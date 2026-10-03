"""Silero VAD against real speech.

Marked ``model`` because it downloads weights on first run. The synthetic
state-machine tests live in test_vad_segmenter.py; this file checks the piece
that state machine depends on actually distinguishes speech from silence, and
does it fast enough to keep up with real time.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from parliamo.audio.vad import (
    SILERO_BLOCK,
    SegmenterConfig,
    SileroVAD,
    segment_audio,
)

pytestmark = pytest.mark.model

BLOCK_SECONDS = SILERO_BLOCK / 16000


@pytest.fixture(scope="module")
def vad() -> SileroVAD:
    v = SileroVAD(device="cpu")
    try:
        v.load()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"silero-vad unavailable: {exc}")
    return v


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    """A real Turkish utterance from FLEURS."""
    from parliamo.paths import configure_model_cache

    configure_model_cache()
    try:
        from parliamo.eval.datasets import load_fleurs

        items = load_fleurs("tr_tr", "test", limit=3)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"FLEURS unavailable: {exc}")
    if not items:
        pytest.skip("FLEURS returned nothing")
    return items[0].audio


def _silence(seconds: float) -> np.ndarray:
    """Room tone, not digital zero - a real microphone never gives pure zeros."""
    rng = np.random.default_rng(20260830)
    return (rng.standard_normal(int(seconds * 16000)) * 1e-4).astype(np.float32)


# ---------------------------------------------------------------------------
# the model discriminates
# ---------------------------------------------------------------------------


def test_silence_scores_low(vad: SileroVAD) -> None:
    probs = [
        vad.probability(_silence(BLOCK_SECONDS)[:SILERO_BLOCK]) for _ in range(20)
    ]
    assert max(probs) < 0.5, f"silence reached {max(probs):.3f}"


def test_speech_scores_high_somewhere(vad: SileroVAD, speech: np.ndarray) -> None:
    vad.reset()
    probs = [
        vad.probability(speech[i : i + SILERO_BLOCK])
        for i in range(0, speech.size - SILERO_BLOCK, SILERO_BLOCK)
    ]
    assert max(probs) > 0.8, f"real speech only reached {max(probs):.3f}"
    # Most of an utterance should read as speech, allowing for internal pauses.
    assert np.mean([p >= 0.5 for p in probs]) > 0.5


def test_wrong_block_size_is_rejected(vad: SileroVAD) -> None:
    with pytest.raises(ValueError, match="512 samples"):
        vad.probability(np.zeros(1024, dtype=np.float32))


def test_reset_is_safe_before_and_after_use(vad: SileroVAD) -> None:
    vad.reset()
    vad.probability(np.zeros(SILERO_BLOCK, dtype=np.float32))
    vad.reset()


# ---------------------------------------------------------------------------
# it has to keep up
# ---------------------------------------------------------------------------


def test_vad_is_far_faster_than_real_time(vad: SileroVAD, speech: np.ndarray) -> None:
    """Each block covers 32 ms of audio, so inference must cost well under that.

    The VAD runs on every block of every second the microphone is open, ahead of
    three much larger models. If it were even close to real time it would eat
    the latency budget before recognition started.
    """
    vad.reset()
    blocks = [
        speech[i : i + SILERO_BLOCK]
        for i in range(0, min(speech.size, 300 * SILERO_BLOCK) - SILERO_BLOCK, SILERO_BLOCK)
    ]
    t0 = time.perf_counter()
    for block in blocks:
        vad.probability(block)
    elapsed = time.perf_counter() - t0

    per_block_ms = elapsed / len(blocks) * 1000
    rtf = elapsed / (len(blocks) * BLOCK_SECONDS)
    assert per_block_ms < 5.0, f"{per_block_ms:.2f} ms per 32 ms block"
    assert rtf < 0.1, f"VAD real-time factor {rtf:.3f}"


# ---------------------------------------------------------------------------
# end to end over real audio
# ---------------------------------------------------------------------------


def test_segments_real_speech_between_silences(
    vad: SileroVAD, speech: np.ndarray
) -> None:
    """Three utterances separated by clear pauses must come back as three."""
    gap = _silence(1.2)
    stream = np.concatenate([gap, speech, gap, speech, gap, speech, gap])

    cfg = SegmenterConfig(min_speech_ms=250, min_silence_ms=700, speech_pad_ms=120,
                          max_segment_ms=60000)
    segments = list(segment_audio(stream, vad, cfg))

    assert len(segments) == 3, f"expected 3 utterances, got {len(segments)}"
    for segment in segments:
        assert segment.duration_s > 1.0
        assert segment.mean_speech_prob > 0.5
        assert not segment.truncated


def test_pure_silence_yields_no_segments(vad: SileroVAD) -> None:
    segments = list(segment_audio(_silence(5.0), vad, SegmenterConfig()))
    assert segments == []


def _true_onset_s(vad: SileroVAD, stream: np.ndarray) -> float:
    """When the detector first sees speech, measured rather than assumed.

    FLEURS clips carry their own leading silence - the sample used here has
    ~2.8 s of it - so the onset cannot be inferred from how much padding the
    test prepends.
    """
    vad.reset()
    for index in range(0, stream.size - SILERO_BLOCK, SILERO_BLOCK):
        if vad.probability(stream[index : index + SILERO_BLOCK]) >= 0.5:
            return index / 16000
    raise AssertionError("no speech found in the stream")


def test_segments_do_not_clip_the_onset(vad: SileroVAD, speech: np.ndarray) -> None:
    """The segment must begin *before* the speech does, not after.

    Regression for a real design bug. Speech is only confirmed once
    min_speech_ms of it has gone past, so a pre-roll buffer sized at
    speech_pad_ms started the segment (min_speech_ms - speech_pad_ms) too late.
    With the shipped defaults that silently removed 130 ms from the front of
    every utterance - the opening consonant, which in Turkish is often what
    distinguishes one word from another (kar/var, ürün/gürün).
    """
    stream = np.concatenate([_silence(1.0), speech, _silence(1.5)])
    onset = _true_onset_s(vad, stream)

    cfg = SegmenterConfig(min_speech_ms=250, min_silence_ms=700, speech_pad_ms=120,
                          max_segment_ms=60000)
    segments = list(segment_audio(stream, vad, cfg))
    assert len(segments) == 1

    segment = segments[0]
    assert segment.start_s < onset, (
        f"segment starts at {segment.start_s:.3f}s, speech at {onset:.3f}s - "
        f"{(segment.start_s - onset) * 1000:.0f} ms of onset clipped"
    )
    # ...but not so far back that the recogniser is handed mostly room tone.
    lead_ms = (onset - segment.start_s) * 1000
    assert 50 <= lead_ms <= 600, f"pre-roll of {lead_ms:.0f} ms is out of range"


@pytest.mark.parametrize("min_speech_ms", [100, 250, 500])
def test_preroll_covers_confirmation_delay(
    vad: SileroVAD, speech: np.ndarray, min_speech_ms: int
) -> None:
    """Holds at every confirmation threshold, not just the shipped default.

    The pre-roll buffer has to be sized from min_speech_ms + speech_pad_ms; if
    it were sized from the padding alone, raising the confirmation threshold
    would quietly eat more of each utterance.
    """
    stream = np.concatenate([_silence(1.0), speech, _silence(1.5)])
    onset = _true_onset_s(vad, stream)

    cfg = SegmenterConfig(
        min_speech_ms=min_speech_ms, min_silence_ms=700, speech_pad_ms=120,
        max_segment_ms=60000,
    )
    segments = list(segment_audio(stream, vad, cfg))
    assert segments, f"min_speech_ms={min_speech_ms}: nothing detected"
    assert segments[0].start_s < onset, (
        f"min_speech_ms={min_speech_ms}: onset clipped by "
        f"{(segments[0].start_s - onset) * 1000:.0f} ms"
    )


def test_segment_covers_the_speech_it_contains(
    vad: SileroVAD, speech: np.ndarray
) -> None:
    """The segment must span from before the first speech to after the last."""
    stream = np.concatenate([_silence(1.0), speech, _silence(1.5)])
    vad.reset()
    probs = [
        vad.probability(stream[i : i + SILERO_BLOCK])
        for i in range(0, stream.size - SILERO_BLOCK, SILERO_BLOCK)
    ]
    speaking = [i for i, p in enumerate(probs) if p >= 0.5]
    first_s = speaking[0] * SILERO_BLOCK / 16000
    last_s = speaking[-1] * SILERO_BLOCK / 16000

    cfg = SegmenterConfig(min_speech_ms=250, min_silence_ms=700, speech_pad_ms=120,
                          max_segment_ms=60000)
    segment = list(segment_audio(stream, vad, cfg))[0]
    assert segment.start_s < first_s
    assert segment.end_s > last_s
