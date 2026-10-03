"""The segmentation policy.

Pure state-machine tests: probabilities are supplied directly, so no VAD model
and no audio device is involved and every case is deterministic. This is where
end-to-end latency is decided, so it gets pinned down rather than tuned by ear.
"""

from __future__ import annotations

import numpy as np
import pytest

from parliamo.audio.vad import (
    SILERO_BLOCK,
    SegmenterConfig,
    SpeechSegmenter,
    State,
)

BLOCK_MS = SILERO_BLOCK / 16000 * 1000  # 32 ms


def _block(value: float = 0.1) -> np.ndarray:
    return np.full(SILERO_BLOCK, value, dtype=np.float32)


def feed(
    segmenter: SpeechSegmenter, pattern: str, block_value: float = 0.1
) -> list:
    """Drive the segmenter with a pattern string: '.' silence, 'S' speech."""
    out = []
    for index, char in enumerate(pattern):
        prob = 0.9 if char == "S" else 0.05
        # Give each block a distinct value so ordering can be asserted.
        segment = segmenter.push(_block(block_value + index * 1e-4), prob)
        if segment is not None:
            out.append(segment)
    return out


@pytest.fixture
def cfg() -> SegmenterConfig:
    # Round numbers in blocks: 32 ms each.
    return SegmenterConfig(
        min_speech_ms=96,     # 3 blocks
        min_silence_ms=192,   # 6 blocks
        speech_pad_ms=64,     # 2 blocks
        max_segment_ms=640,   # 20 blocks
    )


# ---------------------------------------------------------------------------
# basic transitions
# ---------------------------------------------------------------------------


def test_starts_in_silence(cfg: SegmenterConfig) -> None:
    assert SpeechSegmenter(cfg).state is State.SILENCE


def test_silence_alone_emits_nothing(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    assert feed(seg, "." * 50) == []
    assert seg.state is State.SILENCE


def test_speech_then_silence_emits_one_segment(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "..." + "S" * 10 + "." * 8)
    assert len(out) == 1
    assert not out[0].truncated
    assert seg.state is State.SILENCE


def test_two_utterances_emit_two_segments(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "S" * 8 + "." * 8 + "S" * 8 + "." * 8)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# minimum speech: transients must not open a segment
# ---------------------------------------------------------------------------


def test_brief_blip_is_ignored(cfg: SegmenterConfig) -> None:
    """A cough, a door, a keyboard - two blocks of 'speech' is not an utterance."""
    seg = SpeechSegmenter(cfg)
    assert feed(seg, "..SS......." * 3) == []
    assert seg.state is State.SILENCE


def test_speech_exactly_at_the_minimum_opens_a_segment(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    feed(seg, "SSS")
    assert seg.state is State.SPEECH


def test_one_block_short_of_the_minimum_does_not(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    feed(seg, "SS.")
    assert seg.state is State.SILENCE


# ---------------------------------------------------------------------------
# minimum silence: short pauses must not split a sentence
# ---------------------------------------------------------------------------


def test_short_pause_does_not_split_a_sentence(cfg: SegmenterConfig) -> None:
    """Speakers pause mid-sentence. Splitting there hands the translator a fragment."""
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "SSSSS" + "..." + "SSSSS" + "." * 8)
    assert len(out) == 1, "a 3-block pause split one sentence into two"


def test_pause_at_the_threshold_does_split(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "SSSSS" + "." * 6 + "SSSSS" + "." * 8)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# pre-roll: the opening consonant must survive
# ---------------------------------------------------------------------------


def test_segment_includes_audio_from_before_speech_was_confirmed(
    cfg: SegmenterConfig,
) -> None:
    """Without pre-roll every sentence loses its first consonant.

    Confirmation needs min_speech_ms of speech, so by the time the segmenter is
    sure, that much audio has already gone past. It has to come back.
    """
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "....." + "S" * 8 + "." * 8)
    assert len(out) == 1

    pad_blocks = cfg.blocks(cfg.speech_pad_ms)
    speech_blocks = 8
    # pre-roll + the speech itself, minus what the trailing trim removes.
    assert out[0].audio.size >= (speech_blocks + pad_blocks) * SILERO_BLOCK * 0.9


def test_segment_start_time_precedes_confirmation(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "." * 10 + "S" * 8 + "." * 8)
    confirmation_time = (10 + cfg.blocks(cfg.min_speech_ms)) * BLOCK_MS / 1000
    assert out[0].start_s < confirmation_time


# ---------------------------------------------------------------------------
# trailing silence trim
# ---------------------------------------------------------------------------


def test_trailing_silence_is_trimmed_to_the_pad(cfg: SegmenterConfig) -> None:
    """Handing the recogniser half a second of room tone invites hallucination."""
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "S" * 10 + "." * 20)
    silence_blocks = round(out[0].audio.size / SILERO_BLOCK) - 10 - cfg.blocks(
        cfg.speech_pad_ms
    )
    assert silence_blocks <= cfg.blocks(cfg.speech_pad_ms) + 1


# ---------------------------------------------------------------------------
# the hard cut
# ---------------------------------------------------------------------------


def test_endless_speech_is_cut_and_marked(cfg: SegmenterConfig) -> None:
    """A speaker who never pauses must not stall the pipeline forever."""
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "S" * 100)
    assert out, "no segment was emitted during continuous speech"
    assert out[0].truncated is True
    assert out[0].audio.size <= cfg.blocks(cfg.max_segment_ms) * SILERO_BLOCK


def test_continuous_speech_emits_repeatedly(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "S" * 100)
    assert len(out) >= 3, "long speech should produce several truncated segments"
    assert all(s.truncated for s in out)


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


def test_flush_emits_speech_in_progress(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    assert feed(seg, "S" * 8) == []
    tail = seg.flush()
    assert tail is not None
    assert tail.truncated is True


def test_flush_during_silence_emits_nothing(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    feed(seg, "." * 10)
    assert seg.flush() is None


def test_flush_twice_is_safe(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    feed(seg, "S" * 8)
    assert seg.flush() is not None
    assert seg.flush() is None


# ---------------------------------------------------------------------------
# reported metadata
# ---------------------------------------------------------------------------


def test_segment_reports_mean_probability(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "S" * 10 + "." * 8)
    assert 0.5 < out[0].mean_speech_prob <= 0.9


def test_segment_duration_matches_audio_length(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    out = feed(seg, "S" * 10 + "." * 8)
    assert out[0].duration_s == pytest.approx(
        out[0].audio.size / cfg.sample_rate, rel=1e-6
    )
    assert out[0].end_s > out[0].start_s


def test_reset_returns_to_silence(cfg: SegmenterConfig) -> None:
    seg = SpeechSegmenter(cfg)
    feed(seg, "S" * 8)
    assert seg.state is State.SPEECH
    seg.reset()
    assert seg.state is State.SILENCE
    assert seg.flush() is None


# ---------------------------------------------------------------------------
# the latency knob
# ---------------------------------------------------------------------------


def test_min_silence_directly_sets_added_latency() -> None:
    """min_silence_ms is added to every segment's delay - assert the arithmetic.

    Configuring 700 ms means the audience waits 700 ms after the speaker stops
    before the recogniser is even handed the audio. This test exists so the
    cost is visible in the test suite, not buried in a config comment.
    """
    for silence_ms in (200, 700, 1200):
        cfg = SegmenterConfig(
            min_speech_ms=96, min_silence_ms=silence_ms, speech_pad_ms=64,
            max_segment_ms=60000,
        )
        seg = SpeechSegmenter(cfg)
        blocks_needed = cfg.blocks(silence_ms)
        out = feed(seg, "S" * 10 + "." * (blocks_needed - 1))
        assert out == [], f"{silence_ms} ms: closed too early"
        assert seg.push(_block(), 0.05) is not None, f"{silence_ms} ms: never closed"


def test_the_configured_threshold_decides_what_is_speech() -> None:
    """vad.threshold used to stop at the VAD object; the segmenter used 0.5.

    Raising it for a noisy hall - the obvious adjustment on the day - would
    have changed nothing, with no sign that it had not.
    """
    import numpy as np

    from parliamo.audio.vad import SegmenterConfig, SpeechSegmenter

    block = np.zeros(512, dtype=np.float32)
    strict = SpeechSegmenter(SegmenterConfig(speech_threshold=0.7, min_speech_ms=64))
    lenient = SpeechSegmenter(SegmenterConfig(speech_threshold=0.5, min_speech_ms=64))
    for _ in range(10):
        strict.push(block, 0.6)
        lenient.push(block, 0.6)
    assert not strict.in_speech, "0.6 is under a 0.7 threshold: room noise, not speech"
    assert lenient.in_speech
