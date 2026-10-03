"""Provisional subtitles: shown while the speaker is still talking.

The commit wait is 700 ms of the ~3 s an audience waits, and it is spent
proving a sentence has ended. A partial skips that proof: it reads the
utterance so far and shows a guess, replaced when the sentence is committed.

Two properties carry the whole design, and both are tested here:

* **A partial is never spoken.** Turkish puts the verb, and the negation
  suffix, at the end - "bu teknik bir mesele *değil*" reverses everything
  before it. A subtitle can be corrected on screen; audio cannot be un-said.
* **A partial never delays the sentence it previews.** It runs in the
  recogniser's idle time and is dropped whenever there is real work queued.
"""

from __future__ import annotations

import time

import numpy as np

from parliamo.audio.vad import SegmenterConfig, SpeechSegmenter

RATE = 16000
BLOCK = 512
BLOCK_MS = BLOCK / RATE * 1000.0  # 32 ms


def _cfg(**kwargs) -> SegmenterConfig:
    base = {
        "min_speech_ms": 100,
        "min_silence_ms": 300,
        "speech_pad_ms": 64,
        "max_segment_ms": 30000,
        "partial_interval_ms": 320,
        "min_partial_ms": 320,
    }
    base.update(kwargs)
    return SegmenterConfig(**base)


def _block() -> np.ndarray:
    return np.full(BLOCK, 0.1, dtype=np.float32)


def _speak(seg: SpeechSegmenter, blocks: int, prob: float = 0.9) -> list:
    """Push *blocks* of speech, collecting any partials offered."""
    partials = []
    for _ in range(blocks):
        seg.push(_block(), prob)
        partial = seg.take_partial()
        if partial is not None:
            partials.append(partial)
    return partials


# ---------------------------------------------------------------------------
# when a partial is offered
# ---------------------------------------------------------------------------


def test_disabled_by_default() -> None:
    """Off unless asked for: it costs GPU time and changes what is displayed."""
    seg = SpeechSegmenter(SegmenterConfig())
    assert SegmenterConfig().partial_interval_ms == 0
    _speak(seg, 60)
    assert seg.take_partial() is None


def test_no_partial_during_silence() -> None:
    seg = SpeechSegmenter(_cfg())
    for _ in range(40):
        seg.push(_block(), 0.0)
        assert seg.take_partial() is None


def test_no_partial_before_enough_audio() -> None:
    """Whisper hallucinates on very short buffers, so wait for min_partial_ms."""
    seg = SpeechSegmenter(_cfg(min_partial_ms=2000))
    partials = _speak(seg, 20)  # 640 ms of speech
    assert partials == []


def test_partial_is_offered_once_the_interval_passes() -> None:
    seg = SpeechSegmenter(_cfg())
    partials = _speak(seg, 40)  # 1.28 s of speech
    assert partials, "no partial offered during a long utterance"
    assert all(p.partial for p in partials)


def test_partials_are_spaced_by_the_interval() -> None:
    seg = SpeechSegmenter(_cfg(partial_interval_ms=320))
    partials = _speak(seg, 60)
    assert len(partials) >= 2
    gaps = [
        b.audio.size - a.audio.size for a, b in zip(partials, partials[1:], strict=False)
    ]
    expected = SegmenterConfig(partial_interval_ms=320).blocks(320) * BLOCK
    assert all(g == expected for g in gaps), gaps


def test_partials_grow(nothing_is_lost=None) -> None:
    """Each partial contains everything the previous one did, and more."""
    seg = SpeechSegmenter(_cfg())
    partials = _speak(seg, 60)
    sizes = [p.audio.size for p in partials]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_partials_are_numbered_from_one() -> None:
    seg = SpeechSegmenter(_cfg())
    partials = _speak(seg, 60)
    assert [p.partial_index for p in partials] == list(range(1, len(partials) + 1))


def test_partial_index_resets_for_the_next_utterance() -> None:
    seg = SpeechSegmenter(_cfg())
    _speak(seg, 40)
    for _ in range(20):  # silence closes the segment
        seg.push(_block(), 0.0)
    second = _speak(seg, 40)
    assert second, "no partial in the second utterance"
    assert second[0].partial_index == 1


def test_committed_segment_is_not_marked_partial() -> None:
    seg = SpeechSegmenter(_cfg())
    _speak(seg, 40)
    closed = None
    for _ in range(20):
        closed = seg.push(_block(), 0.0) or closed
    assert closed is not None
    assert closed.partial is False
    assert closed.partial_index == 0


def test_partial_covers_the_speech_so_far() -> None:
    seg = SpeechSegmenter(_cfg())
    partials = _speak(seg, 40)
    assert partials
    # 40 blocks of speech plus the pre-roll, so at least the speech itself.
    assert partials[-1].duration_s >= 30 * BLOCK_MS / 1000.0


# ---------------------------------------------------------------------------
# the delivery contract
# ---------------------------------------------------------------------------


class _FakeMT:
    source_lang = "tr"
    loaded = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def unload(self) -> None: ...

    def translate(self, text, source_lang=None, target_lang=None):
        from parliamo.mt.base import Translation

        self.calls.append(text)
        return Translation(text=f"IT({text})", source=text, source_lang="tr",
                           target_lang="it", backend="fake")


class _FakeTTS:
    sample_rate = 24000
    loaded = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def unload(self) -> None: ...

    def speak(self, text, language=None, voice=None):
        from parliamo.tts.base import Speech

        self.calls.append(text)
        return Speech(audio=np.zeros(2400, dtype=np.float32), sample_rate=self.sample_rate,
                      text=text, language="it", backend="fake")


def _translator(**kwargs):
    from parliamo.pipeline.transcriber import LiveTranscriber
    from parliamo.pipeline.translator import LiveTranslator

    class _Backend:
        loaded = True
        model = "fake"

        def load(self) -> None: ...
        def unload(self) -> None: ...
        def warmup(self, seconds: float = 1.0) -> float:
            return 0.0

        def transcribe(self, audio, sample_rate=16000, language=None):
            from parliamo.asr.base import Transcript

            return Transcript(text="merhaba", backend="fake")

    transcriber = LiveTranscriber(_Backend())
    mt, tts = _FakeMT(), _FakeTTS()
    translator = LiveTranslator(transcriber, mt, tts, playback=None, speak=True, **kwargs)
    return translator, mt, tts


def _event(partial: bool):
    from parliamo.audio.vad import SpeechSegment
    from parliamo.pipeline.transcriber import TranscriptEvent

    segment = SpeechSegment(
        audio=np.zeros(16000, dtype=np.float32), start_s=0.0, end_s=1.0, partial=partial
    )
    return TranscriptEvent(
        index=1, text="bu teknik bir mesele", segment=segment,
        transcript=None, commit_wait_s=0.7, queue_wait_s=0.0, asr_s=0.1,
        partial=partial,
    )


def test_a_partial_is_translated_but_never_synthesised() -> None:
    """The property the whole design rests on."""
    translator, mt, tts = _translator()
    delivery = translator.deliver(_event(partial=True))
    assert delivery is not None
    assert delivery.partial is True
    assert delivery.spoken is False
    assert mt.calls == ["bu teknik bir mesele"]
    assert tts.calls == [], "a provisional sentence reached the synthesiser"


def test_a_committed_sentence_is_synthesised() -> None:
    translator, mt, tts = _translator()
    delivery = translator.deliver(_event(partial=False))
    assert delivery is not None
    assert delivery.partial is False
    assert tts.calls == ["IT(bu teknik bir mesele)"]


def test_a_partial_does_not_advance_the_sentence_counter() -> None:
    """Otherwise the subtitle numbering jumps as previews come and go."""
    translator, _, _ = _translator()
    translator.deliver(_event(partial=True))
    translator.deliver(_event(partial=True))
    committed = translator.deliver(_event(partial=False))
    assert committed is not None
    assert committed.index == 1


def test_partials_are_counted_separately() -> None:
    translator, _, _ = _translator()
    translator.deliver(_event(partial=True))
    translator.deliver(_event(partial=False))
    summary = translator.stats.summary()
    assert summary["partials_delivered"] == 1
    assert summary["delivered"] == 1


def test_a_failed_partial_is_dropped_quietly() -> None:
    """A preview that fails must not report an error the audience would see."""
    translator, mt, _ = _translator()

    def boom(*_args, **_kwargs):
        raise RuntimeError("translation exploded")

    mt.translate = boom
    assert translator.deliver(_event(partial=True)) is None
    assert translator.stats.translation_failures == 0


def test_partial_lag_excludes_the_commit_wait() -> None:
    """A partial has not waited for silence - that is the point of it."""
    import pytest

    event = _event(partial=True)
    assert event.total_lag_s == pytest.approx(0.1)  # asr only, no 0.7 s commit
    assert _event(partial=False).total_lag_s == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# the partial must never delay the sentence
# ---------------------------------------------------------------------------


def test_partial_is_dropped_when_the_recogniser_has_work() -> None:
    from parliamo.audio.vad import SpeechSegment
    from parliamo.pipeline.transcriber import LiveTranscriber

    class _Backend:
        loaded = True
        model = "fake"

        def load(self) -> None: ...
        def unload(self) -> None: ...
        def warmup(self, seconds: float = 1.0) -> float:
            return 0.0

        def transcribe(self, audio, sample_rate=16000, language=None):  # pragma: no cover
            from parliamo.asr.base import Transcript

            return Transcript(text="x", backend="fake")

    transcriber = LiveTranscriber(_Backend())
    committed = SpeechSegment(audio=np.zeros(16000, dtype=np.float32), start_s=0, end_s=1)
    partial = SpeechSegment(
        audio=np.zeros(16000, dtype=np.float32), start_s=0, end_s=1, partial=True
    )

    transcriber._enqueue(committed)          # real work is now waiting
    transcriber._enqueue_partial(partial)    # the preview must give way

    assert transcriber.stats.partials_emitted == 0
    assert transcriber.stats.partials_skipped_busy == 1
    assert transcriber._segments.qsize() == 1


def test_partial_is_queued_when_the_recogniser_is_idle() -> None:
    from parliamo.audio.vad import SpeechSegment
    from parliamo.pipeline.transcriber import LiveTranscriber

    class _Backend:
        loaded = True
        model = "fake"

        def load(self) -> None: ...
        def unload(self) -> None: ...
        def warmup(self, seconds: float = 1.0) -> float:
            return 0.0

        def transcribe(self, audio, sample_rate=16000, language=None):  # pragma: no cover
            from parliamo.asr.base import Transcript

            return Transcript(text="x", backend="fake")

    transcriber = LiveTranscriber(_Backend())
    partial = SpeechSegment(
        audio=np.zeros(16000, dtype=np.float32), start_s=0, end_s=1, partial=True
    )
    transcriber._enqueue_partial(partial)
    assert transcriber.stats.partials_emitted == 1


def test_taking_a_partial_is_cheap() -> None:
    """It runs on every audio block, so it must not cost anything measurable."""
    seg = SpeechSegmenter(_cfg(partial_interval_ms=10_000))  # never due
    _speak(seg, 30)
    t0 = time.perf_counter()
    for _ in range(1000):
        seg.take_partial()
    per_call_us = (time.perf_counter() - t0) / 1000 * 1e6
    assert per_call_us < 50, f"{per_call_us:.1f} us per call"


# ---------------------------------------------------------------------------
# the soft cut, which measurement told us to leave off
# ---------------------------------------------------------------------------


def test_soft_cut_is_off_by_default() -> None:
    """It bought 0.9 s off the worst case for 0.67 WER points and a fragment.

    Kept as a knob because another speaker may pause differently, but a
    default that costs accuracy needs a test holding it shut.
    """
    assert SegmenterConfig().soft_cut_after_ms == 0


def test_soft_cut_closes_at_a_short_pause_when_enabled() -> None:
    seg = SpeechSegmenter(_cfg(min_silence_ms=700, soft_cut_after_ms=640,
                               soft_cut_silence_ms=96, partial_interval_ms=0))
    _speak(seg, 30)                       # ~0.96 s, past soft_cut_after_ms
    closed = None
    for _ in range(4):                    # ~128 ms of silence, under min_silence
        closed = seg.push(_block(), 0.0) or closed
    assert closed is not None, "soft cut did not fire"
    assert closed.soft_cut is True
    assert closed.truncated is False, "a soft cut is a real pause, not a clock"


def test_a_short_utterance_is_not_soft_cut() -> None:
    """Waiting the full silence is right at the start - it keeps a clause whole."""
    seg = SpeechSegmenter(_cfg(min_silence_ms=700, soft_cut_after_ms=3000,
                               soft_cut_silence_ms=96, partial_interval_ms=0))
    _speak(seg, 10)
    for _ in range(4):
        assert seg.push(_block(), 0.0) is None


def test_an_ordinary_close_is_not_marked_soft_cut() -> None:
    seg = SpeechSegmenter(_cfg(partial_interval_ms=0))
    _speak(seg, 40)
    closed = None
    for _ in range(20):
        closed = seg.push(_block(), 0.0) or closed
    assert closed is not None
    assert closed.soft_cut is False
