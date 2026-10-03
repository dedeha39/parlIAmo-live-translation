"""The live transcription loop.

The recogniser is faked, so these tests exercise the *orchestration* - thread
lifecycle, backpressure, latency accounting - deterministically and in a couple
of seconds. Whether the real model transcribes correctly is answered by
tests/test_asr_backend.py and the bake-off, not here.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from parliamo.asr.base import ASRBackend, Segment, Transcript
from parliamo.audio.vad import SegmenterConfig, SpeechSegment
from parliamo.pipeline.transcriber import LiveTranscriber, TranscriptEvent


class FakeBackend(ASRBackend):
    """A recogniser that returns fixed text after a controllable delay."""

    name = "fake"

    def __init__(self, text: str = "merhaba dünya", delay_s: float = 0.0, **kw) -> None:
        super().__init__(model="fake", device="cpu", language="tr", **kw)
        self.text = text
        self.delay_s = delay_s
        self.calls = 0
        self.fail_next = False

    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def transcribe(self, audio, sample_rate=16000, language=None) -> Transcript:
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("simulated recognition failure")
        if self.delay_s:
            time.sleep(self.delay_s)
        duration = np.asarray(audio).size / sample_rate
        return Transcript(
            text=self.text,
            segments=[Segment(0.0, duration, self.text)],
            language="tr",
            audio_duration_s=duration,
            compute_s=self.delay_s,
            backend="fake",
        )


def _segment(seconds: float = 1.0) -> SpeechSegment:
    return SpeechSegment(
        audio=np.zeros(int(seconds * 16000), dtype=np.float32),
        start_s=0.0,
        end_s=seconds,
        sample_rate=16000,
    )


@pytest.fixture
def transcriber() -> LiveTranscriber:
    return LiveTranscriber(
        FakeBackend(),
        segmenter_config=SegmenterConfig(min_silence_ms=700),
        max_queue_depth=3,
    )


# ---------------------------------------------------------------------------
# latency accounting
# ---------------------------------------------------------------------------


def test_total_lag_is_the_sum_of_its_parts() -> None:
    event = TranscriptEvent(
        index=1, text="x", segment=_segment(), transcript=Transcript(text="x"),
        commit_wait_s=0.7, queue_wait_s=0.1, asr_s=0.3,
    )
    assert event.total_lag_s == pytest.approx(1.1)


def test_commit_wait_comes_from_min_silence(transcriber: LiveTranscriber) -> None:
    """The dominant latency term is a policy choice, and must be reported as one."""
    assert transcriber._commit_wait_s == pytest.approx(0.7)

    faster = LiveTranscriber(FakeBackend(), segmenter_config=SegmenterConfig(min_silence_ms=300))
    assert faster._commit_wait_s == pytest.approx(0.3)


def test_event_serialises_the_breakdown() -> None:
    event = TranscriptEvent(
        index=2, text="merhaba", segment=_segment(2.0),
        transcript=Transcript(text="merhaba", audio_duration_s=2.0, compute_s=0.4),
        commit_wait_s=0.7, queue_wait_s=0.05, asr_s=0.4,
    )
    payload = event.as_dict()
    assert payload["total_lag_s"] == pytest.approx(1.15)
    assert payload["audio_s"] == 2.0
    assert payload["asr_s"] == 0.4


# ---------------------------------------------------------------------------
# the recogniser loop
# ---------------------------------------------------------------------------


def _run_with_consumer(
    transcriber: LiveTranscriber,
    events: list,
    segments: int,
    expected: int,
    timeout: float = 5.0,
) -> None:
    """Start the recogniser thread *first*, then feed it.

    Enqueueing before the consumer exists fills the bounded queue and triggers
    the backpressure path, which is correct behaviour but not what these tests
    are about - the drop tests exercise it deliberately.
    """
    transcriber._running.set()
    thread = threading.Thread(target=transcriber._recognise_loop, daemon=True)
    thread.start()
    try:
        for _ in range(segments):
            transcriber._enqueue(_segment())
            time.sleep(0.01)
        deadline = time.monotonic() + timeout
        while len(events) < expected and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        transcriber._running.clear()
        transcriber._segments.put(None)
        thread.join(timeout=2.0)


def test_segments_become_transcript_events(transcriber: LiveTranscriber) -> None:
    events: list[TranscriptEvent] = []
    transcriber.on_transcript = events.append
    _run_with_consumer(transcriber, events, segments=3, expected=3)

    assert len(events) == 3
    assert [e.index for e in events] == [1, 2, 3]
    assert all(e.text == "merhaba dünya" for e in events)
    assert transcriber.stats.segments_transcribed == 3
    assert transcriber.stats.segments_dropped_backlog == 0


def test_empty_transcripts_are_counted_not_emitted(transcriber: LiveTranscriber) -> None:
    transcriber.backend = FakeBackend(text="   ")
    events: list[TranscriptEvent] = []
    transcriber.on_transcript = events.append
    _run_with_consumer(transcriber, events, segments=1, expected=0, timeout=1.0)

    assert events == []
    assert transcriber.stats.segments_empty == 1
    assert transcriber.stats.segments_transcribed == 1


def test_recognition_failure_does_not_kill_the_loop(transcriber: LiveTranscriber) -> None:
    backend = FakeBackend()
    backend.fail_next = True
    transcriber.backend = backend
    events: list[TranscriptEvent] = []
    transcriber.on_transcript = events.append

    # First segment raises, second must still be processed.
    _run_with_consumer(transcriber, events, segments=2, expected=1)

    assert len(events) == 1, "the loop stopped after one failure"
    assert backend.calls == 2


def test_callback_exception_does_not_kill_the_loop(transcriber: LiveTranscriber) -> None:
    seen: list[int] = []

    def bad_callback(event: TranscriptEvent) -> None:
        seen.append(event.index)
        raise ValueError("callback blew up")

    transcriber.on_transcript = bad_callback
    for _ in range(3):
        transcriber._enqueue(_segment())

    transcriber._running.set()
    thread = threading.Thread(target=transcriber._recognise_loop, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5.0
    while len(seen) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    transcriber._running.clear()
    transcriber._segments.put(None)
    thread.join(timeout=2.0)

    assert seen == [1, 2, 3], "a raising callback stopped the pipeline"


# ---------------------------------------------------------------------------
# backpressure
# ---------------------------------------------------------------------------


def test_full_queue_drops_the_oldest_segment(transcriber: LiveTranscriber) -> None:
    """A speaker who has moved on is not helped by old audio.

    max_queue_depth is 3 here; pushing five without a consumer must keep the
    three newest and count two drops.
    """
    for _ in range(5):
        transcriber._enqueue(_segment())

    assert transcriber.stats.segments_detected == 5
    assert transcriber.stats.segments_dropped_backlog == 2
    assert transcriber._segments.qsize() == 3


def test_slow_recogniser_causes_drops_not_unbounded_growth() -> None:
    slow = LiveTranscriber(
        FakeBackend(delay_s=0.15),
        segmenter_config=SegmenterConfig(min_silence_ms=700),
        max_queue_depth=2,
    )
    events: list[TranscriptEvent] = []
    slow.on_transcript = events.append

    slow._running.set()
    thread = threading.Thread(target=slow._recognise_loop, daemon=True)
    thread.start()
    for _ in range(12):
        slow._enqueue(_segment())
        time.sleep(0.01)
    time.sleep(0.8)
    slow._running.clear()
    slow._segments.put(None)
    thread.join(timeout=2.0)

    assert slow._segments.qsize() <= 3, "queue grew without bound"
    assert slow.stats.segments_dropped_backlog > 0
    assert slow.stats.segments_transcribed > 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def test_summary_reports_lag_percentiles(transcriber: LiveTranscriber) -> None:
    events: list[TranscriptEvent] = []
    transcriber.on_transcript = events.append
    _run_with_consumer(transcriber, events, segments=5, expected=5)

    summary = transcriber.stats.summary()
    assert summary["segments_transcribed"] == 5
    assert {"mean", "p50", "p95", "max"} <= set(summary["lag_s"])
    assert summary["lag_s"]["mean"] >= 0.7, "commit wait must be included in the lag"


def test_summary_without_events_is_safe() -> None:
    summary = LiveTranscriber(FakeBackend()).stats.summary()
    assert summary["segments_transcribed"] == 0
    assert "lag_s" not in summary


def test_stop_before_start_is_safe() -> None:
    t = LiveTranscriber(FakeBackend())
    assert t.stop().segments_transcribed == 0
    assert not t.running


def test_a_segment_that_produced_no_text_is_kept_for_listening(tmp_path) -> None:
    """"A sentence went missing" has to be something someone can listen to.

    The log says *why* a segment was dropped - a repetition loop, an empty
    decode - and nothing about what was said. Keeping the audio settles it.
    """
    import json

    import numpy as np

    from parliamo.asr.base import Transcript
    from parliamo.audio.vad import SpeechSegment
    from parliamo.pipeline.transcriber import LiveTranscriber

    transcriber = LiveTranscriber.__new__(LiveTranscriber)
    transcriber.keep_dropped_dir = tmp_path / "dropped"
    transcriber._dropped_count = 0

    segment = SpeechSegment(audio=np.zeros(16000, dtype=np.float32), start_s=3.0,
                            end_s=4.0, sample_rate=16000, truncated=False,
                            mean_speech_prob=0.8)
    transcript = Transcript(text="", segments=[], language="tr", language_probability=1.0,
                            audio_duration_s=1.0, compute_s=0.1, backend="test",
                            dropped_segments=[{"reason": "loop"}])
    transcriber._keep_dropped(segment, transcript)

    assert (tmp_path / "dropped" / "dropped-001.wav").is_file()
    meta = json.loads((tmp_path / "dropped" / "dropped-001.json").read_text(encoding="utf-8"))
    assert meta["start_s"] == 3.0 and meta["dropped_segments"] == [{"reason": "loop"}]


def test_nothing_is_kept_unless_asked() -> None:
    from parliamo.pipeline.transcriber import LiveTranscriber

    transcriber = LiveTranscriber.__new__(LiveTranscriber)
    transcriber.keep_dropped_dir = None
    transcriber._keep_dropped(None, None)  # must not raise, must not touch disk
