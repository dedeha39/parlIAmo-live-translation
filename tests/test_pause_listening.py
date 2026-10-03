"""P on the operator page: stop listening, for anything the room's speakers
play that is not the system's own voice.

The half-duplex gate closes the microphone while the translation plays. It
knows nothing about a video on a slide, applause or a question from the
floor: the microphone hears those, and they were transcribed, put on the
screen as subtitles and spoken over the video. Mute (M) stops only the voice.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time

import numpy as np

from parliamo.audio.capture import AudioCapture
from parliamo.audio.vad import SegmenterConfig


def _loud(frames: int = 512) -> np.ndarray:
    return np.full((frames, 1), 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# capture: ignored, but visibly heard
# ---------------------------------------------------------------------------


def test_a_paused_capture_queues_nothing_but_the_meter_still_moves() -> None:
    capture = AudioCapture(gate=None)
    capture.paused = True
    capture._callback(_loud(), 512, None, None)
    assert capture.read(timeout=0.01) is None
    assert capture.stats.blocks_paused == 1
    assert capture.stats.recent_peak == 0.5, "the operator must see what is being ignored"

    capture.paused = False
    capture._callback(_loud(), 512, None, None)
    assert capture.read(timeout=0.01) is not None


def test_pausing_is_not_counted_as_speech_lost_to_the_gate() -> None:
    """The gate's counter is about the system's own voice; mixing a video in
    would make it unreadable."""
    from parliamo.audio.gate import HalfDuplexGate

    gate = HalfDuplexGate(enabled=True)
    capture = AudioCapture(gate=gate)
    capture.paused = True
    capture._callback(_loud(), 512, None, None)
    assert gate.stats.speech_during_mute_blocks == 0
    assert capture.stats.blocks_muted == 0


# ---------------------------------------------------------------------------
# transcriber: what was said before the press is still translated
# ---------------------------------------------------------------------------


class _Capture:
    def __init__(self) -> None:
        self.blocks: queue.Queue[np.ndarray] = queue.Queue()
        self.paused = False

    def read(self, timeout: float | None = None) -> np.ndarray | None:
        try:
            return self.blocks.get(timeout=timeout)
        except queue.Empty:
            return None


class _VAD:
    def reset(self) -> None:
        pass

    def probability(self, block: np.ndarray) -> float:
        return 0.9 if float(np.abs(block).max()) > 0.1 else 0.0


def _segment_loop():
    from parliamo.asr.base import ASRBackend, Transcript
    from parliamo.pipeline.transcriber import LiveTranscriber

    class _Backend(ASRBackend):
        name = "fake"

        def __init__(self) -> None:
            super().__init__(model="fake", device="cpu", language="tr")

        def _load(self) -> None:
            pass

        def _unload(self) -> None:
            pass

        def transcribe(self, audio, sample_rate=16000, language=None) -> Transcript:
            return Transcript(text="x", backend="fake")

    transcriber = LiveTranscriber(_Backend(), vad=_VAD(),  # type: ignore[arg-type]
                                  segmenter_config=SegmenterConfig(min_silence_ms=700))
    capture = _Capture()
    transcriber._capture = capture  # type: ignore[assignment]
    transcriber._running.set()
    thread = threading.Thread(target=transcriber._segment_loop, daemon=True)
    thread.start()
    return transcriber, capture, thread


def _speak(capture: _Capture, seconds: float) -> None:
    for _ in range(int(seconds * 16000 / 512)):
        capture.blocks.put(np.full(512, 0.5, dtype=np.float32))


def _committed(transcriber, wait_s: float) -> list:
    deadline = time.monotonic() + wait_s
    got = []
    while time.monotonic() < deadline:
        with contextlib.suppress(queue.Empty):
            got.append(transcriber._segments.get(timeout=0.05)[0])
    return got


def test_the_sentence_before_the_press_is_committed_without_waiting_for_silence() -> None:
    """Pressed mid-sentence, or straight after it: no silence will come while
    paused, so the segmenter would have held the sentence until resume and
    then joined it to whatever was said first."""
    transcriber, capture, thread = _segment_loop()
    try:
        _speak(capture, 1.5)
        assert _committed(transcriber, 0.6) == [], "still waiting for the end of the sentence"

        transcriber.set_paused(True)
        assert capture.paused, "the capture is what drops the audio"
        segments = _committed(transcriber, 0.8)
        assert len(segments) == 1
        assert abs(segments[0].end_s - segments[0].start_s - 1.5) < 0.2
    finally:
        transcriber._running.clear()
        thread.join(timeout=2)


def test_after_resume_the_next_sentence_starts_clean() -> None:
    transcriber, capture, thread = _segment_loop()
    try:
        _speak(capture, 1.0)
        transcriber.set_paused(True)
        assert len(_committed(transcriber, 0.8)) == 1

        transcriber.set_paused(False)
        _speak(capture, 0.8)
        for _ in range(40):  # 1.3 s of silence ends it
            capture.blocks.put(np.zeros(512, dtype=np.float32))
        segments = _committed(transcriber, 1.0)
        assert len(segments) == 1
        assert segments[0].end_s - segments[0].start_s < 1.6, (
            "the sentence after resume must not carry the one before the pause")
    finally:
        transcriber._running.clear()
        thread.join(timeout=2)


def test_a_pause_set_before_start_reaches_the_capture() -> None:
    from parliamo.pipeline.transcriber import LiveTranscriber

    transcriber = LiveTranscriber.__new__(LiveTranscriber)
    transcriber._paused = False
    transcriber._capture = None
    transcriber.set_paused(True)
    assert transcriber.paused


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------


def test_the_translator_pauses_listening_and_leaves_output_alone() -> None:
    from parliamo.pipeline.translator import LiveTranslator

    calls: list[bool] = []

    class _Transcriber:
        def set_paused(self, paused: bool) -> None:
            calls.append(paused)

    translator = LiveTranslator.__new__(LiveTranslator)
    translator.transcriber = _Transcriber()  # type: ignore[assignment]
    translator.muted = False
    translator.set_paused(True)
    translator.set_paused(False)
    assert calls == [True, False]
    assert translator.muted is False


def test_the_controller_keeps_the_pause_across_a_restart() -> None:
    """Otherwise the page says paused while the new pipeline listens."""
    from parliamo.ui.controller import PipelineController

    seen: list[bool] = []

    class _Translator:
        def set_paused(self, paused: bool) -> None:
            seen.append(paused)

        def start(self) -> None:
            pass

    controller = PipelineController(factory=lambda: (_Translator(), None))
    controller.set_paused(True)
    controller._start_blocking()
    assert seen == [True]
    assert controller.running


def test_pause_reaches_the_pipeline_and_toggles() -> None:
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from parliamo.ui.server import SubtitleServer

    calls: list[bool] = []
    server = SubtitleServer(on_pause=calls.append)
    with TestClient(server.build_app()) as c:
        assert c.post("/pause", json={"paused": True}).json()["paused"] is True
        assert c.post("/pause").json()["paused"] is False
    assert calls == [True, False]


def test_the_page_has_the_p_key_and_the_banner() -> None:
    from parliamo.ui.server import APP

    page = APP.read_text(encoding="utf-8")
    assert 'if (k === "p") togglePause();' in page
    assert 'id="pause-banner"' in page and "LISTENING PAUSED" in page
    assert 'post("/pause", {})' in page
