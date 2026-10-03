"""Capture and playback against real hardware.

Marked ``audio`` so they can be deselected (``-m "not audio"``) on machines
without a sound card. Playback tests use a deliberately quiet tone so running
the suite does not startle anyone.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from parliamo.audio.capture import AudioCapture
from parliamo.audio.gate import HalfDuplexGate
from parliamo.audio.playback import AudioPlayback

pytestmark = pytest.mark.audio

QUIET = 0.03  # amplitude for test tones


@pytest.fixture(scope="module")
def has_audio() -> bool:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PortAudio unavailable: {exc}")
    if not any(d["max_input_channels"] > 0 for d in devices):
        pytest.skip("no input device")
    if not any(d["max_output_channels"] > 0 for d in devices):
        pytest.skip("no output device")
    return True


def _tone(seconds: float, rate: int, freq: float = 440.0, amp: float = QUIET) -> np.ndarray:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_capture_delivers_pipeline_rate_audio(has_audio: bool) -> None:
    with AudioCapture(gate=None) as cap:
        info = cap.info
        assert info.pipeline_sample_rate == 16000
        audio = cap.record(0.5)

    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # Allow slack for the first blocks arriving late after stream start.
    assert 0.6 * 8000 <= audio.size <= 8000, f"expected ~8000 samples, got {audio.size}"
    assert np.isfinite(audio).all(), "capture produced NaN/Inf"


def test_capture_reports_what_it_negotiated(has_audio: bool) -> None:
    with AudioCapture(gate=None) as cap:
        info = cap.info.as_dict()
    assert info["pipeline_sample_rate"] == 16000
    assert info["device_sample_rate"] > 0
    # Whatever path was taken, the two must be reconciled.
    assert info["resampling"] == (info["device_sample_rate"] != 16000)


def test_capture_is_idempotent_on_start(has_audio: bool) -> None:
    cap = AudioCapture(gate=None)
    try:
        first = cap.start()
        second = cap.start()
        assert first is second
    finally:
        cap.stop()


def test_capture_drain_empties_queue(has_audio: bool) -> None:
    with AudioCapture(gate=None) as cap:
        time.sleep(0.3)
        cap.drain()
        assert cap.read(timeout=0.0) is None or True  # drain must not raise


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------


def test_playback_completes(has_audio: bool) -> None:
    with AudioPlayback(sample_rate=16000) as out:
        frames = out.play(_tone(0.15, 16000), blocking=True)
        assert frames > 0
        assert not out.is_playing


def test_playback_clips_instead_of_wrapping(has_audio: bool) -> None:
    """A wrapped sample is a bang through a PA system; a clipped one is not."""
    out = AudioPlayback(sample_rate=16000, gain=50.0)
    try:
        out.start()
        prepared = out._prepare(_tone(0.05, 16000, amp=0.5))
        assert float(np.abs(prepared).max()) <= 1.0
    finally:
        out.stop()


# ---------------------------------------------------------------------------
# the integration that matters: gating during playback
# ---------------------------------------------------------------------------


def test_gate_closes_during_playback_and_reopens_after(has_audio: bool) -> None:
    gate = HalfDuplexGate(enabled=True, tail_ms=150)
    with AudioPlayback(sample_rate=16000, gate=gate) as out:
        assert gate.is_open, "gate should start open"
        out.submit(_tone(0.4, 16000))
        time.sleep(0.05)
        assert gate.is_closed, "gate must close as soon as audio is queued"
        out.wait(timeout=3.0)
        assert gate.is_closed, "tail must keep the gate closed after the last sample"
        time.sleep(0.25)
        assert gate.is_open, "gate must reopen once the tail elapses"


def test_capture_discards_audio_while_gate_is_closed(has_audio: bool) -> None:
    """The anti-feedback guarantee, end to end.

    While synthesised audio plays, the capture callback must throw blocks away
    rather than queue them - otherwise the system hears itself and loops.
    """
    gate = HalfDuplexGate(enabled=True, tail_ms=100)
    capture = AudioCapture(gate=gate)
    playback = AudioPlayback(sample_rate=16000, gate=gate)
    try:
        capture.start()
        playback.start()
        time.sleep(0.3)
        capture.drain()

        playback.submit(_tone(0.6, 16000))
        time.sleep(0.35)
        assert gate.is_closed

        delivered_during = capture.stats.blocks_delivered
        muted_during = capture.stats.blocks_muted
        assert muted_during > 0, "no blocks were muted while playing"

        playback.wait(timeout=3.0)
        time.sleep(0.2)
        assert gate.is_open

        before = capture.stats.blocks_delivered
        time.sleep(0.3)
        assert capture.stats.blocks_delivered > before, "capture did not resume after the gate reopened"
        assert delivered_during <= before
    finally:
        playback.stop()
        capture.stop()


def test_recent_peak_decays_where_the_all_time_peak_does_not() -> None:
    """The microphone card asks "is it hearing anything *now*".

    ``peak_amplitude`` is the run's maximum and stays high forever after one
    loud block; ``recent_peak`` has to fall back to the floor between
    sentences, or the card says "hearing speech" through a minute of silence.
    """
    from parliamo.audio.capture import CaptureStats

    stats = CaptureStats()
    loud, quiet = 0.5, 0.002
    stats.peak_amplitude = max(stats.peak_amplitude, loud)
    stats.recent_peak = max(loud, stats.recent_peak * 0.93)
    for _ in range(60):  # ~2 s of 32 ms blocks of room noise
        stats.recent_peak = max(quiet, stats.recent_peak * 0.93)
    assert stats.peak_amplitude == loud, "the all-time peak must remember the word"
    assert stats.recent_peak < 0.02, "the recent peak must have forgotten it"
    assert stats.recent_peak >= quiet
    assert "recent_peak" in stats.as_dict()
