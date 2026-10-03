"""The gate must outlive the sound actually leaving the speakers.

Regression coverage for a bug found by measurement, not by reading code: on the
reference laptop the acoustic round-trip through the built-in speakers was
~430 ms while PortAudio reported 3 ms. A gate sized from the reported number
reopens the microphone while the previous sentence is still audible in the room,
which is exactly the feedback loop the gate exists to prevent.
"""

from __future__ import annotations

import pytest

from parliamo.audio.gate import HalfDuplexGate
from parliamo.audio.playback import AudioPlayback, PlaybackInfo


class _FakeDevice:
    label = "Fake Speakers [Test]"
    index = 0
    max_output_channels = 2
    default_samplerate = 48000.0
    likely_bluetooth = False
    name = "Fake Speakers"


def _playback_with_info(gate: HalfDuplexGate, reported_ms: float, measured_ms: float) -> AudioPlayback:
    """Build a playback object with a pre-filled info block, no hardware needed."""
    out = AudioPlayback(sample_rate=16000, gate=gate, output_latency_ms=measured_ms)
    out._info = PlaybackInfo(
        device=_FakeDevice(),  # type: ignore[arg-type]
        sample_rate=16000,
        channels=2,
        latency_ms=reported_ms,
        resampling=False,
    )
    return out


def test_tail_covers_measured_latency_not_reported() -> None:
    gate = HalfDuplexGate(enabled=True, tail_ms=250)
    out = _playback_with_info(gate, reported_ms=3.0, measured_ms=430.0)
    out._extend_gate_tail()
    assert gate.tail_ms == 680, "tail must be measured latency + reverb margin"


def test_falls_back_to_reported_latency_when_unmeasured() -> None:
    gate = HalfDuplexGate(enabled=True, tail_ms=250)
    out = _playback_with_info(gate, reported_ms=90.0, measured_ms=0.0)
    out._extend_gate_tail()
    assert gate.tail_ms == 340


def test_tail_is_never_shortened() -> None:
    """A generous operator-set tail must survive a low-latency device."""
    gate = HalfDuplexGate(enabled=True, tail_ms=900)
    out = _playback_with_info(gate, reported_ms=3.0, measured_ms=10.0)
    out._extend_gate_tail()
    assert gate.tail_ms == 910


def test_disabled_gate_is_untouched() -> None:
    gate = HalfDuplexGate(enabled=False, tail_ms=250)
    out = _playback_with_info(gate, reported_ms=3.0, measured_ms=430.0)
    out._extend_gate_tail()
    assert gate.tail_ms == 250


def test_no_gate_is_not_an_error() -> None:
    out = _playback_with_info(HalfDuplexGate(), 3.0, 430.0)
    out.gate = None
    out._extend_gate_tail()  # must not raise


def test_warning_when_latency_unmeasured(caplog: pytest.LogCaptureFixture) -> None:
    gate = HalfDuplexGate(enabled=True, tail_ms=250)
    out = _playback_with_info(gate, reported_ms=3.0, measured_ms=0.0)
    with caplog.at_level("WARNING"):
        out._extend_gate_tail()
    assert any("output_latency_ms is 0" in r.message for r in caplog.records)


def test_config_exposes_output_latency() -> None:
    from pathlib import Path

    from parliamo.config import load_config

    cfg = load_config(local_path=Path("does-not-exist.yaml"))
    assert cfg.pipeline.output_latency_ms == 0, "ships at 0 so the warning fires until measured"
