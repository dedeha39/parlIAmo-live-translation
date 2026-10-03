"""The verdicts the feedback-loop check draws from its two passes.

The measurement needs hardware; the reasoning about it does not, and the
reasoning is where this went wrong twice.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_feedback_loop.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_feedback_loop_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pass(*, floor: float, during: float, after: float,
          delivered: float = 4.0, peak: float = 0.4, held: float = 0.68) -> dict:
    return {"floor_dbfs": floor, "during_dbfs": during, "after_reopen_dbfs": after,
            "seconds_delivered_during": delivered, "during_peak": peak,
            "gate_held_s": held}


def _levels(rows) -> list[str]:
    return [r["level"] for r in rows]


def _text(rows) -> str:
    return " ".join(r["text"] for r in rows)


# ---------------------------------------------------------------------------
# the run has to earn the right to credit the gate
# ---------------------------------------------------------------------------


def test_no_leak_with_the_gate_off_proves_nothing() -> None:
    """Silence is not evidence.

    A microphone that delivers nothing while the speakers play looks identical
    whether the gate is working, the speakers are muted, the output went to
    headphones, or the input device is the wrong one.
    """
    rows = _module().verdicts(
        _pass(floor=-95.0, during=-94.0, after=-95.0, delivered=4.0),
        _pass(floor=-95.0, during=-120.0, after=-120.0, delivered=0.0),
        250, 432.0,
    )
    assert _levels(rows) == ["fail"]
    assert "no loop was demonstrated" in _text(rows)
    assert "headphones" in _text(rows)


def test_a_real_loop_is_recognised_and_the_gate_credited() -> None:
    """Measured on this laptop: 36 dB above the floor, and nothing through the gate."""
    rows = _module().verdicts(
        _pass(floor=-93.6, during=-57.6, after=-111.8, delivered=4.0, peak=0.4),
        _pass(floor=-99.7, during=-120.0, after=-126.4, delivered=0.0, peak=0.0),
        250, 432.0,
    )
    assert "fail" not in _levels(rows)
    assert "36.0 dB above the room floor" in _text(rows)
    assert "loop is closed" in _text(rows)


def test_a_leak_too_quiet_to_transcribe_is_flagged_not_celebrated() -> None:
    """A measurable rise is not the same as a dangerous one.

    Measured here at one point: the leak sat 19 dB above the floor but peaked
    at 0.0011, far below the 0.02 the capture path treats as speech. The gate
    was blocking something that would never have reached the recogniser, which
    is not the test anyone needs passed.
    """
    rows = _module().verdicts(
        _pass(floor=-99.9, during=-81.0, after=-110.0, delivered=4.0, peak=0.0011),
        _pass(floor=-99.9, during=-120.0, after=-120.0, delivered=0.0, peak=0.0),
        250, 432.0,
    )
    assert "warn" in _levels(rows)
    assert "would not have been transcribed" in _text(rows)
    assert "Raise the system volume" in _text(rows)


def test_thresholds_are_relative_not_absolute() -> None:
    """The regression this had.

    An earlier version failed any run whose levels sat below -75 dBFS, and
    called a clear 19 dB rise "nothing at all" - because the Intel array's noise
    suppression puts a quiet room at about -100 dBFS. The judgement has to be
    against the room, not against a number chosen on different hardware.
    """
    quiet = _module().verdicts(
        _pass(floor=-99.0, during=-63.0, after=-110.0, peak=0.4),
        _pass(floor=-99.0, during=-120.0, after=-120.0, delivered=0.0),
        250, 432.0,
    )
    loud = _module().verdicts(
        _pass(floor=-60.0, during=-24.0, after=-71.0, peak=0.4),
        _pass(floor=-60.0, during=-81.0, after=-81.0, delivered=0.0),
        250, 432.0,
    )
    assert "fail" not in _levels(quiet), "a quiet room must not fail on its floor alone"
    assert _levels(quiet) == _levels(loud), "36 dB is 36 dB at any absolute level"


# ---------------------------------------------------------------------------
# the failures worth catching
# ---------------------------------------------------------------------------


def test_a_gate_that_does_not_block_is_a_failure() -> None:
    rows = _module().verdicts(
        _pass(floor=-93.0, during=-57.0, after=-110.0, peak=0.4),
        _pass(floor=-93.0, during=-60.0, after=-110.0, delivered=4.0, peak=0.4),
        250, 432.0,
    )
    assert "fail" in _levels(rows)
    assert "half_duplex is true" in _text(rows)


def test_audio_still_arriving_after_the_gate_reopens_is_a_failure() -> None:
    """This is the whole reason pipeline.output_latency_ms exists.

    The gate is sized from output_latency_ms + half_duplex_tail_ms. If sound is
    still coming out of the speakers when the microphone comes back, the number
    is wrong for this room, and the system will start transcribing itself.
    """
    rows = _module().verdicts(
        _pass(floor=-93.0, during=-57.0, after=-110.0, peak=0.4),
        _pass(floor=-93.0, during=-120.0, after=-70.0, delivered=0.0),
        250, 432.0,
    )
    assert "fail" in _levels(rows)
    assert "too short for this room" in _text(rows)
    assert "half_duplex_tail_ms" in _text(rows)


def test_an_unmeasured_output_latency_is_called_out_even_on_a_pass() -> None:
    rows = _module().verdicts(
        _pass(floor=-93.0, during=-57.0, after=-110.0, peak=0.4),
        _pass(floor=-93.0, during=-120.0, after=-120.0, delivered=0.0),
        250, 0.0,
    )
    assert "fail" not in _levels(rows)
    assert "output_latency_ms is 0" in _text(rows)


# ---------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------


def test_dbfs_of_silence_does_not_return_negative_infinity() -> None:
    module = _module()
    assert module.dbfs(np.zeros(0, dtype=np.float32)) == -120.0
    assert np.isfinite(module.dbfs(np.zeros(1000, dtype=np.float32)))


def test_dbfs_of_a_full_scale_tone_is_about_minus_three() -> None:
    module = _module()
    tone = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000).astype(np.float32)
    assert module.dbfs(tone) == pytest.approx(-3.0, abs=0.1)


def test_the_probe_signal_is_normalised_and_band_limited() -> None:
    module = _module()
    signal = module.speech_band_signal(1.0, 24000)
    assert signal.dtype == np.float32
    assert float(np.abs(signal).max()) == pytest.approx(0.6, abs=0.01)

    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / 24000)
    out_of_band = spectrum[(freqs > 5000)].max()
    in_band = spectrum[(freqs > 500) & (freqs < 3000)].max()
    assert out_of_band < in_band * 0.01
