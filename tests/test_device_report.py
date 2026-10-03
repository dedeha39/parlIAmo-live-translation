"""Reporting logic for the device measurement tool.

Regression coverage for a real misreport: a bandwidth probe that captured
silence produced ``verdict: full-band``, ``covers_whisper_band: true`` and the
recommendation "No problems detected", from a recording whose peak amplitude was
0.0003. A measurement tool that says a device is fine when it measured nothing
is worse than no tool.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    """Load scripts/measure_audio_device.py as a module."""
    path = REPO_ROOT / "scripts" / "measure_audio_device.py"
    spec = importlib.util.spec_from_file_location("measure_audio_device", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_audio_device"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# failed probes must not read as passes
# ---------------------------------------------------------------------------


def test_failed_bandwidth_is_not_reported_as_healthy(mod) -> None:
    report = {
        "bandwidth": {
            "error": "recording was effectively silent - no speech captured",
            "hint": "wait for the SPEAK NOW prompt",
        }
    }
    lines = mod.build_recommendations(report)
    joined = " ".join(lines)
    assert "No problems detected" not in joined
    assert "NOT been assessed" in joined
    assert "wait for the SPEAK NOW prompt" in joined


def test_failed_roundtrip_is_surfaced(mod) -> None:
    report = {"roundtrip": {"error": "no confident measurement"}}
    joined = " ".join(mod.build_recommendations(report))
    assert "roundtrip probe did not produce a result" in joined
    assert "No problems detected" not in joined


def _bw(**overrides) -> dict:
    """A healthy bandwidth result, with fields overridden per test."""
    base = {
        "native_rate": 48000,
        "nyquist_hz": 24000.0,
        "cliff_hz": None,
        "usable_hz": 24000.0,
        "covers_whisper_band": True,
        "clipping": False,
        "quiet": False,
        "peak": 0.31,
        "rms_dbfs": -28.0,
        "likely_bluetooth": False,
    }
    base.update(overrides)
    return base


def test_clean_report_says_so(mod) -> None:
    report = {"bandwidth": _bw(), "roundtrip": {"median_ms": 60.0, "spread_ms": 8.0}}
    assert mod.build_recommendations(report) == [
        "No problems detected. This device is suitable for the pipeline."
    ]


# ---------------------------------------------------------------------------
# the Bluetooth verdict this project exists to make concrete
# ---------------------------------------------------------------------------


def test_narrowband_bluetooth_gets_a_hard_verdict(mod) -> None:
    """A brick wall below 8 kHz on a device that could physically go higher."""
    report = {
        "bandwidth": _bw(
            cliff_hz=3900.0, usable_hz=3900.0, covers_whisper_band=False,
            likely_bluetooth=True,
        )
    }
    joined = " ".join(mod.build_recommendations(report))
    assert "below the 8 kHz band" in joined
    assert "no software fix" in joined
    assert "wired" in joined


def test_16k_capture_is_explained_by_its_sample_rate(mod) -> None:
    """The AirPods case: the ceiling comes from the sample rate, not a filter.

    Blaming a 'roll-off' would be misleading - a 16 kHz capture cannot represent
    anything above 8 kHz no matter how good the microphone is.
    """
    report = {
        "bandwidth": _bw(
            native_rate=16000, nyquist_hz=8000.0, usable_hz=7000.0, cliff_hz=7000.0,
            covers_whisper_band=False, likely_bluetooth=True,
        )
    }
    joined = " ".join(mod.build_recommendations(report))
    assert "16000 Hz" in joined
    assert "cannot carry anything above" in joined
    assert "wired" in joined
    # Both ceilings must be reported, not just the first one hit.
    assert "7000 Hz" in joined
    assert "codec cutoff" in joined


def test_cliff_at_nyquist_is_not_double_reported(mod) -> None:
    """A 16 kHz device whose content simply runs to Nyquist has one problem, not two."""
    report = {
        "bandwidth": _bw(
            native_rate=16000, nyquist_hz=8000.0, usable_hz=8000.0, cliff_hz=None,
            covers_whisper_band=False, likely_bluetooth=True,
        )
    }
    lines = mod.build_recommendations(report)
    assert sum("codec cutoff" in line for line in lines) == 0
    assert any("cannot carry anything above" in line for line in lines)


def test_wideband_but_still_short_of_whisper(mod) -> None:
    report = {
        "bandwidth": _bw(
            cliff_hz=7000.0, usable_hz=7000.0, covers_whisper_band=False,
            likely_bluetooth=True,
        )
    }
    assert any("8 kHz" in line for line in mod.build_recommendations(report))


def test_clipping_and_quiet_signal_are_both_flagged(mod) -> None:
    clipping = mod.build_recommendations({"bandwidth": _bw(clipping=True, peak=1.0)})
    assert any("clipping" in line.lower() for line in clipping)

    quiet = mod.build_recommendations(
        {"bandwidth": _bw(quiet=True, peak=0.021, rms_dbfs=-55.0)}
    )
    assert any("quiet" in line.lower() for line in quiet)


def test_clipping_takes_priority_over_quiet(mod) -> None:
    lines = mod.build_recommendations({"bandwidth": _bw(clipping=True, quiet=True)})
    assert not any("quiet" in line.lower() for line in lines)


# ---------------------------------------------------------------------------
# upstream noise gating
# ---------------------------------------------------------------------------


def test_digital_silence_noise_floor_is_called_out(mod) -> None:
    """-104 dBFS is not a quiet room; it is a driver zeroing non-speech."""
    report = {"noise_floor": {"dbfs": -104.3, "noise_suppression_active": True}}
    joined = " ".join(mod.build_recommendations(report))
    assert "digital silence" in joined
    assert "audio enhancements" in joined


def test_realistic_noise_floor_is_not_flagged(mod) -> None:
    report = {
        "bandwidth": _bw(),
        "noise_floor": {"dbfs": -52.0, "noise_suppression_active": False},
        "roundtrip": {"median_ms": 60.0, "spread_ms": 8.0},
    }
    assert mod.build_recommendations(report) == [
        "No problems detected. This device is suitable for the pipeline."
    ]


def test_high_latency_and_jitter_are_flagged(mod) -> None:
    lines = mod.build_recommendations(
        {"roundtrip": {"median_ms": 515.0, "spread_ms": 120.0}}
    )
    joined = " ".join(lines)
    assert "515" in joined
    assert "jitter" in joined


# ---------------------------------------------------------------------------
# probe accounting
# ---------------------------------------------------------------------------


def test_probes_that_ran_counts_successes(mod) -> None:
    report = {
        "bandwidth": {"error": "silent"},
        "noise_floor": {"dbfs": -46.0},
        "roundtrip": {"median_ms": 60.0},
    }
    assert mod.probes_that_ran(report) == (2, 3)


def test_probes_that_ran_ignores_skipped(mod) -> None:
    assert mod.probes_that_ran({"noise_floor": {"dbfs": -46.0}}) == (1, 1)


# ---------------------------------------------------------------------------
# the chirp itself
# ---------------------------------------------------------------------------


def test_chirp_stays_inside_the_hfp_band(mod) -> None:
    """The probe must survive a 16 kHz Bluetooth link, or it cannot measure one."""
    import numpy as np

    rate = 16000
    chirp = mod._chirp(0.08, rate)
    spectrum = np.abs(np.fft.rfft(chirp * np.hanning(chirp.size)))
    freqs = np.fft.rfftfreq(chirp.size, 1.0 / rate)
    energy_above_4k = spectrum[freqs > 4200].sum()
    assert energy_above_4k / spectrum.sum() < 0.05, "chirp puts energy above the HFP band"


def test_chirp_is_tapered_to_avoid_speaker_clicks(mod) -> None:
    chirp = mod._chirp(0.08, 16000)
    assert abs(float(chirp[0])) < 0.01
    assert abs(float(chirp[-1])) < 0.01
    assert float(abs(chirp).max()) <= 0.61
