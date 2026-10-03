"""Codec-cutoff detection, validated against signals of known bandwidth.

Why this file exists
--------------------
The first version of this detector thresholded the spectrum at -25 dB below its
peak. Speech falls naturally at roughly -6 to -12 dB per octave, so by 8 kHz it
already sits 30-40 dB under its 300 Hz peak. The rule therefore measured the
*talker's* spectral tilt, not the *microphone's* bandwidth, and reported a clean
48 kHz capture as rolling off at 2.2 kHz — a confident, precise, wrong number.

The replacement looks for the cliff a codec leaves behind (tens of dB across a
fraction of an octave) instead of an absolute level. These tests pin that
distinction down with synthetic signals, so the detector is never again judged
only by whatever hardware happened to be plugged in.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 20260829


@pytest.fixture(scope="module")
def mod():
    path = REPO_ROOT / "scripts" / "measure_audio_device.py"
    spec = importlib.util.spec_from_file_location("measure_audio_device", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_audio_device"] = module
    spec.loader.exec_module(module)
    return module


def speech_like(rate: int, seconds: float = 5.0, tilt_db_per_octave: float = -9.0) -> np.ndarray:
    """Noise shaped like speech: full-band, with a natural downward tilt.

    This is the signal that broke the old detector. It has real energy all the
    way to Nyquist, so any honest bandwidth measurement must call it full-band.
    """
    rng = np.random.default_rng(SEED)
    n = int(rate * seconds)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    ref = 300.0
    with np.errstate(divide="ignore"):
        octaves = np.log2(np.maximum(freqs, 1e-6) / ref)
    gain = 10 ** ((tilt_db_per_octave * octaves) / 20.0)
    gain[freqs < 80] = 0.0  # no DC or rumble
    shaped = np.fft.irfft(spectrum * gain, n)
    shaped /= max(float(np.abs(shaped).max()), 1e-9)
    return (shaped * 0.5).astype(np.float32)


def band_limit(signal: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    """Brick-wall low-pass, standing in for a codec cutoff."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1.0 / rate)
    spectrum[freqs > cutoff_hz] = 0.0
    out = np.fft.irfft(spectrum, signal.size)
    return (out / max(float(np.abs(out).max()), 1e-9) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# the regression
# ---------------------------------------------------------------------------


def test_full_band_speech_is_not_called_narrowband(mod) -> None:
    """The exact failure that produced 'rolls off at 2227 Hz' on a 48 kHz mic."""
    rate = 48000
    result = mod.analyse_bandwidth(speech_like(rate), rate)

    assert "error" not in result
    assert result["cliff_hz"] is None, f"invented a cliff at {result['cliff_hz']} Hz"
    assert result["usable_hz"] == pytest.approx(24000.0)
    assert result["covers_whisper_band"] is True
    assert result["usable_hz"] > 8000


@pytest.mark.parametrize("tilt", [-3.0, -6.0, -9.0, -12.0, -15.0])
def test_no_cliff_invented_at_any_plausible_speech_tilt(mod, tilt: float) -> None:
    rate = 48000
    result = mod.analyse_bandwidth(speech_like(rate, tilt_db_per_octave=tilt), rate)
    assert result["cliff_hz"] is None, f"tilt {tilt} dB/oct misread as a cliff"
    assert result["covers_whisper_band"] is True


# ---------------------------------------------------------------------------
# real cutoffs must still be caught
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutoff", [3400.0, 7000.0, 11000.0])
def test_codec_cliff_is_located(mod, cutoff: float) -> None:
    rate = 48000
    signal = band_limit(speech_like(rate), rate, cutoff)
    result = mod.analyse_bandwidth(signal, rate)

    assert result["cliff_hz"] is not None, f"missed a brick wall at {cutoff} Hz"
    assert result["cliff_hz"] == pytest.approx(cutoff, rel=0.25)
    assert result["usable_hz"] < rate / 2


def test_narrowband_telephony_verdict(mod) -> None:
    rate = 48000
    result = mod.analyse_bandwidth(band_limit(speech_like(rate), rate, 3400.0), rate)
    assert result["covers_whisper_band"] is False
    assert "narrowband" in result["verdict"]


def test_wideband_hfp_verdict(mod) -> None:
    rate = 48000
    result = mod.analyse_bandwidth(band_limit(speech_like(rate), rate, 7000.0), rate)
    assert result["covers_whisper_band"] is False
    assert "wideband" in result["verdict"]


# ---------------------------------------------------------------------------
# sample rate is a hard ceiling, no measurement required
# ---------------------------------------------------------------------------


def test_16k_capture_is_capped_at_the_whisper_band(mod) -> None:
    """An AirPods HFP microphone reports 16 kHz native, i.e. 8 kHz Nyquist.

    16 kHz is Whisper's own rate, so a clean 16 kHz capture is acceptable - the
    problem with HFP is that the codec cuts well below Nyquist, which the
    narrowband/wideband tests cover.
    """
    rate = 16000
    result = mod.analyse_bandwidth(speech_like(rate), rate)
    assert result["nyquist_hz"] == 8000.0
    assert result["usable_hz"] <= 8000.0


def test_anti_alias_skirt_is_not_treated_as_a_codec_limit(mod) -> None:
    """A real 16 kHz chain filters at 7.6-7.9 kHz. That must still count as covered.

    Judging coverage at a strict 8000 Hz would fail the exact audio format
    Whisper was trained on.
    """
    rate = 48000
    result = mod.analyse_bandwidth(band_limit(speech_like(rate), rate, 7800.0), rate)
    assert result["covers_whisper_band"] is True


def test_hfp_wideband_is_still_rejected(mod) -> None:
    """7 kHz is a codec limit, not a filter skirt, and must not slip through."""
    rate = 48000
    result = mod.analyse_bandwidth(band_limit(speech_like(rate), rate, 7000.0), rate)
    assert result["covers_whisper_band"] is False


def test_usable_is_capped_by_nyquist_not_the_cliff(mod) -> None:
    rate = 16000
    result = mod.analyse_bandwidth(band_limit(speech_like(rate), rate, 7000.0), rate)
    assert result["usable_hz"] <= 8000.0


# ---------------------------------------------------------------------------
# level reporting
# ---------------------------------------------------------------------------


def test_quiet_signal_is_flagged(mod) -> None:
    rate = 48000
    result = mod.analyse_bandwidth(speech_like(rate) * 0.02, rate)
    assert result["quiet"] is True
    assert result["clipping"] is False


def test_loud_signal_is_not_flagged_quiet(mod) -> None:
    rate = 48000
    result = mod.analyse_bandwidth(speech_like(rate), rate)
    assert result["quiet"] is False


def test_octave_levels_descend_with_a_tilted_source(mod) -> None:
    rate = 48000
    levels = list(mod.analyse_bandwidth(speech_like(rate), rate)["octave_levels_db"].values())
    assert len(levels) >= 5
    assert levels[0] > levels[-1], "tilted noise should decrease across octaves"


def test_short_signal_reports_an_error_not_a_verdict(mod) -> None:
    result = mod.analyse_bandwidth(np.zeros(500, dtype=np.float32), 48000)
    assert "error" in result
    assert "covers_whisper_band" not in result


# ---------------------------------------------------------------------------
# upsampled sources: the container rate is not the real rate
# ---------------------------------------------------------------------------


def upsampled(rate_out: int, rate_in: int, seconds: float = 5.0) -> np.ndarray:
    """Speech-like noise band-limited to *rate_in* but delivered at *rate_out*.

    This is what a laptop voice DSP or a Bluetooth HFP link produces: a
    container claiming 48 kHz whose content stops dead at the source Nyquist,
    with a flat dither floor above it.
    """
    return band_limit(speech_like(rate_out, seconds), rate_out, rate_in / 2 * 0.98)


def test_16k_source_in_a_48k_container_is_identified(mod) -> None:
    """The observed case: 'Microphone Array' reported 48 kHz, content stopped at 8 kHz."""
    result = mod.analyse_bandwidth(upsampled(48000, 16000), 48000)
    assert result["upsampled"] is True
    assert result["effective_source_rate_hz"] == 16000
    assert result["nyquist_hz"] == 24000.0
    # Still covers what Whisper consumes - the flag is about honesty, not doom.
    assert result["covers_whisper_band"] is True


def test_8k_source_is_identified_and_disqualifying(mod) -> None:
    result = mod.analyse_bandwidth(upsampled(48000, 8000), 48000)
    assert result["upsampled"] is True
    assert result["effective_source_rate_hz"] == 8000
    assert result["covers_whisper_band"] is False


def test_genuine_full_rate_capture_is_not_flagged(mod) -> None:
    result = mod.analyse_bandwidth(speech_like(48000), 48000)
    assert result["upsampled"] is False
    assert result["effective_source_rate_hz"] is None


def test_native_16k_device_is_not_called_upsampled(mod) -> None:
    """A device that honestly runs at 16 kHz is not misreporting anything."""
    result = mod.analyse_bandwidth(speech_like(16000), 16000)
    assert result["upsampled"] is False


def test_analysis_is_deterministic(mod) -> None:
    """The same signal must not yield two different verdicts.

    The first implementation located the cutoff as the single steepest slope.
    A recording with two comparable transitions - the codec edge and the descent
    into the quantisation floor - flipped between them, so a live measurement
    said 15633 Hz and re-analysing the saved file said 8742 Hz.
    """
    signal = upsampled(48000, 16000)
    first = mod.analyse_bandwidth(signal, 48000)
    second = mod.analyse_bandwidth(signal.copy(), 48000)
    assert first["cliff_hz"] == second["cliff_hz"]
    assert first["effective_source_rate_hz"] == second["effective_source_rate_hz"]


def test_survives_16_bit_quantisation(mod) -> None:
    """A dither floor near -96 dBFS must not move the verdict.

    Recordings are saved as float now, but any 16-bit source must still analyse
    the same way, or the tool disagrees with itself depending on file format.
    """
    signal = upsampled(48000, 16000)
    quantised = (np.round(signal * 32767.0) / 32767.0).astype(np.float32)

    clean = mod.analyse_bandwidth(signal, 48000)
    dithered = mod.analyse_bandwidth(quantised, 48000)

    assert dithered["effective_source_rate_hz"] == clean["effective_source_rate_hz"]
    assert dithered["cliff_hz"] == pytest.approx(clean["cliff_hz"], rel=0.15)


def test_wav_round_trip_preserves_the_verdict(mod, tmp_path: Path) -> None:
    """Analysing a saved recording must match analysing the live signal."""
    import soundfile as sf

    signal = upsampled(48000, 16000)
    live = mod.analyse_bandwidth(signal, 48000)

    path = tmp_path / "rec.wav"
    sf.write(path, signal, 48000, subtype="FLOAT")
    reloaded, rate = sf.read(path, dtype="float32")
    saved = mod.analyse_bandwidth(np.asarray(reloaded, dtype=np.float32), rate)

    assert saved["cliff_hz"] == live["cliff_hz"]
    assert saved["effective_source_rate_hz"] == live["effective_source_rate_hz"]
    assert saved["usable_hz"] == live["usable_hz"]


def test_flat_floor_and_tilt_are_reported(mod) -> None:
    """The two quantities the decision rests on must be visible in the output."""
    band_limited = mod.analyse_bandwidth(upsampled(48000, 16000), 48000)
    full_band = mod.analyse_bandwidth(speech_like(48000), 48000)

    # A dither floor is flat; natural decay keeps sloping.
    assert abs(band_limited["hf_tilt_db_per_octave"]) < 12.0
    assert full_band["hf_tilt_db_per_octave"] < -3.0


def test_upsampling_is_reported_to_the_user(mod) -> None:
    report = {
        "bandwidth": {
            "native_rate": 48000, "nyquist_hz": 24000.0, "cliff_hz": 8000.0,
            "usable_hz": 8000.0, "covers_whisper_band": True, "upsampled": True,
            "effective_source_rate_hz": 16000, "clipping": False, "quiet": False,
            "peak": 0.14, "rms_dbfs": -35.6, "likely_bluetooth": False,
        }
    }
    joined = " ".join(mod.build_recommendations(report))
    assert "real capture path is 16000 Hz" in joined
    assert "does not disqualify" in joined
