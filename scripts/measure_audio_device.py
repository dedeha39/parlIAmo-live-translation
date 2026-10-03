#!/usr/bin/env python
"""Measure what a microphone actually delivers, so device choice stops being an opinion.

Three probes, each answering a question that matters for this pipeline:

1. **Bandwidth**  - where does the microphone's usable frequency content stop?
   A Bluetooth headset in HFP mode walls off around 4 kHz (narrowband) or
   7-8 kHz (wideband mSBC). Whisper was trained on 16 kHz audio with content up
   to 8 kHz; losing the top half of that costs recognition accuracy on exactly
   the consonants Turkish uses to mark case and tense.

2. **Noise floor** - how loud is the room through this microphone when nobody
   is talking? Drives the VAD threshold.

3. **Round-trip latency** - play a chirp, hear it back, cross-correlate. This is
   the real acoustic loop: output buffer + air + input buffer. It is also a
   direct measurement of how fast feedback would build up without the gate.

Run once per candidate microphone and compare the reports:

    python scripts/measure_audio_device.py --seconds 5
    python scripts/measure_audio_device.py --input "AirPods" --skip-latency
    python scripts/measure_audio_device.py --input "lavalier" --compare-with runs/audio/...json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from parliamo.audio.capture import AudioCapture
from parliamo.audio.devices import resolve_device
from parliamo.logging_setup import configure_console
from parliamo.paths import ensure_dir, resolve

# Whisper-family models consume 16 kHz audio, so content above 8 kHz is
# discarded anyway. A microphone that cannot reach 8 kHz is losing information
# the model would otherwise have used.
USEFUL_BANDWIDTH_HZ = 8000

# Coverage is judged at 7600 Hz, not at 8000. Whisper's own native rate is
# 16 kHz, and every real 16 kHz capture chain puts its anti-alias filter at
# 7.6-7.9 kHz - so a strict ">= 8000" test would fail the exact format the
# model was trained on. Anything genuinely narrower than this is a codec
# limitation, not a filter skirt.
WHISPER_BAND_MIN_HZ = 7600
HFP_NARROWBAND_HZ = 4200
HFP_WIDEBAND_HZ = 7200


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


def probe_bandwidth(device_spec: int | str | None, seconds: float) -> dict[str, Any]:
    """Record at the device's native rate and find the spectral roll-off."""
    import sounddevice as sd

    device = resolve_device(device_spec, "input")
    rate = int(round(device.default_samplerate))
    channels = 1

    print(f"  device: {device.label}")
    print(f"  recording {seconds:.0f}s at {rate} Hz")
    for n in (3, 2, 1):
        print(f"  starting in {n} ...", end="\r", flush=True)
        time.sleep(1.0)
    print("  >>> SPEAK NOW, normally, as you would on stage <<<        ")

    frames = int(rate * seconds)
    recording = sd.rec(
        frames, samplerate=rate, channels=channels, dtype="float32", device=device.index
    )
    sd.wait()
    print("  done.")
    signal = np.asarray(recording, dtype=np.float32).reshape(-1)

    if signal.size == 0:
        return {"error": "no audio captured"}

    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.abs(signal).max())
    rms_dbfs = 20 * np.log10(max(rms, 1e-9))

    # Guard against reporting a roll-off derived from silence. The spectrum of
    # a silent recording is the noise floor's spectrum, which happily produces
    # a confident-looking "full-band" verdict from nothing at all. Intel Smart
    # Sound in particular gates non-speech down to digital silence, so this is
    # not a hypothetical failure.
    if peak < 0.005 or rms_dbfs < -70:
        return {
            "error": "recording was effectively silent - no speech captured",
            "device": device.label,
            "native_rate": rate,
            "rms_dbfs": round(rms_dbfs, 1),
            "peak": round(peak, 6),
            "hint": (
                "wait for the SPEAK NOW prompt and talk for the whole countdown; "
                "if it stays silent, the wrong input device is selected - pass "
                "--input with a name substring"
            ),
        }

    analysis = analyse_bandwidth(signal, rate)
    if "error" in analysis:
        return {**analysis, "device": device.label, "native_rate": rate}

    return {
        "device": device.label,
        "native_rate": rate,
        **analysis,
        "likely_bluetooth": device.likely_bluetooth,
        "recording": _save_wav(signal, rate, device.name),
    }


def analyse_bandwidth(signal: np.ndarray, rate: int) -> dict[str, Any]:
    """Locate a codec cutoff in *signal*, independently of how it was captured.

    Pure function so the detector can be tested against synthetic signals with
    known bandwidth, rather than only against whatever hardware is to hand.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0
    peak = float(np.abs(signal).max()) if signal.size else 0.0
    rms_dbfs = 20 * np.log10(max(rms, 1e-9))

    # Average magnitude spectrum over Hann-windowed frames, ignoring the
    # quietest half so that silence does not dominate the average.
    frame = 2048
    hop = 1024
    if signal.size < frame * 2:
        return {"error": f"need at least {frame * 2} samples to analyse, got {signal.size}"}
    window = np.hanning(frame).astype(np.float32)
    n_frames = max(1, (signal.size - frame) // hop)
    powers = np.zeros((n_frames, frame // 2 + 1), dtype=np.float64)
    energies = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = signal[i * hop : i * hop + frame]
        if seg.size < frame:
            break
        seg = seg * window
        energies[i] = float(np.sum(seg**2))
        powers[i] = np.abs(np.fft.rfft(seg)) ** 2

    if n_frames > 4:
        loud = energies >= np.median(energies)
        spectrum = powers[loud].mean(axis=0)
    else:
        spectrum = powers.mean(axis=0)

    freqs = np.fft.rfftfreq(frame, 1.0 / rate)
    # Ignore mains hum and DC when locating the peak.
    band = freqs >= 150
    if not band.any() or spectrum[band].max() <= 0:
        return {"error": "signal too quiet to analyse", "rate": rate}

    db = 10 * np.log10(np.maximum(spectrum, 1e-20))
    db -= db[band].max()
    # Smooth with *edge* padding, not zero padding. These are dB values, so
    # every entry is negative and np.convolve(mode="same") pads with 0 dB - the
    # peak level - which lifts both ends of the spectrum into a fake plateau.
    # That plateau sat exactly where the cutoff search looks, so the detector
    # reported the top of the frequency axis as the cutoff every time.
    kernel = np.ones(9) / 9.0
    pad = kernel.size // 2
    smooth = np.convolve(np.pad(db, pad, mode="edge"), kernel, mode="valid")

    nyquist = rate / 2.0

    # Finding the cutoff by thresholding against the spectral peak does not
    # work, and the first version of this tool got it wrong. Speech has a
    # natural tilt of roughly -6 to -12 dB per octave, so at 8 kHz it already
    # sits 30-40 dB below its 300 Hz peak. Any fixed "within N dB of peak" rule
    # therefore measures the talker, not the microphone, and reports a perfect
    # 48 kHz capture as rolling off at 2 kHz.
    #
    # What actually distinguishes a band-limited device is the *shape* of the
    # transition: a codec cutoff is a cliff, tens of dB across a fraction of an
    # octave, while speech decays gently. So look for the cliff.
    # Locating the cutoff as the single steepest point is not stable: a
    # recording can contain two comparable transitions - the real codec edge and
    # the descent into the quantisation floor - and floating-point noise decides
    # which one wins. The same file analysed twice then gives two answers, which
    # is exactly what happened on the first pair of runs.
    #
    # Ask a better-posed question instead: is there a noise floor up there, and
    # if so where does real content stop? A noise floor is *flat* (dither and
    # quantisation are spectrally white) and *deep*. Natural spectral decay is
    # neither - it keeps sloping all the way to Nyquist.
    ref_mask = band & (freqs >= 300) & (freqs <= 3000)
    top_mask = band & (freqs >= nyquist * 0.60) & (freqs <= nyquist * 0.95)

    cliff_hz: float | None = None
    max_slope = 0.0
    floor_db = 0.0
    top_tilt = 0.0

    if ref_mask.any() and top_mask.sum() > 8:
        ref_db = float(np.median(smooth[ref_mask]))
        floor_db = float(np.median(smooth[top_mask]))
        depth = ref_db - floor_db

        # Tilt of the top band, in dB per octave.
        top_f = np.log2(freqs[top_mask])
        top_c = smooth[top_mask]
        span = float(top_f[-1] - top_f[0])
        top_tilt = float((top_c[-1] - top_c[0]) / span) if span > 1e-6 else 0.0

        # Dither and quantisation noise are spectrally white, so a real noise
        # floor sits within a couple of dB per octave of flat. Speech-shaped
        # decay measures -6 to -15 dB/octave up here, so 5 separates them
        # cleanly while leaving room for a floor that is not perfectly white.
        flat_floor = abs(top_tilt) < 5.0
        deep_floor = depth > 30.0

        if flat_floor and deep_floor:
            # Content ends at the highest frequency still clearly above the
            # floor. Anchor the threshold to the in-band level rather than to
            # the floor: with a floor 140 dB down, "a bit above the floor" lands
            # deep in the window's leakage skirt and overshoots the true edge by
            # 50%. Sitting 35 dB under the in-band level tracks the real cutoff,
            # and the floor term keeps it meaningful when the floor is shallow.
            threshold = max(floor_db + 10.0, ref_db - 35.0)
            candidates = np.where(band & (freqs >= 500) & (smooth > threshold))[0]
            if candidates.size:
                cliff_hz = float(freqs[candidates[-1]])

        # Kept as a diagnostic only - no longer used to decide anything.
        idx = np.where(band & (freqs >= 400) & (freqs <= nyquist * 0.97))[0]
        if idx.size > 8:
            log_f = np.log2(freqs[idx])
            curve = smooth[idx]
            step = max(2, idx.size // 64)
            slopes = (curve[step:] - curve[:-step]) / np.maximum(
                log_f[step:] - log_f[:-step], 1e-9
            )
            if slopes.size:
                max_slope = float(slopes.min())

    usable_hz = min(nyquist, cliff_hz) if cliff_hz else nyquist

    # A device can report 48 kHz while the audio reaching it came through a
    # 16 kHz path and was upsampled - laptop voice DSP pipelines and Bluetooth
    # HFP links both do this. The giveaway is a brick wall at exactly half a
    # common capture rate, with a dead-flat dither floor above it. Naming the
    # effective rate turns "the spectrum looks odd" into "you are not getting
    # the sample rate you asked for".
    effective_rate: int | None = None
    if cliff_hz:
        for candidate in (8000, 16000, 22050, 24000, 32000, 44100, 48000):
            if candidate >= rate:
                break
            if abs(cliff_hz - candidate / 2) <= candidate * 0.09:
                effective_rate = candidate
                break

    # Octave-band levels, so a human can sanity-check the verdict instead of
    # trusting a single derived number.
    octaves: dict[str, float] = {}
    edges = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    for lo_hz, hi_hz in zip(edges[:-1], edges[1:], strict=True):
        if lo_hz >= nyquist:
            break
        sel = (freqs >= lo_hz) & (freqs < min(hi_hz, nyquist))
        if sel.any():
            octaves[f"{lo_hz}-{min(hi_hz, int(nyquist))}"] = round(
                float(smooth[sel].mean()), 1
            )

    if usable_hz <= HFP_NARROWBAND_HZ:
        verdict = "narrowband (HFP 8 kHz-class) - unusable for accurate ASR"
    elif usable_hz <= HFP_WIDEBAND_HZ:
        verdict = "wideband (HFP mSBC-class) - degraded, Bluetooth profile active"
    elif usable_hz < WHISPER_BAND_MIN_HZ:
        verdict = "below the 8 kHz Whisper band"
    else:
        verdict = "covers the 8 kHz Whisper band"

    return {
        "nyquist_hz": round(nyquist, 1),
        "cliff_hz": round(cliff_hz, 1) if cliff_hz else None,
        "hf_floor_db": round(floor_db, 1),
        "hf_tilt_db_per_octave": round(top_tilt, 1),
        "effective_source_rate_hz": effective_rate,
        "upsampled": bool(effective_rate is not None and effective_rate < rate),
        "steepest_slope_db_per_octave": round(max_slope, 1),
        "usable_hz": round(usable_hz, 1),
        "covers_whisper_band": bool(usable_hz >= WHISPER_BAND_MIN_HZ),
        "verdict": verdict,
        "octave_levels_db": octaves,
        "rms": round(rms, 5),
        "rms_dbfs": round(rms_dbfs, 1),
        "peak": round(peak, 4),
        "quiet": bool(peak < 0.05),
        "clipping": bool(peak >= 0.999),
    }


def _save_wav(signal: np.ndarray, rate: int, device_name: str) -> str | None:
    """Keep the raw recording so a disputed verdict can be re-examined."""
    try:
        import soundfile as sf

        out_dir = ensure_dir("runs/audio")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in device_name)[:40]
        path = out_dir / f"rec-{safe}-{datetime.now():%Y%m%d-%H%M%S}.wav"
        # 32-bit float, not soundfile's PCM_16 default. Quantising to 16 bits
        # plants a dither floor around -96 dBFS, which sits right where we look
        # for a codec floor - so the saved file analysed differently from the
        # live signal it came from, and the tool contradicted itself.
        sf.write(path, signal, rate, subtype="FLOAT")
        return str(path)
    except Exception:  # pragma: no cover - diagnostics must not break the probe
        return None


def probe_noise_floor(device_spec: int | str | None, seconds: float = 3.0) -> dict[str, Any]:
    """RMS of the room with nobody speaking - sets the VAD threshold."""
    print(f"  measuring noise floor for {seconds:.0f}s")
    for n in (3, 2, 1):
        print(f"  starting in {n} ...", end="\r", flush=True)
        time.sleep(1.0)
    print("  >>> STAY SILENT <<<                    ")
    with AudioCapture(device=device_spec, gate=None) as cap:
        time.sleep(0.3)
        cap.drain()
        audio = cap.record(seconds)

    if audio.size == 0:
        return {"error": "no audio captured"}

    rms = float(np.sqrt(np.mean(audio**2)))
    dbfs = 20 * np.log10(max(rms, 1e-9))
    # 20 ms windows: the 95th percentile catches intermittent noise (fans
    # cycling, a cough) that a global RMS would average away.
    win = 320
    n = audio.size // win
    frame_rms = (
        np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1)) if n else np.array([rms])
    )
    # A real microphone in a real room lands somewhere around -70 to -45 dBFS.
    # Anything below about -90 dBFS is not a quiet room, it is digital silence:
    # the driver is gating non-speech to zero. Windows "audio enhancements" and
    # Intel Smart Sound noise suppression both do this. It matters here because
    # a gate upstream of ours chews the onsets off words, and because it makes
    # the measured noise floor useless for setting a VAD threshold.
    gated = dbfs < -90.0

    return {
        "rms": round(rms, 6),
        "dbfs": round(dbfs, 1),
        "p95_dbfs": round(float(20 * np.log10(max(np.percentile(frame_rms, 95), 1e-9))), 1),
        "max_dbfs": round(float(20 * np.log10(max(frame_rms.max(), 1e-9))), 1),
        "noise_suppression_active": bool(gated),
        "suggested_vad_threshold": 0.5 if dbfs < -45 else 0.65,
    }


def _chirp(duration_s: float, rate: int, f0: float = 300.0, f1: float = 3800.0) -> np.ndarray:
    """Linear chirp confined below 4 kHz so it survives even an HFP link."""
    t = np.linspace(0, duration_s, int(rate * duration_s), endpoint=False, dtype=np.float64)
    k = (f1 - f0) / duration_s
    sig = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    # Taper the ends so the loudspeaker does not click, which would bias the
    # correlation peak toward the transient instead of the chirp body.
    ramp = int(0.005 * rate)
    if ramp > 0 and sig.size > 2 * ramp:
        envelope = np.ones_like(sig)
        envelope[:ramp] = np.linspace(0, 1, ramp)
        envelope[-ramp:] = np.linspace(1, 0, ramp)
        sig *= envelope
    return (sig * 0.6).astype(np.float32)


def probe_roundtrip_latency(
    input_spec: int | str | None,
    output_spec: int | str | None,
    repeats: int = 5,
) -> dict[str, Any]:
    """Play a chirp and record it back on a *duplex* stream, then cross-correlate.

    The measurement only means anything if playback and capture share a clock.
    Two independent streams do not: the recording buffer starts at an arbitrary
    point relative to the first output sample, and the correlation peak then
    measures that arbitrary offset rather than the acoustic delay. PortAudio's
    duplex mode (``sounddevice.playrec``) drives both directions from one
    callback, which is what makes the number trustworthy.
    """
    import sounddevice as sd

    in_dev = resolve_device(input_spec, "input")
    out_dev = resolve_device(output_spec, "output")

    # Duplex needs a rate both endpoints accept; prefer the shared native rate.
    candidates = [
        int(round(in_dev.default_samplerate)),
        int(round(out_dev.default_samplerate)),
        48000,
        44100,
        16000,
    ]
    rate = None
    for candidate in dict.fromkeys(candidates):
        try:
            sd.check_input_settings(device=in_dev.index, channels=1, samplerate=candidate, dtype="float32")
            sd.check_output_settings(device=out_dev.index, channels=1, samplerate=candidate, dtype="float32")
            rate = candidate
            break
        except Exception:
            continue
    if rate is None:
        return {"error": "no sample rate accepted by both devices in duplex mode"}

    lead_s = 0.20          # silence before the chirp: lets the stream settle
    chirp = _chirp(0.08, rate)
    lead = np.zeros(int(rate * lead_s), dtype=np.float32)
    tail = np.zeros(int(rate * 0.60), dtype=np.float32)
    probe = np.concatenate([lead, chirp, tail]).reshape(-1, 1)
    chirp_offset = lead.size

    reported = in_dev.latency_ms("input") + out_dev.latency_ms("output")
    print(f"  in : {in_dev.label} (reported {in_dev.latency_ms('input')} ms)")
    print(f"  out: {out_dev.label} (reported {out_dev.latency_ms('output')} ms)")
    print(f"  duplex @ {rate} Hz; driver round-trip floor ~{reported:.0f} ms")
    print(f"  playing {repeats} chirps - keep the room quiet ...")

    measurements: list[float] = []
    for i in range(repeats):
        try:
            recorded = sd.playrec(
                probe,
                samplerate=rate,
                channels=1,
                dtype="float32",
                device=(in_dev.index, out_dev.index),
                blocking=True,
            )
        except Exception as exc:
            return {"error": f"duplex stream failed: {type(exc).__name__}: {exc}"}

        signal = np.asarray(recorded, dtype=np.float32).reshape(-1)
        if signal.size < chirp.size * 2:
            print(f"    chirp {i + 1}: too little audio captured, skipped")
            continue

        n = 1 << int(np.ceil(np.log2(signal.size + chirp.size)))
        corr = np.abs(
            np.fft.irfft(np.fft.rfft(signal, n) * np.conj(np.fft.rfft(chirp, n)), n)
        )[: signal.size]

        # The echo cannot arrive before the chirp was emitted, and anything
        # beyond ~600 ms is a room reflection or unrelated noise.
        lo = chirp_offset
        hi = min(corr.size, chirp_offset + int(rate * 0.6))
        window = corr[lo:hi]
        if window.size == 0:
            continue
        peak_idx = int(np.argmax(window))
        peak_val = float(window[peak_idx])
        noise = float(np.median(corr)) or 1e-12
        snr = peak_val / noise

        delay_ms = peak_idx / rate * 1000.0
        confident = snr >= 12.0 and delay_ms > 0.5
        print(
            f"    chirp {i + 1}: {delay_ms:7.1f} ms  (corr SNR {snr:8.1f}) "
            f"{'ok' if confident else 'low confidence'}"
        )
        if confident:
            measurements.append(delay_ms)
        time.sleep(0.15)

    if not measurements:
        return {
            "error": "no confident measurement - is the output audible to the microphone?",
            "hint": "raise the speaker volume, or pass --skip-latency when monitoring on headphones",
            "duplex_rate": rate,
        }

    arr = np.array(measurements)
    return {
        "samples": len(measurements),
        "duplex_rate": rate,
        "driver_floor_ms": round(reported, 1),
        "median_ms": round(float(np.median(arr)), 1),
        "min_ms": round(float(arr.min()), 1),
        "max_ms": round(float(arr.max()), 1),
        "spread_ms": round(float(arr.max() - arr.min()), 1),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def write_output_latency(
    report: dict[str, Any], config_path: Path, tail_ms: int = 250
) -> dict[str, Any]:
    """Write the measured round-trip into ``pipeline.output_latency_ms``.

    This step used to be a sentence in a report and a manual edit afterwards,
    which is exactly the step that gets skipped half an hour before a talk. It
    is also the one the pre-flight list calls the only remaining item that can
    ruin the demonstration outright: with ``output_latency_ms`` at 0 the gate is
    sized from PortAudio's reported latency, which excludes the vendor DSP - 3
    ms reported against ~430 ms real on this laptop - so the microphone reopens
    while the speakers are still talking and the system translates its own
    output.

    The measured *median* is used rather than the maximum: the gate already
    carries ``half_duplex_tail_ms`` on top, and sizing from an outlier would
    hold the microphone shut through the start of the next sentence.

    That only holds while the tail actually covers the spread, so the result
    says whether it does. Measured twice on the reference laptop's speakers:
    medians 432 and 366 ms, with chirps ranging 229-552 within a single run.
    A median of 366 plus a 250 ms tail leaves 64 ms over the worst chirp seen -
    covered, but not by much, and worth knowing before the room is full rather
    than after the system starts translating itself.

    **This edits the file as text, not by re-serialising it.** A YAML
    round-trip is one line shorter and destroys every comment in the file - and
    ``config/local.yaml`` is where the *consent record* for the cloned voice
    lives, with a paragraph explaining whose voice it is and when it was
    recorded. Losing that to a convenience function is not a trade this project
    can make. Only the one line changes; everything else is left byte for byte.
    """
    roundtrip = report.get("roundtrip") or {}
    if "error" in roundtrip or "median_ms" not in roundtrip:
        return {"written": False,
                "reason": roundtrip.get("error", "no round-trip measurement in this report")}

    measured = int(round(float(roundtrip["median_ms"])))
    provenance = (
        f"measured {datetime.now():%Y-%m-%d %H:%M} from {roundtrip['samples']} chirps"
        f", median {roundtrip['median_ms']} ms, spread {roundtrip.get('spread_ms', '?')} ms"
    )

    # newline="" so the original line endings survive the read.
    original = ""
    if config_path.exists():
        with open(config_path, encoding="utf-8", newline="") as handle:
            original = handle.read()
    lines = original.splitlines()

    # An *active* `pipeline:` block, not a commented-out one. The shipped
    # local.yaml carries the whole explanation commented out, ready to
    # uncomment, and matching that would write the value inside a comment.
    pipeline_at = next(
        (i for i, line in enumerate(lines) if line.rstrip() == "pipeline:"), None
    )
    previous: int | None = None
    new_line = f"  output_latency_ms: {measured}  # {provenance}"

    if pipeline_at is None:
        block = [
            "",
            "# Written by scripts/measure_audio_device.py --write-config.",
            "# This is a property of the room and the output device, not of the",
            "# software: re-measure in the venue, and after changing speakers.",
            "pipeline:",
            new_line,
        ]
        lines = lines + block
    else:
        end = pipeline_at + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
            end += 1
        for i in range(pipeline_at + 1, end):
            stripped = lines[i].strip()
            if stripped.startswith("output_latency_ms:"):
                value = stripped.split(":", 1)[1].split("#")[0].strip()
                try:
                    previous = int(value)
                except ValueError:
                    previous = None
                lines[i] = new_line
                break
        else:
            lines.insert(pipeline_at + 1, new_line)

    # Keep whatever line endings the file already had. Rewriting a CRLF file
    # as LF makes every line look modified to anything diffing it, for one
    # changed value.
    newline = "\r\n" if "\r\n" in original else "\n"
    updated = newline.join(lines) + newline

    # Never hand back a file the pipeline will refuse to start on. Parsing the
    # result costs nothing and turns a broken talk into a printed reason.
    import yaml

    try:
        parsed = yaml.safe_load(updated)
    except Exception as exc:
        return {"written": False, "reason": f"the edit produced invalid YAML ({exc}); "
                                            "nothing was changed"}
    if not isinstance(parsed, dict) or \
            (parsed.get("pipeline") or {}).get("output_latency_ms") != measured:
        return {"written": False,
                "reason": "the edit did not land where expected; nothing was changed"}

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    worst = float(roundtrip.get("max_ms", roundtrip["median_ms"]))
    gate_ms = measured + tail_ms
    return {"written": True, "value_ms": measured, "previous": previous,
            "path": str(config_path), "worst_ms": round(worst, 1),
            "gate_ms": gate_ms, "margin_ms": round(gate_ms - worst, 1),
            "covers_worst": gate_ms > worst}


def render(report: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print(f"DEVICE REPORT  -  {report['label']}")
    print("=" * 78)

    bw = report.get("bandwidth", {})
    if "error" in bw:
        print(f"  bandwidth      : {bw['error']}")
    elif bw:
        mark = "OK  " if bw["covers_whisper_band"] else "WARN"
        cliff = f"codec cliff at {bw['cliff_hz']:.0f} Hz" if bw["cliff_hz"] else "no codec cliff"
        print(f"  bandwidth      : [{mark}] usable to {bw['usable_hz']:.0f} Hz - {bw['verdict']}")
        print(f"                   Nyquist {bw['nyquist_hz']:.0f} Hz, {cliff} "
              f"(steepest {bw['steepest_slope_db_per_octave']:.0f} dB/oct)")
        if bw.get("octave_levels_db"):
            bands = "  ".join(f"{k}:{v:+.0f}" for k, v in bw["octave_levels_db"].items())
            print(f"  spectrum (dB)  : {bands}")
        print(f"  level          : {bw['rms_dbfs']} dBFS rms, peak {bw['peak']:.3f}"
              + ("  *** CLIPPING ***" if bw["clipping"] else "")
              + ("  (quiet)" if bw.get("quiet") else ""))
        if bw.get("recording"):
            print(f"  recording      : {bw['recording']}")

    nf = report.get("noise_floor", {})
    if nf and "error" not in nf:
        print(f"  noise floor    : {nf['dbfs']} dBFS (p95 {nf['p95_dbfs']}) "
              f"-> suggested vad.threshold {nf['suggested_vad_threshold']}")

    lat = report.get("roundtrip", {})
    if "error" in lat:
        print(f"  round-trip     : {lat['error']}")
    elif lat:
        print(f"  round-trip     : {lat['median_ms']} ms median "
              f"({lat['min_ms']}-{lat['max_ms']} ms over {lat['samples']} chirps)")

    print("-" * 78)
    for line in report.get("recommendations", []):
        print(f"  -> {line}")
    print("=" * 78)


def build_recommendations(report: dict[str, Any]) -> list[str]:
    out: list[str] = []
    bw = report.get("bandwidth", {})
    lat = report.get("roundtrip", {})

    # A probe that failed is not a probe that passed. Saying "no problems
    # detected" after a failed measurement is worse than saying nothing.
    failed = [name for name in ("bandwidth", "noise_floor", "roundtrip")
              if isinstance(report.get(name), dict) and "error" in report[name]]
    if failed:
        for name in failed:
            detail = report[name]
            out.append(f"{name} probe did not produce a result: {detail['error']}")
            if detail.get("hint"):
                out.append(f"   {detail['hint']}")
        out.append("This device has NOT been assessed. Re-run the failed probes.")

    if bw and "error" not in bw:
        if not bw["covers_whisper_band"]:
            # Two independent ceilings, and a device can hit both at once: the
            # sample rate caps what is representable at all, and a codec can cut
            # below that. Reporting only one of them misdiagnoses the fix.
            nyquist = bw["nyquist_hz"]
            cliff = bw.get("cliff_hz")
            if nyquist <= USEFUL_BANDWIDTH_HZ:
                out.append(
                    f"This device captures at {bw['native_rate']} Hz, so it physically cannot "
                    f"carry anything above {nyquist:.0f} Hz - at or below the 8 kHz band Whisper "
                    "expects. No setting changes that; only different hardware does."
                )
            if cliff and cliff < nyquist * 0.98:
                lead = "On top of that, content" if nyquist <= USEFUL_BANDWIDTH_HZ else "Content"
                out.append(
                    f"{lead} stops at {cliff:.0f} Hz - a codec cutoff rather than a gentle "
                    "roll-off, and below the 8 kHz band Whisper expects. Expect noticeably "
                    "worse Turkish recognition."
                )
            elif nyquist > USEFUL_BANDWIDTH_HZ:
                out.append(
                    f"Usable content reaches only {bw['usable_hz']:.0f} Hz, below the 8 kHz band "
                    "Whisper expects. Expect noticeably worse Turkish recognition."
                )
            if bw["likely_bluetooth"]:
                out.append(
                    "The Bluetooth HFP profile is the cause. There is no software fix: "
                    "use a wired microphone for the stage."
                )
        if bw.get("upsampled"):
            eff = bw["effective_source_rate_hz"]
            note = (
                f"Device reports {bw['native_rate']} Hz, but the audio is band-limited to "
                f"{bw['usable_hz']:.0f} Hz with a flat floor above it - the real capture path is "
                f"{eff} Hz, upsampled. Something upstream (laptop voice DSP, or a Bluetooth HFP "
                "link) is the actual source."
            )
            if bw["covers_whisper_band"]:
                note += (
                    " Whisper resamples to 16 kHz anyway, so this alone does not disqualify the "
                    "device - but it means the sample rate in the config is not what you are "
                    "really getting, and it points at processing you did not ask for."
                )
            out.append(note)

        if bw["clipping"]:
            out.append("Input is clipping. Lower the device gain in Windows sound settings.")
        elif bw.get("quiet"):
            out.append(
                f"Signal peaked at only {bw['peak']:.3f} ({bw['rms_dbfs']:.0f} dBFS rms). Raise the "
                "input level in Windows sound settings, disable microphone 'audio enhancements', "
                "and speak closer. Quiet input costs recognition accuracy on its own."
            )

    nf = report.get("noise_floor", {})
    if nf and "error" not in nf and nf.get("noise_suppression_active"):
        out.append(
            f"Noise floor reads {nf['dbfs']:.0f} dBFS, which is digital silence rather than a "
            "quiet room: the driver is gating non-speech to zero. Turn off 'audio enhancements' "
            "for this microphone in Windows sound settings before measuring or presenting - an "
            "upstream gate clips word onsets and hides the real noise floor from our VAD."
        )

    if lat and "error" not in lat:
        if lat["median_ms"] > 250:
            out.append(
                f"Round-trip is {lat['median_ms']:.0f} ms. That is added to every segment on top "
                "of model time - a wired WASAPI path should be under 80 ms."
            )
        if lat["spread_ms"] > 60:
            out.append(
                f"Latency varies by {lat['spread_ms']:.0f} ms between chirps, which means jitter. "
                "Bluetooth and MME both do this; WASAPI on wired hardware does not."
            )

    if not out:
        out.append("No problems detected. This device is suitable for the pipeline.")
    return out


def probes_that_ran(report: dict[str, Any]) -> tuple[int, int]:
    """(succeeded, attempted) across the three probes."""
    attempted = [n for n in ("bandwidth", "noise_floor", "roundtrip") if n in report]
    ok = [n for n in attempted if "error" not in report[n]]
    return len(ok), len(attempted)


def main() -> int:
    configure_console()  # Windows consoles default to cp1252; see logging_setup
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="input device index or name substring")
    parser.add_argument("--output", default=None, help="output device index or name substring")
    parser.add_argument("--seconds", type=float, default=5.0, help="bandwidth probe duration")
    parser.add_argument("--repeats", type=int, default=5, help="chirps for the latency probe")
    parser.add_argument("--label", default=None, help="name for this report (e.g. 'lavalier')")
    parser.add_argument("--skip-bandwidth", action="store_true")
    parser.add_argument("--skip-noise", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--write-config", action="store_true",
        help="write the measured round-trip into config/local.yaml as "
             "pipeline.output_latency_ms - the step that otherwise gets skipped",
    )
    args = parser.parse_args()

    spec: int | str | None = args.input
    if isinstance(spec, str) and spec.isdigit():
        spec = int(spec)
    out_spec: int | str | None = args.output
    if isinstance(out_spec, str) and out_spec.isdigit():
        out_spec = int(out_spec)

    device = resolve_device(spec, "input")
    out_device = resolve_device(out_spec, "output")
    label = args.label or device.name

    # Say plainly which hardware is about to be measured. --label only names the
    # report; it does not select anything, and mixing the two produces a report
    # confidently describing the wrong microphone.
    print("\nMeasuring:")
    print(f"  input  : {device.label}"
          f"  [{device.max_input_channels} ch, {int(device.default_samplerate)} Hz]"
          + ("  BLUETOOTH?" if device.likely_bluetooth else ""))
    print(f"  output : {out_device.label}"
          f"  [{out_device.max_output_channels} ch, {int(out_device.default_samplerate)} Hz]"
          + ("  BLUETOOTH?" if out_device.likely_bluetooth else ""))
    if args.input is None:
        print("  (no --input given, so this is the system default; "
              "pass --input \"<name>\" to measure a specific microphone)")

    report: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "label": label,
        "device": device.label,
        "device_index": device.index,
        "hostapi": device.hostapi_name,
        "likely_bluetooth": device.likely_bluetooth,
        "reported_latency_ms": device.latency_ms("input"),
    }

    if not args.skip_bandwidth:
        print("\n[1/3] bandwidth")
        report["bandwidth"] = probe_bandwidth(spec, args.seconds)
    if not args.skip_noise:
        print("\n[2/3] noise floor")
        report["noise_floor"] = probe_noise_floor(spec)
    if not args.skip_latency:
        print("\n[3/3] round-trip latency")
        report["roundtrip"] = probe_roundtrip_latency(spec, out_spec, args.repeats)

    report["recommendations"] = build_recommendations(report)

    out_dir = ensure_dir("runs/audio")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)[:48]
    out_path = out_dir / f"device-{safe}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    written: dict[str, Any] | None = None
    if args.write_config:
        # The real tail, not the shipped default: whether the gate covers the
        # worst measured chirp depends on both numbers together.
        tail_ms = 250
        try:
            from parliamo.config import load_config

            tail_ms = load_config().pipeline.half_duplex_tail_ms
        except Exception:  # pragma: no cover - a broken config is reported elsewhere
            pass
        written = write_output_latency(report, resolve("config/local.yaml"), tail_ms)
        report["config_write"] = written

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        render(report)
        print(f"\nreport written to {out_path}")
        if written is not None:
            if written["written"]:
                was = written["previous"]
                print(
                    "config updated: pipeline.output_latency_ms = "
                    f"{written['value_ms']} ms"
                    + (f" (was {was})" if was is not None else "")
                    + f"  in {written['path']}"
                )
                # The median is what gets written; the gate has to survive the
                # worst chirp, not the typical one.
                if not written["covers_worst"]:
                    print(
                        f"  WARNING: the gate closes for {written['gate_ms']} ms but the "
                        f"slowest chirp came back after {written['worst_ms']} ms. The "
                        "microphone will reopen while the speakers are still audible. "
                        "Raise pipeline.half_duplex_tail_ms by at least "
                        f"{int(abs(written['margin_ms'])) + 50} ms, or move the "
                        "microphone away from the speakers."
                    )
                elif written["margin_ms"] < 100:
                    print(
                        f"  Gate {written['gate_ms']} ms against a worst chirp of "
                        f"{written['worst_ms']} ms - only {written['margin_ms']} ms spare. "
                        "That holds here, but re-measure in the venue before trusting it."
                    )
            else:
                print(f"config NOT updated: {written['reason']}")
        elif isinstance(report.get("roundtrip"), dict) and "median_ms" in report["roundtrip"]:
            print(
                "\n  Re-run with --write-config to put this number where the "
                "gate can actually use it."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
