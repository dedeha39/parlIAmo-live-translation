#!/usr/bin/env python
"""Prove the half-duplex gate actually stops the speakers reaching the microphone.

The one thing file replay cannot test
-------------------------------------
``live_translate.py --file`` exercises every model in the pipeline and none of
the hardware around it. The gate, real playback, and the acoustic loop between
the speakers and the microphone are exactly what a recording cannot reach - and
the feedback loop is the failure that ends a demonstration rather than
degrading it: the system hears its own Italian, transcribes it, translates it,
and speaks it again, faster each time.

This needs no person. It plays recorded speech through the speakers and
measures what the *pipeline* received while it played.

Why it runs twice
-----------------
"The microphone delivered nothing while the speakers were on" is not evidence
of a working gate. It is also what happens when the speakers are muted, the
volume is at zero, the output went to headphones, or the microphone is dead. So
the same test runs with the gate **off**, and the run is only meaningful if that
one shows the loop clearly. The gate is credited for a difference, never for a
silence.

What it measures
----------------
1. **Room floor** - RMS of the room with nothing playing.
2. **Gate off** - what reaches the pipeline while the speakers play. This is the
   feedback that would be transcribed.
3. **Gate on** - the same, expecting the room floor.
4. **The tail** - the seconds *after* the gate reopens. This is what
   ``pipeline.output_latency_ms`` exists to cover: if audio is still arriving
   when the microphone comes back, the gate is too short, and the number in the
   config is wrong for this room.

Run it in the venue, at performance volume, with the microphone where it will
actually stand::

    python scripts/verify_feedback_loop.py
    python scripts/verify_feedback_loop.py --seconds 4 --json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from typing import Any

import numpy as np

from parliamo.audio.capture import AudioCapture
from parliamo.audio.gate import HalfDuplexGate
from parliamo.audio.playback import AudioPlayback
from parliamo.config import load_config
from parliamo.logging_setup import configure_console
from parliamo.paths import ensure_dir

#: How far above *this run's own room floor* the leaked audio must sit for the
#: gate-off pass to count as having demonstrated a loop at all. Relative, not
#: absolute: the Intel array's noise suppression puts a quiet room at about
#: -100 dBFS here, so any fixed threshold judges the microphone rather than
#: the leak.
AUDIBLE_MARGIN_DB = 6.0

#: The level AudioCapture treats as speech. A leak below this would not have
#: reached the recogniser even with the gate open.
SPEECH_PEAK = 0.02


def dbfs(block: np.ndarray) -> float:
    """RMS of *block* in dBFS. Returns a floor rather than -inf for silence."""
    if block.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def load_probe_audio(path: Any, rate: int) -> tuple[np.ndarray, str]:
    """Real recorded speech, resampled to the playback rate.

    **Use speech, not a synthetic probe.** The first version played
    amplitude-modulated noise in the speech band. Measured on this laptop it
    reached the microphone 10 dB *below* the room floor - nothing at all - while
    the same recording of real speech came through 19 dB above it. Whatever the
    mechanism in the Intel array, it treats the two very differently, and the
    one that matters is speech.

    Speech is also the honest probe on its own terms: what feeds back on stage
    is the system's own synthesised Italian, so the test signal should be a
    voice.
    """
    import soundfile as sf

    audio, file_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_rate != rate:
        import soxr

        audio = np.asarray(soxr.resample(audio, file_rate, rate), dtype=np.float32)
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = (audio / peak * 0.7).astype(np.float32)
    return audio, f"{path} ({audio.size / rate:.1f} s of recorded speech)"


def speech_band_signal(seconds: float, rate: int) -> np.ndarray:
    """Modulated noise in the speech band. The fallback, when no recording exists.

    Kept because a machine with no reference recording still deserves a check,
    but see :func:`load_probe_audio`: on this laptop it arrived *below* the room
    floor, in which case the run reports that it proved nothing rather than
    passing.
    """
    n = int(seconds * rate)
    rng = np.random.default_rng(20260905)
    noise = rng.standard_normal(n).astype(np.float32)

    # Band-limit to roughly 300-3400 Hz by zeroing everything else in the
    # spectrum. Crude, and entirely adequate for a level measurement.
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    spectrum[(freqs < 300.0) | (freqs > 3400.0)] = 0.0
    band = np.fft.irfft(spectrum, n).astype(np.float32)

    # 4 Hz amplitude modulation: the syllable rate of ordinary speech.
    envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * np.arange(n) / rate))
    signal = band * (0.35 + 0.65 * envelope).astype(np.float32)

    peak = float(np.abs(signal).max())
    return (signal / peak * 0.6).astype(np.float32) if peak > 0 else signal


class Collector:
    """Drains the capture queue on its own thread, keeping what and when."""

    def __init__(self, capture: AudioCapture) -> None:
        self.capture = capture
        self.blocks: list[tuple[float, np.ndarray]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="collector", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            block = self.capture.read(timeout=0.2)
            if block is not None and block.size:
                self.blocks.append((time.monotonic(), block))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def between(self, start: float, end: float) -> np.ndarray:
        kept = [b for t, b in self.blocks if start <= t < end]
        return np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)


def run_pass(
    *,
    gate_enabled: bool,
    signal: np.ndarray,
    tone_rate: int,
    input_spec: Any,
    output_spec: Any,
    tail_ms: int,
    output_latency_ms: float,
    settle_s: float,
) -> dict[str, Any]:
    """One playback, with the gate on or off, reporting what the pipeline got."""
    gate = HalfDuplexGate(enabled=gate_enabled, tail_ms=tail_ms)
    capture = AudioCapture(device=input_spec, gate=gate)
    playback = AudioPlayback(
        device=output_spec, sample_rate=tone_rate, gate=gate,
        output_latency_ms=output_latency_ms,
    )

    capture.start()
    playback.start()
    collector = Collector(capture)
    collector.start()

    # A moment of room before anything plays, as this pass's own baseline.
    time.sleep(settle_s)
    floor_start = time.monotonic() - settle_s
    play_start = time.monotonic()

    playback.play(signal, blocking=True)
    play_end = time.monotonic()

    # Wait out the gate, then listen to what arrives once it has reopened.
    while gate.is_closed:
        time.sleep(0.01)
    reopen = time.monotonic()
    time.sleep(settle_s)
    tail_end = time.monotonic()

    collector.stop()
    stats = capture.stop()
    playback.stop()

    during = collector.between(play_start, play_end)
    after = collector.between(reopen, tail_end)
    floor = collector.between(floor_start, play_start)

    return {
        "gate_enabled": gate_enabled,
        "device_in": capture.info.device.label if capture._info else "?",
        "device_out": playback.info.device.label if playback._info else "?",
        "floor_dbfs": round(dbfs(floor), 1),
        "during_dbfs": round(dbfs(during), 1),
        "during_peak": round(float(np.abs(during).max()) if during.size else 0.0, 4),
        "after_reopen_dbfs": round(dbfs(after), 1),
        "seconds_delivered_during": round(during.size / capture.sample_rate, 2),
        "gate_held_s": round(reopen - play_start, 2),
        "playback_s": round(play_end - play_start, 2),
        "capture": stats.as_dict(),
        "gate": gate.stats.as_dict(),
    }


def verdicts(off: dict[str, Any], on: dict[str, Any], tail_ms: int,
             output_latency_ms: float) -> list[dict[str, str]]:
    """Turn the two passes into statements someone can act on."""
    rows: list[dict[str, str]] = []

    def add(level: str, text: str) -> None:
        rows.append({"level": level, "text": text})

    # Everything here is judged against this pass's own room floor, never
    # against a fixed dBFS threshold. An earlier version used -75 dBFS and
    # called a clear 19 dB rise "nothing at all", because the Intel array's
    # noise suppression puts the floor of a quiet room at about -100 dBFS.
    leak = off["during_dbfs"] - off["floor_dbfs"]
    if leak < AUDIBLE_MARGIN_DB:
        add("fail", f"With the gate OFF the pipeline received only {leak:.1f} dB "
                    "above the room floor, so no loop was demonstrated and the "
                    "gate cannot be credited for anything. The output may be on "
                    "headphones, the volume may be down, or --input may be the "
                    "wrong microphone. Fix that and run it again.")
        return rows

    add("ok", f"With the gate OFF the pipeline received {leak:.1f} dB above the "
              f"room floor for {off['seconds_delivered_during']:.1f} s. The "
              "speakers do reach this microphone.")

    # Whether that leak would actually have been transcribed is a separate
    # question from whether it is measurable. AudioCapture counts a block as
    # speech at 0.02, and VAD wants more than that again.
    if off["during_peak"] < SPEECH_PEAK:
        add("warn", f"But it peaked at only {off['during_peak']:.4f}, below the "
                    f"{SPEECH_PEAK} the capture path treats as speech - at this volume the "
                    "loop would not have been transcribed even with the gate off, "
                    "so this run shows the gate working against a leak that was "
                    "never dangerous. Raise the system volume to performance "
                    "level and run it again; that is the test that matters.")

    blocked = on["during_dbfs"] - on["floor_dbfs"]
    if on["seconds_delivered_during"] <= 0.05:
        add("ok", "With the gate ON the pipeline received nothing while the "
                  "speakers played. The loop is closed.")
    elif blocked < AUDIBLE_MARGIN_DB:
        add("ok", f"With the gate ON the pipeline received "
                  f"{on['seconds_delivered_during']:.1f} s at the room floor "
                  f"({blocked:+.1f} dB). Nothing of the output got through.")
    else:
        add("fail", f"With the gate ON the pipeline still received "
                    f"{blocked:.1f} dB above the room floor. The gate is not "
                    "doing its job - check pipeline.half_duplex is true.")

    # The tail, which is the whole reason output_latency_ms exists.
    residual = on["after_reopen_dbfs"] - on["floor_dbfs"]
    if residual >= AUDIBLE_MARGIN_DB:
        add("fail", f"After the gate reopened, audio was still arriving "
                    f"{residual:.1f} dB above the room floor. The gate is too "
                    f"short for this room: it held for {on['gate_held_s']:.2f} s "
                    f"({output_latency_ms:.0f} + {tail_ms} ms). Re-measure with "
                    "scripts/measure_audio_device.py --write-config, or raise "
                    "pipeline.half_duplex_tail_ms.")
    else:
        add("ok", f"After the gate reopened the room was back to its floor "
                  f"({residual:+.1f} dB). "
                  f"{output_latency_ms:.0f} + {tail_ms} ms is long enough here.")

    if output_latency_ms <= 0:
        add("warn", "pipeline.output_latency_ms is 0. It passed here, but the "
                    "gate is sized from PortAudio's reported latency only, which "
                    "excludes the vendor DSP. Measure it: "
                    "scripts/measure_audio_device.py --write-config")
    return rows


def render(report: dict[str, Any]) -> None:
    off, on = report["gate_off"], report["gate_on"]
    print("\n" + "=" * 78)
    print("FEEDBACK LOOP CHECK")
    print("=" * 78)
    print(f"  in  : {on['device_in']}")
    print(f"  out : {on['device_out']}")
    print(f"  gate: {report['output_latency_ms']:.0f} ms output latency "
          f"+ {report['tail_ms']} ms tail")
    print("-" * 78)
    print(f"  {'':<14}{'room floor':>12}{'while playing':>16}{'after reopen':>15}"
          f"{'delivered':>12}")
    for label, row in (("gate OFF", off), ("gate ON", on)):
        print(f"  {label:<14}{row['floor_dbfs']:>10.1f} dB{row['during_dbfs']:>13.1f} dB"
              f"{row['after_reopen_dbfs']:>12.1f} dB"
              f"{row['seconds_delivered_during']:>10.1f} s")
    print("-" * 78)
    for row in report["verdicts"]:
        mark = {"ok": "  ->", "warn": "  !!", "fail": "  XX"}[row["level"]]
        text = row["text"]
        print(f"{mark} {text[:72]}")
        for i in range(72, len(text), 72):
            print(f"      {text[i:i + 72]}")
    print("=" * 78)


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=None, help="input device index or name")
    parser.add_argument("--output", default=None, help="output device index or name")
    parser.add_argument("--seconds", type=float, default=4.0,
                        help="how long to play the test signal")
    parser.add_argument("--audio", default=None,
                        help="a WAV of speech to play (default: the configured "
                             "reference voice, which is real speech)")
    parser.add_argument("--tone", action="store_true",
                        help="use synthetic modulated noise instead of speech; "
                             "voice-tuned noise suppression may remove it entirely")
    parser.add_argument("--settle", type=float, default=1.2,
                        help="quiet seconds measured before and after playback")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    tail_ms = cfg.pipeline.half_duplex_tail_ms
    output_latency_ms = float(cfg.pipeline.output_latency_ms)
    rate = cfg.audio.output_sample_rate

    if not cfg.pipeline.half_duplex:
        print("\n  pipeline.half_duplex is false in the config, so the gate would "
              "not run at all on stage.\n  This check forces it on for the second "
              "pass; fix the config before rehearsing.")

    source = "synthetic modulated noise in the speech band"
    signal = speech_band_signal(args.seconds, rate)
    if not args.tone:
        from parliamo.paths import resolve

        candidate = args.audio or cfg.tts.conversion.reference_voice
        path = resolve(candidate) if candidate else None
        if path is not None and path.exists():
            try:
                signal, source = load_probe_audio(path, rate)
                signal = signal[: int(args.seconds * rate)]
            except Exception as exc:
                print(f"  could not read {path} ({exc}); falling back to noise")
        elif args.audio:
            print(f"  {args.audio} does not exist; falling back to noise")

    played_s = signal.size / rate
    print(f"\nPlaying {played_s:.1f} s, twice.")
    print(f"  signal: {source}")
    print("Set the volume to what the room will hear, and keep the room quiet.\n")

    print("[1/2] gate OFF - this is what feedback looks like")
    off = run_pass(
        gate_enabled=False, signal=signal, tone_rate=rate,
        input_spec=args.input if args.input is not None else cfg.audio.input_device,
        output_spec=args.output if args.output is not None else cfg.audio.output_device,
        tail_ms=tail_ms, output_latency_ms=output_latency_ms, settle_s=args.settle,
    )
    time.sleep(0.5)

    print("[2/2] gate ON - this is what should reach the pipeline")
    on = run_pass(
        gate_enabled=True, signal=signal, tone_rate=rate,
        input_spec=args.input if args.input is not None else cfg.audio.input_device,
        output_spec=args.output if args.output is not None else cfg.audio.output_device,
        tail_ms=tail_ms, output_latency_ms=output_latency_ms, settle_s=args.settle,
    )

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "seconds": round(played_s, 2),
        "signal": source,
        "tail_ms": tail_ms,
        "output_latency_ms": output_latency_ms,
        "gate_off": off,
        "gate_on": on,
    }
    report["verdicts"] = verdicts(off, on, tail_ms, output_latency_ms)

    out_dir = ensure_dir("runs/audio")
    path = out_dir / f"feedback-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        render(report)
        print(f"\nreport written to {path}")

    return 1 if any(v["level"] == "fail" for v in report["verdicts"]) else 0


if __name__ == "__main__":
    sys.exit(main())
