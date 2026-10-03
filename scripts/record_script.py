#!/usr/bin/env python
"""Record a reading script sentence by sentence, aligned automatically.

    python scripts/record_script.py
    python scripts/record_script.py --input "Wireless GO" --out data/insitu/lavalier

Shows one sentence at a time, records it, and writes both the wav and a
manifest line pairing it with the text. Because you are reading a known script,
**the script is the reference transcript** - there is no transcription work
afterwards. Feed the manifest to the bake-off to get a WER for your own voice
through your own microphone:

    python scripts/bench_asr.py --manifest data/insitu/<name>/manifest.tsv

Controls
--------
Enter    start recording; it stops on its own after a pause
r        redo the sentence just recorded
s        skip this sentence
q        stop and keep what has been recorded so far

The manifest is written after every sentence, so quitting or crashing halfway
loses nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from parliamo.audio.capture import AudioCapture
from parliamo.audio.devices import resolve_device
from parliamo.audio.reblock import Reblocker
from parliamo.audio.vad import SILERO_BLOCK, SileroVAD
from parliamo.config import load_config
from parliamo.logging_setup import configure_console
from parliamo.paths import ensure_dir


def load_script(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def record_until_silence(
    capture: AudioCapture,
    vad: SileroVAD,
    *,
    silence_s: float = 1.5,
    max_s: float = 30.0,
    min_s: float = 0.6,
) -> np.ndarray:
    """Record until the speaker has been quiet for *silence_s*."""
    reblocker = Reblocker(SILERO_BLOCK)
    vad.reset()
    capture.drain()

    frames: list[np.ndarray] = []
    silent_run = 0.0
    heard_speech = False
    block_s = SILERO_BLOCK / 16000
    started = time.monotonic()

    while time.monotonic() - started < max_s:
        chunk = capture.read(timeout=0.5)
        if chunk is None:
            continue
        for block in reblocker.push(chunk):
            frames.append(block)
            if vad.probability(block) >= 0.5:
                heard_speech = True
                silent_run = 0.0
            elif heard_speech:
                silent_run += block_s

        elapsed = time.monotonic() - started
        if heard_speech and silent_run >= silence_s and elapsed >= min_s:
            break
        # Live level meter so the speaker can see it is working.
        if frames:
            level = float(np.abs(frames[-1]).max())
            bar = "#" * min(40, int(level * 120))
            state = "REC" if heard_speech else "..."
            print(f"\r  {state} {elapsed:4.1f}s |{bar:<40}|", end="", flush=True)

    print("\r" + " " * 60 + "\r", end="")
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


def main() -> int:
    configure_console()  # Windows consoles default to cp1252; see logging_setup
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--script", default="data/insitu/script-tr.txt")
    parser.add_argument("--out", default=None,
                        help="output directory (default: data/insitu/<timestamp>)")
    parser.add_argument("--input", default=None, help="input device index or name substring")
    parser.add_argument("--silence", type=float, default=1.5,
                        help="pause that ends a recording")
    parser.add_argument("--start-at", type=int, default=1, help="resume from sentence N")
    args = parser.parse_args()

    cfg = load_config()
    script_path = Path(args.script)
    if not script_path.is_absolute():
        from parliamo.paths import resolve

        script_path = resolve(script_path)
    sentences = load_script(script_path)
    if not sentences:
        print(f"no sentences in {script_path}")
        return 1

    out_dir = ensure_dir(
        args.out or f"data/insitu/{datetime.now():%Y%m%d-%H%M%S}"
    )
    manifest_path = out_dir / "manifest.tsv"

    device_spec = args.input if args.input is not None else cfg.audio.input_device
    if isinstance(device_spec, str) and device_spec.isdigit():
        device_spec = int(device_spec)
    device = resolve_device(device_spec, "input")

    print("\n" + "=" * 72)
    print("parlIAmo - in-situ recording")
    print("=" * 72)
    print(f"  microphone : {device.label}")
    if device.likely_bluetooth:
        print("               WARNING: this looks like a Bluetooth device. Its "
              "microphone runs at 16 kHz mono over HFP.")
    print(f"  script     : {script_path.name} ({len(sentences)} sentences)")
    print(f"  output     : {out_dir}")
    print(f"  stops after: {args.silence}s of silence")
    print("\n  Read at your presenting pace. Do not over-articulate - ordinary")
    print("  delivery is what we need to measure.")
    print("\n  Enter = record   r = redo   s = skip   q = quit and keep\n")

    vad = SileroVAD(device="cpu", threshold=cfg.vad.threshold)
    vad.load()

    import soundfile as sf

    recorded: list[tuple[str, str]] = []
    capture = AudioCapture(device=device_spec, sample_rate=16000, block_ms=32, gate=None)
    capture.start()

    try:
        index = args.start_at - 1
        while index < len(sentences):
            sentence = sentences[index]
            print(f"\n[{index + 1}/{len(sentences)}]  {sentence}")
            command = input("  > ").strip().lower()

            if command == "q":
                break
            if command == "s":
                index += 1
                continue
            if command == "r" and recorded:
                recorded.pop()
                index = max(0, index - 1)
                continue

            audio = record_until_silence(
                capture, vad, silence_s=args.silence
            )
            duration = audio.size / 16000
            peak = float(np.abs(audio).max()) if audio.size else 0.0

            if duration < 0.4 or peak < 0.01:
                print(f"  nothing captured ({duration:.1f}s, peak {peak:.3f}) - press Enter to retry")
                continue

            name = f"{index + 1:03d}.wav"
            sf.write(out_dir / name, audio, 16000, subtype="FLOAT")
            recorded.append((name, sentence))
            manifest_path.write_text(
                "\n".join(f"{n}\t{t}" for n, t in recorded) + "\n", encoding="utf-8"
            )
            print(f"  saved {name}  ({duration:.1f}s, peak {peak:.2f})")
            index += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        capture.stop()

    total_s = 0.0
    for name, _ in recorded:
        info = sf.info(out_dir / name)
        total_s += info.duration

    print("\n" + "=" * 72)
    print(f"  {len(recorded)} of {len(sentences)} sentences, {total_s / 60:.1f} minutes")
    print(f"  manifest: {manifest_path}")
    if recorded:
        print("\n  Measure this microphone and this voice:")
        print(f"    python scripts/bench_asr.py --manifest {manifest_path} \\")
        print("        --models large-v3-turbo,large-v3")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
