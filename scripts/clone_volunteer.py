#!/usr/bin/env python
"""Clone a volunteer's voice on stage in fifteen seconds, and delete it after.

    python scripts/clone_volunteer.py --name "Maria" --consent "signed, ref 007"
    python scripts/clone_volunteer.py --list
    python scripts/clone_volunteer.py --forget "Maria"
    python scripts/clone_volunteer.py --forget-all          # after the talk

Why a separate tool
-------------------
`record_script.py` records sentence by sentence against a prepared text, which
is right for building an evaluation set and wrong for a stage. Here the whole
interaction is: someone comes up, talks for fifteen seconds, and the next
sentence is in their voice. A restart at that moment would end the
demonstration - the point being made is that it takes fifteen seconds, not a
coffee break.

So this records, builds a proper reference (the laptop microphone gates 16-50%
of what it captures to digital silence, which is why the first cloning attempt
sounded wrong), and POSTs the result into the running pipeline through the
operator page.

Consent is not a formality here
-------------------------------
`--consent` is required and is written next to the audio. A volunteer asked in
front of two hundred people is under pressure to agree, so ask before the
segment rather than during it, and make refusing easy. The form is in
docs/07-ethics-and-consent.md.

**Deleting afterwards is part of the promise**, and until now nothing in this
repository did it. `--forget-all` removes every volunteer recording, every
reference built from one, and the consent records - and says what it removed.
Run it before closing the laptop.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from parliamo.audio.capture import AudioCapture
from parliamo.audio.devices import resolve_device
from parliamo.config import load_config
from parliamo.logging_setup import configure_console
from parliamo.paths import ensure_dir, resolve
from parliamo.tts.base import VoiceProfile
from parliamo.tts.reference import build_reference, measure_clip

#: Everything a volunteer leaves behind lives here, so forgetting is one path.
VOLUNTEER_DIR = "data/voices/volunteers"

TARGET_RATE = 16000


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")[:40]


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------


def record(seconds: float, device_spec, threshold: float) -> np.ndarray:
    """Record for *seconds*, showing a level meter so the room can see it work."""
    capture = AudioCapture(device=device_spec, sample_rate=TARGET_RATE, block_ms=32, gate=None)
    capture.start()
    try:
        time.sleep(0.3)
        capture.drain()
        print()
        for n in (3, 2, 1):
            print(f"  starting in {n} ...", end="\r", flush=True)
            time.sleep(1.0)
        print("  >>> SPEAK NOW - anything at all, for the whole countdown <<<   ")

        chunks: list[np.ndarray] = []
        collected = 0
        wanted = int(seconds * TARGET_RATE)
        deadline = time.monotonic() + seconds * 3 + 3.0
        while collected < wanted and time.monotonic() < deadline:
            block = capture.read(timeout=1.0)
            if block is None:
                continue
            chunks.append(block)
            collected += block.size
            level = float(np.abs(block).max())
            bar = "#" * min(40, int(level * 120))
            left = max(0.0, (wanted - collected) / TARGET_RATE)
            print(f"\r  {left:4.1f}s left  |{bar:<40}|", end="", flush=True)
        print("\r" + " " * 66 + "\r", end="")
    finally:
        capture.stop()

    return np.concatenate(chunks)[:wanted] if chunks else np.zeros(0, dtype=np.float32)


def push_to_pipeline(path: Path, consent: str, port: int) -> bool:
    """Tell a running `live_translate.py --ui` to use this voice now."""
    import urllib.error
    import urllib.request

    payload = json.dumps({"path": str(path), "consent": consent}).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/voice", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            json.loads(response.read().decode("utf-8"))
        return True
    except urllib.error.HTTPError as exc:
        print(f"  the pipeline refused it: {exc.read().decode('utf-8', 'replace')}")
    except OSError as exc:
        print(f"  no running pipeline on port {port} ({exc}).")
        print(f"  Start it with --ui, or pass --voice {path} at startup.")
    return False


# ---------------------------------------------------------------------------
# forgetting
# ---------------------------------------------------------------------------


def existing() -> list[Path]:
    directory = resolve(VOLUNTEER_DIR)
    return sorted(p for p in directory.glob("*") if p.is_dir()) if directory.is_dir() else []


def forget(target: Path) -> dict[str, object]:
    """Delete one volunteer's recordings, reference and consent record."""
    files = [p for p in target.rglob("*") if p.is_file()]
    detail = {
        "name": target.name,
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
    }
    shutil.rmtree(target, ignore_errors=True)
    detail["gone"] = not target.exists()
    return detail


# ---------------------------------------------------------------------------


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default=None, help="the volunteer's name, for the record")
    parser.add_argument("--consent", default="", help="reference to the signed form")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--input", default=None, help="microphone index or name substring")
    parser.add_argument("--port", type=int, default=8770, help="the operator page's port")
    parser.add_argument("--no-push", action="store_true",
                        help="build the reference but do not load it into the pipeline")
    parser.add_argument("--list", action="store_true", help="who is currently on disk")
    parser.add_argument("--forget", default=None, metavar="NAME")
    parser.add_argument("--forget-all", action="store_true",
                        help="delete every volunteer recording - run before closing the laptop")
    args = parser.parse_args()

    cfg = load_config()
    root = ensure_dir(VOLUNTEER_DIR)

    # -- listing and forgetting come first: they must work even if audio does not
    if args.list:
        people = existing()
        if not people:
            print("\nNo volunteer recordings on disk.")
            return 0
        print(f"\n{len(people)} volunteer recording(s) still on this machine:")
        for person in people:
            meta = person / "consent.json"
            record_ref = "?"
            if meta.exists():
                record_ref = json.loads(meta.read_text(encoding="utf-8")).get("consent", "?")
            size = sum(p.stat().st_size for p in person.rglob("*") if p.is_file())
            print(f"  {person.name:<24} {size / 1e6:>6.1f} MB   consent: {record_ref}")
        print("\nDelete them with --forget-all. The consent form promises it.")
        return 0

    if args.forget_all or args.forget:
        targets = existing() if args.forget_all else [root / slug(args.forget)]
        removed = []
        for target in targets:
            if not target.exists():
                print(f"  nothing on disk for {target.name}")
                continue
            detail = forget(target)
            removed.append(detail)
            print(f"  deleted {detail['name']}: {detail['files']} files, "
                  f"{detail['bytes'] / 1e6:.1f} MB")
        if not removed:
            print("\nNothing to delete.")
        else:
            print(f"\n{len(removed)} volunteer(s) removed. Tell them it is done.")
        return 0

    # -- recording -----------------------------------------------------------
    if not args.name:
        print("--name is required when recording. Use --list or --forget otherwise.")
        return 2
    if not args.consent.strip():
        print("\n  --consent is required before recording anyone.")
        print("  Put the reference of the signed form there, e.g.")
        print('    --consent "signed 2026-09-14, ref 007"')
        print("\n  The form is in docs/07-ethics-and-consent.md. Ask before the")
        print("  segment rather than during it, and make refusing easy.")
        return 2

    device_spec = args.input if args.input is not None else cfg.audio.input_device
    if isinstance(device_spec, str) and device_spec.isdigit():
        device_spec = int(device_spec)
    device = resolve_device(device_spec, "input")

    person = ensure_dir(Path(VOLUNTEER_DIR) / slug(args.name))
    print("\n" + "=" * 68)
    print(f"  cloning: {args.name}")
    print(f"  consent: {args.consent}")
    print(f"  mic    : {device.label}")
    print(f"  length : {args.seconds:.0f}s")
    print("=" * 68)
    print("\n  Say anything - what you had for breakfast is fine. Ordinary")
    print("  speaking voice, not a performance.")

    audio = record(args.seconds, device_spec, cfg.vad.threshold)
    stats = measure_clip(audio, TARGET_RATE)
    print(f"  captured {stats.duration_s:.1f}s, peak {stats.peak:.3f}, "
          f"{stats.rms_dbfs:.0f} dBFS, {100 * stats.gated_fraction:.0f}% gated")

    if stats.duration_s < 3.0 or stats.peak < 0.01:
        print("\n  Nothing usable was captured. Check the microphone and try again.")
        return 1
    if stats.gated_fraction > 0.4:
        print("\n  Warning: the driver removed most of that recording. The clone")
        print("  will sound poor. Try a different microphone if there is one.")

    import soundfile as sf

    raw_path = person / "raw.wav"
    sf.write(raw_path, audio, TARGET_RATE, subtype="FLOAT")

    reference = build_reference([(audio, TARGET_RATE)], target_seconds=args.seconds,
                                sample_rate=TARGET_RATE)
    ref_path = person / "reference.wav"
    sf.write(ref_path, reference, TARGET_RATE, subtype="FLOAT")

    # The consent record sits next to the audio, so the two cannot be separated.
    (person / "consent.json").write_text(json.dumps({
        "name": args.name,
        "consent": args.consent,
        "recorded": datetime.now().astimezone().isoformat(),
        "seconds": round(stats.duration_s, 1),
        "delete_after": "the end of this session - scripts/clone_volunteer.py --forget-all",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # Fails loudly rather than writing a voice with no record beside it.
    VoiceProfile(name=slug(args.name), reference_path=str(ref_path),
                 consent=args.consent).validate()

    print(f"\n  reference: {ref_path}  ({reference.size / TARGET_RATE:.1f}s)")

    if not args.no_push:
        print(f"  loading into the running pipeline on port {args.port} ...")
        if push_to_pipeline(ref_path, args.consent, args.port):
            print("  done - the next sentence will be in this voice.")

    print("\n  When the talk is over:")
    print("    python scripts/clone_volunteer.py --forget-all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
