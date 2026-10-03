#!/usr/bin/env python
"""Print the audio device inventory, ranked and annotated.

Use this to find the exact string to put in ``config/local.yaml`` under
``audio.input_device``. Prefer a name substring over an index: indices shift
whenever a Bluetooth device connects or disconnects.

    python scripts/list_audio_devices.py
    python scripts/list_audio_devices.py --direction input --hostapi WASAPI
    python scripts/list_audio_devices.py --json
"""

from __future__ import annotations

import argparse
import json
import sys

from parliamo.audio.devices import describe_devices, preferred_hostapi
from parliamo.logging_setup import configure_console


def main() -> int:
    configure_console()  # Windows consoles default to cp1252; see logging_setup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=["input", "output", "both"], default="both")
    parser.add_argument("--hostapi", default=None, help="filter by host API substring")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    directions = ["input", "output"] if args.direction == "both" else [args.direction]
    payload: dict[str, list[dict]] = {}
    for direction in directions:
        rows = describe_devices(direction)  # type: ignore[arg-type]
        if args.hostapi:
            needle = args.hostapi.lower()
            rows = [r for r in rows if needle in r["hostapi"].lower()]
        payload[direction] = rows

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    best = preferred_hostapi()
    print(f"\nPreferred host API on this system: {best}\n")

    for direction, rows in payload.items():
        print(f"{direction.upper()} DEVICES ({len(rows)})")
        print("-" * 104)
        print(f"{'idx':>4}  {'ch':>3}  {'sr':>7}  {'lat':>7}  {'hostapi':<22} name")
        print("-" * 104)
        for r in rows:
            ch = r["in_ch"] if direction == "input" else r["out_ch"]
            lat = r["low_in_latency_ms"] if direction == "input" else r["low_out_latency_ms"]
            flags = []
            if r["default_input"] and direction == "input":
                flags.append("DEFAULT")
            if r["default_output"] and direction == "output":
                flags.append("DEFAULT")
            if r["likely_bluetooth"]:
                flags.append("BLUETOOTH?")
            suffix = ("  <- " + " ".join(flags)) if flags else ""
            print(
                f"{r['index']:>4}  {ch:>3}  {int(r['default_sr']):>7}  {lat:>6.1f}ms  "
                f"{r['hostapi']:<22} {r['name']}{suffix}"
            )
        print()

    bt = [r for rows in payload.values() for r in rows if r["likely_bluetooth"]]
    if bt:
        print(
            "Note: devices flagged BLUETOOTH? drop to mono 16 kHz (HFP) the moment their\n"
            "      microphone is opened, and that also degrades their playback quality.\n"
            "      Measure before committing: python scripts/measure_audio_device.py\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
