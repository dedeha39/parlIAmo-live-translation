#!/usr/bin/env python
"""Re-analyse a saved recording without touching the hardware.

``measure_audio_device.py`` keeps every bandwidth recording under ``runs/audio``.
This script runs the same analysis over one of them, so a verdict can be
revisited after the detector changes, or compared between devices, without
asking anyone to speak into a microphone again.

    python scripts/analyse_recording.py                    # newest recording
    python scripts/analyse_recording.py path/to/file.wav
    python scripts/analyse_recording.py --all              # every saved recording
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_analyser():
    """Import the sibling script without executing its CLI."""
    path = REPO_ROOT / "scripts" / "measure_audio_device.py"
    spec = importlib.util.spec_from_file_location("measure_audio_device", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_audio_device"] = module
    spec.loader.exec_module(module)
    return module


def find_recordings() -> list[Path]:
    out_dir = REPO_ROOT / "runs" / "audio"
    if not out_dir.is_dir():
        return []
    return sorted(out_dir.glob("rec-*.wav"), key=lambda p: p.stat().st_mtime)


def render(path: Path, rate: int, result: dict[str, Any], advice: list[str]) -> None:
    print("\n" + "=" * 78)
    print(f"{path.name}")
    print(f"container rate: {rate} Hz")
    print("=" * 78)

    if "error" in result:
        print(f"  {result['error']}")
        return

    eff = result.get("effective_source_rate_hz")
    print(f"  Nyquist              {result['nyquist_hz']:.0f} Hz")
    print("  codec cliff          "
          + (f"{result['cliff_hz']:.0f} Hz "
             f"(steepest {result['steepest_slope_db_per_octave']:.0f} dB/oct)"
             if result["cliff_hz"] else "none detected"))
    print(f"  usable to            {result['usable_hz']:.0f} Hz")
    print("  effective source     " + (f"{eff} Hz  <-- UPSAMPLED" if result.get("upsampled")
                                        else "matches container"))
    print(f"  covers Whisper band  {'yes' if result['covers_whisper_band'] else 'NO'}")
    print(f"  verdict              {result['verdict']}")
    print(f"  level                {result['rms_dbfs']} dBFS rms, peak {result['peak']:.4f}"
          + ("  (quiet)" if result["quiet"] else "")
          + ("  *** CLIPPING ***" if result["clipping"] else ""))

    print("\n  octave levels (dB relative to spectral peak)")
    for band, level in result["octave_levels_db"].items():
        bar = "#" * max(0, min(48, int((level + 100) / 2)))
        print(f"    {band:>12} Hz  {level:7.1f}  {bar}")

    if advice:
        print()
        for line in advice:
            print(f"  -> {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", nargs="?", default=None,
                        help="wav file to analyse (default: newest under runs/audio)")
    parser.add_argument("--all", action="store_true", help="analyse every saved recording")
    args = parser.parse_args()

    import soundfile as sf

    mod = _load_analyser()

    if args.recording:
        targets = [Path(args.recording)]
    else:
        found = find_recordings()
        if not found:
            print("No recordings under runs/audio. Run scripts/measure_audio_device.py first.")
            return 1
        targets = found if args.all else [found[-1]]

    for path in targets:
        if not path.is_file():
            print(f"not found: {path}")
            continue
        signal, rate = sf.read(path, dtype="float32")
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim > 1:
            signal = signal[:, 0]

        result = mod.analyse_bandwidth(signal, rate)
        advice: list[str] = []
        if "error" not in result:
            advice = mod.build_recommendations(
                {"bandwidth": {**result, "native_rate": rate, "likely_bluetooth": False}}
            )
        render(path, rate, result, advice)

    return 0


if __name__ == "__main__":
    sys.exit(main())
