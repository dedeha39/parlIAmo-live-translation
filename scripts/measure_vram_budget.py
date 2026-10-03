#!/usr/bin/env python
"""Measure what the pipeline actually costs in VRAM, with the models co-resident.

    python scripts/measure_vram_budget.py
    python scripts/measure_vram_budget.py --asr large-v3 --mt-device cpu
    python scripts/measure_vram_budget.py --skip-tts

Why this exists
---------------
The project's VRAM budget was assembled from per-model estimates, and the
estimates were wrong. Translation was budgeted at ~3100 MiB and measured at
640. On that arithmetic a model was eliminated that in fact fits comfortably.

Adding up separately measured models is also not the same as measuring them
together: allocator fragmentation, CUDA context, and cuBLAS workspaces are paid
once, not once per model. So this loads them **in sequence into one process**
and reports the real total after each addition.

The number that matters is the peak with everything resident, against what the
card actually has free once the desktop session has taken its share.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

from parliamo.paths import configure_model_cache

configure_model_cache()

import logging  # noqa: E402

from parliamo.logging_setup import configure_console  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402

log = logging.getLogger("vram")


def gpu_used_mib(samples: int = 5, settle_s: float = 0.4) -> float:
    """Peak GPU memory over a few samples.

    A single reading is noisy: the same model measured 950, 1011, 1084 and
    1292 MiB across earlier runs. Taking the peak of several readings after a
    settle delay is stable enough to budget from, and erring high is the safe
    direction.
    """
    time.sleep(settle_s)
    readings: list[float] = []
    for _ in range(samples):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                readings.append(float(out.stdout.strip().splitlines()[0]))
        except Exception:  # pragma: no cover
            pass
        time.sleep(0.15)
    return max(readings) if readings else 0.0


def gpu_total_mib() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0:
            return float(out.stdout.strip().splitlines()[0])
    except Exception:  # pragma: no cover
        pass
    return 0.0


def main() -> int:
    configure_console()  # Windows consoles default to cp1252; see logging_setup
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--asr", default="large-v3-turbo", help="or large-v3, medium, ...")
    parser.add_argument("--asr-compute", default="int8_float16")
    parser.add_argument("--mt", default="models/ct2/nllb-200-distilled-600M")
    parser.add_argument("--mt-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--tts", default="kokoro", choices=["kokoro", "chatterbox"],
                        help="synthesiser to measure; kokoro is what the live path uses")
    parser.add_argument("--tts-language", default="it")
    parser.add_argument("--conversion", action="store_true",
                        help="also account for the Seed-VC service (start it first)")
    parser.add_argument("--conversion-reference", default=None,
                        help="wav to warm the conversion service with")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-mt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    total = gpu_total_mib()
    stages: list[dict[str, Any]] = []
    held: list[Any] = []

    baseline = gpu_used_mib()
    print("\n" + "=" * 78)
    print("VRAM BUDGET  -  measured with models co-resident, not summed")
    print("=" * 78)
    print(f"  card total          {total:>7.0f} MiB")
    print(f"  before loading      {baseline:>7.0f} MiB   (desktop session, browsers, etc.)")
    print(f"  available to us     {total - baseline:>7.0f} MiB")
    if args.conversion:
        # Said plainly, because it changes how the last line reads: the service
        # is a separate process that was started before this script, so its
        # weights are already inside `baseline`. The stage recorded later is
        # the *activation* cost of one conversion, not the model's footprint.
        print("  NOTE: the conversion service was already running, so its weights are")
        print("        counted in the baseline above, not in the stage below.")
    print("-" * 78)

    def record(label: str, detail: str) -> None:
        used = gpu_used_mib()
        added = used - (stages[-1]["used_mib"] if stages else baseline)
        stages.append({"stage": label, "detail": detail, "used_mib": used, "added_mib": added})
        print(f"  + {label:<18} {added:>7.0f} MiB   total now {used:>7.0f} MiB   {detail}")

    try:
        if not args.skip_asr:
            from parliamo.asr import create_backend as make_asr

            asr = make_asr(
                "faster_whisper", model=args.asr, device="cuda", language="tr",
                compute_type=args.asr_compute,
            )
            asr.load()
            asr.warmup(seconds=3.0)
            held.append(asr)
            record("ASR", f"{args.asr} {args.asr_compute}")

        if not args.skip_mt:
            from parliamo.mt import create_backend as make_mt

            mt = make_mt(
                "ctranslate2_nllb", model=args.mt, device=args.mt_device,
                source_lang="tr", target_lang="it",
            )
            mt.load()
            mt.warmup()
            held.append(mt)
            # Name the model that was loaded, not the one that used to be the
            # default. This line said "NLLB-600M" while measuring 1.3B.
            from pathlib import Path as _Path

            record("MT", f"{_Path(args.mt).name} on {args.mt_device}")

        if not args.skip_tts:
            from parliamo.tts import create_backend as make_tts

            # Go through the registry rather than importing one synthesiser
            # directly. The direct import is how this script spent a week
            # measuring Chatterbox after ADR 0006 had replaced it with Kokoro
            # on the live path - the budget it printed was for an architecture
            # that no longer shipped.
            tts = make_tts(args.tts, device="cuda", language=args.tts_language)
            tts.load()
            held.append(tts)
            record("TTS (loaded)", f"{args.tts}")

            # Generation allocates activations and a KV cache on top of the
            # weights, and that peak is what has to fit - not the idle figure.
            tts.speak(
                "Questa è una frase di prova per misurare la memoria occupata.",
                language=args.tts_language,
            )
            record("TTS (after gen)", "peak including activations")

        if args.conversion:
            # Seed-VC lives in another process, so nothing here can load it.
            # nvidia-smi reports the whole card, which is exactly right: the
            # question is what the *machine* holds while the pipeline runs, and
            # a second process competing for the same 8 GB counts fully.
            from parliamo.tts.conversion import VoiceConverter

            converter = VoiceConverter(diffusion_steps=4)
            if not converter.available():
                print(f"\n  conversion service not reachable at {converter.address}")
                print("  start it first, in the other environment:")
                print("    <seedvc-env>/python.exe scripts/voice_conversion_server.py")
                print("  continuing without it - the peak below EXCLUDES voice conversion.\n")
            else:
                reference = args.conversion_reference
                if reference:
                    import numpy as np

                    from parliamo.paths import resolve

                    # One real conversion, so the measurement includes the
                    # activations and not just the resident weights.
                    converter.convert(
                        np.zeros(int(0.5 * 24000), dtype=np.float32),
                        24000,
                        str(resolve(reference)),
                    )
                    record("Voice conversion", "Seed-VC, separate process, after one convert")
                else:
                    record("Voice conversion", "Seed-VC, separate process, weights only")

        peak = gpu_used_mib()
        print("-" * 78)
        print(f"  PEAK with everything resident   {peak:>7.0f} MiB")
        print(f"  headroom against card total     {total - peak:>7.0f} MiB")
        verdict = "FITS" if peak < total - 200 else "DOES NOT FIT"
        print(f"  verdict                         {verdict}")
        print("=" * 78)

        report = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "card_total_mib": total,
            "baseline_mib": baseline,
            "peak_mib": peak,
            "headroom_mib": total - peak,
            "fits": peak < total - 200,
            "config": {
                "asr": None if args.skip_asr else args.asr,
                "asr_compute": args.asr_compute,
                "mt": None if args.skip_mt else args.mt,
                "mt_device": args.mt_device,
                "tts": None if args.skip_tts else args.tts,
                "conversion": bool(args.conversion),
            },
            "stages": stages,
        }
        out_dir = ensure_dir("runs/vram")
        path = args.out or out_dir / f"budget-{datetime.now():%Y%m%d-%H%M%S}.json"
        from pathlib import Path

        Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport written to {path}")
        return 0
    except Exception as exc:
        print(f"\nFAILED at stage {len(stages) + 1}: {type(exc).__name__}: {exc}")
        print("Stages that did load:")
        for s in stages:
            print(f"  {s['stage']:<18} +{s['added_mib']:.0f} MiB")
        # An out-of-memory here is a *result*, not an error - it is the answer
        # to whether this combination fits.
        return 2
    finally:
        for item in reversed(held):
            try:
                if hasattr(item, "unload"):
                    item.unload()
            except Exception:  # pragma: no cover
                pass


if __name__ == "__main__":
    sys.exit(main())
