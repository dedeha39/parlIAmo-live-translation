#!/usr/bin/env python
"""Separate the fixed and per-character cost of one synthesis call.

    python scripts/profile_tts_cost.py
    python scripts/profile_tts_cost.py --language tr --repeats 3

Why this matters
----------------
The obvious cheap fix for a slow non-streaming synthesiser is to split a
sentence into clauses, synthesise each, and play them back to back: the first
clause is short, so the audience hears something sooner.

That only works if the cost is mostly *per character*. If a large part of it is
paid once per call - model setup, a fixed-step vocoder - then splitting one
sentence into three pays that cost three times and makes the total worse.

Fitting generation time against text length answers which it is, and therefore
whether chunking is a fix or a trap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402
from parliamo.tts import create_backend  # noqa: E402

log = logging.getLogger("profile_tts")

# A length ladder built from real presentation phrasing rather than padding, so
# the fit is not distorted by unnatural text.
LADDER = {
    "it": [
        "Sì.",
        "Buongiorno a tutti.",
        "Questo sistema funziona sul portatile.",
        "Questo sistema funziona interamente su questo computer portatile.",
        "La clonazione vocale non richiede più lunghe registrazioni: bastano quindici secondi.",
        "Le lingue minoritarie restano sistematicamente indietro nel mondo digitale, "
        "e questo non è un problema tecnico.",
        "Le lingue minoritarie restano sistematicamente indietro nel mondo digitale, "
        "e questo non è un problema tecnico ma una scelta collettiva su dove investire "
        "il tempo e il denaro disponibili.",
    ],
    "tr": [
        "Evet.",
        "Herkese günaydın.",
        "Bu sistem dizüstünde çalışıyor.",
        "Bu sistem tamamen bu dizüstü bilgisayarda çalışıyor.",
        "Ses klonlama için artık uzun kayıtlara ihtiyaç yok, on beş saniye yetiyor.",
        "Azınlık dilleri dijital dünyada sistematik olarak geride kalıyor ve bu teknik "
        "bir sorun değil.",
        "Azınlık dilleri dijital dünyada sistematik olarak geride kalıyor ve bu teknik "
        "bir sorun değil, elimizdeki zamanı ve parayı nereye yatıracağımıza dair ortak "
        "bir tercih.",
    ],
}


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return slope, intercept, r2


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--language", default="it", choices=sorted(LADDER))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging(level="WARNING", jsonl=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "diffusers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    backend = create_backend("chatterbox", device=args.device, language=args.language)
    backend.load()
    backend.warmup()

    rows: list[dict[str, Any]] = []
    print(f"\n{'chars':>6} {'audio s':>8} {'gen s':>7} {'RTF':>6}")
    print("-" * 34)
    for text in LADDER[args.language]:
        times = []
        speech = None
        for _ in range(args.repeats):
            speech = backend.speak(text, language=args.language)
            times.append(speech.compute_s)
        assert speech is not None
        median = sorted(times)[len(times) // 2]
        rows.append({
            "chars": len(text),
            "audio_s": round(speech.duration_s, 3),
            "gen_s": round(median, 3),
            "rtf": round(median / max(speech.duration_s, 1e-9), 3),
            "text": text,
        })
        print(f"{len(text):>6} {speech.duration_s:>8.2f} {median:>7.2f} "
              f"{rows[-1]['rtf']:>6.2f}")

    chars = [float(r["chars"]) for r in rows]
    gens = [float(r["gen_s"]) for r in rows]
    audio = [float(r["audio_s"]) for r in rows]

    slope, intercept, r2 = fit_line(chars, gens)
    a_slope, a_intercept, a_r2 = fit_line(chars, audio)

    print("\n" + "=" * 74)
    print("COST STRUCTURE OF ONE SYNTHESIS CALL")
    print("=" * 74)
    print(f"  generation = {intercept:.2f} s fixed  +  {slope * 1000:.1f} ms per character"
          f"   (R^2 {r2:.3f})")
    print(f"  audio      = {a_intercept:.2f} s        +  {a_slope * 1000:.1f} ms per character"
          f"   (R^2 {a_r2:.3f})")

    longest = max(rows, key=lambda r: r["chars"])
    fixed_share = intercept / longest["gen_s"]
    print(f"\n  On the longest sentence ({longest['chars']} chars, "
          f"{longest['gen_s']:.2f} s) the fixed cost is {100 * fixed_share:.0f}% of the total.")

    print("\n  Does splitting a sentence help?")
    for parts in (2, 3, 4):
        per_part_chars = longest["chars"] / parts
        total = parts * (intercept + slope * per_part_chars)
        first = intercept + slope * per_part_chars
        verdict = "worse" if total > longest["gen_s"] else "better"
        print(f"    into {parts}: total {total:5.2f} s ({verdict} than "
              f"{longest['gen_s']:.2f}), first chunk after {first:.2f} s")

    if intercept > 1.0:
        print(f"\n  VERDICT: {intercept:.1f} s is paid once per call regardless of length.")
        print("  Chunking multiplies that cost - it is a trap, not a fix. What is needed")
        print("  is incremental generation inside a single call, i.e. real streaming.")
    else:
        print("\n  VERDICT: cost is dominated by length, so chunking genuinely helps.")

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "language": args.language,
        "fit": {
            "generation_fixed_s": round(intercept, 3),
            "generation_ms_per_char": round(slope * 1000, 2),
            "generation_r2": round(r2, 4),
            "audio_fixed_s": round(a_intercept, 3),
            "audio_ms_per_char": round(a_slope * 1000, 2),
        },
        "measurements": rows,
    }
    path = args.out or ensure_dir("runs/tts") / f"cost-{datetime.now():%Y%m%d-%H%M%S}.json"
    from pathlib import Path

    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {path}")

    backend.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
