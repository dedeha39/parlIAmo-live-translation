#!/usr/bin/env python
"""Measure translation quality, speed and VRAM for one or more backends.

    python scripts/bench_mt.py --limit 200
    python scripts/bench_mt.py --target en --limit 100
    python scripts/bench_mt.py --device cpu --limit 100

The question this exists to answer first: **is a dedicated 600M NMT model good
enough, or is a multi-billion-parameter LLM actually needed?** An LLM costs
roughly four times the VRAM and an order of magnitude more latency per
sentence, and brings failure modes NMT does not have - it can refuse, explain
itself, or answer a question it found in the text. That price is worth paying
only if the quality difference earns it, which is a measurement, not an
assumption.

chrF++ is the headline. See parliamo/eval/mt_metrics.py for why not BLEU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.eval.datasets import (  # noqa: E402
    SentencePair,
    load_fleurs_parallel,
    load_tsv_pairs,
    summarise_pairs,
)
from parliamo.eval.mt_metrics import score, worst_translations  # noqa: E402
from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.mt import create_backend  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402

log = logging.getLogger("bench_mt")


def run_backend(
    label: str,
    backend_name: str,
    options: dict[str, Any],
    pairs: list[SentencePair],
    batch_size: int,
) -> dict[str, Any]:
    log.info("-" * 70)
    log.info("%s  (%s)", label, backend_name)

    result: dict[str, Any] = {"name": label, "backend": backend_name, "options": dict(options)}
    backend = None
    try:
        backend = create_backend(backend_name, **options)
        backend.load()
        result["warmup_s"] = round(backend.warmup(), 3)
        result["load_vram_mb"] = backend.load_vram_mb
        result["load_seconds"] = round(backend.load_seconds, 2)

        sources = [p.source for p in pairs]
        references = [p.reference for p in pairs]
        hypotheses: list[str] = []
        repaired = 0

        # Sentence-at-a-time latency is what the stage experiences; batch
        # throughput is what an offline pass would see. Measure both.
        t0 = time.perf_counter()
        for start in range(0, len(sources), batch_size):
            chunk = sources[start : start + batch_size]
            for tr in backend.translate_many(chunk):
                hypotheses.append(tr.text)
                if tr.repaired:
                    repaired += 1
            if start and start % (batch_size * 10) == 0:
                log.info("  %d/%d", start, len(sources))
        batch_seconds = time.perf_counter() - t0

        single: list[float] = []
        for source in sources[: min(30, len(sources))]:
            t1 = time.perf_counter()
            backend.translate(source)
            single.append(time.perf_counter() - t1)

        metrics = score(hypotheses, references, target_lang=pairs[0].target_lang)
        result["metrics"] = metrics.as_dict()
        result["repaired"] = repaired
        result["batch_ms_per_sentence"] = round(batch_seconds / max(len(sources), 1) * 1000, 1)
        if single:
            ordered = sorted(single)
            result["single_ms"] = {
                "mean": round(sum(ordered) / len(ordered) * 1000, 1),
                "p50": round(ordered[len(ordered) // 2] * 1000, 1),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000, 1),
            }
        result["worst"] = worst_translations(sources, hypotheses, references, top=6)
        result["hypotheses"] = hypotheses
        result["ok"] = True
    except Exception as exc:
        log.exception("%s failed", label)
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.unload()
    return result


def render(results: list[dict[str, Any]], direction: str) -> None:
    usable = [r for r in results if r.get("ok")]
    usable.sort(key=lambda r: -r["metrics"]["chrf"])

    print("\n" + "=" * 96)
    print(f"TRANSLATION  {direction}   (chrF++ is the headline; higher is better)")
    print("=" * 96)
    print(f"{'backend':<38}{'chrF++':>8}{'BLEU':>7}{'len':>7}{'ms/sent':>9}{'p95 ms':>8}{'VRAM':>7}")
    print("-" * 96)
    for r in usable:
        m = r["metrics"]
        p95 = r.get("single_ms", {}).get("p95", 0.0)
        print(
            f"{r['name'][:37]:<38}{m['chrf']:>8.2f}{m['bleu']:>7.2f}"
            f"{m['length_ratio']:>7.2f}{r['batch_ms_per_sentence']:>9.1f}"
            f"{p95:>8.1f}{r.get('load_vram_mb', 0):>7.0f}"
        )
    print("-" * 96)
    for r in results:
        if not r.get("ok"):
            print(f"{r['name'][:37]:<38} FAILED: {r.get('error')}")

    if not usable:
        return
    best = usable[0]
    print(f"\nBest: {best['name']}  chrF++ {best['metrics']['chrf']:.2f}, "
          f"{best['metrics']['empty_outputs']} empty, {best.get('repaired', 0)} repaired")
    print("\nWorst sentences for the leader:")
    for row in best.get("worst", [])[:4]:
        print(f"  chrF {row['chrf']:5.1f}")
        print(f"    src: {row['source'][:82]}")
        print(f"    out: {row['hypothesis'][:82]}")
        print(f"    ref: {row['reference'][:82]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", default="tr")
    parser.add_argument("--target", default="it")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--pairs", default=None, help="TSV of source<TAB>reference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nllb", default="models/ct2/nllb-200-distilled-600M")
    parser.add_argument("--beam-sizes", default="4", help="comma-separated NLLB beam sizes")
    parser.add_argument("--split-sentences", action="store_true",
                        help="translate sentence by sentence (see parliamo.mt.sentences)")
    parser.add_argument("--ollama", action="append", default=[],
                        help="also measure a model served by a local Ollama, by its name "
                             "(repeatable); the prompt comes from mt.ollama_backend")
    parser.add_argument("--skip-nllb", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging(level="INFO", jsonl=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "datasets", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    fleurs = {"tr": "tr_tr", "it": "it_it", "en": "en_us", "de": "de_de", "fr": "fr_fr",
              "es": "es_419"}
    if args.pairs:
        pairs = load_tsv_pairs(args.pairs, args.source, args.target)
    else:
        pairs = load_fleurs_parallel(
            fleurs[args.source], fleurs[args.target], limit=args.limit or None
        )
    if not pairs:
        log.error("no sentence pairs loaded")
        return 1
    dataset = summarise_pairs(pairs)
    print("\ndataset:", json.dumps(dataset, indent=2, ensure_ascii=False))

    # Name the row after the model that was actually loaded. The label used to
    # be the literal string "NLLB-600M", so running --nllb against the 1.3B
    # checkpoint produced a report confidently attributing 1.3B's numbers to
    # 600M - a measurement that lies is worse than no measurement.
    model_label = Path(args.nllb).name.replace("nllb-200-distilled-", "NLLB-")
    if args.split_sentences:
        model_label += " split"

    results = []
    for model in args.ollama:
        # One sentence per request, as on stage: an LLM gains nothing from a
        # batch here, and a batch would hide the per-sentence latency.
        results.append(
            run_backend(
                f"{model} (ollama)" + (" split" if args.split_sentences else ""),
                "ollama",
                {"model": model, "source_lang": args.source, "target_lang": args.target,
                 "split_sentences": args.split_sentences},
                pairs,
                1,
            )
        )
    beams = [] if args.skip_nllb else [int(b) for b in args.beam_sizes.split(",") if b.strip()]
    for beam in beams:
        results.append(
            run_backend(
                f"{model_label} {args.device} beam={beam}",
                "ctranslate2_nllb",
                {
                    "model": args.nllb,
                    "device": args.device,
                    "source_lang": args.source,
                    "target_lang": args.target,
                    "beam_size": beam,
                    "split_sentences": args.split_sentences,
                },
                pairs,
                args.batch_size,
            )
        )

    out_dir = ensure_dir("runs/mt")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"mt-{stamp}-hypotheses.json").write_text(
        json.dumps({r["name"]: r.pop("hypotheses", []) for r in results},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset": dataset,
        "results": results,
    }
    out_path = out_dir / f"mt-{stamp}.json" if not args.out else args.out
    Path(out_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    render(results, dataset["direction"])
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
