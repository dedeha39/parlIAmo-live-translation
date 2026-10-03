#!/usr/bin/env python
"""Evaluate many recogniser configurations from one YAML file.

    python scripts/sweep_asr.py config/sweeps/asr-decoding.yaml
    python scripts/sweep_asr.py config/sweeps/asr-decoding.yaml --dry-run
    python scripts/sweep_asr.py my-sweep.yaml --only "beam_size=5"

Trying a different model, quantisation or decoding setting is a change to the
YAML, not to any code. Each variant is loaded alone and unloaded afterwards, so
its VRAM and timing numbers belong to it.

The sweep file has three sections:

    dataset    where the audio and references come from
    base       settings shared by every variant
    grid       cartesian product of settings to try
    variants   one-off configurations outside the grid

Small-sample honesty: on a 40-utterance set one reference word is worth about
0.15 WER points, so differences of a few tenths are noise. The report prints
the gap in *words* as well as points and refuses to call small gaps a ranking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.eval.datasets import (  # noqa: E402
    load_fleurs,
    load_wav_manifest,
    summarise,
)
from parliamo.eval.normalize import TurkishNormalizer  # noqa: E402
from parliamo.eval.runner import (  # noqa: E402
    Variant,
    expand_grid,
    rank,
    run_variant,
    significance_note,
)
from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.paths import ensure_dir, resolve  # noqa: E402

log = logging.getLogger("sweep_asr")


def load_hotwords(path: str | None) -> str | None:
    if not path:
        return None
    file = resolve(path)
    if not file.exists():
        log.warning("hotword file %s not found", file)
        return None
    words = [
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return " ".join(words) if words else None


def materialise(options: dict[str, Any]) -> dict[str, Any]:
    """Turn sweep-file options into backend keyword arguments.

    ``hotwords_file`` is a path in the YAML and a joined string in the backend,
    so the sweep file can name a list instead of embedding one.
    """
    out = dict(options)
    if "hotwords_file" in out:
        out["hotwords"] = load_hotwords(out.pop("hotwords_file"))
    return out


def build_variants(spec: dict[str, Any]) -> tuple[list[Variant], dict[str, Any]]:
    base = dict(spec.get("base") or {})
    variants: list[Variant] = []

    for variant in expand_grid(base, spec.get("grid") or {}):
        variants.append(Variant(variant.name, materialise(variant.options)))

    for entry in spec.get("variants") or []:
        entry = dict(entry)
        name = entry.pop("name", None) or "unnamed"
        options = dict(base)
        options.update(entry)
        variants.append(Variant(name, materialise(options)))

    return variants, materialise(base)


def load_dataset(spec: dict[str, Any]):
    if spec.get("manifest"):
        return load_wav_manifest(resolve(spec["manifest"]))
    return load_fleurs(
        spec.get("fleurs", "tr_tr"),
        spec.get("split", "test"),
        limit=spec.get("limit") or None,
    )


def render(results: list[dict[str, Any]], dataset: dict[str, Any]) -> None:
    usable = rank(results)
    print("\n" + "=" * 104)
    print(f"SWEEP  -  {dataset['utterances']} utterances, {dataset['total_minutes']} min")
    print("=" * 104)
    if not usable:
        print("no variant produced a result")
        for r in results:
            print(f"  {r['name']}: {r.get('error')}")
        return

    print(f"{'variant':<44}{'WER%':>7}{'CER%':>7}{'RTF':>7}{'p95 s':>8}{'VRAM':>7}{'loops':>7}")
    print("-" * 104)
    for r in usable:
        m = r["metrics"]
        p95 = r.get("latency_s", {}).get("p95", 0.0)
        print(
            f"{r['name'][:43]:<44}"
            f"{m['wer']['percent']:>7.2f}"
            f"{m['cer']['percent']:>7.2f}"
            f"{r['rtf']:>7.3f}"
            f"{p95:>8.3f}"
            f"{r.get('load_vram_mb', 0):>7.0f}"
            f"{r.get('repetition_loops', 0):>7}"
        )
    print("-" * 104)

    for r in results:
        if not r.get("ok"):
            print(f"{r['name'][:43]:<44} FAILED: {r.get('error')}")

    best = usable[0]
    print(f"\nBest: {best['name']}  ({best['metrics']['wer']['percent']:.2f}% WER, "
          f"RTF {best['rtf']:.3f}, {best.get('load_vram_mb', 0):.0f} MiB)")
    print("\nAgainst the best, with sample size taken into account:")
    for r in usable[1:6]:
        print(f"  {r['name'][:43]:<44} {significance_note(best, r)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sweep", help="path to a sweep YAML file")
    parser.add_argument("--only", default=None, help="run only variants whose name contains this")
    parser.add_argument("--dry-run", action="store_true", help="list variants and exit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging(level="INFO", jsonl=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "faster_whisper", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    spec = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    variants, _ = build_variants(spec)
    if args.only:
        variants = [v for v in variants if args.only.lower() in v.name.lower()]
    if not variants:
        print("no variants matched")
        return 1

    if args.dry_run:
        print(f"\n{len(variants)} variants:")
        for v in variants:
            print(f"  {v.name:<44} {v.describe()}")
        return 0

    items = load_dataset(spec.get("dataset") or {})
    if not items:
        log.error("no evaluation items loaded")
        return 1
    dataset = summarise(items)
    print("\ndataset:", json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"\n{len(variants)} variants to run\n")

    normalizer = TurkishNormalizer()
    results = [run_variant(v, items, {}, normalizer) for v in variants]

    # Hypotheses are kept per variant for later inspection but bloat the report;
    # write them to a sidecar instead of the summary.
    out_dir = ensure_dir("runs/asr")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    hyp_path = out_dir / f"sweep-{stamp}-hypotheses.json"
    hyp_path.write_text(
        json.dumps(
            {r["name"]: r.pop("hypotheses", []) for r in results}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "sweep_file": str(args.sweep),
        "dataset": dataset,
        "results": results,
    }
    out_path = Path(args.out) if args.out else out_dir / f"sweep-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    render(results, dataset)
    print(f"\nreport      {out_path}")
    print(f"hypotheses  {hyp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
