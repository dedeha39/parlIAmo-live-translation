#!/usr/bin/env python
"""Rank speech recognisers on Turkish, by measurement rather than by reputation.

    python scripts/bench_asr.py --limit 200
    python scripts/bench_asr.py --models large-v3-turbo,large-v3 --limit 400
    python scripts/bench_asr.py --manifest data/insitu/manifest.tsv

Each model is loaded alone, warmed up, run over the whole set, then unloaded, so
the VRAM and timing figures belong to that model and not to whatever was
resident beforehand.

What the numbers mean
---------------------
**WER** is the headline, but read it next to **CER**. Turkish packs grammar into
suffixes, so a single wrong suffix costs a whole word by WER while damaging only
a few characters. A model that loses to another on WER but matches it on CER is
mangling endings, not misunderstanding words - which matters, because the
translation stage downstream can often recover from the latter and not the
former.

**RTF** is compute seconds per audio second, measured on this GPU. It must stay
well under 1.0: translation and synthesis share the same 8 GB card, so an ASR
stage that only just keeps up leaves nothing for them.

Public data ranks models. It does not predict accuracy on a particular speaker,
microphone and room - run ``--manifest`` against an in-situ recording for that.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from parliamo.paths import configure_model_cache

configure_model_cache()  # before any model library is imported

from parliamo.asr import create_backend  # noqa: E402
from parliamo.eval.datasets import (  # noqa: E402
    Utterance,
    load_fleurs,
    load_wav_manifest,
    summarise,
)
from parliamo.eval.normalize import TurkishNormalizer  # noqa: E402
from parliamo.eval.wer import evaluate, worst_examples  # noqa: E402
from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402

log = logging.getLogger("bench_asr")

DEFAULT_MODELS = ["small", "medium", "large-v3-turbo", "large-v3"]


def load_hotwords(path: str | None) -> str | None:
    """Read a hotword list into the single string faster-whisper expects."""
    if not path:
        return None
    from parliamo.paths import resolve

    file = resolve(path)
    if not file.exists():
        log.warning("hotword file %s not found; continuing without bias", file)
        return None
    words = [
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not words:
        return None
    log.info("biasing recognition toward %d hotwords from %s", len(words), file.name)
    return " ".join(words)


def run_model(
    model: str,
    items: list[Utterance],
    language: str,
    compute_type: str,
    device: str,
    normalizer: TurkishNormalizer,
    hotwords: str | None = None,
) -> dict[str, Any]:
    """Load one model, transcribe everything, unload, and report."""
    log.info("=" * 70)
    log.info("model: %s (%s, %s)%s", model, device, compute_type,
             " +hotwords" if hotwords else "")

    backend = create_backend(
        "faster_whisper",
        model=model,
        device=device,
        language=language,
        compute_type=compute_type,
        hotwords=hotwords,
    )

    result: dict[str, Any] = {
        "model": model + (" +hotwords" if hotwords else ""),
        "compute_type": compute_type,
        "hotwords": bool(hotwords),
    }
    try:
        backend.load()
        warmup_s = backend.warmup(seconds=3.0)
        result["warmup_s"] = round(warmup_s, 3)
        # describe() reports the bare model name, which would overwrite the
        # label distinguishing the biased run from the unbiased one - so the
        # comparison table showed two identical rows.
        label = result["model"]
        result.update(backend.describe())
        result["model"] = label

        hypotheses: list[str] = []
        total_audio = 0.0
        total_compute = 0.0
        failures = 0

        for index, item in enumerate(items, start=1):
            try:
                transcript = backend.transcribe(item.audio, item.sample_rate)
                hypotheses.append(transcript.text)
                total_audio += transcript.audio_duration_s
                total_compute += transcript.compute_s
            except Exception as exc:
                log.warning("utterance %s failed: %s", item.uid, exc)
                hypotheses.append("")
                failures += 1
            if index % 50 == 0:
                log.info("  %d/%d  (rtf so far %.3f)", index, len(items),
                         total_compute / max(total_audio, 1e-9))

        references = [i.reference for i in items]
        result["metrics"] = evaluate(references, hypotheses, normalizer)
        result["rtf"] = round(total_compute / max(total_audio, 1e-9), 4)
        result["total_audio_s"] = round(total_audio, 1)
        result["total_compute_s"] = round(total_compute, 1)
        result["failures"] = failures
        result["worst"] = worst_examples(
            [i.uid for i in items], references, hypotheses, normalizer, top=8
        )
        result["ok"] = True
    except Exception as exc:
        log.exception("model %s failed to run", model)
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        backend.unload()

    return result


def render_table(results: list[dict[str, Any]]) -> None:
    usable = [r for r in results if r.get("ok")]
    if not usable:
        print("\nNo model produced a result.")
        return

    usable.sort(key=lambda r: r["metrics"]["wer"]["rate"])

    print("\n" + "=" * 92)
    print("ASR COMPARISON  (sorted by normalised WER, lower is better)")
    print("=" * 92)
    header = f"{'model':<20} {'WER%':>7} {'CER%':>7} {'WERraw%':>8} {'RTF':>7} {'VRAM MiB':>9} {'load s':>7}"
    print(header)
    print("-" * 92)
    for r in usable:
        m = r["metrics"]
        print(
            f"{r['model']:<20} "
            f"{m['wer']['percent']:>7.2f} "
            f"{m['cer']['percent']:>7.2f} "
            f"{m['wer_raw']['percent']:>8.2f} "
            f"{r['rtf']:>7.3f} "
            f"{r.get('load_vram_mb', 0):>9.0f} "
            f"{r.get('load_seconds', 0):>7.1f}"
        )
    print("-" * 92)

    failed = [r for r in results if not r.get("ok")]
    for r in failed:
        print(f"{r['model']:<20} FAILED: {r.get('error')}")

    best = usable[0]
    print(f"\nLowest WER: {best['model']} at {best['metrics']['wer']['percent']:.2f}% "
          f"(CER {best['metrics']['cer']['percent']:.2f}%, RTF {best['rtf']:.3f})")

    if len(usable) > 1:
        second = usable[1]
        gap = second["metrics"]["wer"]["percent"] - best["metrics"]["wer"]["percent"]
        speed = second["rtf"] / max(best["rtf"], 1e-9)
        print(f"Runner-up:  {second['model']} at {second['metrics']['wer']['percent']:.2f}% "
              f"(+{gap:.2f} points, {speed:.2f}x the RTF)")

    print("\nWorst utterances for the leading model:")
    for row in best.get("worst", [])[:5]:
        print(f"  WER {row['wer']:.2f}  ref: {row['reference'][:70]}")
        print(f"              hyp: {row['hypothesis'][:70]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated model names or Hugging Face ids")
    parser.add_argument("--dataset", default="fleurs", choices=["fleurs"])
    parser.add_argument("--fleurs-config", default="tr_tr")
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", default=None,
                        help="TSV of 'path<TAB>reference' instead of a public set")
    parser.add_argument("--limit", type=int, default=200,
                        help="cap the number of utterances (0 = all)")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--hotwords", default="config/hotwords.tr.txt",
                        help="vocabulary to bias toward; '' to disable")
    parser.add_argument("--compare-hotwords", action="store_true",
                        help="run each model twice, with and without the bias")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging(level="INFO", jsonl=True)
    # These libraries narrate every HTTP request at INFO, which buries ours.
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.manifest:
        items = load_wav_manifest(args.manifest)
    else:
        items = load_fleurs(
            args.fleurs_config, args.split, limit=args.limit or None
        )
    if not items:
        log.error("no evaluation items loaded")
        return 1

    print("\nEvaluation set:", json.dumps(summarise(items), indent=2, ensure_ascii=False))

    normalizer = TurkishNormalizer()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    hotwords = load_hotwords(args.hotwords)

    results = []
    for m in models:
        results.append(
            run_model(m, items, args.language, args.compute_type, args.device,
                      normalizer, hotwords=None)
        )
        if hotwords and args.compare_hotwords:
            # Same model, same audio, biased decoding - so the difference is
            # attributable to the bias and nothing else.
            results.append(
                run_model(m, items, args.language, args.compute_type, args.device,
                          normalizer, hotwords=hotwords)
            )

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset": summarise(items),
        "device": args.device,
        "compute_type": args.compute_type,
        "language": args.language,
        "results": results,
    }

    out_dir = ensure_dir("runs/asr")
    out_path = (
        args.out and __import__("pathlib").Path(args.out)
        or out_dir / f"bench-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    render_table(results)
    print(f"\nreport written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
