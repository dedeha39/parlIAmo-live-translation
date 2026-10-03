"""Run one recogniser configuration over one dataset and report.

Extracted so that a single-model benchmark and a parameter sweep share exactly
the same evaluation path. If they did not, a sweep result and a benchmark
result would not be comparable, and the whole point is comparing them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ..asr import create_backend
from .datasets import Utterance
from .normalize import TurkishNormalizer
from .wer import evaluate, worst_examples

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Variant:
    """One named recogniser configuration to evaluate."""

    name: str
    options: dict[str, Any] = field(default_factory=dict)

    def backend_kwargs(self, defaults: dict[str, Any]) -> dict[str, Any]:
        merged = dict(defaults)
        merged.update(self.options)
        merged.pop("backend", None)
        return merged

    def describe(self) -> str:
        interesting = {
            k: v for k, v in self.options.items()
            if k not in {"backend", "device", "language"}
        }
        return ", ".join(f"{k}={v}" for k, v in sorted(interesting.items()))


def run_variant(
    variant: Variant,
    items: list[Utterance],
    defaults: dict[str, Any],
    normalizer: TurkishNormalizer | None = None,
    progress_every: int = 50,
) -> dict[str, Any]:
    """Load, transcribe everything, unload. One variant resident at a time.

    Keeping exactly one model loaded is what makes the VRAM and timing figures
    attributable to that variant rather than to whatever was already resident.
    """
    normalizer = normalizer or TurkishNormalizer()
    kwargs = variant.backend_kwargs(defaults)
    backend_name = variant.options.get("backend", defaults.get("backend", "faster_whisper"))

    log.info("-" * 70)
    log.info("variant: %s  [%s]", variant.name, variant.describe() or "defaults")

    result: dict[str, Any] = {
        "name": variant.name,
        "backend": backend_name,
        "options": {k: v for k, v in variant.options.items() if k != "backend"},
    }

    backend = None
    try:
        backend = create_backend(backend_name, **kwargs)
        backend.load()
        result["warmup_s"] = round(backend.warmup(seconds=3.0), 3)
        result["load_vram_mb"] = backend.load_vram_mb
        result["load_seconds"] = round(backend.load_seconds, 2)

        hypotheses: list[str] = []
        total_audio = 0.0
        total_compute = 0.0
        per_utterance_s: list[float] = []
        failures = 0
        loops = 0

        for index, item in enumerate(items, start=1):
            try:
                t0 = time.perf_counter()
                transcript = backend.transcribe(item.audio, item.sample_rate)
                elapsed = time.perf_counter() - t0
                hypotheses.append(transcript.text)
                per_utterance_s.append(elapsed)
                total_audio += transcript.audio_duration_s
                total_compute += elapsed
                loops += len(transcript.dropped_segments)
            except Exception as exc:
                log.warning("utterance %s failed: %s", item.uid, exc)
                hypotheses.append("")
                failures += 1
            if progress_every and index % progress_every == 0:
                log.info("  %d/%d", index, len(items))

        references = [i.reference for i in items]
        result["metrics"] = evaluate(references, hypotheses, normalizer)
        result["rtf"] = round(total_compute / max(total_audio, 1e-9), 4)
        result["total_audio_s"] = round(total_audio, 1)
        result["total_compute_s"] = round(total_compute, 1)
        result["failures"] = failures
        result["repetition_loops"] = loops
        if per_utterance_s:
            ordered = sorted(per_utterance_s)
            result["latency_s"] = {
                "mean": round(sum(ordered) / len(ordered), 3),
                "p50": round(ordered[len(ordered) // 2], 3),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
                "max": round(ordered[-1], 3),
            }
        result["worst"] = worst_examples(
            [i.uid for i in items], references, hypotheses, normalizer, top=8
        )
        result["hypotheses"] = hypotheses
        result["ok"] = True
    except Exception as exc:
        log.exception("variant %s failed", variant.name)
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.unload()

    return result


def expand_grid(base: dict[str, Any], grid: dict[str, list[Any]]) -> list[Variant]:
    """Cartesian product of *grid* over *base*, as named variants."""
    import itertools

    if not grid:
        return []
    keys = sorted(grid)
    variants: list[Variant] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        options = dict(base)
        options.update(dict(zip(keys, combo, strict=True)))
        label = " ".join(f"{k}={v}" for k, v in zip(keys, combo, strict=True))
        variants.append(Variant(name=label, options=options))
    return variants


def rank(results: list[dict[str, Any]], by: str = "wer") -> list[dict[str, Any]]:
    usable = [r for r in results if r.get("ok")]
    usable.sort(key=lambda r: r["metrics"][by]["rate"])
    return usable


def significance_note(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Say whether a WER difference is worth acting on, given the sample size.

    A 40-utterance set is small. Treating a 0.3-point difference on it as a
    real ranking is how a measurement culture turns into a superstition, so the
    comparison prints the reference-word count alongside the gap and refuses to
    call small differences meaningful.
    """
    wa, wb = a["metrics"]["wer"], b["metrics"]["wer"]
    gap = wb["percent"] - wa["percent"]
    n_words = wa["reference_length"]
    # One word in the reference is worth 100/n points, so a gap smaller than a
    # few words' worth is inside the noise of this dataset.
    per_word = 100.0 / max(n_words, 1)
    if abs(gap) < 3 * per_word:
        return (
            f"{gap:+.2f} points over {n_words} words - within ~3 words of each "
            "other, treat as a tie"
        )
    return f"{gap:+.2f} points over {n_words} words (~{abs(gap) / per_word:.0f} words)"
