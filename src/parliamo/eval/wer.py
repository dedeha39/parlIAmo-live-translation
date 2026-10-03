"""Word and character error rates, with the caveats that make them comparable.

WER alone is a poor summary for Turkish. Turkish is agglutinative: a single
orthographic word carries what English spreads across several
(``gelemeyeceklerini`` = "that they will not be able to come"). One wrong suffix
destroys a whole word by WER, while the same amount of phonetic damage in
English would cost a fraction of one. CER is reported alongside for that reason
- it degrades proportionally to how much of the word actually went wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .normalize import TurkishNormalizer


@dataclass(slots=True)
class ErrorRate:
    """One error-rate measurement plus the counts it was derived from."""

    rate: float
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def total_reference(self) -> int:
        return self.hits + self.substitutions + self.deletions

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate": round(self.rate, 4),
            "percent": round(self.rate * 100, 2),
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.total_reference,
        }


def _measure(references: Sequence[str], hypotheses: Sequence[str], char: bool) -> ErrorRate:
    import jiwer

    # jiwer raises on empty references; drop those pairs rather than letting one
    # blank line make the whole run unmeasurable.
    pairs = [(r, h) for r, h in zip(references, hypotheses, strict=True) if r.strip()]
    if not pairs:
        return ErrorRate(rate=float("nan"), hits=0, substitutions=0, deletions=0, insertions=0)

    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]

    output = (
        jiwer.process_characters(refs, hyps) if char else jiwer.process_words(refs, hyps)
    )
    rate = output.cer if char else output.wer
    return ErrorRate(
        rate=float(rate),
        hits=int(output.hits),
        substitutions=int(output.substitutions),
        deletions=int(output.deletions),
        insertions=int(output.insertions),
    )


def evaluate(
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalizer: TurkishNormalizer | None = None,
) -> dict[str, Any]:
    """Compute WER and CER, both raw and normalised.

    Reporting both is deliberate. The raw figure includes punctuation and
    casing differences that no listener would notice; the normalised figure
    removes them. A large gap between the two means the models disagree about
    formatting rather than about what was said, and that difference should not
    drive the choice of model.
    """
    normalizer = normalizer or TurkishNormalizer()
    norm_refs = [normalizer(r) for r in references]
    norm_hyps = [normalizer(h) for h in hypotheses]

    return {
        "n": len(references),
        "wer": _measure(norm_refs, norm_hyps, char=False).as_dict(),
        "cer": _measure(norm_refs, norm_hyps, char=True).as_dict(),
        "wer_raw": _measure(list(references), list(hypotheses), char=False).as_dict(),
        "cer_raw": _measure(list(references), list(hypotheses), char=True).as_dict(),
    }


def worst_examples(
    uids: Sequence[str],
    references: Sequence[str],
    hypotheses: Sequence[str],
    normalizer: TurkishNormalizer | None = None,
    top: int = 10,
) -> list[dict[str, Any]]:
    """The utterances a model got most wrong, for eyeballing.

    An aggregate number tells you which model to pick; these tell you what it
    will do to you on stage.
    """
    import jiwer

    normalizer = normalizer or TurkishNormalizer()
    rows: list[dict[str, Any]] = []
    for uid, ref, hyp in zip(uids, references, hypotheses, strict=True):
        nref, nhyp = normalizer(ref), normalizer(hyp)
        if not nref.strip():
            continue
        try:
            rate = float(jiwer.wer(nref, nhyp or " "))
        except Exception:  # pragma: no cover
            continue
        rows.append(
            {
                "uid": uid,
                "wer": round(rate, 3),
                "reference": nref,
                "hypothesis": nhyp,
                "words": len(nref.split()),
            }
        )
    # Sort by rate, then by length: a 100% WER on a two-word utterance is less
    # informative than 60% on a twenty-word one.
    rows.sort(key=lambda r: (-r["wer"], -r["words"]))
    return rows[:top]
