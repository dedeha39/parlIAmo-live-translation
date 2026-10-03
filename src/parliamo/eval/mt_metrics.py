"""Translation quality metrics.

**chrF++ is the primary number, not BLEU.**

BLEU counts matching word n-grams. Turkish and Italian both inflect heavily,
and Turkish is agglutinative on top of that: ``gelemeyeceklerini`` is one token
carrying what English spreads over six. A translation that gets the meaning
right but picks a different valid inflection scores zero on that word by BLEU,
while a fluent-sounding mistranslation of a common phrase can score well. On
morphologically rich language pairs BLEU is close to noise at the sentence
level.

chrF++ scores character n-grams plus a little word n-gram signal, so partial
credit survives inflection. It is what WMT uses for exactly this reason, and it
correlates far better with human judgement on these pairs.

Both are reported, because a large BLEU/chrF++ divergence is itself
informative: it usually means the system is producing the right content with
different surface forms, which downstream speech synthesis does not care about.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MTScore:
    chrf: float
    bleu: float
    n: int
    #: Mean output length relative to reference, in characters. Far from 1.0
    #: signals truncation or runaway generation rather than bad word choice.
    length_ratio: float
    empty_outputs: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "chrf": round(self.chrf, 2),
            "bleu": round(self.bleu, 2),
            "n": self.n,
            "length_ratio": round(self.length_ratio, 3),
            "empty_outputs": self.empty_outputs,
        }


def score(
    hypotheses: Sequence[str],
    references: Sequence[str],
    target_lang: str = "it",
) -> MTScore:
    """Score *hypotheses* against *references*.

    No text normalisation is applied. sacrebleu owns tokenisation so that our
    numbers are comparable with published ones; normalising first would make
    them incomparable with everything, including our own earlier runs.
    """
    import sacrebleu

    if len(hypotheses) != len(references):
        raise ValueError(
            f"got {len(hypotheses)} hypotheses for {len(references)} references"
        )
    if not references:
        return MTScore(chrf=0.0, bleu=0.0, n=0, length_ratio=0.0, empty_outputs=0)

    hyps = [h.strip() for h in hypotheses]
    refs = [r.strip() for r in references]

    # word_order=2 is what makes it chrF++ rather than plain chrF.
    chrf = sacrebleu.CHRF(word_order=2).corpus_score(hyps, [refs]).score
    bleu = sacrebleu.BLEU(trg_lang=target_lang).corpus_score(hyps, [refs]).score

    ref_chars = sum(len(r) for r in refs) or 1
    hyp_chars = sum(len(h) for h in hyps)

    return MTScore(
        chrf=float(chrf),
        bleu=float(bleu),
        n=len(refs),
        length_ratio=hyp_chars / ref_chars,
        empty_outputs=sum(1 for h in hyps if not h),
    )


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """chrF++ for one sentence, used for ranking individual failures."""
    import sacrebleu

    if not reference.strip():
        return 0.0
    return float(
        sacrebleu.CHRF(word_order=2).sentence_score(hypothesis.strip(), [reference.strip()]).score
    )


def worst_translations(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Sequence[str],
    top: int = 8,
) -> list[dict[str, Any]]:
    """The sentences a system handled worst, for reading rather than counting.

    An aggregate score chooses the system; these show what it will do on stage.
    """
    rows: list[dict[str, Any]] = []
    for src, hyp, ref in zip(sources, hypotheses, references, strict=True):
        if not ref.strip():
            continue
        rows.append(
            {
                "chrf": round(sentence_chrf(hyp, ref), 2),
                "source": src,
                "hypothesis": hyp,
                "reference": ref,
                "chars": len(ref),
            }
        )
    rows.sort(key=lambda r: (r["chrf"], -r["chars"]))
    return rows[:top]
