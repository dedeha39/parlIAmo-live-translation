"""Translation metrics.

The point these tests pin down: **chrF++ and BLEU disagree in a specific,
predictable way on morphologically rich languages**, and that is why chrF++ is
the primary number. If someone later swaps the primary metric to BLEU because
it is more familiar, these tests should make the consequence obvious.
"""

from __future__ import annotations

import pytest

from parliamo.eval.mt_metrics import score, sentence_chrf, worst_translations


def test_perfect_translation_scores_maximum() -> None:
    refs = ["Il traduttore funziona bene.", "Questo è un test."]
    result = score(refs, refs)
    assert result.chrf == pytest.approx(100.0, abs=0.01)
    assert result.bleu == pytest.approx(100.0, abs=0.01)
    assert result.length_ratio == pytest.approx(1.0)


def test_unrelated_translation_scores_low() -> None:
    result = score(["Il gatto dorme sul divano."], ["La borsa valori ha chiuso in rialzo."])
    assert result.chrf < 30
    assert result.bleu < 10


def test_inflection_difference_keeps_partial_credit_in_chrf() -> None:
    """The reason chrF++ is primary.

    A different but valid inflection is a total loss under BLEU's word n-grams
    and a small loss under chrF++'s character n-grams. On Turkish and Italian
    that difference is the difference between a usable metric and noise.
    """
    reference = "I ricercatori hanno pubblicato i risultati dello studio."
    inflected = "Il ricercatore ha pubblicato il risultato dello studio."

    result = score([inflected], [reference])
    # Measured: chrF++ 59.8, BLEU 19.1. The claim is about the *ratio*, not
    # about either absolute value - the first version of this test asserted
    # thresholds picked by intuition and both were wrong.
    assert result.chrf > 2.5 * result.bleu, (
        f"chrF++ {result.chrf:.1f} should be far above BLEU {result.bleu:.1f} here"
    )
    assert result.chrf > 50, "meaning-preserving inflection should keep real credit"
    assert result.bleu < 25, "BLEU should collapse on changed word forms"


def test_agglutinative_suffix_change_is_survivable() -> None:
    """One Turkish word carries what Italian spreads over several.

    Dropping the final ``i`` from ``gelemeyeceklerini`` is a whole word lost by
    any word-level metric. chrF++ scores it 80.8 - damaged, not destroyed.
    """
    damaged = sentence_chrf("gelemeyeceklerin", "gelemeyeceklerini")
    assert damaged > 75, f"one lost character scored {damaged:.1f}"
    unrelated = sentence_chrf("kahvaltı", "gelemeyeceklerini")
    assert damaged > 3 * unrelated, "a suffix slip must score far above an unrelated word"


def test_empty_output_is_counted_and_scores_zero() -> None:
    result = score(["", "Questo è un test."], ["Qualcosa.", "Questo è un test."])
    assert result.empty_outputs == 1
    assert 0 < result.chrf < 100


def test_length_ratio_detects_truncation() -> None:
    """Runaway or truncated generation shows up here before it shows up in chrF."""
    reference = "Questa è una frase piuttosto lunga con molte parole al suo interno."
    truncated = score(["Questa è"], [reference])
    assert truncated.length_ratio < 0.3

    runaway = score([reference * 4], [reference])
    assert runaway.length_ratio > 3.0


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="hypotheses"):
        score(["uno"], ["uno", "due"])


def test_empty_corpus_is_safe() -> None:
    result = score([], [])
    assert result.n == 0
    assert result.chrf == 0.0


# ---------------------------------------------------------------------------
# worst-case inspection
# ---------------------------------------------------------------------------


def test_worst_translations_ranks_by_chrf() -> None:
    rows = worst_translations(
        ["kaynak bir", "kaynak iki", "kaynak üç"],
        ["Questo è un test.", "Completamente sbagliato qui.", "Quasi un test."],
        ["Questo è un test.", "Il gatto dorme.", "Questo è un test."],
        top=3,
    )
    assert rows[0]["chrf"] < rows[-1]["chrf"]
    assert rows[-1]["chrf"] == pytest.approx(100.0, abs=0.01)


def test_worst_translations_keeps_the_source_for_context() -> None:
    rows = worst_translations(["merhaba dünya"], ["sbagliato"], ["ciao mondo"], top=1)
    assert rows[0]["source"] == "merhaba dünya"
    assert rows[0]["reference"] == "ciao mondo"


def test_worst_translations_skips_empty_references() -> None:
    assert worst_translations(["a"], ["x"], [""], top=5) == []


def test_sentence_chrf_with_empty_reference_is_zero() -> None:
    assert sentence_chrf("qualsiasi cosa", "") == 0.0
