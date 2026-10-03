"""Error-rate measurement."""

from __future__ import annotations

import math

import pytest

from parliamo.eval.wer import evaluate, worst_examples


def test_identical_text_scores_zero() -> None:
    result = evaluate(["merhaba dünya"], ["merhaba dünya"])
    assert result["wer"]["rate"] == 0.0
    assert result["cer"]["rate"] == 0.0


def test_one_substitution_in_four_words() -> None:
    result = evaluate(["bir iki üç dört"], ["bir iki beş dört"])
    assert result["wer"]["rate"] == pytest.approx(0.25)
    assert result["wer"]["substitutions"] == 1
    assert result["wer"]["hits"] == 3


def test_deletion_and_insertion_are_counted_separately() -> None:
    deleted = evaluate(["bir iki üç"], ["bir üç"])
    assert deleted["wer"]["deletions"] == 1
    inserted = evaluate(["bir üç"], ["bir iki üç"])
    assert inserted["wer"]["insertions"] == 1


def test_normalisation_removes_formatting_differences() -> None:
    """Punctuation and digit style must not count as recognition errors."""
    result = evaluate(["2026'da İstanbul'da."], ["iki bin yirmi altıda İstanbulda"])
    assert result["wer"]["rate"] == 0.0
    # The raw figure should show the difference the normaliser absorbed.
    assert result["wer_raw"]["rate"] > 0.0


def test_turkish_casing_is_not_an_error() -> None:
    """Guards the dotted-i trap end to end, through the metric."""
    assert evaluate(["IRAK ışık"], ["ırak ışık"])["wer"]["rate"] == 0.0


def test_cer_is_gentler_than_wer_on_a_suffix_error() -> None:
    """Turkish packs grammar into suffixes, so WER over-penalises small damage.

    'gelemeyeceklerini' vs 'gelemeyeceklerin' is one wrong character, but WER
    scores it as a whole word lost. CER is reported alongside for this reason.
    """
    result = evaluate(["gelemeyeceklerini biliyorum"], ["gelemeyeceklerin biliyorum"])
    assert result["wer"]["rate"] == pytest.approx(0.5)
    assert result["cer"]["rate"] < 0.1
    assert result["cer"]["rate"] < result["wer"]["rate"]


def test_empty_reference_is_skipped_not_fatal() -> None:
    result = evaluate(["", "bir iki"], ["bir şey", "bir iki"])
    assert result["wer"]["rate"] == 0.0, "only the non-empty pair should be scored"
    assert result["n"] == 2


def test_all_empty_references_returns_nan_not_a_crash() -> None:
    result = evaluate(["", "  "], ["bir", "iki"])
    assert math.isnan(result["wer"]["rate"])


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        evaluate(["bir", "iki"], ["bir"])


# ---------------------------------------------------------------------------
# worst examples
# ---------------------------------------------------------------------------


def test_worst_examples_ranks_by_error_rate() -> None:
    rows = worst_examples(
        ["a", "b", "c"],
        ["bir iki üç dört", "bir iki üç dört", "bir iki üç dört"],
        ["bir iki üç dört", "bir iki üç beş", "tamamen farklı bir cümle"],
        top=3,
    )
    assert rows[0]["uid"] == "c"
    assert rows[-1]["uid"] == "a"
    assert rows[-1]["wer"] == 0.0


def test_worst_examples_prefers_longer_utterances_on_ties() -> None:
    rows = worst_examples(
        ["short", "long"],
        ["bir iki", "bir iki üç dört beş altı"],
        ["x y", "a b c d e f"],
        top=2,
    )
    assert rows[0]["uid"] == "long", "a 100% error on more words is more informative"


def test_worst_examples_returns_normalised_text() -> None:
    rows = worst_examples(["u1"], ["2026'DA!"], ["yanlış"], top=1)
    assert rows[0]["reference"] == "iki bin yirmi altıda"


def test_worst_examples_skips_empty_references() -> None:
    assert worst_examples(["u1"], [""], ["bir şey"], top=5) == []
