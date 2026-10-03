"""Sentence splitting, and the Turkish it must not break.

The splitter exists because NLLB-600M drops trailing sentences (see
parliamo/mt/sentences.py). Its risk is the opposite failure: splitting where
Turkish does not have a sentence boundary, which would hand the translator
fragments and produce nonsense in a different way.
"""

from __future__ import annotations

from parliamo.mt.sentences import split_sentences

# ---------------------------------------------------------------------------
# the case this was written for
# ---------------------------------------------------------------------------


def test_the_measured_failure_is_split() -> None:
    """The exact sentence whose second half NLLB-600M discarded."""
    parts = split_sentences("Aklımdan geçen ilk soru şuydu. Acaba gerçekten çalışacak mı?")
    assert parts == [
        "Aklımdan geçen ilk soru şuydu.",
        "Acaba gerçekten çalışacak mı?",
    ]


def test_three_sentences() -> None:
    parts = split_sentences("Birinci cümle budur. İkinci cümle şudur. Üçüncü cümle de vardır.")
    assert len(parts) == 3


def test_question_and_exclamation_end_sentences() -> None:
    assert len(split_sentences("Gerçekten mi? Evet! Kesinlikle.")) == 3


def test_single_sentence_is_returned_whole() -> None:
    text = "Bu teknik bir mesele değil, etik bir mesele."
    assert split_sentences(text) == [text]


def test_colon_is_not_a_boundary() -> None:
    """Measured: splitting at the colon made the translation worse, not better.

    The colon carries the link between the two halves, and NLLB handles the
    construction correctly when it is left intact - it is the full stop that
    breaks it.
    """
    text = "Bu sunumu hazırlarken aklımdan geçen ilk soru şuydu: acaba çalışacak mı?"
    assert split_sentences(text) == [text]


# ---------------------------------------------------------------------------
# Turkish that must not be split
# ---------------------------------------------------------------------------


def test_ordinals_are_not_sentence_ends() -> None:
    """Turkish writes ordinals with a full stop: '1. Dünya Savaşı'."""
    text = "1. Dünya Savaşı 1914 yılında başladı."
    assert split_sentences(text) == [text]


def test_ordinal_mid_sentence() -> None:
    text = "Öğrenciler 3. sınıfta bu konuyu görüyor."
    assert split_sentences(text) == [text]


def test_abbreviations_do_not_end_a_sentence() -> None:
    text = "Dr. Yılmaz sunumu yaptı."
    assert split_sentences(text) == [text]


def test_vb_abbreviation() -> None:
    text = "Modeller, veri kümeleri vb. konular ele alındı."
    assert split_sentences(text) == [text]


def test_initials_are_not_sentence_ends() -> None:
    text = "A. Yılmaz bu makaleyi yazdı."
    assert split_sentences(text) == [text]


def test_thousands_separator_is_untouched() -> None:
    """40.000 has no space after the stop, so it was never at risk - but check."""
    text = "Salonda 40.000 kişi vardı."
    assert split_sentences(text) == [text]


def test_decimal_comma_is_untouched() -> None:
    text = "Değer 3,14 olarak ölçüldü."
    assert split_sentences(text) == [text]


def test_abbreviation_then_a_real_sentence() -> None:
    parts = split_sentences("Dr. Yılmaz sunumu yaptı. Herkes dinledi.")
    assert parts == ["Dr. Yılmaz sunumu yaptı.", "Herkes dinledi."]


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_no_terminal_punctuation() -> None:
    """ASR output often has none at all - it must still translate."""
    text = "bugün sizinle biraz teknolojiden bahsetmek istiyorum"
    assert split_sentences(text) == [text]


def test_trailing_punctuation_does_not_produce_an_empty_piece() -> None:
    assert split_sentences("Evet. ") == ["Evet."]


def test_closing_quote_after_the_stop() -> None:
    parts = split_sentences('Adam "bu doğru." dedi. Sonra gitti.')
    assert all(p.strip() for p in parts)
    assert len(parts) == 2


def test_runaway_text_is_not_shattered() -> None:
    """A recogniser loop must not turn into 200 translation calls."""
    text = " ".join("Bu bir cümle." for _ in range(50))
    assert split_sentences(text) == [text.strip()]


def test_nothing_is_lost() -> None:
    """Every word in must appear in the output - the whole point is not losing text."""
    text = "Merhaba. Bugün buradayım! Neden mi? Çünkü önemli."
    parts = split_sentences(text)
    assert " ".join(parts).split() == text.split()
