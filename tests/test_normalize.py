"""Turkish text normalisation for WER comparison."""

from __future__ import annotations

import pytest

from parliamo.eval.normalize import (
    NormalizerOptions,
    TurkishNormalizer,
    number_to_turkish,
    spell_numbers,
    turkish_lower,
    turkish_upper,
)

# ---------------------------------------------------------------------------
# the dotted-i trap
# ---------------------------------------------------------------------------


def test_dotless_capital_I_lowercases_to_dotless() -> None:
    """Python's str.lower() gets this wrong, and it changes word identity.

    'Irak' (Iraq) lowercases to 'ırak' (far away) in Turkish. Under the default
    Unicode mapping it becomes 'irak', which is a different string from the
    reference and counts as a substitution error that never happened.
    """
    assert turkish_lower("IRAK") == "ırak"
    assert "IRAK".lower() == "irak"  # the behaviour we must not use


def test_dotted_capital_I_lowercases_to_dotted() -> None:
    assert turkish_lower("İSTANBUL") == "istanbul"
    assert turkish_lower("İzmir") == "izmir"


def test_mixed_i_forms_in_one_word() -> None:
    assert turkish_lower("IŞIK") == "ışık"
    assert turkish_lower("İLGİ") == "ilgi"


def test_uppercase_round_trip() -> None:
    assert turkish_upper("ışık") == "IŞIK"
    assert turkish_upper("ilgi") == "İLGİ"


def test_non_turkish_letters_are_untouched() -> None:
    assert turkish_lower("ABCÇDEFGĞHJKLMNOÖPRSŞTUÜVYZ") == "abcçdefgğhjklmnoöprsştuüvyz"


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "sıfır"),
        (1, "bir"),
        (9, "dokuz"),
        (10, "on"),
        (11, "on bir"),
        (20, "yirmi"),
        (42, "kırk iki"),
        (90, "doksan"),
        (100, "yüz"),          # not "bir yüz"
        (101, "yüz bir"),
        (200, "iki yüz"),
        (999, "dokuz yüz doksan dokuz"),
        (1000, "bin"),         # not "bir bin"
        (1001, "bin bir"),
        (2000, "iki bin"),
        (2026, "iki bin yirmi altı"),
        (1_000_000, "bir milyon"),   # but "bir" IS said here
        (2_500_000, "iki milyon beş yüz bin"),
        (1_000_000_000, "bir milyar"),
    ],
)
def test_number_to_turkish(value: int, expected: str) -> None:
    assert number_to_turkish(value) == expected


def test_negative_numbers() -> None:
    assert number_to_turkish(-5) == "eksi beş"


def test_spell_numbers_in_a_sentence() -> None:
    assert spell_numbers("2026 yılında 3 dil") == "iki bin yirmi altı yılında üç dil"


def test_thousands_separator_is_a_full_stop_in_turkish() -> None:
    """Regression: '40.000' was being split into 40 and 000 -> 'kırk sıfır'.

    Turkish groups thousands with a full stop, so a bare \\d+ match corrupts
    every grouped number in the reference text - and the reference is what
    every model is being scored against.
    """
    assert spell_numbers("40.000") == "kırk bin"
    assert spell_numbers("1.234.567") == "bir milyon iki yüz otuz dört bin beş yüz altmış yedi"


def test_decimal_comma_is_read_as_virgul() -> None:
    assert spell_numbers("3,14") == "üç virgül on dört"


def test_grouped_and_plain_numbers_in_one_sentence() -> None:
    assert spell_numbers("40.000 kişi ve 5 dil") == "kırk bin kişi ve beş dil"


def test_full_stop_after_a_number_is_not_a_separator() -> None:
    """A sentence-ending period must not be read as digit grouping."""
    assert spell_numbers("Toplam 40. Bitti.") == "Toplam kırk. Bitti."


# ---------------------------------------------------------------------------
# the normaliser
# ---------------------------------------------------------------------------


@pytest.fixture
def norm() -> TurkishNormalizer:
    return TurkishNormalizer()


def test_punctuation_is_stripped(norm: TurkishNormalizer) -> None:
    assert norm("Merhaba, dünya!") == "merhaba dünya"


def test_apostrophe_suffixes_are_unified(norm: TurkishNormalizer) -> None:
    """Turkish attaches suffixes to proper nouns with an apostrophe.

    ASR systems disagree about emitting it, so both spellings must compare
    equal or the metric measures typography.
    """
    assert norm("Türkiye'nin") == norm("Türkiyenin")
    assert norm("İzmir'den") == "izmirden"
    assert norm("2026'da") == "iki bin yirmi altıda"


def test_curly_and_straight_apostrophes_agree(norm: TurkishNormalizer) -> None:
    assert norm("Türkiye’nin") == norm("Türkiye'nin")


def test_unicode_composition_is_unified(norm: TurkishNormalizer) -> None:
    """NFC vs NFD forms of ç/ğ/ö/ş/ü must not count as errors."""
    composed = "çğöşü"
    decomposed = "çğöşü"
    assert norm(composed) == norm(decomposed)


def test_whitespace_is_collapsed(norm: TurkishNormalizer) -> None:
    assert norm("  bir   iki \n üç  ") == "bir iki üç"


def test_numbers_are_spelled(norm: TurkishNormalizer) -> None:
    assert norm("2026 yılı") == "iki bin yirmi altı yılı"


def test_percent_and_currency_symbols_are_removed(norm: TurkishNormalizer) -> None:
    assert norm("%50 ve 100₺") == "elli ve yüz"


def test_dashes_become_spaces(norm: TurkishNormalizer) -> None:
    assert norm("yapay-zekâ") == "yapay zekâ"


def test_empty_input(norm: TurkishNormalizer) -> None:
    assert norm("") == ""


def test_realistic_asr_pair_normalises_to_equal(norm: TurkishNormalizer) -> None:
    """A reference and a plausible ASR output of the same utterance."""
    reference = "2026'da İstanbul'da yapay zekâ konuştu."
    hypothesis = "iki bin yirmi altıda İstanbulda yapay zekâ konuştu"
    assert norm(reference) == norm(hypothesis)


# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------


def test_options_can_disable_number_spelling() -> None:
    n = TurkishNormalizer(NormalizerOptions(spell_numbers=False))
    assert n("2026") == "2026"


def test_options_can_preserve_case() -> None:
    n = TurkishNormalizer(NormalizerOptions(lowercase=False))
    assert n("İstanbul") == "İstanbul"


def test_normalisation_does_not_merge_distinct_words(norm: TurkishNormalizer) -> None:
    """Sanity guard: normalisation must not be so aggressive it hides errors."""
    assert norm("ışık") != norm("işik")
    assert norm("geliyorum") != norm("gelmiyorum")
    # kar (snow) and kâr (profit) are different words; the circumflex must survive.
    assert norm("kar") != norm("kâr")
    # A dropped negation suffix is the kind of error that must never normalise away.
    assert norm("Anlıyorum.") != norm("Anlamıyorum.")
