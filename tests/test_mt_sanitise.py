"""Defending against an LLM that answers instead of translating.

Dedicated NMT models emit a translation or nothing. Instruction-following
models can also apologise, explain their choices, refuse, or wrap the output in
quotes. On stage every one of those would be spoken aloud by the synthesiser in
the presenter's own cloned voice, in front of the audience.

These cases are all real shapes of LLM translation output.
"""

from __future__ import annotations

import pytest

from parliamo.mt.base import MAX_LENGTH_RATIO, MTBackend, sanitise

SOURCE = "Rıza olmadan birinin sesini kopyalamak etik bir meseledir."
GOOD = "Copiare la voce di qualcuno senza consenso è una questione etica."


def test_clean_output_is_left_alone() -> None:
    text, reason = sanitise(GOOD, SOURCE)
    assert text == GOOD
    assert reason == ""


# ---------------------------------------------------------------------------
# preambles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        "Sure, ",
        "Certainly, ",
        "Here is the translation: ",
        "Here's the translation: ",
        "Translation: ",
        "Traduzione: ",
        "Ecco la traduzione: ",
        "The translation is: ",
    ],
)
def test_preamble_is_stripped(prefix: str) -> None:
    text, reason = sanitise(prefix + GOOD, SOURCE)
    assert text == GOOD
    assert "preamble-stripped" in reason


def test_preamble_matching_is_anchored_to_the_start() -> None:
    """A sentence that merely contains such a word must survive intact."""
    sentence = "Il documento spiega che qui è dove inizia la traduzione: il primo capitolo."
    text, reason = sanitise(sentence, SOURCE)
    assert text == sentence
    assert "preamble" not in reason


def test_refusal_yields_nothing_rather_than_a_fragment() -> None:
    """Better silence than reading an apology aloud in the presenter's voice."""
    text, reason = sanitise("I cannot translate that.", SOURCE)
    assert text == ""
    assert reason == "refusal"


def test_refusal_in_italian() -> None:
    text, reason = sanitise("Mi dispiace, ma non posso.", SOURCE)
    assert text == ""
    assert reason == "refusal"


def test_preamble_followed_by_substance_is_kept() -> None:
    """The strip only applies if a real translation survives it."""
    text, reason = sanitise("Sure, " + GOOD, SOURCE)
    assert len(text) > 20
    assert "refusal" not in reason


# ---------------------------------------------------------------------------
# decoration and commentary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quotes", ['"{}"', "'{}'", "“{}”", "«{}»"])
def test_wrapping_quotes_are_removed(quotes: str) -> None:
    text, reason = sanitise(quotes.format(GOOD), SOURCE)
    assert text == GOOD
    assert "unquoted" in reason


def test_internal_quotes_are_preserved() -> None:
    quoted = 'Ha detto "sì" e poi se ne è andato.'
    text, _ = sanitise(quoted, SOURCE)
    assert text == quoted


def test_commentary_after_a_blank_line_is_dropped() -> None:
    raw = f"{GOOD}\n\nNote: I chose 'questione' rather than 'problema' because..."
    text, reason = sanitise(raw, SOURCE)
    assert text == GOOD
    assert "truncated-at-paragraph" in reason


def test_single_newline_is_not_a_paragraph_break() -> None:
    raw = "Prima riga\nseconda riga"
    text, _ = sanitise(raw, SOURCE)
    assert text == raw


# ---------------------------------------------------------------------------
# runaway generation
# ---------------------------------------------------------------------------


def test_runaway_output_is_capped() -> None:
    text, reason = sanitise(GOOD * 20, SOURCE)
    assert len(text) <= MAX_LENGTH_RATIO * len(SOURCE)
    assert "length-capped" in reason


def test_legitimately_longer_translation_is_not_capped() -> None:
    """Turkish is compact and Italian is not - 1.5x longer is normal, not a fault."""
    source = "Yarın geliyorum."
    target = "Domani verrò a trovarvi presso il vostro ufficio."
    text, reason = sanitise(target, source)
    assert text == target
    assert "length-capped" not in reason


def test_cap_does_not_cut_mid_word() -> None:
    text, _ = sanitise("parola " * 200, SOURCE)
    assert not text.endswith("paro")


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_empty_output_is_reported() -> None:
    assert sanitise("", SOURCE) == ("", "empty")
    assert sanitise("   \n ", SOURCE) == ("", "empty")


def test_multiple_problems_are_all_reported() -> None:
    """Quoted, prefaced, and followed by commentary - all three at once.

    The commentary has to be removed before the quote strip can see a closing
    quote at the end of the string; getting that order wrong was a real bug.
    """
    raw = f'"Sure, {GOOD}"\n\nLet me know if you need anything else.'
    text, reason = sanitise(raw, SOURCE)
    assert "truncated-at-paragraph" in reason
    assert "unquoted" in reason
    assert "preamble-stripped" in reason
    assert text == GOOD
    assert "Let me know" not in text


def test_nested_preambles_are_peeled() -> None:
    """'here is' strips to 'the translation: ...', which is itself a preamble."""
    text, reason = sanitise(f"Here is the translation: {GOOD}", SOURCE)
    assert text == GOOD
    assert "translation:" not in text.lower()
    assert "preamble-stripped" in reason


def test_peeling_terminates() -> None:
    """A pathological input must not loop forever."""
    text, _ = sanitise('"' * 12 + GOOD + '"' * 12, SOURCE)
    assert GOOD in text


# ---------------------------------------------------------------------------
# sentence splitting inside the backend
# ---------------------------------------------------------------------------


class _Recorder(MTBackend):
    """Records what it was asked to translate, so the split path is observable."""

    name = "recorder"

    def __init__(self, **kwargs):
        super().__init__(model="fake", **kwargs)
        self.calls: list[str] = []
        self.batches: list[list[str]] = []
        self._loaded = True

    def _load(self) -> None: ...
    def _unload(self) -> None: ...

    def _translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.calls.append(text)
        return f"<{text}>"

    def _translate_pieces(self, pieces, source_lang, target_lang):
        self.batches.append(list(pieces))
        return [f"<{p}>" for p in pieces]


def test_splitting_off_sends_the_whole_segment() -> None:
    backend = _Recorder(split_sentences=False)
    backend.translate("Birinci cümle budur. İkinci cümle şudur.")
    assert backend.calls == ["Birinci cümle budur. İkinci cümle şudur."]
    assert backend.batches == []


def test_splitting_on_sends_one_batch_of_sentences() -> None:
    """One batch, not one call per sentence - that is what keeps it free."""
    backend = _Recorder(split_sentences=True)
    result = backend.translate("Birinci cümle budur. İkinci cümle şudur.")
    assert backend.batches == [["Birinci cümle budur.", "İkinci cümle şudur."]]
    assert result.text == "<Birinci cümle budur.> <İkinci cümle şudur.>"


def test_single_sentence_takes_the_ordinary_path() -> None:
    backend = _Recorder(split_sentences=True)
    backend.translate("Tek bir cümle.")
    assert backend.batches == []
    assert backend.calls == ["Tek bir cümle."]


def test_split_output_is_sanitised_per_sentence() -> None:
    """A preamble on the second sentence must not survive by being buried."""

    class _Chatty(_Recorder):
        def _translate_pieces(self, pieces, source_lang, target_lang):
            return [f"<{pieces[0]}>", "Sure, here is the translation: seconda frase"]

    backend = _Chatty(split_sentences=True)
    result = backend.translate("Birinci cümle budur. İkinci cümle şudur.")
    assert "Sure" not in result.text
    assert "seconda frase" in result.text
    assert "preamble-stripped" in result.repaired


def test_a_refused_sentence_does_not_blank_the_whole_segment() -> None:
    """Losing one sentence beats losing the sentence that was fine."""

    class _Refuser(_Recorder):
        def _translate_pieces(self, pieces, source_lang, target_lang):
            return [f"<{pieces[0]}>", "I cannot translate that."]

    backend = _Refuser(split_sentences=True)
    result = backend.translate("Birinci cümle budur. İkinci cümle şudur.")
    assert result.text == "<Birinci cümle budur.>"
    assert "refusal" in result.repaired


def test_ordinals_do_not_trigger_the_split_path() -> None:
    backend = _Recorder(split_sentences=True)
    backend.translate("1. Dünya Savaşı 1914 yılında başladı.")
    assert backend.batches == []


# ---------------------------------------------------------------------------
# NMT artefacts: subtitle conventions and padding
# ---------------------------------------------------------------------------
#
# Everything above defends against an LLM answering instead of translating.
# These defend against something different: NLLB-200 faithfully reproducing
# OpenSubtitles conventions from its own training data. Measured on sixteen
# ordinary short Turkish phrases, five came back with a leading "- " and
# several were padded out by repetition.


def test_leading_subtitle_dash_is_removed() -> None:
    text, why = sanitise("- Buongiorno.", "Günaydın.")
    assert text == "Buongiorno."
    assert "subtitle-dash" in why


def test_dash_and_duplicate_together() -> None:
    """The measured case: 'Merhaba.' came back as '- Ciao. - Ciao.'"""
    text, why = sanitise("- Ciao. - Ciao.", "Merhaba.")
    assert text == "Ciao."
    assert "subtitle-dash" in why


def test_thank_you_is_not_doubled() -> None:
    text, _ = sanitise("- Grazie. - Grazie.", "Teşekkür ederim.")
    assert text == "Grazie."


def test_padding_repetition_is_collapsed() -> None:
    """'Hayır.' came back as six negations."""
    text, why = sanitise("No, no, no, no, no, no.", "Hayır.")
    assert text == "No."
    assert "de-duplicated" in why


def test_a_hyphen_inside_the_sentence_is_kept() -> None:
    """Only a *leading* dash is the subtitle convention."""
    text, why = sanitise("Il sistema - tutto locale - funziona.", "Sistem yerel çalışıyor.")
    assert text == "Il sistema - tutto locale - funziona."
    assert "subtitle-dash" not in why


def test_a_long_source_is_not_de_duplicated() -> None:
    """Repetition in a real sentence is more likely the speaker's own."""
    source = "Bu çok önemli, gerçekten çok önemli, bunu unutmayın lütfen."
    raw = "È molto importante, è molto importante, non dimenticatelo."
    text, why = sanitise(raw, source)
    assert text == raw
    assert "de-duplicated" not in why


def test_deliberate_repetition_in_the_source_survives() -> None:
    """If the speaker repeated themselves, the translation should too."""
    text, why = sanitise("Basta. Basta.", "Yeter. Yeter.")
    assert text == "Basta. Basta."
    assert "de-duplicated" not in why


def test_two_different_sentences_are_not_collapsed() -> None:
    text, why = sanitise("Ciao. Come stai?", "Merhaba. Nasılsın?")
    assert text == "Ciao. Come stai?"
    assert "de-duplicated" not in why


def test_ordinary_short_translation_is_untouched() -> None:
    text, why = sanitise("Iniziamo.", "Başlayalım.")
    assert text == "Iniziamo."
    assert why == ""


def test_emphasis_pair_from_a_short_source_is_collapsed() -> None:
    """'Tamam.' -> 'Va bene, va bene.' is padding, not emphasis.

    A single-word source cannot have asked for the doubling.
    """
    text, _ = sanitise("Va bene, va bene.", "Tamam.")
    assert text == "Va bene."


def test_a_collapsed_unit_keeps_the_sentence_ending() -> None:
    """The synthesiser should get a finished sentence, not a fragment."""
    assert sanitise("No, no, no.", "Hayır.")[0].endswith(".")
    assert sanitise("Davvero, davvero?", "Gerçekten mi?")[0].endswith("?")


# ---------------------------------------------------------------------------
# a sentence that begins with the word "translation" is a sentence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentence", [
    "La traduzione non è davvero buona e non so come risolverla.",
    "La traduzione in realtà è bene.",
    "The translation is not good and I do not know how to fix it.",
    "Translation quality matters more than speed here.",
    "Ecco la traduzione che abbiamo fatto ieri.",
])
def test_a_sentence_about_translation_is_left_alone(sentence: str) -> None:
    """Seen live, from the presenter: 'La traduzione non è davvero buona' lost
    its first two words and was spoken as 'non è davvero buona e questo'.
    A preamble ends with a colon or a dash; a sentence does not.
    """
    from parliamo.mt.base import sanitise

    text, reason = sanitise(sentence, source="çeviri aslında iyi değil")
    assert text == sentence, reason
    assert "preamble" not in reason


@pytest.mark.parametrize("raw, expected", [
    ("Ecco la traduzione: Ciao a tutti.", "Ciao a tutti."),
    ("Translation: Ciao a tutti.", "Ciao a tutti."),
    ("Here is the translation - Ciao a tutti.", "Ciao a tutti."),
    ("Sure, Ciao a tutti.", "Ciao a tutti."),
    ("La traduzione è: Ciao a tutti.", "Ciao a tutti."),
])
def test_a_real_preamble_is_still_stripped(raw: str, expected: str) -> None:
    from parliamo.mt.base import sanitise

    text, reason = sanitise(raw, source="herkese merhaba")
    assert text == expected
    assert "preamble-stripped" in reason
