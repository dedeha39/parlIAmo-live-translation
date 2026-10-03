"""Digits spelled out as words, for synthesisers that read them badly.

The translator writes numbers as digits - "una registrazione di 15 secondi",
"25 milioni di dollari". Kokoro and Piper expand them through their phonemiser;
OmniVoice does not (its own expansion needs an optional text-normalisation
package), and read "15 secondi" back as "chi me è secondo". Spelled out before
it is spoken, the number is said.
"""

from __future__ import annotations

import re

#: How each language groups thousands in the translator's output: "600.000"
#: in Italian, German, Spanish and Turkish; "600,000" in English; "600 000"
#: in French.
_GROUPING = {"en": ",", "fr": " "}
#: Languages without a num2words table spoken as the one whose voice they use.
_SPOKEN_AS = {"fur": "it"}


def spell_numbers(text: str, language: str | None) -> str:
    """Replace whole numbers in *text* with words in *language*.

    Unchanged when num2words is missing or has no table for the language, and
    for anything that is not a plain whole number (a decimal's two halves are
    each spelled, which is how it is read aloud anyway).
    """
    if not text or not language or not any(c.isdigit() for c in text):
        return text
    lang = _SPOKEN_AS.get(language, language)
    try:
        from num2words import num2words
    except ImportError:  # pragma: no cover - present wherever OmniVoice is
        return text
    sep = re.escape(_GROUPING.get(lang, "."))
    pattern = re.compile(rf"(?<![\w.,])\d{{1,3}}(?:{sep}\d{{3}})+(?![\d])|\d+")

    def one(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        try:
            return num2words(int(digits), lang=lang)
        except (NotImplementedError, ValueError, OverflowError):
            return match.group()

    return pattern.sub(one, text)
