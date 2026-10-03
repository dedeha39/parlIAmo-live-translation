"""Splitting Turkish text into sentences before translating it.

Why this exists
---------------
NLLB-600M silently drops trailing content. Measured, on the presenter's own
recorded speech::

    tr     Aklımdan geçen ilk soru şuydu. Acaba gerçekten çalışacak mı?
    it     La prima domanda che mi è venuta in mente è stata:

The second sentence - *I wonder whether it will really work?* - is simply gone.
The output is fluent, grammatical Italian, carries no error flag, and is not
short enough to trip the length check in :func:`parliamo.mt.base.sanitise`. On
stage the audience would never learn that anything was missing.

The trigger is the cataphoric ``şuydu`` / ``şudur`` construction ("was this:"),
which points *forward* at the sentence that follows. NLLB renders it as
``è stata:`` and then stops, dropping what the colon promised. That construction
is ordinary presentation Turkish, so this is not an exotic input.

Translating each sentence separately fixes it, and measured on the failing
cases it is also **faster** - 355 ms against 408 ms median - because a batch of
short decodes beats one long autoregressive decode.

Why splitting Turkish needs care
--------------------------------
A naive split on ``[.!?]\s`` breaks Turkish in two common places:

* **Ordinals.** Turkish writes them with a full stop: ``3. sınıf``, ``1. Dünya
  Savaşı``. Splitting there fragments the sentence.
* **Abbreviations.** ``Dr.``, ``vb.``, ``Prof.``, ``M.Ö.`` all end in a period
  mid-sentence.

Thousands separators (``40.000``) are safe without special handling, because
they carry no space after the stop - but the ordinal case does, so it has to be
excluded explicitly.
"""

from __future__ import annotations

import re

#: Turkish abbreviations that end in a full stop mid-sentence. Lowercased for
#: comparison with :func:`parliamo.eval.normalize.turkish_lower`, so the dotted
#: and dotless i behave.
ABBREVIATIONS = frozenset(
    {
        "dr", "prof", "doç", "av", "sn", "sy", "bkz", "örn", "vb", "vs", "yy",
        "no", "tel", "sf", "s", "c", "bl", "yrd", "müh", "alb", "gen", "hz",
        "st", "mah", "cad", "sok", "apt", "tsk", "tbmm", "vd", "çev", "haz",
        "m", "mö", "ms", "ör",
    }
)

# Split after sentence-final punctuation followed by whitespace. Deliberately
# NOT on ':' or ';' - measured, splitting at a colon made the cataphoric case
# *worse* (the two halves lose the link the colon carries), while the same
# sentence written with a full stop is the case that needs fixing.
_BOUNDARY = re.compile(r"(?<=[.!?…])[\"'”»\)]*\s+")

# The token immediately before the boundary. Used to veto ordinals and
# abbreviations.
_PRECEDING = re.compile(r"([^\s]+)[.!?…][\"'”»\)]*\s*$")

_DIGITS_ONLY = re.compile(r"^\d+\.$")

# Reported speech: `Adam "bu doğru." dedi.` The stop belongs to the quotation,
# and the sentence carries on outside it. The tell is that the word after the
# closing quote is lowercase - a new sentence would not be. Applied *only*
# when a closing quote is present, because recogniser output is not reliably
# capitalised and the same rule elsewhere would refuse real boundaries.
_CLOSING_QUOTE = re.compile(r"[\"'”»\)]")


def _is_real_boundary(left: str) -> bool:
    """Is the stop at the end of *left* a sentence end rather than punctuation?"""
    match = _PRECEDING.search(left)
    if match is None:
        return True
    token = match.group(1)

    # "3." / "1945." - a Turkish ordinal or a bare year, not a sentence end.
    if _DIGITS_ONLY.match(token + "."):
        return False

    word = token.rstrip(".!?…").strip("\"'“«(").lower()
    # Turkish i-mapping matters here: "Vb." lowercases correctly either way,
    # but a dotless-I abbreviation would not without it.
    word = word.replace("I", "ı").replace("İ", "i").lower()
    if word in ABBREVIATIONS:
        return False
    # A single letter followed by a stop is an initial ("A. Yılmaz"), not a
    # sentence end.
    return not len(word) <= 1


def split_sentences(text: str, max_sentences: int = 12) -> list[str]:
    """Split *text* into sentences. Returns ``[text]`` when there is nothing to do.

    *max_sentences* is a guard, not a tuning knob: a segmenter fault or a
    recogniser loop could hand this a wall of text, and translating 200 tiny
    fragments would be slower than translating one long one. Past the limit the
    text is returned whole, which is the behaviour we had before.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []

    pieces: list[str] = []
    cursor = 0
    for match in _BOUNDARY.finditer(stripped):
        left = stripped[cursor : match.start()]
        if not _is_real_boundary(stripped[cursor : match.end()]):
            continue
        if _CLOSING_QUOTE.search(match.group()):
            following = stripped[match.end() : match.end() + 1]
            if following and following.islower():
                continue
        piece = left.strip()
        if piece:
            pieces.append(piece)
        cursor = match.end()

    tail = stripped[cursor:].strip()
    if tail:
        pieces.append(tail)

    if not pieces:
        return [stripped]
    if len(pieces) > max_sentences:
        return [stripped]
    return pieces
