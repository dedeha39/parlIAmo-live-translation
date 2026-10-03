"""Text normalisation for word-error-rate comparison.

Normalisation decides how much of a measured error rate is real. Compare raw
ASR output against a reference and you are largely measuring punctuation and
capitalisation conventions; normalise too aggressively and you hide mistakes
that would be audible on stage.

The Turkish dotted-i trap
-------------------------
Turkish has four i-letters that pair up differently from every other Latin
alphabet::

    dotted    i (U+0069)  <->  İ (U+0130)
    dotless   ı (U+0131)  <->  I (U+0049)

Python's ``str.lower()`` follows the default Unicode mapping, so it turns
``I`` into ``i`` — merging the dotless capital with the dotted lowercase. In
Turkish those are different letters in different words: ``ışık`` (light) and
``işik`` are not the same, ``Irak`` (Iraq) lowercases to ``ırak`` (far), not
``irak``. Using ``str.lower()`` here would silently invent errors in the
reference and hide them in the hypothesis, in opposite directions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Applied before any case folding, so the mapping is unambiguous.
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})

# Punctuation is stripped, but the apostrophe is handled separately: Turkish
# writes suffixes on proper nouns with one (Türkiye'nin, 2026'da), and ASR
# systems disagree about whether to emit it. Dropping it entirely makes the
# comparison fair in both directions.
_APOSTROPHES = "'‘’ʼ´`"
_DASHES = "‐‑‒–—―"

_WS_RE = re.compile(r"\s+")

# Turkish groups thousands with a full stop and marks decimals with a comma:
# 40.000 is forty thousand, 3,14 is pi. Matching a bare \d+ splits "40.000"
# into 40 and 000 and spells it "kırk sıfır" instead of "kırk bin", which
# corrupts the reference text the whole measurement is compared against.
# Longest form first so the alternation does not match a prefix.
_NUM_RE = re.compile(
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"   # 40.000 / 1.234.567,5
    r"|\d+,\d+"                        # 3,14
    r"|\d+"                            # 2026
)


def turkish_lower(text: str) -> str:
    """Lowercase *text* with Turkish i-mapping applied first."""
    return text.translate(_TR_LOWER).lower()


def turkish_upper(text: str) -> str:
    """Uppercase *text* with Turkish i-mapping applied first."""
    return text.translate(_TR_UPPER).upper()


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

_ONES = ("", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")
_SCALES = ((10**9, "milyar"), (10**6, "milyon"), (10**3, "bin"), (10**2, "yüz"))


def _under_thousand(n: int) -> list[str]:
    parts: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        # "yüz", not "bir yüz" - Turkish drops the leading one.
        parts.extend([_ONES[hundreds]] if hundreds > 1 else [])
        parts.append("yüz")
    tens, ones = divmod(rest, 10)
    if tens:
        parts.append(_TENS[tens])
    if ones:
        parts.append(_ONES[ones])
    return parts


def number_to_turkish(n: int) -> str:
    """Spell a non-negative integer in Turkish.

    Handles the two places Turkish drops a leading "bir": ``100`` is ``yüz``
    and ``1000`` is ``bin``, but ``1_000_000`` is ``bir milyon``.
    """
    if n < 0:
        return "eksi " + number_to_turkish(-n)
    if n == 0:
        return "sıfır"

    parts: list[str] = []
    for value, name in _SCALES[:3]:  # milyar, milyon, bin
        count, n = divmod(n, value)
        if count:
            if not (count == 1 and name == "bin"):
                parts.extend(_under_thousand(count))
            parts.append(name)
    if n:
        parts.extend(_under_thousand(n))
    return " ".join(p for p in parts if p)


def _spell_match(token: str) -> str:
    """Spell one numeric token, honouring Turkish digit grouping."""
    integer_part, _, fraction = token.partition(",")
    integer_part = integer_part.replace(".", "")  # thousands separator
    try:
        spelled = number_to_turkish(int(integer_part))
    except ValueError:  # pragma: no cover - the regex should prevent this
        return token
    if fraction:
        # "3,14" is read "üç virgül on dört", not "üç virgül bir dört".
        spelled += " virgül " + number_to_turkish(int(fraction))
    return spelled


def spell_numbers(text: str) -> str:
    """Replace numeric tokens with their Turkish spelling.

    ASR systems disagree about whether to emit "2026" or "iki bin yirmi altı";
    both are correct transcriptions of the same speech, so counting one as an
    error would measure formatting rather than recognition.
    """
    return _NUM_RE.sub(lambda m: _spell_match(m.group()), text)


# ---------------------------------------------------------------------------
# the normaliser
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizerOptions:
    lowercase: bool = True
    strip_punctuation: bool = True
    spell_numbers: bool = True
    collapse_whitespace: bool = True


class TurkishNormalizer:
    """Normalise Turkish text for fair WER comparison."""

    def __init__(self, options: NormalizerOptions | None = None) -> None:
        self.options = options or NormalizerOptions()

    def __call__(self, text: str) -> str:
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        if not text:
            return ""

        opts = self.options
        # NFC first so that composed and decomposed forms of ç, ğ, ö, ş, ü
        # compare equal - ASR output and reference corpora differ on this.
        out = unicodedata.normalize("NFC", text)

        for ch in _APOSTROPHES:
            out = out.replace(ch, "")
        for ch in _DASHES:
            out = out.replace(ch, " ")

        if opts.lowercase:
            out = turkish_lower(out)
        if opts.spell_numbers:
            out = spell_numbers(out)
        if opts.strip_punctuation:
            out = "".join(
                " " if unicodedata.category(ch).startswith("P") or ch in "%€$₺" else ch
                for ch in out
            )
        if opts.collapse_whitespace:
            out = _WS_RE.sub(" ", out).strip()
        return out


DEFAULT_NORMALIZERS: dict[str, TurkishNormalizer] = {}


def get_normalizer(language: str) -> TurkishNormalizer:
    """Return a normaliser for *language*.

    Only Turkish has bespoke handling so far; other languages fall back to the
    same pipeline without the i-mapping, which is harmless for them because the
    mapping only touches I and İ.
    """
    key = language.lower()
    if key not in DEFAULT_NORMALIZERS:
        DEFAULT_NORMALIZERS[key] = TurkishNormalizer()
    return DEFAULT_NORMALIZERS[key]
