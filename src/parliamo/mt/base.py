"""The translation backend contract.

Two very different kinds of system sit behind this interface, and the interface
exists because both are needed:

*Dedicated NMT* (NLLB) is a sequence-to-sequence model that translates and does
nothing else. Small, fast, predictable, and — decisively for this project — the
only pretrained model that has seen Friulian at all.

*Instruction-following LLMs* (MiLMMT, TranslateGemma, Gemma) translate by being
asked to. Better on the well-resourced pairs, but they can also refuse, explain
themselves, or answer a question they found inside the text. That failure mode
does not exist for NMT and has to be defended against here.

Language codes are the caller's problem to supply in a neutral form (``tr``,
``it``, ``fur``); each backend maps them to whatever it wants internally.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Translation:
    """One translated sentence, with what it cost."""

    text: str
    source: str
    source_lang: str
    target_lang: str
    compute_s: float = 0.0
    backend: str = ""
    #: Set when the output was rejected or repaired by the sanity checks below.
    repaired: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "compute_ms": round(self.compute_s * 1000, 1),
            "backend": self.backend,
            "repaired": self.repaired,
        }


class MTBackend(ABC):
    """A translator."""

    name: str = "abstract"

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        source_lang: str = "tr",
        target_lang: str = "it",
        split_sentences: bool = False,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.source_lang = source_lang
        self.target_lang = target_lang
        # Translate one sentence at a time rather than a whole segment.
        # Measured: NLLB-600M silently discards a trailing sentence after a
        # cataphoric construction, and splitting recovers it while running
        # *faster* (355 ms against 408 ms median). See parliamo.mt.sentences.
        self.split_sentences = split_sentences
        self.options = kwargs
        self._loaded = False
        self.load_vram_mb: float = 0.0
        self.load_seconds: float = 0.0

    # -- lifecycle --------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from ..asr.base import measure_vram

        t0 = time.perf_counter()
        with measure_vram(f"load {self.name}:{self.model}") as vram:
            self._load()
        self.load_seconds = time.perf_counter() - t0
        self.load_vram_mb = vram.get("vram_delta_mb", 0.0)
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        self._unload()
        self._loaded = False

    def warmup(self, text: str = "Merhaba, bu bir testtir.") -> float:
        """Translate once before it matters, so the first real call is not the slow one."""
        if not self._loaded:
            self.load()
        t0 = time.perf_counter()
        self.translate(text)
        return time.perf_counter() - t0

    def __enter__(self) -> MTBackend:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    # -- to implement -----------------------------------------------------

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _unload(self) -> None: ...

    @abstractmethod
    def _translate(self, text: str, source_lang: str, target_lang: str) -> str: ...

    def _translate_pieces(
        self, pieces: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate several sentences. Overridden by backends that can batch.

        The default loops, which is correct but slow. CTranslate2 handles a
        batch far better, so :class:`CTranslate2NLLBBackend` overrides this -
        which is what makes sentence splitting free rather than N times the
        cost.
        """
        return [self._translate(p, source_lang, target_lang) for p in pieces]

    # -- public API -------------------------------------------------------

    def translate(
        self, text: str, source_lang: str | None = None, target_lang: str | None = None
    ) -> Translation:
        if not self._loaded:
            self.load()
        src = source_lang or self.source_lang
        tgt = target_lang or self.target_lang

        stripped = text.strip()
        if not stripped:
            return Translation(
                text="", source=text, source_lang=src, target_lang=tgt,
                backend=f"{self.name}:{self.model}",
            )

        t0 = time.perf_counter()
        if self.split_sentences:
            from .sentences import split_sentences as _split

            pieces = _split(stripped)
            if len(pieces) > 1:
                raws = self._translate_pieces(pieces, src, tgt)
                # Sanitise per sentence, so a preamble on one does not survive
                # by being buried in the middle of a joined string.
                parts: list[str] = []
                reasons: list[str] = []
                for piece, raw_piece in zip(pieces, raws, strict=True):
                    cleaned_piece, why = sanitise(raw_piece, piece)
                    if cleaned_piece:
                        parts.append(cleaned_piece)
                    if why:
                        reasons.append(why)
                return Translation(
                    text=" ".join(parts),
                    source=stripped,
                    source_lang=src,
                    target_lang=tgt,
                    compute_s=time.perf_counter() - t0,
                    backend=f"{self.name}:{self.model}",
                    repaired="; ".join(dict.fromkeys(reasons)),
                )

        raw = self._translate(stripped, src, tgt)
        elapsed = time.perf_counter() - t0

        cleaned, repaired = sanitise(raw, stripped)
        if repaired:
            log.warning("translation repaired (%s): %r -> %r", repaired, raw[:80], cleaned[:80])

        return Translation(
            text=cleaned,
            source=stripped,
            source_lang=src,
            target_lang=tgt,
            compute_s=elapsed,
            backend=f"{self.name}:{self.model}",
            repaired=repaired,
        )

    def translate_many(
        self, texts: list[str], source_lang: str | None = None, target_lang: str | None = None
    ) -> list[Translation]:
        """Default is sequential; backends that batch should override."""
        return [self.translate(t, source_lang, target_lang) for t in texts]

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "device": self.device,
            "load_seconds": round(self.load_seconds, 2),
            "load_vram_mb": self.load_vram_mb,
            **{k: v for k, v in self.options.items() if isinstance(v, str | int | float | bool)},
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.model} {self.source_lang}->{self.target_lang}>"


# ---------------------------------------------------------------------------
# output sanity
# ---------------------------------------------------------------------------

# Openers an instruction-following model uses when it decides to talk to you
# instead of translating. Anchored at the start, so a sentence that merely
# contains one of these words is untouched.
#
# Matched as a regex rather than as literal prefixes, because the phrasing
# varies in ways a prefix list cannot keep up with: "Here is the translation:",
# "The translation is:", "Translation -", "Ecco la traduzione:". The pattern
# takes a stem, an optional linking verb, and any separator, so one rule covers
# all of them. Anchored at the start and applied repeatedly.
# A preamble has to *end* - with a colon or a dash, or a comma after an
# interjection. The first version let the separator be optional, and then
# "La traduzione non è davvero buona" - the presenter, live, saying the
# translation was not good - lost its first two words and was spoken as
# "non è davvero buona e questo". A sentence that merely begins with the word
# "translation" is a sentence, and the room heard it mangled while this logged
# a successful repair. The rule was written for LLM output; the translator in
# use (NLLB, ADR 0004) never produces a preamble at all.
_PREAMBLE_INTERJECTIONS = (
    r"sure",
    r"certainly",
    r"of course",
)
_PREAMBLE_LABELS = (
    r"here(?:'s| is)(?: the| your)?(?: translation| translated text)?",
    r"ecco(?: la)?(?: traduzione)?",
    r"(?:the )?translation",
    r"(?:la )?traduzione",
    r"translated text",
    r"output",
)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:" + "|".join(_PREAMBLE_INTERJECTIONS) + r")\b\s*[:,\-–—]+"
    r"|"
    r"(?:" + "|".join(_PREAMBLE_LABELS) + r")\b\s*(?:is|are|è|sono)?\s*[:\-–—]+"
    r")\s*",
    re.IGNORECASE,
)

# Refusals are handled separately: there is nothing behind them to salvage.
_REFUSAL_RE = re.compile(
    r"^\s*(?:i cannot|i can'?t|i'?m unable|i am unable|as an ai|"
    r"mi dispiace|non posso|sorry\b)",
    re.IGNORECASE,
)

#: Beyond this multiple of the source length, the model has stopped translating
#: and started generating. Generous, because Turkish is compact and Italian is
#: not: a faithful tr->it translation routinely runs 1.5x longer in characters.
MAX_LENGTH_RATIO = 4.0

# NLLB-200 was trained partly on OpenSubtitles, where a leading "- " marks a
# change of speaker. It reproduces the convention on short utterances, and
# measured on sixteen ordinary short Turkish phrases it did so on five:
#
#   Merhaba.          -> - Ciao. - Ciao.
#   Teşekkür ederim.  -> - Grazie. - Grazie.
#   Günaydın.         -> - Buongiorno.
#
# This is not an LLM misbehaving, which is what the rest of this module was
# written for. It is a dedicated NMT model faithfully reproducing its training
# data, and on stage the dash and the duplicate are both read aloud in the
# presenter's cloned voice.
_SUBTITLE_DASH_RE = re.compile(r"^\s*[-–—]\s+")

#: A short source cannot honestly produce a long repeated output. Above this
#: length, repetition is more likely to be the speaker's own.
_SHORT_SOURCE_CHARS = 40


def sanitise(raw: str, source: str) -> tuple[str, str]:
    """Clean an LLM's translation output. Returns ``(text, reason_if_repaired)``.

    Dedicated NMT models never need this. Instruction-following models do, and
    on stage a preamble like "Sure, here is the translation:" would be read
    aloud by the synthesiser in the presenter's own cloned voice.
    """
    text = (raw or "").strip()
    if not text:
        return "", "empty"

    reasons: list[str] = []

    # Order matters. Commentary goes first: a model that wraps its translation
    # in quotes and then explains itself produces `"...text..."\n\nNote: ...`,
    # where the closing quote is mid-string and the quote strip would not fire.
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
        reasons.append("truncated-at-paragraph")

    # A refusal has nothing behind it worth keeping. Better silence than an
    # apology read aloud in the presenter's own cloned voice.
    if _REFUSAL_RE.match(text):
        return "", "refusal"

    # Peel quotes and preambles until nothing more comes off. One pass is not
    # enough: "Here is the translation:" sheds "here is" and leaves "the
    # translation:", which is a preamble in its own right.
    for _ in range(4):
        before = text

        if len(text) >= 2 and text[0] in "\"'“«" and text[-1] in "\"'”»":
            text = text[1:-1].strip()
            if "unquoted" not in reasons:
                reasons.append("unquoted")

        match = _PREAMBLE_RE.match(text)
        if match and match.end() > 0:
            remainder = text[match.end():].lstrip(" \"'“«")
            # Accept the strip only if something substantial survives, so a
            # bare "Translation:" with nothing after it is not mistaken for
            # a successful repair.
            if len(remainder) < max(8, 0.3 * len(source)):
                return "", "refusal"
            text = remainder
            if "preamble-stripped" not in reasons:
                reasons.append("preamble-stripped")

        if text == before:
            break

    # Subtitle conventions from the training data, stripped repeatedly because
    # a doubled output carries one on each half.
    for _ in range(3):
        stripped = _SUBTITLE_DASH_RE.sub("", text)
        if stripped == text:
            break
        text = stripped
        if "subtitle-dash" not in reasons:
            reasons.append("subtitle-dash")

    repeated = _collapse_repeated_output(text, source)
    if repeated != text:
        text = repeated
        reasons.append("de-duplicated")

    if len(text) > MAX_LENGTH_RATIO * max(len(source), 1):
        text = text[: int(MAX_LENGTH_RATIO * len(source))].rsplit(" ", 1)[0]
        reasons.append("length-capped")

    return text.strip(), "; ".join(reasons)


def _collapse_repeated_output(text: str, source: str) -> str:
    """Keep one copy when a short source produced the same sentence twice over.

    Measured: "Merhaba." came back as "- Ciao. - Ciao." and "Hayır." as
    "No, no, no, no, no, no." The model pads a short input out to a more
    typical sentence length by repeating itself.

    Deliberately narrow. Italian genuinely repeats for emphasis - *va bene, va
    bene* - so this only fires when the **source is short and does not itself
    repeat**, which is the case where the extra copies cannot have come from
    anywhere but the model.
    """
    if len(source) > _SHORT_SOURCE_CHARS:
        return text

    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|,\s*", text) if u.strip()]
    if len(units) < 2:
        return text

    # Compare without punctuation or case: "Ciao." and "ciao" are one unit.
    def key(unit: str) -> str:
        return re.sub(r"[^\w]+", "", unit).casefold()

    keys = [key(u) for u in units]
    if len(set(keys)) != 1 or not keys[0]:
        return text

    # The source repeating itself is the speaker's choice, not the model's, and
    # a translation that dropped it would be losing content rather than
    # cleaning it up. "Yeter. Yeter." should stay "Basta. Basta."
    source_units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|,\s*", source) if u.strip()]
    source_repeats = len(source_units) > 1 and len({key(u) for u in source_units}) < len(
        source_units
    )
    if source_repeats:
        return text

    # Splitting at a comma leaves the kept unit bare - "No, no, no." yields
    # "No". Give it back the sentence's own ending, so the synthesiser gets a
    # finished sentence rather than a fragment to guess the prosody of.
    kept = units[0]
    ending = text.rstrip()[-1:]
    if ending in ".!?" and kept[-1:] not in ".!?":
        kept += ending
    return kept
