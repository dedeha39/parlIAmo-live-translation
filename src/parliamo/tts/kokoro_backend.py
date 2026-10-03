"""Kokoro-82M: fast generic-voice synthesis.

Kokoro carries no speaker identity of its own worth preserving - it ships fixed
voicepacks and cannot clone. That is exactly why it is here. Generating speech
and carrying a speaker's identity are separate jobs, and forcing one model to do
both is what cost this project weeks (see docs/adr/0006). Kokoro does the first
job about fifty times faster than a model that also tries to do the second, and
voice conversion does the second afterwards.

Measured on the RTX 4070 Laptop, Italian, in the main environment:

    3 chars   ->  0.97 s audio, 0.071 s generation   RTF 0.073
   65 chars   ->  4.42 s audio, 0.094 s generation   RTF 0.021
  154 chars   ->  9.72 s audio, 0.195 s generation   RTF 0.020

587 MiB of VRAM, and no silence padding - "Sì." yields 0.97 s where Chatterbox
yielded 3.00 s.

Turkish
-------
Kokoro has no Turkish voicepack. That is not a gap for this project: the stage
language is Turkish and the *output* language is Italian, so nothing needs to be
spoken in Turkish. If a Turkish output is ever required, this backend will
refuse rather than silently mispronounce it with another language's frontend.

Friulian
--------
Synthesised through the Italian frontend, as everywhere else in this project.
Correct-ish prosody, Italian phonetics, and a documented limitation that is
disclosed on stage rather than hidden.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import Speech, TTSBackend, VoiceProfile

log = logging.getLogger(__name__)

#: Kokoro selects a language with a single-letter code, not an ISO tag.
LANG_CODES = {
    "en": "a",   # American English
    "it": "i",
    "es": "e",
    "fr": "f",
    "pt": "p",
    "hi": "h",
    "ja": "j",
    "zh": "z",
}

#: No Friulian voice exists anywhere; speak it with the Italian frontend.
FALLBACK_LANGUAGES = {"fur": "it"}

#: Default voicepack per language. Kokoro names them <lang><gender>_<name>.
DEFAULT_VOICES = {
    "it": "if_sara",
    "en": "af_heart",
    "es": "ef_dora",
    "fr": "ff_siwis",
}


class KokoroBackend(TTSBackend):
    name = "kokoro"
    sample_rate = 24000

    def __init__(
        self,
        model: str = "kokoro-82M",
        device: str = "cuda",
        language: str = "it",
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, device=device, language=language, speed=speed, **kwargs)
        self.voice = voice or DEFAULT_VOICES.get(language, "if_sara")
        self.speed = speed
        self._pipelines: dict[str, Any] = {}

    # -- language ---------------------------------------------------------

    def resolve_language(self, language: str) -> tuple[str, str]:
        """Map a project language to a Kokoro code. Returns ``(code, note)``."""
        if language in LANG_CODES:
            return LANG_CODES[language], ""
        if language in FALLBACK_LANGUAGES:
            substitute = FALLBACK_LANGUAGES[language]
            return LANG_CODES[substitute], (
                f"no {language} voicepack exists; synthesised with the "
                f"{substitute} frontend, so the phonetics are wrong"
            )
        raise KeyError(
            f"Kokoro has no voice for {language!r}. Supported: "
            f"{', '.join(sorted(LANG_CODES) + sorted(FALLBACK_LANGUAGES))}. "
            "Refusing rather than mispronouncing it with another language."
        )

    # -- lifecycle --------------------------------------------------------

    def _pipeline(self, language: str) -> Any:
        """One pipeline per language, built on first use and kept."""
        code, note = self.resolve_language(language)
        if code not in self._pipelines:
            from kokoro import KPipeline

            if note:
                log.warning("%s", note)
            log.info("building Kokoro pipeline for %r (code %r)", language, code)
            self._pipelines[code] = KPipeline(lang_code=code, device=self.device)
        return self._pipelines[code]

    def _load(self) -> None:
        # Building the default pipeline here means load() measures the real
        # cost, rather than deferring it to the first utterance on stage.
        self._pipeline(self.language)

    def _unload(self) -> None:
        self._pipelines.clear()
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- synthesis --------------------------------------------------------

    def _synthesise(
        self, text: str, language: str, voice: VoiceProfile | None
    ) -> np.ndarray:
        if voice is not None:
            # Better to refuse than to hand back audio in the wrong voice and
            # let a caller believe cloning happened.
            raise NotImplementedError(
                "Kokoro cannot clone a voice. Generate here and convert "
                "afterwards - see parliamo.tts.conversion."
            )

        pipeline = self._pipeline(language)
        # The configured voice wins; the language default is the fallback.
        # This was the other way round - the language default first, the
        # configured voice only for a language without one - so `tts.voice`
        # was silently ignored for every language that has a default, which
        # is all of them. A voicepack must match the language's frontend, so
        # a configured voice is only honoured when its prefix agrees.
        spoken = FALLBACK_LANGUAGES.get(language, language)
        default = DEFAULT_VOICES.get(spoken, "if_sara")
        voicepack = self.voice if self.voice and self.voice[0] == default[0] else default
        if voicepack != self.voice and self.voice:
            log.warning("voice %r does not belong to %s; using %s", self.voice, spoken, voicepack)

        chunks = [
            np.asarray(audio, dtype=np.float32).reshape(-1)
            for _, _, audio in pipeline(text, voice=voicepack, speed=self.speed)
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def speak(
        self, text: str, language: str | None = None, voice: str | None = None
    ) -> Speech:
        speech = super().speak(text, language, voice)
        lang = language or self.language
        if lang in FALLBACK_LANGUAGES:
            speech.language = f"{lang} (spoken as {FALLBACK_LANGUAGES[lang]})"
        return speech

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["voicepack"] = self.voice
        info["speed"] = self.speed
        info["can_clone"] = False
        return info
