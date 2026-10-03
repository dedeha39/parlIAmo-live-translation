"""Chatterbox Multilingual v3 (Resemble AI).

Chosen for three reasons, in this order:

1. **It covers Turkish and Italian**, and zero-shot voice cloning from a short
   reference. Few open models cover both languages at all.
2. **MIT licence.** NLLB, by contrast, is CC-BY-NC, which is fine for a
   presentation and a blocker for anything else.
3. **It watermarks every output by default** (Perth). For a talk whose subject
   is how easy voice cloning has become, being able to clone a voice and then
   detect the watermark in the same breath is the demonstration.

Friulian
--------
There is no Friulian TTS model anywhere, and no Friulian language token here.
Friulian text is synthesised through the Italian frontend: correct timbre,
Italian phonetics. That is a documented limitation to be disclosed on stage,
not a defect to hide - and it is a concrete illustration of the point the talk
is making about minority languages.

Dependency warning
------------------
``chatterbox-tts`` pins ``torch==2.6.0`` exactly. Installing it downgrades a
CUDA build of torch to the CPU wheel and silently breaks every other stage. The
pin is over-strict: the package imports and runs correctly against torch 2.11
with CUDA. Install it, then reinstall torch. See docs/01-setup.md.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import Speech, TTSBackend, VoiceProfile

log = logging.getLogger(__name__)

#: Chatterbox Multilingual language ids for the languages this project uses.
LANGUAGE_IDS = {
    "it": "it",
    "tr": "tr",
    "en": "en",
    "de": "de",
    "fr": "fr",
    "es": "es",
}

#: Friulian has no model and no language id. Synthesise it as Italian and say so.
FALLBACK_LANGUAGES = {"fur": "it"}


class ChatterboxBackend(TTSBackend):
    name = "chatterbox"
    sample_rate = 24000

    def __init__(
        self,
        model: str = "chatterbox-multilingual-v3",
        device: str = "cuda",
        language: str = "it",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.6,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, device=device, language=language,
            exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature,
            **kwargs,
        )
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self._model: Any = None

    # -- lifecycle --------------------------------------------------------

    def _load(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        log.info("loading Chatterbox Multilingual on %s", self.device)
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
        self.sample_rate = int(getattr(self._model, "sr", self.sample_rate))

    def _unload(self) -> None:
        self._model = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- language ---------------------------------------------------------

    def resolve_language(self, language: str) -> tuple[str, str]:
        """Map a project language code to a Chatterbox id.

        Returns ``(id, note)``; *note* is non-empty when a substitution was made
        and should be surfaced rather than swallowed.
        """
        if language in LANGUAGE_IDS:
            return LANGUAGE_IDS[language], ""
        if language in FALLBACK_LANGUAGES:
            substitute = FALLBACK_LANGUAGES[language]
            return LANGUAGE_IDS[substitute], (
                f"no {language} voice exists; synthesised with the {substitute} "
                "frontend, so the phonetics are wrong"
            )
        raise KeyError(
            f"language {language!r} is not supported. Known: "
            f"{', '.join(sorted(LANGUAGE_IDS) + sorted(FALLBACK_LANGUAGES))}"
        )

    # -- synthesis --------------------------------------------------------

    def _synthesise(
        self, text: str, language: str, voice: VoiceProfile | None
    ) -> np.ndarray:
        import torch

        language_id, note = self.resolve_language(language)
        if note:
            log.warning("%s", note)

        kwargs: dict[str, Any] = {
            "language_id": language_id,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "temperature": self.temperature,
        }
        if voice is not None:
            kwargs["audio_prompt_path"] = voice.reference_path

        with torch.inference_mode():
            wav = self._model.generate(text, **kwargs)

        if hasattr(wav, "detach"):
            wav = wav.detach().to("cpu").numpy()
        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def speak(
        self, text: str, language: str | None = None, voice: str | None = None
    ) -> Speech:
        speech = super().speak(text, language, voice)
        # Record the substitution on the result so a caller writing subtitles
        # or a report can state it, rather than only a log line nobody reads.
        lang = language or self.language
        if lang in FALLBACK_LANGUAGES:
            speech.language = f"{lang} (spoken as {FALLBACK_LANGUAGES[lang]})"
        return speech
