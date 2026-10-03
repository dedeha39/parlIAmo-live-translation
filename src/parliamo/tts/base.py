"""The speech synthesis contract.

Synthesis is the last stage and the one the audience actually hears, so two
numbers matter and they are not the same:

**Time to first audio.** If output is streamed, this is what the audience
experiences as the delay. Everything after it overlaps with playback.

**Total generation time.** If output is not streamed, this is the delay, and it
scales with sentence length in a way the first-chunk figure does not.

A backend that cannot stream must be judged on the second. Chatterbox generates
a whole utterance before returning, so for now the two are the same and the
distinction is recorded rather than exploited - it becomes real when a streaming
backend appears.

Voice cloning
-------------
A reference recording is turned into a speaker embedding once and cached. Doing
that per utterance would add seconds on stage, and the embedding does not change
between sentences.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Speech:
    """One synthesised utterance."""

    audio: np.ndarray
    sample_rate: int
    text: str
    language: str
    compute_s: float = 0.0
    first_audio_s: float = 0.0
    backend: str = ""
    voice: str = "default"

    @property
    def duration_s(self) -> float:
        return self.audio.size / float(self.sample_rate)

    @property
    def rtf(self) -> float:
        """Compute seconds per second of audio produced.

        Below 1.0 means synthesis outruns playback, which is the condition for
        speaking continuously without falling behind.
        """
        if self.duration_s <= 0:
            return float("inf")
        return self.compute_s / self.duration_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "voice": self.voice,
            "audio_s": round(self.duration_s, 3),
            "compute_s": round(self.compute_s, 3),
            "first_audio_s": round(self.first_audio_s, 3),
            "rtf": round(self.rtf, 3),
            "backend": self.backend,
        }


@dataclass(slots=True)
class VoiceProfile:
    """A cloned voice, with the consent record that authorises it.

    ``consent`` is not decoration. Cloning a voice without documented
    permission is the thing this project exists to warn people about, so the
    profile that carries the voice carries the authorisation with it.
    """

    name: str
    reference_path: str
    consent: str = ""
    notes: str = ""
    embedding: Any = field(default=None, repr=False)

    def validate(self) -> None:
        if not self.consent.strip():
            raise ValueError(
                f"voice profile {self.name!r} has no consent record. "
                "Set consent to the signed authorisation reference before cloning."
            )


class TTSBackend(ABC):
    """A speech synthesiser."""

    name: str = "abstract"
    sample_rate: int = 24000

    def __init__(
        self,
        model: str = "default",
        device: str = "cuda",
        language: str = "it",
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.language = language
        self.options = kwargs
        self._loaded = False
        self.load_vram_mb: float = 0.0
        self.load_seconds: float = 0.0
        self._voices: dict[str, VoiceProfile] = {}

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

    def warmup(self, text: str = "Questa è una prova.") -> float:
        """Synthesise once so the first real utterance is not the slow one."""
        if not self._loaded:
            self.load()
        t0 = time.perf_counter()
        self.speak(text)
        return time.perf_counter() - t0

    def __enter__(self) -> TTSBackend:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.unload()

    # -- voices -----------------------------------------------------------

    def register_voice(self, profile: VoiceProfile) -> None:
        """Add a cloned voice. Refuses profiles without a consent record."""
        profile.validate()
        self._voices[profile.name] = profile
        log.info("voice %r registered from %s", profile.name, profile.reference_path)

    @property
    def voices(self) -> list[str]:
        return sorted(self._voices)

    # -- to implement -----------------------------------------------------

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _unload(self) -> None: ...

    @abstractmethod
    def _synthesise(self, text: str, language: str, voice: VoiceProfile | None) -> np.ndarray: ...

    # -- public API -------------------------------------------------------

    def speak(
        self, text: str, language: str | None = None, voice: str | None = None
    ) -> Speech:
        if not self._loaded:
            self.load()
        lang = language or self.language
        profile = self._voices.get(voice) if voice else None
        if voice and profile is None:
            raise KeyError(
                f"unknown voice {voice!r}. Registered: {', '.join(self.voices) or 'none'}"
            )

        stripped = text.strip()
        if not stripped:
            return Speech(
                audio=np.zeros(0, dtype=np.float32), sample_rate=self.sample_rate,
                text="", language=lang, backend=f"{self.name}:{self.model}",
            )

        t0 = time.perf_counter()
        audio = self._synthesise(stripped, lang, profile)
        elapsed = time.perf_counter() - t0

        return Speech(
            audio=np.asarray(audio, dtype=np.float32).reshape(-1),
            sample_rate=self.sample_rate,
            text=stripped,
            language=lang,
            compute_s=elapsed,
            # Without streaming, nothing is audible until generation finishes.
            first_audio_s=elapsed,
            backend=f"{self.name}:{self.model}",
            voice=voice or "default",
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "device": self.device,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "load_seconds": round(self.load_seconds, 2),
            "load_vram_mb": self.load_vram_mb,
            "voices": self.voices,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.model} {self.language}>"
