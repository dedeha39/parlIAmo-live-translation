"""Speech synthesis backends and the registry that builds them."""

from __future__ import annotations

from typing import Any

from .base import Speech, TTSBackend, VoiceProfile

_REGISTRY: dict[str, type[TTSBackend]] = {}


def register(name: str, cls: type[TTSBackend]) -> None:
    _REGISTRY[name] = cls


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_backend(name: str, **kwargs: Any) -> TTSBackend:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown TTS backend {name!r}. Available: {', '.join(available_backends())}"
        )
    return _REGISTRY[name](**kwargs)


def _register_defaults() -> None:
    # Imported inside the function so that importing this package does not pull
    # in torch and the model weights on a machine that only needs audio.
    #
    # Registration must not fail because one backend's dependency is absent:
    # Kokoro carries the live path and Chatterbox only the cloning
    # demonstration, so a machine set up for one should still work.
    for module, attr in (
        (".kokoro_backend", "KokoroBackend"),
        (".piper_backend", "PiperBackend"),
        (".chatterbox_backend", "ChatterboxBackend"),
    ):
        try:
            import importlib

            cls = getattr(importlib.import_module(module, __package__), attr)
            register(cls.name, cls)
        except Exception as exc:  # pragma: no cover - absence is supported
            import logging

            logging.getLogger(__name__).debug("%s unavailable: %s", attr, exc)


_register_defaults()


def backend_for(language: str, preferred: str = "kokoro") -> str | None:
    """Which backend can speak *language*: the preferred one if it can, else Piper.

    None means no voice at all - that target is subtitles only. Kokoro is
    preferred because it runs on the GPU in ~0.2 s a sentence; Piper covers
    German and Turkish on the CPU, and only when its voice is on disk.
    """
    from .kokoro_backend import FALLBACK_LANGUAGES, LANG_CODES

    if preferred == "kokoro" and (language in LANG_CODES or language in FALLBACK_LANGUAGES):
        return "kokoro"
    if preferred not in ("kokoro", "piper"):
        return preferred
    try:
        from .piper_backend import available_languages
    except Exception:  # pragma: no cover - piper-tts not installed
        return None
    if language in available_languages():
        return "piper"
    return None


__all__ = [
    "Speech",
    "TTSBackend",
    "VoiceProfile",
    "available_backends",
    "backend_for",
    "create_backend",
    "register",
]
