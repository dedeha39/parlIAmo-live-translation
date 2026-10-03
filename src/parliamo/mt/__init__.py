"""Machine translation backends and the registry that builds them."""

from __future__ import annotations

from typing import Any

from .base import MTBackend, Translation, sanitise

_REGISTRY: dict[str, type[MTBackend]] = {}


def register(name: str, cls: type[MTBackend]) -> None:
    _REGISTRY[name] = cls


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_backend(name: str, **kwargs: Any) -> MTBackend:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown MT backend {name!r}. Available: {', '.join(available_backends())}"
        )
    return _REGISTRY[name](**kwargs)


def _register_defaults() -> None:
    from .ctranslate2_nllb import CTranslate2NLLBBackend
    from .ollama_backend import OllamaBackend

    register(CTranslate2NLLBBackend.name, CTranslate2NLLBBackend)
    # Ollama needs no import-time dependency - it talks HTTP - so registering it
    # is free. Whether a server is actually running is discovered on load().
    register(OllamaBackend.name, OllamaBackend)


_register_defaults()

__all__ = [
    "MTBackend",
    "Translation",
    "available_backends",
    "create_backend",
    "register",
    "sanitise",
]
