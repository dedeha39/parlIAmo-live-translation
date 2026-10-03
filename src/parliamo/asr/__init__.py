"""Speech recognition backends and the registry that builds them."""

from __future__ import annotations

from typing import Any

from .base import ASRBackend, Segment, Transcript, cuda_allocated_mb, measure_vram

_REGISTRY: dict[str, type[ASRBackend]] = {}


def register(name: str, cls: type[ASRBackend]) -> None:
    _REGISTRY[name] = cls


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_backend(name: str, **kwargs: Any) -> ASRBackend:
    """Instantiate a registered backend, failing loudly on an unknown name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown ASR backend {name!r}. Available: {', '.join(available_backends())}"
        )
    return _REGISTRY[name](**kwargs)


def _register_defaults() -> None:
    # Imported lazily inside the function so that importing this package does
    # not drag in CTranslate2 (and therefore CUDA) on machines that only need
    # the audio layer.
    from .faster_whisper_backend import FasterWhisperBackend

    register(FasterWhisperBackend.name, FasterWhisperBackend)


_register_defaults()

__all__ = [
    "ASRBackend",
    "Segment",
    "Transcript",
    "available_backends",
    "create_backend",
    "cuda_allocated_mb",
    "measure_vram",
    "register",
]
