"""The ASR backend contract.

Every recogniser we might use on stage sits behind this interface, so the
bake-off compares them on identical terms and the pipeline can swap one for
another without knowing which is loaded.

Two things every backend must report honestly, because they are the numbers the
hardware budget is built from:

* how long it took to produce the transcript, and
* how much VRAM it occupied while doing so.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Segment:
    """One timestamped chunk of recognised speech."""

    start: float
    end: float
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class Transcript:
    """The result of recognising one audio buffer."""

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    audio_duration_s: float = 0.0
    compute_s: float = 0.0
    backend: str = ""
    #: Segments discarded as decoder repetition loops, with the reason why.
    dropped_segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        """Real-time factor: compute seconds per audio second.

        Below 1.0 means the model runs faster than the speech it is
        transcribing, which is the bare minimum for a streaming pipeline. In
        practice we need a good deal of headroom, because translation and
        synthesis share the same GPU.
        """
        if self.audio_duration_s <= 0:
            return float("inf")
        return self.compute_s / self.audio_duration_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "audio_duration_s": round(self.audio_duration_s, 3),
            "compute_s": round(self.compute_s, 3),
            "rtf": round(self.rtf, 4),
            "n_segments": len(self.segments),
            "n_dropped": len(self.dropped_segments),
            "text": self.text,
        }


def cuda_allocated_mb() -> float:
    """VRAM currently held by this process, in MiB. Zero when CUDA is absent.

    Reads the driver's view rather than PyTorch's allocator, because
    CTranslate2 (which faster-whisper uses) allocates outside PyTorch entirely
    and would otherwise report as zero.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().splitlines()[0])
    except Exception:  # pragma: no cover - diagnostics only
        pass
    return 0.0


@contextmanager
def measure_vram(label: str = "") -> Iterator[dict[str, float]]:
    """Record GPU memory before and after a block.

    The delta is attributed to whatever ran inside, which is only meaningful if
    nothing else is competing for the card - hence the idle-VRAM warning in
    ``check_env.py``.
    """
    stats: dict[str, float] = {}
    before = cuda_allocated_mb()
    stats["vram_before_mb"] = before
    try:
        yield stats
    finally:
        after = cuda_allocated_mb()
        stats["vram_after_mb"] = after
        stats["vram_delta_mb"] = round(after - before, 1)
        if label:
            log.info("%s: VRAM %.0f -> %.0f MiB (%+.0f)", label, before, after,
                     stats["vram_delta_mb"])


class ASRBackend(ABC):
    """A speech recogniser."""

    #: Short identifier used in reports and config.
    name: str = "abstract"

    def __init__(self, model: str, device: str = "cuda", language: str = "tr", **kwargs: Any):
        self.model = model
        self.device = device
        self.language = language
        self.options = kwargs
        self._loaded = False
        self.load_vram_mb: float = 0.0
        self.load_seconds: float = 0.0

    # -- lifecycle --------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load weights, recording how long it took and what it cost in VRAM."""
        if self._loaded:
            return
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

    def warmup(self, seconds: float = 1.0, sample_rate: int = 16000) -> float:
        """Run one throwaway inference so the first real one is not the slow one.

        A cold first call can take several seconds while kernels compile and
        weights page in. On stage that is the difference between a demo that
        starts and one that appears to have hung.
        """
        if not self._loaded:
            self.load()
        silence = np.zeros(int(seconds * sample_rate), dtype=np.float32)
        t0 = time.perf_counter()
        self.transcribe(silence, sample_rate)
        return time.perf_counter() - t0

    def __enter__(self) -> ASRBackend:
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
    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> Transcript:
        """Recognise a mono float32 buffer. Must not mutate *audio*."""

    # -- reporting --------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model,
            "device": self.device,
            "language": self.language,
            "load_seconds": round(self.load_seconds, 2),
            "load_vram_mb": self.load_vram_mb,
            **{k: v for k, v in self.options.items() if isinstance(v, (str, int, float, bool))},
        }

    def __repr__(self) -> str:  # pragma: no cover
        state = "loaded" if self._loaded else "unloaded"
        return f"<{type(self).__name__} {self.model} on {self.device} ({state})>"
