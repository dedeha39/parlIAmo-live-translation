"""The half-duplex gate: the single most important stage-safety component.

The problem it solves
---------------------
Synthesised Italian comes out of the room speakers. The microphone hears it.
ASR transcribes it. MT translates it. TTS speaks it. Within a few seconds the
system is translating its own output in a loop that gets louder each pass, and
the demo is over.

Acoustic echo cancellation is the sophisticated answer, but it needs tuning per
room and fails in exactly the conditions a live stage produces. The blunt answer
works reliably: while we are speaking, we do not listen.

The gate is closed by playback and reopened a configurable tail after playback
ends, because room reverberation outlives the last sample.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GateStats:
    """Bookkeeping so a rehearsal can answer 'how much speech did we drop?'."""

    closes: int = 0
    muted_frames: int = 0
    muted_seconds: float = 0.0
    speech_during_mute_blocks: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "closes": self.closes,
            "muted_frames": self.muted_frames,
            "muted_seconds": round(self.muted_seconds, 3),
            "speech_during_mute_blocks": self.speech_during_mute_blocks,
        }


class HalfDuplexGate:
    """Thread-safe open/closed gate shared by the capture and playback threads.

    ``enabled=False`` turns the gate into a permanent open, which is what you
    want when the operator is monitoring on headphones and there is no acoustic
    path from the speakers back to the microphone.
    """

    def __init__(self, enabled: bool = True, tail_ms: int = 250) -> None:
        self.enabled = enabled
        self.tail_ms = max(0, int(tail_ms))
        self._lock = threading.Lock()
        self._closed_until: float = 0.0
        self._hold_count: int = 0
        self.stats = GateStats()

    # -- state ------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True when the microphone should be listened to."""
        if not self.enabled:
            return True
        with self._lock:
            if self._hold_count > 0:
                return False
            return time.monotonic() >= self._closed_until

    @property
    def is_closed(self) -> bool:
        return not self.is_open

    def remaining_ms(self) -> float:
        """Milliseconds until the gate reopens (0 when already open)."""
        if not self.enabled:
            return 0.0
        with self._lock:
            if self._hold_count > 0:
                return float("inf")
            return max(0.0, (self._closed_until - time.monotonic()) * 1000.0)

    # -- control ----------------------------------------------------------

    def close(self) -> None:
        """Hold the gate closed until :meth:`release` is called."""
        if not self.enabled:
            return
        with self._lock:
            self._hold_count += 1
            if self._hold_count == 1:
                self.stats.closes += 1
        log.debug("gate closed (holds=%d)", self._hold_count)

    def release(self) -> None:
        """Drop one hold; when the last one goes, start the reopen tail."""
        if not self.enabled:
            return
        with self._lock:
            if self._hold_count > 0:
                self._hold_count -= 1
            if self._hold_count == 0:
                self._closed_until = time.monotonic() + self.tail_ms / 1000.0
        log.debug("gate released (holds=%d, tail=%d ms)", self._hold_count, self.tail_ms)

    def force_open(self) -> None:
        """Emergency override - used by the operator panic control."""
        with self._lock:
            self._hold_count = 0
            self._closed_until = 0.0
        log.warning("gate force-opened")

    # -- accounting -------------------------------------------------------

    def note_muted_block(self, frames: int, sample_rate: int, had_speech: bool = False) -> None:
        """Record that *frames* of audio were discarded while the gate was closed."""
        with self._lock:
            self.stats.muted_frames += frames
            self.stats.muted_seconds += frames / float(sample_rate)
            if had_speech:
                self.stats.speech_during_mute_blocks += 1

    def __enter__(self) -> HalfDuplexGate:
        self.close()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "open" if self.is_open else "closed"
        return f"<HalfDuplexGate {state} enabled={self.enabled} tail={self.tail_ms}ms>"
