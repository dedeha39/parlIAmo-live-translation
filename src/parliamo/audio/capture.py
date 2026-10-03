"""Microphone capture.

Delivers fixed-size blocks of mono ``float32`` audio at the pipeline sample rate
(16 kHz), regardless of what the device natively runs at.

Design notes
------------
*Never block inside the PortAudio callback.* The callback runs on a realtime
thread; a slow consumer there produces dropouts, not backpressure. So the
callback does the cheap work (downmix, resample, enqueue) and drops the oldest
block when the queue is full, recording the drop for later inspection.

WASAPI shared mode will not always open at an arbitrary sample rate, so we try
the pipeline rate first and fall back to the device's native rate with soxr
resampling. Which path was taken is reported in :attr:`CaptureInfo`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .devices import AudioDevice, DeviceResolutionError, open_with_fallback
from .gate import HalfDuplexGate

log = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 16000


@dataclass(slots=True)
class CaptureInfo:
    """What actually happened when the stream opened, as opposed to what we asked for."""

    device: AudioDevice
    device_sample_rate: int
    pipeline_sample_rate: int
    channels: int
    block_frames: int
    resampling: bool
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.label,
            "device_index": self.device.index,
            "device_sample_rate": self.device_sample_rate,
            "pipeline_sample_rate": self.pipeline_sample_rate,
            "channels": self.channels,
            "block_frames": self.block_frames,
            "resampling": self.resampling,
            "latency_ms": self.latency_ms,
            "likely_bluetooth": self.device.likely_bluetooth,
        }


@dataclass(slots=True)
class CaptureStats:
    blocks_delivered: int = 0
    blocks_dropped: int = 0
    blocks_muted: int = 0
    #: Discarded because the operator paused listening (P on the page).
    blocks_paused: int = 0
    callback_overflows: int = 0
    peak_amplitude: float = 0.0
    #: The peak over roughly the last second, decaying. ``peak_amplitude`` is
    #: the run's all-time maximum, which answers "did the microphone ever hear
    #: anything" and nothing else: once one loud block has passed it stays high
    #: for the rest of the talk. This one answers "is it hearing anything
    #: *now*" - the first question on a stage when no subtitle appears.
    recent_peak: float = 0.0
    started_at: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        elapsed = max(1e-6, time.monotonic() - self.started_at)
        return {
            "blocks_delivered": self.blocks_delivered,
            "blocks_dropped": self.blocks_dropped,
            "blocks_muted": self.blocks_muted,
            "blocks_paused": self.blocks_paused,
            "callback_overflows": self.callback_overflows,
            "peak_amplitude": round(self.peak_amplitude, 4),
            "recent_peak": round(self.recent_peak, 4),
            "elapsed_s": round(elapsed, 2),
        }


class AudioCapture:
    """Threaded microphone capture producing 16 kHz mono float32 blocks."""

    def __init__(
        self,
        device: int | str | None = None,
        *,
        sample_rate: int = PIPELINE_SAMPLE_RATE,
        block_ms: int = 32,
        gate: HalfDuplexGate | None = None,
        max_queue_blocks: int = 64,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.block_frames = int(sample_rate * block_ms / 1000)
        self.gate = gate
        self.stats = CaptureStats()
        #: Set from the operator page; read by the audio callback.
        self.paused = False

        self._device_spec = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_blocks)
        self._stream: Any = None
        self._info: CaptureInfo | None = None
        self._resampler: Any = None
        self._running = threading.Event()

    # -- lifecycle --------------------------------------------------------

    @property
    def info(self) -> CaptureInfo:
        if self._info is None:
            raise RuntimeError("capture has not been started")
        return self._info

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self) -> CaptureInfo:
        """Open the input stream, walking down the candidate devices. Idempotent.

        Opening can fail for reasons that have nothing to do with the choice of
        device. PortAudio addresses devices by *index*, and the index list is
        rebuilt whenever anything connects - a Bluetooth headset pairing
        mid-start is enough to make the index resolved a moment ago point at
        something else. Observed on this machine as::

            Unanticipated host error [PaErrorCode -9999]
            'GetNameFromCategory: usbTerminalGUID = ...' [Windows WDM-KS error]

        for a microphone that opened cleanly a minute later.

        So a failure walks to the *next* device rather than retrying the same
        one, which is the only kind of retry that can help. A working
        microphone at 90 ms on MME beats a correct one that is shut, and on
        stage the difference is the whole talk. See
        :func:`~parliamo.audio.devices.candidates` for the order.
        """
        if self._running.is_set():
            return self.info

        import sounddevice as sd

        def opener(device: AudioDevice) -> CaptureInfo:
            # verify_index (inside open_with_fallback) re-enumerates, which is
            # all that is needed. Re-initialising PortAudio here would be worse
            # than the bug: sd._terminate() invalidates *every* open stream, so
            # opening the speakers would silently kill the microphone. A test
            # caught that within a minute of it being written.
            try:
                return self._open(sd, device)
            except Exception:
                self._stream = None
                self._resampler = None
                raise

        try:
            return open_with_fallback(self._device_spec, "input", opener)
        except DeviceResolutionError as exc:
            raise RuntimeError(
                f"{exc} Pick a different device in Setup, or check that nothing "
                "else is using it."
            ) from exc

    def _open(self, sd: Any, device: AudioDevice) -> CaptureInfo:
        channels = 1 if device.max_input_channels >= 1 else device.max_input_channels

        device_rate, resampling = self._negotiate_rate(sd, device, channels)
        device_block = int(round(self.block_frames * device_rate / self.sample_rate))

        if resampling:
            import soxr

            self._resampler = soxr.ResampleStream(
                device_rate, self.sample_rate, channels, dtype="float32", quality="VHQ"
            )

        self._stream = sd.InputStream(
            device=device.index,
            channels=channels,
            samplerate=device_rate,
            blocksize=device_block,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        self._running.set()

        self._info = CaptureInfo(
            device=device,
            device_sample_rate=device_rate,
            pipeline_sample_rate=self.sample_rate,
            channels=channels,
            block_frames=self.block_frames,
            resampling=resampling,
            latency_ms=round(float(self._stream.latency) * 1000.0, 2),
        )
        log.info(
            "capture open: %s @ %d Hz%s, block %d frames, latency %.1f ms",
            device.label,
            device_rate,
            f" (resampling to {self.sample_rate})" if resampling else "",
            device_block,
            self._info.latency_ms,
        )
        if device.likely_bluetooth:
            log.warning(
                "%s looks like a Bluetooth device: opening its microphone forces the "
                "HFP profile (mono, <=16 kHz, compressed) and adds 100-250 ms of "
                "latency. Prefer a wired lavalier for the stage.",
                device.name,
            )
        return self._info

    def _negotiate_rate(self, sd: Any, device: AudioDevice, channels: int) -> tuple[int, bool]:
        """Prefer the pipeline rate; fall back to the device native rate."""
        try:
            sd.check_input_settings(
                device=device.index,
                channels=channels,
                samplerate=self.sample_rate,
                dtype="float32",
            )
            return self.sample_rate, False
        except Exception as exc:
            native = int(round(device.default_samplerate))
            log.debug(
                "device %s rejected %d Hz (%s); using native %d Hz with resampling",
                device.name, self.sample_rate, exc, native,
            )
            return native, native != self.sample_rate

    def stop(self) -> CaptureStats:
        """Close the stream and return the run statistics."""
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover
                log.warning("error closing input stream: %s", exc)
            self._stream = None
        self._resampler = None
        log.info("capture closed: %s", self.stats.as_dict())
        return self.stats

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- callback ---------------------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.stats.callback_overflows += 1
            log.debug("input stream status: %s", status)

        block = indata[:, 0].astype(np.float32, copy=True) if indata.ndim > 1 else indata.astype(np.float32, copy=True)

        if self._resampler is not None:
            block = self._resampler.resample_chunk(block)
            if block.size == 0:
                return
            block = np.asarray(block, dtype=np.float32).reshape(-1)

        peak = float(np.abs(block).max()) if block.size else 0.0
        if peak > self.stats.peak_amplitude:
            self.stats.peak_amplitude = peak
        # ~1 s half-life at 32 ms blocks: loud enough to register a word,
        # quick enough to fall back to the floor between sentences.
        self.stats.recent_peak = max(peak, self.stats.recent_peak * 0.93)

        # Paused by the operator - a video through the room's speakers,
        # applause, a question from the floor. The gate only knows the
        # system's own voice; everything else the speakers play the microphone
        # hears, and was transcribed, subtitled and spoken over the video.
        # The meter above still moves, so the operator can see it is ignored.
        if self.paused:
            self.stats.blocks_paused += 1
            return

        # Half-duplex: discard audio captured while we are speaking.
        if self.gate is not None and self.gate.is_closed:
            self.stats.blocks_muted += 1
            self.gate.note_muted_block(block.size, self.sample_rate, had_speech=peak > 0.02)
            return

        try:
            self._queue.put_nowait(block)
            self.stats.blocks_delivered += 1
        except queue.Full:
            # Drop the oldest block so the newest audio always wins: falling
            # behind by a second is recoverable, drifting forever is not.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(block)
            except (queue.Empty, queue.Full):  # pragma: no cover
                pass
            self.stats.blocks_dropped += 1

    # -- consumption ------------------------------------------------------

    def read(self, timeout: float | None = 1.0) -> np.ndarray | None:
        """Pop one block, or None if nothing arrived within *timeout* seconds."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def blocks(self, timeout: float | None = 1.0) -> Iterator[np.ndarray]:
        """Yield blocks until :meth:`stop` is called."""
        while self._running.is_set():
            block = self.read(timeout=timeout)
            if block is not None:
                yield block

    def record(self, seconds: float) -> np.ndarray:
        """Collect *seconds* of audio into one array. Blocking; for tooling only."""
        wanted = int(seconds * self.sample_rate)
        chunks: list[np.ndarray] = []
        collected = 0
        deadline = time.monotonic() + seconds * 3 + 2.0
        while collected < wanted and time.monotonic() < deadline:
            block = self.read(timeout=1.0)
            if block is None:
                continue
            chunks.append(block)
            collected += block.size
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)[:wanted]

    def drain(self) -> int:
        """Discard everything currently queued. Returns the number of blocks dropped."""
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped
