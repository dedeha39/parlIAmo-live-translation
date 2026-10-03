"""Audio playback with automatic half-duplex gating.

Playback owns the gate: it closes it before the first sample leaves the card and
releases it once the buffer has drained, so the microphone never hears the
synthesised voice. Callers do not have to remember to do this - if they use
:class:`AudioPlayback`, feedback protection is on by construction.

The stream stays open between utterances. Opening a WASAPI stream costs tens of
milliseconds, which is real money in a 2-second latency budget.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .devices import (
    STUTTERING_OUTPUT_APIS,
    AudioDevice,
    DeviceResolutionError,
    open_with_fallback,
)
from .gate import HalfDuplexGate

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PlaybackInfo:
    device: AudioDevice
    sample_rate: int
    channels: int
    latency_ms: float
    resampling: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.label,
            "device_index": self.device.index,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "latency_ms": self.latency_ms,
            "resampling": self.resampling,
        }


class AudioPlayback:
    """Queued playback of mono float32 audio."""

    def __init__(
        self,
        device: int | str | None = None,
        *,
        sample_rate: int = 24000,
        gate: HalfDuplexGate | None = None,
        gain: float = 1.0,
        channels: int = 2,
        output_latency_ms: float = 0.0,
    ) -> None:
        self.requested_sample_rate = sample_rate
        self.gate = gate
        self.gain = gain
        self.requested_channels = channels
        # Measured end-to-end delay of this output path, from
        # ``scripts/measure_audio_device.py``. PortAudio's reported latency only
        # covers its own buffers; it knows nothing about the vendor DSP chain
        # (Realtek/Nahimic effects added ~170 ms on the reference laptop). The
        # gate has to stay shut for the *real* delay, not the reported one, or
        # it reopens while the speakers are still emitting the last utterance.
        self.output_latency_ms = max(0.0, float(output_latency_ms))

        self._device_spec = device
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: Any = None
        self._info: PlaybackInfo | None = None
        self._resample_to: int | None = None
        self._idle = threading.Event()
        self._idle.set()
        self._pending_frames = 0
        #: Times the speakers ran dry mid-stream - heard as a gap. Counted in
        #: the callback (an increment, nothing more: the callback is the one
        #: place that must not block), reported from the delivery thread.
        self.underflows = 0
        self._underflows_reported = 0
        self._underflow_reported_at = 0.0
        #: One lock for everything the callback and the submitting thread
        #: share. The callback holds it for one buffer's worth of copying.
        self._lock = threading.Lock()
        self._gate_held = False
        #: The sentence the callback is part-way through, and how far. Owned
        #: by the callback: it used to put the remainder back on the queue,
        #: and a sentence submitted in that instant was played in the middle
        #: of the current one.
        self._current: np.ndarray | None = None
        self._offset = 0
        #: Set by stop(). A delivery thread that outlives stop() must not
        #: reopen the speakers by submitting one last sentence.
        self._stopped = False
        #: The gate's own tail before any output latency was added to it, so
        #: opening again does not add the latency a second time.
        self._gate_base_tail: int | None = None

    # -- lifecycle --------------------------------------------------------

    @property
    def info(self) -> PlaybackInfo:
        if self._info is None:
            raise RuntimeError("playback has not been started")
        return self._info

    @property
    def is_playing(self) -> bool:
        return not self._idle.is_set()

    def start(self) -> PlaybackInfo:
        """Open the output stream, walking down the candidate devices.

        Each candidate is tried once. The previous version re-resolved the same
        specification on every attempt, so three tries produced three identical
        failures - a retry that could not possibly succeed. See
        :func:`~parliamo.audio.devices.candidates`.
        """
        if self._stream is not None:
            return self.info
        self._stopped = False

        import sounddevice as sd

        def opener(device: AudioDevice) -> PlaybackInfo:
            # No sd._terminate() here: it invalidates every open stream,
            # including the microphone that is already running.
            try:
                return self._open(sd, device)
            except Exception:
                self._stream = None
                raise

        try:
            return open_with_fallback(self._device_spec, "output", opener)
        except DeviceResolutionError as exc:
            raise RuntimeError(
                f"{exc} Pick a different device in Setup, or run with --no-speak "
                "for subtitles only."
            ) from exc

    def _open(self, sd: Any, device: AudioDevice) -> PlaybackInfo:
        channels = min(self.requested_channels, max(1, device.max_output_channels))

        rate, resampling = self._negotiate_rate(sd, device, channels)
        self._resample_to = rate if resampling else None

        self._stream = sd.OutputStream(
            device=device.index,
            channels=channels,
            samplerate=rate,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

        self._info = PlaybackInfo(
            device=device,
            sample_rate=rate,
            channels=channels,
            latency_ms=round(float(self._stream.latency) * 1000.0, 2),
            resampling=resampling,
        )
        log.info(
            "playback open: %s @ %d Hz, %d ch, latency %.1f ms%s",
            device.label, rate, channels, self._info.latency_ms,
            f" (resampling from {self.requested_sample_rate})" if resampling else "",
        )
        self._extend_gate_tail()
        return self._info

    def _extend_gate_tail(self) -> None:
        """Make sure the gate outlives the sound actually leaving the speakers."""
        if self.gate is None or not self.gate.enabled:
            return
        if self._gate_base_tail is None:
            self._gate_base_tail = self.gate.tail_ms
        hardware_ms = max(self.output_latency_ms, self._info.latency_ms if self._info else 0.0)
        required = int(round(hardware_ms + self._gate_base_tail))
        if required > self.gate.tail_ms:
            log.info(
                "extending half-duplex tail %d -> %d ms to cover %.0f ms of output latency",
                self.gate.tail_ms, required, hardware_ms,
            )
            self.gate.tail_ms = required
        if self.output_latency_ms <= 0.0:
            log.warning(
                "pipeline.output_latency_ms is 0: the gate is sized from PortAudio's "
                "reported latency only, which excludes vendor DSP. Run "
                "scripts/measure_audio_device.py and set the measured round-trip."
            )

    def _negotiate_rate(self, sd: Any, device: AudioDevice, channels: int) -> tuple[int, bool]:
        try:
            sd.check_output_settings(
                device=device.index,
                channels=channels,
                samplerate=self.requested_sample_rate,
                dtype="float32",
            )
            return self.requested_sample_rate, False
        except Exception:
            native = int(round(device.default_samplerate))
            return native, native != self.requested_sample_rate

    def stop(self) -> None:
        self._stopped = True
        self.report_underflows(force=True)
        self.flush()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover
                log.warning("error closing output stream: %s", exc)
            self._stream = None
        self._release_gate()

    def __enter__(self) -> AudioPlayback:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- gate -------------------------------------------------------------

    def _hold_gate_locked(self) -> None:
        if self.gate is not None and not self._gate_held:
            self.gate.close()
            self._gate_held = True

    def _release_gate_locked(self) -> None:
        if self.gate is not None and self._gate_held:
            self.gate.release()
            self._gate_held = False

    def _release_gate(self) -> None:
        with self._lock:
            self._release_gate_locked()

    # -- callback ---------------------------------------------------------

    def _callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        # A gap only counts while something is playing: starved silence is
        # still silence.
        if status and getattr(status, "output_underflow", False) and not self._idle.is_set():
            self.underflows += 1

        outdata.fill(0.0)
        cursor = 0
        # Under the lock throughout: submit() and flush() change the same
        # state, and the decision "nothing is left, release the gate" is only
        # true if nothing can be added while it is being made.
        with self._lock:
            while cursor < frames:
                if self._current is None:
                    try:
                        chunk = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if chunk is None:  # end-of-utterance marker
                        continue
                    self._current, self._offset = chunk, 0

                take = min(frames - cursor, self._current.shape[0] - self._offset)
                segment = self._current[self._offset:self._offset + take]
                if outdata.shape[1] == 1:
                    outdata[cursor:cursor + take, 0] = segment
                else:
                    outdata[cursor:cursor + take, :] = segment[:, None]
                cursor += take
                self._offset += take
                self._pending_frames = max(0, self._pending_frames - take)
                if self._offset >= self._current.shape[0]:
                    self._current = None

            drained = self._current is None and self._queue.empty()
            if drained and not self._idle.is_set():
                self._idle.set()
                self._release_gate_locked()

    # -- submission -------------------------------------------------------

    def _prepare(self, samples: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        # The stream runs at one rate and a sentence can arrive at another:
        # RVC answers at 48 kHz, Seed-VC at 22.05. Written to a 24 kHz stream
        # as they were, RVC played at half speed an octave down - the cloned
        # voice "slowed down and robotic" - and Seed-VC 9% fast.
        source = int(sample_rate or self.requested_sample_rate)
        stream = self._resample_to or self.requested_sample_rate
        if source != stream and audio.size:
            import soxr

            audio = np.asarray(soxr.resample(audio, source, stream), dtype=np.float32)
        if self.gain != 1.0:
            audio = audio * self.gain
        # Hard-limit rather than let the card wrap: a clipped word is
        # intelligible, a wrapped one is a bang through a PA system.
        return np.clip(audio, -1.0, 1.0)

    #: At most one underflow warning per this many seconds: a starving stream
    #: underflows dozens of times a second, and the log is for reading.
    UNDERFLOW_REPORT_S = 10.0

    def report_underflows(self, force: bool = False) -> int:
        """Log the gaps since the last report; returns how many were new.

        Before 2026-10-02 they were logged at debug level and the run log could
        not show the choppy translation the room heard.
        """
        new = self.underflows - self._underflows_reported
        now = time.monotonic()
        if new <= 0 or (not force and now - self._underflow_reported_at < self.UNDERFLOW_REPORT_S):
            return 0
        self._underflows_reported = self.underflows
        self._underflow_reported_at = now
        device = self._info.device if self._info is not None else None
        cause = ("MME and DirectSound starve while the pipeline works; choose the WASAPI entry."
                 if getattr(device, "hostapi_name", "") in STUTTERING_OUTPUT_APIS
                 else "The machine is overloaded, or the output buffer is too small.")
        log.warning("the speakers ran dry %d time(s) on %s - the room heard gaps. %s",
                    new, getattr(device, "label", "the speakers"), cause)
        return new

    def submit(self, samples: np.ndarray, sample_rate: int | None = None) -> int:
        """Queue *samples* for playback and return the number of frames queued.

        *sample_rate* is the rate the samples were made at; it is converted to
        the stream's. Omitted, they are taken to be at the rate playback was
        opened with.
        """
        self.report_underflows()
        if self._stream is None:
            if self._stopped:
                log.warning("playback stopped; a late sentence was not played")
                return 0
            self.start()
        audio = self._prepare(samples, sample_rate)
        if audio.size == 0:
            return 0
        # All four together, under the callback's lock. Separately, a callback
        # landing between "busy" and "queued" saw nothing pending, decided the
        # speakers had drained, and released the gate - and the sentence then
        # played to an open microphone.
        with self._lock:
            self._hold_gate_locked()
            self._pending_frames += audio.shape[0]
            self._queue.put(audio)
            self._idle.clear()
        return int(audio.shape[0])

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queue has drained. Returns False on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.is_playing:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                return False
            if self._idle.wait(timeout=min(0.1, remaining or 0.1)):
                break
        # Let the card's own buffer drain before declaring silence.
        if self._info is not None:
            time.sleep(self._info.latency_ms / 1000.0)
        self.report_underflows()
        return True

    def play(self, samples: np.ndarray, blocking: bool = True,
             sample_rate: int | None = None) -> int:
        """Convenience: submit and optionally wait for completion."""
        frames = self.submit(samples, sample_rate)
        if blocking:
            self.wait()
        return frames

    def flush(self) -> None:
        """Discard everything queued and stop immediately (panic control)."""
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._current, self._offset = None, 0
            self._pending_frames = 0
            self._idle.set()
            self._release_gate_locked()
