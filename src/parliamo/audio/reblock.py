"""Re-chunking a stream into fixed-size frames without losing samples.

Why this exists
---------------
The capture layer resamples 48 kHz device audio down to the 16 kHz the pipeline
runs at, and ``soxr``'s streaming resampler does not return a fixed number of
samples per call — it buffers internally and emits whatever is ready. Measured
on the reference laptop: 1536 input frames per callback produced chunks of
1100 samples, not the 512 the arithmetic suggests.

Silero VAD, meanwhile, requires *exactly* 512 samples per call.

The first version of the pipeline bridged that gap by trimming each chunk to
512 samples, which silently discarded 53% of the audio — the microphone was
working perfectly and half of what it heard went in the bin. This class does it
properly: accumulate, emit whole frames, keep the remainder for next time.
"""

from __future__ import annotations

import numpy as np


class Reblocker:
    """Buffers a stream of arbitrary-length chunks into fixed-size frames.

    >>> rb = Reblocker(4)
    >>> [f.tolist() for f in rb.push(np.arange(6, dtype=np.float32))]
    [[0.0, 1.0, 2.0, 3.0]]
    >>> [f.tolist() for f in rb.push(np.arange(6, 10, dtype=np.float32))]
    [[4.0, 5.0, 6.0, 7.0]]
    """

    def __init__(self, frame_size: int) -> None:
        if frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {frame_size}")
        self.frame_size = frame_size
        self._buffer = np.zeros(0, dtype=np.float32)
        self.samples_in = 0
        self.frames_out = 0

    @property
    def pending(self) -> int:
        """Samples held back because they do not fill a whole frame yet."""
        return int(self._buffer.size)

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Add *chunk* and return every complete frame it made available."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []
        self.samples_in += int(chunk.size)

        self._buffer = (
            chunk if self._buffer.size == 0 else np.concatenate([self._buffer, chunk])
        )

        n_frames = self._buffer.size // self.frame_size
        if n_frames == 0:
            return []

        cut = n_frames * self.frame_size
        # Copy each frame out: the caller may hold on to it (the segmenter
        # buffers frames for the length of an utterance) and the source buffer
        # is about to be replaced.
        frames = [
            self._buffer[i * self.frame_size : (i + 1) * self.frame_size].copy()
            for i in range(n_frames)
        ]
        self._buffer = self._buffer[cut:].copy()
        self.frames_out += n_frames
        return frames

    def flush(self) -> np.ndarray | None:
        """Return the remainder zero-padded to one frame, or None if empty."""
        if self._buffer.size == 0:
            return None
        frame = np.zeros(self.frame_size, dtype=np.float32)
        frame[: self._buffer.size] = self._buffer
        self._buffer = np.zeros(0, dtype=np.float32)
        self.frames_out += 1
        return frame

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    def accounting(self) -> dict[str, int]:
        """Samples in versus samples emitted - these must agree."""
        return {
            "samples_in": self.samples_in,
            "frames_out": self.frames_out,
            "samples_out": self.frames_out * self.frame_size,
            "pending": self.pending,
        }
