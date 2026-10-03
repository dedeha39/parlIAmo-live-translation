"""Building a good reference recording for voice cloning.

Voice conversion derives a speaker embedding from one reference recording, so
the reference decides how much the result sounds like the person. Three things
about it matter, and on this project all three went wrong at once:

**Length.** `config/default.yaml` asks for 15-30 s. The first live test used a
single 7.8 s sentence, and the result was judged not good enough.

**Continuity.** The reference laptop's microphone runs through Intel Smart Sound,
which gates non-speech to *digital silence*. Measured across 40 in-situ
recordings, between 7% and 39% of every file is exactly zero. Those holes are
not silence the speaker produced; they are the driver removing audio, including
the quiet onsets and tails of words. A reference made of the wrong file is
mostly holes.

**Level.** The same recordings sit around -30 dBFS peak ~0.24. Quiet input costs
accuracy everywhere, and a speaker encoder given a signal 12 dB down is being
asked to characterise a voice from less of it.

So this module assembles a reference the way you would if you were choosing by
ear: prefer the recordings with the least gating, drop the dead air inside them,
join them with short natural pauses, and normalise the level.

None of that invents signal. It selects and presents what was actually recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Samples below this magnitude are treated as the driver's gate rather than as
#: quiet speech. The gate writes exact zeros, so this is deliberately far below
#: any real noise floor - a genuinely quiet room measures around -70 dBFS.
SILENCE_FLOOR = 1e-5

#: Pause inserted between joined recordings. Enough to read as a sentence
#: boundary, short enough not to dilute the reference with silence.
JOIN_PAUSE_S = 0.25

#: Leave this much silence at each edge of a kept region, so word onsets and
#: tails are not clipped by the trimmer itself.
EDGE_KEEP_S = 0.05


@dataclass(slots=True)
class ClipStats:
    """What one candidate recording offers as reference material."""

    path: Path
    duration_s: float
    voiced_s: float
    peak: float
    rms_dbfs: float
    gated_fraction: float

    @property
    def usable_s(self) -> float:
        return self.voiced_s

    def as_dict(self) -> dict[str, float | str]:
        return {
            "file": self.path.name,
            "duration_s": round(self.duration_s, 2),
            "voiced_s": round(self.voiced_s, 2),
            "peak": round(self.peak, 3),
            "rms_dbfs": round(self.rms_dbfs, 1),
            "gated_pct": round(100 * self.gated_fraction, 1),
        }


#: Zeros shorter than this are not the gate. Every waveform crosses zero, and
#: at 16 kHz a 220 Hz tone spends ~0.25% of its samples below the floor doing
#: exactly that - which a naive count reports as "gated" and is not.
MIN_GATE_RUN_S = 0.01


def _gated_mask(audio: np.ndarray, sample_rate: int, min_run_s: float = MIN_GATE_RUN_S):
    """Samples belonging to a run of silence long enough to be the driver's gate."""
    quiet = np.abs(audio) < SILENCE_FLOOR
    if not quiet.any():
        return quiet

    min_run = max(1, int(min_run_s * sample_rate))
    mask = np.zeros(audio.size, dtype=bool)
    edges = np.diff(quiet.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if quiet[0]:
        starts.insert(0, 0)
    if quiet[-1]:
        ends.append(audio.size)
    for start, end in zip(starts, ends, strict=True):
        if end - start >= min_run:
            mask[start:end] = True
    return mask


def measure_clip(audio: np.ndarray, sample_rate: int, path: Path | None = None) -> ClipStats:
    """Describe one recording as reference material."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return ClipStats(path or Path("-"), 0.0, 0.0, 0.0, -np.inf, 1.0)

    gated = _gated_mask(audio, sample_rate)
    rms = float(np.sqrt(np.mean(audio**2)))
    return ClipStats(
        path=path or Path("-"),
        duration_s=audio.size / sample_rate,
        voiced_s=float((~gated).sum()) / sample_rate,
        peak=float(np.abs(audio).max()),
        rms_dbfs=20 * float(np.log10(max(rms, 1e-9))),
        gated_fraction=float(gated.mean()),
    )


def trim_gated_regions(
    audio: np.ndarray,
    sample_rate: int,
    min_gap_s: float = 0.20,
    edge_keep_s: float = EDGE_KEEP_S,
) -> np.ndarray:
    """Remove stretches the driver gated to zero, keeping a margin at each edge.

    Only runs of *at least* ``min_gap_s`` are removed. Short zero runs inside a
    word are left alone: cutting them would splice two halves of a phoneme
    together and produce a click, which is worse for the embedding than the
    hole was.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio

    quiet = np.abs(audio) < SILENCE_FLOOR
    if not quiet.any():
        return audio

    min_gap = max(1, int(min_gap_s * sample_rate))
    margin = int(edge_keep_s * sample_rate)

    # Boundaries of every run of silence.
    edges = np.diff(quiet.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if quiet[0]:
        starts.insert(0, 0)
    if quiet[-1]:
        ends.append(audio.size)

    drop = np.zeros(audio.size, dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        if end - start < min_gap:
            continue
        drop[start + margin : max(start + margin, end - margin)] = True

    kept = audio[~drop]
    return kept if kept.size else audio


def normalise(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Scale to *target_peak*. Does not compress, so dynamics are preserved."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak <= 0.0:
        return audio
    return np.clip(audio * (target_peak / peak), -1.0, 1.0).astype(np.float32)


def rank_candidates(stats: list[ClipStats]) -> list[ClipStats]:
    """Best reference material first.

    Sorted by how little of the file the driver removed, then by level. Gating
    is the dominant term because a hole is missing signal, while a quiet file
    is merely scaled - and scaling is something we can fix afterwards.
    """
    return sorted(stats, key=lambda c: (c.gated_fraction, -c.rms_dbfs))


def build_reference(
    clips: list[tuple[np.ndarray, int]],
    target_seconds: float = 25.0,
    sample_rate: int | None = None,
    trim: bool = True,
    normalise_peak: float | None = 0.95,
) -> np.ndarray:
    """Join *clips* into one reference of roughly *target_seconds*.

    Clips are consumed in the order given, so rank them first. The result stops
    once the target is reached rather than using everything: a longer reference
    is not better once the embedding has enough, and the extra material is more
    useful kept back for a second reference to compare against.
    """
    if not clips:
        return np.zeros(0, dtype=np.float32)
    rate = sample_rate or clips[0][1]
    pause = np.zeros(int(JOIN_PAUSE_S * rate), dtype=np.float32)

    pieces: list[np.ndarray] = []
    collected = 0.0
    for audio, clip_rate in clips:
        if clip_rate != rate:
            raise ValueError(f"clip at {clip_rate} Hz among {rate} Hz clips; resample first")
        piece = trim_gated_regions(audio, rate) if trim else np.asarray(audio, dtype=np.float32)
        if piece.size == 0:
            continue
        if pieces:
            pieces.append(pause)
            collected += JOIN_PAUSE_S
        pieces.append(piece)
        collected += piece.size / rate
        if collected >= target_seconds:
            break

    if not pieces:
        return np.zeros(0, dtype=np.float32)
    out = np.concatenate(pieces)
    if normalise_peak is not None:
        out = normalise(out, normalise_peak)
    return out
