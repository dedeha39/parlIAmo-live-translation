"""The median pitch of a voice, and the shift that moves it to another.

RVC keeps the pitch it is given: fed Kokoro's Italian woman at 223 Hz it
produced the presenter's timbre at a woman's pitch. The fix was a semitone
shift, and the first one (-8) was found by ear. Measured the same way, the
voices this project speaks with differ by 14 semitones (Kokoro if_sara 223 Hz,
em_alex 139, Piper thorsten 125, dfki 105), so one shift cannot serve every
language; the shift is computed from the voice that will be converted.
"""

from __future__ import annotations

import math

import numpy as np

#: Wide enough for any adult voice, narrow enough that pYIN does not chase
#: harmonics or breath.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0


def median_f0(audio: np.ndarray, sample_rate: int) -> float | None:
    """Median fundamental frequency of the voiced frames, or None if none are voiced."""
    import librosa

    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    if samples.size < sample_rate // 4:
        return None
    f0, _voiced, _prob = librosa.pyin(samples, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=sample_rate)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if f0.size else None


def semitones(from_hz: float, to_hz: float) -> float:
    """The shift, in semitones, that moves *from_hz* to *to_hz*."""
    return 12.0 * math.log2(to_hz / from_hz)
