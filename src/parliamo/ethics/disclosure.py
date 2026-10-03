"""Speaking the disclosure into the audio, so it cannot be trimmed off.

Why this exists, and why it is better than the watermark
--------------------------------------------------------
The watermark measurement in :mod:`parliamo.ethics.watermark` produced an
uncomfortable table. Perth marks Chatterbox output and the detector recovers it
cleanly. It recovers **nothing** from Kokoro, from Seed-VC, or from real
speech - correctly, because none of them mark anything. Which is the honest
finding: a watermark tells you the generator chose to mark its output, and an
attacker simply picks a generator that does not. The pipeline in this
repository is one.

An **audible** disclosure has the property the watermark lacks: it cannot be
removed without leaving a hole. It is not steganography and does not pretend to
be. Anyone can cut it out - but not without cutting audio, and the result stops
being a clean recording of someone saying something.

Three placements, and the middle one is the point
-------------------------------------------------
* **Start** - so a listener who hears the clip from the beginning is told.
* **End** - so a listener who joins late is told.
* **Middle, at a random position** - so the clip cannot be made usable by
  trimming a known offset off each end. This is what turns a courtesy into a
  safeguard, and it was the presenter's own suggestion.

The disclosure is spoken in a **different voice** from the cloned one, and
deliberately so: a warning delivered in the impersonated voice is one more
sentence that person never said.

What this is honestly for
-------------------------
It makes a demonstration clip unusable as a fabricated record. It does not make
cloning safe, it does not substitute for consent - :class:`VoiceProfile` still
refuses without a record - and it protects the person whose voice was cloned
rather than the person listening.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

#: Spoken in the target language, because the room is Italian.
DEFAULT_TEXT_IT = "Attenzione: questa voce è sintetica. Non è una registrazione reale."

#: A short pause each side, so the disclosure does not collide with speech and
#: is heard as an interruption rather than as part of the sentence.
PAD_S = 0.35

#: Never interrupt within this much of either edge - the start and end already
#: carry their own copy, and an interior one landing there is wasted.
EDGE_MARGIN_S = 1.0

#: Below this, one interruption in the middle is enough; the clip is too short
#: for trimming to be a useful attack anyway.
MULTI_INTERRUPT_SECONDS = 20.0


@dataclass(slots=True)
class DisclosureReport:
    """Where the disclosure was placed, so the placement can be checked."""

    original_s: float
    result_s: float
    positions_s: list[float] = field(default_factory=list)
    text: str = ""

    @property
    def overhead_s(self) -> float:
        return self.result_s - self.original_s

    def as_dict(self) -> dict[str, object]:
        return {
            "original_s": round(self.original_s, 2),
            "result_s": round(self.result_s, 2),
            "overhead_s": round(self.overhead_s, 2),
            "positions_s": [round(p, 2) for p in self.positions_s],
            "text": self.text,
        }


def interior_positions(
    duration_s: float,
    count: int | None = None,
    rng: secrets.SystemRandom | None = None,
) -> list[float]:
    """Random interior points to interrupt at, in seconds, sorted.

    Randomised with :mod:`secrets` rather than :mod:`random`: the whole value of
    the interior placement is that its offset cannot be predicted, and a seeded
    PRNG is predictable by definition.
    """
    usable = duration_s - 2 * EDGE_MARGIN_S
    if usable <= 1.0:
        return []
    if count is None:
        count = 1 if duration_s < MULTI_INTERRUPT_SECONDS else 2

    rng = rng or secrets.SystemRandom()
    # Spread them across the clip rather than letting two land together, which
    # would leave a long clean stretch - exactly what a trimmer wants.
    band = usable / count
    positions = [
        EDGE_MARGIN_S + i * band + rng.uniform(0.1 * band, 0.9 * band)
        for i in range(count)
    ]
    return sorted(positions)


def _nearest_quiet(audio: np.ndarray, sample_rate: int, at_s: float,
                   search_s: float = 0.6) -> int:
    """Move a cut to the quietest sample nearby, so it lands between words.

    Splicing mid-vowel gives a click, and a click is a seam an editor can find.
    A cut at the quietest point in the neighbourhood is both less audible and
    harder to spot.
    """
    centre = int(at_s * sample_rate)
    half = int(search_s * sample_rate)
    lo = max(0, centre - half)
    hi = min(audio.size, centre + half)
    if hi - lo < 2:
        return centre

    window = np.abs(audio[lo:hi])
    # Energy over 20 ms, so a single quiet sample inside a loud vowel does not
    # win against a genuine gap between words.
    frame = max(1, int(0.02 * sample_rate))
    if window.size >= frame:
        kernel = np.ones(frame) / frame
        smooth = np.convolve(window, kernel, mode="same")
        return lo + int(np.argmin(smooth))
    return centre


def add_disclosure(
    audio: np.ndarray,
    sample_rate: int,
    announcement: np.ndarray,
    announcement_rate: int | None = None,
    interior_count: int | None = None,
    text: str = DEFAULT_TEXT_IT,
    rng: secrets.SystemRandom | None = None,
) -> tuple[np.ndarray, DisclosureReport]:
    """Interleave *announcement* into *audio* at the start, end and inside.

    *announcement* should be synthesised in a different voice from the cloned
    one. Returns the audio and a report of where the interruptions landed.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    announcement = np.asarray(announcement, dtype=np.float32).reshape(-1)
    if announcement_rate not in (None, sample_rate):
        import soxr

        announcement = np.asarray(
            soxr.resample(announcement, announcement_rate, sample_rate, quality="VHQ"),
            dtype=np.float32,
        )

    pad = np.zeros(int(PAD_S * sample_rate), dtype=np.float32)
    marker = np.concatenate([pad, announcement, pad])
    duration_s = audio.size / sample_rate

    if audio.size == 0:
        return marker, DisclosureReport(0.0, marker.size / sample_rate, [], text)

    positions = interior_positions(duration_s, interior_count, rng)
    cuts = sorted({_nearest_quiet(audio, sample_rate, p) for p in positions})

    pieces: list[np.ndarray] = [marker]
    previous = 0
    placed: list[float] = [0.0]
    for cut in cuts:
        if cut <= previous:
            continue
        pieces.append(audio[previous:cut])
        pieces.append(marker)
        # Report where it lands in the *result*, which is what someone
        # inspecting the file will measure.
        placed.append(sum(p.size for p in pieces[:-1]) / sample_rate)
        previous = cut
    pieces.append(audio[previous:])
    tail_at = sum(p.size for p in pieces) / sample_rate
    pieces.append(marker)
    placed.append(tail_at)

    out = np.concatenate(pieces)
    report = DisclosureReport(
        original_s=duration_s,
        result_s=out.size / sample_rate,
        positions_s=placed,
        text=text,
    )
    log.info(
        "disclosure spoken %d times over %.1fs (was %.1fs)",
        len(placed), report.result_s, report.original_s,
    )
    return out, report
