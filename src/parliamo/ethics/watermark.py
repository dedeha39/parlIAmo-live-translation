"""Detecting an audio watermark, and being honest about what that proves.

Chatterbox watermarks every output by default, using Resemble's **Perth**: an
imperceptible pattern embedded in the waveform that a detector can recover
afterwards. It is the reason Chatterbox is kept in this project at all, having
been replaced on the live path by Kokoro and Seed-VC (ADR 0006).

The demonstration this exists for is a comparison, not a reassurance:

1. Clone a voice with Chatterbox. Recover the watermark. It is there.
2. Clone the same voice with the pipeline that actually ships here - Kokoro
   generating, Seed-VC applying identity. Look for a watermark. There is none.

Both take fifteen seconds of reference audio. Only one of them marks its
output, and **the unmarked one is the one running on this laptop**. That is the
point: watermarking is a property of some tools, not of the technology, and an
attacker picks the tool.

What a watermark is and is not
------------------------------
It is **evidence**, not protection. It survives ordinary handling - re-encoding,
a bit of noise, being played through a phone - which is what makes it useful
against casual re-sharing. It does not survive someone who does not want it to,
and it was never in the file if the generator did not put it there.

So the honest framing on stage is: a watermark tells you *this was made by a
tool that marks its output*. Silence from the detector tells you **nothing at
all**.

The EU AI Act, Article 50, requires providers of generative systems to mark
outputs as artificial. In force since 2 August 2026; the watermarking grace
period ends 2 December 2026. That obligation binds providers, not someone
demonstrating on a laptop - and this module is a demonstration of exactly what
that obligation is worth and where it stops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Recovered-bit agreement above which we call a watermark present. Perth
#: returns a bit sequence; noise recovers roughly half of it by chance, so the
#: threshold sits well above chance rather than at it.
DETECTION_THRESHOLD = 0.75


@dataclass(slots=True)
class WatermarkReport:
    """What the detector found, and what that is worth saying about it."""

    present: bool
    confidence: float
    #: Whatever the detector handed back, kept so a sceptical audience member
    #: can be shown the raw output rather than a boolean.
    raw: Any = None
    error: str = ""

    @property
    def verdict(self) -> str:
        if self.error:
            return f"could not check: {self.error}"
        if self.present:
            return f"watermark found ({self.confidence:.0%} of bits recovered)"
        # Phrased deliberately. "No watermark" invites the audience to hear
        # "not synthetic", which is the opposite of true.
        return (
            f"no watermark ({self.confidence:.0%} recovered, chance is ~50%) - "
            "which means only that whatever made this did not mark it"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "confidence": round(self.confidence, 4),
            "verdict": self.verdict,
            "error": self.error,
        }


def _watermarker() -> Any:
    from perth import PerthImplicitWatermarker

    return PerthImplicitWatermarker()


def detect(audio: np.ndarray, sample_rate: int) -> WatermarkReport:
    """Look for a Perth watermark in *audio*.

    Never raises. A detector that throws in front of an audience is worse than
    one that says it could not tell.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return WatermarkReport(present=False, confidence=0.0, error="empty audio")

    try:
        recovered = _watermarker().get_watermark(audio, sample_rate=sample_rate)
    except Exception as exc:  # pragma: no cover - detector failures are data
        log.warning("watermark detection failed: %s", exc)
        return WatermarkReport(
            present=False, confidence=0.0, error=f"{type(exc).__name__}: {exc}"
        )

    bits = np.asarray(recovered).reshape(-1)
    if bits.size == 0:
        return WatermarkReport(present=False, confidence=0.0, raw=recovered)

    # Perth embeds a fixed pattern of ones. Recovering nearly all of them is
    # the signal; recovering about half is what noise does on its own, so the
    # confidence is reported as-is rather than rescaled to look decisive.
    confidence = float(np.mean(bits))
    return WatermarkReport(
        present=confidence >= DETECTION_THRESHOLD,
        confidence=confidence,
        raw=recovered,
    )


def is_watermarked(audio: np.ndarray, sample_rate: int) -> bool:
    return detect(audio, sample_rate).present


def survives(
    audio: np.ndarray,
    sample_rate: int,
    transform: Any,
    label: str = "",
) -> tuple[str, WatermarkReport]:
    """Apply *transform* to the audio and re-check. Returns ``(label, report)``.

    The interesting half of the demonstration. A watermark that vanishes when
    the file is re-encoded protects nobody, and one that survives being played
    through a phone speaker and recorded again is doing real work. Both claims
    should be shown rather than asserted.
    """
    try:
        altered = transform(np.asarray(audio, dtype=np.float32).reshape(-1))
    except Exception as exc:  # pragma: no cover
        return label, WatermarkReport(present=False, confidence=0.0, error=str(exc))
    return label, detect(altered, sample_rate)
