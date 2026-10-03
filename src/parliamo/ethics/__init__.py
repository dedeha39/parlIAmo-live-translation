"""The part of the talk that the audience can act on.

Cloning a voice is the demonstration; noticing that a voice was cloned is the
only part anyone takes home. This package holds the detection side.
"""

from .watermark import (
    WatermarkReport,
    detect,
    is_watermarked,
    survives,
)

__all__ = ["WatermarkReport", "detect", "is_watermarked", "survives"]
