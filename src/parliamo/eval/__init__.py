"""Evaluation: metrics, normalisation and public test sets.

Kept separate from the runtime pipeline on purpose. Nothing in
:mod:`parliamo.eval` is imported by the live system, so heavyweight evaluation
dependencies (``datasets``, ``jiwer``) never load on stage.
"""

from .normalize import (
    NormalizerOptions,
    TurkishNormalizer,
    get_normalizer,
    number_to_turkish,
    turkish_lower,
    turkish_upper,
)

__all__ = [
    "NormalizerOptions",
    "TurkishNormalizer",
    "get_normalizer",
    "number_to_turkish",
    "turkish_lower",
    "turkish_upper",
]
