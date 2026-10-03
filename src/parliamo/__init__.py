"""parlIAmo - fully offline real-time speech translation on a single laptop.

The package is organised as one module per pipeline stage:

    parliamo.audio     microphone capture, playback, device discovery, VAD
    parliamo.asr       speech -> source-language text
    parliamo.mt        source text -> target text
    parliamo.tts       target text -> speech
    parliamo.pipeline  the orchestrator that wires the stages together

Nothing in this package contacts the network at run time. Model weights are
fetched once by ``scripts/download_models.py`` and then read from disk.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
