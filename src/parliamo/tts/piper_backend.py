"""Piper: speech for the languages Kokoro has no voice for.

Kokoro-82M speaks Italian, Spanish, English, French, Portuguese, Hindi,
Japanese and Mandarin - not German, and not Turkish. Piper (VITS, one ONNX file
per voice) has both, runs on the CPU and takes no GPU memory, which on an 8 GB
card beside the recogniser, the translator and a voice service is the
constraint that decides everything.

Measured on this laptop's CPU (i9-13900HX), 2026-09-25:

    de_DE-thorsten-high   5.39 s of audio in 0.69 s   RTF 0.13   median F0 125 Hz
    tr_TR-dfki-medium     5.14 s of audio in 0.75 s   RTF 0.15   median F0 105 Hz

Slower than Kokoro on the GPU (~0.2 s a sentence), faster than anything that
would have to share the GPU. Both are male voices near the presenter's 139 Hz,
so the voice service has little pitch to move.

The voices are files under ``models/piper``, from ``rhasspy/piper-voices``:
``<name>.onnx`` and ``<name>.onnx.json``. A language whose voice is not on disk
is not offered - the page shows it as subtitles only rather than failing at
Start. Thorsten's dataset is CC0; the Piper runtime (``piper-tts``) is GPL-3.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import TTSBackend, VoiceProfile

log = logging.getLogger(__name__)

#: The voice used for each language unless the config names another.
VOICES = {
    "de": "de_DE-thorsten-high",
    "tr": "tr_TR-dfki-medium",
}

DEFAULT_VOICES_DIR = "models/piper"


def voice_path(name: str, voices_dir: str | Path = DEFAULT_VOICES_DIR) -> Path:
    from ..paths import resolve

    return Path(resolve(voices_dir)) / f"{name}.onnx"


def available_languages(voices_dir: str | Path = DEFAULT_VOICES_DIR) -> set[str]:
    """Languages whose default voice is on disk, model and config both."""
    ready = set()
    for language, name in VOICES.items():
        path = voice_path(name, voices_dir)
        if path.exists() and path.with_suffix(".onnx.json").exists():
            ready.add(language)
    return ready


def _voice_language(name: str) -> str:
    """``de_DE-thorsten-high`` -> ``de``."""
    return name.split("_", 1)[0].lower()


class PiperBackend(TTSBackend):
    name = "piper"
    sample_rate = 22050

    def __init__(
        self,
        model: str = "piper",
        device: str = "cpu",
        language: str = "de",
        voice: str | None = None,
        speed: float = 1.0,
        voices_dir: str | Path = DEFAULT_VOICES_DIR,
        **kwargs: Any,
    ) -> None:
        # Piper runs on the CPU whatever the config says for Kokoro: its GPU
        # path needs onnxruntime-gpu, which would replace the CPU runtime the
        # VAD shares.
        super().__init__(model=model, device="cpu", language=language, **kwargs)
        self.voice = voice
        self.speed = speed
        self.voices_dir = voices_dir
        self._loaded_voices: dict[str, Any] = {}

    def _voice_name(self, language: str) -> str:
        # A configured voice is used only for its own language: a German voice
        # reading Turkish would be worse than the Turkish default.
        if self.voice and _voice_language(self.voice) == language:
            return self.voice
        if language not in VOICES:
            raise KeyError(
                f"Piper has no voice configured for {language!r}. "
                f"Configured: {', '.join(sorted(VOICES))}"
            )
        return VOICES[language]

    #: onnxruntime threads for synthesis. Its default is every core, and on
    #: this hybrid CPU (8 performance + 16 efficiency cores) that was the
    #: slowest setting measured: one German sentence took 2.0-3.1 s with the
    #: default, 1.37-1.43 s with 8 threads, 1.30-1.37 with 12, 11.7 with 1.
    #: Eight leaves the other cores to the recogniser's and the VAD's threads.
    threads: int = 8

    def _open(self, path: Path) -> Any:
        """Load a voice with our own session options, not Piper's defaults."""
        import json

        import onnxruntime as ort
        import piper.voice as piper_voice
        from piper import PiperVoice
        from piper.config import PiperConfig

        config = PiperConfig.from_dict(
            json.loads(path.with_suffix(".onnx.json").read_text(encoding="utf-8")))
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        providers: list[Any] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        log.info("Piper voice %s on %s", path.stem, session.get_providers()[0])
        return PiperVoice(config=config, session=session,
                          espeak_data_dir=Path(piper_voice.ESPEAK_DATA_DIR),
                          download_dir=Path.cwd())

    def _voice(self, language: str) -> Any:
        name = self._voice_name(language)
        if name not in self._loaded_voices:
            path = voice_path(name, self.voices_dir)
            if not path.exists():
                raise FileNotFoundError(
                    f"Piper voice {name} is not on disk at {path}. Download "
                    f"{name}.onnx and {name}.onnx.json from rhasspy/piper-voices."
                )
            self._loaded_voices[name] = self._open(path)
        return self._loaded_voices[name]

    def _load(self) -> None:
        voice = self._voice(self.language)
        self.sample_rate = int(voice.config.sample_rate)

    def _unload(self) -> None:
        self._loaded_voices.clear()

    def _synthesise(self, text: str, language: str, voice: VoiceProfile | None) -> np.ndarray:
        if voice is not None:
            raise NotImplementedError(
                "Piper cannot clone a voice. Generate here and convert "
                "afterwards - see parliamo.tts.conversion."
            )
        piper_voice = self._voice(language)
        syn_config = None
        if self.speed != 1.0:
            from piper import SynthesisConfig

            # Piper's knob is the length of each phoneme, the inverse of speed.
            syn_config = SynthesisConfig(length_scale=1.0 / self.speed)
        chunks = [
            np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
            for chunk in piper_voice.synthesize(text, syn_config=syn_config)
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def describe(self) -> dict[str, Any]:
        info = super().describe()
        info["voice"] = self._voice_name(self.language) if self.language in VOICES else None
        info["can_clone"] = False
        return info
