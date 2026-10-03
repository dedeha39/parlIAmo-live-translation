"""Whisper reciting its hotword prompt must not be spoken.

Every "recital" below is a transcript the pipeline translated and spoke at the
2026-09-27 and 2026-10-02 rehearsals, read from the run logs: with the
microphone about -40 dBFS, Whisper answered speech it could not read with the
list it is biased toward. Every "speech" below must survive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from parliamo.asr.plausibility import hotword_echo

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def hotwords() -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from live_translate import load_hotwords  # type: ignore[import-not-found]

    words = load_hotwords(str(ROOT / "config" / "hotwords.tr.txt"), "tr")
    assert words and words.startswith("İpek Nur Yıldız Acme")
    return words


RECITALS = [
    "İpek Nur Yıldız Acme",                                     # 13 times
    "İpek Nur Yıldız Acme Valdastra",
    "İpek Nur Yıldız Acme Valdastra Friuli",                    # 33 times
    "İpek Nur Yıldız Acme Valdastra Friuli Yeni",
    "İpek Nur Yıldız Acme Valdastra Fırmızı İpek Nur Yıldız Acme Valdastra Fırmızı",
    "İpek Nur Yıldız Acme İpek Nur Yıldız Acme İpek",
    "İpek Nur Yıldız Acme Valdastra Friuli Hong Kong Crosetti",
    "Acme Valdastra Friuli Hong Kong Crosetto Ferrari Gemini",
    "İpek Nur Yıldız Acme Valdastra Friuli Hong Kong Crosetto İpek Nur Yıldız "
    "Acme Valdastra Friuli Hong Kong Crosetto İpek",
]

SPEECH = [
    "İpek Nur Yıldız.",                       # how the talk opens: three, not four
    "Ben İpek Nur Yıldız.",
    "Ben İpek Nur Yıldız. Bir yıldır Acme'nin teknik ofisinde çalışıyorum. "
    "Yapay zekâ üzerine çalışıyorum.",         # spoken at rehearsal, correctly
    "Ben İpek Nur Yıldız, Acme'de çalışıyorum.",  # four in order, four of seven
    "Yapay zekâ ile ses klonlama.",            # hotwords, but not in the list's order
    "Deepfake ses klonlama yapay zekâ.",
    "Valdastra'da, Acme'de.",
    "Hong Kong'da bir şirket yirmi beş milyon dolar kaybetti.",
    "Crosetto'nun sesi taklit edildi.",
]


@pytest.mark.parametrize("text", RECITALS)
def test_the_recital_is_recognised(text: str, hotwords: str) -> None:
    assert hotword_echo(text, hotwords)


@pytest.mark.parametrize("text", SPEECH)
def test_speech_with_the_same_names_is_kept(text: str, hotwords: str) -> None:
    assert not hotword_echo(text, hotwords)


def test_no_hotwords_no_judgement() -> None:
    assert not hotword_echo("İpek Nur Yıldız Acme Valdastra Friuli", None)
    assert not hotword_echo("İpek Nur Yıldız Acme Valdastra Friuli", "")


def test_the_recogniser_drops_the_recital_and_says_why(hotwords: str) -> None:
    """Long enough to pass the rate check - which is how they got through."""
    import types

    import numpy as np

    from parliamo.asr.faster_whisper_backend import FasterWhisperBackend

    text = "İpek Nur Yıldız Acme Valdastra Friuli"
    backend = FasterWhisperBackend(device="cpu", hotwords=hotwords)
    segment = types.SimpleNamespace(start=0.0, end=4.0, text=text, no_speech_prob=0.1,
                                    avg_logprob=-0.3)
    backend._model = types.SimpleNamespace(
        transcribe=lambda *a, **k: (iter([segment]),
                                    types.SimpleNamespace(language="tr", language_probability=0.9)))
    backend._loaded = True

    transcript = backend.transcribe(np.zeros(int(4.5 * 16000), dtype=np.float32))
    assert transcript.text == ""
    assert "hotword" in transcript.dropped_segments[-1]["reason"]

    backend.hotwords = None
    assert backend.transcribe(np.zeros(int(4.5 * 16000), dtype=np.float32)).text == text, (
        "without a hotword prompt there is nothing to recite - nothing is dropped")
