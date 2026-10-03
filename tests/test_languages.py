"""Spanish, German and Turkish as spoken targets, and what each language needs.

Kokoro speaks Italian and Spanish on the GPU; Piper speaks German and Turkish on
the CPU. Each voice sits at its own pitch, so the voice service is told the
shift for the voice it is converting, and a hotword list written for one
language is not applied to another.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _live_translate():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("live_translate_under_test",
                                                  SCRIPTS / "live_translate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# which synthesiser speaks which language
# ---------------------------------------------------------------------------


def test_kokoro_keeps_the_languages_it_has(monkeypatch) -> None:
    from parliamo.tts import backend_for

    monkeypatch.setattr("parliamo.tts.piper_backend.available_languages", lambda *a: {"de", "tr"})
    assert backend_for("it") == "kokoro" and backend_for("es") == "kokoro"
    assert backend_for("fur") == "kokoro", "Friulian is still spoken with the Italian frontend"


def test_piper_speaks_german_and_turkish_only_when_their_voices_are_on_disk(monkeypatch) -> None:
    from parliamo.tts import backend_for

    monkeypatch.setattr("parliamo.tts.piper_backend.available_languages", lambda *a: {"de", "tr"})
    assert backend_for("de") == "piper" and backend_for("tr") == "piper"
    monkeypatch.setattr("parliamo.tts.piper_backend.available_languages", lambda *a: set())
    assert backend_for("de") is None, "no voice on disk: subtitles only, not a failure at Start"


def test_a_piper_voice_is_used_only_for_its_own_language() -> None:
    from parliamo.tts.piper_backend import PiperBackend

    backend = PiperBackend(language="tr", voice="de_DE-thorsten-high")
    assert backend._voice_name("de") == "de_DE-thorsten-high"
    assert backend._voice_name("tr") == "tr_TR-dfki-medium", "a German voice must not read Turkish"
    with pytest.raises(KeyError):
        backend._voice_name("it")


@pytest.mark.skipif(not (ROOT / "models/piper/de_DE-thorsten-high.onnx").exists(),
                    reason="German Piper voice not downloaded")
def test_german_is_spoken() -> None:
    from parliamo.tts import create_backend

    tts = create_backend("piper", language="de")
    speech = tts.speak("Legen Sie auf und rufen Sie zurück.", language="de")
    assert speech.sample_rate == 22050 and speech.duration_s > 1.0
    assert np.abs(speech.audio).max() > 0.05


# ---------------------------------------------------------------------------
# hotwords belong to one language
# ---------------------------------------------------------------------------


def test_the_turkish_hotwords_are_not_applied_to_italian_speech(tmp_path, monkeypatch) -> None:
    module = _live_translate()
    (tmp_path / "hotwords.tr.txt").write_text("# names\nFriulian\nklonlama\n", encoding="utf-8")
    monkeypatch.setattr(module, "resolve", lambda p: Path(p))
    path = str(tmp_path / "hotwords.tr.txt")
    assert module.load_hotwords(path, "tr") == "Friulian klonlama"
    assert module.load_hotwords(path, "it") is None
    (tmp_path / "hotwords.it.txt").write_text("Friuli\n", encoding="utf-8")
    assert module.load_hotwords(path, "it") == "Friuli", "a list for the language is used when it exists"
    assert module.load_hotwords(path) == "Friulian klonlama", "callers naming no language are unchanged"


# ---------------------------------------------------------------------------
# pitch: the shift is measured from the voice being converted
# ---------------------------------------------------------------------------


def _voice_at(hz: float, rate: int = 24000, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (0.3 * np.sin(2 * np.pi * hz * t) + 0.1 * np.sin(2 * np.pi * 2 * hz * t)).astype(np.float32)


def test_pitch_is_measured_and_turned_into_semitones() -> None:
    from parliamo.audio.pitch import median_f0, semitones

    assert abs(median_f0(_voice_at(223.0), 24000) - 223.0) < 5
    assert round(semitones(223.0, 139.0), 1) == -8.2, "if_sara to the presenter: the -8 found by ear"
    assert round(semitones(139.0, 139.0), 1) == 0.0


class _TTS:
    sample_rate = 24000

    def __init__(self, hz: float) -> None:
        self.hz = hz

    def speak(self, text, language=None):
        return type("S", (), {"audio": _voice_at(self.hz), "sample_rate": 24000})()


class _Converter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def convert(self, audio, sample_rate, reference_path, diffusion_steps=None, **options):
        self.calls.append(options)
        return type("C", (), {"audio": audio, "sample_rate": sample_rate, "round_trip_s": 0.1,
                              "duration_s": audio.size / sample_rate})()


def _translator(tts, converter, presenter_f0):
    from parliamo.pipeline.translator import LiveTranslator

    transcriber = type("T", (), {"on_transcript": None})()
    mt = type("M", (), {"source_lang": "tr"})()
    translator = LiveTranslator(transcriber, mt, tts, converter=converter,
                                reference_voice="ref.wav", target_lang="it")
    translator.presenter_f0_hz = presenter_f0
    return translator


def test_the_voice_service_is_told_the_shift_for_the_voice_it_converts() -> None:
    """Kokoro's Italian woman at 223 Hz needs -8; Piper's German man at 125 Hz needs +2."""
    for hz, expected in ((223.0, -8.2), (125.0, 1.8)):
        converter = _Converter()
        _translator(_TTS(hz), converter, 139.0)._warm_conversion()
        assert abs(converter.calls[0]["pitch"] - expected) < 0.4, (hz, converter.calls)


def test_without_the_presenter_pitch_the_service_keeps_its_own() -> None:
    converter = _Converter()
    _translator(_TTS(223.0), converter, None)._warm_conversion()
    assert len(converter.calls) == 1
    assert "pitch" not in converter.calls[0], "no pitch sent: the server's --pitch applies"


def test_the_client_puts_the_pitch_in_the_request() -> None:
    from parliamo.tts import conversion

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    seen: list[dict] = []

    def serve() -> None:
        conn, _ = server.accept()
        header, audio = conversion.recv_message(conn)
        seen.append(header)
        conversion.send_message(conn, {"sample_rate": 48000}, audio)
        conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        client = conversion.VoiceConverter(port=server.getsockname()[1], timeout=5)
        client.convert(np.zeros(160, np.float32), 24000, "ref.wav", pitch=-8.24)
        thread.join(timeout=5)
    finally:
        server.close()
    assert seen[0]["pitch"] == -8.24


# ---------------------------------------------------------------------------
# the invented-sentence filter counts German and Spanish syllables
# ---------------------------------------------------------------------------


def test_german_and_spanish_vowels_are_counted() -> None:
    from parliamo.asr.plausibility import syllables

    assert syllables("Mädchen", "de") == 2
    assert syllables("está aquí", "es") == 4
