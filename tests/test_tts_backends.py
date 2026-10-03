"""Synthesis backends: language handling, consent, and the honest refusals.

The behaviour worth pinning here is what each backend does when asked for
something it cannot do. A synthesiser that quietly substitutes a language, or
returns audio in the wrong voice while a caller believes cloning happened, is
worse than one that raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from parliamo.tts import VoiceProfile, available_backends, create_backend
from parliamo.tts.base import Speech, TTSBackend
from parliamo.tts.kokoro_backend import KokoroBackend


class FakeBackend(TTSBackend):
    """A synthesiser that returns silence, for testing the base contract."""

    name = "fake"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.calls: list[tuple[str, str, str | None]] = []

    def _load(self) -> None:
        pass

    def _unload(self) -> None:
        pass

    def _synthesise(self, text, language, voice):
        self.calls.append((text, language, voice.name if voice else None))
        return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_kokoro_is_registered() -> None:
    assert "kokoro" in available_backends()


def test_unknown_backend_lists_the_alternatives() -> None:
    with pytest.raises(KeyError, match="Available"):
        create_backend("nonexistent")


# ---------------------------------------------------------------------------
# the base contract
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty_audio_not_an_error() -> None:
    backend = FakeBackend()
    speech = backend.speak("   ")
    assert speech.audio.size == 0
    assert speech.duration_s == 0.0
    assert backend.calls == []


def test_speech_reports_rtf() -> None:
    speech = Speech(
        audio=np.zeros(24000, dtype=np.float32), sample_rate=24000,
        text="x", language="it", compute_s=0.5,
    )
    assert speech.duration_s == 1.0
    assert speech.rtf == pytest.approx(0.5)


def test_rtf_of_empty_audio_is_infinite_not_a_crash() -> None:
    speech = Speech(audio=np.zeros(0, dtype=np.float32), sample_rate=24000,
                    text="", language="it", compute_s=0.1)
    assert speech.rtf == float("inf")


def test_unknown_voice_name_is_rejected() -> None:
    backend = FakeBackend()
    with pytest.raises(KeyError, match="unknown voice"):
        backend.speak("ciao", voice="nobody")


# ---------------------------------------------------------------------------
# consent
# ---------------------------------------------------------------------------


def test_voice_without_consent_cannot_be_registered() -> None:
    """The thing this project warns people about must not be doable silently.

    A cloned voice carries its authorisation with it, or it does not get
    registered at all.
    """
    backend = FakeBackend()
    profile = VoiceProfile(name="volunteer", reference_path="x.wav")
    with pytest.raises(ValueError, match="no consent record"):
        backend.register_voice(profile)
    assert backend.voices == []


def test_voice_with_consent_is_accepted() -> None:
    backend = FakeBackend()
    backend.register_voice(
        VoiceProfile(name="presenter", reference_path="x.wav",
                     consent="signed 2026-08-31, ref 001")
    )
    assert backend.voices == ["presenter"]


def test_whitespace_is_not_a_consent_record() -> None:
    backend = FakeBackend()
    with pytest.raises(ValueError, match="no consent record"):
        backend.register_voice(
            VoiceProfile(name="v", reference_path="x.wav", consent="   ")
        )


# ---------------------------------------------------------------------------
# Kokoro language handling
# ---------------------------------------------------------------------------


def test_italian_maps_to_kokoro_code() -> None:
    code, note = KokoroBackend().resolve_language("it")
    assert code == "i"
    assert note == ""


def test_friulian_falls_back_to_italian_and_says_so() -> None:
    """The substitution is real and must be surfaced, not swallowed."""
    code, note = KokoroBackend().resolve_language("fur")
    assert code == "i"
    assert "phonetics are wrong" in note


def test_turkish_is_refused_rather_than_mispronounced() -> None:
    """Kokoro has no Turkish voicepack.

    Turkish is the *stage* language, not an output language, so this never
    arises in normal use. If it ever does, a clear failure beats Italian
    phonetics applied to Turkish text.
    """
    with pytest.raises(KeyError, match="no voice for"):
        KokoroBackend().resolve_language("tr")


def test_unknown_language_is_refused() -> None:
    with pytest.raises(KeyError, match="Supported"):
        KokoroBackend().resolve_language("xx")


def test_kokoro_refuses_to_pretend_it_cloned() -> None:
    """Returning a default voice when asked for a cloned one would be a lie."""
    backend = KokoroBackend()
    backend._loaded = True  # skip the real model load
    profile = VoiceProfile(name="p", reference_path="x.wav", consent="ref 001")
    with pytest.raises(NotImplementedError, match="cannot clone"):
        backend._synthesise("ciao", "it", profile)


def test_kokoro_describes_itself_as_non_cloning() -> None:
    info = KokoroBackend().describe()
    assert info["can_clone"] is False
    assert info["backend"] == "kokoro"


# ---------------------------------------------------------------------------
# real synthesis
# ---------------------------------------------------------------------------


@pytest.mark.model
@pytest.mark.slow
def test_kokoro_synthesises_italian_fast() -> None:
    """The property the whole architecture rests on: RTF well under 1."""
    from parliamo.paths import configure_model_cache

    configure_model_cache()
    try:
        backend = create_backend("kokoro", device="cuda", language="it")
        backend.load()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Kokoro unavailable: {exc}")

    try:
        backend.warmup()
        speech = backend.speak(
            "Questo sistema funziona interamente su questo computer portatile.",
            language="it",
        )
        assert speech.audio.size > 0
        assert np.isfinite(speech.audio).all()
        assert 2.0 < speech.duration_s < 10.0
        # Measured at 0.02; assert an order of magnitude of headroom so this
        # fails loudly if a future change makes synthesis the bottleneck again.
        assert speech.rtf < 0.3, f"RTF {speech.rtf:.3f} - synthesis got slow"
    finally:
        backend.unload()


@pytest.mark.model
@pytest.mark.slow
def test_kokoro_speaks_friulian_as_italian_and_labels_it() -> None:
    from parliamo.paths import configure_model_cache

    configure_model_cache()
    try:
        backend = create_backend("kokoro", device="cuda", language="it")
        backend.load()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Kokoro unavailable: {exc}")
    try:
        speech = backend.speak("Bundì, o soi ca.", language="fur")
        assert speech.audio.size > 0
        assert "spoken as it" in speech.language
    finally:
        backend.unload()


def test_the_configured_kokoro_voice_is_honoured(monkeypatch) -> None:
    """The regression this guards.

    The voicepack lookup tried the language default first and the configured
    voice only for a language without one - which is no language - so
    `tts.voice` was silently ignored and every Italian sentence came out as
    if_sara whatever the config said. Found when a male source was wanted
    for RVC, which keeps the source pitch, and im_nicola measured 216 Hz.
    """
    from parliamo.tts.kokoro_backend import KokoroBackend

    seen: dict = {}

    class _Pipeline:
        def __call__(self, text, voice, speed):
            seen["voice"] = voice
            yield None, None, np.zeros(2400, dtype=np.float32)

    backend = KokoroBackend(model="hexgrad/Kokoro-82M", device="cpu", language="it",
                            voice="im_nicola")
    backend._loaded = True
    monkeypatch.setattr(backend, "_pipeline", lambda language: _Pipeline())
    backend._synthesise("Ciao.", "it", None)
    assert seen["voice"] == "im_nicola"


def test_a_voice_from_another_language_falls_back_with_a_warning(monkeypatch, caplog) -> None:
    """An English voicepack cannot drive the Italian frontend."""
    import logging

    from parliamo.tts.kokoro_backend import KokoroBackend

    seen: dict = {}

    class _Pipeline:
        def __call__(self, text, voice, speed):
            seen["voice"] = voice
            yield None, None, np.zeros(2400, dtype=np.float32)

    backend = KokoroBackend(model="hexgrad/Kokoro-82M", device="cpu", language="it",
                            voice="af_heart")
    backend._loaded = True
    monkeypatch.setattr(backend, "_pipeline", lambda language: _Pipeline())
    with caplog.at_level(logging.WARNING):
        backend._synthesise("Ciao.", "it", None)
    assert seen["voice"] == "if_sara"
    assert "does not belong to it" in caplog.text
