"""Cloning a volunteer on stage, and deleting them afterwards.

The consent form promises two things: the recording stays on this machine, and
it is deleted at the end of the session. Until this existed, nothing in the
repository did the second one - the promise was in a document and nowhere else.

The other half is the swap. Someone gives fifteen seconds and the *next*
sentence is in their voice; a restart at that moment would end the
demonstration, whose whole point is that it takes fifteen seconds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _script():
    """Import the CLI without running it."""
    import importlib.util

    from parliamo.paths import REPO_ROOT

    path = REPO_ROOT / "scripts" / "clone_volunteer.py"
    spec = importlib.util.spec_from_file_location("clone_volunteer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# forgetting, which the consent form promises
# ---------------------------------------------------------------------------


def test_forget_removes_everything_for_one_person(tmp_path) -> None:
    module = _script()
    person = tmp_path / "maria"
    person.mkdir()
    (person / "raw.wav").write_bytes(b"\x00" * 2048)
    (person / "reference.wav").write_bytes(b"\x00" * 1024)
    (person / "consent.json").write_text('{"consent": "ref 007"}', encoding="utf-8")

    detail = module.forget(person)

    assert detail["files"] == 3
    assert detail["gone"] is True
    assert not person.exists(), "the recording is still on disk"


def test_forget_reports_what_it_removed(tmp_path) -> None:
    """The operator has to be able to tell the volunteer it is done."""
    module = _script()
    person = tmp_path / "luca"
    person.mkdir()
    (person / "raw.wav").write_bytes(b"\x00" * 5000)

    detail = module.forget(person)
    assert detail["name"] == "luca"
    assert detail["bytes"] == 5000


def test_forget_removes_nested_files(tmp_path) -> None:
    module = _script()
    person = tmp_path / "anna"
    (person / "clips").mkdir(parents=True)
    (person / "clips" / "one.wav").write_bytes(b"\x00" * 100)
    (person / "raw.wav").write_bytes(b"\x00" * 100)

    detail = module.forget(person)
    assert detail["files"] == 2
    assert not person.exists()


def test_forget_on_a_missing_person_is_not_a_crash(tmp_path) -> None:
    module = _script()
    detail = module.forget(tmp_path / "nobody")
    assert detail["files"] == 0
    assert detail["gone"] is True


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def test_slug_survives_accents_and_spaces() -> None:
    module = _script()
    assert module.slug("Maria Rossi") == "Maria-Rossi"
    assert module.slug("Anna/../etc") == "Anna----etc"


def test_slug_cannot_escape_the_directory() -> None:
    """A name is typed in a hurry on stage; it must not become a path."""
    module = _script()
    for hostile in ("../../etc/passwd", "..\\..\\windows", "/absolute"):
        assert "/" not in module.slug(hostile)
        assert "\\" not in module.slug(hostile)


def test_slug_is_bounded() -> None:
    module = _script()
    assert len(module.slug("x" * 500)) <= 40


# ---------------------------------------------------------------------------
# the swap, and the consent check on it
# ---------------------------------------------------------------------------


def _translator():
    from parliamo.pipeline.transcriber import LiveTranscriber
    from parliamo.pipeline.translator import LiveTranslator

    class _MT:
        source_lang = "tr"
        loaded = True

        def load(self) -> None: ...
        def warmup(self) -> None: ...
        def unload(self) -> None: ...

        def translate(self, text, source_lang=None, target_lang=None):
            from parliamo.mt.base import Translation

            return Translation(text="IT", source=text, source_lang="tr",
                               target_lang="it", backend="fake")

    class _TTS:
        sample_rate = 24000
        loaded = True

        def load(self) -> None: ...
        def warmup(self) -> None: ...
        def unload(self) -> None: ...

        def speak(self, text, language=None, voice=None):
            from parliamo.tts.base import Speech

            return Speech(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000,
                          text=text, language="it", backend="fake")

    class _Backend:
        loaded = True
        model = "fake"

        def load(self) -> None: ...
        def unload(self) -> None: ...
        def warmup(self, seconds: float = 1.0) -> float:
            return 0.0

        def transcribe(self, audio, sample_rate=16000, language=None):  # pragma: no cover
            from parliamo.asr.base import Transcript

            return Transcript(text="x", backend="fake")

    return LiveTranslator(LiveTranscriber(_Backend()), _MT(), _TTS(),
                          playback=None, speak=True)


def test_a_voice_can_be_swapped_in_mid_talk(tmp_path) -> None:
    reference = tmp_path / "volunteer.wav"
    reference.write_bytes(b"\x00" * 64)

    translator = _translator()
    translator.set_reference_voice(str(reference), consent="signed, ref 007")
    assert translator.reference_voice == str(reference)


def test_a_swap_without_consent_is_refused(tmp_path) -> None:
    """The moment most likely to skip the check is the moment on stage."""
    reference = tmp_path / "volunteer.wav"
    reference.write_bytes(b"\x00" * 64)

    translator = _translator()
    with pytest.raises(ValueError, match="consent"):
        translator.set_reference_voice(str(reference), consent="")
    assert translator.reference_voice is None, "the voice loaded despite the refusal"


def test_whitespace_is_not_a_consent_record(tmp_path) -> None:
    reference = tmp_path / "volunteer.wav"
    reference.write_bytes(b"\x00" * 64)
    translator = _translator()
    with pytest.raises(ValueError):
        translator.set_reference_voice(str(reference), consent="   ")


def test_the_voice_can_be_cleared_back_to_generic(tmp_path) -> None:
    reference = tmp_path / "volunteer.wav"
    reference.write_bytes(b"\x00" * 64)
    translator = _translator()
    translator.set_reference_voice(str(reference), consent="ref 007")
    translator.set_reference_voice(None)
    assert translator.reference_voice is None


# ---------------------------------------------------------------------------
# over HTTP, which is how the stage flow reaches the pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from parliamo.ui.server import SubtitleServer

    calls: list[tuple] = []

    def on_voice(path, consent):
        # Mirrors LiveTranslator.set_reference_voice: clearing back to the
        # generic voice needs no consent, because it removes a clone rather
        # than creating one. On stage "get that voice off, now" has to work.
        if path is not None and not consent.strip():
            raise ValueError("voice profile has no consent record")
        calls.append((path, consent))

    server = SubtitleServer(on_voice=on_voice)
    with TestClient(server.build_app()) as c:
        yield c, server, calls


def test_posting_a_voice_reaches_the_pipeline(client) -> None:
    c, server, calls = client
    response = c.post("/voice", json={"path": "data/voices/x.wav", "consent": "ref 007"})
    assert response.status_code == 200
    assert calls == [("data/voices/x.wav", "ref 007")]
    assert server.state.voice == "x"


def test_posting_without_consent_is_refused_with_a_reason(client) -> None:
    c, _, calls = client
    response = c.post("/voice", json={"path": "data/voices/x.wav"})
    assert response.status_code == 400
    assert "consent" in response.json()["error"]
    assert calls == [], "the pipeline was told to load it anyway"


def test_clearing_the_voice_over_http(client) -> None:
    c, server, _ = client
    c.post("/voice", json={"path": "data/voices/x.wav", "consent": "ref 007"})
    c.post("/voice", json={"path": None})
    assert server.state.voice == "generic"


def test_voice_endpoint_without_a_pipeline_says_so() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from parliamo.ui.server import SubtitleServer

    with TestClient(SubtitleServer().build_app()) as c:
        assert c.post("/voice", json={"path": "x.wav", "consent": "y"}).status_code == 503


def test_the_current_voice_is_visible_to_the_operator(client) -> None:
    """The operator has to know whether a volunteer's recording is still loaded."""
    c, _, _ = client
    assert c.get("/state").json()["voice"] == "generic"
    c.post("/voice", json={"path": "data/voices/maria.wav", "consent": "ref 007"})
    assert c.get("/state").json()["voice"] == "maria"


# ---------------------------------------------------------------------------
# the consent record lives with the audio
# ---------------------------------------------------------------------------


def test_consent_json_shape(tmp_path) -> None:
    """Whatever the format, it must carry who, what and when - so a recording
    found later can be traced to a signature."""
    record = {
        "name": "Maria",
        "consent": "signed 2026-09-14, ref 007",
        "recorded": "2026-09-14T18:03:00+02:00",
        "seconds": 15.0,
        "delete_after": "the end of this session",
    }
    path = tmp_path / "consent.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["consent"]
    assert loaded["name"]
    assert loaded["recorded"]
