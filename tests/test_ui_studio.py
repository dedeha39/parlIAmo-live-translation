"""The studio speaks typed sentences through OmniVoice - in consented voices,
invented ones, or the model's own - with every setting OmniVoice has."""

from __future__ import annotations

import base64
import io
import json
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from parliamo.tts.conversion import recv_message, send_message
from parliamo.tts.omnivoice_options import ADVANCED, DESIGN, clean_options, instruct_for
from parliamo.ui.studio import AUTO_VOICE, DESIGN_SAMPLE, Studio, StudioError


class _FakeOmniVoice(socketserver.BaseRequestHandler):
    seen: list[dict] = []

    def handle(self) -> None:
        header, _ = recv_message(self.request)
        empty = np.zeros(0, dtype=np.float32)
        op = header.get("op")
        if op == "ping":
            send_message(self.request, {"service": "omnivoice", "steps": 16}, empty)
            return
        type(self).seen.append(header)
        if op == "languages":
            send_message(self.request, {"languages": [["it", "Italian"], ["scn", "Sicilian"]]},
                         empty)
            return
        if op == "prepare":
            send_message(self.request, {"text": "Buongiorno a tutti.", "start": 0.0,
                                        "seconds": 3.2}, empty)
            return
        send_message(self.request, {"sample_rate": 24000, "compute_s": 0.9,
                                    "seed": header.get("seed", 4242)},
                     np.full(12000, 0.1, dtype=np.float32))


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture
def service():
    _FakeOmniVoice.seen = []
    srv = _Server(("127.0.0.1", 0), _FakeOmniVoice)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def _wav(path: Path, seconds: float = 0.1) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(int(24000 * seconds), dtype=np.float32), 24000)
    return str(path)


@pytest.fixture
def studio(tmp_path, service):
    voices_dir = tmp_path / "data" / "voices"
    mine = _wav(voices_dir / "phone-sentences.wav")
    stranger = _wav(voices_dir / "someone.wav")
    volunteer = _wav(voices_dir / "volunteers" / "elena" / "reference.wav")
    listed = [
        {"name": "phone-sentences", "path": mine, "consent": "presenter's own voice",
         "configured": True},
        {"name": "someone", "path": stranger, "consent": ""},
        {"name": "elena", "path": volunteer, "consent": "signed, ref 007", "volunteer": True},
    ]
    return Studio(lambda: listed, port=service, root=tmp_path), {
        "mine": mine, "stranger": stranger, "volunteer": volunteer}


# -- whose voice ----------------------------------------------------------------

def test_a_voice_without_consent_is_refused(studio) -> None:
    s, paths = studio
    with pytest.raises(StudioError, match="consent"):
        s.speak("Ciao a tutti.", "it", paths["stranger"])
    assert _FakeOmniVoice.seen == [], "nothing was sent to the model"


def test_only_consented_voices_are_offered(studio) -> None:
    s, _ = studio
    assert {v["name"] for v in s.usable_voices()} == {"phone-sentences", "elena"}


def test_the_models_own_voice_needs_no_recording_and_is_asked_for_by_name(studio) -> None:
    s, _ = studio
    clip = s.speak("Ciao.", "it", AUTO_VOICE)
    sent = _FakeOmniVoice.seen[-1]
    assert sent["auto"] is True and "reference_path" not in sent
    assert clip["auto"] and clip["voice"] == "auto"


# -- speaking ---------------------------------------------------------------------

def test_a_sentence_is_spoken_kept_and_listed(studio) -> None:
    s, paths = studio
    clip = s.speak("Nonna, sono io.", "it", paths["mine"], steps=16)
    sent = _FakeOmniVoice.seen[-1]
    assert (sent["text"], sent["language"], sent["options"]["num_step"]) == (
        "Nonna, sono io.", "it", 16)
    assert sent["reference_path"] == paths["mine"]
    assert clip["seconds"] == 0.5 and clip["voice"] == "phone-sentences"
    assert s.clip_file(clip["id"]) is not None
    assert [c["id"] for c in s.clips()] == [clip["id"]]
    assert s.delete(clip["id"]) and s.clips() == []


def test_settings_are_cleaned_before_they_reach_the_model(studio) -> None:
    s, paths = studio
    clip = s.speak("Ciao.", "it", paths["mine"],
                   options={"guidance_scale": 99, "speed": 1.2, "denoise": False,
                            "bogus": 1, "duration": ""})
    sent = _FakeOmniVoice.seen[-1]["options"]
    assert sent == {"guidance_scale": ADVANCED["guidance_scale"]["max"], "speed": 1.2,
                    "denoise": False}
    assert clip["options"] == sent, "the clip records what it was made with"


def test_a_seed_is_sent_and_the_one_used_is_kept(studio) -> None:
    s, paths = studio
    assert s.speak("Ciao.", "it", paths["mine"], seed=77)["seed"] == 77
    assert _FakeOmniVoice.seen[-1]["seed"] == 77
    assert s.speak("Ciao.", "it", paths["mine"])["seed"] == 4242, "the service's choice is kept"
    with pytest.raises(StudioError, match="seed"):
        s.speak("Ciao.", "it", paths["mine"], seed="abc")


def test_a_volunteers_clips_live_with_their_recording(studio) -> None:
    # Deleting the volunteer's folder - the promise on the consent form -
    # must take everything made from their voice with it.
    s, paths = studio
    clip = s.speak("Nonna, sono io.", "it", paths["volunteer"])
    assert s.clip_file(clip["id"]).parent == Path(paths["volunteer"]).parent / "studio"


@pytest.mark.parametrize("bad", ["", " " * 3, "x" * 601])
def test_empty_or_endless_text_is_refused(studio, bad) -> None:
    s, paths = studio
    with pytest.raises(StudioError):
        s.speak(bad, "it", paths["mine"])


def test_a_malformed_language_or_setting_is_refused(studio) -> None:
    s, paths = studio
    with pytest.raises(StudioError):
        s.speak("Ciao.", "Italian!", paths["mine"])
    with pytest.raises(StudioError, match="number"):
        s.speak("Ciao.", "it", paths["mine"], options={"speed": "fast"})


def test_every_omnivoice_language_is_listed_once_it_answers(studio) -> None:
    s, _ = studio
    assert ["scn", "Sicilian"] in s.languages()


# -- new voices -------------------------------------------------------------------

def test_an_invented_voice_belongs_to_no_one_and_can_speak_again(studio) -> None:
    s, _ = studio
    voice = s.design("Nonna Inventata", "female", "elderly", "low pitch")
    sent = _FakeOmniVoice.seen[-1]
    assert sent["instruct"] == "female, elderly, low pitch"
    assert "reference_path" not in sent, "no one's recording is involved"
    folder = Path(voice["path"]).parent
    assert json.loads((folder / "consent.json").read_text(encoding="utf-8"))["consent"].endswith(
        "no real person")
    # Its words are known, so OmniVoice does not transcribe it.
    assert json.loads((folder / "reference.transcript.json").read_text(encoding="utf-8"))["text"]
    assert "nonna-inventata" in {v["name"] for v in s.usable_voices()}
    s.speak("Ciao.", "it", voice["path"])


def test_an_accent_is_designed_in_english_where_the_model_has_accents(studio) -> None:
    s, _ = studio
    s.design("Uncle", "male", "middle-aged", "low pitch", accent="british accent", whisper=True)
    sent = _FakeOmniVoice.seen[-1]
    assert sent["instruct"] == "male, middle-aged, low pitch, british accent, whisper"
    assert sent["language"] == "en" and sent["text"] == DESIGN_SAMPLE["en"]


def test_invented_voices_take_only_attributes_the_model_knows(studio) -> None:
    s, _ = studio
    with pytest.raises(StudioError):
        s.design("x", "female", "elderly", "Donald Trump")
    with pytest.raises(StudioError):
        s.design("x", "female", "elderly", "low pitch", accent="italian accent")
    assert "elderly" in DESIGN["age"]


def _wav64(seconds: float, amplitude: float = 0.3) -> str:
    buf = io.BytesIO()
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(buf, (amplitude * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 16000,
             format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def test_an_upload_becomes_a_volunteer_with_their_consent(studio, tmp_path) -> None:
    s, _ = studio
    voice = s.upload("Marta R.", "signed, ref 012", _wav64(6.0), "memo.m4a")
    folder = Path(voice["path"]).parent
    assert folder.parent == tmp_path / "data" / "voices" / "volunteers"
    meta = json.loads((folder / "consent.json").read_text(encoding="utf-8"))
    assert meta["consent"] == "signed, ref 012" and "memo.m4a" in meta["source"]


@pytest.mark.parametrize("name,consent,data,why", [
    ("Marta", "", "UNUSED", "consent"),
    ("", "ref 1", "UNUSED", "name"),
    ("Marta", "ref 1", "not base64!", "intact"),
    ("Marta", "ref 1", base64.b64encode(b"hello").decode(), "readable"),
])
def test_an_upload_without_consent_or_sound_is_refused(studio, name, consent, data, why) -> None:
    s, _ = studio
    payload = _wav64(6.0) if data == "UNUSED" else data
    with pytest.raises(StudioError, match=why):
        s.upload(name, consent, payload)


def test_too_short_an_upload_is_refused(studio) -> None:
    s, _ = studio
    with pytest.raises(StudioError, match="3 seconds"):
        s.upload("Marta", "ref 1", _wav64(1.0))


# -- what a reference says --------------------------------------------------------

def test_a_transcript_can_be_prepared_read_and_corrected(studio) -> None:
    s, paths = studio
    assert s.transcript(paths["mine"]) is None
    assert s.prepare(paths["mine"])["text"] == "Buongiorno a tutti."
    assert _FakeOmniVoice.seen[-1] == {"op": "prepare", "reference_path": paths["mine"],
                                       "audio_bytes": 0}
    fixed = s.set_transcript(paths["mine"], "  Buongiorno   a tutti voi. ")
    assert fixed["text"] == "Buongiorno a tutti voi." and fixed["edited"]
    assert s.transcript(paths["mine"])["text"] == "Buongiorno a tutti voi."


def test_a_transcript_cannot_be_written_for_a_voice_without_consent(studio) -> None:
    s, paths = studio
    with pytest.raises(StudioError, match="consent"):
        s.set_transcript(paths["stranger"], "anything")


# -- clips and service ---------------------------------------------------------------

def test_a_clip_name_cannot_reach_outside_the_studio(studio) -> None:
    s, _ = studio
    assert s.clip_file("../../data/voices/phone-sentences") is None
    assert s.clip_file("..\\x") is None


def test_status_says_when_omnivoice_is_down(tmp_path) -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    s = Studio(lambda: [], port=free, root=tmp_path)
    assert s.status()["up"] is False
    assert s.languages() == []
    with pytest.raises(StudioError, match="not running"):
        s.design("x", "female", "elderly", "low pitch")


def test_status_says_when_it_is_up(studio) -> None:
    s, _ = studio
    assert s.status()["up"] is True


# -- the settings table ---------------------------------------------------------------

def test_settings_are_clamped_not_refused_and_unknown_ones_dropped() -> None:
    cleaned = clean_options({"num_step": 1000, "class_temperature": -1, "speed": None,
                             "postprocess_output": 0, "evil": "rm -rf"})
    assert cleaned == {"num_step": ADVANCED["num_step"]["max"],
                       "class_temperature": ADVANCED["class_temperature"]["min"],
                       "postprocess_output": False}


def test_steps_are_whole_numbers() -> None:
    assert clean_options({"num_step": 15.6}) == {"num_step": 16}


def test_the_design_instruction_is_built_only_from_known_words() -> None:
    assert instruct_for("female", "child", "very high pitch") == "female, child, very high pitch"
    with pytest.raises(ValueError):
        instruct_for("female", "robot", "low pitch")


# -- derived voices, deletion, re-design ------------------------------------------

def _speech(path: Path, seconds: float = 4.0, hz: float = 140.0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(24000 * seconds)) / 24000
    sf.write(path, (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32), 24000)
    return str(path)


def test_a_derived_voice_moves_the_pitch_and_keeps_the_consent(studio) -> None:
    from parliamo.audio.pitch import median_f0

    s, paths = studio
    _speech(Path(paths["mine"]))
    words = Path(paths["mine"]).with_name("phone-sentences.transcript.json")
    words.write_text(json.dumps({"text": "Buongiorno.", "start": 0.0, "seconds": 4.0}),
                     encoding="utf-8")
    voice = s.derive("Presenter lower", paths["mine"], semitones=-6, tempo=1.25)
    audio, rate = sf.read(voice["path"], dtype="float32")
    assert median_f0(audio, rate) == pytest.approx(140 * 2 ** (-6 / 12), rel=0.05)
    folder = Path(voice["path"]).parent
    meta = json.loads((folder / "consent.json").read_text(encoding="utf-8"))
    assert meta["derived_from"] == "phone-sentences" and "presenter's own voice" in meta["consent"]
    # With the volunteers, so "Delete all volunteer recordings" takes it too.
    assert folder.parent == s.volunteers_dir
    moved = json.loads((folder / "reference.transcript.json").read_text(encoding="utf-8"))
    assert moved["text"] == "Buongiorno." and moved["seconds"] == pytest.approx(3.2)


def test_a_voice_without_consent_cannot_be_derived(studio) -> None:
    s, paths = studio
    with pytest.raises(StudioError, match="consent"):
        s.derive("x", paths["stranger"], semitones=2)


@pytest.mark.parametrize("semitones,tempo", [(12, 1.0), (-9, 1.0), (0, 2.0), (0, 0.5)])
def test_a_derivation_stays_where_the_words_survive(studio, semitones, tempo) -> None:
    s, paths = studio
    _speech(Path(paths["mine"]))
    with pytest.raises(StudioError):
        s.derive("x", paths["mine"], semitones=semitones, tempo=tempo)


def test_volunteers_and_invented_voices_can_be_deleted_with_their_clips(studio) -> None:
    s, paths = studio
    clip = s.speak("Ciao.", "it", paths["volunteer"])
    s.delete_voice(paths["volunteer"])
    assert not Path(paths["volunteer"]).parent.exists()
    assert s.clip_file(clip["id"]) is None
    invented = s.design("Nonno", "male", "elderly", "low pitch")
    s.delete_voice(invented["path"])
    assert not Path(invented["path"]).parent.exists()


def test_the_presenters_own_recordings_are_not_deleted_from_the_page(studio) -> None:
    s, paths = studio
    with pytest.raises(StudioError, match="own recordings"):
        s.delete_voice(paths["mine"])
    assert Path(paths["mine"]).exists()


def test_an_invented_voice_can_be_spoken_from_its_description(studio) -> None:
    s, _ = studio
    voice = s.design("Uncle", "male", "middle-aged", "low pitch", accent="british accent")
    clip = s.speak("Nonna, sono io.", "it", voice["path"], redesign=True)
    sent = _FakeOmniVoice.seen[-1]
    assert sent["instruct"] == "male, middle-aged, low pitch, british accent"
    assert "reference_path" not in sent and sent["language"] == "it"
    assert clip["redesigned"]
    s.speak("Nonna, sono io.", "it", voice["path"])
    assert "reference_path" in _FakeOmniVoice.seen[-1], "cloned from its sample by default"
