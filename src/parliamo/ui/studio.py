"""The studio: everything OmniVoice can do, on the operator page.

The live pipeline speaks what the presenter says. The studio speaks what the
operator types - the grandchild's phone call in a volunteer's voice, the same
sentence in three languages, a laugh in the middle of it - through the
OmniVoice service (port 8767), and keeps each result as a clip that can be
played in the browser or through the speakers. It needs OmniVoice running,
not the pipeline.

Voices, and nothing else:

- a **consented** recording: the presenter's own, or a volunteer's - recorded
  here, or uploaded from a file - with the signed form reference the Voices
  tab already requires;
- an **invented** voice, designed from attributes OmniVoice understands
  (gender, age, pitch, accent, whisper). It belongs to no one, and is saved as
  a reference so the same character can speak again - and be used live;
- the model's **own choice** (auto): a voice it picks, which belongs to no one
  either, and changes with the seed;
- a **derived** voice: a consented recording with its pitch and tempo moved,
  saved as a new reference. OmniVoice follows the reference - measured, a
  reference moved -6/+4/+8 semitones gave clones at -5.1/+4.0/+8.3 - whereas a
  design instruction added to a clone changed nothing (pitch within 4 Hz for
  "very high" and "very low") and once inserted a "non". So a recording is
  changed by changing the recording, not by describing it. A derived voice is
  still that person's, carries their consent, and lives with the volunteers so
  it is deleted with them.

There is deliberately no fourth kind. A public figure's voice, or a cartoon
character's (which is a voice actor's), has no consent record, and the whole
point of the talk is what happens when that is ignored.

Everything made from a volunteer - their upload, their clips - is written in
their own folder, so "Delete all volunteer recordings" removes it all.
"""

from __future__ import annotations

import base64
import io
import json
import re
import socket
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..tts.omnivoice_options import DESIGN, clean_options, instruct_for

OMNIVOICE_PORT = 8767

#: The voice value that asks the model to pick one itself.
AUTO_VOICE = "@auto"

#: Offered first; OmniVoice has 646 in all, listed after these when it runs.
LANGUAGES = {"it": "Italiano", "tr": "Türkçe", "fur": "Furlan (with Italian phonetics)",
             "es": "Español", "de": "Deutsch", "en": "English", "fr": "Français"}

#: What an invented voice says to become a reference: whole sentences, so the
#: cut falls on sentence ends (a reference cut mid-phrase made OmniVoice drop
#: the first word of every sentence it spoke - ADR 0015). In English when an
#: accent is chosen: the model's accents are English accents.
DESIGN_SAMPLE = {
    "it": "Buongiorno a tutti. Questa è una voce inventata dal computer. "
          "Non appartiene a nessuna persona vera.",
    "en": "Good morning, everyone. This is a voice invented by the computer. "
          "It belongs to no real person.",
}

MAX_TEXT = 600
#: How far a derived voice may move. Beyond about 8 semitones the clone's words
#: started to break ("ho abuto inizidenza" at +8).
DERIVE_SEMITONES = (-8.0, 8.0)
DERIVE_TEMPO = (0.8, 1.25)
#: An upload is cut to this before it is kept; OmniVoice uses at most 12 s.
MAX_UPLOAD_S = 60.0
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


class StudioError(ValueError):
    """A request the studio refuses, with a reason fit to show the operator."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "voice"


def _transcript_path(reference: Path) -> Path:
    # The same file the OmniVoice service writes and reads.
    return reference.with_name(reference.stem + ".transcript.json")


class Studio:
    def __init__(
        self,
        voices: Callable[[], list[dict]],
        *,
        host: str = "127.0.0.1",
        port: int = OMNIVOICE_PORT,
        root: Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        from ..paths import resolve

        #: The Voices tab's list, with consent attached - the one place that
        #: knows which recordings may be used.
        self.voices = voices
        self.host = host
        self.port = port
        self.timeout = timeout
        self.root = Path(root) if root is not None else resolve(".")
        self.clips_dir = self.root / "runs" / "studio"
        self.designed_dir = self.root / "data" / "voices" / "designed"
        self.volunteers_dir = self.root / "data" / "voices" / "volunteers"
        self._languages: list[list[str]] | None = None

    # -- the service ------------------------------------------------------

    def status(self) -> dict[str, Any]:
        from ..tts.conversion import VoiceConverter

        try:
            who = VoiceConverter(host=self.host, port=self.port, timeout=2.0).ping()
        except Exception as exc:
            return {"up": False, "port": self.port, "reason": str(exc)}
        up = who.get("service") == "omnivoice"
        return {"up": up, "port": self.port, "steps": who.get("steps"),
                "reason": "" if up else f"port {self.port} answers as {who.get('service')!r}"}

    def _call(self, header: dict[str, Any]) -> tuple[dict, np.ndarray]:
        from ..tts.conversion import recv_message, send_message

        try:
            sock = socket.create_connection((self.host, self.port), timeout=5.0)
        except OSError as exc:
            raise StudioError(
                f"OmniVoice is not running on port {self.port}. Start it with "
                r"C:\Users\you\venvs\omnivoice\Scripts\python.exe scripts\omnivoice_server.py"
            ) from exc
        with sock:
            sock.settimeout(self.timeout)
            send_message(sock, header, np.zeros(0, dtype=np.float32))
            reply, audio = recv_message(sock)
        if reply.get("error"):
            raise StudioError(f"OmniVoice: {reply['error']}")
        return reply, audio

    def languages(self) -> list[list[str]]:
        """Every language OmniVoice speaks, as [code, name], once it answers."""
        if self._languages is None:
            try:
                reply, _ = self._call({"op": "languages"})
            except (StudioError, OSError):
                return []
            self._languages = [list(pair) for pair in reply.get("languages", [])]
        return self._languages

    # -- voices -----------------------------------------------------------

    def usable_voices(self) -> list[dict]:
        """Every voice with a consent record, designed voices included."""
        out = [v for v in self.voices() if v.get("consent")]
        known = {str(Path(v["path"]).resolve()) for v in out}
        for voice in self.designed_voices():
            if str(Path(voice["path"]).resolve()) not in known:
                out.append(voice)
        return out

    def designed_voices(self) -> list[dict]:
        out = []
        if not self.designed_dir.is_dir():
            return out
        for folder in sorted(p for p in self.designed_dir.glob("*") if p.is_dir()):
            reference = folder / "reference.wav"
            meta = folder / "consent.json"
            if not reference.exists() or not meta.exists():
                continue
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"name": folder.name, "path": str(reference), "designed": True,
                        "consent": data.get("consent", ""), "design": data.get("design", "")})
        return out

    def _voice(self, path: str) -> dict:
        wanted = Path(path).resolve()
        for voice in self.usable_voices():
            if Path(voice["path"]).resolve() == wanted:
                return voice
        raise StudioError("That voice has no consent record. Use one from the Voices tab "
                          "with its signed form reference, record or upload one with "
                          "consent, or invent one.")

    # -- speaking ---------------------------------------------------------

    def speak(self, text: str, language: str, voice_path: str,
              options: dict[str, Any] | None = None, seed: int | None = None,
              steps: int | None = None, redesign: bool = False) -> dict:
        text = " ".join(str(text or "").split())
        if not text:
            raise StudioError("Write the sentence first.")
        if len(text) > MAX_TEXT:
            raise StudioError(f"Keep it under {MAX_TEXT} characters; split longer text.")
        if not re.fullmatch(r"[a-z]{2,3}", language or ""):
            raise StudioError(f"Unknown language {language!r}.")
        try:
            opts = clean_options(options)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc
        if steps is not None:
            opts.setdefault("num_step", int(steps))
        if seed not in (None, ""):
            try:
                seed = int(seed)
            except (TypeError, ValueError) as exc:
                raise StudioError("The seed must be a whole number.") from exc
            if not 0 <= seed < 2**31:
                raise StudioError("The seed must be between 0 and 2147483647.")
        else:
            seed = None

        header: dict[str, Any] = {"op": "convert", "sample_rate": 24000, "text": text,
                                  "language": language, "options": opts}
        if seed is not None:
            header["seed"] = seed
        if voice_path == AUTO_VOICE:
            voice = {"name": "auto", "auto": True}
            header["auto"] = True
        else:
            voice = self._voice(voice_path)
            from ..tts.base import VoiceProfile

            VoiceProfile(name=voice["name"], reference_path=voice["path"],
                         consent=voice["consent"]).validate()
            if redesign and voice.get("designed") and voice.get("design"):
                # An invented voice spoken from its description in the target
                # language, instead of cloned from its saved sample - the
                # accent attributes then apply to that language. The voice
                # may vary between renders; the seed holds it.
                header["instruct"] = voice["design"]
            else:
                header["reference_path"] = voice["path"]
        reply, audio = self._call(header)
        rate = int(reply.get("sample_rate", 24000))
        return self._save(audio, rate, {
            "text": text, "language": language, "voice": voice["name"],
            "voice_path": voice_path, "options": opts, "seed": reply.get("seed", seed),
            "compute_s": round(float(reply.get("compute_s", 0.0)), 2),
            "volunteer": bool(voice.get("volunteer")), "designed": bool(voice.get("designed")),
            "auto": bool(voice.get("auto")), "redesigned": "instruct" in header},
            folder=self._clip_folder(voice))

    def _clip_folder(self, voice: dict) -> Path:
        # A volunteer's clips live with their recording: deleting the
        # volunteer deletes everything made from their voice.
        if voice.get("volunteer"):
            return Path(voice["path"]).parent / "studio"
        return self.clips_dir

    def _save(self, audio: np.ndarray, rate: int, meta: dict, folder: Path) -> dict:
        import soundfile as sf

        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        clip_id = f"{stamp}-{_slug(meta['voice'])}"
        sf.write(folder / f"{clip_id}.wav", np.asarray(audio, dtype=np.float32), rate)
        meta = {**meta, "id": clip_id, "seconds": round(audio.size / max(rate, 1), 2),
                "created": time.time()}
        (folder / f"{clip_id}.json").write_text(json.dumps(meta, ensure_ascii=False),
                                               encoding="utf-8")
        return meta

    # -- new voices -------------------------------------------------------

    def design(self, name: str, gender: str, age: str, pitch: str, accent: str = "",
               whisper: bool = False, seed: int | None = None) -> dict:
        slug = _slug(name)
        if not name.strip():
            raise StudioError("Give the invented voice a name.")
        try:
            instruct = instruct_for(gender, age, pitch, accent, whisper)
        except ValueError as exc:
            raise StudioError(str(exc)) from exc
        language = "en" if accent else "it"
        header: dict[str, Any] = {"op": "convert", "sample_rate": 24000,
                                  "text": DESIGN_SAMPLE[language], "language": language,
                                  "instruct": instruct}
        if seed not in (None, ""):
            header["seed"] = int(seed)
        reply, audio = self._call(header)
        rate = int(reply.get("sample_rate", 24000))
        import soundfile as sf

        folder = self.designed_dir / slug
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "reference.wav", np.asarray(audio, dtype=np.float32), rate)
        consent = f"invented voice, designed from attributes ({instruct}); no real person"
        (folder / "consent.json").write_text(json.dumps(
            {"consent": consent, "design": instruct, "seed": reply.get("seed"),
             "created": time.time()}, ensure_ascii=False), encoding="utf-8")
        # The words the reference says, so OmniVoice need not transcribe it.
        _transcript_path(folder / "reference.wav").write_text(json.dumps(
            {"text": DESIGN_SAMPLE[language], "start": 0.0,
             "seconds": round(audio.size / rate, 2), "language": language},
            ensure_ascii=False), encoding="utf-8")
        return {"name": slug, "path": str(folder / "reference.wav"), "designed": True,
                "consent": consent, "design": instruct, "seed": reply.get("seed"),
                "compute_s": round(float(reply.get("compute_s", 0.0)), 2)}

    def upload(self, name: str, consent: str, data_b64: str, filename: str = "") -> dict:
        """Keep an uploaded recording as a volunteer's voice.

        The page decodes whatever the browser can play (an iPhone voice memo
        included) and sends plain WAV, so nothing here depends on a codec.
        """
        import soundfile as sf

        from ..tts.reference import build_reference

        name, consent = name.strip(), consent.strip()
        if not name:
            raise StudioError("A name is required.")
        if not consent:
            raise StudioError("A consent record is required before keeping anyone's voice.")
        try:
            blob = base64.b64decode(data_b64 or "", validate=True)
        except (ValueError, TypeError) as exc:
            raise StudioError("The file did not arrive intact.") from exc
        if not blob:
            raise StudioError("The file is empty.")
        if len(blob) > MAX_UPLOAD_BYTES:
            raise StudioError("The file is too large; send at most a minute of audio.")
        try:
            audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        except Exception as exc:
            raise StudioError(f"Not a readable recording: {exc}") from exc
        audio = audio.mean(axis=1)[: int(MAX_UPLOAD_S * rate)]
        if audio.size < 3 * rate or float(np.abs(audio).max()) < 0.01:
            raise StudioError("Too short or too quiet: at least 3 seconds of speech are needed.")

        person = self.volunteers_dir / _slug(name)
        person.mkdir(parents=True, exist_ok=True)
        sf.write(person / "raw.wav", audio, rate, subtype="FLOAT")
        reference = build_reference([(audio, rate)], target_seconds=20.0, sample_rate=rate)
        sf.write(person / "reference.wav", reference, rate, subtype="FLOAT")
        (person / "consent.json").write_text(json.dumps({
            "name": name, "consent": consent,
            "recorded": datetime.now().astimezone().isoformat(),
            "source": f"uploaded file {Path(filename).name}" if filename else "uploaded file",
            "seconds": round(audio.size / rate, 1),
            "delete_after": "the end of this session",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"name": person.name, "path": str(person / "reference.wav"), "volunteer": True,
                "consent": consent, "seconds": round(reference.size / rate, 1)}

    def derive(self, name: str, source_path: str, semitones: float = 0.0,
               tempo: float = 1.0) -> dict:
        """A new voice from a consented recording, its pitch and tempo moved."""
        import librosa
        import soundfile as sf

        name = name.strip()
        if not name:
            raise StudioError("Give the new voice a name.")
        source = self._voice(source_path)
        try:
            semitones, tempo = float(semitones), float(tempo)
        except (TypeError, ValueError) as exc:
            raise StudioError("Pitch and tempo must be numbers.") from exc
        if not DERIVE_SEMITONES[0] <= semitones <= DERIVE_SEMITONES[1]:
            raise StudioError(f"Pitch must stay within {DERIVE_SEMITONES[0]:+.0f} and "
                              f"{DERIVE_SEMITONES[1]:+.0f} semitones.")
        if not DERIVE_TEMPO[0] <= tempo <= DERIVE_TEMPO[1]:
            raise StudioError(f"Tempo must stay within {DERIVE_TEMPO[0]} and {DERIVE_TEMPO[1]}.")
        folder = self.volunteers_dir / _slug(name)
        if folder.resolve() == Path(source["path"]).parent.resolve():
            raise StudioError("Choose a different name from the source voice.")

        audio, rate = sf.read(source["path"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if semitones:
            audio = librosa.effects.pitch_shift(audio, sr=rate, n_steps=semitones)
        if tempo != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=tempo)
        peak = float(np.abs(audio).max()) or 1.0
        audio = (audio / peak * 0.9).astype(np.float32)

        folder.mkdir(parents=True, exist_ok=True)
        reference = folder / "reference.wav"
        sf.write(reference, audio, rate, subtype="FLOAT")
        consent = f"derived from {source['name']} - {source['consent']}"
        (folder / "consent.json").write_text(json.dumps({
            "name": name, "consent": consent, "derived_from": source["name"],
            "semitones": semitones, "tempo": tempo,
            "recorded": datetime.now().astimezone().isoformat(),
            "delete_after": "the end of this session",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        # Same words, at the new tempo: no need to transcribe it again.
        words = _transcript_path(Path(source["path"]))
        if words.exists():
            try:
                data = json.loads(words.read_text(encoding="utf-8"))
                data["start"] = round(float(data.get("start", 0.0)) / tempo, 2)
                data["seconds"] = round(min(float(data["seconds"]) / tempo,
                                            audio.size / rate), 2)
                _transcript_path(reference).write_text(json.dumps(data, ensure_ascii=False),
                                                       encoding="utf-8")
            except (OSError, ValueError, KeyError):
                pass
        return {"name": folder.name, "path": str(reference), "volunteer": True,
                "consent": consent, "semitones": semitones, "tempo": tempo}

    def delete_voice(self, voice_path: str) -> dict:
        """Delete a volunteer's, uploaded, derived or invented voice, with its clips.

        The presenter's own recordings are not deleted from a web page.
        """
        import shutil

        voice = self._voice(voice_path)
        folder = Path(voice["path"]).parent.resolve()
        if folder.parent not in (self.volunteers_dir.resolve(), self.designed_dir.resolve()):
            raise StudioError("Your own recordings are not deleted from here; volunteers', "
                              "uploaded, derived and invented voices are.")
        shutil.rmtree(folder, ignore_errors=True)
        return {"deleted": voice["name"], "path": voice["path"]}

    # -- what a reference says --------------------------------------------

    def transcript(self, voice_path: str) -> dict | None:
        voice = self._voice(voice_path)
        path = _transcript_path(Path(voice["path"]))
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def set_transcript(self, voice_path: str, text: str) -> dict:
        """Correct what a reference says. A transcript that matches the audio
        makes a better clone; the service rebuilds the voice when it changes."""
        import soundfile as sf

        voice = self._voice(voice_path)
        text = " ".join(str(text or "").split())
        if not text:
            raise StudioError("The transcript cannot be empty.")
        reference = Path(voice["path"])
        current = self.transcript(voice_path) or {}
        seconds = current.get("seconds")
        if seconds is None:
            seconds = round(min(sf.info(str(reference)).duration, 20.0), 2)
        data = {**current, "text": text, "start": current.get("start", 0.0),
                "seconds": seconds, "edited": True}
        _transcript_path(reference).write_text(json.dumps(data, ensure_ascii=False),
                                               encoding="utf-8")
        return data

    def prepare(self, voice_path: str) -> dict:
        """Have the service transcribe and encode a voice now, not on first use."""
        voice = self._voice(voice_path)
        reply, _ = self._call({"op": "prepare", "reference_path": voice["path"]})
        return {"text": reply.get("text", ""), "start": reply.get("start"),
                "seconds": reply.get("seconds")}

    # -- clips ------------------------------------------------------------

    def _clip_folders(self) -> list[Path]:
        folders = [self.clips_dir]
        if self.volunteers_dir.is_dir():
            folders += [p / "studio" for p in self.volunteers_dir.glob("*") if p.is_dir()]
        return [f for f in folders if f.is_dir()]

    def clips(self, limit: int = 40) -> list[dict]:
        out = []
        for folder in self._clip_folders():
            for meta in folder.glob("*.json"):
                try:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if (folder / f"{data.get('id')}.wav").exists():
                    out.append(data)
        out.sort(key=lambda c: c.get("created", 0), reverse=True)
        return out[:limit]

    def clip_file(self, clip_id: str) -> Path | None:
        if not re.fullmatch(r"[0-9a-z-]{1,80}", clip_id or ""):
            return None
        for folder in self._clip_folders():
            path = folder / f"{clip_id}.wav"
            if path.is_file() and path.resolve().parent == folder.resolve():
                return path
        return None

    def delete(self, clip_id: str) -> bool:
        path = self.clip_file(clip_id)
        if path is None:
            return False
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        return True

    def load(self, clip_id: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        path = self.clip_file(clip_id)
        if path is None:
            raise StudioError("No such clip.")
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(rate)


__all__ = ["AUTO_VOICE", "DESIGN", "LANGUAGES", "Studio", "StudioError"]
