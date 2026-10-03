#!/usr/bin/env python
"""OmniVoice: the sentence spoken in the reference voice, served over the voice socket.

Why a third voice service
-------------------------
RVC and Seed-VC *convert*: Kokoro or Piper speaks the sentence, and the service
repaints the timbre. The rhythm and intonation stay the synthesiser's, which is
much of why a converted voice sounds like a machine wearing someone's voice.
OmniVoice (k2-fsa, March 2026, 646 languages; code Apache-2.0, weights
CC-BY-NC) *speaks*: given the
sentence and a few seconds of someone, it says the sentence in that voice. And
it is zero-shot like Seed-VC, so the presenter and a volunteer are the same
service with a different reference - no swapping services mid-talk.

Same socket protocol as the other two, so the pipeline's client and its
fallback (a failed or slow service means the sentence goes out in the generic
voice) work unchanged. The client adds ``text`` and ``language`` to each
request; the synthesiser's audio it also sends is not used here. A request
without ``text`` is refused - an old client would otherwise get silence.

The reference's transcript is needed for cloning. OmniVoice would fetch a
Whisper from the internet to make one; instead this transcribes the reference
once with the project's own faster-whisper on the CPU (the GPU is the
pipeline's) and keeps it beside the recording as ``<name>.transcript.json``.

Run with the OmniVoice environment (the main one plus the ``omnivoice`` wheel)::

    C:\\Users\\you\\venvs\\omnivoice\\Scripts\\python.exe scripts\\omnivoice_server.py

Friulian has no OmniVoice voice (checked against its 646-language list); it is
spoken with Italian, as everywhere else in this project.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import socket
import socketserver
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Before anything imports huggingface_hub: it reads the cache location once.
from parliamo.paths import configure_model_cache  # noqa: E402

configure_model_cache()
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are local; the venue has no network

from parliamo.tts.numbers import spell_numbers  # noqa: E402
from parliamo.tts.omnivoice_options import clean_options  # noqa: E402

log = logging.getLogger("omnivoice-server")

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_PORT = 8767
MODEL_REPO = "k2-fsa/OmniVoice"
#: The pipeline's recogniser, already in the model cache - faster-whisper's own name
#: for "large-v3-turbo" points at another repository, which is not.
WHISPER_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"
#: Languages OmniVoice has no voice for, spoken as another. Friulian is not in
#: its list; the project speaks it with Italian phonetics and says so.
SPEAK_AS = {"fur": "it"}


# -- the wire format, identical to voice_conversion_server.py ----------------
# Copied rather than imported, like the other two servers: each runs in its
# own environment and the protocol is small enough to keep in step by hand.

def send_message(sock: socket.socket, header: dict[str, Any], audio: np.ndarray) -> None:
    payload = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    header = {**header, "audio_bytes": len(payload)}
    blob = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">I", len(blob)))
    sock.sendall(blob)
    if payload:
        sock.sendall(payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], np.ndarray]:
    (header_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    if header_len > 1024 * 1024:
        raise ValueError(f"header claims {header_len} bytes")
    header = json.loads(_recv_exactly(sock, header_len).decode("utf-8"))
    audio_bytes = int(header.get("audio_bytes", 0))
    if audio_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload claims {audio_bytes} bytes")
    payload = _recv_exactly(sock, audio_bytes) if audio_bytes else b""
    return header, np.frombuffer(payload, dtype="<f4").copy()


# -- the reference: its words, and how much of it -----------------------------

#: OmniVoice asks for 3-10 s of reference. More costs memory more than quality:
#: its audio tokenizer peaked at 5.6 GB encoding 12 s on the GPU and 12.5 GB on
#: 35 s - beside the running pipeline, that is the whole card. So a reference
#: is cut at word boundaries to at most this, and encoded on the CPU.
MAX_REFERENCE_S = 12.0
#: Where the reference starts and ends matters more than its length. Cut
#: mid-phrase ("...kolay taklit edilebildiğini..."), the presenter's clip lost
#: the first word of 6 of 6 Turkish test sentences (WER 25.6%); cut on sentence
#: boundaries from the same recording, 1 of 6 (7.0-9.3%). So the cut ends on a
#: sentence end when there is one past this, and starts on the first word.
MIN_SENTENCE_S = 3.0


def transcript_path(reference: Path) -> Path:
    return reference.with_name(reference.stem + ".transcript.json")


class Transcriber:
    """faster-whisper on the CPU, loaded on first need, for references only.

    Returns the words and the span they occupy: the transcript must match the
    audio it is given exactly, so the audio is cut where the words are, not
    the other way round.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def __call__(self, reference: Path) -> tuple[str, float, float]:
        cached = transcript_path(reference)
        if cached.exists() and cached.stat().st_mtime >= reference.stat().st_mtime:
            data = json.loads(cached.read_text(encoding="utf-8"))
            return str(data["text"]), float(data.get("start", 0.0)), float(data["seconds"])
        if self._model is None:
            from faster_whisper import WhisperModel

            t0 = time.perf_counter()
            self._model = WhisperModel(WHISPER_REPO, device="cpu", compute_type="int8",
                                       cpu_threads=8)
            log.info("reference transcriber loaded in %.1f s", time.perf_counter() - t0)
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(str(reference), beam_size=5,
                                                condition_on_previous_text=False,
                                                word_timestamps=True)
        timed = [w for segment in segments for w in (segment.words or [])]
        if not timed:
            raise ValueError(f"no speech found in the reference {reference.name}")
        begin = timed[0].start
        kept = [w for w in timed if w.end - begin <= MAX_REFERENCE_S] or timed[:1]
        # End on a sentence if one ends late enough to leave a usable clip.
        ends = [i for i, w in enumerate(kept)
                if w.word.strip()[-1:] in ".?!" and w.end - begin >= MIN_SENTENCE_S]
        if ends:
            kept = kept[: ends[-1] + 1]
        text = "".join(w.word for w in kept).strip()
        start = round(max(0.0, begin - 0.1), 2)
        seconds = round(min(kept[-1].end + 0.15, info.duration), 2)
        log.info("transcribed reference %s (%s) in %.1f s, keeping %.1f-%.1f of %.1f s: %s",
                 reference.name, info.language, time.perf_counter() - t0, start, seconds,
                 info.duration, text[:80])
        # What a person said in their recording is theirs too: this file sits
        # beside the recording, is git-ignored with it, and goes when it goes.
        # Correct the text by hand if the recogniser got a word wrong - a
        # transcript that does not match the audio makes a worse clone.
        cached.write_text(json.dumps({"text": text, "start": start, "seconds": seconds,
                                      "language": info.language}, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        return text, start, seconds


# -- the model ----------------------------------------------------------------

class Model:
    def __init__(self, *, steps: int, device: str = "cuda:0") -> None:
        self.steps = steps
        self.device = device
        self._model: Any = None
        self._local = ""
        self._cpu_tokenizer: Any = None
        self._prompts: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._transcribe = Transcriber()
        self.loaded_at: float | None = None
        self.sample_rate = 24000

    def load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from omnivoice import OmniVoice

        t0 = time.perf_counter()
        # A local path, never a repo name: given a name, OmniVoice asks the Hub
        # first, and the venue has no network.
        self._local = snapshot_download(MODEL_REPO, local_files_only=True)
        self._model = OmniVoice.from_pretrained(self._local, device_map=self.device,
                                                dtype=torch.float16)
        self.sample_rate = int(self._model.sampling_rate)
        self.loaded_at = time.time()
        log.info("OmniVoice loaded in %.1f s (%d Hz, %d steps)", time.perf_counter() - t0,
                 self.sample_rate, self.steps)

    def _encoder(self) -> Any:
        """A CPU copy of the audio tokenizer, for encoding references only."""
        if self._cpu_tokenizer is None:
            from transformers import HiggsAudioV2TokenizerModel

            self._cpu_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
                os.path.join(self._local, "audio_tokenizer"), device_map="cpu")
        return self._cpu_tokenizer

    def _stamp(self, path: Path) -> float:
        """Changes when the recording changes - or its transcript, which the
        Studio lets the operator correct: a corrected transcript must rebuild
        the prompt, or the correction does nothing."""
        stamp = path.stat().st_mtime
        words = transcript_path(path)
        if words.exists():
            stamp = max(stamp, words.stat().st_mtime)
        return stamp

    def prompt(self, reference: str) -> Any:
        import dataclasses

        import soundfile as sf
        import torch

        path = Path(reference)
        if not path.exists():
            raise FileNotFoundError(f"reference voice not found: {reference}")
        stamp = self._stamp(path)
        cached = self._prompts.get(reference)
        if cached and cached[0] == stamp:
            return cached[1]
        text, start, seconds = self._transcribe(path)
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)[int(start * rate): int(seconds * rate)]
        t0 = time.perf_counter()
        on_gpu = self._model.audio_tokenizer
        self._model.audio_tokenizer = self._encoder()
        try:
            prompt = self._model.create_voice_clone_prompt(
                ref_audio=(torch.from_numpy(np.ascontiguousarray(audio)), rate), ref_text=text)
        finally:
            self._model.audio_tokenizer = on_gpu
        prompt = dataclasses.replace(prompt,
                                     ref_audio_tokens=prompt.ref_audio_tokens.to(on_gpu.device))
        log.info("voice prompt for %s (%.1f s) built in %.2f s", path.name, seconds - start,
                 time.perf_counter() - t0)
        self._prompts[reference] = (self._stamp(path), prompt)
        return prompt

    def prepare(self, reference: str) -> dict[str, Any]:
        """Transcribe and encode a reference now, and say what it says."""
        with self._lock:
            self.prompt(reference)
        text, start, seconds = self._transcribe(Path(reference))
        return {"text": text, "start": start, "seconds": seconds}

    def languages(self) -> list[list[str]]:
        from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

        return sorted(([code, lang_display_name(name)] for name, code in LANG_NAME_TO_ID.items()),
                      key=lambda pair: pair[1])

    def generate(self, text: str, language: str | None, *, reference: str = "",
                 instruct: str = "", options: dict[str, Any] | None = None,
                 seed: int | None = None) -> tuple[np.ndarray, int, float, int]:
        """Speak *text*: in the reference's voice, in a designed one, or - with
        neither - in a voice the model picks. Returns the seed it used, so a
        render worth keeping can be made again."""
        import torch

        opts = dict(options or {})
        opts.setdefault("num_step", self.steps)
        speed = opts.pop("speed", None)
        duration = opts.pop("duration", None)
        lang = SPEAK_AS.get(language or "", language) or None
        if lang and lang not in self._model.supported_language_ids():
            raise ValueError(f"OmniVoice has no language {language!r}")
        # OmniVoice reads digits badly ("15 secondi" came back as "chi me è
        # secondo"); the translator writes them. Spelled out first.
        text = spell_numbers(text, lang)
        used = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "little") >> 1
        with self._lock:
            prompt = self.prompt(reference) if reference else None
            torch.manual_seed(used)
            t0 = time.perf_counter()
            kwargs: dict[str, Any] = {"text": text, "language": lang, "speed": speed,
                                      "duration": duration, **opts}
            if prompt is not None:
                kwargs["voice_clone_prompt"] = prompt
            elif instruct:
                kwargs["instruct"] = instruct
            audios = self._model.generate(**kwargs)
            compute = time.perf_counter() - t0
        wave = np.asarray(audios[0], dtype=np.float32).reshape(-1)
        return wave, self.sample_rate, compute, used

    def speak(self, text: str, language: str | None, reference: str,
              steps: int | None = None) -> tuple[np.ndarray, int, float]:
        wave, rate, compute, _ = self.generate(
            text, language, reference=reference,
            options={"num_step": int(steps)} if steps else None)
        return wave, rate, compute


MODEL: Model | None = None

#: What a browser gets. The first thing anyone does with a port number is
#: open it in a browser; the protocol then read "GET " as a length of
#: 1195725856 bytes and the log filled with "malformed request".
BROWSER_REPLY = (
    b"HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nConnection: close\r\n\r\n"
    b"This is the OmniVoice voice service of parlIAmo, not a web page.\n"
    b"It is running. The application talks to it; people use the operator page:\n"
    b"http://127.0.0.1:8770/  (the Studio tab speaks typed sentences through it)\n"
)

EMPTY = np.zeros(0, dtype=np.float32)


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        assert MODEL is not None
        with contextlib.suppress(OSError):
            if self.request.recv(4, socket.MSG_PEEK) in (b"GET ", b"HEAD", b"POST"):
                self.request.sendall(BROWSER_REPLY)
                return
        try:
            header, _audio = recv_message(self.request)
        except Exception as exc:
            log.warning("malformed request: %s", exc)
            return
        op = header.get("op", "convert")
        try:
            if op == "ping":
                send_message(self.request, {
                    "service": "omnivoice", "model": MODEL_REPO, "steps": MODEL.steps,
                    "model_loaded": MODEL.loaded_at is not None, "loaded_at": MODEL.loaded_at,
                }, EMPTY)
                return
            if op == "languages":
                send_message(self.request, {"languages": MODEL.languages()}, EMPTY)
                return
            if op == "prepare":
                send_message(self.request, MODEL.prepare(str(header.get("reference_path") or "")),
                             EMPTY)
                return
            if op != "convert":
                raise ValueError(f"unknown op {op!r}")
            text = str(header.get("text") or "").strip()
            if not text:
                raise ValueError("OmniVoice speaks the sentence itself and needs its text; "
                                 "this client did not send one")
            reference = str(header.get("reference_path") or "")
            instruct = str(header.get("instruct") or "").strip()
            if not reference and not instruct and not header.get("auto"):
                # A missing reference must not quietly become a stranger's
                # voice; the model's own choice is asked for by name.
                raise ValueError("no voice: send a reference, an instruct, or auto")
            options = clean_options(header.get("options"))
            if header.get("steps") and "num_step" not in options:
                options["num_step"] = int(header["steps"])
            seed = header.get("seed")
            wave, rate, compute, used = MODEL.generate(
                text, header.get("language"), reference=reference, instruct=instruct,
                options=options, seed=None if seed in (None, "") else int(seed))
            who = Path(reference).stem if reference else (instruct or "auto")
            log.info("spoke %.2fs in %.2fs (%s, %s, seed %d): %s", wave.size / max(rate, 1),
                     compute, header.get("language"), who, used, text[:60])
            send_message(self.request, {"sample_rate": rate, "compute_s": round(compute, 4),
                                        "voice": who, "seed": used}, wave)
        except Exception as exc:
            log.exception("request failed")
            with contextlib.suppress(OSError):
                send_message(self.request, {"error": f"{type(exc).__name__}: {exc}"}, EMPTY)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--steps", type=int, default=16,
                        help="decoding steps: fewer is faster, more is cleaner "
                             "(the model's default is 32; see docs/00-state.md)")
    parser.add_argument("--warm", default="data/voices/phone-sentences.wav",
                        help="reference to prepare at start, so the first sentence is not "
                             "the one that pays for transcribing it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "transformers", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    global MODEL
    MODEL = Model(steps=args.steps)
    MODEL.load()
    warm = ROOT / args.warm if args.warm else None
    if warm and warm.exists():
        t0 = time.perf_counter()
        MODEL.speak("Buongiorno a tutti.", "it", str(warm))
        log.info("warm in %.1f s", time.perf_counter() - t0)

    with Server((args.host, args.port), Handler) as server:
        log.info("listening on %s:%d", args.host, args.port)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
