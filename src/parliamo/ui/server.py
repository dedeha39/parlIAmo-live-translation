"""A local page showing the translation, and one control that silences it.

Server-sent events, not WebSockets
----------------------------------
The push is one-way - the pipeline has things to tell the page, the page has
almost nothing to tell the pipeline - and SSE is one-way. It is plain HTTP, it
needs no extra dependency (``websockets`` is not installed and does not need to
be), and browsers reconnect on their own when a stream drops. On a laptop that
will be carried to a stage, "reconnects by itself" is worth more than duplex.

The one thing the page sends back is the panic control, which is an ordinary
POST.

Never block the pipeline
------------------------
The UI is a passive observer and must stay one. Each connected browser gets a
bounded queue; when it fills, events are dropped for *that* browser and counted.
A slow or wedged browser tab must not be able to stall the delivery thread, and
a laptop on a stage will have a browser tab that is slow at some point.
"""

# Deliberately NOT `from __future__ import annotations`.
#
# FastAPI resolves a handler's annotations at runtime to decide what each
# parameter is. With the future import every annotation becomes a string, and
# `Request` is imported inside build_app() rather than at module scope - so
# FastAPI cannot resolve the name, falls back to treating `request` as a query
# parameter, and every POST comes back 422 with
# `{"loc": ["query", "request"], "msg": "Field required"}`.
#
# The lazy import is worth keeping: it means a machine without fastapi can
# still run the pipeline headless. So the future import goes instead. Nothing
# here needs it on Python 3.12.

import asyncio
import contextlib
import json
import logging
import math
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The operator's window, at "/". Light or dark, laptop distance, CDS-flavoured.
APP = Path(__file__).with_name("app.html")
#: The room's window, at "/screen". Dark, huge, projected - a different problem
#: at a different distance, so deliberately not the same page.
SCREEN = Path(__file__).with_name("page.html")

#: Events buffered per connected browser before dropping. A few seconds of
#: subtitles; past that the browser is not keeping up and the newest text
#: matters more than the backlog.
QUEUE_DEPTH = 32

#: Committed sentences kept for a late-joining browser. Matches what the page
#: shows, so a reload restores the screen rather than approximating it.
HISTORY = 4


@dataclass(slots=True)
class UIState:
    """What the page shows besides the subtitles themselves."""

    source_lang: str = "tr"
    target_lang: str = "it"
    speaking: bool = True
    muted: bool = False
    #: Listening paused by the operator (P): the microphone is ignored.
    paused: bool = False
    #: Whose voice the system is currently speaking in, for the operator strip.
    #: A voice swapped in mid-talk must be visible - the operator has to know
    #: whether the volunteer's recording is still loaded.
    voice: str = "generic"
    #: Set when output latency is unmeasured and speakers are live - the one
    #: misconfiguration that reliably ruins a demonstration.
    gate_warning: str = ""
    delivered: int = 0
    partials: int = 0
    last_lag_s: float = 0.0
    listening: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "speaking": self.speaking,
            "muted": self.muted,
            "paused": self.paused,
            "voice": self.voice,
            "gate_warning": self.gate_warning,
            "delivered": self.delivered,
            "partials": self.partials,
            "last_lag_s": round(self.last_lag_s, 2),
            "listening": self.listening,
        }


class SubtitleServer:
    """Serves the page, broadcasts deliveries, and carries the mute control.

    Runs uvicorn on its own thread so the caller keeps its own. Construction is
    cheap; nothing is imported from fastapi until :meth:`start`, so a machine
    without it can still run the pipeline headless.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8770,
        state: UIState | None = None,
        on_mute: Callable[[bool], None] | None = None,
        on_pause: Callable[[bool], None] | None = None,
        on_voice: Callable[[str | None, str], None] | None = None,
        controller: Any = None,
    ) -> None:
        self.host = host
        self.port = port
        self.state = state or UIState()
        #: Called with True to silence output, False to restore it.
        self.on_mute = on_mute
        self.on_pause = on_pause
        #: Called with (reference path or None, consent) to change the voice
        #: mid-talk. This is the volunteer demonstration.
        self.on_voice = on_voice
        #: Owns the pipeline, so the page can start and stop it. Optional: a
        #: run started from the CLI has no controller and simply shows no
        #: start button.
        self.controller = controller
        #: The reference voice named in the config and the consent recorded
        #: beside it, as (resolved path, consent). The Voices tab lists the
        #: file, and pressing *use* on it must carry that consent - the first
        #: version sent an empty string and the presenter's own voice, with
        #: its consent sitting in local.yaml, was refused on stage.
        self.configured_voice: tuple[str, str] | None = None
        #: (host, port) of the Seed-VC service, so the pre-flight list can say
        #: whether cloning is possible at all before anyone presses start.
        self.conversion_address: tuple[str, int] | None = None
        self.conversion_enabled = False
        #: Typed sentences spoken through OmniVoice; see ui/studio.py.
        self._studio: Any = None

        # The last few committed sentences, replayed to a browser that connects
        # late. Without it, reloading the page mid-talk shows a blank screen
        # until the speaker finishes their next sentence - which on stage looks
        # exactly like the system having died.
        self._history: deque = deque(maxlen=HISTORY)
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self.dropped = 0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # -- publishing -------------------------------------------------------

    def publish(self, kind: str, payload: dict[str, Any]) -> None:
        """Push one event to every connected browser. Never raises, never blocks.

        Called from the delivery thread, so both of those matter more than
        delivery: a subtitle that does not arrive is a subtitle, a delivery
        thread that blocks is the show.
        """
        loop = self._loop
        if loop is None:
            return
        message = json.dumps({"kind": kind, **payload}, ensure_ascii=False)
        with self._lock:
            queues = list(self._subscribers)
        for queue in queues:
            with contextlib.suppress(RuntimeError):  # the loop may be closing
                loop.call_soon_threadsafe(self._offer, queue, message)

    def _offer(self, queue: asyncio.Queue, message: str) -> None:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            self.dropped += 1

    def publish_delivery(self, event: Any) -> None:
        """Publish a :class:`~parliamo.pipeline.translator.DeliveryEvent`."""
        data = event.as_dict()
        if event.partial:
            self.state.partials += 1
        else:
            self.state.delivered += 1
            self.state.last_lag_s = event.total_lag_s
            # Only committed sentences are worth replaying. A partial is a
            # guess that was already superseded by the time anyone reconnects.
            self._history.append(data)
        self.publish("delivery", {"event": data, "state": self.state.as_dict()})

    def publish_state(self) -> None:
        self.publish("state", {"state": self.state.as_dict()})

    # -- what the operator needs to know ----------------------------------

    def checks(self) -> dict:
        """The pre-flight list, as data.

        It exists because of one failure mode: starting with speakers live and
        `output_latency_ms` at 0, which turns the microphone into a feedback
        loop within a few sentences. That is the only remaining item that can
        ruin the demonstration outright, so it is shown where it cannot be
        missed rather than logged where it can.

        Nothing here blocks starting. Whether a warning is acceptable is a
        judgement for the person on stage, not for this function.
        """
        rows: list[dict] = []

        def add(level: str, title: str, detail: str, action: str = "") -> None:
            rows.append({"level": level, "title": title, "detail": detail,
                         "action": action})

        if self.state.gate_warning:
            add("warn", "Output latency not measured",
                "The gate is sized from PortAudio's reported latency, which "
                "excludes the vendor DSP - 3 ms reported against ~430 ms real "
                "on this laptop. With speakers live the microphone hears them "
                "and the system starts translating its own output.",
                "scripts/measure_audio_device.py --label venue")
        else:
            add("ok", "Output latency measured", "The half-duplex gate is sized correctly.")

        if self.state.speaking:
            add("ok", "Audio output on", "The room will hear the translation.")
        else:
            add("warn", "Subtitles only", "Nothing will be spoken.")

        if self.state.voice == "generic":
            add("ok", "Generic voice", "No cloned voice is loaded.")
        else:
            add("warn", f"Cloned voice loaded: {self.state.voice}",
                "Delete the recording when the session ends - the consent form "
                "promises it.", "clone_volunteer.py --forget-all")

        # Whether a cloned voice is *possible*. Seed-VC runs in its own
        # process and its own conda environment; when it is not up, every
        # sentence is spoken in the generic voice and the page said nothing
        # about why. This is the row that answers "my voice is configured, so
        # why is it not being used".
        if self.conversion_address is not None:
            host, port = self.conversion_address
            reachable = self.conversion_reachable()
            if not self.conversion_enabled:
                add("ok", "Voice conversion off",
                    "tts.conversion.enabled is false; the generic voice is used.")
            elif reachable:
                # Two services speak this protocol now: Seed-VC (zero-shot,
                # any reference, ~0.93 s) and RVC (one trained voice, ~0.43 s).
                # Which one answers changes what the Voices tab means, so the
                # row says - and on which port, since Start uses whichever runs.
                who = self.conversion_service()
                port = who.get("port", port)
                if who.get("service") == "omnivoice":
                    add("ok", f"Voice: OmniVoice at {host}:{port}",
                        "Zero-shot, any consented reference; speaks each sentence itself "
                        "rather than converting the generic voice, ~1 s per sentence. "
                        "Holds ~2 GB: choose NLLB 600M. A new reference takes 5-15 s to "
                        "prepare on first use.")
                    chosen_mt = str(getattr(self.controller, "mt_model", "") or "")
                    if "1.3B" in chosen_mt:
                        add("warn", "NLLB 1.3B chosen beside OmniVoice",
                            "OmniVoice holds ~1.4 GB more than RVC; the larger translator "
                            "leaves too little. Choose NLLB 600M on the Setup tab.")
                elif who.get("service") == "rvc":
                    add("ok", f"Voice conversion: RVC '{who.get('model')}' at {host}:{port}",
                        "Trained once, ~0.43 s per sentence. The Voices tab is inert: "
                        "this service speaks in its trained voice whatever is selected. "
                        "For a volunteer, switch to Seed-VC.")
                else:
                    add("ok", f"Voice conversion: Seed-VC at {host}:{port}",
                        "Zero-shot, any consented reference, ~0.93 s per sentence.")
                    # Seed-VC holds ~2.2 GB more than RVC. The larger
                    # translator fits beside RVC (6942 MiB peak, measured)
                    # and leaves ~350 MiB beside Seed-VC - one browser tab
                    # from running out mid-sentence.
                    chosen_mt = str(getattr(self.controller, "mt_model", "") or "")
                    if "1.3B" in chosen_mt:
                        add("warn", "NLLB 1.3B chosen beside Seed-VC",
                            "Seed-VC holds ~2.2 GB more GPU memory than RVC; with the "
                            "larger translator about 350 MiB is left. Choose NLLB 600M "
                            "on the Setup tab, or use RVC.")
            else:
                add("warn", "Voice service not running (RVC 8766, Seed-VC 8765, OmniVoice 8767)",
                    "Every sentence will be spoken in the generic voice, whatever "
                    "voice is selected. Start RVC for your own voice, or Seed-VC "
                    "for a volunteer, before pressing start.",
                    "<Applio-env>\\python.exe scripts\\rvc_server.py --model presenter --port 8766")

        # The devices that will be opened: the page's choice, else the
        # config's, else Windows' default. Checking the default when something
        # else was chosen answered a question nobody asked.
        chosen_in = getattr(self.controller, "input_device", None) if self.controller else None
        chosen_out = getattr(self.controller, "output_device", None) if self.controller else None
        with contextlib.suppress(Exception):
            from ..config import load_config

            audio_cfg = load_config().audio
            chosen_in = chosen_in if chosen_in is not None else audio_cfg.input_device
            chosen_out = chosen_out if chosen_out is not None else audio_cfg.output_device
        from ..audio.devices import resolve_device, stutters_under_load

        def opened(spec: object, direction: str) -> tuple[Any, str]:
            """The device a start would open, and a note when it is not the choice."""
            if spec is None:
                return resolve_device(None, direction), ""  # type: ignore[arg-type]
            try:
                return resolve_device(spec, direction), ""  # type: ignore[arg-type]
            except Exception:
                # Start falls back to Windows' default (see candidates()).
                return (resolve_device(None, direction),  # type: ignore[arg-type]
                        f"  - {spec} is not here; Windows' default would open instead")

        try:
            mic, note = opened(chosen_in, "input")
            level = "ok" if mic.latency_ms("input") < 30 and not note else "warn"
            add(level, f"Microphone: {mic.name}",
                f"{mic.hostapi_name}, {mic.latency_ms('input')} ms" + note
                + ("  - a Bluetooth microphone drops to 16 kHz mono over HFP"
                   if mic.likely_bluetooth else ""))
        except Exception as exc:  # pragma: no cover - no sound card
            add("blocked", "No microphone", str(exc))

        try:
            speakers, note = opened(chosen_out, "output")
            if stutters_under_load(speakers):
                add("warn", f"Speakers on {speakers.hostapi_name}: {speakers.name}",
                    f"{speakers.hostapi_name} starves while the pipeline works - measured 45 "
                    "gaps in 8 s, heard as a choppy translation at rehearsal. Choose the "
                    "same speakers' WASAPI entry in the Devices picker." + note)
            else:
                add("warn" if note else "ok", f"Speakers: {speakers.name}",
                    f"{speakers.hostapi_name}, {speakers.latency_ms('output')} ms" + note)
        except Exception as exc:  # pragma: no cover - no sound card
            add("warn", "No speakers resolved", str(exc))

        volunteers = [v for v in self.list_voices() if v["volunteer"]]
        if volunteers:
            add("warn", f"{len(volunteers)} volunteer recording(s) on disk",
                ", ".join(v["name"] for v in volunteers),
                "clone_volunteer.py --forget-all")

        worst = "blocked" if any(r["level"] == "blocked" for r in rows) else (
            "warn" if any(r["level"] == "warn" for r in rows) else "ok")
        return {"level": worst, "checks": rows}

    #: Human names for the codes the backends know. Anything the backends can
    #: do but this table cannot name still appears, under its code.
    _LANG_NAMES = {"tr": "Turkish", "it": "Italian", "en": "English", "es": "Spanish",
                   "fr": "French", "de": "German", "pt": "Portuguese", "fur": "Friulian"}

    def languages(self) -> dict:
        """Sources and targets, from the backends' own tables."""
        from ..mt.ctranslate2_nllb import LANG_TAGS
        from ..tts import backend_for
        from ..tts.kokoro_backend import FALLBACK_LANGUAGES

        names = self._LANG_NAMES
        # Anything NLLB has a tag for can be recognised (Whisper is
        # multilingual) and translated. Whether it can be *spoken* is the
        # synthesisers' business - Kokoro on the GPU, Piper on the CPU for
        # what Kokoro lacks; a target neither can voice is subtitles only.
        sources = sorted(LANG_TAGS)
        targets = []
        for code in sorted(LANG_TAGS):
            speaker = backend_for(code)
            if speaker == "kokoro" and code in FALLBACK_LANGUAGES:
                voice, note = True, (f"no {names.get(code, code)} voice exists; spoken "
                                     f"with the {names.get(FALLBACK_LANGUAGES[code])} "
                                     "frontend, phonetics wrong, disclosed on stage")
            elif speaker == "piper":
                voice, note = True, "spoken by Piper on the CPU - about 0.4 s slower a sentence"
            elif speaker:
                voice, note = True, ""
            else:
                voice, note = False, (f"no {names.get(code, code)} voice on this machine - "
                                      "subtitles only")
            targets.append({"code": code, "name": names.get(code, code),
                            "voice": voice, "note": note})
        chosen_source = getattr(self.controller, "source_lang", None) if self.controller else None
        chosen_target = getattr(self.controller, "target_lang", None) if self.controller else None
        return {
            "sources": sources,
            "names": {c: names.get(c, c) for c in sources},
            "targets": targets,
            "current": {"source": self.state.source_lang, "target": self.state.target_lang},
            "selected": {"source": chosen_source, "target": chosen_target},
            "applies": "at the next start",
        }

    #: What each option costs, as measured in this project. The numbers are the
    #: ones in the ADRs; a model without a measurement is listed without one
    #: rather than with a guess.
    _ASR_OPTIONS = [
        ("large-v3-turbo", "large-v3-turbo",
         "default · 14.00% WER in situ · p95 293 ms · 1213 MiB", "ADR 0002"),
        ("large-v3", "large-v3",
         "+0.56 WER better on FLEURS, 3.93 worse on this voice with hotwords · p95 724 ms",
         "ADR 0002"),
        ("medium", "medium", "smaller and faster; not measured on this voice", ""),
        ("small", "small", "smaller and faster; not measured on this voice", ""),
    ]
    _MT_OPTIONS = [
        ("models/ct2/nllb-200-distilled-600M", "NLLB-200 600M",
         "default · chrF++ 45.4 · 844 MiB · ~0.5 s/sentence", "ADR 0007"),
        # Measured 2026-09-23, 68 s reference, RVC: peak 6942 MiB against
        # 6038 for 600M, mean lag 1.92 s against 1.72. Beside Seed-VC, which
        # holds ~2.2 GB more, it leaves ~350 MiB and is the wrong choice.
        ("models/ct2/nllb-200-distilled-1.3B", "NLLB-200 1.3B",
         "better translation (chrF++ 47.1) · +0.2 s/sentence · fits beside RVC, "
         "NOT beside Seed-VC",
         "ADR 0007, 0013"),
    ]

    #: ADR 0009, 68 s of continuous reading, one speaker. Segments, WER, and
    #: what happened at the extremes. 400 and 500 measured identically.
    _TEMPO_OPTIONS = [
        {"ms": 300, "note": "fastest feel · 14 segments, two fragments under 1.5 s - sentences split at breaths",
         "level": "warn"},
        {"ms": 400, "note": "11 segments · WER 11.41 · identical to 500 on this speaker, 0.1 s sooner",
         "level": "ok"},
        {"ms": 500, "note": "default · 11 segments · WER 11.41 · no truncation · the measured best",
         "level": "ok"},
        {"ms": 700, "note": "8 segments · WER 12.08 · one sentence hit the 15 s hard cut mid-phrase",
         "level": "warn"},
    ]

    #: OmniVoice speaking rates offered, with what they measured as.
    VOICE_SPEEDS = [
        {"speed": 0, "note": "the reference's own pace - measured 6.2 syllables/s, and 9 of 40 "
                             "short sentences came out as another sentence"},
        {"speed": 0.8, "note": "slower still - not measured"},
        {"speed": 0.85, "note": "measured 5.3 syllables/s, Kokoro's pace; 4 of 40 garbled - "
                                "recommended for this audience"},
        {"speed": 0.9, "note": "measured 5.6 syllables/s; 7 of 40 garbled"},
        {"speed": 0.95, "note": "not measured; about the presenter's own reading pace"},
        {"speed": 1.0, "note": "the model's estimate from the reference, no change"},
    ]

    def voice_speed(self) -> dict:
        chosen = getattr(self.controller, "voice_speed", None) if self.controller else None
        configured = None
        with contextlib.suppress(Exception):
            from ..config import load_config

            configured = load_config().tts.conversion.speed
        running = getattr(getattr(self.controller, "translator", None), "voice_speed", None)
        return {"options": self.VOICE_SPEEDS, "selected": chosen, "configured": configured,
                "running": running, "applies": "at once, and at every start",
                "only": "OmniVoice; RVC and Seed-VC keep the synthesiser's pace"}

    def tempo(self) -> dict:
        chosen = getattr(self.controller, "min_silence_ms", None) if self.controller else None
        configured = None
        with contextlib.suppress(Exception):
            from ..config import load_config

            configured = load_config().vad.min_silence_ms
        return {"options": self._TEMPO_OPTIONS, "selected": chosen,
                "configured": configured, "applies": "at the next start"}

    def models(self) -> dict:
        """Recogniser and translator options that exist on this machine."""
        from ..paths import resolve

        hub = resolve("models/huggingface/hub")

        def whisper_cached(name: str) -> bool:
            # faster-whisper resolves these to Systran/deepdml repos in the
            # HF cache; anything not already there would download on stage.
            # Exact on the model name: "large-v3" must not be satisfied by
            # "large-v3-turbo-ct2" sitting in the cache.
            if not hub.is_dir():
                return False
            for entry in hub.glob("models--*--faster-whisper-*"):
                tail = entry.name.split("--")[-1].removeprefix("faster-whisper-")
                if tail in {name, f"{name}-ct2"}:
                    return True
            return False

        asr = [{"id": i, "name": n, "cost": c, "adr": a}
               for i, n, c, a in self._ASR_OPTIONS if whisper_cached(i)]
        mt = [{"id": i, "name": n, "cost": c, "adr": a}
              for i, n, c, a in self._MT_OPTIONS if resolve(i).is_dir()]
        chosen = self.controller
        return {
            "asr": asr,
            "mt": mt,
            "selected": {"asr": getattr(chosen, "asr_model", None) if chosen else None,
                         "mt": getattr(chosen, "mt_model", None) if chosen else None},
            "applies": "at the next start",
        }

    def live_conversion_address(self) -> tuple[str, int] | None:
        """The voice service that is running: the configured port, else the other known one.

        Start does the same search, so this row names the service the next
        Start will actually use.
        """
        if self.conversion_address is None:
            return None
        from ..tts.conversion import find_voice_service

        host, port = self.conversion_address
        found = find_voice_service(host, port)
        return (host, found) if found else None

    def conversion_service(self) -> dict:
        """Ask the running service what it is. Empty when nothing answers."""
        address = self.live_conversion_address()
        if address is None:
            return {}
        host, port = address
        info: dict = {"port": port}
        try:
            from ..tts.conversion import VoiceConverter

            info.update(VoiceConverter(host=host, port=port, timeout=2.0).ping())
        except Exception:
            pass
        return info

    def conversion_reachable(self) -> bool:
        """Is either voice service answering? A one-second socket probe each."""
        return self.live_conversion_address() is not None

    def metrics(self) -> dict:
        """Gate statistics and the lag distribution, or empty when nothing runs."""
        out: dict[str, Any] = {"running": False, "gate": None, "lag": None}
        controller = self.controller
        translator = getattr(controller, "translator", None) if controller else None
        if translator is None:
            return out
        out["running"] = True

        gate = None
        playback = getattr(translator, "playback", None)
        if playback is not None:
            gate = getattr(playback, "gate", None)
        if gate is not None:
            # `remaining_ms` is a method on HalfDuplexGate and `is_open` is a
            # property. The first version of this got that backwards, and the
            # test stand-in - a bare object with attributes - agreed with the
            # mistake. The test below now uses the real gate.
            stats = gate.stats.as_dict()
            remaining = float(
                gate.remaining_ms() if callable(gate.remaining_ms) else gate.remaining_ms
            )
            # While playback holds the gate the remaining time is infinite -
            # it reopens when the audio ends, not on a clock - and JSON has no
            # infinity. `null` here means "held", and the page shows it as
            # such. Found by polling this route while a sentence was actually
            # being spoken, which is the only time the number matters.
            out["gate"] = {
                **stats,
                "enabled": gate.enabled,
                "open": bool(gate.is_open),
                "held": not math.isfinite(remaining),
                "tail_ms": gate.tail_ms,
                "remaining_ms": round(remaining, 1) if math.isfinite(remaining) else None,
            }

        # Is the microphone hearing anything at all? The first question on a
        # stage when nothing appears, and the one this page could not answer:
        # `peak_amplitude` says whether audio arrives, `segments_detected`
        # whether the VAD thinks any of it is speech. Between them they locate
        # a silent run in one glance instead of a log dive.
        transcriber = getattr(translator, "transcriber", None)
        capture = getattr(transcriber, "_capture", None) if transcriber else None
        if capture is not None and getattr(capture, "stats", None) is not None:
            out["capture"] = capture.stats.as_dict()
        if transcriber is not None and getattr(transcriber, "stats", None) is not None:
            with contextlib.suppress(Exception):
                summary_t = transcriber.stats.summary()
                out["transcriber"] = {k: summary_t[k] for k in (
                    "segments_detected", "segments_transcribed", "segments_empty",
                    "partials_emitted", "repetition_loops_caught", "blocks_seen",
                ) if k in summary_t}

        summary = translator.stats.summary()
        out["lag"] = summary.get("lag_s")
        out["counts"] = {
            "delivered": summary.get("delivered", 0),
            "partials": summary.get("partials_delivered", 0),
            "dropped_backlog": summary.get("dropped_backlog", 0),
            "translation_failures": summary.get("translation_failures", 0),
            "synthesis_failures": summary.get("synthesis_failures", 0),
            "conversion_failures": summary.get("conversion_failures", 0),
            "conversion_skipped": summary.get("conversion_skipped", 0),
        }
        return out

    def friulian_rows(self) -> list[dict]:
        """Parse the review sheet into rows. Returns [] when it has not been built.

        Reads ``data/friulian/review-sheet.md`` - the file a native speaker is
        asked to mark up - rather than keeping a second copy of the sentences
        in the page. One list, and the reviewer's file is the one that is true.
        """
        import re as _re

        from ..paths import resolve

        path = resolve("data/friulian/review-sheet.md")
        if not path.is_file():
            return []

        rows: list[dict] = []
        current: dict[str, Any] | None = None

        def field(line: str, label: str) -> str | None:
            match = _re.match(rf"^-\s+\*\*{label}[^:]*:\*\*\s*(.*)$", line.strip())
            if match is None:
                return None
            # The Friulian is bolded inside the bullet; strip that emphasis.
            return match.group(1).strip().strip("*").strip()

        for line in path.read_text(encoding="utf-8").splitlines():
            heading = _re.match(r"^###\s+(\d+)\.", line.strip())
            if heading:
                if current is not None:
                    rows.append(current)
                current = {"n": int(heading.group(1)), "friulian": "",
                           "italian": "", "correction": ""}
                continue
            if current is None:
                continue
            for key, label in (("italian", "Italian"), ("friulian", "Friulian"),
                               ("correction", "Correction")):
                value = field(line, label)
                if value is not None:
                    current[key] = value
                    break
        if current is not None:
            rows.append(current)
        return [r for r in rows if r["friulian"] or r["italian"]]

    def friulian_native(self) -> dict | None:
        """The native speaker's clips and their attribution, if a recording is here.

        ``data/friulian/marco/clips/index.json`` is written by the segmenting
        step and carries the speaker, the source, the licence and the list.
        Nothing is invented here: no index, no section on the page.
        """
        from ..paths import resolve

        index = resolve("data/friulian/marco/clips/index.json")
        if not index.is_file():
            return None
        with contextlib.suppress(Exception):
            data = json.loads(index.read_text(encoding="utf-8"))
            clips = [c for c in data.get("clips", [])
                     if (index.parent / c.get("file", "")).is_file()]
            # The WAVs are not in git. A fresh clone has the index and none of
            # the files, and a section with no play buttons is not a section.
            if not clips:
                return None
            return {k: data.get(k, "") for k in ("speaker", "source", "licence", "note")} | \
                {"clips": clips}
        return None

    def measurements(self) -> dict:
        """The most recent VRAM and output-latency reports, if they were run."""
        from ..paths import resolve

        out: dict[str, Any] = {"vram": None, "latency_runs": []}

        vram_dir = resolve("runs/vram")
        if vram_dir.is_dir():
            reports = sorted(vram_dir.glob("*.json"))
            if reports:
                with contextlib.suppress(Exception):
                    out["vram"] = json.loads(reports[-1].read_text(encoding="utf-8"))

        audio_dir = resolve("runs/audio")
        if audio_dir.is_dir():
            # Newest last, so the table reads in the order they were taken.
            for report in sorted(audio_dir.glob("device-*.json"))[-3:]:
                with contextlib.suppress(Exception):
                    data = json.loads(report.read_text(encoding="utf-8"))
                    trip = data.get("roundtrip") or {}
                    if "median_ms" not in trip:
                        continue
                    out["latency_runs"].append({
                        "label": data.get("label", report.stem),
                        "median_ms": trip["median_ms"],
                        "min_ms": trip.get("min_ms"),
                        "max_ms": trip.get("max_ms"),
                        "spread_ms": trip.get("spread_ms"),
                    })
        return out

    def list_voices(self) -> list[dict]:
        """Reference recordings on disk, and whether each is a volunteer's."""
        from ..paths import resolve

        out: list[dict] = []
        root = resolve("data/voices")
        if not root.is_dir():
            return out
        configured_path, configured_consent = self.configured_voice or (None, "")
        for path in sorted(root.glob("*.wav")):
            configured = configured_path is not None and                 Path(configured_path).resolve() == path.resolve()
            out.append({"name": path.stem, "path": str(path), "volunteer": False,
                        "consent": configured_consent if configured else "",
                        "configured": configured, **self._clip_quality(path)})
        volunteers = root / "volunteers"
        if volunteers.is_dir():
            for person in sorted(p for p in volunteers.glob("*") if p.is_dir()):
                reference = person / "reference.wav"
                if not reference.exists():
                    continue
                about: dict = {}
                meta = person / "consent.json"
                if meta.exists():
                    with contextlib.suppress(Exception):
                        about = json.loads(meta.read_text(encoding="utf-8"))
                out.append({"name": person.name, "path": str(reference),
                            "volunteer": True, "consent": about.get("consent", ""),
                            "derived_from": about.get("derived_from", ""),
                            **self._clip_quality(reference)})
        # Invented in the Studio from attributes: no one's voice, and the
        # consent record says so. Usable live like any other reference.
        designed = root / "designed"
        if designed.is_dir():
            for folder in sorted(p for p in designed.glob("*") if p.is_dir()):
                reference = folder / "reference.wav"
                meta = folder / "consent.json"
                if not reference.exists() or not meta.exists():
                    continue
                data: dict = {}
                with contextlib.suppress(Exception):
                    data = json.loads(meta.read_text(encoding="utf-8"))
                out.append({"name": folder.name, "path": str(reference), "volunteer": False,
                            "designed": True, "consent": data.get("consent", ""),
                            "design": data.get("design", ""),
                            **self._clip_quality(reference)})
        return out

    #: Measuring a clip costs a file read, and the Voices tab is polled. Keyed
    #: by (path, mtime) so a rebuilt reference is re-measured and an unchanged
    #: one is not.
    _quality_cache: dict[tuple[str, float], dict] = {}

    def _clip_quality(self, path: Path) -> dict:
        """How much of this recording the microphone driver removed.

        The single number that decided cloned-voice quality in this project.
        The laptop's noise suppression gates 16-50% of everything it captures
        to digital silence, and the first reference chosen by ear was 49.5%
        holes - which is what made the clone sound poor and sent the diffusion
        step count up to compensate for it. A phone recording with no gating
        took WER from 14.00% to 11.41% and let the steps come back down.

        So it is a column in the table rather than a warning behind something:
        an operator picking a voice thirty seconds before a talk should see
        which recording is damaged without having to ask.
        """
        try:
            key = (str(path), path.stat().st_mtime)
        except OSError:
            return {}
        cached = self._quality_cache.get(key)
        if cached is not None:
            return cached

        try:
            import soundfile as sf

            from ..tts.reference import measure_clip

            audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            stats = measure_clip(audio, rate, path)
            result = {
                "duration_s": round(stats.duration_s, 1),
                "gated_fraction": round(stats.gated_fraction, 3),
                "peak": round(stats.peak, 3),
                "rms_dbfs": round(stats.rms_dbfs, 1),
            }
        except Exception as exc:  # pragma: no cover - a bad file must not 500
            log.debug("could not measure %s: %s", path, exc)
            result = {}
        self._quality_cache[key] = result
        return result

    def _record_volunteer(self, name: str, consent: str, seconds: float) -> dict:
        """Record from the microphone and build a reference. Blocking.

        Run on a worker thread by the endpoint, because opening the sound card
        and waiting fifteen seconds inside an async handler would stall every
        other request - including the subtitle stream.
        """
        import numpy as np
        import soundfile as sf

        from ..audio.capture import AudioCapture
        from ..paths import ensure_dir
        from ..tts.base import VoiceProfile
        from ..tts.reference import build_reference, measure_clip

        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")[:40]
        if not safe:
            return {"error": "that name has no usable characters"}

        rate = 16000
        capture = AudioCapture(device=None, sample_rate=rate, block_ms=32, gate=None)
        capture.start()
        try:
            import time

            time.sleep(0.3)
            capture.drain()
            chunks: list = []
            collected = 0
            wanted = int(seconds * rate)
            deadline = time.monotonic() + seconds * 3 + 3.0
            while collected < wanted and time.monotonic() < deadline:
                block = capture.read(timeout=1.0)
                if block is None:
                    continue
                chunks.append(block)
                collected += block.size
        finally:
            capture.stop()

        if not chunks:
            return {"error": "nothing was captured - check the microphone"}
        audio = np.concatenate(chunks)[:wanted]
        stats = measure_clip(audio, rate)
        if stats.duration_s < 3.0 or stats.peak < 0.01:
            return {"error": f"only {stats.duration_s:.1f}s captured at peak "
                             f"{stats.peak:.3f} - check the microphone"}

        person = ensure_dir(f"data/voices/volunteers/{safe}")
        sf.write(person / "raw.wav", audio, rate, subtype="FLOAT")
        reference = build_reference([(audio, rate)], target_seconds=seconds,
                                    sample_rate=rate)
        ref_path = person / "reference.wav"
        sf.write(ref_path, reference, rate, subtype="FLOAT")

        # The consent record sits beside the audio so the two cannot drift
        # apart, and a recording found later can be traced to a signature.
        (person / "consent.json").write_text(json.dumps({
            "name": name,
            "consent": consent,
            "recorded": datetime.now().astimezone().isoformat(),
            "seconds": round(stats.duration_s, 1),
            "delete_after": "the end of this session",
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        VoiceProfile(name=safe, reference_path=str(ref_path), consent=consent).validate()
        log.warning("recorded %s (%.1fs, %.0f%% gated)", name, stats.duration_s,
                    100 * stats.gated_fraction)
        return {
            "name": safe,
            "path": str(ref_path),
            "seconds": round(stats.duration_s, 1),
            "peak": round(stats.peak, 3),
            "gated_pct": round(100 * stats.gated_fraction, 1),
            "warning": ("the microphone removed most of that recording, so the "
                        "clone will sound poor")
            if stats.gated_fraction > 0.4 else "",
        }

    def forget_volunteers(self) -> dict:
        """Delete every volunteer recording, and say what went."""
        import shutil

        from ..paths import resolve

        root = resolve("data/voices/volunteers")
        removed: list[str] = []
        freed = 0
        if root.is_dir():
            for person in [p for p in root.glob("*") if p.is_dir()]:
                freed += sum(f.stat().st_size for f in person.rglob("*") if f.is_file())
                shutil.rmtree(person, ignore_errors=True)
                removed.append(person.name)
        # A voice whose recording no longer exists must not stay loaded.
        if removed and self.on_voice is not None:
            with contextlib.suppress(Exception):
                self.on_voice(None, "")
            self.state.voice = "generic"
            self.publish_state()
        log.warning("deleted %d volunteer recording(s)", len(removed))
        return {"removed": removed, "bytes": freed}

    # -- the studio -------------------------------------------------------

    @property
    def studio(self) -> Any:
        if self._studio is None:
            from .studio import Studio

            self._studio = Studio(self.list_voices)
        return self._studio

    def pipeline_running(self) -> bool:
        return bool(self.controller is not None and getattr(self.controller, "running", False))

    def play_audio(self, audio: Any, rate: int) -> str:
        """Play a studio clip through the speakers, and say which way it went.

        With the pipeline running it goes through the pipeline's own playback,
        whose gate keeps the microphone shut while it plays - otherwise the
        system would hear the clip and translate it. With the pipeline
        stopped, a playback of its own on the configured output.
        """
        translator = getattr(self.controller, "translator", None) if self.controller else None
        playback = getattr(translator, "playback", None)
        if playback is not None and self.pipeline_running():
            playback.submit(audio, rate)
            return "pipeline"
        from ..audio.playback import AudioPlayback

        device = None
        with contextlib.suppress(Exception):
            from ..config import load_config

            device = load_config().audio.output_device
        out = AudioPlayback(device=device, sample_rate=rate)

        def run() -> None:
            try:
                out.play(audio, blocking=True)
            finally:
                out.stop()

        threading.Thread(target=run, name="studio-play", daemon=True).start()
        return "speakers"

    # -- lifecycle --------------------------------------------------------

    def build_app(self) -> Any:
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

        app = FastAPI(title="parlIAmo")

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return APP.read_text(encoding="utf-8")

        @app.get("/screen", response_class=HTMLResponse)
        async def screen() -> str:
            """What the room sees. Opened on the second display."""
            return SCREEN.read_text(encoding="utf-8")

        @app.get("/devices")
        async def devices() -> JSONResponse:
            """Audio devices, so the operator can see what is selected.

            Windows exposes the same hardware once per host API with wildly
            different latency - the system default here was an MME clone at
            90 ms against 2 ms on WASAPI - so the list shows the host API and
            the latency rather than just a name.
            """
            from ..audio.devices import describe_devices

            # `selected` and `live` used to be returned only from POST, so the
            # page could not show which device was actually open unless the
            # operator changed one. That matters now that a device which
            # refuses to open is silently replaced by one that works: the
            # substitution has to be visible, or the operator believes they are
            # on the lavalier while the room is hearing the laptop.
            chosen = (await asyncio.to_thread(self.controller.devices)
                      if self.controller is not None else {})
            try:
                return JSONResponse({
                    "input": await asyncio.to_thread(describe_devices, "input"),
                    "output": await asyncio.to_thread(describe_devices, "output"),
                    **chosen,
                })
            except Exception as exc:  # pragma: no cover - no sound card
                return JSONResponse(
                    {"error": str(exc), "input": [], "output": [], **chosen}
                )

        @app.get("/pipeline")
        async def pipeline() -> JSONResponse:
            if self.controller is None:
                return JSONResponse({"status": "external", "detail": "started from the CLI"})
            return JSONResponse(self.controller.state())

        @app.post("/pipeline/{action}")
        async def pipeline_control(action: str) -> JSONResponse:
            if self.controller is None:
                return JSONResponse(
                    {"error": "this run was started from the CLI"}, status_code=409)
            if action == "start":
                return JSONResponse(await asyncio.to_thread(self.controller.start))
            if action == "stop":
                return JSONResponse(await asyncio.to_thread(self.controller.stop))
            return JSONResponse({"error": f"unknown action {action}"}, status_code=400)

        @app.post("/voices/record")
        async def record_voice(request: Request) -> JSONResponse:
            """Record a volunteer, build a reference, and load it.

            The whole point of the demonstration is that it takes fifteen
            seconds, so it cannot require leaving the room for a terminal.

            Consent is checked before the microphone opens, not after: a
            recording made and then refused is a recording that existed.
            """
            body = await request.json() if await request.body() else {}
            name = str(body.get("name", "")).strip()
            consent = str(body.get("consent", "")).strip()
            seconds = float(body.get("seconds", 15))
            use_now = bool(body.get("use", True))

            if not name:
                return JSONResponse({"error": "a name is required"}, status_code=400)
            if not consent:
                return JSONResponse(
                    {"error": "a consent record is required before recording anyone"},
                    status_code=400)

            try:
                result = await asyncio.to_thread(
                    self._record_volunteer, name, consent, seconds)
            except Exception as exc:
                log.exception("recording failed")
                return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                                    status_code=500)

            if result.get("error"):
                return JSONResponse(result, status_code=400)

            if use_now and self.on_voice is not None:
                with contextlib.suppress(Exception):
                    self.on_voice(result["path"], consent)
                    self.state.voice = name
                    self.publish_state()
            return JSONResponse(result)

        @app.post("/devices")
        async def choose_device(request: Request) -> JSONResponse:
            """Pick the microphone and speakers from the page.

            Taking effect at the next start rather than immediately: PortAudio
            streams are opened when the pipeline is built, and swapping a
            device under a running stream is how you get a dead microphone
            with no error. The page says which is selected and which is live.
            """
            body = await request.json() if await request.body() else {}
            if self.controller is None:
                return JSONResponse({"error": "this run was started from the CLI"},
                                    status_code=409)
            for key in ("input", "output"):
                if key in body:
                    value = body[key]
                    setattr(self.controller, f"{key}_device",
                            None if value in (None, "", "default") else value)
            return JSONResponse(await asyncio.to_thread(self.controller.devices))

        @app.get("/preflight")
        async def preflight() -> JSONResponse:
            """What would go wrong if you started right now.

            The pre-flight list exists because of one failure mode: starting
            with speakers live and `output_latency_ms` at 0, which turns the
            microphone into a feedback loop within a few sentences. That is the
            only remaining item that can ruin the demonstration outright, so it
            is checked where it cannot be missed rather than logged where it
            can.
            """
            return JSONResponse(await asyncio.to_thread(self.checks))

        @app.get("/metrics")
        async def metrics() -> JSONResponse:
            """The gate, and the shape of the lag. What the operator watches.

            Split from ``/state`` because these are polled on a timer while the
            state is pushed on every delivery, and because none of it exists
            until a pipeline is running.
            """
            return JSONResponse(await asyncio.to_thread(self.metrics))

        @app.post("/flush")
        async def flush() -> JSONResponse:
            """Stop the audio that is playing *now*.

            Mute prevents the next sentence; this drops the one already in the
            speakers and releases the gate with it. They are different controls
            and on stage the difference is a few seconds of the wrong thing
            still being said.
            """
            if self.controller is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            dropped = await asyncio.to_thread(self.controller.flush)
            log.warning("playback flushed from the operator page")
            return JSONResponse({"flushed": dropped})

        @app.get("/streaming")
        async def streaming() -> JSONResponse:
            if self.controller is None:
                return JSONResponse({"mode": "off", "enabled": False, "available": False})
            mode = self.controller.streaming()
            return JSONResponse({"mode": mode, "enabled": mode != "off", "available": True})

        @app.post("/streaming")
        async def set_streaming(request: Request) -> JSONResponse:
            """Speak sentences as they end, or wait for the pause. Live switch.

            The presenter's call, made knowing the cost: a sentence goes out
            on the strength of where the recogniser put its full stop, and a
            stop one word early speaks a Turkish sentence without the verb
            and the negation that finish it.
            """
            if self.controller is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            body = await request.json() if await request.body() else {}
            if "mode" in body:
                wanted = body["mode"]
            elif "enabled" in body:
                wanted = "sentence" if body["enabled"] else "off"
            else:  # no body: cycle off -> sentence -> chunk -> off
                order = self.controller.STREAM_MODES
                wanted = order[(order.index(self.controller.streaming()) + 1) % len(order)]
            try:
                mode = self.controller.set_streaming(wanted)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            log.warning("streaming mode %s from the operator page - %s", mode, {
                "off": "sentences wait for the pause",
                "sentence": "sentences are spoken as they end",
                "chunk": "words are spoken in chunks as they settle",
            }[mode])
            return JSONResponse({"mode": mode, "enabled": mode != "off", "available": True})

        @app.get("/friulian")
        async def friulian() -> JSONResponse:
            """The prepared sentences, and the blanks a native speaker fills in."""
            return JSONResponse({"rows": await asyncio.to_thread(self.friulian_rows),
                                 "native": await asyncio.to_thread(self.friulian_native)})

        @app.get("/friulian/audio/{kind}/{name}")
        async def friulian_audio(kind: str, name: str):
            """One clip: the pipeline's rendering, or the native speaker's.

            The native clips are a recording of a real person, played as
            recorded. They are served from the repo so the talk needs no
            network, and never handed to the synthesiser: the licence covers
            the recording, not the speaker's voice.
            """
            from fastapi.responses import FileResponse

            from ..paths import resolve

            if kind not in {"pipeline", "native"} or "/" in name or "\\" in name \
                    or not name.endswith(".wav"):
                return JSONResponse({"error": "no such clip"}, status_code=404)
            folder = resolve("data/friulian/audio" if kind == "pipeline"
                             else "data/friulian/marco/clips")
            path = (folder / name)
            if not path.is_file() or path.resolve().parent != folder.resolve():
                return JSONResponse({"error": "no such clip"}, status_code=404)
            return FileResponse(str(path), media_type="audio/wav")

        @app.get("/measurements")
        async def measurements() -> JSONResponse:
            """VRAM and output latency, read back from the reports that measured them.

            Shown in Setup rather than quoted in a document, because the number
            that matters on the night is the one measured in *that* room, and an
            operator who cannot see it has to trust a figure from a laptop on a
            different continent.
            """
            return JSONResponse(await asyncio.to_thread(self.measurements))

        @app.get("/languages")
        async def languages() -> JSONResponse:
            """Which pairs the loaded backends can actually do.

            Read from the backends' own tables rather than listed here, so the
            page cannot offer a language the translator has no tag for or the
            synthesiser no voice for. A target with no voice is still offered
            - as subtitles only - and says so.
            """
            return JSONResponse(self.languages())

        @app.post("/languages")
        async def choose_languages(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            if self.controller is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            known = self.languages()
            source = body.get("source")
            target = body.get("target")
            if source is not None and source not in known["sources"]:
                return JSONResponse({"error": f"no source support for {source!r}"},
                                    status_code=400)
            if target is not None and target not in {t["code"] for t in known["targets"]}:
                return JSONResponse({"error": f"no target support for {target!r}"},
                                    status_code=400)
            if source is not None:
                self.controller.source_lang = source
            if target is not None:
                self.controller.target_lang = target
            log.warning("languages chosen from the operator page: %s -> %s (next start)",
                        self.controller.source_lang, self.controller.target_lang)
            return JSONResponse(self.languages())

        @app.get("/tempo")
        async def tempo() -> JSONResponse:
            """The commit wait and what each setting was measured to cost.

            From ADR 0009, on 68 s of the presenter's continuous speech. The
            figures are that speaker on that day; a different pace measures
            differently, and the page says so.
            """
            return JSONResponse(self.tempo())

        @app.post("/tempo")
        async def choose_tempo(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            if self.controller is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            wanted = body.get("min_silence_ms")
            allowed = {o["ms"] for o in self.tempo()["options"]}
            if wanted not in allowed:
                return JSONResponse({"error": f"min_silence_ms must be one of {sorted(allowed)}"},
                                    status_code=400)
            self.controller.min_silence_ms = int(wanted)
            log.warning("commit wait %d ms chosen from the operator page (next start)", wanted)
            return JSONResponse(self.tempo())

        @app.get("/voice_speed")
        async def voice_speed_get() -> JSONResponse:
            return JSONResponse(self.voice_speed())

        @app.post("/voice_speed")
        async def voice_speed_set(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            if self.controller is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            wanted = float(body.get("speed") or 0)
            if wanted and not 0.6 <= wanted <= 1.4:
                return JSONResponse({"error": "speed must be between 0.6 and 1.4, or 0 for the "
                                              "reference's own pace"}, status_code=400)
            self.controller.voice_speed = wanted
            translator = getattr(self.controller, "translator", None)
            if translator is not None:
                # From the next sentence: the rate is sent with each one.
                translator.voice_speed = wanted or None
            log.warning("cloned voice speed %s from the operator page",
                        wanted or "as the reference")
            return JSONResponse(self.voice_speed())

        @app.get("/models")
        async def models() -> JSONResponse:
            """The recogniser and translator options, each with its measured cost.

            Only models that are actually on this machine are listed - the
            venue has no internet, and an option that downloads on first use
            is an option that fails on stage.
            """
            return JSONResponse(await asyncio.to_thread(self.models))

        @app.post("/models")
        async def choose_models(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            if self.controller is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            known = self.models()
            for kind in ("asr", "mt"):
                if kind not in body:
                    continue
                wanted = body[kind]
                if wanted is not None and wanted not in {o["id"] for o in known[kind]}:
                    return JSONResponse({"error": f"{kind}: {wanted!r} is not on this machine"},
                                        status_code=400)
                setattr(self.controller, f"{kind}_model", wanted)
            log.warning("models chosen from the operator page: asr=%s mt=%s (next start)",
                        self.controller.asr_model, self.controller.mt_model)
            return JSONResponse(await asyncio.to_thread(self.models))

        @app.get("/voices")
        async def voices() -> JSONResponse:
            return JSONResponse({"voices": await asyncio.to_thread(self.list_voices)})

        @app.get("/studio")
        async def studio_state() -> JSONResponse:
            from ..tts.omnivoice_options import ADVANCED, DESIGN, FLAGS, NONVERBAL
            from .studio import AUTO_VOICE, LANGUAGES

            def gather() -> dict:
                service = self.studio.status()
                return {"service": service, "voices": self.studio.usable_voices(),
                        "clips": self.studio.clips(), "languages": LANGUAGES,
                        "all_languages": self.studio.languages() if service["up"] else [],
                        "advanced": ADVANCED, "flags": FLAGS, "nonverbal": NONVERBAL,
                        "design": DESIGN, "auto_voice": AUTO_VOICE,
                        "pipeline_running": self.pipeline_running()}

            return JSONResponse(await asyncio.to_thread(gather))

        async def studio_call(fn, *args) -> JSONResponse:
            """Run a studio action off the event loop; a refusal is a 400 with its reason."""
            from .studio import StudioError

            try:
                return JSONResponse(await asyncio.to_thread(fn, *args))
            except (StudioError, ValueError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception as exc:  # pragma: no cover - a control must not fail silently
                log.exception("studio action failed")
                return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

        @app.post("/studio/speak")
        async def studio_speak(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}

            def run() -> dict:
                clip = self.studio.speak(
                    str(body.get("text", "")), str(body.get("language", "it")),
                    str(body.get("voice", "")), body.get("options") or {},
                    body.get("seed"), body.get("steps"), bool(body.get("redesign")))
                log.warning("studio: %s spoke %.1f s (%s, seed %s)", clip["voice"],
                            clip["seconds"], clip["language"], clip.get("seed"))
                if body.get("play"):
                    audio, rate = self.studio.load(clip["id"])
                    clip["played"] = self.play_audio(audio, rate)
                return clip

            return await studio_call(run)

        @app.post("/studio/design")
        async def studio_design(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            return await studio_call(
                self.studio.design, str(body.get("name", "")), str(body.get("gender", "")),
                str(body.get("age", "")), str(body.get("pitch", "")),
                str(body.get("accent", "") or ""), bool(body.get("whisper", False)),
                body.get("seed"))

        @app.post("/studio/upload")
        async def studio_upload(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            return await studio_call(
                self.studio.upload, str(body.get("name", "")), str(body.get("consent", "")),
                str(body.get("data", "")), str(body.get("filename", "")))

        @app.post("/studio/derive")
        async def studio_derive(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            return await studio_call(
                self.studio.derive, str(body.get("name", "")), str(body.get("source", "")),
                body.get("semitones", 0), body.get("tempo", 1))

        @app.post("/studio/voice/delete")
        async def studio_delete_voice(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}

            def run() -> dict:
                gone = self.studio.delete_voice(str(body.get("voice", "")))
                # A voice whose recording is gone must not stay loaded.
                path = Path(gone["path"])
                if self.state.voice in {gone["deleted"], path.stem, path.parent.name}:
                    if self.on_voice is not None:
                        with contextlib.suppress(Exception):
                            self.on_voice(None, "")
                    self.state.voice = "generic"
                    self.publish_state()
                log.warning("studio: deleted voice %s", gone["deleted"])
                return gone

            return await studio_call(run)

        @app.get("/studio/transcript")
        async def studio_transcript(voice: str) -> JSONResponse:
            return await studio_call(lambda: {"transcript": self.studio.transcript(voice)})

        @app.post("/studio/transcript")
        async def studio_set_transcript(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            return await studio_call(self.studio.set_transcript, str(body.get("voice", "")),
                                     str(body.get("text", "")))

        @app.post("/studio/prepare")
        async def studio_prepare(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            return await studio_call(self.studio.prepare, str(body.get("voice", "")))

        @app.get("/studio/clip/{clip_id}")
        async def studio_clip(clip_id: str):
            from fastapi.responses import FileResponse

            path = self.studio.clip_file(clip_id)
            if path is None:
                return JSONResponse({"error": "no such clip"}, status_code=404)
            return FileResponse(str(path), media_type="audio/wav",
                                filename=f"{clip_id}.wav")

        @app.post("/studio/play/{clip_id}")
        async def studio_play(clip_id: str) -> JSONResponse:
            from .studio import StudioError

            try:
                audio, rate = await asyncio.to_thread(self.studio.load, clip_id)
            except StudioError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            return JSONResponse({"played": await asyncio.to_thread(self.play_audio, audio, rate)})

        @app.delete("/studio/clip/{clip_id}")
        async def studio_delete(clip_id: str) -> JSONResponse:
            gone = await asyncio.to_thread(self.studio.delete, clip_id)
            return JSONResponse({"deleted": gone}, status_code=200 if gone else 404)

        @app.delete("/voices")
        async def forget_voices() -> JSONResponse:
            """Delete every volunteer recording. The consent form promises it."""
            return JSONResponse(await asyncio.to_thread(self.forget_volunteers))

        @app.get("/state")
        async def state() -> JSONResponse:
            return JSONResponse(self.state.as_dict())

        @app.post("/mute")
        async def mute(request: Request) -> JSONResponse:
            body = await request.json() if await request.body() else {}
            wanted = bool(body.get("muted", not self.state.muted))
            self.state.muted = wanted
            if self.on_mute is not None:
                try:
                    await asyncio.to_thread(self.on_mute, wanted)
                except Exception:  # pragma: no cover - a control must not 500
                    log.exception("mute handler raised")
            log.warning("output %s from the operator page", "MUTED" if wanted else "unmuted")
            self.publish_state()
            return JSONResponse(self.state.as_dict())

        @app.post("/pause")
        async def pause(request: Request) -> JSONResponse:
            """Stop or resume listening. The microphone keeps running - its
            meter still moves - but nothing it hears is transcribed."""
            body = await request.json() if await request.body() else {}
            wanted = bool(body.get("paused", not self.state.paused))
            self.state.paused = wanted
            if self.on_pause is not None:
                try:
                    await asyncio.to_thread(self.on_pause, wanted)
                except Exception:  # pragma: no cover - a control must not 500
                    log.exception("pause handler raised")
            log.warning("listening %s from the operator page", "PAUSED" if wanted else "resumed")
            self.publish_state()
            return JSONResponse(self.state.as_dict())

        @app.post("/voice")
        async def voice(request: Request) -> JSONResponse:
            """Swap the reference voice mid-talk, or drop back to generic.

            Refuses without a consent record, exactly as the startup path does.
            The moment a voice is swapped in front of an audience is when that
            check is most likely to be skipped, so it is not skippable here.
            """
            body = await request.json() if await request.body() else {}
            path = body.get("path") or None
            consent = str(body.get("consent", ""))
            # The presenter's own voice carries its consent in the config. A
            # request for that file with no consent attached means that one -
            # the page has no other way to know it, and refusing the one voice
            # the whole setup was built around is not the safe direction.
            if path and not consent and self.configured_voice:
                configured_path, configured_consent = self.configured_voice
                with contextlib.suppress(OSError):
                    if Path(path).resolve() == Path(configured_path).resolve():
                        consent = configured_consent
            if self.on_voice is None:
                return JSONResponse({"error": "no pipeline attached"}, status_code=503)
            try:
                await asyncio.to_thread(self.on_voice, path, consent)
            except ValueError as exc:
                # The consent refusal. A 400 and the reason, not a 500.
                log.warning("voice change refused: %s", exc)
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception as exc:  # pragma: no cover
                log.exception("voice change failed")
                return JSONResponse({"error": str(exc)}, status_code=500)

            self.state.voice = "generic" if path is None else Path(path).stem
            log.warning("voice is now %s", self.state.voice)
            self.publish_state()
            return JSONResponse(self.state.as_dict())

        @app.get("/events")
        async def events(request: Request) -> StreamingResponse:
            queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_DEPTH)
            with self._lock:
                self._subscribers.append(queue)

            async def stream():
                # State first, then the recent sentences, so a browser that
                # connects mid-talk shows the room what it missed rather than
                # a blank screen.
                yield _sse(json.dumps({"kind": "state", "state": self.state.as_dict()}))
                for past in list(self._history):
                    yield _sse(json.dumps(
                        {"kind": "delivery", "event": past,
                         "state": self.state.as_dict()}, ensure_ascii=False))
                try:
                    while True:
                        if await request.is_disconnected():
                            return
                        try:
                            message = await asyncio.wait_for(queue.get(), timeout=15.0)
                        except TimeoutError:
                            # A comment frame keeps proxies and the browser
                            # from deciding the connection died.
                            yield ": keep-alive\n\n"
                            continue
                        yield _sse(message)
                finally:
                    with self._lock:
                        if queue in self._subscribers:
                            self._subscribers.remove(queue)

            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return app

    def bind(self, attempts: int = 8) -> socket.socket:
        """Claim the port *before* anything is printed, and return the socket.

        Binding here rather than leaving it to uvicorn on its own thread is the
        whole point. The previous version signalled "ready" as soon as the event
        loop existed, which is before the bind is even attempted, so a port
        already in use produced::

            parlIAmo is at http://127.0.0.1:8770/
            ERROR: [Errno 10048] error while attempting to bind on address ...

        - a URL that was never going to work, printed as though it would, with
        the failure below it in a colour nobody reads. The address handed to the
        operator has to be the address that is actually held.

        A busy port is nearly always an earlier run that never exited, so step
        forward to the next free one rather than refuse to start. Whoever prints
        :attr:`url` afterwards gets the port we actually won.
        """
        first = self.port
        last: OSError | None = None
        for offset in range(attempts):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Deliberately no SO_REUSEADDR. On Windows it lets a second process
            # bind a port another process is actively listening on, and requests
            # then land on whichever socket the kernel happens to pick. That
            # already cost hours once, on the conversion server.
            try:
                sock.bind((self.host, first + offset))
            except OSError as exc:
                sock.close()
                last = exc
                continue
            sock.listen(64)
            self.port = first + offset
            if offset:
                log.warning(
                    "port %d is in use (an earlier run still holding it?); "
                    "serving on %d instead", first, self.port,
                )
            return sock

        raise OSError(
            f"no free port in {first}-{first + attempts - 1} for the operator page. "
            f"Last error: {last}. Close the earlier parlIAmo, or pass --ui-port."
        ) from last

    def start(self) -> str:
        """Start serving on a background thread. Returns the URL that works."""
        import uvicorn

        sock = self.bind()
        # log_config=None is load-bearing. uvicorn's default is to run
        # logging.config.dictConfig on its own LOGGING_CONFIG, and dictConfig
        # begins by calling logging.shutdown() on *every* handler in the
        # process - ours included. The JSONL handler stayed attached to the
        # root logger with its file closed, and the first warning from the
        # model-loading thread produced "I/O operation on closed file" in the
        # console and a run with no events.jsonl. uvicorn's own loggers are
        # left to propagate to root, where our handlers already are.
        config = uvicorn.Config(
            self.build_app(), host=self.host, port=self.port,
            log_level="warning", access_log=False, log_config=None,
        )
        self._server = uvicorn.Server(config)

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._server.serve(sockets=[sock]))
            finally:
                with contextlib.suppress(Exception):
                    sock.close()

        self._thread = threading.Thread(target=run, name="ui", daemon=True)
        self._thread.start()

        # Wait for uvicorn to report that it is serving, not merely that a
        # thread exists. Ten seconds is generous; it is normally milliseconds.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if getattr(self._server, "started", False):
                break
            time.sleep(0.01)
        else:
            log.warning("the operator page did not report ready within 10 s")

        log.info("operator page on %s", self.url)
        return self.url

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._loop = None


def running_instance(host: str, port: int, span: int = 4, timeout: float = 0.5) -> str | None:
    """The URL of a parlIAmo already serving on *port* or the next few, or None.

    Identified by the shape of ``/state``, not by the port being busy: some
    other program on 8770 is not a reason to refuse, and the bind step will
    step past it as before.
    """
    import urllib.request

    for candidate in range(port, port + span):
        url = f"http://{host}:{candidate}/"
        try:
            with urllib.request.urlopen(url + "state", timeout=timeout) as response:
                body = json.loads(response.read() or b"{}")
        except Exception:
            continue
        if isinstance(body, dict) and {"source_lang", "target_lang", "muted"} <= set(body):
            return url
    return None


def _sse(message: str) -> str:
    """One server-sent event. Newlines inside data must be prefixed per line."""
    body = "\n".join(f"data: {line}" for line in message.splitlines() or [""])
    return f"{body}\n\n"


@dataclass(slots=True)
class _Noop:
    """Stand-in used when the UI is switched off, so callers need no branches."""

    state: UIState = field(default_factory=UIState)
    url: str = ""

    def publish(self, *_args: Any, **_kwargs: Any) -> None: ...
    def publish_delivery(self, *_args: Any, **_kwargs: Any) -> None: ...
    def publish_state(self, *_args: Any, **_kwargs: Any) -> None: ...
    def start(self) -> str:
        return ""

    def stop(self, timeout: float = 5.0) -> None: ...


def disabled() -> _Noop:
    """A server that does nothing, for runs without a UI."""
    return _Noop()
