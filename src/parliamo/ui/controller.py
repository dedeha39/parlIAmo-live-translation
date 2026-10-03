"""Owning the pipeline, so the application can start and stop it.

The inversion this fixes
------------------------
The first version had `live_translate.py` build the pipeline and *then* start
the page. That meant the microphone opened the moment the command ran: there
was no start button because there was nothing to press it on, and the operator
watched the system translate the room while they were still finding their
notes.

Here the application starts first and empty. The pipeline is built when someone
presses start and torn down when they press stop, which is the order a person
expects and the only order in which "start" means anything.

Building takes time - models load, the recogniser warms - so it happens on a
worker thread and the page is told what stage it has reached. A page that says
nothing for forty seconds is indistinguishable from a page that has crashed.
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


def _substituted(chosen: object, live: Any) -> bool:
    """Did the operator ask for one device and get another?

    Only an explicit choice can be overridden - ``None`` means "you decide", so
    whatever the resolver picked is not a substitution.
    """
    if chosen is None or live is None:
        return False
    if isinstance(chosen, int):
        return chosen != live.index
    text = str(chosen).split("@")[0].strip().casefold()
    return text not in live.name.casefold()


class PipelineController:
    """Builds, starts and stops the translation pipeline on demand.

    *factory* is called with no arguments and returns
    ``(translator, teardown)``. Keeping construction outside this class is
    deliberate: the CLI knows about flags and config layering, and this does
    not need to.
    """

    def __init__(
        self,
        factory: Callable[[], tuple[Any, Callable[[], None]]],
        on_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self.factory = factory
        #: Called with (status, detail) so the page can show progress.
        self.on_status = on_status
        self.translator: Any = None
        self.status = "stopped"
        self.detail = ""
        self._teardown: Callable[[], None] | None = None
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # -- reporting --------------------------------------------------------

    def _set(self, status: str, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        log.info("pipeline %s %s", status, detail)
        if self.on_status is not None:
            try:
                self.on_status(status, detail)
            except Exception:  # pragma: no cover - a display must not stop this
                log.exception("status callback raised")

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def busy(self) -> bool:
        """Starting or stopping. Pressing the button again must do nothing."""
        return self.status in {"starting", "stopping"}

    def state(self) -> dict:
        return {"status": self.status, "detail": self.detail}

    # -- control ----------------------------------------------------------

    def start(self) -> dict:
        """Build and start, on a worker thread. Returns immediately."""
        with self._lock:
            if self.running or self.busy:
                return self.state()
            self._set("starting", "loading models")
            self._worker = threading.Thread(target=self._start_blocking,
                                            name="pipeline-start", daemon=True)
            self._worker.start()
        return self.state()

    def _start_blocking(self) -> None:
        translator = teardown = None
        try:
            translator, teardown = self.factory()
            self._set("starting", "warming up")
            if self.paused:
                translator.set_paused(True)
            translator.start()
        except Exception as exc:
            log.exception("pipeline failed to start")
            # Whatever was built must be taken down again. The factory has
            # loaded ~2.5 GB of models and start() may have opened the
            # speakers and started the delivery thread before failing - on
            # this laptop, "could not open any audio output". Left alone, all
            # of it stayed resident, and the next Start loaded a second copy.
            self._dispose(translator, teardown)
            # The reason has to reach the page. An operator looking at a stuck
            # "starting" has no way to find out what went wrong.
            self._set("failed", f"{type(exc).__name__}: {exc}")
            return
        self.translator = translator
        self._teardown = teardown
        self._set("running", "listening")

    def stop(self) -> dict:
        with self._lock:
            if self.translator is None or self.busy:
                return self.state()
            self._set("stopping", "")
            translator, teardown = self.translator, self._teardown
            self.translator, self._teardown = None, None

        self._dispose(translator, teardown)
        self._set("stopped", "")
        return self.state()

    @staticmethod
    def _dispose(translator: Any, teardown: Callable[[], None] | None) -> None:
        """Stop what runs and unload what was loaded; never raise."""
        if translator is not None:
            try:
                translator.stop()
            except Exception:  # pragma: no cover
                log.exception("error stopping the pipeline")
        if teardown is not None:
            try:
                teardown()
            except Exception:  # pragma: no cover
                log.exception("error unloading models")

    # -- pass-through, so the page talks to one object --------------------

    def set_muted(self, muted: bool) -> None:
        if self.translator is not None:
            self.translator.set_muted(muted)

    #: Listening paused from the page (P). Kept across Stop and Start, so the
    #: page's banner never says paused while the microphone is listening.
    paused: bool = False

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        if self.translator is not None:
            self.translator.set_paused(self.paused)

    #: Speak sentences as they end, chosen from the page. None means the
    #: config; a live pipeline is switched immediately, a stopped one at the
    #: next start.
    stream_sentences: bool | None = None
    STREAM_MODES = ("off", "sentence", "chunk")
    stream_mode: str | None = None

    def set_streaming(self, mode) -> str:
        """Accepts a mode name, or a bool for the callers that predate chunks."""
        if isinstance(mode, bool):
            mode = "sentence" if mode else "off"
        if mode not in self.STREAM_MODES:
            raise ValueError(f"unknown streaming mode {mode!r}")
        self.stream_mode = mode
        self.stream_sentences = mode == "sentence"
        if self.translator is not None:
            self.translator.stream_mode = mode
            self.translator.stream_sentences = mode == "sentence"
            # A mode change discards what the old policy was tracking, or the
            # next commit skips sentences the new policy never spoke.
            self.translator.reset_streaming()
        return mode

    def streaming(self) -> str:
        if self.translator is not None:
            return getattr(self.translator, "stream_mode", "off") or (
                "sentence" if self.translator.stream_sentences else "off")
        return self.stream_mode or "off"

    def flush(self) -> bool:
        """Drop the audio already in the speakers, and release the gate with it.

        Distinct from mute, which stops the *next* sentence. This one stops the
        one currently being said - the case where the system is confidently
        speaking something wrong and every second of it costs the room's
        attention. Returns whether there was a pipeline to flush.
        """
        playback = getattr(self.translator, "playback", None) if self.translator else None
        if playback is None:
            return False
        playback.flush()
        return True

    def set_reference_voice(self, path: str | None, consent: str = "") -> None:
        """Change the voice, running or not.

        When nothing is running the choice is remembered and applied at the
        next start - otherwise picking a voice before pressing start would
        silently do nothing, which is worse than refusing.
        """
        if self.translator is not None:
            self.translator.set_reference_voice(path, consent)
            return
        if path is not None:
            from ..paths import resolve
            from ..tts.base import VoiceProfile

            VoiceProfile(name="pending", reference_path=str(resolve(path)),
                         consent=consent).validate()
        self.pending_voice = (path, consent)

    #: Chosen before the pipeline was started, applied when it is.
    pending_voice: tuple[str | None, str] | None = None

    #: Devices chosen from the page. None means the system default, which the
    #: resolver then upgrades to the fastest host API offering the same
    #: hardware - on Windows the default is usually an MME clone carrying
    #: ~90 ms against 2 ms for the same microphone on WASAPI.
    input_device: object = None
    output_device: object = None

    #: Languages chosen from the page, applied at the next start. None means
    #: the config. Not changed under a running pipeline: the recogniser, the
    #: translator and the voice are all built for a pair, and a swap mid-talk
    #: would be a rebuild with a different name.
    source_lang: str | None = None
    target_lang: str | None = None

    #: Models chosen from the page, applied at the next start. None means the
    #: config. The defaults were chosen by measurement and the alternatives
    #: are offered with their measured cost beside them; the choice is the
    #: presenter's, and it is made with the numbers in view.
    asr_model: str | None = None
    mt_model: str | None = None

    #: The commit wait, chosen from the page, applied at the next start. The
    #: one term in the whole budget that is a setting rather than compute;
    #: measured in ADR 0009, offered with those numbers beside it.
    min_silence_ms: int | None = None

    #: OmniVoice's speaking rate chosen on the page; 0 means "copy the
    #: reference". Applied to the running pipeline at once, and at each start.
    voice_speed: float | None = None

    def devices(self) -> dict:
        """What is selected, and what is actually open.

        The two differ between choosing a device and pressing start, and an
        operator who cannot see the difference will assume a click took effect
        when it has not.
        """
        live_in = live_out = None
        device_in = device_out = None
        if self.translator is not None:
            capture = getattr(self.translator.transcriber, "_capture", None)
            if capture is not None:
                with_info = getattr(capture, "_info", None)
                if with_info is not None:
                    device_in = with_info.device
                    live_in = device_in.label
            playback = self.translator.playback
            if playback is not None and getattr(playback, "_info", None) is not None:
                device_out = playback._info.device
                live_out = device_out.label
        return {
            "selected": {"input": self.input_device, "output": self.output_device},
            "live": {"input": live_in, "output": live_out},
            # A device that will not open is replaced by one that will. That is
            # the right behaviour on stage and the wrong thing to do quietly:
            # an operator who cannot see the substitution believes the room is
            # hearing the lavalier when it is hearing the laptop.
            "substituted": {
                "input": _substituted(self.input_device, device_in),
                "output": _substituted(self.output_device, device_out),
            },
            "applies": "now" if not self.running else "at the next start",
        }
