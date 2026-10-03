"""The subtitle and operator page.

Two properties carry it, and neither is about the display:

* **It cannot stall the pipeline.** The UI is a passive observer running on
  another thread. A browser tab that stops reading, or a page nobody opened,
  must not be able to hold up delivery - on stage the sentence matters and the
  screen does not.
* **Mute means now.** The panic control has to stop the sentence *already
  playing*, not merely the next one, because the one you want to stop is the
  one coming out of the speakers.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from parliamo.ui.server import QUEUE_DEPTH, SubtitleServer, UIState, _sse, disabled


def _delivery(index: int = 1, partial: bool = False, translated: str = "Ciao."):
    from parliamo.pipeline.translator import DeliveryEvent

    return DeliveryEvent(
        index=index, source_text="Merhaba.", translated_text=translated,
        source_lang="tr", target_lang="it", recognition_lag_s=0.8,
        translation_s=0.2, synthesis_s=0.1, conversion_s=0.5, audio_s=1.0,
        spoken=not partial, partial=partial,
    )


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


def test_state_serialises() -> None:
    state = UIState(source_lang="tr", target_lang="it")
    payload = state.as_dict()
    assert payload["source_lang"] == "tr"
    assert payload["muted"] is False
    json.dumps(payload)  # must survive the wire


def test_committed_and_partial_counts_are_separate() -> None:
    server = SubtitleServer()
    server.publish_delivery(_delivery(partial=True))
    server.publish_delivery(_delivery(partial=True))
    server.publish_delivery(_delivery(partial=False))
    assert server.state.partials == 2
    assert server.state.delivered == 1


def test_only_a_committed_sentence_updates_the_lag() -> None:
    """A partial has not waited for silence, so its lag is not comparable."""
    server = SubtitleServer()
    server.publish_delivery(_delivery(partial=False))
    committed_lag = server.state.last_lag_s
    server.publish_delivery(_delivery(partial=True))
    assert server.state.last_lag_s == committed_lag


# ---------------------------------------------------------------------------
# it must never stall the pipeline
# ---------------------------------------------------------------------------


def test_publishing_with_no_server_running_is_a_noop() -> None:
    """The common case at startup, and after stop(). Must not raise."""
    server = SubtitleServer()
    server.publish_delivery(_delivery())
    server.publish_state()
    server.publish("anything", {"a": 1})


def test_publishing_is_fast_enough_to_sit_in_the_delivery_thread() -> None:
    server = SubtitleServer()
    t0 = time.perf_counter()
    for i in range(500):
        server.publish_delivery(_delivery(index=i))
    per_call_us = (time.perf_counter() - t0) / 500 * 1e6
    assert per_call_us < 200, f"{per_call_us:.0f} us per publish"


def test_a_full_subscriber_queue_drops_rather_than_blocks() -> None:
    """A browser that stopped reading must lose subtitles, not stop the show."""
    import asyncio

    server = SubtitleServer()
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_DEPTH)
    for _ in range(QUEUE_DEPTH):
        queue.put_nowait("x")

    server._offer(queue, "one more")   # would block if it awaited
    assert server.dropped == 1
    assert queue.qsize() == QUEUE_DEPTH


def test_disabled_server_absorbs_every_call() -> None:
    """So callers need no `if ui is not None` around each publish."""
    noop = disabled()
    assert noop.start() == ""
    noop.publish_delivery(_delivery())
    noop.publish_state()
    noop.publish("kind", {})
    noop.stop()


# ---------------------------------------------------------------------------
# the wire format
# ---------------------------------------------------------------------------


def test_sse_frame_shape() -> None:
    assert _sse('{"a":1}') == 'data: {"a":1}\n\n'


def test_multiline_payload_is_prefixed_per_line() -> None:
    """A raw newline inside data would end the event early and lose the rest."""
    frame = _sse("one\ntwo")
    assert frame == "data: one\ndata: two\n\n"


def test_turkish_and_italian_survive_the_json() -> None:
    event = _delivery(translated="perché è così")
    event.source_text = "ışık İstanbul"
    payload = json.dumps({"kind": "delivery", "event": event.as_dict()}, ensure_ascii=False)
    assert "perché è così" in payload
    assert "ışık İstanbul" in payload


# ---------------------------------------------------------------------------
# the mute control, end to end over HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    assert fastapi
    calls: list[bool] = []
    server = SubtitleServer(on_mute=calls.append)
    with TestClient(server.build_app()) as c:
        yield c, server, calls


def test_page_is_served_and_self_contained(client) -> None:
    """No external asset may be fetched: the venue has no internet, and the
    talk's whole claim is that this runs with the network unplugged."""
    c, _, _ = client
    page = c.get("/").text
    assert "<title>parlIAmo" in page
    for forbidden in ("http://", "https://", "//cdn", "fonts.googleapis"):
        assert forbidden not in page, f"page fetches something external: {forbidden}"


def test_mute_reaches_the_pipeline(client) -> None:
    c, server, calls = client
    response = c.post("/mute", json={"muted": True})
    assert response.json()["muted"] is True
    assert calls == [True], "the mute button did not reach the translator"
    assert server.state.muted is True


def test_unmute_reaches_the_pipeline(client) -> None:
    c, _, calls = client
    c.post("/mute", json={"muted": True})
    c.post("/mute", json={"muted": False})
    assert calls == [True, False]


def test_mute_with_no_body_toggles(client) -> None:
    c, server, _ = client
    c.post("/mute")
    assert server.state.muted is True
    c.post("/mute")
    assert server.state.muted is False


def test_a_raising_mute_handler_does_not_break_the_control(client) -> None:
    """The operator pressing the panic button must always get an answer."""
    _, server, _ = client
    from fastapi.testclient import TestClient

    def boom(_muted: bool) -> None:
        raise RuntimeError("pipeline exploded")

    server.on_mute = boom
    with TestClient(server.build_app()) as c:
        assert c.post("/mute", json={"muted": True}).status_code == 200


def test_state_endpoint(client) -> None:
    c, _, _ = client
    assert c.get("/state").json()["source_lang"] == "tr"


# ---------------------------------------------------------------------------
# the translator side of mute
# ---------------------------------------------------------------------------


class _Playback:
    def __init__(self) -> None:
        self.submitted: list[int] = []
        self.flushes = 0

    def start(self):  # pragma: no cover - not exercised here
        return None

    def stop(self) -> None: ...
    def wait(self, timeout=None) -> bool:
        return True

    def submit(self, audio, sample_rate=None) -> int:
        self.submitted.append(int(np.asarray(audio).size))
        return int(np.asarray(audio).size)

    def flush(self) -> None:
        self.flushes += 1


class _FakeMT:
    source_lang = "tr"
    loaded = True

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def unload(self) -> None: ...

    def translate(self, text, source_lang=None, target_lang=None):
        from parliamo.mt.base import Translation

        return Translation(text=f"IT({text})", source=text, source_lang="tr",
                           target_lang="it", backend="fake")


class _FakeTTS:
    sample_rate = 24000
    loaded = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def unload(self) -> None: ...

    def speak(self, text, language=None, voice=None):
        from parliamo.tts.base import Speech

        self.calls.append(text)
        return Speech(audio=np.zeros(2400, dtype=np.float32), sample_rate=self.sample_rate,
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


def _muteable_translator():
    from parliamo.pipeline.transcriber import LiveTranscriber
    from parliamo.pipeline.translator import LiveTranslator

    playback = _Playback()
    translator = LiveTranslator(
        LiveTranscriber(_Backend()), _FakeMT(), _FakeTTS(),
        playback=playback, speak=True,
    )
    return translator, playback


def _committed_event():
    from parliamo.audio.vad import SpeechSegment
    from parliamo.pipeline.transcriber import TranscriptEvent

    return TranscriptEvent(
        index=1, text="merhaba",
        segment=SpeechSegment(audio=np.zeros(16000, dtype=np.float32), start_s=0, end_s=1),
        transcript=None, commit_wait_s=0.5, queue_wait_s=0.0, asr_s=0.1,
    )


def test_mute_flushes_what_is_already_playing() -> None:
    """The whole point of a panic control."""
    translator, playback = _muteable_translator()
    translator.set_muted(True)
    assert playback.flushes == 1


def test_muted_delivery_reaches_no_speaker() -> None:
    translator, playback = _muteable_translator()
    translator.set_muted(True)
    delivery = translator.deliver(_committed_event())
    assert delivery is not None
    assert delivery.spoken is False
    assert playback.submitted == []


def test_muted_delivery_still_produces_the_subtitle() -> None:
    """Silencing the audio must not silence the screen."""
    translator, _ = _muteable_translator()
    translator.set_muted(True)
    delivery = translator.deliver(_committed_event())
    assert delivery is not None
    assert delivery.translated_text


def test_muted_delivery_skips_synthesis() -> None:
    """It would only be thrown away, and the GPU is shared."""
    translator, _ = _muteable_translator()
    translator.set_muted(True)
    translator.deliver(_committed_event())
    assert translator.synthesiser.calls == []


def test_unmuting_restores_audio() -> None:
    translator, playback = _muteable_translator()
    translator.set_muted(True)
    translator.deliver(_committed_event())
    translator.set_muted(False)
    delivery = translator.deliver(_committed_event())
    assert delivery is not None
    assert delivery.spoken is True
    assert playback.submitted


def test_mute_is_safe_without_playback() -> None:
    """--no-speak runs have no playback object to flush."""
    from parliamo.pipeline.transcriber import LiveTranscriber
    from parliamo.pipeline.translator import LiveTranslator

    translator = LiveTranslator(
        LiveTranscriber(_Backend()), _FakeMT(), _FakeTTS(), playback=None, speak=False
    )
    translator.set_muted(True)  # must not raise


def test_mute_is_thread_safe_enough_to_press_twice() -> None:
    """The operator will press it twice. It is a panic button."""
    translator, playback = _muteable_translator()
    threads = [
        threading.Thread(target=translator.set_muted, args=(i % 2 == 0,))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert isinstance(translator.muted, bool)


# ---------------------------------------------------------------------------
# the operator application
# ---------------------------------------------------------------------------


def test_operator_app_and_audience_screen_are_different_pages(client) -> None:
    """One is read at laptop distance by a person who is talking; the other
    across a room by two hundred who are not. Same data, different problem."""
    c, _, _ = client
    app = c.get("/").text
    screen = c.get("/screen").text
    assert app != screen

    # Distinct titles, because both windows are open at once on the night and
    # the browser tab is the only thing telling them apart.
    assert "<title>parlIAmo — operator</title>" in app
    assert "<title>parlIAmo</title>" in screen

    assert 'id="preflight"' in app, "the operator app has no pre-flight list"
    assert 'id="preflight"' not in screen, "the room must not be shown the checklist"
    assert 'id="stack"' in screen, "the audience screen is not the subtitle page"
    # The room's page is dark and the operator's is not. Projecting a white
    # page into a dark hall is the difference between reading it and not.
    assert "#0b0d10" in screen and "#0b0d10" not in app


def test_both_pages_fetch_nothing_external(client) -> None:
    c, _, _ = client
    for route in ("/", "/screen"):
        page = c.get(route).text
        for forbidden in ("http://", "https://", "//cdn", "fonts.googleapis"):
            assert forbidden not in page, f"{route} fetches {forbidden}"


def test_preflight_reports_the_gate_warning(client) -> None:
    """The one item that can ruin the demonstration outright."""
    c, server, _ = client
    server.state.gate_warning = "output latency is 0"
    body = c.get("/preflight").json()
    assert body["level"] == "warn"
    titles = [row["title"] for row in body["checks"]]
    assert any("latency" in t.lower() for t in titles)


def test_preflight_is_clean_when_everything_is_set(client) -> None:
    c, server, _ = client
    server.state.gate_warning = ""
    server.state.speaking = True
    server.state.voice = "generic"
    body = c.get("/preflight").json()
    assert all(row["level"] != "blocked" for row in body["checks"])


def test_preflight_flags_a_loaded_cloned_voice(client) -> None:
    """Someone has to remember to delete it, so the list remembers instead."""
    c, server, _ = client
    server.state.voice = "maria"
    body = c.get("/preflight").json()
    assert any("maria" in row["title"] for row in body["checks"])


def test_preflight_never_blocks_on_a_warning(client) -> None:
    """Whether a warning is acceptable is the presenter's judgement, not ours."""
    c, server, _ = client
    server.state.gate_warning = "output latency is 0"
    server.state.voice = "maria"
    assert c.get("/preflight").json()["level"] in {"warn", "blocked"}


def test_devices_endpoint_answers_even_without_a_sound_card(client) -> None:
    c, _, _ = client
    body = c.get("/devices").json()
    assert "input" in body and "output" in body


def test_listing_voices_separates_volunteers(tmp_path, monkeypatch, client) -> None:
    c, server, _ = client
    root = tmp_path / "data" / "voices"
    (root / "volunteers" / "maria").mkdir(parents=True)
    (root / "presenter.wav").write_bytes(b"\x00" * 32)
    (root / "volunteers" / "maria" / "reference.wav").write_bytes(b"\x00" * 32)
    (root / "volunteers" / "maria" / "consent.json").write_text(
        '{"consent": "ref 007"}', encoding="utf-8")

    import parliamo.paths as paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    voices = server.list_voices()
    names = {v["name"]: v for v in voices}
    assert names["presenter"]["volunteer"] is False
    assert names["maria"]["volunteer"] is True
    assert names["maria"]["consent"] == "ref 007"


def test_forgetting_volunteers_also_unloads_the_voice(tmp_path, monkeypatch) -> None:
    """A voice whose recording no longer exists must not stay loaded."""
    from parliamo.ui.server import SubtitleServer

    root = tmp_path / "data" / "voices" / "volunteers" / "maria"
    root.mkdir(parents=True)
    (root / "reference.wav").write_bytes(b"\x00" * 128)

    cleared: list = []
    server = SubtitleServer(on_voice=lambda p, c: cleared.append(p))
    server.state.voice = "maria"

    import parliamo.paths as paths

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    result = server.forget_volunteers()

    assert result["removed"] == ["maria"]
    assert not root.exists()
    assert cleared == [None], "the deleted voice was left loaded"
    assert server.state.voice == "generic"


def test_forgetting_nothing_is_not_an_error(tmp_path, monkeypatch) -> None:
    import parliamo.paths as paths
    from parliamo.ui.server import SubtitleServer

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    assert SubtitleServer().forget_volunteers()["removed"] == []


def test_a_late_browser_is_shown_what_it_missed() -> None:
    """Reloading mid-talk must not show a blank screen.

    On stage a blank subtitle area looks exactly like the system having died,
    and the operator has no way to tell the difference in the moment.
    """
    server = SubtitleServer()
    for i in range(3):
        server.publish_delivery(_delivery(index=i, translated=f"frase {i}"))
    assert len(server._history) == 3
    assert [e["translation"] for e in server._history] == ["frase 0", "frase 1", "frase 2"]


def test_only_committed_sentences_are_replayed() -> None:
    """A partial was already superseded by the time anyone reconnects."""
    server = SubtitleServer()
    server.publish_delivery(_delivery(partial=True, translated="guess"))
    server.publish_delivery(_delivery(partial=False, translated="final"))
    assert [e["translation"] for e in server._history] == ["final"]


def test_the_replay_is_bounded() -> None:
    from parliamo.ui.server import HISTORY

    server = SubtitleServer()
    for i in range(HISTORY * 5):
        server.publish_delivery(_delivery(index=i))
    assert len(server._history) == HISTORY


# ---------------------------------------------------------------------------
# the pages' own JavaScript
# ---------------------------------------------------------------------------


def _script_of(name: str) -> str:
    from parliamo.paths import REPO_ROOT

    html = (REPO_ROOT / "src" / "parliamo" / "ui" / name).read_text(encoding="utf-8")
    return html.split("<script>", 1)[1].split("</script>", 1)[0]


@pytest.mark.parametrize("page", ["app.html", "page.html"])
def test_page_javascript_parses(page: str) -> None:
    """A syntax error takes out the whole script, so every control dies at once.

    That happened: an edit landed inside a function body and the page loaded
    looking correct with nothing working - no start button, no mute, no
    subtitles - and the only clue was one line in the browser console. Nothing
    in the test suite could see it, because the tests checked the transport and
    the endpoints, never the page's own code.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; cannot parse-check the page")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.js"
        path.write_text('"use strict";\n' + _script_of(page), encoding="utf-8")
        result = subprocess.run([node, "--check", str(path)], capture_output=True,
                                text=True, timeout=30)
    assert result.returncode == 0, f"{page} has a JavaScript syntax error:\n{result.stderr}"


@pytest.mark.parametrize("page", ["app.html", "page.html"])
def test_every_element_the_script_reaches_for_exists(page: str) -> None:
    """`getElementById` on a missing id returns null, and the next line throws.

    A renamed or removed element is invisible until the control is pressed on
    stage.
    """
    import re

    from parliamo.paths import REPO_ROOT

    html = (REPO_ROOT / "src" / "parliamo" / "ui" / page).read_text(encoding="utf-8")
    body = html.split("</script>", 1)[0]
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', body))
    wanted |= set(re.findall(r'\$\("([^"]+)"\)', body))
    present = set(re.findall(r'id="([^"]+)"', html))
    missing = wanted - present
    assert not missing, f"{page} reaches for ids that do not exist: {sorted(missing)}"


# ---------------------------------------------------------------------------
# binding, and the URL that is printed
# ---------------------------------------------------------------------------


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_start_reports_the_port_it_actually_holds() -> None:
    """The regression this guards.

    ``start()`` used to signal ready as soon as the event loop existed - before
    the bind was even attempted - so an occupied port printed a working-looking
    URL with uvicorn's bind failure logged underneath it::

        parlIAmo is at http://127.0.0.1:8770/
        ERROR: [Errno 10048] error while attempting to bind on address ...

    An address that is printed has to be an address that is held.
    """
    import urllib.request

    port = _free_port()
    first = SubtitleServer(port=port)
    second = SubtitleServer(port=port)
    try:
        url_a = first.start()
        url_b = second.start()

        assert url_a != url_b, "two servers must not claim the same URL"
        assert second.port == port + 1, "the second should step forward one port"

        for url in (url_a, url_b):
            with urllib.request.urlopen(url + "state", timeout=5) as response:
                assert response.status == 200, f"{url} was printed but does not serve"
    finally:
        first.stop()
        second.stop()


def test_bind_raises_when_the_whole_range_is_taken() -> None:
    """Silently serving nothing is worse than refusing to start."""
    import socket

    base = _free_port()
    held = []
    try:
        for offset in range(3):
            sock = socket.socket()
            try:
                sock.bind(("127.0.0.1", base + offset))
            except OSError:  # pragma: no cover - a neighbour took it first
                sock.close()
                pytest.skip("could not reserve a contiguous port range")
            sock.listen(1)
            held.append(sock)

        with pytest.raises(OSError, match="no free port"):
            SubtitleServer(port=base).bind(attempts=3)
    finally:
        for sock in held:
            sock.close()


def test_bind_does_not_set_reuseaddr() -> None:
    """SO_REUSEADDR on Windows lets two processes listen on one port.

    That is how two voice-conversion servers once ran at once, holding 5.8 GB
    and making every conversion 5-10x slower, with the symptom looking exactly
    like a model regression.
    """
    import socket

    server = SubtitleServer(port=_free_port())
    sock = server.bind()
    try:
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 0
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# a substituted device must be visible
# ---------------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name
        self.hostapi_name = "Windows WASAPI"

    @property
    def label(self) -> str:
        return f"{self.name} [{self.hostapi_name}]"


def _controller_with(selected_out, live_out):
    from parliamo.ui.controller import PipelineController

    controller = PipelineController(factory=lambda: (None, None))
    controller.output_device = selected_out

    class _Playback:
        _info = type("I", (), {"device": live_out})()

    class _Translator:
        transcriber = type("T", (), {"_capture": None})()
        playback = _Playback()

    controller.translator = _Translator()
    return controller


def test_a_device_that_was_replaced_is_reported_as_substituted() -> None:
    """The operator has to see it, or they trust the wrong speaker.

    A device that will not open is replaced by one that will - the right
    behaviour thirty seconds before a talk, and the wrong thing to do quietly.
    """
    controller = _controller_with(19, _FakeDevice(16, "Speakers (Realtek(R) Audio)"))
    report = controller.devices()
    assert report["substituted"]["output"] is True
    assert report["live"]["output"] == "Speakers (Realtek(R) Audio) [Windows WASAPI]"


def test_getting_the_device_you_asked_for_is_not_a_substitution() -> None:
    controller = _controller_with(16, _FakeDevice(16, "Speakers (Realtek(R) Audio)"))
    assert controller.devices()["substituted"]["output"] is False


def test_no_choice_means_no_substitution() -> None:
    """`None` means "you decide", so the resolver's pick is not an override."""
    controller = _controller_with(None, _FakeDevice(16, "Speakers (Realtek(R) Audio)"))
    assert controller.devices()["substituted"]["output"] is False


def test_a_name_choice_is_matched_by_name() -> None:
    controller = _controller_with("Realtek", _FakeDevice(16, "Speakers (Realtek(R) Audio)"))
    assert controller.devices()["substituted"]["output"] is False

    moved = _controller_with("Realtek", _FakeDevice(14, "Headphones (AirPods Pro)"))
    assert moved.devices()["substituted"]["output"] is True


def test_get_devices_reports_what_is_live(client) -> None:
    """GET used to return only the lists, so the page could not show the truth.

    `selected` and `live` were reachable only from POST, which meant the
    substitution was invisible unless the operator happened to change a device.
    """
    http, _server, _calls = client
    response = http.get("/devices")
    assert response.status_code == 200
    body = response.json()
    assert "input" in body and "output" in body
    if body.get("live") is not None:
        assert "selected" in body and "substituted" in body


# ---------------------------------------------------------------------------
# what the redesigned operator window needs from the server
# ---------------------------------------------------------------------------


def test_metrics_is_empty_and_honest_when_nothing_runs(client) -> None:
    """Not running is a state, not an error. The page has to render it."""
    http, _server, _calls = client
    body = http.get("/metrics").json()
    assert body["running"] is False
    assert body["gate"] is None
    assert body["lag"] is None


def test_metrics_reports_the_gate_and_the_lag_shape(client) -> None:
    """The two things an operator watches that are not the subtitles.

    `speech_during_mute_blocks` is the one worth surfacing: it counts real
    speech thrown away because the gate was shut, which is the cost of the
    protection rather than the protection itself.
    """
    http, server, _calls = client

    # The real gate, not a stand-in. A bare object with attributes agreed with
    # a bug - `remaining_ms` is a method, and the first version rounded the
    # method itself - and the route returned 500 on every poll on stage
    # hardware while every test stayed green.
    from parliamo.audio.gate import HalfDuplexGate

    gate = HalfDuplexGate(enabled=True, tail_ms=250)
    gate.close()
    gate.note_muted_block(512, 16000, had_speech=True)
    gate.note_muted_block(512, 16000, had_speech=False)

    class _Translator:
        playback = type("P", (), {"gate": gate})()
        stats = type("T", (), {"summary": staticmethod(lambda: {
            "delivered": 11, "partials_delivered": 38,
            "lag_s": {"mean": 2.78, "p50": 2.56, "p95": 3.66, "max": 3.66}})})()

    server.controller = type("C", (), {"translator": _Translator()})()
    response = http.get("/metrics")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["running"] is True
    assert body["gate"]["closes"] == 1
    assert body["gate"]["open"] is False
    assert body["gate"]["tail_ms"] == 250
    assert body["gate"]["speech_during_mute_blocks"] == 1
    assert body["gate"]["muted_frames"] == 1024
    # Held by playback: infinite remaining time, which JSON cannot carry.
    assert body["gate"]["held"] is True
    assert body["gate"]["remaining_ms"] is None

    gate.release()  # the tail starts; now there is a finite number
    body = http.get("/metrics").json()
    assert body["gate"]["held"] is False
    assert isinstance(body["gate"]["remaining_ms"], float)
    assert 0.0 <= body["gate"]["remaining_ms"] <= 250.0
    assert body["lag"]["p95"] == 3.66
    assert body["counts"]["delivered"] == 11


def test_flush_without_a_pipeline_says_so_rather_than_crashing(client) -> None:
    http, _server, _calls = client
    assert http.post("/flush").status_code == 503


def test_flush_reaches_the_playback(client) -> None:
    """Mute stops the next sentence; this stops the one being said now."""
    http, server, _calls = client
    flushed: list[bool] = []
    server.controller = type("C", (), {"flush": lambda self: flushed.append(True) or True})()
    assert http.post("/flush").status_code == 200
    assert flushed == [True]


def test_friulian_rows_come_from_the_review_sheet(tmp_path, monkeypatch) -> None:
    """One list of sentences, and the reviewer's own file is the one that is true."""
    from parliamo.ui import server as mod

    sheet = tmp_path / "data" / "friulian"
    sheet.mkdir(parents=True)
    (sheet / "review-sheet.md").write_text(
        "# Friulian demonstration sentences\n\n"
        "### 1.\n\n"
        "- **Italian (meaning):** Il friulano e una lingua parlata.\n"
        "- **Friulian (to check):** **Il furlan al e une lenghe.**\n"
        "- **Correction:**\n\n"
        "### 2.\n\n"
        "- **Italian (meaning):** Circa seicentomila persone.\n"
        "- **Friulian (to check):** **Cirche siscent mil personis.**\n"
        "- **Correction:** already corrected\n",
        encoding="utf-8")
    monkeypatch.setattr(mod, "resolve", lambda p: tmp_path / p, raising=False)
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)

    rows = SubtitleServer().friulian_rows()
    assert [r["n"] for r in rows] == [1, 2]
    assert rows[0]["friulian"] == "Il furlan al e une lenghe.", "bold markers must be stripped"
    assert rows[0]["italian"] == "Il friulano e una lingua parlata."
    assert rows[0]["correction"] == ""
    assert rows[1]["correction"] == "already corrected"


def test_friulian_is_empty_rather_than_broken_without_the_sheet(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)
    assert SubtitleServer().friulian_rows() == []


def test_measurements_survive_a_machine_that_measured_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)
    body = SubtitleServer().measurements()
    assert body["vram"] is None
    assert body["latency_runs"] == []


def test_measurements_read_back_the_latency_runs(tmp_path, monkeypatch) -> None:
    """Setup shows the numbers from this machine, not a figure from a document."""
    import json as _json

    audio = tmp_path / "runs" / "audio"
    audio.mkdir(parents=True)
    for i, median in enumerate((432.4, 366.4)):
        (audio / f"device-run{i}.json").write_text(_json.dumps({
            "label": f"run{i}",
            "roundtrip": {"median_ms": median, "min_ms": 229.0, "max_ms": 597.0,
                          "spread_ms": 368.0, "samples": 5},
        }), encoding="utf-8")
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)

    runs = SubtitleServer().measurements()["latency_runs"]
    assert [r["median_ms"] for r in runs] == [432.4, 366.4]
    assert runs[0]["spread_ms"] == 368.0


def test_a_failed_probe_is_left_out_of_the_latency_table(tmp_path, monkeypatch) -> None:
    """A report with no measurement must not become a row with no numbers."""
    import json as _json

    audio = tmp_path / "runs" / "audio"
    audio.mkdir(parents=True)
    (audio / "device-bad.json").write_text(
        _json.dumps({"label": "bad", "roundtrip": {"error": "no confident measurement"}}),
        encoding="utf-8")
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)

    assert SubtitleServer().measurements()["latency_runs"] == []


# ---------------------------------------------------------------------------
# the design, as a contract
# ---------------------------------------------------------------------------


def _page(name: str) -> str:
    from parliamo.paths import REPO_ROOT

    return (REPO_ROOT / "src" / "parliamo" / "ui" / name).read_text(encoding="utf-8")


def test_the_operator_window_has_all_five_tabs() -> None:
    """Live, Voices, Friulian, Ethics, Setup.

    Each one is a section of the talk. Losing one to a refactor loses that part
    of the demonstration, silently, until someone clicks for it on stage.
    """
    html = _page("app.html")
    for tab in ("live", "voices", "friulian", "ethics", "setup"):
        assert f'data-t="{tab}"' in html, f"the {tab} tab is gone"
        assert f'id="{tab}"' in html, f"the {tab} section is gone"


def test_the_five_stage_breakdown_covers_the_whole_journey() -> None:
    """Commit wait, recognition, translation, synthesis, conversion.

    These five sum to what the audience waited. Drop one and the bar still
    renders, still looks plausible, and no longer adds up.
    """
    html = _page("app.html")
    for key in ("commit_wait_s", "asr_s", "translation_s", "synthesis_s", "conversion_s"):
        assert f'"{key}"' in html, f"the breakdown lost {key}"


def test_the_operator_keeps_every_stage_control() -> None:
    html = _page("app.html")
    for control in ("/mute", "/flush", "/voice", "/pipeline/", "/voices/record", "/devices"):
        assert control in html, f"the page no longer calls {control}"


def test_the_three_shortcuts_are_bound() -> None:
    """M, G and S. Documented in the sidebar, so they have to work."""
    html = _page("app.html")
    body = html.split("<script>", 1)[1]
    for key in ('"m"', '"g"', '"s"'):
        assert key in body, f"shortcut {key} is not handled"
    assert "<kbd>M</kbd>" in html and "<kbd>G</kbd>" in html and "<kbd>S</kbd>" in html


def test_a_provisional_line_is_marked_as_provisional_on_both_pages() -> None:
    """It is a guess, still being revised, and it is never spoken.

    Turkish puts the verb and its negation last, so the meaning can reverse on
    the final word. Anything that reads as a statement before it is committed
    will be read as one by the room.
    """
    app, screen = _page("app.html"), _page("page.html")
    assert "provisional · revised in place · never spoken" in app
    assert ".said.prov" in app and ".said.prov" in screen
    assert "italic" in app and "italic" in screen


def test_the_audience_screen_carries_no_operator_controls() -> None:
    """The room must not be shown the checklist, the devices or the volunteers."""
    screen = _page("page.html")
    for operator_only in ("/preflight", "/devices", "/voices", "/metrics", "/friulian"):
        assert operator_only not in screen, f"the projected page reaches for {operator_only}"


def test_neither_page_fetches_a_font_or_a_script_from_anywhere() -> None:
    """The venue has no internet, and the talk's claim is that it needs none."""
    for name in ("app.html", "page.html"):
        html = _page(name)
        for forbidden in ("http://", "https://", "//cdn", "@import", "fonts.g"):
            assert forbidden not in html, f"{name} fetches {forbidden}"


def test_hidden_actually_hides() -> None:
    """The bug this guards, which shipped for one afternoon.

    Every rule that sets `display` on a banner outranks the browser's default
    `[hidden]{display:none}`, so the red AUDIO MUTED banner sat on screen with
    nothing muted. The override has to exist and has to be marked important.
    """
    html = _page("app.html")
    assert "[hidden]{display:none!important}" in html.replace(" ", "")


def test_starting_the_server_does_not_close_the_run_log(tmp_path) -> None:
    """The regression this guards.

    uvicorn.Config runs logging.config.dictConfig by default, and dictConfig
    starts by calling logging.shutdown() on every handler in the process. Our
    JSONL handler stayed attached to the root logger with its file closed, and
    the first warning from the model-loading thread produced
    "I/O operation on closed file" - and a run with no events.jsonl.
    """
    import logging

    from parliamo.logging_setup import JsonlHandler, setup_logging

    pytest.importorskip("uvicorn")
    setup_logging(level="WARNING", jsonl=True, runs_dir=tmp_path)
    root = logging.getLogger()
    jsonl = next(h for h in root.handlers if isinstance(h, JsonlHandler))
    assert not jsonl._fh.closed

    server = SubtitleServer(port=_free_port())
    try:
        server.start()
        assert jsonl in root.handlers, "the run log was detached"
        assert not jsonl._fh.closed, "the run log was closed by starting the page"
        # And it still writes.
        logging.getLogger("test").warning("still here")
        jsonl.flush()
        assert "still here" in jsonl.path.read_text(encoding="utf-8")
    finally:
        server.stop()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()


# ---------------------------------------------------------------------------
# the presenter's own voice, and the consent that sits beside it
# ---------------------------------------------------------------------------


def test_the_configured_voice_carries_its_consent_in_the_list(tmp_path, monkeypatch) -> None:
    """The bug this guards, seen on stage.

        voice change refused: voice profile 'pending' has no consent record.

    The presenter pressed *use* on their own recording. Its consent sits in
    config/local.yaml beside the path, and the list never attached it, so the
    page sent an empty string and the one voice the setup was built around was
    refused.
    """
    voices = tmp_path / "data" / "voices"
    voices.mkdir(parents=True)
    import numpy as np
    import soundfile as sf

    sf.write(str(voices / "mine.wav"), np.zeros(16000, dtype=np.float32), 16000)
    sf.write(str(voices / "other.wav"), np.zeros(16000, dtype=np.float32), 16000)
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)

    server = SubtitleServer()
    server.configured_voice = (str(voices / "mine.wav"), "presenter's own voice, 2026-09-01")
    rows = {v["name"]: v for v in server.list_voices()}

    assert rows["mine"]["consent"] == "presenter's own voice, 2026-09-01"
    assert rows["mine"]["configured"] is True
    assert rows["other"]["consent"] == ""
    assert rows["other"]["configured"] is False


def test_using_the_configured_voice_without_consent_falls_back_to_the_recorded_one(
    client, tmp_path
) -> None:
    http, server, _calls = client
    seen: list[tuple] = []
    server.on_voice = lambda path, consent: seen.append((path, consent))
    ref = tmp_path / "mine.wav"
    ref.write_bytes(b"")
    server.configured_voice = (str(ref), "signed 2026-09-01")

    r = http.post("/voice", json={"path": str(ref), "consent": ""})
    assert r.status_code == 200, r.text
    assert seen == [(str(ref), "signed 2026-09-01")]


def test_an_unconfigured_voice_without_consent_is_still_refused(client, tmp_path) -> None:
    """The fallback is for the one recorded voice, not a way around the check."""
    http, server, _calls = client

    def refuse(path, consent):
        if not consent:
            raise ValueError("no consent record")

    server.on_voice = refuse
    server.configured_voice = (str(tmp_path / "mine.wav"), "signed")
    r = http.post("/voice", json={"path": str(tmp_path / "someone-else.wav"), "consent": ""})
    assert r.status_code == 400


def test_preflight_says_when_the_conversion_service_is_down(client, monkeypatch) -> None:
    """"My voice is configured, so why is it not being used?" - this row."""
    import parliamo.tts.conversion as conversion

    # Discovery also tries the known ports; a real service running on this
    # machine (as during a rehearsal) must not answer for the test.
    monkeypatch.setattr(conversion, "KNOWN_PORTS", ())
    http, server, _calls = client
    server.conversion_address = ("127.0.0.1", _free_port())  # nothing listens there
    server.conversion_enabled = True
    titles = [c["title"] for c in http.get("/preflight").json()["checks"]]
    assert any("Voice service not running" in t and "OmniVoice 8767" in t for t in titles), titles


def _service_says(server, who: dict) -> None:
    server.conversion_address = ("127.0.0.1", 9)
    server.conversion_enabled = True
    server.conversion_reachable = lambda: True
    server.conversion_service = lambda: who


@pytest.mark.parametrize("service, warned", [("seedvc", True), ("rvc", False)])
def test_the_larger_translator_is_warned_about_only_beside_seedvc(client, service, warned) -> None:
    """Measured: 1.3B beside RVC peaks at 6942 MiB; beside Seed-VC ~350 MiB is left."""
    from parliamo.ui.controller import PipelineController

    http, server, _calls = client
    server.controller = PipelineController(factory=lambda: (None, None))
    server.controller.mt_model = "models/ct2/nllb-200-distilled-1.3B"
    _service_says(server, {"service": service, "model": "presenter"})
    titles = [c["title"] for c in http.get("/preflight").json()["checks"]]
    assert any("NLLB 1.3B chosen beside Seed-VC" in t for t in titles) is warned, titles


def test_preflight_says_when_conversion_is_simply_off(client) -> None:
    http, server, _calls = client
    server.conversion_address = ("127.0.0.1", 1)
    server.conversion_enabled = False
    titles = [c["title"] for c in http.get("/preflight").json()["checks"]]
    assert any("Voice conversion off" in t for t in titles), titles


# ---------------------------------------------------------------------------
# languages
# ---------------------------------------------------------------------------


def test_languages_come_from_the_backends_tables(client) -> None:
    """Nothing is listed the translator cannot tag; voice means some synthesiser can speak it."""
    from parliamo.mt.ctranslate2_nllb import LANG_TAGS
    from parliamo.tts import backend_for

    http, _server, _calls = client
    body = http.get("/languages").json()
    assert set(body["sources"]) == set(LANG_TAGS)
    for target in body["targets"]:
        assert target["code"] in LANG_TAGS
        assert target["voice"] is (backend_for(target["code"]) is not None), target


def test_turkish_is_subtitles_only_without_its_piper_voice(client, monkeypatch) -> None:
    """Kokoro has no Turkish voice. Without Piper's on disk: offered, marked, not refused."""
    monkeypatch.setattr("parliamo.tts.piper_backend.available_languages", lambda *a: set())
    http, _server, _calls = client
    tr = next(t for t in http.get("/languages").json()["targets"] if t["code"] == "tr")
    assert tr["voice"] is False
    assert "subtitles only" in tr["note"]


def test_german_and_turkish_are_spoken_by_piper_when_their_voices_are_on_disk(client, monkeypatch) -> None:
    monkeypatch.setattr("parliamo.tts.piper_backend.available_languages", lambda *a: {"de", "tr"})
    http, _server, _calls = client
    targets = {t["code"]: t for t in http.get("/languages").json()["targets"]}
    for code in ("de", "tr"):
        assert targets[code]["voice"] is True and "Piper" in targets[code]["note"], targets[code]
    assert targets["es"]["voice"] is True and targets["es"]["note"] == "", "Spanish stays on Kokoro"


def test_friulian_is_a_target_spoken_with_the_italian_frontend(client) -> None:
    http, _server, _calls = client
    fur = next(t for t in http.get("/languages").json()["targets"] if t["code"] == "fur")
    assert fur["voice"] is True
    assert "Italian frontend" in fur["note"]


def test_choosing_languages_is_remembered_for_the_next_start(client) -> None:
    http, server, _calls = client
    server.controller = type("C", (), {"source_lang": None, "target_lang": None,
                                       "translator": None})()
    r = http.post("/languages", json={"source": "it", "target": "tr"})
    assert r.status_code == 200, r.text
    assert server.controller.source_lang == "it"
    assert server.controller.target_lang == "tr"
    assert r.json()["selected"] == {"source": "it", "target": "tr"}
    assert r.json()["applies"] == "at the next start"


def test_an_unknown_language_is_refused_with_a_reason(client) -> None:
    http, server, _calls = client
    server.controller = type("C", (), {"source_lang": None, "target_lang": None,
                                       "translator": None})()
    r = http.post("/languages", json={"target": "klingon"})
    assert r.status_code == 400
    assert "klingon" in r.json()["error"]
    assert server.controller.target_lang is None


# ---------------------------------------------------------------------------
# the native speaker's clips
# ---------------------------------------------------------------------------


def _native_fixture(tmp_path, monkeypatch) -> None:
    import json as _json

    import numpy as np
    import soundfile as sf

    clips = tmp_path / "data" / "friulian" / "marco" / "clips"
    clips.mkdir(parents=True)
    sf.write(str(clips / "marco-001.wav"), np.zeros(16000, dtype=np.float32), 16000)
    (clips / "index.json").write_text(_json.dumps({
        "speaker": "Marco Moroldo", "source": "Wikitongues", "licence": "CC BY-SA 4.0",
        "note": "Played as recorded.",
        "clips": [{"n": 1, "file": "marco-001.wav", "start_s": 2.3, "end_s": 13.1,
                   "duration_s": 10.8},
                  {"n": 2, "file": "marco-002.wav", "start_s": 14.0, "end_s": 17.0,
                   "duration_s": 3.0}],
    }), encoding="utf-8")
    (tmp_path / "data" / "friulian" / "audio").mkdir()
    (tmp_path / "data" / "friulian" / "marco" / "secret.wav").write_bytes(b"x")
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)


def test_native_clips_carry_the_attribution_and_only_files_that_exist(tmp_path, monkeypatch) -> None:
    """A native speaker's recording is played as recorded, with its licence beside it.

    The index lists two clips; one file is missing. The missing one must not
    become a broken play button on stage.
    """
    _native_fixture(tmp_path, monkeypatch)
    native = SubtitleServer().friulian_native()
    assert native["speaker"] == "Marco Moroldo"
    assert native["licence"] == "CC BY-SA 4.0"
    assert [c["file"] for c in native["clips"]] == ["marco-001.wav"]


def test_no_recording_means_no_section_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)
    assert SubtitleServer().friulian_native() is None


def test_clip_routes_serve_wavs_and_nothing_else(client, tmp_path, monkeypatch) -> None:
    """The clips folder is served; the folder above it is not."""
    http, _server, _calls = client
    _native_fixture(tmp_path, monkeypatch)

    ok = http.get("/friulian/audio/native/marco-001.wav")
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("audio/wav")

    for bad in ("/friulian/audio/native/index.json",
                "/friulian/audio/native/../secret.wav",
                "/friulian/audio/native/..%2Fsecret.wav",
                "/friulian/audio/elsewhere/marco-001.wav"):
        assert http.get(bad).status_code == 404, bad


def test_the_native_clips_are_never_offered_as_a_voice(tmp_path, monkeypatch) -> None:
    """The licence covers the recording, not the speaker's voice.

    `data/friulian/marco/` must not appear in the Voices tab, where a click
    would register it as a reference for the synthesiser.
    """
    _native_fixture(tmp_path, monkeypatch)
    (tmp_path / "data" / "voices").mkdir()
    names = [v["name"] for v in SubtitleServer().list_voices()]
    assert not any("marco" in n.lower() for n in names)


def test_an_index_whose_files_are_gone_is_no_section(tmp_path, monkeypatch) -> None:
    """A fresh clone has index.json and none of the WAVs (they are not in git)."""
    import json as _json

    clips = tmp_path / "data" / "friulian" / "marco" / "clips"
    clips.mkdir(parents=True)
    (clips / "index.json").write_text(_json.dumps({
        "speaker": "Marco Moroldo", "clips": [{"n": 1, "file": "marco-001.wav"}]}),
        encoding="utf-8")
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)
    assert SubtitleServer().friulian_native() is None


# ---------------------------------------------------------------------------
# models: only what is on the machine, each with its measured cost
# ---------------------------------------------------------------------------


def test_models_lists_only_what_is_on_disk(tmp_path, monkeypatch) -> None:
    """The venue has no internet: an option that downloads on first use fails on stage."""
    hub = tmp_path / "models" / "huggingface" / "hub"
    (hub / "models--deepdml--faster-whisper-large-v3-turbo-ct2").mkdir(parents=True)
    (tmp_path / "models" / "ct2" / "nllb-200-distilled-600M").mkdir(parents=True)
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)

    body = SubtitleServer().models()
    assert [o["id"] for o in body["asr"]] == ["large-v3-turbo"]
    assert [o["name"] for o in body["mt"]] == ["NLLB-200 600M"]
    assert "1213 MiB" in body["asr"][0]["cost"]


def test_the_bigger_translator_says_which_voice_service_it_fits_beside(tmp_path, monkeypatch) -> None:
    """It said "does NOT fit beside the voice service" - true of Seed-VC, measured false of RVC."""
    (tmp_path / "models" / "ct2" / "nllb-200-distilled-1.3B").mkdir(parents=True)
    (tmp_path / "models" / "huggingface" / "hub").mkdir(parents=True)
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)
    big = SubtitleServer().models()["mt"][0]
    assert "fits beside RVC" in big["cost"] and "NOT beside Seed-VC" in big["cost"]


def test_choosing_a_model_is_remembered_and_an_absent_one_refused(client, tmp_path, monkeypatch) -> None:
    http, server, _calls = client
    (tmp_path / "models" / "ct2" / "nllb-200-distilled-600M").mkdir(parents=True)
    (tmp_path / "models" / "huggingface" / "hub" /
     "models--Systran--faster-whisper-large-v3").mkdir(parents=True)
    monkeypatch.setattr("parliamo.paths.resolve", lambda p: tmp_path / p)
    server.controller = type("C", (), {"asr_model": None, "mt_model": None, "translator": None})()

    ok = http.post("/models", json={"asr": "large-v3"})
    assert ok.status_code == 200, ok.text
    assert server.controller.asr_model == "large-v3"

    bad = http.post("/models", json={"mt": "models/ct2/nllb-200-distilled-1.3B"})
    assert bad.status_code == 400
    assert "not on this machine" in bad.json()["error"]
    assert server.controller.mt_model is None


def test_streaming_toggle_reaches_a_running_pipeline(client) -> None:
    """The live switch: on flips the translator now, off clears its ledger."""
    from parliamo.ui.controller import PipelineController

    http, server, _calls = client
    resets: list[int] = []
    translator = type("T", (), {"stream_sentences": False, "stream_mode": "off",
                                "reset_streaming": lambda self: resets.append(1)})()
    controller = PipelineController(factory=lambda: (None, None))
    controller.translator = translator
    server.controller = controller

    assert http.get("/streaming").json() == {"mode": "off", "enabled": False, "available": True}
    on = http.post("/streaming", json={"enabled": True}).json()   # the old bool form
    assert on["mode"] == "sentence" and translator.stream_sentences is True
    assert resets == [1], "a mode change discards what the old policy was tracking"
    chunk = http.post("/streaming", json={}).json()               # no body: cycle
    assert chunk["mode"] == "chunk" and translator.stream_mode == "chunk"
    off = http.post("/streaming", json={}).json()
    assert off["mode"] == "off" and off["enabled"] is False
    bad = http.post("/streaming", json={"mode": "firehose"})
    assert bad.status_code == 400


def test_streaming_choice_is_kept_for_the_next_start(client) -> None:
    from parliamo.ui.controller import PipelineController

    http, server, _calls = client
    server.controller = PipelineController(factory=lambda: (None, None))
    http.post("/streaming", json={"mode": "chunk"})
    assert server.controller.stream_mode == "chunk"
    assert http.get("/streaming").json()["mode"] == "chunk"


def test_the_commit_wait_is_offered_with_its_measured_cost_and_remembered(client) -> None:
    """ADR 0009's table, as a setting. 300 is offered - it is what the presenter
    liked in the earlier prototype - and marked for what it measured."""
    from parliamo.ui.controller import PipelineController

    http, server, _calls = client
    server.controller = PipelineController(factory=lambda: (None, None))
    body = http.get("/tempo").json()
    by_ms = {o["ms"]: o for o in body["options"]}
    assert set(by_ms) == {300, 400, 500, 700}
    assert by_ms[300]["level"] == "warn" and "fragment" in by_ms[300]["note"]
    assert by_ms[500]["level"] == "ok"

    ok = http.post("/tempo", json={"min_silence_ms": 300})
    assert ok.status_code == 200 and server.controller.min_silence_ms == 300
    bad = http.post("/tempo", json={"min_silence_ms": 123})
    assert bad.status_code == 400


def test_the_page_says_when_its_server_is_gone() -> None:
    """Seen live: the presenter started a second copy, it moved to 8771 and
    said so, the first was closed, and the tab on 8770 sat blank with a red
    dot the size of a pea. "Everything was empty." The banner says what
    happened and what to do, and that reloading will not help."""
    html = _page("app.html")
    assert 'id="dead-banner"' in html
    assert "NOT CONNECTED" in html
    assert "8771" in html and "reloading this tab will not help" in html
    body = html.split("<script>", 1)[1]
    assert "dead-banner" in body and "onerror" in body


def test_a_slow_operator_action_does_not_freeze_the_subtitles() -> None:
    """The event loop carries the subtitle stream for every browser.

    Measured before the fix: the pre-flight list blocked it for 1.46 s, Stop
    for as long as unloading the models took, and a voice swap for a whole
    warm-up conversion - the volunteer demonstration's most watched moment.
    A slow handler must run on a worker thread while /state stays instant.
    """
    import threading
    import time
    import urllib.request

    from parliamo.ui.controller import PipelineController

    def slow_stop():
        time.sleep(1.5)
        return {"status": "stopped", "detail": ""}

    controller = PipelineController(factory=lambda: (None, None))
    controller.stop = slow_stop
    server = SubtitleServer(port=_free_port(), controller=controller)
    url = server.start()
    try:
        worker = threading.Thread(target=lambda: urllib.request.urlopen(
            urllib.request.Request(url + "pipeline/stop", data=b"{}", method="POST"),
            timeout=10).read())
        worker.start()
        time.sleep(0.2)                      # the slow stop is now in flight
        t0 = time.perf_counter()
        urllib.request.urlopen(url + "state", timeout=10).read()
        waited = time.perf_counter() - t0
        worker.join(timeout=10)
        assert waited < 0.5, f"/state waited {waited:.2f} s behind a slow handler"
    finally:
        server.stop()


def test_a_running_parliamo_is_found_and_a_stranger_is_not() -> None:
    """A second copy must refuse to start, not quietly move to 8771."""
    import socket as _socket

    from parliamo.ui.server import running_instance

    port = _free_port()
    server = SubtitleServer(port=port)
    url = server.start()
    try:
        assert running_instance("127.0.0.1", port) == url
    finally:
        server.stop()

    # Something else holding the port is not a parlIAmo.
    other = _socket.socket()
    other.bind(("127.0.0.1", 0))
    other.listen(1)
    try:
        assert running_instance("127.0.0.1", other.getsockname()[1], span=1, timeout=0.3) is None
    finally:
        other.close()


def test_preflight_finds_the_voice_service_that_is_actually_running(client, monkeypatch) -> None:
    """Configured for RVC, Seed-VC running: the row names Seed-VC, not 'not running'."""
    import socket as socket_module

    from parliamo.tts import conversion

    http, server, _calls = client
    live = socket_module.socket()
    live.bind(("127.0.0.1", 0))
    live.listen(4)
    port = live.getsockname()[1]
    monkeypatch.setattr(conversion, "KNOWN_PORTS", (_free_port(), port))
    server.conversion_address = ("127.0.0.1", _free_port())   # the configured one is down
    server.conversion_enabled = True
    try:
        titles = [c["title"] for c in http.get("/preflight").json()["checks"]]
    finally:
        live.close()
    assert not any("not running" in t for t in titles), titles
    assert any(str(port) in t for t in titles), titles


def test_the_studio_offers_only_consented_voices_and_refuses_others(client, tmp_path) -> None:
    import socket

    import numpy as np
    import soundfile as sf

    from parliamo.ui.studio import Studio

    http, server, _calls = client
    mine, stranger = tmp_path / "mine.wav", tmp_path / "stranger.wav"
    for path in (mine, stranger):
        sf.write(path, np.zeros(2400, dtype=np.float32), 24000)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    server._studio = Studio(lambda: [
        {"name": "mine", "path": str(mine), "consent": "presenter's own voice"},
        {"name": "stranger", "path": str(stranger), "consent": ""}], port=free, root=tmp_path)

    state = http.get("/studio").json()
    assert [v["name"] for v in state["voices"]] == ["mine"]
    assert state["service"]["up"] is False and "it" in state["languages"]

    refused = http.post("/studio/speak", json={"text": "Ciao.", "voice": str(stranger)})
    assert refused.status_code == 400 and "consent" in refused.json()["error"]
    down = http.post("/studio/speak", json={"text": "Ciao.", "voice": str(mine)})
    assert down.status_code == 400 and "not running" in down.json()["error"]
    assert http.get("/studio/clip/..%2F..%2Fsecret").status_code == 404


def test_every_tab_has_a_section_the_page_can_show() -> None:
    # show() un-hides only the sections named in SECTIONS. The Studio tab was
    # added without its entry, and clicking it hid everything: a blank page.
    import re

    from parliamo.ui.server import APP

    page = APP.read_text(encoding="utf-8")
    tabs = set(re.findall(r'class="tab[^"]*" data-t="([a-z]+)"', page))
    listed = set(re.findall(r'"([a-z]+)"', re.search(r"const SECTIONS = \[([^\]]*)\]", page).group(1)))
    sections = set(re.findall(r'<section id="([a-z]+)"', page))
    assert tabs, "no tabs found - the pattern no longer matches the page"
    assert tabs <= listed, f"tabs show() cannot show: {sorted(tabs - listed)}"
    assert tabs <= sections, f"tabs without a section: {sorted(tabs - sections)}"


def _fake_devices(monkeypatch, by_spec: dict) -> list:
    """resolve_device stand-in: *by_spec* maps (spec, direction) to a device."""
    import parliamo.audio.devices as devices

    asked: list = []

    def resolve(spec, direction, devices_=None, **_):
        asked.append((spec, direction))
        if (spec, direction) not in by_spec:
            raise devices.DeviceResolutionError(f"no {spec!r}")
        return by_spec[(spec, direction)]

    monkeypatch.setattr(devices, "resolve_device", resolve)
    # The config's devices fill in what the page left unset; not these tests'.
    import parliamo.config as config

    real = config.load_config

    def no_config_devices(*a, **k):
        cfg = real(*a, **k)
        cfg.audio.input_device = None
        cfg.audio.output_device = None
        return cfg

    monkeypatch.setattr(config, "load_config", no_config_devices)
    return asked


def _device(name: str, api: str, out: bool = False):
    from parliamo.audio.devices import AudioDevice

    return AudioDevice(index=1, name=name, hostapi_name=api,
                       max_input_channels=0 if out else 2, max_output_channels=2 if out else 0,
                       default_samplerate=48000.0, default_low_input_latency=0.003,
                       default_low_output_latency=0.003)


def test_preflight_warns_about_speakers_on_mme(client, monkeypatch) -> None:
    """The choppy translation at the 2026-10-02 rehearsal: speakers on MME."""
    c, server, _ = client
    dante = "Speakers (Dante USB I/O Module)"
    server.controller = type("C", (), {"input_device": None,
                                       "output_device": f"{dante}@MME"})()
    _fake_devices(monkeypatch, {
        (None, "input"): _device("Microphone (Realtek(R) Audio)", "Windows WASAPI"),
        (f"{dante}@MME", "output"): _device(dante, "MME", out=True),
        (None, "output"): _device("Speakers (Realtek(R) Audio)", "Windows WASAPI", out=True),
    })
    rows = {r["title"]: r for r in c.get("/preflight").json()["checks"]}
    row = rows[f"Speakers on MME: {dante}"]
    assert row["level"] == "warn" and "WASAPI" in row["detail"]


def test_preflight_checks_the_chosen_devices_not_windows_default(client, monkeypatch) -> None:
    """The microphone row used to describe Windows' default whatever was picked."""
    c, server, _ = client
    jack = "Microphone (Realtek(R) Audio)@Windows WASAPI"
    dante = "Speakers (Dante USB I/O Module)@Windows WASAPI"
    server.controller = type("C", (), {"input_device": jack, "output_device": dante})()
    asked = _fake_devices(monkeypatch, {
        (jack, "input"): _device("Microphone (Realtek(R) Audio)", "Windows WASAPI"),
        (dante, "output"): _device("Speakers (Dante USB I/O Module)", "Windows WASAPI", out=True),
    })
    rows = {r["title"]: r for r in c.get("/preflight").json()["checks"]}
    assert rows["Microphone: Microphone (Realtek(R) Audio)"]["level"] == "ok"
    assert rows["Speakers: Speakers (Dante USB I/O Module)"]["level"] == "ok"
    assert (None, "input") not in asked and (None, "output") not in asked


def test_preflight_says_when_the_chosen_device_is_gone(client, monkeypatch) -> None:
    c, server, _ = client
    server.controller = type("C", (), {"input_device": "USB Mic@Windows WASAPI",
                                       "output_device": None})()
    _fake_devices(monkeypatch, {
        (None, "input"): _device("Microphone Array (Intel)", "Windows WASAPI"),
        (None, "output"): _device("Speakers (Realtek(R) Audio)", "Windows WASAPI", out=True),
    })
    rows = {r["title"]: r for r in c.get("/preflight").json()["checks"]}
    row = rows["Microphone: Microphone Array (Intel)"]
    assert row["level"] == "warn" and "USB Mic" in row["detail"] and "not here" in row["detail"]
