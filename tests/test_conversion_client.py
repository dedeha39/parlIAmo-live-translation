"""The voice-conversion socket protocol.

Tested against a fake server rather than Seed-VC, because what needs pinning
here is the *framing* - short reads, size limits, error propagation - not the
model. A protocol bug shows up on stage as a hang, which is the worst way to
find one.
"""

from __future__ import annotations

import socket
import socketserver
import threading

import numpy as np
import pytest

from parliamo.tts.conversion import (
    Conversion,
    ConversionError,
    VoiceConverter,
    recv_message,
    send_message,
)


class _EchoHandler(socketserver.BaseRequestHandler):
    """Returns the audio it was sent, scaled, so round-trips are verifiable."""

    def handle(self) -> None:
        header, audio = recv_message(self.request)
        if header.get("op") == "ping":
            send_message(self.request, {"service": "fake", "model_loaded": True},
                         np.zeros(0, dtype=np.float32))
            return
        if header.get("reference_path") == "ECHO_HEADER":
            import json as _json

            send_message(self.request, {"sample_rate": 24000, "compute_s": 0.0,
                                        "echo": _json.dumps(header)},
                         np.zeros(0, dtype=np.float32))
            return
        if header.get("reference_path") == "MAKE_IT_FAIL":
            send_message(self.request, {"error": "simulated failure"},
                         np.zeros(0, dtype=np.float32))
            return
        send_message(
            self.request,
            {"sample_rate": 22050, "compute_s": 0.5},
            audio * 0.5,
        )


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address
    srv.shutdown()
    srv.server_close()


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------


def test_audio_survives_the_round_trip(server) -> None:
    host, port = server
    client = VoiceConverter(host=host, port=port)
    source = np.linspace(-1, 1, 24000, dtype=np.float32)

    result = client.convert(source, 24000, reference_path="ref.wav")

    assert result.audio.size == source.size
    np.testing.assert_allclose(result.audio, source * 0.5, rtol=1e-5)
    assert result.sample_rate == 22050
    assert result.compute_s == 0.5


def test_large_payload_is_not_truncated(server) -> None:
    """One recv() does not return a megabyte. Short reads must be handled."""
    host, port = server
    client = VoiceConverter(host=host, port=port)
    source = np.random.default_rng(0).standard_normal(600_000).astype(np.float32)

    result = client.convert(source, 24000, reference_path="ref.wav")
    assert result.audio.size == source.size
    np.testing.assert_allclose(result.audio, source * 0.5, rtol=1e-4)


def test_empty_audio_round_trips(server) -> None:
    host, port = server
    result = VoiceConverter(host=host, port=port).convert(
        np.zeros(0, dtype=np.float32), 24000, "ref.wav"
    )
    assert result.audio.size == 0


def test_diffusion_steps_are_forwarded_and_reported(server) -> None:
    host, port = server
    client = VoiceConverter(host=host, port=port, diffusion_steps=10)
    assert client.convert(np.zeros(100, dtype=np.float32), 24000, "r.wav").diffusion_steps == 10
    assert client.convert(
        np.zeros(100, dtype=np.float32), 24000, "r.wav", diffusion_steps=4
    ).diffusion_steps == 4


def test_the_sentence_text_and_language_are_sent_when_given(server, monkeypatch) -> None:
    # OmniVoice speaks the sentence rather than converting the audio.
    import json

    import parliamo.tts.conversion as module

    seen: list[dict] = []
    real = module.recv_message

    def spy(sock):
        header, audio = real(sock)
        seen.append(header)
        return header, audio

    monkeypatch.setattr(module, "recv_message", spy)
    host, port = server
    client = VoiceConverter(host=host, port=port)
    client.convert(np.zeros(10, dtype=np.float32), 24000, "ECHO_HEADER",
                   text="Ciao, come state?", language="it")
    sent = json.loads(seen[0]["echo"])
    assert sent["text"] == "Ciao, come state?" and sent["language"] == "it"

    client.convert(np.zeros(10, dtype=np.float32), 24000, "ECHO_HEADER")
    sent = json.loads(seen[1]["echo"])
    assert "text" not in sent and "language" not in sent


def test_a_call_can_wait_longer_than_the_client_timeout() -> None:
    # A warm-up on OmniVoice may prepare a new reference first; it must be
    # allowed to take longer than a sentence without changing the sentence
    # timeout.
    import socket as _socket
    import time as _time

    class _Slow(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            recv_message(self.request)
            _time.sleep(0.6)
            send_message(self.request, {"sample_rate": 24000}, np.zeros(1, dtype=np.float32))

    srv = _Server(("127.0.0.1", 0), _Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        host, port = srv.server_address
        client = VoiceConverter(host=host, port=port, timeout=0.2)
        from parliamo.tts.conversion import ConversionTimeout

        with pytest.raises(ConversionTimeout):
            client.convert(np.zeros(10, dtype=np.float32), 24000, "r.wav")
        result = client.convert(np.zeros(10, dtype=np.float32), 24000, "r.wav", timeout=3.0)
        assert result.audio.size == 1
        assert client.timeout == 0.2
    finally:
        srv.shutdown()
        srv.server_close()
        del _socket


def test_omnivoice_is_one_of_the_known_services() -> None:
    from parliamo.tts.conversion import KNOWN_PORTS

    assert 8767 in KNOWN_PORTS


# ---------------------------------------------------------------------------
# failure modes
# ---------------------------------------------------------------------------


def test_unreachable_service_names_the_fix() -> None:
    """The error has to say how to start the server, not just that it failed."""
    client = VoiceConverter(host="127.0.0.1", port=1, timeout=1.0)
    with pytest.raises(ConversionError, match="voice_conversion_server"):
        client.convert(np.zeros(100, dtype=np.float32), 24000, "ref.wav")


def test_server_side_error_is_raised_not_swallowed(server) -> None:
    host, port = server
    with pytest.raises(ConversionError, match="simulated failure"):
        VoiceConverter(host=host, port=port).convert(
            np.zeros(100, dtype=np.float32), 24000, "MAKE_IT_FAIL"
        )


def test_available_reports_reachability(server) -> None:
    host, port = server
    assert VoiceConverter(host=host, port=port).available() is True
    assert VoiceConverter(host="127.0.0.1", port=1).available() is False


def test_ping_reports_service_state(server) -> None:
    host, port = server
    info = VoiceConverter(host=host, port=port).ping()
    assert info["service"] == "fake"
    assert info["model_loaded"] is True


def test_oversized_payload_is_refused_before_allocating() -> None:
    """A header claiming 4 GB must not cause a 4 GB allocation."""
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _LyingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        with socket.create_connection((host, port), timeout=5) as sock:
            send_message(sock, {"op": "ping"}, np.zeros(0, dtype=np.float32))
            with pytest.raises(ConversionError, match="refusing"):
                recv_message(sock)
    finally:
        server.shutdown()
        server.server_close()


class _LyingHandler(socketserver.BaseRequestHandler):
    """Claims a huge payload it does not send."""

    def handle(self) -> None:
        import json
        import struct

        recv_message(self.request)
        blob = json.dumps({"audio_bytes": 4 * 1024 * 1024 * 1024}).encode()
        self.request.sendall(struct.pack(">I", len(blob)))
        self.request.sendall(blob)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def test_conversion_reports_ipc_overhead_separately() -> None:
    """Compute and round-trip are different numbers and both matter.

    The whole argument for a second process is that IPC is negligible against
    conversion. That claim should be visible in the numbers, not assumed.
    """
    conversion = Conversion(
        audio=np.zeros(22050, dtype=np.float32), sample_rate=22050,
        compute_s=0.57, round_trip_s=0.58, diffusion_steps=10,
    )
    payload = conversion.as_dict()
    assert payload["ipc_overhead_s"] == pytest.approx(0.01, abs=1e-4)
    assert payload["rtf"] == pytest.approx(0.57)


def test_rtf_of_empty_conversion_is_infinite() -> None:
    conversion = Conversion(
        audio=np.zeros(0, dtype=np.float32), sample_rate=22050,
        compute_s=0.1, round_trip_s=0.1, diffusion_steps=10,
    )
    assert conversion.rtf == float("inf")


# ---------------------------------------------------------------------------
# the server must not start twice
# ---------------------------------------------------------------------------


def _server_module():
    """Import the server script without running its CLI."""
    import importlib.util

    from parliamo.paths import REPO_ROOT

    path = REPO_ROOT / "scripts" / "voice_conversion_server.py"
    spec = importlib.util.spec_from_file_location("voice_conversion_server", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_does_not_allow_a_second_bind() -> None:
    """`allow_reuse_address` on Windows lets two live servers share a port.

    A server left running overnight was still listening on 8765 when a second
    was started for a measurement. Both held ~2.9 GB, the card sat at 7001 MiB
    of 8188, and conversions ran 4.8x to 9.8x slower than their recorded
    figures - which was very nearly written down as a Seed-VC regression.

    Refusing to start is the correct behaviour: a service that starts
    invisibly wrong is worse than one that will not start.
    """
    module = _server_module()
    assert module.Server.allow_reuse_address is False


def test_port_in_use_detects_a_listener() -> None:
    module = _server_module()
    with socketserver.TCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as srv:
        host, port = srv.server_address
        assert module.port_in_use(host, port) is True


def test_port_in_use_is_false_for_a_free_port() -> None:
    module = _server_module()
    with socketserver.TCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as srv:
        _, port = srv.server_address
    # The server is closed now, so nothing is listening there.
    assert module.port_in_use("127.0.0.1", port) is False


def test_a_service_that_accepts_and_never_answers_is_a_timeout() -> None:
    """Hung, not absent: the translator pauses conversion instead of waiting again."""
    import socket as socket_module

    import numpy as np

    from parliamo.tts.conversion import ConversionTimeout, VoiceConverter

    server = socket_module.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    try:
        converter = VoiceConverter(port=server.getsockname()[1], timeout=0.5)
        try:
            converter.convert(np.zeros(160, np.float32), 16000, "ref.wav")
        except ConversionTimeout:
            pass
        else:
            raise AssertionError("expected ConversionTimeout")
    finally:
        server.close()


# ---------------------------------------------------------------------------
# whichever voice service is running is the one used
# ---------------------------------------------------------------------------


def _listening():
    import socket as socket_module

    server = socket_module.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(4)
    return server, server.getsockname()[1]


def _closed_port() -> int:
    import socket as socket_module

    s = socket_module.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_the_running_service_is_found_when_the_configured_one_is_not_up(monkeypatch) -> None:
    """The volunteer switch: RVC stopped, Seed-VC started - Start must find Seed-VC.

    The application was pinned to one port. Swapping services, as the talk
    does for the volunteer, left it knocking on the closed one, and the
    volunteer would have been spoken in the generic voice.
    """
    from parliamo.tts import conversion

    server, other = _listening()
    configured = _closed_port()
    monkeypatch.setattr(conversion, "KNOWN_PORTS", (configured, other))
    try:
        assert conversion.find_voice_service("127.0.0.1", configured) == other
    finally:
        server.close()


def test_the_configured_service_wins_when_it_is_up(monkeypatch) -> None:
    from parliamo.tts import conversion

    first, preferred = _listening()
    second, other = _listening()
    monkeypatch.setattr(conversion, "KNOWN_PORTS", (other, preferred))
    try:
        assert conversion.find_voice_service("127.0.0.1", preferred) == preferred
    finally:
        first.close()
        second.close()


def test_no_service_running_is_none(monkeypatch) -> None:
    from parliamo.tts import conversion

    monkeypatch.setattr(conversion, "KNOWN_PORTS", (_closed_port(), _closed_port()))
    assert conversion.find_voice_service("127.0.0.1", _closed_port()) is None
