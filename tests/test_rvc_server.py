"""The RVC server speaks the pipeline's conversion protocol byte for byte.

The server itself runs in Applio's environment and cannot be imported here
whole; its framing functions can, and they have to round-trip with
parliamo.tts.conversion or the pipeline talks to a wall.
"""

from __future__ import annotations

import importlib.util
import socket
import threading
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rvc_server.py"


def _framing():
    spec = importlib.util.spec_from_file_location("rvc_server_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_server_framing_round_trips_with_the_client() -> None:
    from parliamo.tts import conversion as client

    server = _framing()
    a, b = socket.socketpair()
    audio = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)

    def serve() -> None:
        header, got = server.recv_message(b)
        server.send_message(b, {"sample_rate": 48000, "voice": "presenter",
                                "echo": header.get("op")}, got[::2])

    t = threading.Thread(target=serve)
    t.start()
    client.send_message(a, {"op": "convert", "sample_rate": 24000}, audio)
    header, back = client.recv_message(a)
    t.join(timeout=5)

    assert header["echo"] == "convert" and header["voice"] == "presenter"
    assert header["sample_rate"] == 48000
    assert back.dtype == np.float32 and back.shape == (12000,)
    np.testing.assert_allclose(back, audio[::2])


def test_a_ping_carries_no_audio_either_way() -> None:
    from parliamo.tts import conversion as client

    server = _framing()
    a, b = socket.socketpair()

    def serve() -> None:
        header, got = server.recv_message(b)
        assert got.size == 0
        server.send_message(b, {"service": "rvc", "model": "presenter"},
                            np.zeros(0, dtype=np.float32))

    t = threading.Thread(target=serve)
    t.start()
    client.send_message(a, {"op": "ping"}, np.zeros(0, dtype=np.float32))
    header, back = client.recv_message(a)
    t.join(timeout=5)
    assert header["service"] == "rvc" and back.size == 0


def _touch(folder: Path, *names: str) -> None:
    for name in names:
        (folder / name).write_bytes(b"")


def test_the_best_epoch_is_served_when_training_marked_one(tmp_path) -> None:
    """Past overtraining the last epoch is worse; the best is what training kept."""
    server = _framing()
    _touch(tmp_path, "presenter_100e_2100s.pth", "presenter_200e_4200s_best_epoch.pth",
           "presenter_300e_6300s.pth", "G_2333333.pth", "D_2333333.pth")
    assert server.pick_weights(tmp_path, "presenter").name == "presenter_200e_4200s_best_epoch.pth"


def test_without_a_best_epoch_the_last_epoch_wins_by_number_not_by_name(tmp_path) -> None:
    """"50e" sorts after "300e" as text; the epoch is compared as a number."""
    server = _framing()
    _touch(tmp_path, "presenter_300e_6300s.pth", "presenter_50e_1050s.pth")
    assert server.pick_weights(tmp_path, "presenter").name == "presenter_300e_6300s.pth"


def test_the_choice_does_not_depend_on_file_times(tmp_path) -> None:
    """The real folder: both 300e files written in the same second."""
    import os

    server = _framing()
    _touch(tmp_path, "presenter_300e_6300s_best_epoch.pth", "presenter_300e_6300s.pth",
           "presenter_250e_5250s.pth")
    os.utime(tmp_path / "presenter_250e_5250s.pth", (2e9, 2e9))   # newest on disk
    assert server.pick_weights(tmp_path, "presenter").name == "presenter_300e_6300s_best_epoch.pth"


def test_no_weights_is_said_plainly(tmp_path) -> None:
    import pytest

    server = _framing()
    _touch(tmp_path, "G_2333333.pth")
    with pytest.raises(FileNotFoundError, match="no trained weights"):
        server.pick_weights(tmp_path, "presenter")
