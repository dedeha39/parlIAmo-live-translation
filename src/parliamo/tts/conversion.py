"""Voice conversion over a local socket.

Seed-VC cannot live in the main environment: it needs an older
``huggingface_hub`` than ``datasets`` and ``transformers`` do, and forcing that
downgrade is the exact trade that broke this environment once already. So it
runs in its own ``seedvc`` environment behind a socket, started by
``scripts/voice_conversion_server.py``.

The cost of the split is about a millisecond of loopback against ~570 ms of
conversion compute, so it is free in latency terms and permanently isolates a
dependency conflict instead of re-litigating it at every upgrade.

Protocol
--------
Deliberately tiny, stdlib-only on this side, and framed rather than delimited so
a partial read can never be mistaken for a complete message::

    4 bytes   big-endian length of the JSON header
    N bytes   UTF-8 JSON header
    M bytes   raw float32 little-endian samples (M is given in the header)

The same shape in both directions. No pickling: a socket that unpickles
whatever it is sent is a remote-code-execution hole, and this one listens on a
laptop that will be on a conference network.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: The voice services this project runs: RVC (the presenter's trained voice),
#: Seed-VC (zero-shot, for a volunteer) and OmniVoice (zero-shot, speaks the
#: sentence itself). One at a time on the GPU, so the one that answers is the
#: one in use.
KNOWN_PORTS = (8766, 8765, 8767)


def find_voice_service(host: str, preferred: int | None, timeout: float = 1.0) -> int | None:
    """The port of the voice service that is running: *preferred* first, then the known ones.

    The talk swaps services mid-way - RVC stopped, Seed-VC started for the
    volunteer - and a pipeline pinned to one port knocked on the closed one
    and spoke the volunteer in the generic voice.
    """
    candidates = [p for p in (preferred, *KNOWN_PORTS) if p]
    for port in dict.fromkeys(candidates):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return int(port)
        except OSError:
            continue
    return None

#: Refuse anything larger rather than allocating on a claimed length. One
#: minute of 24 kHz float32 audio is about 5.8 MB; 64 MB is generous.
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class ConversionError(RuntimeError):
    """Raised when the conversion service is unreachable or fails."""


class ConversionTimeout(ConversionError):
    """The service accepted the request and did not answer in time.

    Different from unreachable, which fails in milliseconds: a hung service
    costs the whole timeout on every sentence sent to it, so the caller should
    stop sending for a while rather than retry at once.
    """


# ---------------------------------------------------------------------------
# framing
# ---------------------------------------------------------------------------


def send_message(sock: socket.socket, header: dict[str, Any], audio: np.ndarray) -> None:
    """Send one framed message."""
    payload = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    header = {**header, "audio_bytes": len(payload)}
    blob = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack(">I", len(blob)))
    sock.sendall(blob)
    if payload:
        sock.sendall(payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes, or raise. recv() may return short reads."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConversionError(
                f"connection closed with {remaining} of {n} bytes still expected"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], np.ndarray]:
    """Read one framed message."""
    (header_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    if header_len > 1 << 20:
        raise ConversionError(f"header claims {header_len} bytes; refusing")
    header = json.loads(_recv_exactly(sock, header_len).decode("utf-8"))

    audio_bytes = int(header.get("audio_bytes", 0))
    if audio_bytes > MAX_PAYLOAD_BYTES:
        raise ConversionError(f"payload claims {audio_bytes} bytes; refusing")
    if audio_bytes == 0:
        return header, np.zeros(0, dtype=np.float32)
    raw = _recv_exactly(sock, audio_bytes)
    return header, np.frombuffer(raw, dtype="<f4").astype(np.float32)


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Conversion:
    """One converted utterance."""

    audio: np.ndarray
    sample_rate: int
    compute_s: float
    round_trip_s: float
    diffusion_steps: int

    @property
    def duration_s(self) -> float:
        return self.audio.size / float(self.sample_rate)

    @property
    def rtf(self) -> float:
        if self.duration_s <= 0:
            return float("inf")
        return self.compute_s / self.duration_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "audio_s": round(self.duration_s, 3),
            "compute_s": round(self.compute_s, 3),
            "round_trip_s": round(self.round_trip_s, 3),
            "ipc_overhead_s": round(self.round_trip_s - self.compute_s, 4),
            "rtf": round(self.rtf, 3),
            "diffusion_steps": self.diffusion_steps,
        }


class VoiceConverter:
    """Client for the Seed-VC conversion service.

    ``diffusion_steps`` is the quality/speed dial. Measured on 9.72 s of audio:
    4 steps 1.02 s, 10 steps 1.29 s, 25 steps 1.96 s.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        diffusion_steps: int = 10,
        timeout: float = 60.0,
    ) -> None:
        self.host = host
        self.port = port
        self.diffusion_steps = diffusion_steps
        self.timeout = timeout

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    def available(self) -> bool:
        """Is the service reachable? Cheap enough to call before a rehearsal."""
        try:
            with socket.create_connection((self.host, self.port), timeout=2.0):
                return True
        except OSError:
            return False

    def convert(
        self,
        audio: np.ndarray,
        sample_rate: int,
        reference_path: str,
        diffusion_steps: int | None = None,
        pitch: float | None = None,
        text: str | None = None,
        language: str | None = None,
        timeout: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> Conversion:
        """Convert *audio* into the voice of the speaker in *reference_path*.

        *pitch*, in semitones, is for RVC, which keeps the pitch it is given;
        Seed-VC ignores it. None leaves the service's own setting.

        *text* and *language* are the sentence the audio says. OmniVoice speaks
        the sentence itself rather than converting the audio, and refuses a
        request without them; RVC and Seed-VC ignore both.

        *timeout* overrides the client's for this call - for a warm-up, which
        on OmniVoice may include preparing a new reference on the CPU.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        steps = diffusion_steps if diffusion_steps is not None else self.diffusion_steps
        wait = float(timeout) if timeout is not None else self.timeout

        t0 = time.perf_counter()
        try:
            # Connecting and answering fail differently. Not connecting is an
            # absent service - on Windows sometimes a connect timeout rather
            # than a refusal. Connecting and then not answering is a hung one.
            sock = socket.create_connection((self.host, self.port), timeout=wait)
        except OSError as exc:
            raise ConversionError(
                f"voice conversion service unreachable at {self.address}: {exc}. "
                "Start it with: <seedvc-env>/python scripts/voice_conversion_server.py"
            ) from exc
        try:
            with sock:
                sock.settimeout(wait)
                header_out: dict[str, Any] = {
                    "op": "convert",
                    "sample_rate": int(sample_rate),
                    "reference_path": reference_path,
                    "diffusion_steps": int(steps),
                }
                if pitch is not None:
                    header_out["pitch"] = round(float(pitch), 2)
                if text:
                    header_out["text"] = text
                if language:
                    header_out["language"] = language
                if options:
                    # OmniVoice's generation settings (speed and the like);
                    # the other services ignore them.
                    header_out["options"] = options
                send_message(sock, header_out, audio)
                header, converted = recv_message(sock)
        except TimeoutError as exc:
            raise ConversionTimeout(
                f"voice conversion service at {self.address} did not answer in "
                f"{wait:.0f} s"
            ) from exc
        except OSError as exc:
            raise ConversionError(
                f"voice conversion service at {self.address} dropped the connection: {exc}"
            ) from exc
        except ValueError as exc:
            # A reply that is not the protocol - JSON that does not parse, a
            # length that does not add up. Raised as anything but
            # ConversionError it escaped the caller's fallback, and the
            # sentence was dropped instead of spoken in the generic voice.
            raise ConversionError(f"unreadable reply from {self.address}: {exc}") from exc
        round_trip = time.perf_counter() - t0

        if header.get("error"):
            raise ConversionError(f"conversion failed: {header['error']}")

        return Conversion(
            audio=converted,
            sample_rate=int(header.get("sample_rate", sample_rate)),
            compute_s=float(header.get("compute_s", 0.0)),
            round_trip_s=round_trip,
            diffusion_steps=steps,
        )

    def ping(self) -> dict[str, Any]:
        """Ask the service what it is and whether its model is loaded."""
        try:
            with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                send_message(sock, {"op": "ping"}, np.zeros(0, dtype=np.float32))
                header, _ = recv_message(sock)
                return header
        except OSError as exc:
            raise ConversionError(f"service unreachable at {self.address}: {exc}") from exc
