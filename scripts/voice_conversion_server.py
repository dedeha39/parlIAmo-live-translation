#!/usr/bin/env python
"""Seed-VC voice conversion, served over a local socket.

Run this with the **seedvc** environment's interpreter, not the main one::

    C:\\Users\\you\\miniconda3\\envs\\seedvc\\python.exe scripts\\voice_conversion_server.py

Why a separate process
----------------------
Seed-VC needs an older ``huggingface_hub`` than ``datasets`` and
``transformers`` do. Forcing that downgrade into the main environment is the
same trade that once replaced a CUDA torch build with the CPU wheel and broke
every stage silently. Conversion costs ~570 ms of compute per sentence against
about a millisecond of loopback, so the split is free and the conflict stays
isolated.

Binds to loopback only. The protocol carries JSON and raw float32 samples and
never unpickles anything.
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
from typing import Any

import numpy as np

log = logging.getLogger("vc-server")

# The client lives in the other environment, so the framing is duplicated here
# rather than imported. Keeping it small is what makes that acceptable; if it
# grows, move it to a file both environments can read.
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def send_message(sock: socket.socket, header: dict[str, Any], audio: np.ndarray) -> None:
    payload = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    blob = json.dumps({**header, "audio_bytes": len(payload)}).encode("utf-8")
    sock.sendall(struct.pack(">I", len(blob)))
    sock.sendall(blob)
    if payload:
        sock.sendall(payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(f"closed with {remaining} of {n} bytes expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], np.ndarray]:
    (header_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    if header_len > 1 << 20:
        raise ValueError(f"header claims {header_len} bytes")
    header = json.loads(_recv_exactly(sock, header_len).decode("utf-8"))
    audio_bytes = int(header.get("audio_bytes", 0))
    if audio_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload claims {audio_bytes} bytes")
    if audio_bytes == 0:
        return header, np.zeros(0, dtype=np.float32)
    raw = _recv_exactly(sock, audio_bytes)
    return header, np.frombuffer(raw, dtype="<f4").astype(np.float32)


class Model:
    """Holds the loaded converter. One instance, guarded by a lock.

    Seed-VC is not safe to call concurrently and the GPU has room for one copy,
    so requests are serialised. The pipeline sends one sentence at a time
    anyway; the lock exists so a stray second client cannot corrupt state.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._wrapper: Any = None
        self._lock = threading.Lock()
        self.loaded_at: float | None = None

    def load(self) -> None:
        import torch
        from seed_vc.seed_vc_wrapper import SeedVCWrapper

        t0 = time.perf_counter()
        log.info("loading Seed-VC on %s ...", self.device)
        # Must be a torch.device: the library calls .type on this argument and
        # a plain string fails deep inside convert_voice.
        self._wrapper = SeedVCWrapper(device=torch.device(self.device))
        self.loaded_at = time.time()
        log.info("loaded in %.1fs", time.perf_counter() - t0)

    def convert(
        self, audio: np.ndarray, sample_rate: int, reference: str, steps: int
    ) -> tuple[np.ndarray, int, float]:
        import soundfile as sf

        if not os.path.exists(reference):
            raise FileNotFoundError(f"reference voice not found: {reference}")

        # Seed-VC reads from paths, so the incoming audio goes to a temp file.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            source_path = tmp.name
        try:
            sf.write(source_path, audio, sample_rate, subtype="FLOAT")
            with self._lock:
                t0 = time.perf_counter()
                last = None
                for item in self._wrapper.convert_voice(
                    source=source_path,
                    target=reference,
                    diffusion_steps=steps,
                    inference_cfg_rate=0.7,
                    stream_output=True,
                ):
                    last = item
                compute = time.perf_counter() - t0
            if last is None:
                raise RuntimeError("converter produced no output")
            # The stream yields (mp3_bytes, (sample_rate, waveform)).
            _mp3, (out_rate, wave) = last
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            # Some builds return int16-scaled floats.
            if np.abs(wave).max() > 1.5:
                wave = wave / 32768.0
            return wave, int(out_rate), compute
        finally:
            # A temp file that outlives us is untidy, not fatal.
            with contextlib.suppress(OSError):
                os.unlink(source_path)


MODEL: Model | None = None


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        assert MODEL is not None
        try:
            header, audio = recv_message(self.request)
        except Exception as exc:
            log.warning("malformed request: %s", exc)
            return

        op = header.get("op", "convert")
        try:
            if op == "ping":
                send_message(
                    self.request,
                    {
                        "service": "seed-vc",
                        "device": MODEL.device,
                        "model_loaded": MODEL.loaded_at is not None,
                        "loaded_at": MODEL.loaded_at,
                    },
                    np.zeros(0, dtype=np.float32),
                )
                return

            if op != "convert":
                raise ValueError(f"unknown op {op!r}")

            wave, rate, compute = MODEL.convert(
                audio,
                int(header.get("sample_rate", 24000)),
                str(header.get("reference_path", "")),
                int(header.get("diffusion_steps", 10)),
            )
            log.info(
                "converted %.2fs -> %.2fs in %.2fs (%d steps)",
                audio.size / max(int(header.get("sample_rate", 24000)), 1),
                wave.size / max(rate, 1), compute, int(header.get("diffusion_steps", 10)),
            )
            send_message(
                self.request,
                {"sample_rate": rate, "compute_s": round(compute, 4)},
                wave,
            )
        except Exception as exc:
            log.exception("conversion failed")
            # Report the failure to the client rather than dropping the
            # connection: the pipeline can then skip one sentence instead of
            # stalling on a socket timeout.
            # If the client has already gone, there is nobody to tell.
            with contextlib.suppress(OSError):
                send_message(
                    self.request,
                    {"error": f"{type(exc).__name__}: {exc}"},
                    np.zeros(0, dtype=np.float32),
                )


class Server(socketserver.ThreadingTCPServer):
    # NOT allow_reuse_address. On POSIX that flag only permits reuse of a port
    # in TIME_WAIT; on Windows it lets a *second live server* bind the same
    # port, and which one receives a connection is undefined.
    #
    # That is not theoretical. A server left running overnight was still
    # listening on 8765 when a second was started for a measurement; both held
    # ~2.9 GB, the card sat at 7001 MiB of 8188, and conversions ran six times
    # slower than their recorded figures - 7.66 s against 1.29 s. The numbers
    # were nearly written down as a regression in Seed-VC.
    #
    # So: refuse to start rather than start invisibly wrong.
    allow_reuse_address = False
    daemon_threads = True


def port_in_use(host: str, port: int) -> bool:
    """Is something already listening? Cheaper than interpreting a bind error."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="loopback only by default")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-home", default=None,
                        help="model cache; defaults to ./models/huggingface")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    cache = args.hf_home or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "huggingface")
    )
    os.environ.setdefault("HF_HOME", cache)

    # Check before loading, not after: the model costs ~12 s and ~2.9 GB, and
    # finding out afterwards that the port is taken means paying both for
    # nothing - or worse, holding the memory while another server serves.
    if port_in_use(args.host, args.port):
        log.error("something is already listening on %s:%d", args.host, args.port)
        log.error("Another conversion service is running. Two of them hold ~5.8 GB")
        log.error("between them and make every conversion several times slower.")
        log.error("Stop the other one first, or pass --port for a second instance.")
        return 1

    global MODEL
    MODEL = Model(device=args.device)
    try:
        MODEL.load()
    except Exception as exc:
        log.error("could not load Seed-VC: %s", exc)
        log.error("Is this running with the seedvc environment's interpreter?")
        return 1

    with Server((args.host, args.port), Handler) as server:
        log.info("listening on %s:%d", args.host, args.port)
        log.info("stop with Ctrl+C")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("stopping")
    return 0


if __name__ == "__main__":
    sys.exit(main())
