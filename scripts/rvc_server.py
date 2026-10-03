#!/usr/bin/env python
"""RVC voice conversion, served over the same socket protocol as Seed-VC.

Why a second converter
----------------------
Seed-VC is *zero-shot*: it conditions on a reference recording at every call,
which is what makes the fifteen-second volunteer demonstration possible - and
what costs 0.93 s per sentence, every sentence, forever. Measured on the 68 s
reference recording that is 41% of the whole latency.

RVC is the other trade. The voice is *trained* once - here, ~7 minutes of the
presenter's own recordings, ~40 minutes on this GPU - and then applied at a
fraction of that cost per sentence. Clone once, use always. It cannot clone a
volunteer on stage, so it does not replace Seed-VC; it sits beside it, and the
pipeline picks one by port.

Same protocol as ``voice_conversion_server.py`` deliberately: the pipeline's
``VoiceConverter`` client does not know which is behind the socket. The
``reference_path`` the client sends is ignored - the identity is the trained
model, chosen when this starts - and reported back so a mismatch is visible.

Run from Applio's own interpreter, which has RVC's dependencies::

    C:\\Users\\you\\Applio\\env\\python.exe scripts\\rvc_server.py --model presenter

then point the pipeline at it: ``tts.conversion.port: 8766``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import socket
import socketserver
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("rvc-server")

APPLIO = Path(os.environ.get("APPLIO_DIR", r"C:\Users\you\Applio"))


# -- the wire format, identical to voice_conversion_server.py ----------------
# Copied rather than imported: this runs in Applio's environment, which does
# not have the parliamo package, and a shared module would drag its imports.

def send_message(sock: socket.socket, header: dict[str, Any], audio: np.ndarray) -> None:
    payload = np.ascontiguousarray(audio, dtype="<f4").tobytes()
    header = {**header, "audio_bytes": len(payload)}
    blob = json.dumps(header).encode("utf-8")
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
            raise ConnectionError(f"closed with {remaining} of {n} bytes still expected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> tuple[dict[str, Any], np.ndarray]:
    (header_len,) = struct.unpack(">I", _recv_exactly(sock, 4))
    if header_len > 1 << 20:
        raise ValueError(f"header claims {header_len} bytes; refusing")
    header = json.loads(_recv_exactly(sock, header_len).decode("utf-8"))
    audio_bytes = int(header.get("audio_bytes", 0))
    if audio_bytes > 64 << 20:
        raise ValueError(f"payload claims {audio_bytes} bytes; refusing")
    if audio_bytes == 0:
        return header, np.zeros(0, dtype=np.float32)
    raw = _recv_exactly(sock, audio_bytes)
    return header, np.frombuffer(raw, dtype="<f4").astype(np.float32)


# -- the model ---------------------------------------------------------------

_EPOCH = re.compile(r"_(\d+)e_")


def pick_weights(folder: Path, name: str) -> Path:
    """The checkpoint to serve: the best epoch if training marked one, else the last.

    Applio writes ``<name>_<N>e_<steps>s.pth`` every save interval and, when
    the loss improves, ``..._best_epoch.pth`` beside it. The best one is what
    training meant to keep - past the point of overtraining the last epoch is
    worse, not better. Chosen by the epoch in the name, not by file time:
    two files written in the same second tied on time, and which one loaded
    came down to directory order.
    """
    pths = [p for p in folder.glob(f"{name}_*.pth") if _EPOCH.search(p.name)]
    if not pths:
        raise FileNotFoundError(f"no trained weights under {folder}")

    def epoch(p: Path) -> int:
        return int(_EPOCH.search(p.name).group(1))

    best = [p for p in pths if p.stem.endswith("_best_epoch")]
    return max(best or pths, key=lambda p: (epoch(p), not p.stem.endswith("_best_epoch")))


class Model:
    """One trained RVC voice, loaded once, converting under a lock."""

    def __init__(self, name: str, *, f0_method: str, index_rate: float, protect: float,
                 pitch: int, weights: str | None = None) -> None:
        self.name = name
        self.weights = weights
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.protect = protect
        self.pitch = pitch
        self.loaded_at: float | None = None
        self._lock = threading.Lock()
        self._vc: Any = None
        self.pth = ""
        self.index = ""

    def load(self) -> None:
        sys.path.insert(0, str(APPLIO))
        os.chdir(APPLIO)  # Applio resolves its own assets relative to cwd
        from rvc.infer.infer import VoiceConverter

        folder = APPLIO / "logs" / self.name
        self.pth = str(self.weights or pick_weights(folder, self.name))
        indexes = sorted(folder.glob("*.index"))
        self.index = str(indexes[-1]) if indexes else ""

        t0 = time.perf_counter()
        self._vc = VoiceConverter()
        self._vc.get_vc(self.pth, 0)
        self._vc.load_hubert("contentvec")
        self._vc.last_embedder_model = "contentvec"
        # Warm the pitch extractor and the network so the first sentence on
        # stage is not the slow one.
        self._convert16(np.zeros(16000, dtype=np.float32))
        self.loaded_at = time.time()
        log.info("loaded %s in %.1fs (index: %s)", Path(self.pth).name,
                 time.perf_counter() - t0, Path(self.index).name if self.index else "none")

    def _convert16(self, audio16: np.ndarray, pitch: float | None = None) -> np.ndarray:
        vc = self._vc
        return vc.vc.pipeline(
            model=vc.hubert_model, net_g=vc.net_g, sid=0, audio=audio16,
            pitch=self.pitch if pitch is None else pitch, f0_method=self.f0_method,
            file_index=self.index.replace("trained", "added"), index_rate=self.index_rate,
            pitch_guidance=vc.use_f0, volume_envelope=1.0, version=vc.version,
            protect=self.protect, f0_autotune=False, f0_autotune_strength=1.0,
            proposed_pitch=False, proposed_pitch_threshold=155.0,
        )

    def convert(self, audio: np.ndarray, sample_rate: int,
                pitch: float | None = None) -> tuple[np.ndarray, int, float]:
        import librosa

        t0 = time.perf_counter()
        wave = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=16000,
                                    res_type="soxr_hq").astype(np.float32)
        peak = float(np.abs(wave).max()) / 0.95 if wave.size else 0.0
        if peak > 1.0:
            wave = wave / peak
        with self._lock:
            out = self._convert16(wave, pitch)
        # Applio's pipeline returns float audio already scaled to a 0.99 peak
        # (rvc/infer/pipeline.py, end of pipeline()); nothing to rescale.
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return out, int(self._vc.tgt_sr), time.perf_counter() - t0


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
                send_message(self.request, {
                    "service": "rvc", "model": MODEL.name, "weights": Path(MODEL.pth).name,
                    "model_loaded": MODEL.loaded_at is not None, "loaded_at": MODEL.loaded_at,
                }, np.zeros(0, dtype=np.float32))
                return
            if op != "convert":
                raise ValueError(f"unknown op {op!r}")
            rate_in = int(header.get("sample_rate", 24000))
            # The client may name the shift for this sentence: its voice
            # depends on the language, and one --pitch cannot fit Kokoro's
            # Italian woman (223 Hz) and Piper's German man (125 Hz) at once.
            pitch = header.get("pitch")
            pitch = None if pitch is None else float(pitch)
            wave, rate, compute = MODEL.convert(audio, rate_in, pitch)
            log.info("converted %.2fs -> %.2fs in %.2fs (rvc %s, pitch %+.1f)",
                     audio.size / max(rate_in, 1), wave.size / max(rate, 1), compute, MODEL.name,
                     MODEL.pitch if pitch is None else pitch)
            send_message(self.request, {
                "sample_rate": rate, "compute_s": round(compute, 4),
                # The client sent a reference path meant for Seed-VC; say
                # plainly whose voice this actually is.
                "voice": MODEL.name, "ignored_reference": header.get("reference_path", ""),
            }, wave)
        except Exception as exc:
            log.exception("conversion failed")
            with contextlib.suppress(OSError):
                send_message(self.request, {"error": f"{type(exc).__name__}: {exc}"},
                             np.zeros(0, dtype=np.float32))


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False  # two servers on one port cost hours once


def port_in_use(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main() -> int:
    global MODEL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="name under Applio/logs/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "fcpe", "crepe", "crepe-tiny"])
    parser.add_argument("--index-rate", type=float, default=0.5)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--pitch", type=int, default=0,
                        help="semitones. RVC keeps the pitch it is given, and the generic "
                             "voice is not the presenter's: -8 from if_sara (135 Hz), +9 from "
                             "im_nicola (159 Hz) put it at the presenter's 139 Hz")
    parser.add_argument("--weights", default=None,
                        help="a specific .pth; default: the best epoch, else the last")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    if port_in_use(args.host, args.port):
        log.error("port %d is already in use - another server is running there", args.port)
        return 2

    MODEL = Model(args.model, f0_method=args.f0_method, index_rate=args.index_rate,
                  protect=args.protect, pitch=args.pitch, weights=args.weights)
    MODEL.load()
    with Server((args.host, args.port), Handler) as server:
        log.info("listening on %s:%d - stop with Ctrl+C", args.host, args.port)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
