#!/usr/bin/env python
"""Does the microphone still hear the translation when the gate reopens?

The half-duplex gate deafens the microphone while the system speaks, and for a
tail afterwards: ``pipeline.output_latency_ms`` + ``pipeline.half_duplex_tail_ms``.
In a hall the last word keeps sounding - the PA chain, then the room's
reverberation. If it is still speech to the voice detector when the gate
reopens, the system translates the end of its own sentence.

``measure_audio_device.py`` measures when a chirp *arrives*. This measures
when the echo of a real sentence *stops* - the number the tail is for. It plays
an Italian sentence through the speakers with the pipeline's own playback and
gate, records everything the microphone hears, including what the gate would
have thrown away, and runs the pipeline's own voice detector over it. For each
sentence: when the echo last counted as speech, and when the gate reopened.
The difference is the margin.

Run it in the room, the speakers at the volume of the talk, the microphone
where it will be during the talk - and stay silent while it plays. Stop the
pipeline on the page first (the page itself may stay open):

    python scripts/measure_room_echo.py --input "Realtek@Windows WASAPI" --output "Dante@Windows WASAPI"
    python scripts/measure_room_echo.py --input ... --output ... --write-config

``--write-config`` sets ``pipeline.half_duplex_tail_ms`` in config/local.yaml
when the margin is short, and leaves it alone when it is not. The next Start
reads it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.audio.capture import AudioCapture  # noqa: E402
from parliamo.audio.devices import resolve_device  # noqa: E402
from parliamo.audio.gate import HalfDuplexGate  # noqa: E402
from parliamo.audio.playback import AudioPlayback  # noqa: E402
from parliamo.audio.vad import SILERO_BLOCK, SileroVAD  # noqa: E402
from parliamo.config import load_config  # noqa: E402
from parliamo.logging_setup import configure_console  # noqa: E402
from parliamo.paths import ensure_dir, resolve  # noqa: E402

RATE = 16000
FRAME_MS = SILERO_BLOCK / RATE * 1000.0
#: Listened to after each reopening, for echo that outlives the gate.
LISTEN_AFTER_S = 1.5
#: Margin wanted over the worst sentence: rooms vary with how full they are,
#: and the PA operator may turn it up.
SAFETY_MS = 150

SENTENCE = ("Buongiorno a tutti. Questa è una prova del sistema di traduzione: "
            "misuriamo quanto dura l'eco della sala.")


# ---------------------------------------------------------------------------
# analysis - pure, so it is tested without a sound card
# ---------------------------------------------------------------------------


def gate_episodes(closed: list[bool]) -> list[tuple[int, int]]:
    """(first closed frame, first open frame after it) for every closing."""
    out: list[tuple[int, int]] = []
    start = None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            out.append((start, i))
            start = None
    return out


def echo_margins(probs: list[float], closed: list[bool], threshold: float,
                 listen_after: int = int(LISTEN_AFTER_S * 1000 / FRAME_MS),
                 peaks: list[float] | None = None) -> list[dict]:
    """Per sentence: the last frame the detector called speech, against the
    frame the gate reopened at. Positive margin: the echo was over in time.
    *peaks*, when given, adds how loud the speakers reached the microphone."""
    results = []
    episodes = gate_episodes(closed)
    for n, (start, reopen) in enumerate(episodes):
        end = min(len(probs), reopen + listen_after)
        if n + 1 < len(episodes):
            end = min(end, episodes[n + 1][0])
        speech = [i for i in range(start, end) if probs[i] >= threshold]
        heard = bool(speech)
        last = speech[-1] + 1 if heard else start
        after = [i for i in speech if i >= reopen]
        # How long a run of speech frames after reopening was - one stray
        # frame does not open a segment; min_speech_ms of them does.
        run = longest = 0
        for i in range(reopen, end):
            run = run + 1 if probs[i] >= threshold else 0
            longest = max(longest, run)
        loudest = max(peaks[start:end], default=0.0) if peaks is not None else None
        results.append({
            "sentence": n + 1,
            "peak_dbfs": None if loudest is None else round(_dbfs(loudest), 1),
            "heard_as_speech": heard,
            "closed_ms": round((reopen - start) * FRAME_MS),
            "margin_ms": round((reopen - last) * FRAME_MS),
            "speech_after_reopen_ms": round(len(after) * FRAME_MS),
            "longest_run_after_reopen_ms": round(longest * FRAME_MS),
        })
    return results


def _dbfs(peak: float) -> float:
    return 20 * math.log10(max(peak, 1e-6))


def recommend(margins: list[dict], tail_ms: int, safety_ms: int = SAFETY_MS) -> dict:
    """The tail that leaves *safety_ms* over the worst sentence measured."""
    if not margins:
        return {"verdict": "no sentence was measured", "tail_ms": tail_ms, "change": False}
    worst = min(m["margin_ms"] for m in margins)
    if not any(m["heard_as_speech"] for m in margins):
        return {"verdict": "the microphone does not hear the speakers as speech at all - "
                           "the echo cannot be translated", "worst_margin_ms": worst,
                "tail_ms": tail_ms, "change": False}
    if worst >= safety_ms:
        return {"verdict": f"covered: the echo stops {worst} ms before the gate reopens",
                "worst_margin_ms": worst, "tail_ms": tail_ms, "change": False}
    needed = int(math.ceil((tail_ms + safety_ms - worst) / 50.0) * 50)
    verdict = (f"the echo outlives the gate by {-worst} ms" if worst < 0
               else f"covered by only {worst} ms")
    return {"verdict": f"{verdict}; half_duplex_tail_ms {tail_ms} -> {needed}",
            "worst_margin_ms": worst, "tail_ms": needed, "change": True}


def write_pipeline_value(config_path: Path, key: str, value: int, provenance: str) -> dict:
    """Set ``pipeline.<key>`` in a YAML file by editing one line.

    As text, like measure_audio_device.py's write: config/local.yaml carries
    the consent record for the cloned voice in comments, and a YAML round-trip
    would erase them.
    """
    original = ""
    if config_path.exists():
        with open(config_path, encoding="utf-8", newline="") as handle:
            original = handle.read()
    lines = original.splitlines()
    new_line = f"  {key}: {value}  # {provenance}"
    previous = None
    at = next((i for i, line in enumerate(lines) if line.rstrip() == "pipeline:"), None)
    if at is None:
        lines += ["", "pipeline:", new_line]
    else:
        end = at + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
            end += 1
        for i in range(at + 1, end):
            if lines[i].strip().startswith(f"{key}:"):
                previous = lines[i].strip().split(":", 1)[1].split("#")[0].strip()
                lines[i] = new_line
                break
        else:
            lines.insert(at + 1, new_line)
    newline = "\r\n" if "\r\n" in original else "\n"
    updated = newline.join(lines) + newline

    import yaml

    try:
        parsed = yaml.safe_load(updated)
    except Exception as exc:
        return {"written": False, "reason": f"the edit produced invalid YAML ({exc})"}
    if (parsed.get("pipeline") or {}).get(key) != value:
        return {"written": False, "reason": "the edit did not land where expected"}
    with open(config_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    return {"written": True, "path": str(config_path), "previous": previous, "value": value}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


class _Witness:
    """Stands in for the gate at the capture: tells it the gate is open, so
    every block is kept, and remembers what the gate really was."""

    enabled = True

    def __init__(self, gate: HalfDuplexGate) -> None:
        self.gate = gate
        self.states: list[bool] = []

    @property
    def is_closed(self) -> bool:
        self.states.append(self.gate.is_closed)
        return False

    def note_muted_block(self, *a: Any, **k: Any) -> None:  # pragma: no cover
        pass


def sentence_audio(cfg, text: str, wav: str | None) -> tuple[np.ndarray, int]:
    if wav:
        import soundfile as sf

        audio, rate = sf.read(str(resolve(wav)), dtype="float32", always_2d=True)
        return audio.mean(axis=1), int(rate)
    from parliamo.tts import backend_for
    from parliamo.tts import create_backend as make_tts

    target = "it"
    tts = make_tts(backend_for(target, cfg.tts.backend) or cfg.tts.backend,
                   device=cfg.tts.device, language=target,
                   voice=(cfg.tts.voices or {}).get(target) or cfg.tts.voice,
                   speed=cfg.tts.speed)
    speech = tts.speak(text, language=target)
    return speech.audio, speech.sample_rate


def record(cfg, input_spec, output_spec, audio, rate, repeats: int):
    gate = HalfDuplexGate(enabled=True, tail_ms=cfg.pipeline.half_duplex_tail_ms)
    witness = _Witness(gate)
    playback = AudioPlayback(device=output_spec, sample_rate=rate, gate=gate,
                             output_latency_ms=cfg.pipeline.output_latency_ms)
    capture = AudioCapture(device=input_spec, sample_rate=RATE,
                           block_ms=int(FRAME_MS), gate=witness,  # type: ignore[arg-type]
                           max_queue_blocks=100_000)
    chunks: list[np.ndarray] = []
    running = threading.Event()
    running.set()

    def drain() -> None:
        while running.is_set():
            block = capture.read(timeout=0.1)
            if block is not None:
                chunks.append(block)

    out_info = playback.start()
    in_info = capture.start()
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        print(f"  speakers  : {out_info.device.label}")
        print(f"  microphone: {in_info.device.label}")
        print(f"  gate tail : {gate.tail_ms} ms after the last sample "
              f"(output_latency_ms {cfg.pipeline.output_latency_ms} + "
              f"half_duplex_tail_ms {cfg.pipeline.half_duplex_tail_ms})")
        print("  quiet, please - listening to the room for 1 s ...")
        time.sleep(1.0)
        for n in range(repeats):
            print(f"  sentence {n + 1}/{repeats} ...")
            playback.submit(audio, rate)
            playback.wait()
            while gate.is_closed:
                time.sleep(0.01)
            time.sleep(LISTEN_AFTER_S + 0.5)
    finally:
        running.clear()
        reader.join(timeout=2)
        capture.stop()
        playback.stop()
        while (block := capture.read(timeout=0.05)) is not None:
            chunks.append(block)
    return chunks, witness.states[:len(chunks)], gate.tail_ms


def frames_of(chunks: list[np.ndarray], states: list[bool]) -> tuple[np.ndarray, list[bool]]:
    """The recording cut into the detector's frames, each with the gate state
    the pipeline's capture would have seen for it."""
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    per_sample = np.concatenate([np.full(len(c), s, dtype=bool) for c, s in zip(chunks, states, strict=True)]) \
        if chunks else np.zeros(0, dtype=bool)
    n = audio.size // SILERO_BLOCK
    frames = audio[: n * SILERO_BLOCK].reshape(n, SILERO_BLOCK)
    closed = [bool(per_sample[(i + 1) * SILERO_BLOCK - 1]) for i in range(n)]
    return frames, closed


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help='microphone, e.g. "Realtek@Windows WASAPI"')
    parser.add_argument("--output", default=None, help='speakers, e.g. "Dante@Windows WASAPI"')
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--text", default=SENTENCE, help="Italian sentence to play")
    parser.add_argument("--wav", default=None, help="play this file instead (e.g. a cloned voice)")
    parser.add_argument("--write-config", action="store_true",
                        help="raise pipeline.half_duplex_tail_ms in config/local.yaml if needed")
    args = parser.parse_args()

    cfg = load_config()
    input_spec = args.input if args.input is not None else cfg.audio.input_device
    output_spec = args.output if args.output is not None else cfg.audio.output_device
    resolve_device(input_spec, "input")
    resolve_device(output_spec, "output")

    print("\nPreparing the test sentence ...")
    audio, rate = sentence_audio(cfg, args.text, args.wav)
    vad = SileroVAD(device="cpu", threshold=cfg.vad.threshold)
    vad.load()

    print("\nMeasuring the echo:")
    chunks, states, gate_ms = record(cfg, input_spec, output_spec, audio, rate, args.repeats)
    frames, closed = frames_of(chunks, states)
    vad.reset()
    probs = [vad.probability(f) for f in frames]
    floor = float(np.percentile(np.abs(frames[: int(1000 / FRAME_MS)]).max(axis=1), 50)) \
        if len(frames) else 0.0

    peaks = [float(np.abs(f).max()) for f in frames]
    margins = echo_margins(probs, closed, cfg.vad.threshold, peaks=peaks)
    advice = recommend(margins, cfg.pipeline.half_duplex_tail_ms)

    print(f"\n  room noise before the test: peak {floor:.4f}"
          f" ({_dbfs(floor):.0f} dBFS)")
    for m in margins:
        if not m["heard_as_speech"]:
            print(f"  sentence {m['sentence']}: not speech to the microphone "
                  f"(it reached {m['peak_dbfs']} dBFS)")
            continue
        state = "ok" if m["margin_ms"] >= SAFETY_MS else "SHORT" if m["margin_ms"] >= 0 else "LEAK"
        print(f"  sentence {m['sentence']}: echo ended {m['margin_ms']:5d} ms before the gate "
              f"reopened  [{state}], loudest {m['peak_dbfs']} dBFS"
              + (f" - {m['speech_after_reopen_ms']} ms of it after reopening, longest run "
                 f"{m['longest_run_after_reopen_ms']} ms" if m["speech_after_reopen_ms"] else ""))
    print(f"\n  {advice['verdict']}")

    report = {"timestamp": datetime.now().astimezone().isoformat(),
              "input": str(input_spec), "output": str(output_spec),
              "gate_ms": gate_ms, "half_duplex_tail_ms": cfg.pipeline.half_duplex_tail_ms,
              "output_latency_ms": cfg.pipeline.output_latency_ms,
              "vad_threshold": cfg.vad.threshold, "noise_peak": round(floor, 5),
              "sentences": margins, "advice": advice}
    out = ensure_dir("runs/audio") / f"room-echo-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  report: {out}")

    if args.write_config and advice["change"]:
        written = write_pipeline_value(
            resolve("config/local.yaml"), "half_duplex_tail_ms", advice["tail_ms"],
            f"measured {datetime.now():%Y-%m-%d %H:%M} by measure_room_echo.py, "
            f"worst margin {advice['worst_margin_ms']} ms")
        print("  config updated - press Start again to use it" if written["written"]
              else f"  config NOT updated: {written['reason']}")
    elif args.write_config:
        print("  config left as it is")
    return 0


if __name__ == "__main__":
    sys.exit(main())
