#!/usr/bin/env python
"""Live Turkish transcription from the microphone, with the latency breakdown.

    python scripts/live_transcribe.py
    python scripts/live_transcribe.py --input "Microphone Array" --model large-v3-turbo
    python scripts/live_transcribe.py --file recording.wav      # same pipeline, offline

Speak, pause, and the recognised sentence appears with how long each part of the
delay took. Ctrl+C to stop and print a summary.

The offline ``--file`` mode runs the identical VAD and recogniser over a
recording instead of a microphone, which makes a result reproducible: the same
file gives the same segments and the same text every time.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from datetime import datetime

from parliamo.paths import configure_model_cache

configure_model_cache()

import logging  # noqa: E402

from parliamo.asr import create_backend  # noqa: E402
from parliamo.audio.gate import HalfDuplexGate  # noqa: E402
from parliamo.audio.vad import SegmenterConfig, SileroVAD, segment_audio  # noqa: E402
from parliamo.config import load_config  # noqa: E402
from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402
from parliamo.pipeline import LiveTranscriber, TranscriptEvent  # noqa: E402

log = logging.getLogger("live_transcribe")


def print_event(event: TranscriptEvent) -> None:
    d = event.as_dict()
    marker = " [cut]" if d["truncated"] else ""
    print(f"\n[{d['index']:>3}] {event.text}{marker}")
    print(
        f"      {d['audio_s']:.1f}s audio | lag {d['total_lag_s']:.2f}s "
        f"(commit {d['commit_wait_s']:.2f} + queue {d['queue_wait_s']:.2f} + "
        f"asr {d['asr_s']:.2f}) | rtf {d['rtf']:.3f}"
    )
    if d["dropped_segments"]:
        print(f"      {d['dropped_segments']} repetition loop(s) discarded")


def run_live(args: argparse.Namespace, cfg) -> int:
    backend = create_backend(
        "faster_whisper",
        model=args.model or cfg.asr.model,
        device=cfg.asr.device,
        language=cfg.asr.language,
        compute_type=cfg.asr.compute_type,
        beam_size=cfg.asr.beam_size,
        condition_on_previous_text=cfg.asr.condition_on_previous_text,
        temperature=cfg.asr.temperature,
    )
    seg_cfg = SegmenterConfig(
        speech_threshold=cfg.vad.threshold,
        min_speech_ms=cfg.vad.min_speech_ms,
        min_silence_ms=args.min_silence_ms or cfg.vad.min_silence_ms,
        speech_pad_ms=cfg.vad.speech_pad_ms,
        max_segment_ms=cfg.vad.max_segment_ms,
    )
    gate = HalfDuplexGate(
        enabled=cfg.pipeline.half_duplex, tail_ms=cfg.pipeline.half_duplex_tail_ms
    )

    events: list[TranscriptEvent] = []

    def collect(event: TranscriptEvent) -> None:
        events.append(event)
        print_event(event)

    transcriber = LiveTranscriber(
        backend,
        device=args.input if args.input is not None else cfg.audio.input_device,
        vad=SileroVAD(device="cpu", threshold=cfg.vad.threshold),
        segmenter_config=seg_cfg,
        gate=gate,
        on_transcript=collect,
        max_queue_depth=cfg.pipeline.max_queue_depth,
    )

    stopping = threading.Event()

    def handle_sigint(*_: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, handle_sigint)

    print("\nLoading models ...")
    transcriber.start()
    print(f"\nListening. Commit wait is {seg_cfg.min_silence_ms} ms of silence.")
    print("Speak Turkish, pause between sentences. Ctrl+C to stop.\n")

    started = time.monotonic()
    try:
        while not stopping.is_set():
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            time.sleep(0.2)
    finally:
        print("\nstopping ...")
        stats = transcriber.stop()
        backend.unload()

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": backend.model,
        "min_silence_ms": seg_cfg.min_silence_ms,
        "elapsed_s": round(time.monotonic() - started, 1),
        "stats": stats.summary(),
        "events": [e.as_dict() for e in events],
    }
    out = ensure_dir("runs/live") / f"live-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(json.dumps(stats.summary(), indent=2))
    print("=" * 70)
    print(f"report written to {out}")
    return 0


def run_file(args: argparse.Namespace, cfg) -> int:
    """Same VAD and recogniser, run over a recording instead of a microphone."""
    import numpy as np
    import soundfile as sf
    import soxr

    audio, rate = sf.read(args.file, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if rate != 16000:
        audio = np.asarray(soxr.resample(audio, rate, 16000, quality="VHQ"), dtype=np.float32)

    backend = create_backend(
        "faster_whisper",
        model=args.model or cfg.asr.model,
        device=cfg.asr.device,
        language=cfg.asr.language,
        compute_type=cfg.asr.compute_type,
        beam_size=cfg.asr.beam_size,
        condition_on_previous_text=cfg.asr.condition_on_previous_text,
        temperature=cfg.asr.temperature,
    )
    backend.load()
    backend.warmup(2.0)

    vad = SileroVAD(device="cpu", threshold=cfg.vad.threshold)
    seg_cfg = SegmenterConfig(
        speech_threshold=cfg.vad.threshold,
        min_speech_ms=cfg.vad.min_speech_ms,
        min_silence_ms=args.min_silence_ms or cfg.vad.min_silence_ms,
        speech_pad_ms=cfg.vad.speech_pad_ms,
        max_segment_ms=cfg.vad.max_segment_ms,
    )

    print(f"\n{args.file}: {audio.size / 16000:.1f}s")
    print(f"segmenting at min_silence_ms={seg_cfg.min_silence_ms} ...\n")

    total_asr = 0.0
    total_audio = 0.0
    for index, segment in enumerate(segment_audio(audio, vad, seg_cfg), start=1):
        t0 = time.perf_counter()
        transcript = backend.transcribe(segment.audio, segment.sample_rate)
        elapsed = time.perf_counter() - t0
        total_asr += elapsed
        total_audio += segment.duration_s

        marker = " [cut]" if segment.truncated else ""
        print(f"[{index:>3}] {segment.start_s:6.2f}s–{segment.end_s:6.2f}s{marker}")
        print(f"      {transcript.text}")
        print(f"      asr {elapsed * 1000:.0f} ms, rtf {transcript.rtf:.3f}")
        for dropped in transcript.dropped_segments:
            print(f"      discarded repetition: {dropped['reason']}")

    backend.unload()
    if total_audio:
        print(f"\n{total_audio:.1f}s of speech, {total_asr:.1f}s of compute, "
              f"overall rtf {total_asr / total_audio:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=None, help="input device index or name substring")
    parser.add_argument("--model", default=None, help="override asr.model from config")
    parser.add_argument("--file", default=None, help="transcribe a wav instead of the mic")
    parser.add_argument("--min-silence-ms", type=int, default=None,
                        help="override the commit wait - the dominant latency term")
    parser.add_argument("--seconds", type=float, default=None,
                        help="stop automatically after this long")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "WARNING", jsonl=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "faster_whisper", "datasets"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    cfg = load_config()
    if isinstance(args.input, str) and args.input.isdigit():
        args.input = int(args.input)

    return run_file(args, cfg) if args.file else run_live(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
