#!/usr/bin/env python
"""Could echo cancellation replace the half-duplex gate in this room?

Why ask
-------
The gate closes the microphone while the translation plays, so the system never
hears itself. It also means the presenter cannot speak over the translation:
the streaming modes lose words to it, and the gate card counts them. WebRTC's
echo canceller (AEC3) does the opposite - it is told exactly what the speakers
play and subtracts it from the microphone. It ships in the ``livekit`` package
(``AudioProcessingModule``), runs locally, and needs no LiveKit server.

Whether it works is a property of the room, the speakers and the microphone,
so it is measured, never assumed. This plays Italian synthesis through the
speakers, records the microphone on its own stream - as the pipeline does -
and runs AEC3 over the recording offline, with the reference placed on the
same clock the callbacks would see.

What it reports
---------------
1. **Echo over the room floor**, before anything is cancelled. Under ~20 dB
   the test proves little: turn the volume up to what the room will hear.
2. **Suppression** (ERLE) after three seconds of convergence, and where the
   residual sits against the room floor.
3. **What the pipeline would hear** in the echo, before and after: Silero VAD
   exactly as live, then Whisper on each segment it passes. After AEC this
   must be nothing, or the gate stays.
4. **Talking over it**: the presenter's recorded reading added on top at three
   loudness ratios, recognised with and without AEC, scored against the same
   reading recognised alone.

Measured on the reference laptop, 2026-09-23, at low volume
------------------------------------------------------------
In shared mode the Intel microphone array's own driver removes the speakers -
and then the room - to digital silence (-120 dBFS) within seven seconds, and
does not recover when playback stops. That is driver processing, not the
acoustics, so the microphone is opened in WASAPI **exclusive** mode, which
bypasses it: the echo is then present throughout. At that (low) volume the echo
sat 10 dB over the floor and AEC3 with a delay hint of 232 ms brought it to the
floor. The case that decides it - performance volume, where laptop speakers
distort and cancellation is hardest - has not been measured yet.

Run it in the room, at performance volume, with the microphone where it will
stand::

    python scripts/measure_echo_cancellation.py
    python scripts/measure_echo_cancellation.py --shared      # keep driver effects
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from parliamo.logging_setup import configure_console
from parliamo.paths import configure_model_cache, ensure_dir, resolve

configure_model_cache()

RATE = 48000
FRAME = RATE // 100          # AEC3 takes exactly 10 ms
SETTLE_S = 3.0               # convergence allowed before suppression is judged


def db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.asarray(x, np.float64) ** 2) + 1e-12))


def italian_speech(paths: list[Path]) -> np.ndarray:
    import soundfile as sf
    import soxr

    parts = []
    for path in paths:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        parts += [soxr.resample(audio, rate, RATE).astype(np.float32),
                  np.zeros(int(0.4 * RATE), np.float32)]
    speech = np.concatenate(parts)
    return (speech / (np.abs(speech).max() + 1e-9) * 0.7).astype(np.float32)


def record(reference: np.ndarray, exclusive: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Play *reference*, record the microphone; both placed on PortAudio's clock."""
    import sounddevice as sd

    from parliamo.audio.devices import ensure_com, resolve_device

    ensure_com()
    mic_dev, spk_dev = resolve_device(None, "input"), resolve_device(None, "output")
    mic_blocks: list[tuple[float, np.ndarray]] = []
    out_blocks: list[tuple[float, np.ndarray]] = []
    position = [0]
    finished = threading.Event()

    def on_input(indata, frames, t, status) -> None:
        mic_blocks.append((t.inputBufferAdcTime, indata.mean(axis=1).copy()))

    def on_output(outdata, frames, t, status) -> None:
        start = position[0]
        chunk = reference[start:start + frames]
        outdata[:len(chunk), 0] = chunk
        outdata[len(chunk):, 0] = 0
        out_blocks.append((t.outputBufferDacTime, outdata[:, 0].copy()))
        position[0] += frames
        if start >= reference.size + RATE:
            finished.set()

    extra = sd.WasapiSettings(exclusive=True) if exclusive else None
    channels = min(2, int(sd.query_devices(mic_dev.index)["max_input_channels"]))
    with sd.InputStream(device=mic_dev.index, samplerate=RATE, channels=channels,
                        dtype="float32", blocksize=FRAME, callback=on_input,
                        extra_settings=extra):
        time.sleep(2.0)                                   # the room, alone
        with sd.OutputStream(device=spk_dev.index, samplerate=RATE, channels=1,
                             dtype="float32", blocksize=FRAME, callback=on_output):
            finished.wait(timeout=reference.size / RATE + 10)
        time.sleep(0.5)

    t0 = min(mic_blocks[0][0], out_blocks[0][0])
    length = int((max(mic_blocks[-1][0], out_blocks[-1][0]) - t0 + 1.0) * RATE)
    mic, played = np.zeros(length, np.float32), np.zeros(length, np.float32)
    for blocks, target in ((mic_blocks, mic), (out_blocks, played)):
        for t, block in blocks:
            i = int(round((t - t0) * RATE))
            target[i:i + block.size] = block[:max(0, length - i)]
    n = length // FRAME * FRAME
    devices = {"microphone": mic_dev.name, "speakers": spk_dev.name,
               "mode": "exclusive" if exclusive else "shared"}
    return mic[:n], played[:n], devices


def echo_delay_ms(mic: np.ndarray, played: np.ndarray) -> float:
    size = 1 << int(np.ceil(np.log2(2 * mic.size)))
    xc = np.fft.irfft(np.fft.rfft(mic, size) * np.conj(np.fft.rfft(played, size)), size)
    return float(np.argmax(np.abs(xc[: int(1.5 * RATE)])) / RATE * 1000)


def cancel(mic: np.ndarray, played: np.ndarray, delay_ms: int) -> np.ndarray:
    from livekit import rtc

    apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)

    def pcm(x: np.ndarray) -> bytes:
        return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()

    out = np.empty_like(mic)
    for i in range(0, mic.size, FRAME):
        apm.process_reverse_stream(rtc.AudioFrame(pcm(played[i:i + FRAME]), RATE, 1, FRAME))
        apm.set_stream_delay_ms(delay_ms)
        frame = rtc.AudioFrame(pcm(mic[i:i + FRAME]), RATE, 1, FRAME)
        apm.process_stream(frame)
        out[i:i + FRAME] = np.frombuffer(bytes(frame.data), np.int16).astype(np.float32) / 32767
    return out


class Listener:
    """The pipeline's ears: Silero VAD as configured, then Whisper per segment."""

    def __init__(self) -> None:
        from live_translate import load_hotwords  # type: ignore[import-not-found]
        from parliamo.asr import create_backend
        from parliamo.audio.vad import SegmenterConfig, SileroVAD
        from parliamo.config import load_config

        cfg = load_config()
        self.asr = create_backend(
            "faster_whisper", model=cfg.asr.model, device=cfg.asr.device,
            language=cfg.asr.language, compute_type=cfg.asr.compute_type,
            beam_size=cfg.asr.beam_size, hotwords=load_hotwords(cfg.asr.hotwords))
        self.vad = SileroVAD(device="cpu", threshold=cfg.vad.threshold)
        self.seg_cfg = SegmenterConfig(
            speech_threshold=cfg.vad.threshold,
            min_speech_ms=cfg.vad.min_speech_ms, min_silence_ms=cfg.vad.min_silence_ms,
            speech_pad_ms=cfg.vad.speech_pad_ms, max_segment_ms=cfg.vad.max_segment_ms,
            partial_interval_ms=0)

    def hear(self, audio48: np.ndarray) -> list[str]:
        import soxr

        from parliamo.audio.vad import segment_audio

        audio = soxr.resample(audio48, RATE, 16000).astype(np.float32)
        texts = []
        for segment in segment_audio(audio, self.vad, self.seg_cfg):
            if segment.partial:
                continue
            text = self.asr.transcribe(segment.audio, segment.sample_rate).text.strip()
            if text:
                texts.append(text)
        return texts


def main() -> int:
    configure_console()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shared", action="store_true",
                        help="open the microphone in shared mode, driver effects included")
    parser.add_argument("--speech", nargs="*", default=None,
                        help="WAVs of Italian speech to play (default: runs/replay/audio/*.wav)")
    parser.add_argument("--talker", default="data/insitu/continuous/reading-48k.wav",
                        help="the presenter's recorded speech, for talking over the echo")
    parser.add_argument("--from", dest="from_dir", default=None,
                        help="analyse an earlier recording (mic.wav, played.wav) instead of "
                             "playing; every run saves one")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    import jiwer
    import soundfile as sf

    from parliamo.eval.normalize import get_normalizer

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    if args.from_dir:
        folder = Path(args.from_dir)
        mic, _ = sf.read(str(folder / "mic.wav"), dtype="float32")
        played, _ = sf.read(str(folder / "played.wav"), dtype="float32")
        n = min(mic.size, played.size) // FRAME * FRAME
        mic, played = mic[:n], played[:n]
        devices = json.loads((folder / "devices.json").read_text(encoding="utf-8"))
    else:
        speech_paths = [Path(p) for p in args.speech] if args.speech else sorted(
            resolve("runs/replay/audio").glob("*.wav"))[:5]
        if not speech_paths:
            print("  no Italian speech to play: pass --speech file.wav")
            return 2
        reference = italian_speech(speech_paths)
        print(f"\nPlaying {reference.size / RATE:.1f} s of Italian synthesis. "
              "Set the volume to what the room will hear.\n")
        mic, played, devices = record(reference, exclusive=not args.shared)
        folder = ensure_dir(f"runs/audio/echo-{stamp}")
        sf.write(str(folder / "mic.wav"), mic, RATE, subtype="FLOAT")
        sf.write(str(folder / "played.wav"), played, RATE, subtype="FLOAT")
        (folder / "devices.json").write_text(json.dumps(devices, ensure_ascii=False), encoding="utf-8")
    active = np.flatnonzero(np.abs(played) > 1e-4)
    p0, p1 = int(active[0]), int(active[-1])
    floor = db(mic[int(0.3 * RATE): max(int(0.4 * RATE), p0 - int(0.2 * RATE))])
    echo = db(mic[p0:p1])
    delay = echo_delay_ms(mic, played)
    print(f"  {devices['microphone']} ({devices['mode']}) <- {devices['speakers']}")
    print(f"  room floor {floor:.1f} dBFS, echo {echo:.1f} dBFS: {echo - floor:+.1f} dB over the floor, "
          f"arriving {delay:.0f} ms after the reference")
    if echo - floor < 20:
        print("  NOTE the echo is weak; at this volume the test proves little. Turn it up.")

    cleaned = cancel(mic, played, int(round(delay)))
    settle = p0 + int(SETTLE_S * RATE)
    erle = db(mic[settle:p1]) - db(cleaned[settle:p1])
    residual = db(cleaned[settle:p1])
    print(f"\n  suppression after {SETTLE_S:.0f} s: {erle:.1f} dB, residual {residual:.1f} dBFS "
          f"({residual - floor:+.1f} dB against the floor)")

    listener = Listener()
    heard_before = listener.hear(mic[p0:p1])
    heard_after = listener.hear(cleaned[p0:p1])
    print(f"\n  the pipeline would hear in the echo, without AEC: {len(heard_before)} segment(s)")
    for text in heard_before[:4]:
        print(f"      {text[:100]}")
    print(f"  ... and with AEC: {len(heard_after)} segment(s)")
    for text in heard_after[:4]:
        print(f"      {text[:100]}")

    talk, rate = sf.read(str(resolve(args.talker)), dtype="float32", always_2d=False)
    if rate != RATE:
        import soxr
        talk = soxr.resample(talk, rate, RATE).astype(np.float32)
    talk = talk[int(12 * RATE): int(12 * RATE) + (p1 - p0)]
    norm = get_normalizer("tr")
    voiced = talk[np.abs(talk) > 1e-4]
    alone = " ".join(listener.hear(talk))
    double_talk = []
    natural = db(voiced) - echo
    print(f"\n  talking over the translation (WER against the same speech recognised alone;"
          f" as recorded, the presenter sits {natural:+.0f} dB against this echo):")
    # As recorded first - the presenter's own level into this microphone -
    # then fixed ratios, because the stage microphone will not be this one.
    for ratio in (natural, -6.0, 0.0, 6.0):
        gain = 10 ** ((echo + ratio - db(voiced)) / 20)
        near = np.zeros_like(mic)
        near[p0:p0 + talk.size] = talk * gain
        both = np.clip(mic + near, -1, 1)
        without = " ".join(listener.hear(both[p0:p1]))
        with_aec = " ".join(listener.hear(cancel(both, played, int(round(delay)))[p0:p1]))
        row = {"talker_over_echo_db": round(ratio, 1), "as_recorded": ratio == natural,
               "wer_without_aec": round(100 * jiwer.wer(norm(alone), norm(without or "-")), 1),
               "wer_with_aec": round(100 * jiwer.wer(norm(alone), norm(with_aec or "-")), 1)}
        double_talk.append(row)
        label = "as recorded" if ratio == natural else f"{ratio:+.0f} dB"
        print(f"    presenter {label:>11}: without AEC {row['wer_without_aec']:5.1f}%"
              f"   with AEC {row['wer_with_aec']:5.1f}%")

    verdict = ("AEC removes what the pipeline would hear" if not heard_after
               else "AEC leaves something the pipeline would transcribe - keep the gate")
    print(f"\n  {verdict}")
    report: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(), **devices,
        "floor_dbfs": round(floor, 1), "echo_dbfs": round(echo, 1),
        "echo_delay_ms": round(delay), "erle_db": round(erle, 1),
        "residual_dbfs": round(residual, 1), "heard_without_aec": heard_before,
        "heard_with_aec": heard_after, "double_talk": double_talk, "verdict": verdict,
    }
    out = Path(folder) / "report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  report written to {out}")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
