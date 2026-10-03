#!/usr/bin/env python
"""Speak Turkish, hear Italian. The whole pipeline, on one machine, offline.

    python scripts/live_translate.py
    python scripts/live_translate.py --no-speak            # subtitles only
    python scripts/live_translate.py --voice data/insitu/.../003.wav \\
        --consent "signed 2026-08-31, ref 001"
    python scripts/live_translate.py --file recording.wav --save-audio runs/replay

``--file`` pushes a recording through the identical pipeline. It is how the
whole chain gets exercised - and listened to - without a microphone, a room or
a person, and unlike a live run it gives the same answer twice.

Nothing here contacts the network. Model weights are read from ``models/``,
which ``scripts/download_models.py`` fills once.

Voice cloning needs the conversion service running in the other environment::

    <seedvc-env>/python.exe scripts/voice_conversion_server.py

Without it the pipeline still works and speaks in Kokoro's generic voice - the
translation is the point, the voice is a layer on top.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.asr import create_backend as make_asr  # noqa: E402
from parliamo.audio.gate import HalfDuplexGate  # noqa: E402
from parliamo.audio.playback import AudioPlayback  # noqa: E402
from parliamo.audio.vad import SegmenterConfig, SileroVAD, segment_audio  # noqa: E402
from parliamo.config import load_config  # noqa: E402
from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.mt import create_backend as make_mt  # noqa: E402
from parliamo.paths import ensure_dir, resolve  # noqa: E402
from parliamo.pipeline import DeliveryEvent, LiveTranscriber, LiveTranslator  # noqa: E402
from parliamo.pipeline.transcriber import TranscriptEvent  # noqa: E402
from parliamo.tts import VoiceProfile, backend_for  # noqa: E402
from parliamo.tts import create_backend as make_tts  # noqa: E402
from parliamo.tts.conversion import KNOWN_PORTS, VoiceConverter, find_voice_service  # noqa: E402
from parliamo.ui import SubtitleServer, UIState  # noqa: E402

log = logging.getLogger("live_translate")


def load_hotwords(path: str | None, language: str | None = None) -> str | None:
    """The hotword list for *language*, or None.

    The configured file is Turkish (``hotwords.tr.txt``). Speaking another
    language, it would have pulled Italian or German recognition toward
    Turkish words, so a list written for one language is only used for it: a
    ``.tr.`` file becomes ``.<language>.`` if that exists, and nothing if not.
    """
    if not path:
        return None
    file = resolve(path)
    if language:
        tagged = file.name.split(".")
        if len(tagged) >= 3 and len(tagged[-2]) in (2, 3) and tagged[-2] != language:
            tagged[-2] = language
            file = file.with_name(".".join(tagged))
    if not file.exists():
        return None
    words = [
        line.strip()
        for line in file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return " ".join(words) if words else None


def print_delivery(event: DeliveryEvent) -> None:
    d = event.as_dict()
    if event.partial:
        # Marked so nobody mistakes a guess for the sentence. On a real
        # subtitle surface this would overwrite the previous partial in place;
        # in a terminal the honest thing is to label it.
        print(f"\n[{d['index']:>3}]. {event.source_text}   ...")
        print(f"       ~ {event.translated_text}")
        return
    print(f"\n[{d['index']:>3}]  {event.source_text}")
    print(f"       -> {event.translated_text or '(no translation)'}")
    parts = (
        f"lag {d['total_lag_s']:.2f}s  "
        f"(asr {d['recognition_lag_s']:.2f} + mt {d['translation_s']:.2f} "
        f"+ tts {d['synthesis_s']:.2f} + vc {d['conversion_s']:.2f})"
    )
    marker = f"  [{d['voice']}]" if d["spoken"] else "  [not spoken]"
    print(f"       {parts}{marker}")
    if d["error"]:
        print(f"       ! {d['error']}")


def warn_about_the_gate(cfg, speaking: bool) -> None:
    """The one misconfiguration that reliably destroys a live demonstration.

    Speakers feed back into the microphone. The gate mutes capture while audio
    plays, but it is sized from `output_latency_ms`, and PortAudio's reported
    latency excludes the vendor DSP - measured at ~430 ms on the reference
    laptop against 3 ms reported. Left at zero, the microphone reopens while
    the previous sentence is still audible in the room.
    """
    if not speaking:
        return
    if cfg.pipeline.output_latency_ms > 0:
        return
    print("\n" + "!" * 72)
    print("  pipeline.output_latency_ms is 0 and audio output is enabled.")
    print()
    print("  The half-duplex gate will be sized from PortAudio's reported")
    print("  latency, which does not include the vendor DSP chain. On the")
    print("  reference laptop that was 3 ms reported against ~430 ms real.")
    print("  The microphone can reopen while the speakers are still playing,")
    print("  and the system will start translating its own output.")
    print()
    print("  Measure it, then set it in config/local.yaml:")
    print("    python scripts/measure_audio_device.py --label venue")
    print("!" * 72)


def replay_file(args, cfg, translator) -> None:
    """Run a recording through the whole pipeline, without a microphone.

    Why this exists
    ---------------
    Every stage has been measured on its own, but a rehearsal needs a person, a
    room and a microphone. This mode takes a wav - the in-situ recordings under
    ``data/insitu`` are exactly the right material - and pushes it through the
    identical VAD, recogniser, translator, synthesiser and converter, calling
    the same :meth:`LiveTranslator.deliver` the live path calls.

    So the numbers it reports are the real pipeline's numbers, minus the
    microphone and the commit wait, which is charged as a policy constant
    rather than actually waited out. And unlike a live run it is reproducible:
    the same file gives the same segments, the same text and the same audio.
    """
    import numpy as np
    import soundfile as sf
    import soxr

    audio, rate = sf.read(resolve(args.file), dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if rate != 16000:
        audio = np.asarray(soxr.resample(audio, rate, 16000, quality="VHQ"), dtype=np.float32)

    seg_cfg = SegmenterConfig(
        speech_threshold=cfg.vad.threshold,
        min_speech_ms=cfg.vad.min_speech_ms,
        min_silence_ms=args.min_silence_ms or cfg.vad.min_silence_ms,
        speech_pad_ms=cfg.vad.speech_pad_ms,
        max_segment_ms=cfg.vad.max_segment_ms,
        partial_interval_ms=(
            cfg.pipeline.partial_interval_ms
            if cfg.pipeline.emit_partial_transcripts and not args.no_partials
            else 0
        ),
    )
    vad = SileroVAD(device="cpu", threshold=cfg.vad.threshold)
    asr = translator.transcriber.backend

    print(f"\nreplaying {args.file}  ({audio.size / 16000:.1f}s)")
    print("warming every stage first - an unwarmed first sentence is not a measurement")
    translator.warm()
    asr.warmup(seconds=3.0)
    if translator.playback is not None:
        translator.playback.start()

    save_dir = ensure_dir(args.save_audio) if args.save_audio else None

    index = 0
    for segment in segment_audio(audio, vad, seg_cfg):
        if not segment.partial:
            index += 1
        t0 = time.monotonic()
        transcript = asr.transcribe(segment.audio, segment.sample_rate)
        asr_s = time.monotonic() - t0
        text = transcript.text.strip()
        if not text:
            continue

        event = TranscriptEvent(
            # A partial previews the sentence that is *coming*, so it carries
            # the number that sentence will get - not the one before it.
            index=index + 1 if segment.partial else index,
            text=text,
            segment=segment,
            transcript=transcript,
            partial=segment.partial,
            # There is no real wait here - the whole file is already on disk -
            # so the policy cost is charged at its configured value rather than
            # measured, and the total stays comparable with a live run.
            commit_wait_s=seg_cfg.min_silence_ms / 1000.0,
            queue_wait_s=0.0,
            asr_s=asr_s,
        )
        # `deliver` invokes translator.on_delivery itself, which is where the
        # printing and collecting happen - doing it again here would double
        # every line and every row in the report.
        delivery = translator.deliver(event)
        if delivery is None:
            continue

        if save_dir is not None and delivery.audio is not None and delivery.audio.size:
            sf.write(
                save_dir / f"{index:03d}.wav",
                delivery.audio,
                delivery.sample_rate,
                subtype="FLOAT",
            )

    if translator.playback is not None:
        translator.playback.wait()
    if save_dir is not None:
        print(f"\naudio written to {save_dir}")


def build(args, cfg):
    """Assemble the pipeline. Returns (translator, teardown)."""
    asr = make_asr(
        "faster_whisper",
        model=args.asr_model or cfg.asr.model,
        device=cfg.asr.device,
        language=cfg.asr.language,
        compute_type=cfg.asr.compute_type,
        beam_size=cfg.asr.beam_size,
        condition_on_previous_text=cfg.asr.condition_on_previous_text,
        temperature=cfg.asr.temperature,
        hotwords=load_hotwords(cfg.asr.hotwords, cfg.asr.language),
    )
    mt = make_mt(
        cfg.mt.backend,
        model=args.mt_model or cfg.mt.model_path,
        device=args.mt_device or cfg.mt.device,
        source_lang=cfg.mt.source_lang,
        target_lang=args.target or cfg.tts.language,
        compute_type=cfg.mt.compute_type,
        beam_size=cfg.mt.beam_size,
        max_decoding_length=cfg.mt.max_decoding_length,
        split_sentences=cfg.mt.split_sentences,
    )
    target = args.target or cfg.tts.language
    # Kokoro where it has the language, Piper (CPU) for German and Turkish.
    # The voice and speed in the config reach the synthesiser: they did not,
    # which left `tts.voice` inert whatever it said.
    tts_backend = backend_for(target, cfg.tts.backend) or cfg.tts.backend
    tts = make_tts(
        tts_backend,
        device=cfg.tts.device,
        language=target,
        voice=(cfg.tts.voices or {}).get(target) or cfg.tts.voice,
        speed=cfg.tts.speed,
    )

    speaking = not args.no_speak
    gate = HalfDuplexGate(
        enabled=cfg.pipeline.half_duplex, tail_ms=cfg.pipeline.half_duplex_tail_ms
    )
    # Replaying a file still synthesises - that is the stage being exercised -
    # but it does not put sound in the room unless asked. A measurement run
    # should not surprise anyone, and --save-audio is the usual way to listen.
    audible = speaking and (not args.file or args.play)
    playback = (
        AudioPlayback(
            device=args.output if args.output is not None else cfg.audio.output_device,
            sample_rate=tts.sample_rate,
            gate=gate,
            output_latency_ms=cfg.pipeline.output_latency_ms,
        )
        if audible
        else None
    )

    converter = None
    reference = None if args.no_clone else (args.voice or cfg.tts.conversion.reference_voice)
    if reference and speaking:
        consent = args.consent or cfg.tts.conversion.consent
        # Registering the profile is what enforces consent; it raises when the
        # record is empty. Do it before anything is cloned, not after.
        VoiceProfile(
            name="target", reference_path=str(resolve(reference)), consent=consent
        ).validate()
        # Whichever voice service is running: the configured port first, then
        # the other known one. The talk swaps RVC for Seed-VC for the volunteer,
        # and a pipeline pinned to one port spoke the volunteer generic.
        preferred = args.converter_port or cfg.tts.conversion.port
        port = find_voice_service(cfg.tts.conversion.host, preferred)
        if port is None:
            tried = ", ".join(str(p) for p in dict.fromkeys((preferred, *KNOWN_PORTS)))
            print(f"\n  no voice service running (tried {tried})")
            print("  continuing in the generic voice; start RVC or Seed-VC first.\n")
        else:
            if port != preferred:
                log.warning("no voice service on %d; using the one running on %d", preferred, port)
            converter = VoiceConverter(
                host=cfg.tts.conversion.host,
                port=port,
                diffusion_steps=args.steps or cfg.tts.conversion.diffusion_steps,
                timeout=cfg.tts.conversion.timeout_s,
            )

    transcriber = LiveTranscriber(
        asr,
        device=args.input if args.input is not None else cfg.audio.input_device,
        vad=SileroVAD(device="cpu", threshold=cfg.vad.threshold),
        segmenter_config=SegmenterConfig(
            speech_threshold=cfg.vad.threshold,
            min_speech_ms=cfg.vad.min_speech_ms,
            min_silence_ms=args.min_silence_ms or cfg.vad.min_silence_ms,
            speech_pad_ms=cfg.vad.speech_pad_ms,
            max_segment_ms=cfg.vad.max_segment_ms,
            partial_interval_ms=(
                cfg.pipeline.partial_interval_ms
                if cfg.pipeline.emit_partial_transcripts and not args.no_partials
                else 0
            ),
        ),
        gate=gate,
        max_queue_depth=cfg.pipeline.max_queue_depth,
    )

    translator = LiveTranslator(
        transcriber,
        mt,
        tts,
        playback=playback,
        converter=converter,
        reference_voice=str(resolve(reference)) if (reference and converter) else None,
        target_lang=args.target or cfg.tts.language,
        speak=speaking,
    )
    # Speak sentences as they end rather than at the pause. Config sets the
    # default; --stream forces it for a replay; the page toggles it live.
    mode = args.stream or ("sentence" if cfg.pipeline.stream_sentences else "off")
    translator.stream_mode = mode
    translator.stream_sentences = mode == "sentence"
    # Each sentence carries its own pitch shift to the presenter's voice,
    # measured from whichever synthesiser voice this language uses.
    translator.presenter_f0_hz = cfg.tts.conversion.presenter_f0_hz
    # 0 from the page or the flag means "the reference's own pace", which
    # must win over a configured rate rather than fall through to it.
    chosen = getattr(args, "voice_speed", None)
    translator.voice_speed = (chosen if chosen is not None else cfg.tts.conversion.speed) or None

    def teardown() -> None:
        for component in (mt, tts, asr):
            # Shutdown is best-effort; a failure here must not mask whatever
            # the run was actually about.
            with contextlib.suppress(Exception):
                component.unload()

    return translator, teardown, speaking


def run_app(args, cfg, speaking: bool, run_dir) -> int:
    """Serve the application and let it start the pipeline.

    The inverse of the ordinary path: nothing is loaded until someone presses
    start, so the operator can open the window, read the pre-flight list and
    pick a voice before the microphone opens.
    """
    from parliamo.ui.controller import PipelineController
    from parliamo.ui.server import running_instance

    # One application per machine. A second copy used to step to the next
    # port and carry on - and then both wanted the microphone and the GPU, and
    # the operator kept looking at the first tab. Seen on 2026-09-20: "I ran it
    # more than once, everything was empty." If a parlIAmo answers on the
    # port, say where it is and stop.
    other = running_instance("127.0.0.1", args.ui_port)
    if other:
        print(f"\n  parlIAmo is already running at {other}")
        print("  Open that address, or close the other terminal (Ctrl+C) and run this again.")
        print("  Two copies would fight over the microphone and the GPU.\n")
        return 3

    state = UIState(
        source_lang=cfg.mt.source_lang,
        target_lang=args.target or cfg.tts.language,
        speaking=speaking,
        gate_warning=(
            "pipeline.output_latency_ms is 0 - measure it before using speakers, "
            "or the microphone will hear them and translate the system's own output"
            if speaking and cfg.pipeline.output_latency_ms <= 0 else ""
        ),
    )

    deliveries: list[DeliveryEvent] = []
    ui_ref: dict = {}

    def factory():
        # The gate's numbers are measured in the room while the page is open
        # (measure_room_echo.py --write-config), so they are read at each
        # Start rather than once when the program started.
        with contextlib.suppress(Exception):
            fresh = load_config().pipeline
            cfg.pipeline.output_latency_ms = fresh.output_latency_ms
            cfg.pipeline.half_duplex_tail_ms = fresh.half_duplex_tail_ms

        # Whatever the page selected wins over the config, so a device picked
        # before start is the device that opens.
        if controller.input_device is not None:
            args.input = controller.input_device
        if controller.output_device is not None:
            args.output = controller.output_device

        # Languages chosen on the page apply here, at start, where every
        # backend is built for the pair. The source goes into both the
        # recogniser and the translator - they must agree or the translator is
        # handed the wrong language with no error.
        if controller.source_lang:
            cfg.mt.source_lang = controller.source_lang
            cfg.asr.language = controller.source_lang
            state.source_lang = controller.source_lang
        if controller.target_lang:
            args.target = controller.target_lang
            state.target_lang = controller.target_lang
        # Models chosen on the page. The CLI flags already exist; the page
        # just sets them before build() reads them.
        if controller.asr_model:
            args.asr_model = controller.asr_model
        if controller.mt_model:
            args.mt_model = controller.mt_model
        if controller.stream_mode is not None:
            args.stream = controller.stream_mode
        if controller.min_silence_ms is not None:
            args.min_silence_ms = controller.min_silence_ms
        if controller.voice_speed is not None:
            args.voice_speed = controller.voice_speed

        # A target nothing can voice is subtitles only, automatically.
        # Refusing to start would be the other honest answer, and it is the
        # worse one thirty seconds before a talk; the page says which.
        target = args.target or cfg.tts.language
        if backend_for(target, cfg.tts.backend) is None:
            log.warning("no voice for %s; running subtitles only", target)
            args.no_speak = True
            state.speaking = False
        else:
            args.no_speak = not speaking
            state.speaking = speaking

        translator, teardown, _ = build(args, cfg)
        # Segments the recogniser made nothing of are kept beside the run
        # log, so "a sentence went missing" can be settled by listening.
        translator.transcriber.keep_dropped_dir = run_dir / "dropped"

        def collect(event: DeliveryEvent) -> None:
            deliveries.append(event)
            print_delivery(event)
            ui = ui_ref.get("ui")
            if ui is not None:
                # A display fault must not cost a sentence.
                with contextlib.suppress(Exception):
                    ui.publish_delivery(event)

        translator.on_delivery = collect
        # A voice chosen before start was pressed is applied now, rather than
        # silently ignored.
        pending = getattr(controller, "pending_voice", None)
        if pending:
            with contextlib.suppress(Exception):
                translator.set_reference_voice(*pending)
        return translator, teardown

    def on_status(status: str, detail: str) -> None:
        ui = ui_ref.get("ui")
        if ui is not None:
            ui.publish("pipeline", {"pipeline": {"status": status, "detail": detail}})

    controller = PipelineController(factory, on_status=on_status)
    ui = SubtitleServer(port=args.ui_port, state=state,
                        on_mute=controller.set_muted,
                        on_pause=controller.set_paused,
                        on_voice=controller.set_reference_voice,
                        controller=controller)
    # The consent recorded beside the configured voice, so the Voices tab can
    # use that voice; and where the conversion service should be, so the
    # pre-flight list can say whether cloning is possible at all.
    reference = None if args.no_clone else (args.voice or cfg.tts.conversion.reference_voice)
    if reference:
        ui.configured_voice = (str(resolve(reference)),
                               args.consent or cfg.tts.conversion.consent)
    ui.conversion_address = (cfg.tts.conversion.host,
                             args.converter_port or cfg.tts.conversion.port)
    ui.conversion_enabled = bool(cfg.tts.conversion.enabled)
    ui_ref["ui"] = ui

    try:
        url = ui.start()
    except OSError as exc:
        # Every candidate port was taken. Say which, and what to do about it; a
        # traceback ending in WinError 10048 tells an operator nothing they can
        # act on thirty seconds before they are due to speak.
        print(f"\n  Could not open the operator page.\n  {exc}")
        return 1
    print(f"\n  parlIAmo is at {url}")
    print("  Open it, read Setup, then press Start. Ctrl+C here to quit.")

    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    started = time.monotonic()
    try:
        while not stopping.is_set():
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            time.sleep(0.2)
    finally:
        print("\nstopping ...")
        controller.stop()
        ui.stop()

    if deliveries:
        summary = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "elapsed_s": round(time.monotonic() - started, 1),
            "direction": f"{cfg.mt.source_lang}->{args.target or cfg.tts.language}",
            "source": "microphone",
            "deliveries": [d.as_dict() for d in deliveries],
        }
        out = ensure_dir("runs/live") / f"translate-{datetime.now():%Y%m%d-%H%M%S}.json"
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=None, help="microphone: index or name substring")
    parser.add_argument("--output", default=None, help="speakers: index or name substring")
    parser.add_argument("--target", default=None, help="target language (default from config)")
    parser.add_argument("--source", default=None,
                        help="spoken language (default from config); sets the recogniser and "
                             "the translator together, as the page does")
    parser.add_argument("--voice", default=None, help="reference wav to clone")
    parser.add_argument("--consent", default="", help="signed consent record reference")
    parser.add_argument("--steps", type=int, default=None, help="conversion diffusion steps")
    parser.add_argument("--no-speak", action="store_true", help="subtitles only")
    parser.add_argument("--converter-port", type=int, default=None,
                        help="voice service port: 8766 RVC (trained), 8765 Seed-VC (zero-shot), "
                             "8767 OmniVoice (zero-shot, speaks the sentence)")
    parser.add_argument("--no-clone", action="store_true",
                        help="speak in the generic voice: skip voice conversion, the largest single term in the latency")
    parser.add_argument("--stream", nargs="?", const="sentence", default=None,
                        choices=["off", "sentence", "chunk"],
                        help="speak before the pause: 'sentence' as each sentence ends, "
                             "'chunk' as words settle (the buffer). Bare --stream means sentence.")
    parser.add_argument("--no-partials", action="store_true",
                        help="wait for the whole sentence before showing a subtitle")
    parser.add_argument("--ui", action="store_true",
                        help="serve the subtitle and operator page on localhost")
    parser.add_argument("--ui-port", type=int, default=8770,
                        help="operator page port; steps forward if it is taken")
    parser.add_argument("--file", default=None,
                        help="replay a wav through the whole pipeline instead of the mic")
    parser.add_argument("--save-audio", default=None,
                        help="with --file: directory to write the produced speech into")
    parser.add_argument("--play", action="store_true",
                        help="with --file: also play the result through the speakers")
    parser.add_argument("--voice-speed", type=float, default=None,
                        help="OmniVoice speaking rate, below 1 slower; 0 copies the reference")
    parser.add_argument("--min-silence-ms", type=int, default=None,
                        help="commit wait - the dominant latency term")
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--mt-model", default=None)
    parser.add_argument("--mt-device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seconds", type=float, default=None, help="stop after this long")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _run_id, run_dir = setup_logging(level="INFO" if args.verbose else "WARNING", jsonl=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "faster_whisper", "datasets",
                  "transformers", "diffusers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    cfg = load_config()
    if args.source:
        # Both or neither: a recogniser and a translator that disagree on the
        # language hand the translator the wrong text with no error.
        cfg.mt.source_lang = args.source
        cfg.asr.language = args.source
    for attr in ("input", "output"):
        value = getattr(args, attr)
        if isinstance(value, str) and value.isdigit():
            setattr(args, attr, int(value))

    # With --ui the application owns the pipeline and builds it when someone
    # presses start. Building it here instead opened the microphone the moment
    # the command ran, which is what left the start button with nothing to do.
    if args.ui and not args.file:
        return run_app(args, cfg, not args.no_speak, run_dir)

    print("\nloading models ...")
    try:
        translator, teardown, speaking = build(args, cfg)
    except ValueError as exc:
        # The consent check lands here. It is a refusal, not a crash.
        print(f"\n  {exc}\n")
        return 2

    # No microphone in replay mode, so there is no acoustic loop to warn about.
    warn_about_the_gate(cfg, speaking and not args.file)

    deliveries: list[DeliveryEvent] = []
    reference_in_use = translator.reference_voice

    ui = None
    if args.ui:
        state = UIState(
            source_lang=cfg.mt.source_lang,
            target_lang=args.target or cfg.tts.language,
            speaking=speaking,
            gate_warning=(
                "pipeline.output_latency_ms is 0 - measure it before using speakers, "
                "or the microphone will hear them and translate the system's own output"
                if speaking and not args.file and cfg.pipeline.output_latency_ms <= 0
                else ""
            ),
        )
        state.voice = "generic" if not reference_in_use else Path(reference_in_use).stem
        ui = SubtitleServer(port=args.ui_port, state=state,
                            on_mute=translator.set_muted,
                            on_pause=translator.set_paused,
                            on_voice=translator.set_reference_voice)
        print(f"\n  operator page: {ui.start()}")
        print("    M mutes audio instantly, P pauses listening, F hides the operator strip, "
              "S shows the Turkish")

    def collect(event: DeliveryEvent) -> None:
        deliveries.append(event)
        print_delivery(event)
        if ui is not None:
            # After printing, and guarded: a display fault must not cost a
            # sentence. The room can follow a terminal; it cannot follow a
            # pipeline that stopped.
            try:
                ui.publish_delivery(event)
            except Exception:
                log.exception("could not publish to the operator page")

    translator.on_delivery = collect

    started = time.monotonic()

    if args.file:
        # Offline replay: the same delivery path, driven by a recording.
        try:
            replay_file(args, cfg, translator)
        finally:
            if translator.playback is not None:
                translator.playback.stop()
            teardown()
        stats = translator.stats
    else:
        stopping = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stopping.set())

        translator.start()
        print("\n" + "=" * 72)
        print(f"  {cfg.mt.source_lang} -> {args.target or cfg.tts.language}"
              f"   commit wait {args.min_silence_ms or cfg.vad.min_silence_ms} ms"
              f"   {'speaking' if speaking else 'subtitles only'}")
        print("  Speak, then pause. Ctrl+C to stop.")
        print("=" * 72)

        try:
            while not stopping.is_set():
                if args.seconds and time.monotonic() - started >= args.seconds:
                    break
                time.sleep(0.2)
        finally:
            print("\nstopping ...")
            stats = translator.stop()
            teardown()

    summary = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "elapsed_s": round(time.monotonic() - started, 1),
        "direction": f"{cfg.mt.source_lang}->{args.target or cfg.tts.language}",
        "spoke": speaking,
        "source": args.file or "microphone",
        "stats": stats.summary(),
        "deliveries": [d.as_dict() for d in deliveries],
    }
    if ui is not None:
        ui.stop()

    out = ensure_dir("runs/live") / f"translate-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(json.dumps(stats.summary(), indent=2))
    print("=" * 72)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
