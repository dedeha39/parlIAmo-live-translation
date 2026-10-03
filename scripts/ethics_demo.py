#!/usr/bin/env python
"""Build the ethics segment: the clone, the disclosure, and the watermark table.

    python scripts/ethics_demo.py
    python scripts/ethics_demo.py --with-chatterbox      # adds the marked half
    python scripts/ethics_demo.py --no-disclosure        # not recommended

Three things the audience needs, in order.

**1. The attack, in a voice they have been listening to for an hour.**
Voice fraud aimed at older people is not a politician impression. It is
someone they love, in trouble, needing money now - so the demonstration is the
presenter's own cloned voice reading the actual scam script, not a celebrity.
A deepfaked public figure teaches a room that public figures get faked. This
teaches them that *they* are the target.

**2. Every clip says out loud that it is synthetic**, at the start, at the end,
and at a random point inside. The interior one is the safeguard: without it the
clip becomes usable by trimming a known offset off each end. It is spoken in a
different voice from the cloned one, because a warning delivered in the
impersonated voice is one more sentence that person never said.

**3. The watermark comparison, which is the uncomfortable part.**
Chatterbox marks its output and the detector recovers it perfectly. The
pipeline that actually ships in this repository - Kokoro generating, Seed-VC
applying identity - marks nothing, and the detector correctly finds nothing.
Both need fifteen seconds of reference audio. The honest conclusion is not
"watermarks protect you"; it is that a watermark tells you the *generator chose
to mark its output*, and an attacker chooses one that does not.

Requires a consent record, like everything else that clones a voice here.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime

import numpy as np

from parliamo.paths import configure_model_cache

configure_model_cache()

import logging  # noqa: E402

from parliamo.config import load_config  # noqa: E402
from parliamo.ethics import detect  # noqa: E402
from parliamo.ethics.disclosure import DEFAULT_TEXT_IT, add_disclosure  # noqa: E402
from parliamo.logging_setup import configure_console  # noqa: E402
from parliamo.paths import ensure_dir, resolve  # noqa: E402
from parliamo.tts import VoiceProfile  # noqa: E402
from parliamo.tts import create_backend as make_tts  # noqa: E402
from parliamo.tts.conversion import VoiceConverter  # noqa: E402

log = logging.getLogger("ethics_demo")


def read_lines(path) -> list[str]:
    return [
        line.strip()
        for line in resolve(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--script", default="data/ethics/fraud-script-it.txt")
    parser.add_argument("--out", default="runs/ethics")
    parser.add_argument("--voice", default=None, help="reference wav; default from config")
    parser.add_argument("--consent", default="", help="signed consent record reference")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--with-chatterbox", action="store_true",
                        help="also generate the watermarked half of the comparison")
    parser.add_argument("--no-disclosure", action="store_true",
                        help="omit the spoken 'this is synthetic' - think first")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "diffusers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    cfg = load_config()
    reference = args.voice or cfg.tts.conversion.reference_voice
    consent = args.consent or cfg.tts.conversion.consent
    if not reference:
        print("no reference voice. Set tts.conversion.reference_voice or pass --voice.")
        return 2
    # Refuses without a record, before anything is loaded or cloned.
    VoiceProfile(name="demo", reference_path=str(resolve(reference)), consent=consent).validate()

    lines = read_lines(args.script)
    out_dir = ensure_dir(args.out)
    tts = make_tts(cfg.tts.backend, device=cfg.tts.device, language=cfg.tts.language)
    tts.load()
    tts.warmup()

    # The disclosure is spoken by the generic voice, never the cloned one.
    announcement = tts.speak(DEFAULT_TEXT_IT, language=cfg.tts.language)

    converter = VoiceConverter(
        host=cfg.tts.conversion.host, port=cfg.tts.conversion.port,
        diffusion_steps=args.steps or cfg.tts.conversion.diffusion_steps,
        timeout=cfg.tts.conversion.timeout_s,
    )
    cloned_available = converter.available()
    if not cloned_available:
        print(f"\n  conversion service not reachable at {converter.address}")
        print("  the clip will be in the generic voice, which weakens the demonstration")
        print("  start it with: <seedvc-env>/python.exe scripts/voice_conversion_server.py\n")

    print(f"\nreading {len(lines)} lines from {args.script}\n")
    pieces: list[np.ndarray] = []
    rate = tts.sample_rate
    for i, line in enumerate(lines, 1):
        speech = tts.speak(line, language=cfg.tts.language)
        audio, rate = speech.audio, speech.sample_rate
        if cloned_available:
            converted = converter.convert(audio, rate, str(resolve(reference)))
            audio, rate = converted.audio, converted.sample_rate
        pieces.append(audio)
        pieces.append(np.zeros(int(0.45 * rate), dtype=np.float32))
        print(f"  {i}. {line}")

    clip = np.concatenate(pieces)

    report: dict[str, object] = {
        "generated": datetime.now().astimezone().isoformat(),
        "voice": "cloned" if cloned_available else "generic",
        "reference": str(reference),
        "consent": consent,
        "lines": lines,
    }

    if not args.no_disclosure:
        clip, disclosure = add_disclosure(
            clip, rate, announcement.audio, announcement.sample_rate
        )
        report["disclosure"] = disclosure.as_dict()
        print(f"\n  disclosure spoken {len(disclosure.positions_s)} times "
              f"at {[round(p, 1) for p in disclosure.positions_s]} s")
    else:
        print("\n  WARNING: no spoken disclosure. This clip is usable as a fake.")

    import soundfile as sf

    fraud_path = out_dir / "1-fraud-call.wav"
    sf.write(fraud_path, clip, rate, subtype="FLOAT")
    print(f"  written to {fraud_path}")

    # -- the watermark table ------------------------------------------------
    print("\n" + "=" * 68)
    print("DOES ANYTHING IN THIS FILE SAY IT IS SYNTHETIC?")
    print("=" * 68)
    rows: list[dict[str, object]] = []

    def check(label: str, audio: np.ndarray, sample_rate: int) -> None:
        found = detect(audio, sample_rate)
        rows.append({"audio": label, **found.as_dict()})
        print(f"  {label:<38} {'YES' if found.present else 'no':>4}  "
              f"({found.confidence:.2f})")

    check("this pipeline: Kokoro + Seed-VC", clip, rate)

    if args.with_chatterbox:
        try:
            chatter = make_tts("chatterbox", device=cfg.tts.device,
                               language=cfg.tts.language)
            chatter.load()
            chatter.register_voice(
                VoiceProfile(name="demo", reference_path=str(resolve(reference)),
                             consent=consent)
            )
            marked = chatter.speak(lines[0], language=cfg.tts.language, voice="demo")
            marked_path = out_dir / "2-chatterbox-watermarked.wav"
            sf.write(marked_path, marked.audio, marked.sample_rate, subtype="FLOAT")
            check("Chatterbox (marks by default)", marked.audio, marked.sample_rate)
            print(f"     written to {marked_path}")
            chatter.unload()
        except Exception as exc:
            print(f"  Chatterbox unavailable: {type(exc).__name__}: {exc}")

    print("-" * 68)
    print("  A watermark says the generator chose to mark its output.")
    print("  Silence from the detector says nothing at all - and the pipeline")
    print("  on this laptop is one of the ones that says nothing.")
    print("=" * 68)

    report["watermark_checks"] = rows
    report["eu_ai_act"] = {
        "article": 50,
        "in_force": "2026-08-02",
        "watermarking_grace_ends": "2026-12-02",
        "note": "binds providers of generative systems, not this demonstration",
    }
    (out_dir / f"report-{date.today().isoformat()}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nreport written to {out_dir}")
    tts.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
