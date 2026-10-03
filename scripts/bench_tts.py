#!/usr/bin/env python
"""Measure speech synthesis latency - the last unmeasured term in the budget.

    python scripts/bench_tts.py
    python scripts/bench_tts.py --language tr
    python scripts/bench_tts.py --voice-reference data/insitu/.../001.wav
    python scripts/bench_tts.py --save runs/tts/samples

Why this blocks a decision
--------------------------
The end-to-end delay the audience experiences is::

    commit wait (700 ms)  +  recognition  +  translation  +  synthesis

Every term but the last has been measured. Until synthesis is known, the choice
between large-v3 (724 ms p95 recognition, 12.78% WER) and large-v3-turbo
(293 ms, 14.00%) cannot be settled: whether 431 ms is affordable depends
entirely on how much of the budget synthesis has already taken.

RTF below 1.0 is necessary but not sufficient. What matters on stage is
*absolute* latency per sentence, because nothing is heard until the whole
utterance is generated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.config import load_config  # noqa: E402
from parliamo.logging_setup import setup_logging  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402
from parliamo.tts import VoiceProfile, create_backend  # noqa: E402

log = logging.getLogger("bench_tts")

# Sentences from the presentation script, spanning the lengths that will
# actually occur. Synthesis time scales with output length, so a single short
# probe would flatter the result.
SENTENCES = {
    "it": [
        "Buongiorno.",
        "Questo sistema funziona interamente su questo computer portatile.",
        "La clonazione vocale non richiede più lunghe registrazioni: quindici secondi bastano.",
        "Le lingue minoritarie restano sistematicamente indietro nel mondo digitale, "
        "e questo non è un problema tecnico ma una scelta collettiva su dove investire.",
    ],
    "tr": [
        "Günaydın.",
        "Bu sistem tamamen bu dizüstü bilgisayarda çalışıyor.",
        "Ses klonlama için artık uzun kayıtlara ihtiyaç yok, on beş saniye yetiyor.",
        "Azınlık dilleri dijital dünyada sistematik olarak geride kalıyor ve bu teknik "
        "bir sorun değil, nereye yatırım yapacağımıza dair ortak bir tercih.",
    ],
    "fur": [
        "Bundì.",
        "Chest sisteme al funzione dut sul computer.",
        "Il furlan al è une lenghe minoritarie e nol à un model di sintesi vocâl.",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--language", default="it", choices=sorted(SENTENCES))
    parser.add_argument("--backend", default=None,
                        help="synthesiser to measure; defaults to tts.backend from config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--voice-reference", default=None,
                        help="wav to clone; requires --consent")
    parser.add_argument("--consent", default="",
                        help="reference to the signed consent record")
    parser.add_argument("--save", default=None, help="directory to write wav samples into")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    setup_logging(level="INFO", jsonl=True)
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "diffusers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    # Read the backend from config rather than naming one here. Hardcoding
    # "chatterbox" is how this script went on measuring the synthesiser that
    # ADR 0006 had already replaced on the live path.
    name = args.backend or load_config().tts.backend
    backend = create_backend(name, device=args.device, language=args.language)
    print(f"\nmeasuring TTS backend: {name}")
    backend.load()
    warm = backend.warmup()
    print(f"\nloaded in {backend.load_seconds:.1f}s, {backend.load_vram_mb:.0f} MiB, "
          f"warmup {warm * 1000:.0f} ms")

    voice_name = None
    if args.voice_reference:
        profile = VoiceProfile(
            name="reference", reference_path=args.voice_reference, consent=args.consent
        )
        # Raises when consent is empty. Cloning a voice is the thing this
        # project warns people about; the code refuses to do it undocumented.
        backend.register_voice(profile)
        voice_name = "reference"
        print(f"voice cloned from {args.voice_reference}")

    save_dir = ensure_dir(args.save) if args.save else None
    rows: list[dict[str, Any]] = []

    print(f"\n{'chars':>6} {'audio s':>8} {'gen s':>7} {'RTF':>6}  text")
    print("-" * 92)
    for sentence in SENTENCES[args.language]:
        per_sentence = []
        speech = None
        for _ in range(args.repeats):
            speech = backend.speak(sentence, language=args.language, voice=voice_name)
            per_sentence.append(speech.compute_s)
        assert speech is not None

        median = sorted(per_sentence)[len(per_sentence) // 2]
        row = {
            "chars": len(sentence),
            "audio_s": round(speech.duration_s, 3),
            "gen_s_median": round(median, 3),
            "gen_s_all": [round(t, 3) for t in per_sentence],
            "rtf": round(median / max(speech.duration_s, 1e-9), 3),
            "text": sentence,
            "language": speech.language,
        }
        rows.append(row)
        print(f"{row['chars']:>6} {row['audio_s']:>8.2f} {row['gen_s_median']:>7.2f} "
              f"{row['rtf']:>6.2f}  {sentence[:52]}")

        if save_dir is not None:
            import soundfile as sf

            path = save_dir / f"{args.language}-{len(sentence):03d}.wav"
            sf.write(path, speech.audio, speech.sample_rate, subtype="FLOAT")

    print("-" * 92)
    gens = [r["gen_s_median"] for r in rows]
    print(f"generation: min {min(gens):.2f}s, median {sorted(gens)[len(gens)//2]:.2f}s, "
          f"max {max(gens):.2f}s")
    print(f"RTF:        min {min(r['rtf'] for r in rows):.2f}, "
          f"max {max(r['rtf'] for r in rows):.2f}")

    # Where this synthesis time lands in the whole budget. Voice conversion is
    # a separate stage downstream (Seed-VC, ~0.92 s at 4 diffusion steps
    # including IPC) and is shown separately rather than folded in, because a
    # run without a cloned voice does not pay it.
    typical = sorted(gens)[len(gens) // 2]
    conversion_s = 0.92
    print("\nEnd-to-end budget with this synthesis time:")
    for asr_name, asr_p95 in (("large-v3-turbo", 0.293), ("large-v3", 0.724)):
        base = 0.700 + asr_p95 + 0.487 + typical
        print(f"  {asr_name:<16} 0.70 commit + {asr_p95:.2f} asr + 0.49 mt + "
              f"{typical:.2f} tts = {base:.2f}s"
              f"   (+{conversion_s:.2f} cloned voice = {base + conversion_s:.2f}s)")

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "language": args.language,
        "device": args.device,
        "backend": backend.describe(),
        "cloned_voice": bool(voice_name),
        "results": rows,
    }
    out_dir = ensure_dir("runs/tts")
    path = args.out or out_dir / f"tts-{datetime.now():%Y%m%d-%H%M%S}.json"
    from pathlib import Path

    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {path}")
    if save_dir:
        print(f"samples written to {save_dir}")

    backend.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
