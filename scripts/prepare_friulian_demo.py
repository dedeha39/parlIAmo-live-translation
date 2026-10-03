#!/usr/bin/env python
"""Translate the Friulian demonstration sentences, and produce a review sheet.

    python scripts/prepare_friulian_demo.py
    python scripts/prepare_friulian_demo.py --speak --out runs/friulian

Per ADR 0010, the Friulian branch does not need a fine-tune. NLLB-200 has seen
Friulian and produces real Friulian - `nol è un probleme`, `lis lenghis
minoritariis a restin` - with ordinary translation mistakes rather than the
collapse a barely-seen language usually gives.

What it does need is **a native speaker looking at ten sentences**, and this
script produces the thing to hand them: Turkish source, Italian for context,
Friulian output, and a blank column to correct in.

Italian is included because a Friulian speaker in Friuli almost certainly reads
Italian, and because the reviewer needs to know what the sentence was *meant*
to say - not to check the pivot, which is no longer used (ADR 0010).

With ``--speak`` it also synthesises the Friulian through the Italian frontend,
which is what the audience will hear. There is no Friulian voice anywhere:
checked on 2026-09-01, HuggingFace's `fur` tag returns translation models only,
and Meta's MMS-TTS covers 1100+ languages without including it. The wrong
phonetics are disclosed on stage rather than hidden - it is the point being
made.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from parliamo.paths import configure_model_cache

configure_model_cache()

import logging  # noqa: E402

from parliamo.config import load_config  # noqa: E402
from parliamo.logging_setup import configure_console  # noqa: E402
from parliamo.mt import create_backend as make_mt  # noqa: E402
from parliamo.paths import ensure_dir, resolve  # noqa: E402


def read_sentences(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sentences", default="data/friulian/demo-sentences-tr.txt")
    parser.add_argument("--out", default="data/friulian")
    parser.add_argument("--speak", action="store_true",
                        help="also synthesise the Friulian, with Italian phonetics")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "datasets"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    cfg = load_config()
    sentences = read_sentences(resolve(args.sentences))
    if not sentences:
        print(f"no sentences in {args.sentences}")
        return 1

    mt = make_mt(
        cfg.mt.backend, model=cfg.mt.model_path, device=cfg.mt.device,
        source_lang="tr", target_lang="fur",
        compute_type=cfg.mt.compute_type, beam_size=cfg.mt.beam_size,
        max_decoding_length=cfg.mt.max_decoding_length,
        split_sentences=cfg.mt.split_sentences,
    )
    mt.load()

    print(f"\ntranslating {len(sentences)} sentences, tr -> fur direct "
          f"(src_token {cfg.mt.friulian.src_token})\n")
    rows = []
    for i, tr in enumerate(sentences, 1):
        fur = mt.translate(tr, source_lang="tr", target_lang="fur").text
        it = mt.translate(tr, source_lang="tr", target_lang="it").text
        rows.append((i, tr, it, fur))
        print(f"{i:>2}. tr : {tr}")
        print(f"    it : {it}")
        print(f"    fur: {fur}\n")
    mt.unload()

    out_dir = ensure_dir(args.out)
    sheet = out_dir / "review-sheet.md"
    lines = [
        "# Friulian demonstration sentences — for review",
        "",
        f"Generated {date.today().isoformat()} by `scripts/prepare_friulian_demo.py`.",
        "",
        "These are for a talk about how AI handles speech, given in Italy. The",
        "Friulian below was produced by **NLLB-200**, a general translation model,",
        "with **no fine-tuning and no Friulian training data of our own**. It was",
        "translated directly from Turkish.",
        "",
        "**What would help enormously:** mark anything that is wrong, and write",
        "what it should be. Rough notes are perfectly useful — the point of the",
        "segment is to be honest about what a general model gets right and wrong",
        "for a language it has barely seen, so corrections are the material, not",
        "a failure.",
        "",
        "Italian is given only so you can see what each sentence was meant to say.",
        "",
        "---",
        "",
    ]
    for i, _tr, it, fur in rows:
        lines += [
            f"### {i}.",
            "",
            f"- **Italian (meaning):** {it}",
            f"- **Friulian (to check):** **{fur}**",
            "- **Correction:**",
            "",
        ]
    lines += [
        "---",
        "",
        "## Two questions, if you have a moment",
        "",
        "1. Does this read as Friulian, or as Italian with Friulian spelling?",
        "2. Is the register right for speaking to an audience, or too written?",
        "",
        "## One thing we already know",
        "",
        "No text-to-speech model exists for Friulian anywhere — we checked. So the",
        "audience will hear these sentences read with **Italian** phonetics. That",
        "gap is exactly what the segment is about, and it is disclosed on stage",
        "rather than hidden.",
        "",
    ]
    sheet.write_text("\n".join(lines), encoding="utf-8")
    print(f"review sheet: {sheet}")

    if args.speak:
        from parliamo.tts import create_backend as make_tts

        tts = make_tts(cfg.tts.backend, device=cfg.tts.device, language="fur")
        tts.load()
        audio_dir = ensure_dir(Path(args.out) / "audio")
        import soundfile as sf

        for i, _tr, _it, fur in rows:
            speech = tts.speak(fur, language="fur")
            sf.write(audio_dir / f"{i:02d}.wav", speech.audio, speech.sample_rate,
                     subtype="FLOAT")
        tts.unload()
        print(f"audio (Italian phonetics, disclosed): {audio_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
