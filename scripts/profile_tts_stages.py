#!/usr/bin/env python
"""Split synthesis time between its two stages, and test the cheap levers.

    python scripts/profile_tts_stages.py

Chatterbox generates in two steps::

    t3.inference(...)     autoregressive speech-token generation, up to 1000 steps
    s3gen.inference(...)  vocoder turning those tokens into a waveform

The measured cost is 5.4 s fixed plus 31 ms per character. Chunking cannot
touch the fixed part, so the only remaining options are to make one of these
stages cheaper or to stream. Knowing which stage holds the time decides which.

Two levers are tested here because they cost nothing to try:

**cfg_weight = 0.** Classifier-free guidance duplicates the text into a batch of
two (``torch.cat([text_tokens, text_tokens])``), so the autoregressive model
runs over twice the sequence. Disabling it should roughly halve T3 time, at some
cost in prosody fidelity.

**Shorter generation cap.** ``max_new_tokens`` is hardcoded to 1000 with a TODO
beside it. Generation stops early on an EOS token, so this should not bind - but
"should not" is not a measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any

from parliamo.paths import configure_model_cache

configure_model_cache()

from parliamo.logging_setup import configure_console  # noqa: E402
from parliamo.paths import ensure_dir  # noqa: E402

log = logging.getLogger("profile_stages")

SENTENCES = [
    "Sì.",
    "Questo sistema funziona interamente su questo computer portatile.",
    "Le lingue minoritarie restano sistematicamente indietro nel mondo digitale, "
    "e questo non è un problema tecnico ma una scelta collettiva.",
]


def timed_generate(model, text: str, language_id: str, cfg_weight: float) -> dict[str, Any]:
    """Reproduce generate() with a timer around each stage."""
    import torch
    import torch.nn.functional as F
    from chatterbox.mtl_tts import drop_invalid_tokens, punc_norm

    text = punc_norm(text)
    t0 = time.perf_counter()
    text_tokens = model.tokenizer.text_to_tokens(text, language_id=language_id).to(model.device)
    # The batch is doubled unconditionally, exactly as generate() does. An
    # earlier version of this script skipped the duplication when cfg_weight
    # was 0 and crashed inside t3.inference: the conditioning tensors are
    # prepared for a batch of two regardless. So cfg_weight=0 does not avoid
    # running the model over two sequences - it only changes how the two
    # logits are combined. Any saving is in arithmetic, not in sequence count.
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    tokenise_s = time.perf_counter() - t0

    with torch.inference_mode():
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=cfg_weight,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        )
        torch.cuda.synchronize()
        t3_s = time.perf_counter() - t1

        speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(model.device)
        n_tokens = int(speech_tokens.numel())

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        wav, _ = model.s3gen.inference(speech_tokens=speech_tokens, ref_dict=model.conds.gen)
        torch.cuda.synchronize()
        s3gen_s = time.perf_counter() - t2

    audio_s = wav.numel() / model.sr
    return {
        "chars": len(text),
        "speech_tokens": n_tokens,
        "tokenise_s": round(tokenise_s, 4),
        "t3_s": round(t3_s, 3),
        "s3gen_s": round(s3gen_s, 3),
        "total_s": round(tokenise_s + t3_s + s3gen_s, 3),
        "audio_s": round(audio_s, 3),
        "t3_tokens_per_s": round(n_tokens / t3_s, 1) if t3_s else 0.0,
    }


def main() -> int:
    configure_console()  # Windows consoles default to cp1252; see logging_setup
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--language", default="it")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "transformers", "diffusers"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    import warnings

    warnings.filterwarnings("ignore")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    print("loading ...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    model.generate("Prova.", language_id=args.language)  # warm

    rows: list[dict[str, Any]] = []
    print(f"\n{'cfg':>5} {'chars':>6} {'tokens':>7} {'T3 s':>7} {'S3Gen s':>8} "
          f"{'total':>7} {'audio':>7} {'tok/s':>7}")
    print("-" * 66)
    for cfg_weight in (0.5, 0.0):
        for text in SENTENCES:
            best = None
            for _ in range(args.repeats):
                r = timed_generate(model, text, args.language, cfg_weight)
                if best is None or r["total_s"] < best["total_s"]:
                    best = r
            assert best is not None
            best["cfg_weight"] = cfg_weight
            rows.append(best)
            print(f"{cfg_weight:>5.1f} {best['chars']:>6} {best['speech_tokens']:>7} "
                  f"{best['t3_s']:>7.2f} {best['s3gen_s']:>8.2f} {best['total_s']:>7.2f} "
                  f"{best['audio_s']:>7.2f} {best['t3_tokens_per_s']:>7.1f}")

    print("\n" + "=" * 66)
    print("WHERE THE TIME GOES")
    print("=" * 66)
    for cfg_weight in (0.5, 0.0):
        subset = [r for r in rows if r["cfg_weight"] == cfg_weight]
        t3 = sum(r["t3_s"] for r in subset)
        s3 = sum(r["s3gen_s"] for r in subset)
        total = t3 + s3
        print(f"  cfg_weight={cfg_weight}:  T3 {100 * t3 / total:4.0f}%   "
              f"S3Gen {100 * s3 / total:4.0f}%   (total {total:.2f} s)")

    on = [r for r in rows if r["cfg_weight"] == 0.5]
    off = [r for r in rows if r["cfg_weight"] == 0.0]
    if on and off:
        t3_on = sum(r["t3_s"] for r in on)
        t3_off = sum(r["t3_s"] for r in off)
        tot_on = sum(r["total_s"] for r in on)
        tot_off = sum(r["total_s"] for r in off)
        print(f"\n  Disabling CFG: T3 {t3_on:.2f} -> {t3_off:.2f} s "
              f"({100 * (1 - t3_off / t3_on):.0f}% faster), "
              f"total {tot_on:.2f} -> {tot_off:.2f} s "
              f"({100 * (1 - tot_off / tot_on):.0f}% faster)")
        print("  Quality effect is NOT measured here - listen before adopting it.")

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "language": args.language,
        "measurements": rows,
    }
    path = args.out or ensure_dir("runs/tts") / f"stages-{datetime.now():%Y%m%d-%H%M%S}.json"
    from pathlib import Path

    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
