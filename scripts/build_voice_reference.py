#!/usr/bin/env python
"""Assemble a good reference recording for voice cloning, from ones you have.

    python scripts/build_voice_reference.py --from data/insitu/20260830-141246
    python scripts/build_voice_reference.py --from data/voices/raw --seconds 30
    python scripts/build_voice_reference.py --from data/insitu/... --report-only

Why this is not just `cat *.wav`
--------------------------------
Voice conversion derives a speaker embedding from one reference recording, so
that recording decides how much the output sounds like the person.

The first live cloning test used a single 7.8 s in-situ sentence, and the result
was judged not good enough. Measuring the 40 recordings afterwards showed why
the choice was poor: **between 7% and 39% of every file is exact digital
silence**, because the laptop's Intel Smart Sound driver gates non-speech to
zero — and the file that was picked was one of the worst, at 35.7%. All of them
also sit around -30 dBFS, which is very quiet.

So this script ranks the candidates by how little of them the driver removed,
drops the dead air inside the ones it keeps, joins them with short pauses and
normalises the level. It reports what it did, because a reference that was
assembled silently is a reference nobody can question.

It is also how a volunteer's fifteen seconds gets prepared on stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from parliamo.logging_setup import configure_console
from parliamo.paths import ensure_dir, resolve
from parliamo.tts.reference import (
    build_reference,
    measure_clip,
    rank_candidates,
)

TARGET_RATE = 16000


def load(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if rate != TARGET_RATE:
        import soxr

        audio = np.asarray(
            soxr.resample(audio, rate, TARGET_RATE, quality="VHQ"), dtype=np.float32
        )
    return audio, TARGET_RATE


def build_from_one_file(args, source: Path) -> int:
    """Build a reference from a single continuous recording."""
    import soundfile as sf

    audio, _ = load(source)
    stats = measure_clip(audio, TARGET_RATE, source)
    print(f"\n{source.name}: {stats.duration_s:.1f}s, peak {stats.peak:.3f}, "
          f"{stats.rms_dbfs:.1f} dBFS, {100 * stats.gated_fraction:.1f}% gated")

    start = int(args.skip_seconds * TARGET_RATE)
    # Take a contiguous stretch. A reference cut from one continuous take keeps
    # the speaker's natural rhythm, which a concatenation of separate sentences
    # does not - and rhythm is part of what the embedding picks up.
    wanted = int(args.seconds * TARGET_RATE * 1.4)   # trimming will shorten it
    excerpt = audio[start : start + wanted]
    if excerpt.size == 0:
        print(f"--skip-seconds {args.skip_seconds} is past the end of the file")
        return 1

    reference = build_reference(
        [(excerpt, TARGET_RATE)],
        target_seconds=args.seconds,
        sample_rate=TARGET_RATE,
        trim=not args.no_trim,
        normalise_peak=None if args.no_normalise else 0.95,
    )
    # build_reference stops *between* clips, and there is only one here, so cut
    # to length afterwards. Reference length drives conversion cost directly -
    # 28 s costs 41% more than 12 s - so overshooting is not free.
    wanted_samples = int(args.seconds * TARGET_RATE)
    if reference.size > wanted_samples:
        reference = reference[:wanted_samples]

    out_path = (
        Path(args.out) if args.out
        else ensure_dir("data/voices") / f"{source.stem}-reference.wav"
    )
    if not out_path.is_absolute():
        out_path = resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, reference, TARGET_RATE, subtype="FLOAT")

    result = measure_clip(reference, TARGET_RATE, out_path)
    print(f"\nbuilt {result.duration_s:.1f}s from {args.skip_seconds:.0f}s in")
    print(f"  peak       {result.peak:.3f}")
    print(f"  written to {out_path}")
    return 0


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--from", dest="source", required=True,
                        help="directory of wav files, or a single recording")
    parser.add_argument("--skip-seconds", type=float, default=0.0,
                        help="ignore this much from the start of a single recording")
    parser.add_argument("--out", default=None,
                        help="output wav (default: data/voices/<dirname>-reference.wav)")
    parser.add_argument("--seconds", type=float, default=25.0,
                        help="target length; config asks for 15-30 s")
    parser.add_argument("--max-gated", type=float, default=0.25,
                        help="skip files more than this fraction digital silence")
    parser.add_argument("--no-trim", action="store_true",
                        help="keep the gated stretches instead of dropping them")
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="rank the candidates and stop")
    args = parser.parse_args()

    source = resolve(args.source)
    if source.is_file():
        # A volunteer on stage hands over one recording, not a directory. Slice
        # it into candidate chunks so the same ranking and trimming applies.
        return build_from_one_file(args, source)

    files = sorted(source.glob("*.wav"))
    if not files:
        print(f"no wav files in {source}")
        return 1

    stats = []
    clips: dict[Path, np.ndarray] = {}
    for path in files:
        audio, _ = load(path)
        clips[path] = audio
        stats.append(measure_clip(audio, TARGET_RATE, path))

    ranked = rank_candidates(stats)

    print(f"\n{len(files)} candidates in {source}")
    print("=" * 74)
    print(f"{'file':<12}{'sec':>7}{'voiced':>8}{'peak':>7}{'rms dBFS':>10}{'gated%':>9}")
    print("-" * 74)
    for s in ranked:
        flag = "  skip" if s.gated_fraction > args.max_gated else ""
        print(f"{s.path.name:<12}{s.duration_s:>7.1f}{s.voiced_s:>8.1f}"
              f"{s.peak:>7.3f}{s.rms_dbfs:>10.1f}{100 * s.gated_fraction:>8.1f}%{flag}")
    print("-" * 74)

    worst = max(stats, key=lambda s: s.gated_fraction)
    print(f"  driver gating removes {100 * min(s.gated_fraction for s in stats):.0f}"
          f"-{100 * worst.gated_fraction:.0f}% of these recordings.")
    print("  That is the microphone's noise suppression, not silence you left.")

    if args.report_only:
        return 0

    usable = [s for s in ranked if s.gated_fraction <= args.max_gated]
    if not usable:
        print(f"\nno file is under --max-gated {args.max_gated}; nothing to build from")
        return 1

    reference = build_reference(
        [(clips[s.path], TARGET_RATE) for s in usable],
        target_seconds=args.seconds,
        sample_rate=TARGET_RATE,
        trim=not args.no_trim,
        normalise_peak=None if args.no_normalise else 0.95,
    )
    if reference.size == 0:
        print("\nnothing usable survived trimming")
        return 1

    out_path = (
        Path(args.out)
        if args.out
        else ensure_dir("data/voices") / f"{source.name}-reference.wav"
    )
    if not out_path.is_absolute():
        out_path = resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    sf.write(out_path, reference, TARGET_RATE, subtype="FLOAT")

    used = []
    collected = 0.0
    for s in usable:
        used.append(s.path.name)
        collected += s.voiced_s
        if collected >= args.seconds:
            break

    print(f"\nbuilt {reference.size / TARGET_RATE:.1f}s from {len(used)} recordings")
    print(f"  files      {', '.join(used)}")
    print(f"  peak       {float(np.abs(reference).max()):.3f}")
    print(f"  written to {out_path}")
    print("\nUse it with:")
    print(f'  scripts/live_translate.py --voice {out_path} --consent "<reference>"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
