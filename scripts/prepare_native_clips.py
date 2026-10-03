#!/usr/bin/env python
"""Cut a native speaker's recording into sentences the Friulian tab can play.

Why a real recording, and why it is only ever *played*
------------------------------------------------------
No Friulian synthesiser exists - MMS-TTS, eSpeak NG and ElevenLabs were all
checked - so the pipeline reads Friulian with the Italian frontend and says so.
The honest half of that demonstration is what the language actually sounds
like, and the only source of that is a person who speaks it.

Wikitongues records native speakers for language documentation and releases
the videos under CC BY-SA 4.0. That licence covers copying and playing the
recording, with attribution. It does not cover the speaker's *voice*: nobody
who sat down to be recorded for documentation agreed to have a synthesiser say
new sentences in their voice. So these clips are served to the page and played
as recorded, and the folder is never listed in the Voices tab.

Usage::

    python scripts/prepare_native_clips.py data/friulian/marco/recording.webm \
        --speaker "Marco Moroldo" \
        --source "https://commons.wikimedia.org/wiki/File:Wikitongues-Friulian-Moroldo.webm" \
        --licence "CC BY-SA 4.0"

Needs ffmpeg on PATH for anything that is not already a 16 kHz mono WAV.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from parliamo.audio.vad import SegmenterConfig, SileroVAD, segment_audio  # noqa: E402
from parliamo.logging_setup import configure_console  # noqa: E402


def to_wav(source: Path, target: Path) -> Path:
    """Mono 16 kHz PCM, which is what the segmenter and the page both want."""
    if source.suffix.lower() == ".wav":
        import soundfile as sf

        info = sf.info(str(source))
        if info.samplerate == 16000 and info.channels == 1:
            return source
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH and the input is not 16 kHz mono WAV")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
        check=True,
    )
    return target


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--source", required=True, help="where the recording came from")
    parser.add_argument("--licence", required=True)
    parser.add_argument("--prefix", default=None, help="clip file prefix (default: from speaker)")
    parser.add_argument("--min-silence-ms", type=int, default=700,
                        help="longer than the live pipeline: a sentence cut at a breath "
                             "is worse than two joined ones when played to a room")
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    args = parser.parse_args()

    import numpy as np
    import soundfile as sf

    folder = args.recording.resolve().parent
    wav = to_wav(args.recording.resolve(), folder / (args.recording.stem + "-16k.wav"))
    audio, rate = sf.read(str(wav), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)

    vad = SileroVAD()
    vad.load()
    config = SegmenterConfig(min_silence_ms=args.min_silence_ms,
                             max_segment_ms=int(args.max_seconds * 1000), min_speech_ms=400)

    prefix = args.prefix or args.speaker.split()[0].lower()
    clips = folder / "clips"
    clips.mkdir(exist_ok=True)
    for old in clips.glob(f"{prefix}-*.wav"):
        old.unlink()

    index: list[dict] = []
    for segment in segment_audio(audio, vad, config):
        if segment.duration_s < args.min_seconds:
            continue
        name = f"{prefix}-{len(index) + 1:03d}.wav"
        sf.write(str(clips / name), segment.audio, segment.sample_rate)
        index.append({"n": len(index) + 1, "file": name,
                      "start_s": round(segment.start_s, 2),
                      "end_s": round(segment.end_s, 2),
                      "duration_s": round(segment.duration_s, 2)})

    (clips / "index.json").write_text(json.dumps({
        "speaker": args.speaker,
        "source": args.source,
        "licence": args.licence,
        "note": ("A native speaker, recorded for language documentation. Played as "
                 "recorded, with attribution. Not used as a voice reference: the "
                 "licence covers the recording, not the speaker's voice."),
        "clips": index,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    durations = [c["duration_s"] for c in index]
    print(f"{len(index)} sentences from {args.speaker}: "
          f"{min(durations):.1f}-{max(durations):.1f} s, {sum(durations):.0f} s in all")
    print(f"index: {clips / 'index.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
