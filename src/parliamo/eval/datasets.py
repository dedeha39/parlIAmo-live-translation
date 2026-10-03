"""Public speech test sets, cached locally.

Model ranking should not wait on anyone recording their own voice, so the
bake-off runs first against public Turkish data. What that data cannot tell us
is how a model behaves on *this* speaker through *this* microphone - FLEURS is
read speech from studio-ish conditions, and the reference laptop's capture path
turned out to be a 16 kHz DSP pipeline with aggressive noise gating. Public
data ranks the models; a short in-situ recording confirms the ranking holds.

Everything downloads once into ``models/`` (see
:func:`parliamo.paths.configure_model_cache`) so the machine can then run
offline.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


@dataclass(slots=True)
class Utterance:
    """One evaluation item: audio plus its reference transcript."""

    uid: str
    audio: np.ndarray
    sample_rate: int
    reference: str
    language: str
    source: str

    @property
    def duration_s(self) -> float:
        return self.audio.size / float(self.sample_rate)


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    import soxr

    return np.asarray(
        soxr.resample(audio, src_rate, dst_rate, quality="VHQ"), dtype=np.float32
    )


def load_fleurs(
    language: str = "tr_tr",
    split: str = "test",
    limit: int | None = None,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
) -> list[Utterance]:
    """Load the FLEURS test set for *language*.

    FLEURS is read speech across 102 languages with clean references, which
    makes it a reasonable neutral ranking set. It is *not* representative of a
    conference room, and the results should never be quoted as "our accuracy".
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    log.info("loading FLEURS %s/%s", language, split)
    dataset = load_dataset("google/fleurs", language, split=split)

    # Decode the audio ourselves rather than letting `datasets` do it.
    #
    # datasets >= 5 delegates decoding to torchcodec, which needs FFmpeg's
    # *shared* libraries on the DLL search path. The common Windows FFmpeg
    # builds ship a static executable and no DLLs, so the import fails with a
    # missing-library error that has nothing to do with this project. FLEURS
    # audio is plain WAV, which soundfile reads directly - one dependency
    # fewer, and one fewer thing for anyone reusing this to debug.
    dataset = dataset.cast_column("audio", Audio(decode=False))

    items: list[Utterance] = []
    for row in dataset:
        audio_field = row["audio"]
        raw = audio_field.get("bytes")
        if raw:
            array, rate = sf.read(io.BytesIO(raw), dtype="float32")
        else:
            array, rate = sf.read(audio_field["path"], dtype="float32")
        array = np.asarray(array, dtype=np.float32)
        if array.ndim > 1:
            array = array[:, 0]
        array = _resample(array, int(rate), TARGET_SAMPLE_RATE)

        duration = array.size / TARGET_SAMPLE_RATE
        if not (min_duration_s <= duration <= max_duration_s):
            continue

        # FLEURS ships both a normalised `transcription` and a `raw_transcription`
        # that keeps punctuation and casing. We take the raw one and let our own
        # normaliser do the work, so every backend is judged by the same rules.
        reference = str(row.get("raw_transcription") or row.get("transcription") or "").strip()
        if not reference:
            continue

        items.append(
            Utterance(
                uid=f"fleurs-{language}-{row.get('id', len(items))}",
                audio=array,
                sample_rate=TARGET_SAMPLE_RATE,
                reference=reference,
                language=language.split("_")[0],
                source=f"fleurs/{language}/{split}",
            )
        )
        if limit and len(items) >= limit:
            break

    log.info(
        "FLEURS %s/%s: %d utterances, %.1f minutes",
        language, split, len(items), sum(u.duration_s for u in items) / 60.0,
    )
    return items


@dataclass(slots=True)
class SentencePair:
    """One source sentence and its reference translation."""

    uid: str
    source: str
    reference: str
    source_lang: str
    target_lang: str


def load_fleurs_parallel(
    source: str = "tr_tr",
    target: str = "it_it",
    split: str = "test",
    limit: int | None = None,
) -> list[SentencePair]:
    """Build a parallel corpus by joining two FLEURS languages on sentence id.

    FLEURS is the spoken version of FLORES, so the same sentence id in two
    languages is the same underlying sentence. That gives an ungated tr-it
    evaluation set without needing FLORES-200 itself, which is gated behind a
    Hugging Face licence acceptance.

    One caveat worth remembering when quoting the numbers: both sides are
    translations *of the English original*, not of each other. Some divergence
    between them is inherent to the corpus rather than caused by the system
    being measured.
    """
    from datasets import Audio, load_dataset

    def texts(config: str) -> dict[int, str]:
        # Audio decoding is irrelevant here and pulls in torchcodec, so switch
        # it off rather than pay for it.
        data = load_dataset("google/fleurs", config, split=split)
        data = data.cast_column("audio", Audio(decode=False))
        out: dict[int, str] = {}
        for row in data:
            text = str(row.get("raw_transcription") or row.get("transcription") or "").strip()
            if text:
                out.setdefault(int(row["id"]), text)
        return out

    src_texts = texts(source)
    tgt_texts = texts(target)
    shared = sorted(set(src_texts) & set(tgt_texts))

    pairs = [
        SentencePair(
            uid=f"fleurs-{sid}",
            source=src_texts[sid],
            reference=tgt_texts[sid],
            source_lang=source.split("_")[0],
            target_lang=target.split("_")[0],
        )
        for sid in shared
    ]
    if limit:
        pairs = pairs[:limit]

    log.info(
        "FLEURS %s->%s/%s: %d aligned pairs (from %d and %d)",
        source, target, split, len(pairs), len(src_texts), len(tgt_texts),
    )
    return pairs


def load_tsv_pairs(path: str | Path, source_lang: str, target_lang: str) -> list[SentencePair]:
    """Load a parallel corpus from a ``source<TAB>reference`` file."""
    file = Path(path)
    pairs: list[SentencePair] = []
    for line_no, raw in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{file}:{line_no}: expected 'source<TAB>reference'")
        pairs.append(
            SentencePair(
                uid=f"{file.stem}-{line_no}",
                source=parts[0].strip(),
                reference="\t".join(parts[1:]).strip(),
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )
    log.info("%s: %d pairs", file, len(pairs))
    return pairs


def summarise_pairs(pairs: list[SentencePair]) -> dict[str, Any]:
    if not pairs:
        return {"pairs": 0}
    src_len = [len(p.source) for p in pairs]
    ref_len = [len(p.reference) for p in pairs]
    return {
        "pairs": len(pairs),
        "direction": f"{pairs[0].source_lang}->{pairs[0].target_lang}",
        "mean_source_chars": round(sum(src_len) / len(src_len), 1),
        "mean_reference_chars": round(sum(ref_len) / len(ref_len), 1),
    }


def load_wav_manifest(manifest: str | Path) -> list[Utterance]:
    """Load a local test set from a TSV manifest of ``path<TAB>reference``.

    This is how an in-situ recording gets evaluated: read a prepared script,
    split it per sentence, and the script itself is the reference - no manual
    transcription required.
    """
    import soundfile as sf

    manifest_path = Path(manifest)
    base = manifest_path.parent
    items: list[Utterance] = []

    for line_no, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(f"{manifest_path}:{line_no}: expected 'path<TAB>reference'")
        rel, reference = parts[0], "\t".join(parts[1:])

        wav_path = Path(rel)
        if not wav_path.is_absolute():
            wav_path = base / wav_path

        audio, rate = sf.read(wav_path, dtype="float32")
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim > 1:
            array = array[:, 0]
        array = _resample(array, rate, TARGET_SAMPLE_RATE)

        items.append(
            Utterance(
                uid=wav_path.stem,
                audio=array,
                sample_rate=TARGET_SAMPLE_RATE,
                reference=reference.strip(),
                language="tr",
                source=str(manifest_path),
            )
        )

    log.info("manifest %s: %d utterances", manifest_path, len(items))
    return items


def iter_batches(items: list[Utterance], size: int) -> Iterator[list[Utterance]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def summarise(items: list[Utterance]) -> dict[str, Any]:
    durations = [u.duration_s for u in items]
    if not durations:
        return {"utterances": 0}
    return {
        "utterances": len(items),
        "total_minutes": round(sum(durations) / 60.0, 2),
        "mean_duration_s": round(sum(durations) / len(durations), 2),
        "min_duration_s": round(min(durations), 2),
        "max_duration_s": round(max(durations), 2),
        "source": items[0].source,
    }
