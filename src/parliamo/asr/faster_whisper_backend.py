"""Whisper via CTranslate2 (faster-whisper).

The default recogniser. CTranslate2 has first-class Windows CUDA support, which
is what kept the project off vLLM (see docs/adr/0001-windows-native.md), and its
int8 quantisation is what makes a Whisper-class model fit alongside translation
and synthesis in 8 GB.

Decoding is deliberately greedy and deterministic:

* ``beam_size=1`` - beam search buys a little accuracy for a lot of latency,
  and on stage latency is the scarcer resource.
* ``temperature=0`` with no fallback - the usual temperature ladder retries a
  segment several times when confidence is low, which turns a predictable
  200 ms into an unpredictable 2 s.
* ``condition_on_previous_text=False`` - conditioning makes Whisper repeat
  itself when it slips, and a repetition loop mid-presentation is unrecoverable.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .base import ASRBackend, Segment, Transcript
from .plausibility import hotword_echo, subtitle_credit, syllables, too_fast
from .repetition import (
    DEFAULT_COMPRESSION_THRESHOLD,
    DEFAULT_MAX_PHRASE_REPEATS,
    analyse,
    collapse_repeats,
)

log = logging.getLogger(__name__)

# Aliases so config can name a model without repeating the Hugging Face path.
MODEL_ALIASES = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}


class FasterWhisperBackend(ASRBackend):
    name = "faster_whisper"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        language: str = "tr",
        compute_type: str = "int8_float16",
        beam_size: int = 1,
        condition_on_previous_text: bool = False,
        vad_filter: bool = False,
        temperature: float = 0.0,
        download_root: str | None = None,
        drop_repetitive: bool = True,
        repetition_threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        phrase_repeat_limit: int = DEFAULT_MAX_PHRASE_REPEATS,
        hotwords: str | None = None,
        initial_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            language=language,
            compute_type=compute_type,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            temperature=temperature,
            drop_repetitive=drop_repetitive,
            **kwargs,
        )
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
        self.temperature = temperature
        self.download_root = download_root
        self.drop_repetitive = drop_repetitive
        self.repetition_threshold = repetition_threshold
        self.phrase_repeat_limit = phrase_repeat_limit
        # Whisper can be told which words to expect. Names it has never seen
        # ("Friulian", "ARLeF", "Gorizia") are where recognition fails hardest,
        # and biasing costs no VRAM and no measurable time - unlike moving to a
        # bigger model. Measured before it was adopted; see ADR 0002.
        self.hotwords = hotwords
        self.initial_prompt = initial_prompt
        self._model: Any = None

    # -- lifecycle --------------------------------------------------------

    def _resolved_model(self) -> str:
        return MODEL_ALIASES.get(self.model, self.model)

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        resolved = self._resolved_model()
        log.info(
            "loading faster-whisper %s (%s) on %s as %s",
            self.model, resolved, self.device, self.compute_type,
        )
        self._model = WhisperModel(
            resolved,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.download_root,
            # One worker: the pipeline serialises requests anyway, and extra
            # workers duplicate the weights in VRAM we do not have.
            num_workers=1,
        )

    def _unload(self) -> None:
        self._model = None
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass

    # -- inference --------------------------------------------------------

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = None
    ) -> Transcript:
        if not self._loaded:
            self.load()
        if sample_rate != 16000:
            raise ValueError(
                f"faster-whisper expects 16 kHz audio, got {sample_rate}. "
                "Resample upstream - AudioCapture already does this."
            )

        buffer = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = buffer.size / float(sample_rate)

        t0 = time.perf_counter()
        segments_iter, info = self._model.transcribe(
            buffer,
            language=language or self.language,
            beam_size=self.beam_size,
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous_text,
            vad_filter=self.vad_filter,
            word_timestamps=False,
            hotwords=self.hotwords,
            initial_prompt=self.initial_prompt,
        )
        # faster-whisper returns a generator; decoding happens on iteration, so
        # the timer has to enclose the list() call to measure anything real.
        raw_segments = [
            Segment(
                start=float(s.start),
                end=float(s.end),
                text=s.text.strip(),
                no_speech_prob=float(getattr(s, "no_speech_prob", 0.0) or 0.0),
                avg_logprob=float(getattr(s, "avg_logprob", 0.0) or 0.0),
            )
            for s in segments_iter
        ]
        compute = time.perf_counter() - t0

        segments, dropped = self._filter_repetition(raw_segments)

        # More text than the buffer could hold: a sentence invented from a
        # breath. See plausibility.py for the measurement behind the limit.
        text = " ".join(s.text for s in segments).strip()
        spoken_in = language or self.language
        invented, rate = too_fast(text, duration, spoken_in)
        if invented:
            log.warning("dropping implausible reading: %d syllables in %.2f s "
                        "(%.1f/s) | %r", syllables(text, spoken_in), duration, rate, text[:80])
            dropped.append({"start": 0.0, "end": duration, "text": text,
                            "syllables_per_s": round(rate, 1),
                            "reason": f"{rate:.1f} syllables/s is faster than speech"})
            segments = []
        elif subtitle_credit(text):
            log.warning("dropping a subtitle credit, not speech: %r", text)
            dropped.append({"start": 0.0, "end": duration, "text": text,
                            "reason": "a subtitle credit from Whisper's training data"})
            segments = []
        elif hotword_echo(text, self.hotwords):
            log.warning("dropping the hotword list read back, not speech: %r", text)
            dropped.append({"start": 0.0, "end": duration, "text": text,
                            "reason": "the hotword prompt recited - usually a quiet microphone"})
            segments = []

        return Transcript(
            text=" ".join(s.text for s in segments).strip(),
            segments=segments,
            dropped_segments=dropped,
            language=str(getattr(info, "language", language or self.language)),
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            audio_duration_s=duration,
            compute_s=compute,
            backend=f"{self.name}:{self.model}",
        )

    #: What a loop said once must still look like speech to be kept: enough
    #: words to be a phrase, a segment long enough to hold it, and a rate
    #: people speak at. "Brad Pitt'in annesinden sonra da" once in 2.7 s is
    #: 3.8 syllables/s; "Friulian" once in 4.1 s of breath is 1.
    SALVAGE_MIN_WORDS = 3
    SALVAGE_MIN_SECONDS = 1.0
    SALVAGE_MIN_RATE = 2.0

    def _salvage(self, text: str, span_s: float) -> str | None:
        once = collapse_repeats(text, self.phrase_repeat_limit)
        if len(once.split()) < self.SALVAGE_MIN_WORDS or span_s < self.SALVAGE_MIN_SECONDS:
            return None
        if analyse(once, compression_threshold=self.repetition_threshold,
                   phrase_repeat_limit=self.phrase_repeat_limit).is_repetitive:
            return None
        rate = syllables(once, self.language) / span_s
        if not self.SALVAGE_MIN_RATE <= rate <= 12.0:
            return None
        return once

    def _filter_repetition(
        self, segments: list[Segment]
    ) -> tuple[list[Segment], list[dict[str, Any]]]:
        """Keep what a decoder loop said once; discard loops that said nothing.

        Greedy decoding is deterministic but cannot escape a repetition cycle on
        its own, so the loop is caught after the fact. Until rehearsal the
        whole segment was dropped - and with it a real sentence the loop had
        started from ("Brad Pitt'in annesinden sonra da", 24 times). Now the
        repetition is cut to one and kept if what remains looks like speech;
        a flood of repeated text still never reaches the translator.
        """
        if not self.drop_repetitive:
            return segments, []

        kept: list[Segment] = []
        dropped: list[dict[str, Any]] = []
        for segment in segments:
            report = analyse(
                segment.text,
                compression_threshold=self.repetition_threshold,
                phrase_repeat_limit=self.phrase_repeat_limit,
            )
            if report.is_repetitive:
                once = self._salvage(segment.text, segment.end - segment.start)
                if once is not None:
                    log.warning("collapsed a repetitive segment [%.2f-%.2f] to one: %r (%s)",
                                segment.start, segment.end, once[:80], report.reason)
                    kept.append(Segment(start=segment.start, end=segment.end, text=once,
                                        no_speech_prob=segment.no_speech_prob,
                                        avg_logprob=segment.avg_logprob))
                    continue
                log.warning(
                    "dropping repetitive segment [%.2f-%.2f]: %s | %r",
                    segment.start, segment.end, report.reason, segment.text[:80],
                )
                dropped.append(
                    {"start": segment.start, "end": segment.end,
                     "text": segment.text, **report.as_dict()}
                )
            else:
                kept.append(segment)
        return kept, dropped
