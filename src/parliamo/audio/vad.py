"""Voice activity detection and the segmentation policy built on it.

Two separate concerns, deliberately kept apart:

:class:`SileroVAD`
    Turns a block of audio into a speech probability. Stateful (it is an RNN),
    needs a model, needs a GPU or CPU to run on.

:class:`SpeechSegmenter`
    Decides where sentences begin and end given a stream of those
    probabilities. Pure logic - no model, no audio device - so its behaviour is
    pinned down by deterministic tests rather than by listening to it.

Why the split matters: the segmenter is where end-to-end latency is really
decided. Every millisecond it waits before declaring a sentence finished is a
millisecond the audience waits to hear the translation, and every millisecond
it *doesn't* wait risks cutting a speaker off mid-clause. That trade-off
deserves to be tested, not tuned by ear.

Pre-roll
--------
The segmenter keeps a short ring buffer of blocks from *before* speech was
confirmed. Without it, every segment starts at the moment the detector became
confident - which is a little after the speaker actually started - and the
opening consonant of each sentence is lost. On the reference laptop that would
compound with the vendor DSP's own noise gate, which already clips onsets.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Silero expects exactly this many samples per call at 16 kHz.
SILERO_BLOCK = 512
SILERO_SAMPLE_RATE = 16000


class State(Enum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(slots=True)
class SpeechSegment:
    """One utterance, ready to hand to the recogniser."""

    audio: np.ndarray
    start_s: float
    end_s: float
    sample_rate: int = SILERO_SAMPLE_RATE
    #: True when the segment was cut by ``max_segment_ms`` rather than by
    #: silence - the speaker was still talking, so the text will run on.
    truncated: bool = False
    mean_speech_prob: float = 0.0
    #: A provisional snapshot of an utterance still in progress. Superseded by
    #: the next partial and finally by the committed segment. Safe to display,
    #: never safe to speak - audio cannot be un-said.
    partial: bool = False
    #: Which partial this is within the current utterance, from 1.
    partial_index: int = 0
    #: Closed early, at a short pause, because the utterance had already run
    #: past ``soft_cut_after_ms``. A real pause, unlike ``truncated``, which is
    #: a clock firing mid-phrase.
    soft_cut: bool = False

    @property
    def duration_s(self) -> float:
        return self.audio.size / float(self.sample_rate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "truncated": self.truncated,
            "mean_speech_prob": round(self.mean_speech_prob, 3),
            "partial": self.partial,
            "partial_index": self.partial_index,
            "soft_cut": self.soft_cut,
        }


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


class SileroVAD:
    """Speech probability per 512-sample block at 16 kHz.

    Runs on CPU by default. The model is tiny (~1 MB) and CPU inference costs
    well under a millisecond per block, which is not worth spending VRAM on
    when three larger models are already competing for the card.
    """

    def __init__(self, device: str = "cpu", threshold: float = 0.5) -> None:
        self.device = device
        self.threshold = threshold
        self._model: Any = None
        self._torch: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from silero_vad import load_silero_vad

        self._torch = torch
        self._model = load_silero_vad()
        if self.device != "cpu":
            self._model = self._model.to(self.device)
        log.info("silero VAD loaded on %s (threshold %.2f)", self.device, self.threshold)

    def reset(self) -> None:
        """Clear the recurrent state between utterances.

        Silero carries hidden state across calls. Leaving it set after a
        segment ends makes the start of the next one depend on the end of the
        previous one, which shows up as inconsistent onset detection.
        """
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def probability(self, block: np.ndarray) -> float:
        """Speech probability for exactly one ``SILERO_BLOCK`` of samples."""
        if self._model is None:
            self.load()
        if block.size != SILERO_BLOCK:
            raise ValueError(
                f"silero needs exactly {SILERO_BLOCK} samples at 16 kHz, got {block.size}"
            )
        torch = self._torch
        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
            if self.device != "cpu":
                tensor = tensor.to(self.device)
            return float(self._model(tensor, SILERO_SAMPLE_RATE).item())

    def is_speech(self, block: np.ndarray) -> bool:
        return self.probability(block) >= self.threshold


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SegmenterConfig:
    sample_rate: int = SILERO_SAMPLE_RATE
    block_size: int = SILERO_BLOCK
    #: Silero probability at or above which a block counts as speech.
    #: ``vad.threshold`` in the config; until 2026-09-24 the segmenter used a
    #: fixed 0.5 and the config value reached nothing, so raising it for a
    #: noisy hall would have changed nothing at all.
    speech_threshold: float = 0.5
    #: Speech shorter than this is a cough or a door, not an utterance.
    min_speech_ms: int = 250
    #: Silence this long ends a sentence. The single biggest latency knob:
    #: every millisecond here is added to every segment's delay.
    min_silence_ms: int = 700
    #: Audio kept from before speech was confirmed, and after it ended.
    speech_pad_ms: int = 120
    #: Hard cut so one long sentence cannot stall the whole pipeline.
    max_segment_ms: int = 15000
    #: Emit a provisional copy of the utterance-so-far this often while the
    #: speaker is still talking. 0 disables it.
    #:
    #: This is what lets subtitles appear before the sentence has ended, and it
    #: is close to free: while someone is speaking, the recogniser thread has
    #: nothing to do - it is waiting for a segment to close. Partial
    #: recognition runs in that idle time.
    #:
    #: Not free for the *audience*, though, if it reached the speakers: a
    #: partial can be revised, and speech cannot be un-said. Partials are for
    #: text only.
    partial_interval_ms: int = 0
    #: Do not emit a partial until the utterance is at least this long. Below
    #: it there is not enough audio for a useful hypothesis, and Whisper
    #: hallucinates on very short buffers.
    min_partial_ms: int = 900
    #: Once an utterance has run this long, close it at the next brief pause
    #: instead of waiting for a full ``min_silence_ms``. 0 disables it.
    #:
    #: **Off by default, on the measurement.** The idea was to shorten the tail:
    #: continuous speech averages 5.5 s per segment but reaches 10.1 s, and the
    #: audience waits the whole of that. Measured on 68 s of real reading:
    #:
    #:   setting          segments  mean  max   fragments  WER%   worst wait
    #:   off                 11     5.5  10.1       0      11.41    10.6 s
    #:   8000 / 240 ms       12     5.0   9.2       1      12.08     9.7 s
    #:   6000 / 240 ms       12     5.0   9.2       1      12.08     9.7 s
    #:   4000 / 240 ms       12     5.0   9.2       1      12.08     9.7 s
    #:
    #: Nine tenths of a second off the worst case, for 0.67 WER points and a
    #: fragment under 1.5 s. Not a good trade.
    #:
    #: And the longest segment stays at 9.2 s whatever the setting, which is
    #: the real answer: **that stretch contains no pause at all**, not even a
    #: 240 ms one. The tail is not the segmenter being too patient. It is a
    #: speaker who did not breathe, and no segmentation policy fixes that.
    #:
    #: Kept, defaulted off, for a speaker who pauses differently - a rehearsal
    #: under stage nerves is not this recording.
    soft_cut_after_ms: int = 0
    #: The shorter silence accepted once ``soft_cut_after_ms`` has passed.
    #: Long enough to be a real breath rather than a stop consonant.
    soft_cut_silence_ms: int = 240

    def blocks(self, milliseconds: int) -> int:
        """How many blocks make up *milliseconds*, rounded up."""
        per_block = self.block_size / self.sample_rate * 1000.0
        return max(1, int(np.ceil(milliseconds / per_block)))


class SpeechSegmenter:
    """Turns a stream of (block, speech probability) into utterances.

    Feed it blocks with :meth:`push`; it yields a :class:`SpeechSegment` at the
    moment it decides one has ended. Call :meth:`flush` when the stream stops,
    to emit whatever was still in progress.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        cfg = self.config

        self._pad_blocks = cfg.blocks(cfg.speech_pad_ms)
        self._min_speech_blocks = cfg.blocks(cfg.min_speech_ms)
        self._min_silence_blocks = cfg.blocks(cfg.min_silence_ms)
        self._max_blocks = cfg.blocks(cfg.max_segment_ms)

        # The ring buffer has to cover the confirmation delay *as well as* the
        # padding. Speech is only confirmed after min_speech_ms of it has gone
        # past, so a buffer holding just speech_pad_ms starts the segment
        # (min_speech_ms - speech_pad_ms) *after* the speaker began - clipping
        # the opening consonant, which is the exact failure the pre-roll exists
        # to prevent. With 250 ms confirmation and 120 ms padding that was
        # 130 ms lost from the front of every utterance.
        self._preroll_blocks = self._min_speech_blocks + self._pad_blocks

        self._partial_blocks = (
            cfg.blocks(cfg.partial_interval_ms) if cfg.partial_interval_ms > 0 else 0
        )
        self._min_partial_blocks = cfg.blocks(cfg.min_partial_ms)
        self._soft_cut_blocks = (
            cfg.blocks(cfg.soft_cut_after_ms) if cfg.soft_cut_after_ms > 0 else 0
        )
        self._soft_cut_silence_blocks = cfg.blocks(cfg.soft_cut_silence_ms)

        self._state = State.SILENCE
        self._preroll: deque[np.ndarray] = deque(maxlen=self._preroll_blocks)
        self._buffer: list[np.ndarray] = []
        self._probs: list[float] = []
        self._speech_run = 0
        self._silence_run = 0
        self._blocks_seen = 0
        self._segment_start_block = 0
        self._blocks_at_last_partial = 0
        self._partial_index = 0

    # -- introspection ----------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def in_speech(self) -> bool:
        return self._state is State.SPEECH

    def take_partial(self) -> SpeechSegment | None:
        """A provisional copy of the utterance so far, or None if not due yet.

        Call after :meth:`push`. Returns a segment marked ``partial``, which
        the caller may recognise and display but must never speak: the next
        partial, or the final segment, supersedes it.

        Returns None when partials are disabled, when the speaker is not
        currently talking, when too little audio has accumulated, or when the
        interval since the last one has not elapsed.
        """
        if self._partial_blocks <= 0 or self._state is not State.SPEECH:
            return None
        if len(self._buffer) < self._min_partial_blocks:
            return None
        if len(self._buffer) - self._blocks_at_last_partial < self._partial_blocks:
            return None

        self._blocks_at_last_partial = len(self._buffer)
        self._partial_index += 1
        audio = np.concatenate(self._buffer)
        start = self._time(self._segment_start_block)
        return SpeechSegment(
            audio=audio,
            start_s=start,
            end_s=start + audio.size / self.config.sample_rate,
            sample_rate=self.config.sample_rate,
            truncated=True,   # by construction: the speaker has not stopped
            mean_speech_prob=float(np.mean(self._probs)) if self._probs else 0.0,
            partial=True,
            partial_index=self._partial_index,
        )

    def _time(self, block_index: int) -> float:
        return block_index * self.config.block_size / self.config.sample_rate

    # -- feeding ----------------------------------------------------------

    def push(self, block: np.ndarray, speech_prob: float) -> SpeechSegment | None:
        """Feed one block. Returns a segment when one has just ended."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        is_speech = speech_prob >= self.config.speech_threshold
        self._blocks_seen += 1

        if self._state is State.SILENCE:
            self._preroll.append(block)
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self._min_speech_blocks:
                    # Confirmed. Open a segment that starts *before* this
                    # point, so the onset is not clipped off.
                    self._state = State.SPEECH
                    self._buffer = list(self._preroll)
                    self._probs = [speech_prob]
                    self._silence_run = 0
                    self._segment_start_block = self._blocks_seen - len(self._buffer)
                    self._preroll.clear()
            else:
                self._speech_run = 0
            return None

        # -- in speech --
        self._buffer.append(block)
        self._probs.append(speech_prob)

        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self._min_silence_blocks:
                return self._close(truncated=False)
            # A speaker who has been going for a while gets closed at the next
            # real breath rather than at a full min_silence_ms. Waiting the
            # usual amount is right at the start of an utterance - it is what
            # keeps a clause together - and wrong once the utterance is already
            # long enough that the audience is paying for it.
            if (
                self._soft_cut_blocks
                and len(self._buffer) >= self._soft_cut_blocks
                and self._silence_run >= self._soft_cut_silence_blocks
            ):
                return self._close(truncated=False, soft_cut=True)

        if len(self._buffer) >= self._max_blocks:
            # The speaker has not paused in max_segment_ms. Cut anyway: a
            # segment that never ends is a pipeline that never produces output.
            return self._close(truncated=True)
        return None

    def flush(self) -> SpeechSegment | None:
        """Emit whatever is buffered, e.g. when capture stops."""
        if self._state is State.SPEECH and self._buffer:
            return self._close(truncated=True)
        return None

    def reset(self) -> None:
        self._state = State.SILENCE
        self._preroll.clear()
        self._buffer = []
        self._probs = []
        self._speech_run = 0
        self._silence_run = 0
        self._blocks_at_last_partial = 0
        self._partial_index = 0

    # -- closing ----------------------------------------------------------

    def _close(self, truncated: bool, soft_cut: bool = False) -> SpeechSegment:
        cfg = self.config
        blocks = self._buffer

        # Trim the trailing silence back to speech_pad_ms so the recogniser is
        # not handed half a second of room tone to hallucinate over.
        if not truncated and self._silence_run > self._pad_blocks:
            keep = len(blocks) - (self._silence_run - self._pad_blocks)
            blocks = blocks[: max(1, keep)]

        audio = (
            np.concatenate(blocks)
            if blocks
            else np.zeros(0, dtype=np.float32)
        )
        start_block = self._segment_start_block
        segment = SpeechSegment(
            audio=audio,
            start_s=self._time(start_block),
            end_s=self._time(start_block) + audio.size / cfg.sample_rate,
            sample_rate=cfg.sample_rate,
            truncated=truncated,
            mean_speech_prob=float(np.mean(self._probs)) if self._probs else 0.0,
            soft_cut=soft_cut,
        )

        self._state = State.SILENCE
        self._buffer = []
        self._probs = []
        self._speech_run = 0
        self._silence_run = 0
        self._preroll.clear()
        self._blocks_at_last_partial = 0
        self._partial_index = 0
        return segment


def segment_audio(
    audio: np.ndarray,
    vad: SileroVAD,
    config: SegmenterConfig | None = None,
) -> Iterator[SpeechSegment]:
    """Run the segmenter over a complete buffer. For offline tooling and tests."""
    cfg = config or SegmenterConfig()
    segmenter = SpeechSegmenter(cfg)
    vad.reset()

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    for start in range(0, audio.size - cfg.block_size + 1, cfg.block_size):
        block = audio[start : start + cfg.block_size]
        segment = segmenter.push(block, vad.probability(block))
        if segment is not None:
            yield segment
            continue
        # Partials too, when configured, so an offline replay exercises the
        # same path a live run takes. Callers that do not want them can leave
        # partial_interval_ms at 0, or filter on `segment.partial`.
        partial = segmenter.take_partial()
        if partial is not None:
            yield partial
    tail = segmenter.flush()
    if tail is not None:
        yield tail
