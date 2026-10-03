"""Microphone to Turkish text, live.

Three threads, because the stages have very different timing:

* **PortAudio's callback thread** delivers 32 ms blocks and must never block.
* **The segmenter thread** runs VAD on every block (under 5 ms each) and
  decides where utterances end. Cheap, must keep up with real time exactly.
* **The recogniser thread** transcribes whole utterances (200–800 ms each).
  Slow and bursty, so it cannot sit in the same thread as the segmenter or it
  would drop audio for the whole time it was working.

Backpressure
------------
If recognition falls behind, the newest audio is the audio worth keeping: a
speaker who has moved on is not helped by a translation of what they said ten
seconds ago. The segment queue therefore drops its *oldest* entry when full,
and counts the drop rather than hiding it.

What "latency" means here
-------------------------
The number that matters is not how long recognition took, but how long the
audience waits after the speaker stops. That is::

    min_silence_ms      the segmenter waiting to be sure the sentence ended
  + queue wait          recognition still busy with the previous utterance
  + recognition time

The first term usually dominates, and it is a policy choice rather than a
hardware limit. :class:`TranscriptEvent` reports all three separately so the
distinction stays visible.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..asr.base import ASRBackend, Transcript
from ..audio.capture import AudioCapture
from ..audio.gate import HalfDuplexGate
from ..audio.reblock import Reblocker
from ..audio.vad import (
    SILERO_BLOCK,
    SegmenterConfig,
    SileroVAD,
    SpeechSegment,
    SpeechSegmenter,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptEvent:
    """One recognised utterance, with the latency breakdown that produced it."""

    index: int
    text: str
    segment: SpeechSegment
    transcript: Transcript
    #: Wall-clock seconds the segmenter spent waiting for silence before it
    #: would commit. A policy cost, not a compute cost.
    commit_wait_s: float
    #: Seconds the segment sat in the queue because recognition was busy.
    queue_wait_s: float
    #: Seconds spent inside the recogniser.
    asr_s: float
    #: A provisional reading of an utterance still being spoken. Superseded by
    #: the next partial and finally by the committed transcript. Display it;
    #: never speak it.
    partial: bool = False

    @property
    def total_lag_s(self) -> float:
        """Delay from the speaker falling silent to the text being available."""
        # A partial has not waited for silence - that is the whole point of it,
        # so charging the commit wait would misreport what the audience saw.
        commit = 0.0 if self.partial else self.commit_wait_s
        return commit + self.queue_wait_s + self.asr_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "partial": self.partial,
            "text": self.text,
            "audio_s": round(self.segment.duration_s, 2),
            "commit_wait_s": round(self.commit_wait_s, 3),
            "queue_wait_s": round(self.queue_wait_s, 3),
            "asr_s": round(self.asr_s, 3),
            "total_lag_s": round(self.total_lag_s, 3),
            "rtf": round(self.transcript.rtf, 3),
            "truncated": self.segment.truncated,
            "dropped_segments": len(self.transcript.dropped_segments),
        }


@dataclass(slots=True)
class TranscriberStats:
    segments_detected: int = 0
    segments_transcribed: int = 0
    segments_dropped_backlog: int = 0
    segments_empty: int = 0
    partials_emitted: int = 0
    partials_skipped_busy: int = 0
    repetition_loops_caught: int = 0
    #: Readings with more syllables than the audio could hold - a sentence
    #: invented from a breath (asr/plausibility.py).
    implausible_readings_caught: int = 0
    blocks_seen: int = 0
    lags: list[float] = field(default_factory=list)
    #: Sample accounting from the re-framer; samples_in must equal
    #: samples_out + pending, or audio is being lost somewhere.
    reblock: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "segments_detected": self.segments_detected,
            "segments_transcribed": self.segments_transcribed,
            "segments_dropped_backlog": self.segments_dropped_backlog,
            "segments_empty": self.segments_empty,
            "partials_emitted": self.partials_emitted,
            "partials_skipped_busy": self.partials_skipped_busy,
            "repetition_loops_caught": self.repetition_loops_caught,
            "implausible_readings_caught": self.implausible_readings_caught,
            "blocks_seen": self.blocks_seen,
            "reblock": self.reblock,
        }
        if self.lags:
            ordered = sorted(self.lags)
            out["lag_s"] = {
                "mean": round(sum(ordered) / len(ordered), 3),
                "p50": round(ordered[len(ordered) // 2], 3),
                "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
                "max": round(ordered[-1], 3),
            }
        return out


class LiveTranscriber:
    """Runs capture → VAD → recognition and hands finished utterances to a callback."""

    def __init__(
        self,
        backend: ASRBackend,
        *,
        device: int | str | None = None,
        vad: SileroVAD | None = None,
        segmenter_config: SegmenterConfig | None = None,
        gate: HalfDuplexGate | None = None,
        on_transcript: Callable[[TranscriptEvent], None] | None = None,
        max_queue_depth: int = 4,
    ) -> None:
        self.backend = backend
        self.device = device
        self.vad = vad or SileroVAD(device="cpu")
        self.segmenter_config = segmenter_config or SegmenterConfig()
        self.gate = gate
        self.on_transcript = on_transcript
        self.stats = TranscriberStats()

        self._segments: queue.Queue[tuple[SpeechSegment, float] | None] = queue.Queue(
            maxsize=max_queue_depth
        )
        self._capture: AudioCapture | None = None
        self._threads: list[threading.Thread] = []
        self._running = threading.Event()
        self._index = 0
        self._commit_wait_s = self.segmenter_config.min_silence_ms / 1000.0
        self._paused = False

    # -- lifecycle --------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if self._running.is_set():
            return

        self.vad.load()
        if not self.backend.loaded:
            self.backend.load()
        # Warm up before the microphone opens: a cold first inference can take
        # seconds, and on stage that reads as a hang.
        warm = self.backend.warmup(seconds=3.0)
        log.info("recogniser warm in %.0f ms", warm * 1000)

        self._capture = AudioCapture(
            device=self.device,
            sample_rate=self.segmenter_config.sample_rate,
            block_ms=int(SILERO_BLOCK / self.segmenter_config.sample_rate * 1000),
            gate=self.gate,
        )
        self._capture.paused = self._paused
        info = self._capture.start()
        log.info("listening on %s", info.device.label)

        self._running.set()
        self._threads = [
            threading.Thread(target=self._segment_loop, name="segmenter", daemon=True),
            threading.Thread(target=self._recognise_loop, name="recogniser", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 10.0) -> TranscriberStats:
        """Stop capture, drain what is in flight, and return the run statistics."""
        if not self._running.is_set():
            return self.stats
        self._running.clear()

        if self._capture is not None:
            self._capture.stop()
            self._capture = None

        # Sentinel so the recogniser thread wakes up and exits. A full queue is
        # fine: the thread also polls _running, so it will notice regardless.
        with contextlib.suppress(queue.Full):
            self._segments.put_nowait(None)

        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads = []
        log.info("transcriber stopped: %s", self.stats.summary())
        return self.stats

    def __enter__(self) -> LiveTranscriber:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        """Stop listening, or listen again. Kept across a restart of capture.

        What was said before the pause is still translated: the segment loop
        commits it as soon as the blocks captured before the press are read,
        rather than waiting for a silence that would only come after resume.
        """
        self._paused = bool(paused)
        if self._capture is not None:
            self._capture.paused = self._paused

    # -- threads ----------------------------------------------------------

    def _segment_loop(self) -> None:
        """VAD every block; push completed utterances to the recogniser."""
        segmenter = SpeechSegmenter(self.segmenter_config)
        self.vad.reset()
        capture = self._capture

        # The resampler emits ragged chunks (1100 samples on the reference
        # machine) while the VAD needs exactly 512. Re-frame rather than trim:
        # trimming threw away 53% of the microphone input.
        reblocker = Reblocker(SILERO_BLOCK)

        committed_for_pause = False
        while self._running.is_set() and capture is not None:
            if not self._paused:
                committed_for_pause = False
            chunk = capture.read(timeout=0.25)
            if chunk is None:
                # Paused, and every block from before the press has been read.
                # Left in the segmenter, the last sentence would wait for a
                # silence that only arrives after resume - and be joined to
                # whatever is said first then.
                if self._paused and not committed_for_pause:
                    self._commit_buffered(segmenter, reblocker)
                    segmenter.reset()
                    self.vad.reset()
                    committed_for_pause = True
                continue
            for block in reblocker.push(chunk):
                self.stats.blocks_seen += 1
                try:
                    prob = self.vad.probability(block)
                except Exception as exc:  # pragma: no cover
                    log.warning("VAD failed on a block: %s", exc)
                    continue
                segment = segmenter.push(block, prob)
                if segment is not None:
                    self._enqueue(segment)
                    continue

                # While the speaker is still talking the recogniser has nothing
                # to do, so a provisional reading costs idle GPU time rather
                # than time the committed sentence needed. Only when it really
                # is idle: a partial that queued behind real work would delay
                # the sentence it was supposed to preview.
                partial = segmenter.take_partial()
                if partial is not None:
                    self._enqueue_partial(partial)

        self._commit_buffered(segmenter, reblocker)
        self.stats.reblock = reblocker.accounting()

    def _commit_buffered(self, segmenter: SpeechSegmenter, reblocker: Reblocker) -> None:
        """Send on whatever speech is buffered, without waiting for silence."""
        tail_block = reblocker.flush()
        if tail_block is not None:
            segment = segmenter.push(tail_block, self.vad.probability(tail_block))
            if segment is not None:
                self._enqueue(segment)
        tail = segmenter.flush()
        if tail is not None:
            self._enqueue(tail)

    def _enqueue_partial(self, segment: SpeechSegment) -> None:
        """Queue a provisional segment, but only if nothing real is waiting.

        A partial is a preview. If the recogniser is already busy or has a
        committed segment queued, the preview would delay the thing it
        previews, which inverts the point of it. Dropping it is correct and is
        counted rather than hidden.
        """
        if not self._segments.empty():
            self.stats.partials_skipped_busy += 1
            return
        try:
            self._segments.put_nowait((segment, time.perf_counter()))
            self.stats.partials_emitted += 1
        except queue.Full:  # pragma: no cover - empty() was just checked
            self.stats.partials_skipped_busy += 1

    def _enqueue(self, segment: SpeechSegment) -> None:
        self.stats.segments_detected += 1
        item = (segment, time.perf_counter())
        try:
            self._segments.put_nowait(item)
        except queue.Full:
            # Recognition is behind. Keep the newest audio, not the oldest.
            try:
                self._segments.get_nowait()
                self._segments.put_nowait(item)
            except (queue.Empty, queue.Full):  # pragma: no cover
                pass
            self.stats.segments_dropped_backlog += 1
            log.warning(
                "recogniser behind: dropped an utterance (%d so far)",
                self.stats.segments_dropped_backlog,
            )

    def _recognise_loop(self) -> None:
        while True:
            try:
                item = self._segments.get(timeout=0.25)
            except queue.Empty:
                if not self._running.is_set():
                    return
                continue
            if item is None:
                return

            segment, enqueued_at = item
            queue_wait = time.perf_counter() - enqueued_at

            t0 = time.perf_counter()
            try:
                transcript = self.backend.transcribe(segment.audio, segment.sample_rate)
            except Exception as exc:
                log.exception("recognition failed: %s", exc)
                continue
            asr_s = time.perf_counter() - t0

            text = transcript.text.strip()
            dropped = transcript.dropped_segments
            invented = sum(1 for d in dropped if "syllables_per_s" in d)
            self.stats.implausible_readings_caught += invented

            if segment.partial:
                # A partial does not advance the sentence counter: it is a
                # preview of the sentence that is coming, and numbering it
                # separately would make the subtitle jump.
                if not text:
                    continue
                event = TranscriptEvent(
                    index=self._index + 1,
                    text=text,
                    segment=segment,
                    transcript=transcript,
                    commit_wait_s=self._commit_wait_s,
                    queue_wait_s=queue_wait,
                    asr_s=asr_s,
                    partial=True,
                )
                self._emit(event)
                continue

            self.stats.segments_transcribed += 1
            self.stats.repetition_loops_caught += len(dropped) - invented

            if not text:
                self.stats.segments_empty += 1
                # A segment the VAD thought was speech and the recogniser made
                # nothing of. From the speaker's side that is "a sentence went
                # missing", and the log line saying why - a repetition loop, an
                # empty decode - does not say what was actually said. Keep the
                # audio, so someone can listen and settle whether it was a
                # cough, a word the model looped on, or a sentence lost.
                self._keep_dropped(segment, transcript)
                continue

            self._index += 1
            event = TranscriptEvent(
                index=self._index,
                text=text,
                segment=segment,
                transcript=transcript,
                commit_wait_s=self._commit_wait_s,
                queue_wait_s=queue_wait,
                asr_s=asr_s,
            )
            self.stats.lags.append(event.total_lag_s)

            self._emit(event)

    #: Where segments that produced no text are written, or None to keep
    #: nothing. Set by the application; tooling that replays files leaves it.
    keep_dropped_dir: Any = None

    def _keep_dropped(self, segment: SpeechSegment, transcript: Transcript) -> None:
        if self.keep_dropped_dir is None:
            return
        try:
            import json
            from pathlib import Path

            import soundfile as sf

            folder = Path(self.keep_dropped_dir)
            folder.mkdir(parents=True, exist_ok=True)
            self._dropped_count += 1
            stem = folder / f"dropped-{self._dropped_count:03d}"
            sf.write(str(stem) + ".wav", segment.audio, segment.sample_rate)
            (stem.with_suffix(".json")).write_text(json.dumps({
                "start_s": round(segment.start_s, 2),
                "end_s": round(segment.end_s, 2),
                "duration_s": round(segment.duration_s, 2),
                "mean_speech_prob": round(segment.mean_speech_prob, 3),
                "dropped_segments": transcript.dropped_segments,
                "raw_text": transcript.text,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            log.warning("kept a segment that produced no text: %s.wav (%.1f s)",
                        stem.name, segment.duration_s)
        except Exception:  # pragma: no cover - a diagnostic must not cost a sentence
            log.debug("could not keep the dropped segment", exc_info=True)

    _dropped_count: int = 0

    def _emit(self, event: TranscriptEvent) -> None:
        if self.on_transcript is None:
            return
        try:
            self.on_transcript(event)
        except Exception:  # pragma: no cover - a bad callback must not kill the loop
            log.exception("on_transcript callback raised")

