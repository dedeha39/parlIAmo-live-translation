"""The full path: microphone in, translated speech out.

Builds on :class:`~parliamo.pipeline.transcriber.LiveTranscriber` by adding a
fourth thread that takes recognised Turkish and produces spoken Italian.

Why a fourth thread rather than doing it in the recogniser's
-----------------------------------------------------------
Recognition and delivery have different jobs. The recogniser must be ready for
the next utterance the moment it finishes one; the delivery stage spends most of
its time waiting for audio to finish playing. Putting them together would mean
the microphone is deaf for the whole duration of every spoken translation -
which is precisely what the half-duplex gate already does *deliberately* and for
a bounded time, and which would otherwise happen accidentally and for longer.

The gate is what keeps this honest: while the speakers are producing sound, the
capture callback discards what it hears, so the system cannot translate its own
output.

Latency, end to end
-------------------
Measured on the reference machine, mean per sentence over the 68 s reference
recording (2026-09-23, RVC; docs/adr/0013)::

    0.50  commit wait      the segmenter waiting to be sure a sentence ended
    0.36  recognition      large-v3-turbo, including queueing
    0.23  translation      NLLB-600M (0.41 with 1.3B)
    0.19  synthesis        Kokoro
    0.44  voice conversion RVC, including IPC (Seed-VC: 0.93)
    ----
    1.72  s

The commit wait is a policy choice; the rest is compute. (Until 2026-09-23 the
documents added the commit wait a second time and read 0.5 s higher.)
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audio.playback import AudioPlayback
from ..mt.base import MTBackend
from ..mt.sentences import split_sentences
from ..tts.base import TTSBackend
from ..tts.conversion import ConversionError, ConversionTimeout, VoiceConverter
from .transcriber import LiveTranscriber, TranscriptEvent

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DeliveryEvent:
    """One sentence carried all the way from speech to speech."""

    index: int
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    #: Delay from the speaker falling silent to the text being recognised.
    recognition_lag_s: float
    translation_s: float
    synthesis_s: float
    conversion_s: float
    audio_s: float
    spoken: bool
    #: The two halves of ``recognition_lag_s``, carried separately because they
    #: are not the same kind of number. The commit wait is a *setting* - the
    #: silence the segmenter waits out before deciding a sentence ended - and it
    #: is the only term in the whole budget anyone can choose. Recognition is
    #: what the model then cost. Summed into one figure, the operator cannot see
    #: which of the two a slow sentence came from.
    commit_wait_s: float = 0.0
    asr_s: float = 0.0
    #: Where in the source audio the recognised segment ended, in seconds.
    #: For a file replay this is the only honest clock: the pipeline runs
    #: faster than real time, so wall-clock says nothing about when the
    #: room would have heard a sentence. Comparing this across delivery
    #: policies says exactly that.
    audio_end_s: float = 0.0
    voice: str = "default"
    error: str = ""
    #: A provisional translation of a sentence still being spoken. Shown as a
    #: subtitle and then replaced; never synthesised, because audio cannot be
    #: withdrawn once it has been played.
    partial: bool = False
    #: The audio that was (or would have been) played, kept so an offline
    #: replay can write it to disk for a listening check. Deliberately absent
    #: from :meth:`as_dict` - a JSON report is for numbers, not waveforms.
    audio: np.ndarray | None = field(default=None, repr=False)
    sample_rate: int = 0

    @property
    def total_lag_s(self) -> float:
        """Delay from the speaker falling silent to audio starting."""
        return (
            self.recognition_lag_s
            + self.translation_s
            + self.synthesis_s
            + self.conversion_s
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "partial": self.partial,
            "source": self.source_text,
            "translation": self.translated_text,
            "direction": f"{self.source_lang}->{self.target_lang}",
            "recognition_lag_s": round(self.recognition_lag_s, 3),
            "commit_wait_s": round(self.commit_wait_s, 3),
            "asr_s": round(self.asr_s, 3),
            "translation_s": round(self.translation_s, 3),
            "synthesis_s": round(self.synthesis_s, 3),
            "conversion_s": round(self.conversion_s, 3),
            "total_lag_s": round(self.total_lag_s, 3),
            "audio_s": round(self.audio_s, 3),
            "audio_end_s": round(self.audio_end_s, 2),
            "spoken": self.spoken,
            "voice": self.voice,
            "error": self.error,
        }


@dataclass(slots=True)
class TranslatorStats:
    delivered: int = 0
    partials_delivered: int = 0
    dropped_backlog: int = 0
    translation_failures: int = 0
    synthesis_failures: int = 0
    conversion_failures: int = 0
    #: Sentences spoken generic because the queue was backed up. Not a
    #: failure: the voice service was fine, there was no time for it.
    conversion_skipped: int = 0
    lags: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "delivered": self.delivered,
            "partials_delivered": self.partials_delivered,
            "dropped_backlog": self.dropped_backlog,
            "translation_failures": self.translation_failures,
            "synthesis_failures": self.synthesis_failures,
            "conversion_failures": self.conversion_failures,
            "conversion_skipped": self.conversion_skipped,
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


class LiveTranslator:
    """Microphone to translated speech, on one machine, with no network."""

    def __init__(
        self,
        transcriber: LiveTranscriber,
        translator: MTBackend,
        synthesiser: TTSBackend,
        *,
        playback: AudioPlayback | None = None,
        converter: VoiceConverter | None = None,
        reference_voice: str | None = None,
        target_lang: str = "it",
        on_delivery: Callable[[DeliveryEvent], None] | None = None,
        max_queue_depth: int = 3,
        speak: bool = True,
    ) -> None:
        self.transcriber = transcriber
        self.translator = translator
        self.synthesiser = synthesiser
        self.playback = playback
        self.converter = converter
        self.reference_voice = reference_voice
        self.target_lang = target_lang
        self.on_delivery = on_delivery
        self.speak = speak
        #: Operator panic control; see :meth:`set_muted`.
        self.muted = False
        #: Speak each sentence as soon as the provisional text shows it has
        #: ended, rather than when the speaker pauses. See
        #: :meth:`_deliver_streaming` for what that buys and what it risks.
        self.stream_sentences = False
        #: Sentences already spoken from partials of the utterance in
        #: progress, normalised, so the committed text does not repeat them.
        self._spoken_early: list[str] = []
        #: Complete sentences seen once in a partial; spoken on the second
        #: sighting. Cleared with the ledger at each commit.
        self._seen_once: set[str] = set()
        #: "off" | "sentence" | "chunk". The bool above is kept for the callers
        #: that only know about sentences; `stream_mode` is the whole setting.
        self.stream_mode = "off"
        #: Chunk mode: the previous partial's words, and how many words of the
        #: utterance have already been spoken. See :meth:`_deliver_chunks`.
        self._prev_words: list[str] = []
        self._chunk_spoken: list[str] = []
        self.stats = TranslatorStats()

        self._queue: queue.Queue[TranscriptEvent | None] = queue.Queue(maxsize=max_queue_depth)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._index = 0

        # The transcriber hands finished utterances here rather than to a
        # printer, so recognition never waits on delivery.
        transcriber.on_transcript = self._enqueue

    # -- lifecycle --------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def warm(self) -> None:
        """Load and warm every delivery stage, without opening the microphone.

        Split out of :meth:`start` so offline replay can warm the same stages
        the live path warms. Warm-up is not optional bookkeeping: the first
        conversion cost 2.28 s against 0.98 s once warm, and an unwarmed first
        sentence made the end-to-end figure read 3.31 s instead of 2.40 s.
        """
        if not self.translator.loaded:
            self.translator.load()
        self.translator.warmup()
        if not self.synthesiser.loaded:
            self.synthesiser.load()
        self.synthesiser.warmup()

        # Warm the conversion service too. Warming the stage you were thinking
        # about and not the one downstream is a mistake this project has
        # already made once.
        if self.converter is not None and self.reference_voice:
            self._warm_conversion()

    def start(self) -> None:
        if self._running.is_set():
            return

        self.warm()

        if self.playback is not None:
            self.playback.start()

        self._running.set()
        self._thread = threading.Thread(target=self._deliver_loop, name="delivery", daemon=True)
        self._thread.start()
        self.transcriber.start()
        log.info("translator running: %s -> %s", self.translator.source_lang, self.target_lang)

    def set_reference_voice(self, path: str | None, consent: str = "") -> None:
        """Change whose voice the system speaks in, without restarting.

        This is what makes the volunteer demonstration possible. Someone comes
        up, gives fifteen seconds, and the next sentence is in their voice - a
        restart at that moment would end the demonstration, because the point
        is that it takes fifteen seconds and not a coffee break.

        Consent is checked here as well as at startup. A voice swapped in
        mid-talk is exactly the moment the check is most likely to be skipped,
        so it is not skippable: this raises before anything is loaded.

        Passing None returns to the generic synthesiser voice.
        """
        if path is None:
            self.reference_voice = None
            log.warning("reference voice cleared; speaking in the generic voice")
            return

        from ..paths import resolve
        from ..tts.base import VoiceProfile

        resolved = str(resolve(path))
        VoiceProfile(name="live", reference_path=resolved, consent=consent).validate()

        self.reference_voice = resolved
        log.warning("reference voice is now %s", resolved)
        # The first conversion against a new reference is slow - the embedding
        # has not been computed - and on stage that lands on the volunteer's
        # first sentence, which is the one everyone is listening to.
        if self.converter is not None:
            self._warm_conversion()

    def set_muted(self, muted: bool) -> None:
        """Silence output immediately, or restore it.

        The operator's panic control. It flushes what is *already playing* as
        well as stopping what comes next - a mute that only applies to future
        sentences is not a panic control, because the sentence you want to stop
        is the one coming out of the speakers now.

        Recognition and translation carry on, so subtitles keep working and the
        room can still follow. Only the audio stops.
        """
        self.muted = muted
        if muted and self.playback is not None:
            self.playback.flush()
        log.warning("output %s", "muted" if muted else "unmuted")

    def set_paused(self, paused: bool) -> None:
        """Stop listening, or listen again - for a video, applause, questions.

        The half-duplex gate covers the system's own voice and nothing else
        the room's speakers play. Output is untouched: a sentence already on
        its way is still spoken. Mute is the control for that.
        """
        self.transcriber.set_paused(paused)
        log.warning("listening %s", "PAUSED" if paused else "resumed")

    #: Spoken once at start to warm the conversion service. Speech, not
    #: silence: on a fresh Seed-VC the first real sentence took 1.12/0.94 s
    #: after a silent warm-up and 0.88/0.87 s after a spoken one - and with a
    #: volunteer that first sentence is the one the room is listening to. On
    #: RVC the difference was 0.05 s, inside the noise.
    WARM_PHRASE = ("Oggi parliamo di come funziona la voce. "
                   "Il computer traduce quello che dico, frase per frase.")
    #: The warm-up phrase in the language being spoken - it is also what the
    #: synthesiser's pitch is measured on, so it must be that language's voice,
    #: and long and level. A short greeting read high: "Buongiorno a tutti,
    #: cominciamo." (2.5 s) put if_sara at -10.2 semitones from the presenter,
    #: these two plain sentences (6.1 s) at -7.5 - the -8 found by ear.
    WARM_PHRASES = {
        "it": WARM_PHRASE,
        "es": "Hoy hablamos de cómo funciona la voz. El ordenador traduce lo que digo, "
              "frase por frase.",
        "de": "Heute sprechen wir darüber, wie die Stimme funktioniert. Der Computer "
              "übersetzt, was ich sage, Satz für Satz.",
        "tr": "Bugün sesin nasıl çalıştığını konuşuyoruz. Bilgisayar söylediklerimi "
              "cümle cümle çeviriyor.",
        "en": "Today we talk about how a voice works. The computer translates what I say, "
              "sentence by sentence.",
        "fr": "Aujourd'hui nous parlons de la voix. L'ordinateur traduit ce que je dis, "
              "phrase par phrase.",
    }

    #: The presenter's median pitch; set from the config. With it, every
    #: sentence goes to the voice service with the shift from the
    #: synthesiser's voice to the presenter's, measured at warm-up.
    presenter_f0_hz: float | None = None
    _pitch_shift: float | None = None

    #: OmniVoice's speaking rate for every sentence; None copies the
    #: reference's pace. Set from the config, and from the page while running.
    voice_speed: float | None = None

    def _conversion_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self._pitch_shift is not None:
            options["pitch"] = self._pitch_shift
        if self.voice_speed:
            options["options"] = {"speed": float(self.voice_speed)}
        return options

    def _measure_pitch_shift(self, audio: np.ndarray, rate: int) -> None:
        """Measure the synthesiser's voice once and derive the shift to the presenter."""
        if not self.presenter_f0_hz or not audio.size or not audio.any():
            return
        from ..audio.pitch import median_f0, semitones

        try:
            voice_hz = median_f0(audio, rate)
        except Exception as exc:  # pragma: no cover - a measurement must not stop Start
            log.warning("could not measure the synthesiser's pitch: %s", exc)
            return
        if voice_hz is None:
            return
        self._pitch_shift = round(semitones(voice_hz, self.presenter_f0_hz), 1)
        log.info("synthesiser voice at %.0f Hz, presenter at %.0f Hz: shifting %+.1f semitones",
                 voice_hz, self.presenter_f0_hz, self._pitch_shift)

    #: The warm-up's own timeout; see :meth:`_warm_conversion`.
    WARM_TIMEOUT_S = 60.0

    def _warm_conversion(self) -> None:
        assert self.converter is not None and self.reference_voice
        rate = self.synthesiser.sample_rate
        phrase = self.WARM_PHRASES.get(self.target_lang, self.WARM_PHRASE)
        try:
            speech = self.synthesiser.speak(phrase, language=self.target_lang)
            audio, rate = np.asarray(speech.audio, dtype=np.float32), speech.sample_rate
        except Exception as exc:  # the warm-up must never be why Start failed
            log.debug("warm-up synthesis failed, warming on silence: %s", exc)
            audio = np.zeros(int(0.5 * rate), dtype=np.float32)
        self._measure_pitch_shift(audio, rate)
        try:
            t0 = time.perf_counter()
            # Longer than a sentence may take: on OmniVoice the first request
            # for a reference also transcribes and encodes it, on the CPU -
            # 5-15 s for a volunteer's recording. Timing out here would leave
            # that work running and the next real sentence waiting behind it.
            self.converter.convert(audio, rate, self.reference_voice,
                                   text=phrase, language=self.target_lang,
                                   timeout=max(getattr(self.converter, "timeout", 0.0), self.WARM_TIMEOUT_S),
                                   **self._conversion_options())
            log.info("conversion service warm in %.0f ms", (time.perf_counter() - t0) * 1000)
        except ConversionError as exc:
            log.warning("could not warm the conversion service: %s", exc)

    def stop(self, timeout: float = 15.0) -> TranslatorStats:
        if not self._running.is_set():
            return self.stats
        self.transcriber.stop()
        self._running.clear()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        if self.playback is not None:
            self.playback.stop()
        log.info("translator stopped: %s", self.stats.summary())
        return self.stats

    def __enter__(self) -> LiveTranslator:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- delivery ---------------------------------------------------------

    def _enqueue(self, event: TranscriptEvent) -> None:
        if event.partial:
            # A preview is worth nothing if it costs the sentence it previews.
            # Committed text always wins the queue slot, so a full queue simply
            # drops the partial instead of displacing anything.
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(event)
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Same rule as upstream: the newest sentence is the one worth
            # saying. A translation of what the speaker said ten seconds ago
            # helps nobody and makes the backlog worse.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except (queue.Empty, queue.Full):  # pragma: no cover
                pass
            self.stats.dropped_backlog += 1
            log.warning(
                "delivery behind: dropped a sentence (%d so far)",
                self.stats.dropped_backlog,
            )

    def _deliver_loop(self) -> None:
        while True:
            try:
                event = self._queue.get(timeout=0.25)
            except queue.Empty:
                if not self._running.is_set():
                    return
                continue
            if event is None:
                return
            try:
                self._deliver(event)
            except Exception:  # pragma: no cover - one bad sentence must not stop the show
                log.exception("delivery failed for sentence %d", event.index)

    def deliver(self, event: TranscriptEvent) -> DeliveryEvent | None:
        """Run one recognised utterance through the delivery path, synchronously.

        The live path reaches :meth:`_deliver` through a queue and a thread.
        Offline replay (``live_translate.py --file``) reaches it through here,
        so a recording exercises *the same* translate-synthesise-convert-speak
        code rather than a parallel implementation of it that can drift.

        Returns the delivery, or None if it raised - one bad sentence must not
        stop a replay any more than it stops a rehearsal.
        """
        try:
            return self._deliver(event)
        except Exception:
            log.exception("delivery failed for sentence %d", event.index)
            return None

    def _deliver_partial(self, event: TranscriptEvent) -> DeliveryEvent | None:
        """Translate a provisional sentence for the subtitle, and stop there.

        Deliberately does not synthesise. A partial is a guess about a sentence
        the speaker has not finished, and Turkish puts the verb - and the
        negation suffix - at the end: "bu teknik bir mesele **değil**" reverses
        everything before it. Displaying a guess that is corrected a second
        later is fine, because the audience sees the correction. Speaking one
        is not, because they cannot un-hear it.
        """
        try:
            t0 = time.perf_counter()
            result = self.translator.translate(event.text, target_lang=self.target_lang)
            translation_s = time.perf_counter() - t0
        except Exception as exc:
            log.debug("partial translation failed, dropping it: %s", exc)
            return None
        if not result.text:
            return None

        delivery = DeliveryEvent(
            index=event.index,
            source_text=event.text,
            translated_text=result.text,
            source_lang=self.translator.source_lang,
            target_lang=self.target_lang,
            recognition_lag_s=event.total_lag_s,
            commit_wait_s=event.commit_wait_s,
            asr_s=event.asr_s,
            audio_end_s=float(getattr(event.segment, 'end_s', 0.0) or 0.0),
            translation_s=translation_s,
            synthesis_s=0.0,
            conversion_s=0.0,
            audio_s=0.0,
            spoken=False,
            partial=True,
        )
        self.stats.partials_delivered += 1
        if self.on_delivery is not None:
            try:
                self.on_delivery(delivery)
            except Exception:  # pragma: no cover
                log.exception("on_delivery callback raised")
        return delivery

    # -- speaking before the pause --------------------------------------

    @staticmethod
    def _normalise(sentence: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", "", sentence.casefold()).split())

    def _already_spoken(self, sentence: str) -> bool:
        """Was this sentence, or a near-identical earlier reading of it, spoken?

        The recogniser revises. A sentence spoken from a partial as "hava
        güzel" may commit as "hava çok güzel"; speaking both would say the
        thing twice. Anything at 80% similarity, or with the spoken version
        as a prefix, counts as the same sentence. That threshold is where a
        revision stops being the same sentence and starts being a different
        one, and it was chosen by reading revisions, not by measuring - which
        is why streaming is an option and not the default.
        """
        wanted = self._normalise(sentence)
        if not wanted:
            return True
        for spoken in self._spoken_early:
            same = (wanted.startswith(spoken) or spoken.startswith(wanted)
                    or difflib.SequenceMatcher(None, spoken, wanted).ratio() >= 0.8)
            if not same:
                continue
            # The one revision that must be spoken again: the committed
            # sentence carries a negation the early reading did not. "Bu
            # teknik bir mesele" spoken early and "... değil" committed is
            # not the same sentence - it is the opposite one, and the room
            # has already heard the wrong half. Saying the correction is
            # the least bad outcome left.
            return not self._negation_added(spoken, wanted)
        return False

    #: Clause-final negations Turkish carries as separate words. Verb suffixes
    #: (-me/-ma) need morphology to spot and are not attempted; these catch
    #: the cases the streaming risk was named for.
    _NEGATIONS = frozenset({"değil", "degil", "yok", "hayır", "hayir", "asla", "hiç", "hic"})

    @classmethod
    def _negation_added(cls, spoken: str, committed: str) -> bool:
        before = set(spoken.split()) & cls._NEGATIONS
        after = set(committed.split()) & cls._NEGATIONS
        return bool(after - before)

    @staticmethod
    def _complete_sentences(text: str) -> tuple[list[str], str]:
        """Sentences that have ended, and whatever is still being said.

        Only a sentence closed by a full stop, question or exclamation mark
        counts as complete. An ellipsis does not: it is what the recogniser
        writes when the audio ran out mid-word, and on the first measurement
        it was the mark on both of the fragments that should never have been
        spoken. The last piece is the one the speaker is in the middle of
        unless it is closed too.
        """
        pieces = split_sentences(text)
        if not pieces:
            return [], ""
        done = [p for p in pieces[:-1] if p.strip() and not p.rstrip().endswith(("…", "..."))]
        last = pieces[-1].strip()
        if last and last[-1] in ".!?" and not last.endswith("..."):
            done.append(last)
            return done, ""
        return done, last

    def _stable(self, sentence: str) -> bool:
        """Has this sentence been read the same way twice in a row?

        Local agreement, the standard policy in simultaneous translation:
        nothing is committed until two consecutive hypotheses agree on it.
        Partials arrive every 800 ms; a real finished sentence reads the same
        in the next one, a hallucinated early reading does not. Measured on
        the reference recording before this existed, streaming spoke 24
        sentences where the speaker said 11, and the extras were confident
        misreadings of the first few words.
        """
        key = self._normalise(sentence)
        seen = self._seen_once
        if key in seen:
            return True
        seen.add(key)
        return False

    def _deliver_streaming(self, event: TranscriptEvent) -> DeliveryEvent | None:
        """Speak sentences the moment they end, without waiting for a pause.

        The default path waits for the speaker to fall silent before saying
        anything, so a ten-second run of speech is heard ten seconds late in
        one piece. Here each sentence goes out as soon as the provisional
        text shows its full stop: the first sentence of that run arrives a
        few seconds in, and the rest follow.

        What it costs, and the presenter has chosen to pay it: a sentence is
        spoken on the strength of where the recogniser put a full stop, and
        Turkish carries the verb and the negation last. A stop placed one
        word early speaks the sentence without its ending. The subtitle would
        have been corrected; the audio cannot be.
        """
        done, tail = self._complete_sentences(event.text)
        last: DeliveryEvent | None = None
        for sentence in done:
            if self._already_spoken(sentence) or not self._stable(sentence):
                continue
            last = self._deliver_text(sentence, event)
            self._spoken_early.append(self._normalise(sentence))
        # The rest is still a guess and stays a subtitle.
        if tail:
            preview = TranscriptEvent(
                index=event.index, text=tail, segment=event.segment,
                transcript=event.transcript, commit_wait_s=event.commit_wait_s,
                queue_wait_s=event.queue_wait_s, asr_s=event.asr_s, partial=True,
            )
            return self._deliver_partial(preview) or last
        return last

    #: Chunk mode speaks once this many stable words have accumulated, or
    #: sooner at a punctuation mark. Four is roughly a clause in Turkish;
    #: fewer and the translator sees word pairs, more and the point is lost.
    CHUNK_WORDS = 4

    #: Queued sentences at which conversion is skipped to keep up. One is
    #: normal - the next sentence arriving while this one converts. Two
    #: means the room is already a sentence behind.
    CONVERSION_BACKLOG_LIMIT = 2

    #: After the conversion service times out, how long every sentence goes
    #: out in the generic voice before it is tried again.
    CONVERSION_PAUSE_S = 30.0
    _conversion_paused_until: float = 0.0

    def _deliver_chunks(self, event: TranscriptEvent) -> DeliveryEvent | None:
        """Speak the utterance in pieces as the words settle - the "buffer".

        The most aggressive policy here, and the one the presenter asked for
        by name. A word counts as settled when two consecutive partials put it
        in the same place (word-level local agreement, as in
        Whisper-Streaming). Settled words that have not been spoken are
        translated and spoken as soon as there are CHUNK_WORDS of them or a
        punctuation mark closes them.

        What it costs is not subtle. The translator sees a clause with no
        verb, because Turkish keeps the verb for the end, and produces the
        best Italian it can for a clause with no verb. The negation arrives
        in a later chunk, after the room has heard the affirmative. This mode
        exists to be tried and heard, with those words next to the switch.
        """
        words = event.text.split()
        prev = self._prev_words
        agreed = 0
        # Compared without punctuation: the recogniser flickers between
        # "istiyorum" and "istiyorum." from one reading to the next, and a
        # full stop is not a different word. Sentence mode already ignored it.
        for a, b in zip(prev, words, strict=False):
            if self._word_key(a) == self._word_key(b):
                agreed += 1
            else:
                break
        self._prev_words = words

        start = self._spoken_end(self._chunk_spoken, words)
        stable_new = words[start:agreed]
        last: DeliveryEvent | None = None
        closes = any(w[-1] in ".!?,;:" for w in stable_new if w)
        if len(stable_new) >= self.CHUNK_WORDS or (stable_new and closes):
            last = self._deliver_text(" ".join(stable_new), event)
            # This reading of what was said, not the first one: the next
            # reading grows out of this one and aligns against it best.
            self._chunk_spoken = words[:agreed]
            start = agreed

        tail = " ".join(words[start:])
        if tail:
            preview = TranscriptEvent(
                index=event.index, text=tail, segment=event.segment,
                transcript=event.transcript, commit_wait_s=event.commit_wait_s,
                queue_wait_s=event.queue_wait_s, asr_s=event.asr_s, partial=True,
            )
            return self._deliver_partial(preview) or last
        return last

    #: How far apart, in words, a spoken word and its match in a new reading
    #: may be. A merge or split moves the boundary by one; a match further
    #: away is more likely the same short word ("ve", "bu") said again later.
    ALIGN_SLACK = 3

    @staticmethod
    def _word_key(word: str) -> str:
        return word.casefold().strip(".,;:!?…\"'«»()-")

    @classmethod
    def _spoken_end(cls, spoken: list[str], words: list[str]) -> int:
        """Where, in the reading *words*, the words already spoken end.

        The chunk ledger used to be a count - five words spoken, so the rest
        starts at word six. Every reading is a fresh decode of the whole
        utterance, and the commit especially may segment it differently:
        "üç te" in the partials, "üçte" in the commit. A count then skips a
        word the room never heard, or repeats one it did. Here the spoken
        words are aligned to the new reading instead; what alignment cannot
        match - a joined or split word, a rewritten one - is covered by its
        length in characters, which is what survives re-segmentation.
        """
        if not spoken:
            return 0
        spoken_keys = [cls._word_key(w) for w in spoken]
        keys = [cls._word_key(w) for w in words]
        matcher = difflib.SequenceMatcher(None, spoken_keys, keys, autojunk=False)
        end_spoken = end_words = 0
        for a, b, size in matcher.get_matching_blocks():
            if size and abs(b - a) <= cls.ALIGN_SLACK:
                end_spoken, end_words = a + size, b + size
        left = sum(len(k) for k in spoken_keys[end_spoken:])
        j = end_words
        while j < len(words) and left >= len(keys[j]) / 2:
            left -= len(keys[j])
            j += 1
        return j

    def reset_streaming(self) -> None:
        """Forget what the streaming policies have spoken of this utterance."""
        self._spoken_early.clear()
        self._seen_once.clear()
        self._prev_words = []
        self._chunk_spoken = []

    def _track_utterance(self, event: TranscriptEvent) -> None:
        """Reset streaming state when a new utterance begins.

        The state was reset only at a commit, and a commit does not always
        come: the recogniser drops a segment that decodes to nothing or to a
        repetition loop, and emits no event. The next utterance then inherited
        the last one's ledger. In chunk mode that meant "5 words already
        spoken" applied to a sentence nobody had heard, and a whole sentence
        - "Toplantı yarın saat üçte başlayacak." in the reproduction - was
        never spoken at all. In sentence mode a sentence the speaker said
        twice was swallowed the second time.

        Partials and the commit of one utterance share its start time
        (verified on the reference recording: 11 commits, 73 partials, no
        mismatch), so a change in start time is a new utterance.
        """
        segment = getattr(event, "segment", None)
        start = getattr(segment, "start_s", None)
        if start is None:
            return
        key = round(float(start), 3)
        if key != self._utterance_key:
            if self._utterance_key is not None:
                self.reset_streaming()
            self._utterance_key = key

    _utterance_key: float | None = None

    def _deliver(self, event: TranscriptEvent) -> DeliveryEvent | None:
        self._track_utterance(event)
        mode = self.stream_mode if self.stream_mode != "off" else (
            "sentence" if self.stream_sentences else "off")
        if event.partial:
            if mode == "chunk":
                return self._deliver_chunks(event)
            if mode == "sentence":
                return self._deliver_streaming(event)
            return self._deliver_partial(event)

        # Committed. Chunk mode: whatever is beyond the spoken words goes out.
        if mode == "chunk" or self._chunk_spoken:
            words = event.text.split()
            remaining = " ".join(words[self._spoken_end(self._chunk_spoken, words):])
            self._prev_words, self._chunk_spoken = [], []
            if not remaining.strip():
                return None
            return self._deliver_text(remaining, event)

        # Whatever sentence streaming already said is not said again; the
        # rest of the utterance - usually its last sentence - goes out now.
        if self._spoken_early:
            remaining = [s for s in split_sentences(event.text)
                         if not self._already_spoken(s)]
            self._spoken_early.clear()
            self._seen_once.clear()
            if not remaining:
                return None
            return self._deliver_text(" ".join(remaining), event)
        self._seen_once.clear()
        return self._deliver_text(event.text, event)

    def _deliver_text(self, text: str, event: TranscriptEvent) -> DeliveryEvent | None:
        """Translate, synthesise, convert and speak *text*, timed against *event*."""
        self._index += 1
        translation_s = synthesis_s = conversion_s = 0.0
        audio_s = 0.0
        spoken = False
        error = ""
        translated = ""
        audio: np.ndarray | None = None
        rate = self.synthesiser.sample_rate

        # -- translate ----------------------------------------------------
        try:
            t0 = time.perf_counter()
            result = self.translator.translate(text, target_lang=self.target_lang)
            translation_s = time.perf_counter() - t0
            translated = result.text
        except Exception as exc:
            self.stats.translation_failures += 1
            error = f"translation: {type(exc).__name__}: {exc}"
            log.exception("translation failed")

        # -- synthesise ---------------------------------------------------
        # Nothing is synthesised while muted. It would only be thrown away, and
        # skipping it hands the GPU back to recognition and translation, which
        # are still running because the subtitles still matter.
        if translated and self.speak and not self.muted:
            try:
                t0 = time.perf_counter()
                speech = self.synthesiser.speak(translated, language=self.target_lang)
                synthesis_s = time.perf_counter() - t0
                audio, rate = speech.audio, speech.sample_rate
                audio_s = speech.duration_s
            except Exception as exc:
                self.stats.synthesis_failures += 1
                error = f"synthesis: {type(exc).__name__}: {exc}"
                log.exception("synthesis failed")

        # -- convert to the cloned voice ----------------------------------
        voice = "default"
        # Conversion is the largest term in the budget - 0.93 s mean, up to
        # 1.5 s - and it runs one sentence at a time on this thread. When
        # sentences arrive faster than that (streaming, or a speaker who does
        # not pause) the queue fills and the oldest sentence is thrown away.
        # Seen live: three sentences dropped in eight seconds and a lag of six
        # on the screen. A sentence in the generic voice is worth more than a
        # sentence not said, which is already the rule when conversion fails;
        # this applies it before the queue overflows rather than after.
        behind = self._queue.qsize() >= self.CONVERSION_BACKLOG_LIMIT
        paused_for = self._conversion_paused_until - time.monotonic()
        wants_voice = audio is not None and self.converter is not None and self.reference_voice
        if wants_voice and paused_for > 0:
            self.stats.conversion_skipped += 1
            error = (f"conversion paused: the service timed out; generic voice for "
                     f"another {paused_for:.0f} s")
        elif behind and wants_voice:
            self.stats.conversion_skipped += 1
            error = "conversion skipped: delivery behind, spoken generic to keep up"
            log.warning("delivery %d sentence(s) behind; speaking this one generic",
                        self._queue.qsize())
        elif wants_voice:
            try:
                converted = self.converter.convert(audio, rate, self.reference_voice,
                                                   text=translated, language=self.target_lang,
                                                   **self._conversion_options())
                conversion_s = converted.round_trip_s
                audio, rate = converted.audio, converted.sample_rate
                audio_s = converted.duration_s
                voice = "cloned"
            except ConversionTimeout as exc:
                # A hung service costs the whole timeout on every sentence
                # sent to it. One is enough: the next sentences go out in the
                # generic voice at once, and the service is tried again later.
                self.stats.conversion_failures += 1
                self._conversion_paused_until = time.monotonic() + self.CONVERSION_PAUSE_S
                error = f"conversion: {exc}; generic voice for {self.CONVERSION_PAUSE_S:.0f} s"
                log.warning("voice conversion timed out; pausing it for %.0f s: %s",
                            self.CONVERSION_PAUSE_S, exc)
            except ConversionError as exc:
                # Speaking in the generic voice is better than saying nothing.
                self.stats.conversion_failures += 1
                error = f"conversion: {exc}"
                log.warning("voice conversion failed, speaking in the default voice: %s", exc)

        # -- speak --------------------------------------------------------
        if audio is not None and audio.size and self.playback is not None and not self.muted:
            # At the rate it was made: converted audio is not the synthesiser's.
            self.playback.submit(audio, rate)
            spoken = True

        delivery = DeliveryEvent(
            index=self._index,
            source_text=text,
            translated_text=translated,
            source_lang=self.translator.source_lang,
            target_lang=self.target_lang,
            recognition_lag_s=event.total_lag_s,
            commit_wait_s=event.commit_wait_s,
            asr_s=event.asr_s,
            audio_end_s=float(getattr(event.segment, 'end_s', 0.0) or 0.0),
            translation_s=translation_s,
            synthesis_s=synthesis_s,
            conversion_s=conversion_s,
            audio_s=audio_s,
            spoken=spoken,
            voice=voice,
            error=error,
            audio=audio,
            sample_rate=rate,
        )
        self.stats.delivered += 1
        self.stats.lags.append(delivery.total_lag_s)

        if self.on_delivery is not None:
            try:
                self.on_delivery(delivery)
            except Exception:  # pragma: no cover
                log.exception("on_delivery callback raised")
        return delivery
