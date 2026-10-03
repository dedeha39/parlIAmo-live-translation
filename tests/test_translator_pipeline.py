"""The full delivery path, with every model faked.

What matters here is the orchestration: that one stage failing degrades the
output instead of stopping the show, that the newest sentence wins when the
pipeline falls behind, and that the latency breakdown adds up. Whether the
models are any good is answered by their own benchmarks.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from parliamo.asr.base import Transcript
from parliamo.audio.vad import SpeechSegment
from parliamo.mt.base import MTBackend
from parliamo.pipeline.transcriber import LiveTranscriber, TranscriptEvent
from parliamo.pipeline.translator import DeliveryEvent, LiveTranslator
from parliamo.tts.base import TTSBackend
from parliamo.tts.conversion import Conversion, ConversionError


class FakeMT(MTBackend):
    name = "fake-mt"

    def __init__(self, **kw):
        super().__init__(model="fake", source_lang="tr", target_lang="it", **kw)
        self.fail = False
        self.calls = 0

    def _load(self): pass
    def _unload(self): pass

    def _translate(self, text, source_lang, target_lang):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated translation failure")
        return f"[{target_lang}] {text}"


class FakeTTS(TTSBackend):
    name = "fake-tts"
    sample_rate = 24000

    def __init__(self, **kw):
        super().__init__(model="fake", **kw)
        self.fail = False
        self.calls = 0

    def _load(self): pass
    def _unload(self): pass

    def _synthesise(self, text, language, voice):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated synthesis failure")
        return np.ones(self.sample_rate, dtype=np.float32) * 0.1


class FakeConverter:
    timeout = 10.0

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0
        self.options: list[dict] = []

    def convert(self, audio, sample_rate, reference_path, diffusion_steps=None, **options):
        self.calls += 1
        self.options.append(options)
        if self.fail:
            raise ConversionError("service unreachable")
        return Conversion(
            audio=np.asarray(audio, dtype=np.float32) * 0.5,
            sample_rate=22050, compute_s=0.4, round_trip_s=0.42, diffusion_steps=4,
        )


class FakePlayback:
    def __init__(self):
        self.submitted: list[np.ndarray] = []
        self.rates: list[int | None] = []
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def submit(self, audio, sample_rate=None):
        self.submitted.append(np.asarray(audio))
        self.rates.append(sample_rate)
        return audio.size


def _event(index: int = 1, text: str = "merhaba dünya") -> TranscriptEvent:
    return TranscriptEvent(
        index=index, text=text,
        segment=SpeechSegment(audio=np.zeros(16000, dtype=np.float32),
                              start_s=0.0, end_s=1.0),
        transcript=Transcript(text=text),
        commit_wait_s=0.7, queue_wait_s=0.05, asr_s=0.25,
    )


def _build(max_queue_depth: int = 8):
    mt, tts, playback = FakeMT(), FakeTTS(), FakePlayback()
    transcriber = LiveTranscriber(backend=None)  # type: ignore[arg-type]
    translator = LiveTranslator(
        transcriber, mt, tts, playback=playback, target_lang="it",
        max_queue_depth=max_queue_depth,
    )
    return translator, mt, tts, playback


@pytest.fixture
def parts():
    # A queue deep enough that normal tests do not race the consumer. An
    # earlier version used depth 2 here and dropped a sentence under load,
    # which made an unrelated assertion fail intermittently. Backpressure gets
    # its own narrow queue in the test that is actually about backpressure.
    return _build(max_queue_depth=8)


def _drain(translator: LiveTranslator, events: list, count: int, expected: int,
           timeout: float = 5.0) -> None:
    """Start the delivery thread first, then feed it."""
    translator._running.set()
    thread = threading.Thread(target=translator._deliver_loop, daemon=True)
    thread.start()
    try:
        for i in range(count):
            translator._enqueue(_event(i + 1))
            time.sleep(0.01)
        deadline = time.monotonic() + timeout
        while len(events) < expected and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        translator._running.clear()
        translator._queue.put(None)
        thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_sentence_goes_all_the_way_to_audio(parts) -> None:
    translator, mt, tts, playback = parts
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=3, expected=3)

    assert len(events) == 3
    assert all(e.translated_text.startswith("[it]") for e in events)
    assert all(e.spoken for e in events)
    assert len(playback.submitted) == 3
    assert mt.calls == 3 and tts.calls == 3


def test_latency_breakdown_sums_to_the_total(parts) -> None:
    translator, _, _, _ = parts
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append
    _drain(translator, events, count=1, expected=1)

    e = events[0]
    assert e.total_lag_s == pytest.approx(
        e.recognition_lag_s + e.translation_s + e.synthesis_s + e.conversion_s
    )
    # Recognition lag carries the upstream commit wait, which dominates.
    assert e.recognition_lag_s >= 0.7


# ---------------------------------------------------------------------------
# degrade, do not stop
# ---------------------------------------------------------------------------


def test_translation_failure_does_not_stop_the_next_sentence(parts) -> None:
    translator, mt, _, _ = parts
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append
    mt.fail = True

    _drain(translator, events, count=2, expected=2)

    assert len(events) == 2, "the loop stopped after a failure"
    assert all("translation" in e.error for e in events)
    assert all(not e.spoken for e in events)
    assert translator.stats.translation_failures == 2


def test_synthesis_failure_still_reports_the_translation(parts) -> None:
    """Subtitles must survive a dead synthesiser - they are the fallback."""
    translator, _, tts, _ = parts
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append
    tts.fail = True

    _drain(translator, events, count=1, expected=1)

    assert events[0].translated_text.startswith("[it]")
    assert not events[0].spoken
    assert "synthesis" in events[0].error


def test_conversion_failure_falls_back_to_the_generic_voice(parts) -> None:
    """Speaking in the wrong voice beats saying nothing at all."""
    translator, _, _, playback = parts
    translator.converter = FakeConverter(fail=True)
    translator.reference_voice = "ref.wav"
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)

    assert events[0].spoken is True
    assert events[0].voice == "default"
    assert "conversion" in events[0].error
    assert len(playback.submitted) == 1
    assert translator.stats.conversion_failures == 1


def test_callback_exception_does_not_stop_delivery(parts) -> None:
    translator, _, _, _ = parts
    seen: list[int] = []

    def bad(event: DeliveryEvent) -> None:
        seen.append(event.index)
        raise ValueError("callback blew up")

    translator.on_delivery = bad
    _drain(translator, [], count=3, expected=0, timeout=2.0)
    assert seen == [1, 2, 3]


# ---------------------------------------------------------------------------
# voice conversion
# ---------------------------------------------------------------------------


def test_conversion_is_applied_and_labelled(parts) -> None:
    translator, _, _, playback = parts
    converter = FakeConverter()
    translator.converter = converter
    translator.reference_voice = "ref.wav"
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)

    assert converter.calls == 1
    assert events[0].voice == "cloned"
    assert events[0].conversion_s > 0
    np.testing.assert_allclose(playback.submitted[0], 0.05, rtol=1e-5)


def test_converted_audio_is_played_at_the_rate_the_service_made_it(parts) -> None:
    # RVC answers at 48 kHz, Seed-VC at 22.05; the synthesiser speaks at 24.
    # Handed over without its rate, RVC's sentence played at half speed an
    # octave down on stage - and every saved file sounded right.
    translator, _, tts, playback = parts
    translator.converter = FakeConverter()
    translator.reference_voice = "ref.wav"
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)

    assert events[0].sample_rate == 22050
    assert playback.rates == [22050]


def test_generic_audio_is_played_at_the_synthesiser_rate(parts) -> None:
    translator, _, tts, playback = parts
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)

    assert playback.rates == [tts.sample_rate]


def test_no_reference_voice_means_no_conversion_attempt(parts) -> None:
    translator, _, _, _ = parts
    converter = FakeConverter()
    translator.converter = converter
    translator.reference_voice = None
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)
    assert converter.calls == 0
    assert events[0].voice == "default"


# ---------------------------------------------------------------------------
# backpressure and reporting
# ---------------------------------------------------------------------------


def test_full_queue_keeps_the_newest_sentence() -> None:
    """With no consumer running, the queue must bound itself and count drops."""
    translator, _, _, _ = _build(max_queue_depth=2)
    for i in range(5):
        translator._enqueue(_event(i + 1))
    assert translator.stats.dropped_backlog == 3
    assert translator._queue.qsize() == 2


def test_summary_reports_percentiles(parts) -> None:
    translator, _, _, _ = parts
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append
    _drain(translator, events, count=4, expected=4)

    summary = translator.stats.summary()
    assert summary["delivered"] == 4
    assert {"mean", "p50", "p95", "max"} <= set(summary["lag_s"])


def test_summary_without_deliveries_is_safe(parts) -> None:
    translator, _, _, _ = parts
    assert translator.stats.summary()["delivered"] == 0
    assert "lag_s" not in translator.stats.summary()


def test_speak_false_produces_text_only(parts) -> None:
    """Subtitle-only mode: no synthesis, no audio, translation still delivered."""
    translator, _, tts, playback = parts
    translator.speak = False
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)

    assert events[0].translated_text.startswith("[it]")
    assert not events[0].spoken
    assert tts.calls == 0
    assert playback.submitted == []


class _RecordingConverter(FakeConverter):
    def __init__(self):
        super().__init__()
        self.received: list[np.ndarray] = []

    def convert(self, audio, sample_rate, reference_path, diffusion_steps=None, **options):
        self.received.append(np.asarray(audio))
        return super().convert(audio, sample_rate, reference_path, diffusion_steps, **options)


def test_the_conversion_service_is_warmed_on_speech_not_silence(parts) -> None:
    """Warmed on silence, a fresh Seed-VC took ~0.15 s longer on the first real sentence."""
    translator, _mt, tts, _pb = parts
    converter = _RecordingConverter()
    translator.converter, translator.reference_voice = converter, "ref.wav"
    translator._warm_conversion()
    assert tts.calls == 1, "the warm-up phrase was synthesised"
    assert np.abs(converter.received[0]).max() > 0, "the service was warmed on sound"


def test_the_service_is_told_what_the_sentence_says(parts) -> None:
    # OmniVoice speaks the sentence itself; without the text it has nothing
    # to say. The translation and its language go with every sentence.
    translator, _, _, _ = parts
    converter = FakeConverter()
    translator.converter, translator.reference_voice = converter, "ref.wav"
    events: list[DeliveryEvent] = []
    translator.on_delivery = events.append

    _drain(translator, events, count=1, expected=1)

    assert converter.options[0]["text"] == events[0].translated_text
    assert converter.options[0]["language"] == "it"


def test_the_warm_up_says_its_phrase_and_waits_long_enough(parts) -> None:
    # On OmniVoice the first request for a reference also prepares it on the
    # CPU - 5-15 s for a volunteer. Timing out at the sentence timeout would
    # leave that running and the first real sentence queued behind it.
    translator, _, _, _ = parts
    converter = FakeConverter()
    translator.converter, translator.reference_voice = converter, "ref.wav"
    translator._warm_conversion()
    options = converter.options[0]
    assert options["text"] == translator.WARM_PHRASES["it"]
    assert options["language"] == "it"
    assert options["timeout"] >= translator.WARM_TIMEOUT_S > converter.timeout


def test_a_warm_up_whose_synthesis_fails_still_warms_on_silence(parts) -> None:
    translator, _mt, tts, _pb = parts
    tts.fail = True
    converter = _RecordingConverter()
    translator.converter, translator.reference_voice = converter, "ref.wav"
    translator._warm_conversion()
    assert len(converter.received) == 1 and not converter.received[0].any()


# ---------------------------------------------------------------------------
# a conversion service that hangs, or answers with nonsense
# ---------------------------------------------------------------------------


class _HangingConverter(FakeConverter):
    def __init__(self):
        super().__init__()

    def convert(self, audio, sample_rate, reference_path, diffusion_steps=None, **options):
        from parliamo.tts.conversion import ConversionTimeout

        self.calls += 1
        raise ConversionTimeout("no answer in 8 s")


def test_a_hung_service_costs_one_timeout_not_one_per_sentence(parts, monkeypatch) -> None:
    """Every sentence waited the full timeout - 60 s each - for a service that had hung."""
    import parliamo.pipeline.translator as module

    translator, _mt, _tts, playback = parts
    converter = _HangingConverter()
    translator.converter, translator.reference_voice = converter, "ref.wav"
    now = [1000.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    first = translator.deliver(_event(1, "bir"))
    second = translator.deliver(_event(2, "iki"))
    assert converter.calls == 1, "the second sentence must not wait for a service that just hung"
    assert first.voice == "default" and second.voice == "default"
    assert "paused" in (second.error or ""), second.error
    assert len(playback.submitted) == 2, "both sentences are still spoken, in the generic voice"

    now[0] += translator.CONVERSION_PAUSE_S + 1
    translator.deliver(_event(3, "üç"))
    assert converter.calls == 2, "after the pause the service is tried again"


def test_a_garbled_answer_is_a_conversion_failure_not_a_lost_sentence() -> None:
    """A ValueError from the wire format escaped as a plain exception, and the sentence was dropped."""
    import socket
    import threading

    import numpy as np

    from parliamo.tts.conversion import ConversionError, VoiceConverter

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def answer_garbage() -> None:
        conn, _ = server.accept()
        conn.recv(1 << 16)
        conn.sendall((5).to_bytes(4, "big") + b"notjs")   # a header that is not JSON
        conn.close()

    threading.Thread(target=answer_garbage, daemon=True).start()
    converter = VoiceConverter(port=port, timeout=2.0)
    try:
        converter.convert(np.zeros(160, np.float32), 16000, "ref.wav")
    except ConversionError:
        pass
    else:
        raise AssertionError("expected ConversionError")
    finally:
        server.close()
