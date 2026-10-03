"""Playback runs on two threads, and the gate depends on them agreeing.

The delivery thread submits a sentence; the sound card's callback drains it and
releases the half-duplex gate when there is nothing left. Every test here
forces one specific interleaving of those two threads, because a race that
happens once in a thousand sentences happens in every hour-long talk.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from parliamo.audio.gate import HalfDuplexGate
from parliamo.audio.playback import AudioPlayback, PlaybackInfo


class _Device:
    label = "Fake Speakers [Test]"
    index = 0
    max_output_channels = 1
    default_samplerate = 16000.0
    likely_bluetooth = False
    name = "Fake Speakers"


def _playback(gate: HalfDuplexGate | None = None) -> AudioPlayback:
    """A playback whose stream is a stand-in: the callback is driven by hand."""
    out = AudioPlayback(sample_rate=16000, gate=gate, channels=1)
    out._stream = object()
    out._info = PlaybackInfo(device=_Device(), sample_rate=16000, channels=1,  # type: ignore[arg-type]
                             latency_ms=0.0, resampling=False)
    return out


def _pull(out: AudioPlayback, frames: int) -> np.ndarray:
    buffer = np.zeros((frames, 1), dtype=np.float32)
    out._callback(buffer, frames, None, None)
    return buffer[:, 0].copy()


def test_the_callback_between_two_lines_of_submit_cannot_open_the_gate() -> None:
    """The feedback race.

    submit() closed the gate, marked playback busy, and only then counted the
    frames. A callback landing in between saw an empty queue and nothing
    pending, decided playback had drained, and released the gate - so the
    sentence that followed played to an open microphone.
    """
    gate = HalfDuplexGate(enabled=True, tail_ms=0)
    out = _playback(gate)

    card: list[threading.Thread] = []

    class CallbackInTheGap(threading.Event):
        def clear(self) -> None:
            super().clear()
            # The sound card's thread runs at the worst moment. Where submit
            # holds the callback's lock, it waits; where it did not, it ran.
            thread = threading.Thread(target=_pull, args=(out, 160))
            thread.start()
            thread.join(timeout=0.2)
            card.append(thread)

    out._idle = CallbackInTheGap()
    out._idle.set()
    out.submit(np.full(1600, 0.5, dtype=np.float32))
    card[0].join(timeout=2)

    assert gate.is_closed, "a sentence is queued and the microphone is open"
    assert out.is_playing


def test_a_sentence_submitted_mid_callback_does_not_jump_the_queue() -> None:
    """The ordering race.

    The callback took the rest of the current sentence and everything behind
    it off the queue, then put it all back. A sentence submitted in between
    landed in front, and the room heard the next sentence before the end of
    this one.
    """
    import queue as queue_module

    out = _playback()
    out.submit(np.full(1000, 0.5, dtype=np.float32))
    delivery: list[threading.Thread] = []

    real_get = out._queue.get_nowait

    def empty_then_submit():
        # The callback finds the queue empty; before it can put the rest of
        # the current sentence back, the delivery thread submits the next one.
        # Where the callback holds the lock, the delivery thread waits.
        try:
            return real_get()
        except queue_module.Empty:
            if not delivery:
                thread = threading.Thread(
                    target=out.submit, args=(np.full(1000, 0.25, dtype=np.float32),))
                thread.start()
                thread.join(timeout=0.2)
                delivery.append(thread)
            raise

    out._queue.get_nowait = empty_then_submit
    played_blocks = []
    for _ in range(20):
        played_blocks.append(_pull(out, 100))
        if delivery:
            delivery[0].join(timeout=2)
    played = np.concatenate(played_blocks)
    first_two = np.flatnonzero(played == 0.25)
    assert (played[:1000] == 0.5).all() and first_two.size and first_two[0] >= 1000, (
        f"the second sentence started at sample {first_two[0] if first_two.size else None}, "
        "inside the first")


def test_flush_stops_the_sentence_already_in_the_callback() -> None:
    out = _playback(HalfDuplexGate(enabled=True, tail_ms=0))
    out.submit(np.full(1000, 0.5, dtype=np.float32))
    _pull(out, 100)
    out.flush()
    assert not _pull(out, 100).any(), "flush must silence the rest of the current sentence"
    assert not out.is_playing


def test_nothing_is_played_after_stop() -> None:
    """A delivery thread that outlives stop() used to reopen the speakers."""
    out = _playback(HalfDuplexGate(enabled=True, tail_ms=0))
    opened = []
    out.start = lambda: opened.append(True)            # would open a real device
    out._stream = None
    out._stopped = True
    assert out.submit(np.ones(1600, dtype=np.float32)) == 0
    assert not opened, "a stopped playback must not reopen the device"


def test_the_gate_tail_does_not_grow_each_time_playback_opens() -> None:
    """Measured latency was added to the tail, then added again on the next open."""
    gate = HalfDuplexGate(enabled=True, tail_ms=250)
    out = AudioPlayback(sample_rate=16000, gate=gate, output_latency_ms=430.0)
    out._info = PlaybackInfo(device=_Device(), sample_rate=16000, channels=1,  # type: ignore[arg-type]
                             latency_ms=3.0, resampling=False)
    out._extend_gate_tail()
    out._extend_gate_tail()
    assert gate.tail_ms == 680


def test_sentences_play_whole_and_in_order() -> None:
    out = _playback(HalfDuplexGate(enabled=True, tail_ms=0))
    out.submit(np.full(250, 0.5, dtype=np.float32))
    out.submit(np.full(250, 0.25, dtype=np.float32))
    played = np.concatenate([_pull(out, 64) for _ in range(10)])
    assert (played[:250] == 0.5).all() and (played[250:500] == 0.25).all()
    assert not played[500:].any()
    time.sleep(0.01)
    assert not out.is_playing and out.gate.is_open


# ---------------------------------------------------------------------------
# a start that fails must not leave anything behind
# ---------------------------------------------------------------------------


def test_a_failed_start_unloads_what_it_loaded() -> None:
    """The speakers would not open; the models stayed on the GPU.

    The factory loads ~2.5 GB of models and translator.start() then fails -
    "could not open any audio output", seen on this laptop. The translator and
    its teardown went out of scope with the delivery thread still running, and
    the next Start loaded everything a second time beside them.
    """
    from parliamo.ui.controller import PipelineController

    calls: list[str] = []

    class Translator:
        def start(self) -> None:
            calls.append("start")
            raise RuntimeError("could not open any audio output after 3 attempts")

        def stop(self) -> None:
            calls.append("stop")

    controller = PipelineController(factory=lambda: (Translator(), lambda: calls.append("teardown")))
    controller.start()
    controller._worker.join(timeout=5)
    assert controller.status == "failed"
    assert calls == ["start", "stop", "teardown"], calls


def test_a_factory_that_fails_part_way_still_reports() -> None:
    from parliamo.ui.controller import PipelineController

    def factory():
        raise RuntimeError("CUDA out of memory")

    controller = PipelineController(factory=factory)
    controller.start()
    controller._worker.join(timeout=5)
    assert controller.status == "failed" and "out of memory" in controller.detail


def test_a_sentence_made_at_another_rate_plays_for_its_own_length() -> None:
    # 48 kHz from RVC into a 16 kHz stream: one second must stay one second,
    # not become three at a third of the pitch.
    out = _playback()
    frames = out.submit(np.full(48000, 0.25, dtype=np.float32), sample_rate=48000)
    assert abs(frames - 16000) <= 2


def test_a_sentence_at_the_stream_rate_is_not_touched() -> None:
    out = _playback()
    tone = np.sin(np.linspace(0, 200 * np.pi, 16000)).astype(np.float32) * 0.5
    assert out.submit(tone, sample_rate=16000) == 16000
    np.testing.assert_array_equal(_pull(out, 16000), tone)


def test_the_device_rate_is_reached_from_the_sentence_rate_directly() -> None:
    # A device that cannot open 16 kHz runs at 48; a 22.05 kHz sentence goes
    # straight there rather than through the requested rate.
    out = _playback()
    out._resample_to = 48000
    frames = out.submit(np.full(22050, 0.25, dtype=np.float32), sample_rate=22050)
    assert abs(frames - 48000) <= 2


# -- underflows --------------------------------------------------------------


class _Status:
    def __init__(self, underflow: bool) -> None:
        self.output_underflow = underflow

    def __bool__(self) -> bool:
        return self.output_underflow


def _gap(out: AudioPlayback) -> None:
    out._callback(np.zeros((160, 1), dtype=np.float32), 160, None, _Status(True))


def test_a_gap_while_a_sentence_plays_reaches_the_run_log(caplog) -> None:
    """At rehearsal the room heard the translation choppy and the log said
    nothing: underflows went to debug level. Now they are counted in the
    callback and reported as a warning from the caller's thread."""
    import logging

    out = _playback()
    out.submit(np.ones(1600, dtype=np.float32))
    for _ in range(3):
        _gap(out)
    assert out.underflows == 3
    with caplog.at_level(logging.WARNING, logger="parliamo.audio.playback"):
        assert out.report_underflows(force=True) == 3
    assert "ran dry 3 time(s)" in caplog.text
    assert out.report_underflows(force=True) == 0, "each gap is reported once"


def test_starved_silence_is_not_a_gap() -> None:
    """Nothing playing, nothing heard: an idle stream's underflows are noise."""
    out = _playback()
    _gap(out)
    assert out.underflows == 0


def test_underflow_warnings_are_rate_limited(caplog) -> None:
    """A starving stream underflows dozens of times a second."""
    import logging

    out = _playback()
    out.submit(np.ones(16000, dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger="parliamo.audio.playback"):
        _gap(out)
        assert out.report_underflows() == 1
        _gap(out)
        assert out.report_underflows() == 0, "within ten seconds of the last report"
        out._underflow_reported_at -= AudioPlayback.UNDERFLOW_REPORT_S
        assert out.report_underflows() == 1
    assert caplog.text.count("ran dry") == 2


def test_the_warning_names_mme_as_the_cause(caplog) -> None:
    import logging

    out = _playback()
    out._info.device = type("D", (), {  # type: ignore[union-attr]
        "label": "Speakers (Dante USB I/O Module) [MME]", "hostapi_name": "MME"})()
    out.submit(np.ones(1600, dtype=np.float32))
    _gap(out)
    with caplog.at_level(logging.WARNING, logger="parliamo.audio.playback"):
        out.report_underflows(force=True)
    assert "Dante" in caplog.text and "WASAPI" in caplog.text


def test_the_submit_after_a_gap_reports_it(caplog) -> None:
    """The delivery thread submits every sentence, so it carries the report."""
    import logging

    out = _playback()
    out.submit(np.ones(1600, dtype=np.float32))
    _gap(out)
    with caplog.at_level(logging.WARNING, logger="parliamo.audio.playback"):
        out.submit(np.ones(1600, dtype=np.float32))
    assert "ran dry 1 time(s)" in caplog.text


def test_stop_reports_what_is_left() -> None:
    out = _playback()
    out.submit(np.ones(1600, dtype=np.float32))
    _gap(out)
    out._underflow_reported_at = 1e18  # a report just went out: not due yet
    out._stream = None
    out.stop()
    assert out._underflows_reported == 1, "stop reports regardless of the rate limit"
