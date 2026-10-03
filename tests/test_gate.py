"""The half-duplex gate.

These tests are pure logic - no audio hardware - because the gate is the
component whose failure mode is "the demo dies on stage", and that deserves
coverage that runs everywhere, every time.
"""

from __future__ import annotations

import threading
import time

from parliamo.audio.gate import HalfDuplexGate


def test_gate_starts_open() -> None:
    assert HalfDuplexGate().is_open


def test_close_then_release_with_tail() -> None:
    gate = HalfDuplexGate(tail_ms=120)
    gate.close()
    assert gate.is_closed

    gate.release()
    # Still closed: the tail covers room reverberation after the last sample.
    assert gate.is_closed
    # The bound needs a tolerance. remaining_ms() is derived from two separate
    # time.monotonic() reads, so with no measurable delay between release and
    # this call it lands a fraction of a nanosecond above the tail
    # (120.00000000000455 was the observed value). Asserting <= 120 exactly
    # makes the test pass or fail on scheduler luck.
    assert 0 < gate.remaining_ms() <= 120 + 1e-3

    time.sleep(0.15)
    assert gate.is_open


def test_zero_tail_reopens_immediately() -> None:
    gate = HalfDuplexGate(tail_ms=0)
    gate.close()
    gate.release()
    assert gate.is_open


def test_nested_holds_require_matching_releases() -> None:
    """Two overlapping utterances must not reopen the mic when the first ends."""
    gate = HalfDuplexGate(tail_ms=0)
    gate.close()
    gate.close()
    gate.release()
    assert gate.is_closed, "gate reopened while a second hold was still active"
    gate.release()
    assert gate.is_open


def test_release_without_close_is_harmless() -> None:
    gate = HalfDuplexGate(tail_ms=0)
    gate.release()
    gate.release()
    assert gate.is_open


def test_context_manager_holds_and_releases() -> None:
    gate = HalfDuplexGate(tail_ms=0)
    with gate:
        assert gate.is_closed
    assert gate.is_open


def test_disabled_gate_is_always_open() -> None:
    gate = HalfDuplexGate(enabled=False, tail_ms=5000)
    gate.close()
    assert gate.is_open
    assert gate.remaining_ms() == 0.0


def test_force_open_clears_all_holds() -> None:
    gate = HalfDuplexGate(tail_ms=10_000)
    gate.close()
    gate.close()
    gate.force_open()
    assert gate.is_open


def test_close_count_tracks_utterances_not_holds() -> None:
    gate = HalfDuplexGate(tail_ms=0)
    gate.close()
    gate.close()
    gate.release()
    gate.release()
    gate.close()
    gate.release()
    assert gate.stats.closes == 2


def test_muted_block_accounting() -> None:
    gate = HalfDuplexGate(tail_ms=0)
    gate.note_muted_block(16000, 16000, had_speech=True)
    gate.note_muted_block(8000, 16000, had_speech=False)
    stats = gate.stats.as_dict()
    assert stats["muted_frames"] == 24000
    assert stats["muted_seconds"] == 1.5
    assert stats["speech_during_mute_blocks"] == 1


def test_gate_is_thread_safe_under_contention() -> None:
    """Capture and playback threads hit this object concurrently, every segment."""
    gate = HalfDuplexGate(tail_ms=0)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(500):
                gate.close()
                assert gate.is_closed
                gate.release()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"gate raised under contention: {errors}"
    assert gate.is_open, "holds and releases did not balance out"
