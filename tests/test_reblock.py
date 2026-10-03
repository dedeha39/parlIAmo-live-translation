"""Fixed-size re-framing.

Regression coverage for a bug that discarded 53% of the microphone input.

The capture layer resamples 48 kHz to 16 kHz with soxr, whose streaming
resampler emits variable-length chunks - 1100 samples on the reference machine,
where the arithmetic suggests 512. Silero VAD needs exactly 512. The pipeline
originally trimmed each chunk to fit, so more than half of every chunk was
thrown away and recognition ran on a silently mutilated signal.

The property these tests exist to protect: **no sample is lost and none is
duplicated**.
"""

from __future__ import annotations

import numpy as np
import pytest

from parliamo.audio.reblock import Reblocker


def test_exact_multiple_passes_straight_through() -> None:
    rb = Reblocker(512)
    frames = rb.push(np.arange(1024, dtype=np.float32))
    assert len(frames) == 2
    assert all(f.size == 512 for f in frames)
    assert rb.pending == 0


def test_remainder_is_carried_to_the_next_chunk() -> None:
    rb = Reblocker(512)
    assert rb.push(np.arange(600, dtype=np.float32)) != []
    assert rb.pending == 88
    frames = rb.push(np.arange(600, 1024, dtype=np.float32))
    assert len(frames) == 1
    assert rb.pending == 0


def test_chunk_smaller_than_a_frame_emits_nothing_yet() -> None:
    rb = Reblocker(512)
    assert rb.push(np.zeros(100, dtype=np.float32)) == []
    assert rb.pending == 100


def test_no_samples_are_lost_across_the_stream() -> None:
    """The exact property the old trimming code violated."""
    rng = np.random.default_rng(20260830)
    source = rng.standard_normal(50_000).astype(np.float32)

    rb = Reblocker(512)
    out: list[np.ndarray] = []
    cursor = 0
    # Feed in the ragged sizes soxr actually produces.
    for size in [1100, 1100, 1024, 1100, 512, 1100, 900, 1100, 1100, 1100] * 5:
        chunk = source[cursor : cursor + size]
        if chunk.size == 0:
            break
        cursor += chunk.size
        out.extend(rb.push(chunk))

    recovered = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    assert recovered.size == (cursor // 512) * 512
    np.testing.assert_array_equal(recovered, source[: recovered.size])
    assert rb.pending == cursor - recovered.size


def test_the_1100_sample_case_specifically() -> None:
    """The measured real-world chunk size, at the real frame size."""
    rb = Reblocker(512)
    total_in = 0
    total_out = 0
    for _ in range(100):
        total_in += 1100
        total_out += sum(f.size for f in rb.push(np.zeros(1100, dtype=np.float32)))

    # Old behaviour trimmed each chunk to 512, keeping 46.5% of the audio.
    assert total_out / total_in > 0.99, (
        f"kept only {100 * total_out / total_in:.1f}% of the input"
    )


def test_ordering_is_preserved() -> None:
    rb = Reblocker(4)
    frames = rb.push(np.arange(10, dtype=np.float32))
    assert np.concatenate(frames).tolist() == list(range(8))
    frames = rb.push(np.arange(10, 14, dtype=np.float32))
    assert np.concatenate(frames).tolist() == [8, 9, 10, 11]


def test_frames_are_independent_copies() -> None:
    """The segmenter holds frames for the length of an utterance."""
    rb = Reblocker(4)
    frames = rb.push(np.arange(8, dtype=np.float32))
    rb.push(np.full(8, 99.0, dtype=np.float32))
    assert frames[0].tolist() == [0.0, 1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# flush and reset
# ---------------------------------------------------------------------------


def test_flush_zero_pads_the_remainder() -> None:
    rb = Reblocker(512)
    rb.push(np.ones(100, dtype=np.float32))
    tail = rb.flush()
    assert tail is not None
    assert tail.size == 512
    assert tail[:100].tolist() == [1.0] * 100
    assert tail[100:].sum() == 0.0


def test_flush_when_empty_returns_none() -> None:
    assert Reblocker(512).flush() is None
    rb = Reblocker(4)
    rb.push(np.zeros(8, dtype=np.float32))
    assert rb.flush() is None


def test_reset_discards_the_remainder() -> None:
    rb = Reblocker(512)
    rb.push(np.ones(100, dtype=np.float32))
    rb.reset()
    assert rb.pending == 0
    assert rb.flush() is None


# ---------------------------------------------------------------------------
# accounting and validation
# ---------------------------------------------------------------------------


def test_accounting_balances() -> None:
    rb = Reblocker(512)
    for _ in range(10):
        rb.push(np.zeros(1100, dtype=np.float32))
    acc = rb.accounting()
    assert acc["samples_in"] == 11000
    assert acc["samples_out"] + acc["pending"] == acc["samples_in"]


def test_empty_chunk_is_a_noop() -> None:
    rb = Reblocker(512)
    assert rb.push(np.zeros(0, dtype=np.float32)) == []
    assert rb.samples_in == 0


def test_multidimensional_input_is_flattened() -> None:
    rb = Reblocker(4)
    frames = rb.push(np.zeros((2, 4), dtype=np.float32))
    assert len(frames) == 2


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_frame_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Reblocker(size)
