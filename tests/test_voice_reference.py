"""Assembling a reference recording for voice cloning.

The first live cloning test used a single 7.8 s recording that was 35.7%
digital silence, and the result was judged not good enough. This module picks
better material and presents it better; these tests pin down that it does not
damage the audio while doing so.
"""

from __future__ import annotations

import numpy as np

from parliamo.tts.reference import (
    ClipStats,
    build_reference,
    measure_clip,
    normalise,
    rank_candidates,
    trim_gated_regions,
)

RATE = 16000


def _tone(seconds: float, amplitude: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _gap(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def test_gating_is_measured() -> None:
    audio = np.concatenate([_tone(1.0), _gap(1.0)])
    stats = measure_clip(audio, RATE)
    assert 0.45 < stats.gated_fraction < 0.55
    assert abs(stats.duration_s - 2.0) < 0.01
    assert abs(stats.voiced_s - 1.0) < 0.05


def test_clean_audio_reports_no_gating() -> None:
    """Zero crossings are not gating.

    A 220 Hz sine at 16 kHz spends about 0.25% of its samples within a hair of
    zero simply by crossing it. Counting those as removed audio would put a
    floor under every measurement and make the real figures look smaller than
    they are - so only runs long enough to be the driver count.
    """
    stats = measure_clip(_tone(2.0), RATE)
    assert stats.gated_fraction == 0.0
    assert stats.peak > 0.29


def test_empty_clip_is_not_a_crash() -> None:
    stats = measure_clip(np.zeros(0, dtype=np.float32), RATE)
    assert stats.duration_s == 0.0
    assert stats.gated_fraction == 1.0


def test_level_is_reported_in_dbfs() -> None:
    loud = measure_clip(_tone(1.0, amplitude=0.9), RATE)
    quiet = measure_clip(_tone(1.0, amplitude=0.09), RATE)
    assert loud.rms_dbfs > quiet.rms_dbfs
    assert 18 < loud.rms_dbfs - quiet.rms_dbfs < 22  # a factor of ten


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------


def test_least_gated_ranks_first() -> None:
    from pathlib import Path

    a = ClipStats(Path("a"), 6.0, 3.0, 0.2, -30.0, 0.50)
    b = ClipStats(Path("b"), 6.0, 5.4, 0.2, -30.0, 0.10)
    c = ClipStats(Path("c"), 6.0, 4.8, 0.2, -30.0, 0.20)
    assert [s.path.name for s in rank_candidates([a, b, c])] == ["b", "c", "a"]


def test_level_breaks_a_gating_tie() -> None:
    from pathlib import Path

    quiet = ClipStats(Path("quiet"), 6.0, 5.0, 0.1, -34.0, 0.15)
    loud = ClipStats(Path("loud"), 6.0, 5.0, 0.3, -28.0, 0.15)
    assert rank_candidates([quiet, loud])[0].path.name == "loud"


# ---------------------------------------------------------------------------
# trimming
# ---------------------------------------------------------------------------


def test_long_gaps_are_removed() -> None:
    audio = np.concatenate([_tone(1.0), _gap(1.0), _tone(1.0)])
    out = trim_gated_regions(audio, RATE)
    assert out.size < audio.size
    assert out.size / RATE > 1.9  # both tones survive


def test_short_gaps_inside_a_word_are_left_alone() -> None:
    """Splicing across a 30 ms hole would click, which is worse than the hole."""
    audio = np.concatenate([_tone(0.5), _gap(0.03), _tone(0.5)])
    out = trim_gated_regions(audio, RATE, min_gap_s=0.20)
    assert out.size == audio.size


def test_trimming_keeps_a_margin_around_speech() -> None:
    """Cutting flush against the gate would clip word onsets - the exact damage
    the pre-roll buffer exists to prevent upstream."""
    audio = np.concatenate([_tone(1.0), _gap(1.0), _tone(1.0)])
    out = trim_gated_regions(audio, RATE, min_gap_s=0.20, edge_keep_s=0.05)
    # 1.0 s gap minus two 50 ms margins is removed, so 0.1 s of it survives.
    assert abs(out.size / RATE - 2.1) < 0.02


def test_audio_without_gaps_is_untouched() -> None:
    audio = _tone(2.0)
    assert np.array_equal(trim_gated_regions(audio, RATE), audio)


def test_all_silence_returns_something_rather_than_nothing() -> None:
    out = trim_gated_regions(_gap(2.0), RATE)
    assert out.size > 0


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------


def test_normalise_reaches_the_target_peak() -> None:
    out = normalise(_tone(1.0, amplitude=0.24), target_peak=0.95)
    assert abs(float(np.abs(out).max()) - 0.95) < 0.01


def test_normalise_preserves_shape() -> None:
    """Scaling, not compression - the dynamics carry the voice."""
    audio = np.concatenate([_tone(0.5, amplitude=0.1), _tone(0.5, amplitude=0.3)])
    out = normalise(audio)
    ratio = float(np.abs(out[: RATE // 2]).max() / np.abs(out[RATE // 2 :]).max())
    assert abs(ratio - 1 / 3) < 0.02


def test_normalise_silence_does_not_divide_by_zero() -> None:
    out = normalise(_gap(1.0))
    assert out.size > 0
    assert float(np.abs(out).max()) == 0.0


def test_normalise_never_clips() -> None:
    out = normalise(_tone(1.0, amplitude=0.9), target_peak=0.95)
    assert float(np.abs(out).max()) <= 1.0


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def test_reference_stops_at_the_target_length() -> None:
    clips = [(_tone(6.0), RATE) for _ in range(10)]
    out = build_reference(clips, target_seconds=25.0)
    duration = out.size / RATE
    assert 25.0 <= duration < 32.0, duration


def test_reference_uses_everything_when_short_of_target() -> None:
    clips = [(_tone(3.0), RATE) for _ in range(2)]
    out = build_reference(clips, target_seconds=25.0)
    assert 6.0 <= out.size / RATE < 6.5  # two clips plus one pause


def test_reference_is_normalised() -> None:
    clips = [(_tone(6.0, amplitude=0.24), RATE) for _ in range(5)]
    out = build_reference(clips, target_seconds=25.0)
    assert abs(float(np.abs(out).max()) - 0.95) < 0.01


def test_reference_can_skip_normalisation() -> None:
    clips = [(_tone(6.0, amplitude=0.24), RATE) for _ in range(5)]
    out = build_reference(clips, target_seconds=25.0, normalise_peak=None)
    assert float(np.abs(out).max()) < 0.3


def test_reference_separates_clips_with_a_pause() -> None:
    clips = [(_tone(2.0), RATE), (_tone(2.0), RATE)]
    out = build_reference(clips, target_seconds=25.0, normalise_peak=None)
    # A quarter-second of silence exists somewhere in the middle.
    middle = out[int(1.9 * RATE) : int(2.35 * RATE)]
    assert float(np.abs(middle).min()) == 0.0


def test_mismatched_sample_rates_are_refused() -> None:
    import pytest

    with pytest.raises(ValueError, match="resample first"):
        build_reference([(_tone(1.0), RATE), (_tone(1.0), 48000)])


def test_no_clips_is_empty_not_a_crash() -> None:
    assert build_reference([]).size == 0
