"""Speaking the disclosure into the audio.

The watermark measurement showed what a watermark is worth: Perth marks
Chatterbox output and recovers cleanly, and recovers nothing at all from the
pipeline that actually ships here, because Kokoro and Seed-VC do not mark
anything. An attacker picks the unmarked tool.

An audible disclosure has the property a watermark lacks - removing it removes
audio. These tests pin down the part that makes it a safeguard rather than a
courtesy: it is inside the clip, not only at the edges, and its position cannot
be predicted.
"""

from __future__ import annotations

import numpy as np

from parliamo.ethics.disclosure import (
    EDGE_MARGIN_S,
    DisclosureReport,
    add_disclosure,
    interior_positions,
)

RATE = 16000


def _speech(seconds: float, freq: float = 180.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _announcement() -> np.ndarray:
    return _speech(1.5, freq=440.0)


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------


def test_disclosure_is_added_at_both_ends() -> None:
    audio = _speech(10.0)
    out, report = add_disclosure(audio, RATE, _announcement())
    assert report.positions_s[0] == 0.0
    assert report.positions_s[-1] > report.positions_s[0]
    assert out.size > audio.size


def test_there_is_at_least_one_interruption_inside() -> None:
    """The edges alone can be trimmed off. The interior one is the safeguard."""
    _out, report = add_disclosure(_speech(10.0), RATE, _announcement())
    interior = [p for p in report.positions_s if 0.0 < p < report.result_s - 2.0]
    assert interior, "nothing was placed inside the clip"


def test_a_long_clip_gets_two_interruptions() -> None:
    _out, report = add_disclosure(_speech(40.0), RATE, _announcement())
    assert len(report.positions_s) >= 4  # start + two interior + end


def test_interior_positions_are_not_predictable() -> None:
    """A fixed offset could simply be cut out; the point is that it is not fixed."""
    seen = {tuple(round(p, 2) for p in interior_positions(30.0)) for _ in range(20)}
    assert len(seen) > 10, "positions repeat far too often to be random"


def test_interior_positions_stay_inside_the_clip() -> None:
    for _ in range(50):
        for p in interior_positions(30.0):
            assert EDGE_MARGIN_S <= p <= 30.0 - EDGE_MARGIN_S


def test_interior_positions_are_sorted() -> None:
    positions = interior_positions(60.0, count=4)
    assert positions == sorted(positions)


def test_positions_are_spread_rather_than_clustered() -> None:
    """Two landing together would leave a long clean stretch to trim to."""
    gaps = []
    for _ in range(30):
        p = interior_positions(40.0, count=2)
        if len(p) == 2:
            gaps.append(p[1] - p[0])
    assert min(gaps) > 1.0, f"closest pair was {min(gaps):.2f}s apart"


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------


def test_a_short_clip_gets_no_interior_position() -> None:
    """Under a couple of seconds there is nowhere to put one, and trimming is
    not a useful attack on a clip that short anyway."""
    assert interior_positions(1.5) == []


def test_a_very_short_clip_still_gets_the_edges() -> None:
    out, report = add_disclosure(_speech(0.8), RATE, _announcement())
    assert len(report.positions_s) >= 2
    assert out.size > _speech(0.8).size


def test_empty_audio_yields_the_announcement_alone() -> None:
    out, report = add_disclosure(np.zeros(0, dtype=np.float32), RATE, _announcement())
    assert out.size > 0
    assert report.original_s == 0.0


# ---------------------------------------------------------------------------
# the audio itself
# ---------------------------------------------------------------------------


def test_no_speech_is_lost() -> None:
    """The disclosure interrupts the clip; it must not delete any of it."""
    audio = _speech(12.0)
    out, report = add_disclosure(audio, RATE, _announcement())
    assert report.result_s > report.original_s
    # Every original sample is still present somewhere, so total energy can
    # only have risen.
    assert float(np.sum(out**2)) > float(np.sum(audio**2))


def test_the_overhead_is_reported() -> None:
    _out, report = add_disclosure(_speech(10.0), RATE, _announcement())
    assert report.overhead_s > 0
    assert report.as_dict()["overhead_s"] == round(report.overhead_s, 2)


def test_announcement_at_a_different_rate_is_resampled() -> None:
    audio = _speech(6.0)
    announcement_48k = np.sin(
        2 * np.pi * 440 * np.arange(int(1.0 * 48000)) / 48000
    ).astype(np.float32)
    out, _ = add_disclosure(audio, RATE, announcement_48k, announcement_rate=48000)
    assert out.size > audio.size
    assert np.isfinite(out).all()


def test_output_does_not_clip() -> None:
    out, _ = add_disclosure(_speech(8.0), RATE, _announcement())
    assert float(np.abs(out).max()) <= 1.0


def test_report_serialises() -> None:
    _out, report = add_disclosure(_speech(10.0), RATE, _announcement())
    payload = report.as_dict()
    assert set(payload) == {"original_s", "result_s", "overhead_s", "positions_s", "text"}
    assert isinstance(report, DisclosureReport)


def test_cuts_land_in_the_quiet_parts() -> None:
    """Splicing mid-vowel clicks, and a click is a seam an editor can find."""
    # Speech, silence, speech - the cut should prefer the silence.
    audio = np.concatenate([_speech(5.0), np.zeros(int(2.0 * RATE), dtype=np.float32),
                            _speech(5.0)]).astype(np.float32)
    from parliamo.ethics.disclosure import _nearest_quiet

    cut = _nearest_quiet(audio, RATE, at_s=5.5, search_s=0.6)
    assert 5.0 * RATE <= cut <= 7.0 * RATE
