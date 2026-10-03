"""measure_room_echo.py: the arithmetic, without a sound card.

The measurement itself plays a sentence in the room; what can be checked here
is that the frames are read the way the pipeline's capture reads them and that
the margin, the advice and the config edit are right.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_room_echo as echo  # noqa: E402  # type: ignore[import-not-found]

THR = 0.5


def _timeline(*parts: tuple[int, bool, float]) -> tuple[list[float], list[bool]]:
    """(frames, gate closed, speech probability) runs, end to end."""
    probs: list[float] = []
    closed: list[bool] = []
    for n, c, p in parts:
        probs += [p] * n
        closed += [c] * n
    return probs, closed


def test_an_echo_that_ends_before_the_gate_reopens_has_a_positive_margin() -> None:
    # 100 frames of sentence, 5 of echo, gate closed 10 frames past the sentence
    probs, closed = _timeline((20, False, 0.0), (100, True, 0.9), (5, True, 0.8),
                              (5, True, 0.1), (40, False, 0.0))
    [m] = echo.echo_margins(probs, closed, THR)
    assert m["heard_as_speech"]
    assert m["margin_ms"] == round(5 * echo.FRAME_MS)
    assert m["speech_after_reopen_ms"] == 0


def test_an_echo_that_outlives_the_gate_is_a_leak() -> None:
    probs, closed = _timeline((20, False, 0.0), (100, True, 0.9),
                              (8, False, 0.7), (40, False, 0.0))
    [m] = echo.echo_margins(probs, closed, THR)
    assert m["margin_ms"] == -round(8 * echo.FRAME_MS)
    assert m["speech_after_reopen_ms"] == round(8 * echo.FRAME_MS)
    assert m["longest_run_after_reopen_ms"] == round(8 * echo.FRAME_MS)


def test_a_microphone_that_never_hears_the_speakers_as_speech() -> None:
    """The close microphone's ideal: the room's sound is not speech to it."""
    probs, closed = _timeline((20, False, 0.0), (100, True, 0.2), (40, False, 0.0))
    margins = echo.echo_margins(probs, closed, THR)
    assert not margins[0]["heard_as_speech"]
    advice = echo.recommend(margins, 250)
    assert not advice["change"] and "does not hear" in advice["verdict"]


def test_each_sentence_is_judged_up_to_the_next_one_only() -> None:
    probs, closed = _timeline((10, False, 0.0), (50, True, 0.9), (5, False, 0.0),
                              (50, True, 0.9), (40, False, 0.0))
    margins = echo.echo_margins(probs, closed, THR, listen_after=100)
    assert [m["sentence"] for m in margins] == [1, 2]
    assert margins[0]["margin_ms"] == 0, "the second sentence is not the first one's echo"


@pytest.mark.parametrize(("worst", "change", "tail"), [
    (400, False, 250),      # plenty
    (100, True, 300),       # 250 + 150 - 100
    (-180, True, 600),      # 250 + 150 + 180 = 580 -> 600
])
def test_the_advice_leaves_the_safety_margin_over_the_worst_sentence(worst, change, tail) -> None:
    margins = [{"heard_as_speech": True, "margin_ms": worst},
               {"heard_as_speech": True, "margin_ms": worst + 300}]
    advice = echo.recommend(margins, 250)
    assert advice["change"] is change
    assert advice["tail_ms"] == tail


def test_frames_carry_the_gate_state_the_capture_saw() -> None:
    chunks = [np.ones(512, dtype=np.float32), np.ones(700, dtype=np.float32),
              np.ones(336, dtype=np.float32)]
    frames, closed = echo.frames_of(chunks, [False, True, False])
    assert frames.shape == (3, 512)
    # frame 1 ends at sample 1024, inside the closed chunk; frame 2 at 1536,
    # inside the last (open) one
    assert closed == [False, True, False]


def test_the_config_edit_keeps_every_comment(tmp_path) -> None:
    cfg = tmp_path / "local.yaml"
    cfg.write_text(
        "# whose voice: consent ref 001, recorded 2026-09-20\n"
        "tts:\n  conversion:\n    speed: 0.85\n\n"
        "pipeline:\n  output_latency_ms: 38  # measured\n  half_duplex_tail_ms: 250\n",
        encoding="utf-8")
    result = echo.write_pipeline_value(cfg, "half_duplex_tail_ms", 600, "measured in the hall")
    assert result["written"] and result["previous"] == "250"
    text = cfg.read_text(encoding="utf-8")
    assert "consent ref 001" in text and "output_latency_ms: 38  # measured" in text
    assert "half_duplex_tail_ms: 600  # measured in the hall" in text


def test_the_config_edit_adds_the_key_when_it_is_missing(tmp_path) -> None:
    cfg = tmp_path / "local.yaml"
    cfg.write_text("pipeline:\n  output_latency_ms: 38\n", encoding="utf-8")
    assert echo.write_pipeline_value(cfg, "half_duplex_tail_ms", 400, "x")["written"]
    import yaml

    assert yaml.safe_load(cfg.read_text(encoding="utf-8"))["pipeline"] == {
        "output_latency_ms": 38, "half_duplex_tail_ms": 400}


def test_the_witness_keeps_every_block_and_remembers_the_gate() -> None:
    from parliamo.audio.capture import AudioCapture
    from parliamo.audio.gate import HalfDuplexGate

    gate = HalfDuplexGate(enabled=True, tail_ms=0)
    witness = echo._Witness(gate)
    capture = AudioCapture(gate=witness)  # type: ignore[arg-type]
    gate.close()
    capture._callback(np.full((512, 1), 0.3, dtype=np.float32), 512, None, None)
    gate.release()
    capture._callback(np.full((512, 1), 0.3, dtype=np.float32), 512, None, None)
    assert witness.states == [True, False]
    assert capture.read(timeout=0.01) is not None and capture.read(timeout=0.01) is not None


def test_start_reads_the_measured_gate_values_again() -> None:
    source = (ROOT / "scripts" / "live_translate.py").read_text(encoding="utf-8")
    factory = source[source.index("    def factory():"):]
    assert "cfg.pipeline.half_duplex_tail_ms = fresh.half_duplex_tail_ms" in factory[:800]
