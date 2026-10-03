"""Writing the measured round-trip into config/local.yaml.

The number this writes is the one the pre-flight list calls the only remaining
item that can ruin the demonstration outright. It used to be a sentence in a
report and a manual edit afterwards, which is the step that gets skipped half an
hour before a talk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_audio_device.py"


def _module():
    spec = importlib.util.spec_from_file_location("measure_audio_device_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = {"roundtrip": {"samples": 5, "median_ms": 428.4, "spread_ms": 6.2,
                        "min_ms": 425.0, "max_ms": 431.2}}

# The real file, comments and all. The consent paragraph is the point.
LOCAL_YAML = """\
# Machine-specific overrides for this laptop. Git-ignored.

tts:
  conversion:
    enabled: true
    reference_voice: data/voices/phone-12s.wav

    # The presenter's own voice, at their own request, recorded by them for
    # this purpose. Written here because it is their voice and their machine.
    consent: "presenter's own voice, recorded 2026-09-01 for this project"

# Not set, and it matters: pipeline.output_latency_ms is still 0.
#
# Measure it in the venue, then uncomment:
# pipeline:
#   output_latency_ms: 430
"""


def test_the_value_is_written(tmp_path) -> None:
    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")

    result = _module().write_output_latency(REPORT, path)

    assert result["written"] is True
    assert result["value_ms"] == 428, "the median, rounded"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["pipeline"]["output_latency_ms"] == 428


def test_the_consent_record_survives(tmp_path) -> None:
    """A YAML round-trip would delete every comment in this file.

    config/local.yaml is where the consent record for the cloned voice lives,
    with a paragraph saying whose voice it is and when it was recorded. Losing
    that to a convenience function is not a trade this project can make.
    """
    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")

    _module().write_output_latency(REPORT, path)
    after = path.read_text(encoding="utf-8")

    assert "# The presenter's own voice, at their own request" in after
    assert "# this purpose. Written here because it is their voice" in after
    assert "recorded 2026-09-01 for this project" in after
    assert "reference_voice: data/voices/phone-12s.wav" in after


def test_a_commented_out_pipeline_block_is_not_matched(tmp_path) -> None:
    """Matching `# pipeline:` would write the value inside a comment."""
    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")

    _module().write_output_latency(REPORT, path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded["pipeline"]["output_latency_ms"] == 428
    assert "#   output_latency_ms: 430" in path.read_text(encoding="utf-8"), (
        "the original commented example should be left alone"
    )


def test_an_existing_value_is_replaced_not_duplicated(tmp_path) -> None:
    path = tmp_path / "local.yaml"
    path.write_text("pipeline:\n  half_duplex_tail_ms: 250\n  output_latency_ms: 100\n",
                    encoding="utf-8")

    result = _module().write_output_latency(REPORT, path)
    after = path.read_text(encoding="utf-8")

    assert result["previous"] == 100
    assert after.count("output_latency_ms:") == 1
    assert "half_duplex_tail_ms: 250" in after, "neighbouring settings must survive"
    assert yaml.safe_load(after)["pipeline"]["output_latency_ms"] == 428


def test_the_key_is_added_to_an_existing_pipeline_block(tmp_path) -> None:
    path = tmp_path / "local.yaml"
    path.write_text("pipeline:\n  half_duplex: true\n", encoding="utf-8")

    _module().write_output_latency(REPORT, path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded["pipeline"] == {"half_duplex": True, "output_latency_ms": 428}


def test_a_missing_file_is_created(tmp_path) -> None:
    path = tmp_path / "does-not-exist.yaml"
    result = _module().write_output_latency(REPORT, path)

    assert result["written"] is True
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["pipeline"]["output_latency_ms"] == 428


def test_a_failed_probe_writes_nothing(tmp_path) -> None:
    """No measurement is not the same as a measurement of zero."""
    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")

    result = _module().write_output_latency(
        {"roundtrip": {"error": "no confident measurement"}}, path
    )

    assert result["written"] is False
    assert "no confident measurement" in result["reason"]
    assert path.read_text(encoding="utf-8") == LOCAL_YAML, "the file must be untouched"


def test_a_report_without_a_roundtrip_writes_nothing(tmp_path) -> None:
    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")

    result = _module().write_output_latency({"bandwidth": {}}, path)

    assert result["written"] is False
    assert path.read_text(encoding="utf-8") == LOCAL_YAML


def test_the_provenance_is_recorded_next_to_the_number(tmp_path) -> None:
    """Six months on, "430" alone does not say which room it was measured in."""
    path = tmp_path / "local.yaml"
    result_text = None
    _module().write_output_latency(REPORT, path)
    result_text = path.read_text(encoding="utf-8")

    assert "median 428.4 ms" in result_text
    assert "5 chirps" in result_text


def test_the_written_file_loads_through_the_real_config_loader(tmp_path) -> None:
    """A file the pipeline refuses to start on would be worse than no file."""
    from parliamo.config import load_config

    path = tmp_path / "local.yaml"
    path.write_text(LOCAL_YAML, encoding="utf-8")
    _module().write_output_latency(REPORT, path)

    default = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(default_path=default, local_path=path)
    assert cfg.pipeline.output_latency_ms == 428


def test_crlf_line_endings_are_preserved(tmp_path) -> None:
    """Rewriting a CRLF file as LF makes every line look modified.

    One changed value should show up as one changed line in any diff.
    """
    path = tmp_path / "local.yaml"
    path.write_bytes(LOCAL_YAML.replace("\n", "\r\n").encode("utf-8"))

    _module().write_output_latency(REPORT, path)
    raw = path.read_bytes()

    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n"), "mixed line endings were written"
    assert yaml.safe_load(raw.decode("utf-8"))["pipeline"]["output_latency_ms"] == 428


def test_lf_files_stay_lf(tmp_path) -> None:
    path = tmp_path / "local.yaml"
    path.write_bytes(LOCAL_YAML.encode("utf-8"))

    _module().write_output_latency(REPORT, path)

    assert b"\r" not in path.read_bytes()


# ---------------------------------------------------------------------------
# the median is written, but the gate has to survive the worst chirp
# ---------------------------------------------------------------------------


def test_a_comfortable_margin_is_reported(tmp_path) -> None:
    path = tmp_path / "local.yaml"
    result = _module().write_output_latency(
        {"roundtrip": {"samples": 5, "median_ms": 100.0, "max_ms": 120.0, "spread_ms": 20.0}},
        path, tail_ms=250,
    )
    assert result["covers_worst"] is True
    assert result["gate_ms"] == 350
    assert result["margin_ms"] == 230.0


def test_a_worst_case_beyond_the_gate_is_flagged(tmp_path) -> None:
    """Measured on this laptop: median 366 ms, chirps ranging 229-552.

    Writing the median and stopping there would leave the microphone reopening
    while the speakers are still audible on the slow chirps - the exact failure
    this whole setting exists to prevent.
    """
    path = tmp_path / "local.yaml"
    result = _module().write_output_latency(
        {"roundtrip": {"samples": 5, "median_ms": 366.4, "max_ms": 900.0, "spread_ms": 600.0}},
        path, tail_ms=250,
    )
    assert result["written"] is True, "the value is still written; the operator is warned"
    assert result["covers_worst"] is False
    assert result["margin_ms"] < 0


def test_the_real_measurement_from_this_laptop_is_covered_but_thin(tmp_path) -> None:
    """366 + 250 against a 547.5 ms worst chirp: 68.5 ms spare."""
    path = tmp_path / "local.yaml"
    result = _module().write_output_latency(
        {"roundtrip": {"samples": 5, "median_ms": 366.4, "min_ms": 265.7,
                       "max_ms": 547.5, "spread_ms": 281.8}},
        path, tail_ms=250,
    )
    assert result["covers_worst"] is True
    assert 0 < result["margin_ms"] < 100, "thin enough that the operator should be told"


def test_a_missing_max_falls_back_to_the_median(tmp_path) -> None:
    """Older reports predate max_ms; they must not crash the writer."""
    path = tmp_path / "local.yaml"
    result = _module().write_output_latency(
        {"roundtrip": {"samples": 3, "median_ms": 200.0}}, path, tail_ms=250
    )
    assert result["written"] is True
    assert result["worst_ms"] == 200.0
