"""Structured logging and latency tracing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from parliamo.logging_setup import (
    JsonlHandler,
    LatencyTracer,
    log_event,
    new_run_id,
    setup_logging,
)


def test_new_run_id_is_sortable() -> None:
    a = new_run_id()
    time.sleep(1.01)
    b = new_run_id()
    assert a < b, "run ids must sort chronologically"
    assert len(a) == len("20260829-182433")


def test_setup_logging_creates_run_dir(tmp_path: Path) -> None:
    run_id, run_dir = setup_logging(runs_dir=tmp_path, run_id="testrun")
    assert run_id == "testrun"
    assert run_dir.is_dir()
    assert (run_dir / "events.jsonl").exists()
    logging.shutdown()


def test_jsonl_handler_writes_structured_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    handler = JsonlHandler(path)
    logger = logging.getLogger("test.jsonl")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_event(logger, "asr done", segment_id="abc123", ms=142.5)
    handler.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "asr done"
    assert payload["segment_id"] == "abc123"
    assert payload["ms"] == 142.5
    assert payload["level"] == "INFO"


def test_jsonl_handler_survives_unicode(tmp_path: Path) -> None:
    """Turkish, Italian and Friulian text all go through this logger."""
    path = tmp_path / "events.jsonl"
    handler = JsonlHandler(path)
    logger = logging.getLogger("test.unicode")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_event(logger, "translated", source="Günaydın, İzmir'den geliyorum.", target="Bundì!")
    handler.close()

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["source"] == "Günaydın, İzmir'den geliyorum."
    assert payload["target"] == "Bundì!"


def test_latency_tracer_records_stages() -> None:
    tracer = LatencyTracer(enabled=True)
    sid = tracer.start_segment(text_len=42)

    with tracer.stage(sid, "asr"):
        time.sleep(0.02)
    with tracer.stage(sid, "mt"):
        time.sleep(0.01)

    trace = tracer.finish_segment(sid)
    assert trace is not None
    assert set(trace.stages) == {"asr", "mt"}
    assert trace.stages["asr"] >= 15.0, "asr stage should measure ~20 ms"
    assert trace.stages["mt"] >= 5.0
    assert trace.meta["text_len"] == 42
    assert trace.total_ms() >= trace.stages["asr"]


def test_latency_tracer_accumulates_repeated_stage() -> None:
    tracer = LatencyTracer(enabled=True)
    sid = tracer.start_segment()
    for _ in range(3):
        with tracer.stage(sid, "tts"):
            time.sleep(0.005)
    trace = tracer.finish_segment(sid)
    assert trace is not None
    assert trace.stages["tts"] >= 12.0, "repeated stage timings must sum, not overwrite"


def test_latency_tracer_disabled_is_a_noop() -> None:
    tracer = LatencyTracer(enabled=False)
    sid = tracer.start_segment()
    with tracer.stage(sid, "asr"):
        time.sleep(0.01)
    trace = tracer.finish_segment(sid)
    assert trace is not None
    assert trace.stages == {}


def test_latency_summary_has_percentiles(tmp_path: Path) -> None:
    tracer = LatencyTracer(run_dir=tmp_path, enabled=True)
    for _ in range(5):
        sid = tracer.start_segment()
        with tracer.stage(sid, "mt"):
            time.sleep(0.002)
        tracer.finish_segment(sid)

    summary = tracer.summary()
    assert summary["segments"] == 5
    assert {"mean_ms", "p50_ms", "p95_ms", "max_ms"} <= set(summary["mt"])
    assert "end_to_end" in summary

    written = tracer.write_summary()
    assert written is not None and written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["segments"] == 5


def test_latency_summary_empty_is_safe() -> None:
    assert LatencyTracer(enabled=True).summary() == {"segments": 0}


# ---------------------------------------------------------------------------
# console encoding
# ---------------------------------------------------------------------------


def test_console_configuration_survives_a_captured_stream() -> None:
    """pytest replaces sys.stdout with an object that has no reconfigure().

    The guard matters beyond tests: anything that wraps stdout - a UI, a log
    collector, a notebook - does the same thing, and a crash there would take
    out the whole run before a single sentence was translated.
    """
    from parliamo.logging_setup import configure_console

    encoding = configure_console()
    assert isinstance(encoding, str)


def test_turkish_and_italian_survive_the_configured_stream(tmp_path: Path) -> None:
    """The characters that actually appear on stage must encode.

    Not a hypothetical: on this machine a bare `print` of Turkish raised
    UnicodeEncodeError under cp1252, and the pipeline swallows callback
    exceptions - so the failure mode was silent, subtitle-free operation.
    """
    import subprocess
    import sys

    script = tmp_path / "probe.py"
    script.write_text(
        "from parliamo.logging_setup import configure_console\n"
        "configure_console()\n"
        "print('ışık İstanbul gelemeyeceklerini perché così è')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, f"printing Turkish failed: {result.stderr}"
    assert "UnicodeEncodeError" not in result.stderr
