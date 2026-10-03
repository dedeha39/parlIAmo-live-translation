"""Logging and latency tracing.

Two outputs, deliberately separate:

* a human-readable console stream, for watching a rehearsal go by;
* a machine-readable JSONL file per run, for answering "why was that segment
  1.8 s late?" afterwards without re-running anything.

The latency tracer is the reason the JSONL exists. Every audio segment gets an
id, and every stage records when it started and finished working on that id.
That turns a vague "it feels laggy" into a table you can act on.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import ensure_dir

_CONSOLE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-22s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class JsonlHandler(logging.Handler):
    """Append one JSON object per log record to a file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            extra = getattr(record, "extra_fields", None)
            if extra:
                payload.update(extra)
            if record.exc_info:
                payload["exception"] = self.format(record)
            with self._lock:
                self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:  # pragma: no cover - logging must never crash the app
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            finally:
                super().close()


def new_run_id() -> str:
    """A sortable, human-readable identifier for one execution."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def configure_console() -> str:
    """Make the console able to print Turkish and Italian. Returns the encoding.

    Windows still hands Python a legacy code page - ``cp1252`` on this machine -
    when output is piped or the console has not been switched to UTF-8. Printing
    ``ışık`` or ``perché`` to that stream raises ``UnicodeEncodeError``.

    That is not cosmetic here. Every subtitle line contains Turkish source text
    and Italian translation, and in the live pipeline the printing happens
    inside the ``on_delivery`` callback, whose exceptions are caught and logged
    so that one bad sentence cannot stop the show. The two behaviours combine
    badly: on a cp1252 console the operator would see **no subtitles at all**
    and no error either, while the pipeline carried on working perfectly.

    So: switch the console itself to UTF-8 where Windows allows it, reconfigure
    the streams, and fall back to replacing unencodable characters. A subtitle
    with a wrong character is readable; a missing subtitle is not.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            # 65001 is UTF-8. Failure is fine - the reconfigure below still
            # prevents the crash, it just may render as mojibake.
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # pragma: no cover - not all consoles allow this
            pass

    for stream in (sys.stdout, sys.stderr):
        # pytest and other harnesses replace these with objects that have no
        # reconfigure(); leaving them alone is correct there.
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8", errors="replace")

    return getattr(sys.stdout, "encoding", "unknown")


def setup_logging(
    level: str = "INFO",
    console: bool = True,
    jsonl: bool = True,
    runs_dir: str | Path = "runs",
    run_id: str | None = None,
) -> tuple[str, Path]:
    """Configure the root logger. Returns ``(run_id, run_directory)``."""
    # Done here rather than in each script, because it is exactly the kind of
    # setup step that gets forgotten in the one place it matters.
    configure_console()
    run_id = run_id or new_run_id()
    run_dir = ensure_dir(Path(runs_dir) / run_id)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(stream)

    if jsonl:
        root.addHandler(JsonlHandler(run_dir / "events.jsonl"))

    logging.getLogger(__name__).info("run %s -> %s", run_id, run_dir)
    return run_id, run_dir


def log_event(logger: logging.Logger, message: str, /, **fields: Any) -> None:
    """Log *message* with structured *fields* attached for the JSONL sink."""
    logger.info(message, extra={"extra_fields": fields})


# ---------------------------------------------------------------------------
# Latency tracing
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SegmentTrace:
    """Per-stage timings for a single audio segment, in milliseconds."""

    segment_id: str
    created_at: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def total_ms(self) -> float:
        return sum(self.stages.values())

    def wall_ms(self) -> float:
        """Time from segment creation to now - includes queueing, not just work."""
        return (time.perf_counter() - self.created_at) * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "stages_ms": {k: round(v, 2) for k, v in self.stages.items()},
            "work_ms": round(self.total_ms(), 2),
            "wall_ms": round(self.wall_ms(), 2),
            **self.meta,
        }


class LatencyTracer:
    """Collects :class:`SegmentTrace` objects and writes them to the run directory."""

    def __init__(self, run_dir: Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.run_dir = run_dir
        self._traces: dict[str, SegmentTrace] = {}
        self._completed: list[SegmentTrace] = []
        self._lock = threading.Lock()
        self._log = logging.getLogger("parliamo.latency")

    def start_segment(self, segment_id: str | None = None, **meta: Any) -> str:
        sid = segment_id or uuid.uuid4().hex[:8]
        with self._lock:
            self._traces[sid] = SegmentTrace(segment_id=sid, meta=dict(meta))
        return sid

    @contextmanager
    def stage(self, segment_id: str, name: str) -> Iterator[None]:
        """Time a named stage for *segment_id*."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                trace = self._traces.get(segment_id)
                if trace is not None:
                    trace.stages[name] = trace.stages.get(name, 0.0) + elapsed

    def annotate(self, segment_id: str, **meta: Any) -> None:
        with self._lock:
            trace = self._traces.get(segment_id)
            if trace is not None:
                trace.meta.update(meta)

    def finish_segment(self, segment_id: str) -> SegmentTrace | None:
        with self._lock:
            trace = self._traces.pop(segment_id, None)
            if trace is not None:
                self._completed.append(trace)
        if trace is not None:
            log_event(self._log, "segment complete", **trace.as_dict())
        return trace

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics over all completed segments."""
        with self._lock:
            completed = list(self._completed)
        if not completed:
            return {"segments": 0}

        stage_names: set[str] = set()
        for t in completed:
            stage_names.update(t.stages)

        def pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
            return ordered[idx]

        out: dict[str, Any] = {"segments": len(completed)}
        for name in sorted(stage_names):
            vals = [t.stages[name] for t in completed if name in t.stages]
            out[name] = {
                "mean_ms": round(sum(vals) / len(vals), 2),
                "p50_ms": round(pct(vals, 0.50), 2),
                "p95_ms": round(pct(vals, 0.95), 2),
                "max_ms": round(max(vals), 2),
            }
        walls = [t.wall_ms() for t in completed]
        out["end_to_end"] = {
            "mean_ms": round(sum(walls) / len(walls), 2),
            "p50_ms": round(pct(walls, 0.50), 2),
            "p95_ms": round(pct(walls, 0.95), 2),
            "max_ms": round(max(walls), 2),
        }
        return out

    def write_summary(self, filename: str = "latency_summary.json") -> Path | None:
        if self.run_dir is None:
            return None
        path = self.run_dir / filename
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return path
