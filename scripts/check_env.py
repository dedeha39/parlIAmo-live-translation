#!/usr/bin/env python
"""Phase 0 gate: verify this machine can host the pipeline, and record the numbers.

Run this before anything else, and again whenever something breaks. It writes a
JSON report to ``runs/env/env-<timestamp>.json`` so that "it worked yesterday"
can be checked against "what changed".

The script degrades gracefully: it is designed to run on a bare interpreter with
nothing installed yet, and to report progressively more as dependencies land.

Exit codes
----------
0   all hard requirements met
1   at least one hard requirement failed
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Hard requirements: the build cannot proceed without these.
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 13)  # exclusive
MIN_FREE_DISK_GB = 40.0
MIN_TOTAL_VRAM_MIB = 7000

# Packages we care about, grouped by the stage that needs them.
#
# Keep this list honest about what actually ships. It named `llama-cpp-python`
# for months after ADR 0004 replaced LLM translation with NLLB, and
# `chatterbox-tts` after ADR 0006 moved the live path to Kokoro - so the report
# asked for one thing and the pipeline used another. Same class of mistake as
# the config that still said `mt.backend: llama_cpp`.
PACKAGE_GROUPS: dict[str, list[str]] = {
    "core": ["pyyaml", "numpy", "rich"],
    "audio": ["sounddevice", "soundfile", "soxr", "silero-vad"],
    "asr": ["faster-whisper", "ctranslate2"],
    "mt": ["ctranslate2", "transformers", "sentencepiece"],
    "tts": ["kokoro", "torch", "torchaudio"],
    "eval": ["sacrebleu", "jiwer", "datasets"],
    "ui": ["fastapi", "uvicorn", "websockets"],
    "dev": ["pytest", "ruff"],
}

# Optional, and absent on purpose. Reported separately so "missing" does not
# read as "broken": Chatterbox is only needed for the cloning demonstration,
# and Seed-VC lives in its own environment behind a socket (see ADR 0006), so
# it is *expected* to be missing from this one.
OPTIONAL_PACKAGES: dict[str, str] = {
    "chatterbox-tts": "ethics demo only - watermarked cloning, not the live path",
    "ollama": "offline LLM translation for Friulian data work (ADR 0004)",
}

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    """Run a command, returning (returncode, combined output). Never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except Exception as exc:  # pragma: no cover
        return 1, f"{type(exc).__name__}: {exc}"


def _pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None
    except Exception:
        return None


def _total_ram_gb() -> float | None:
    """Total physical RAM in GB, without requiring psutil."""
    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MemoryStatusEx()
        stat.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / 1024**3, 1)
        return None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_python() -> dict[str, Any]:
    v = sys.version_info
    ok = MIN_PYTHON <= (v.major, v.minor) < MAX_PYTHON
    return {
        "status": OK if ok else FAIL,
        "version": platform.python_version(),
        "executable": sys.executable,
        "detail": (
            "supported"
            if ok
            else f"need >={MIN_PYTHON[0]}.{MIN_PYTHON[1]},<{MAX_PYTHON[0]}.{MAX_PYTHON[1]}"
        ),
    }


def check_platform() -> dict[str, Any]:
    return {
        "status": OK,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
    }


def check_gpu() -> dict[str, Any]:
    fields = "name,driver_version,memory.total,memory.used,memory.free,compute_cap"
    rc, out = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if rc != 0 or not out:
        return {"status": FAIL, "detail": f"nvidia-smi unavailable: {out}"}

    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    if len(parts) < 6:
        return {"status": FAIL, "detail": f"unexpected nvidia-smi output: {out}"}

    name, driver, total, used, free, cc = parts
    total_i, used_i, free_i = int(total), int(used), int(free)
    status = OK if total_i >= MIN_TOTAL_VRAM_MIB else FAIL

    # A high idle allocation eats directly into the model budget. Flag it here
    # rather than discovering it as an out-of-memory error mid-rehearsal.
    notes = []
    if used_i > 1200:
        notes.append(
            f"{used_i} MiB already allocated by other apps - close browsers/chat "
            "clients before measuring or presenting"
        )
        if status == OK:
            status = WARN

    return {
        "status": status,
        "name": name,
        "driver": driver,
        "compute_capability": cc,
        "vram_total_mib": total_i,
        "vram_used_mib": used_i,
        "vram_free_mib": free_i,
        "notes": notes,
    }


def check_torch() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"status": WARN, "detail": "torch not installed yet"}

    info: dict[str, Any] = {
        "status": OK,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["device_capability"] = f"{major}.{minor}"
        try:
            # Round-trip a real tensor: `is_available()` can be true while the
            # driver/runtime pairing is still broken.
            x = torch.randn(256, 256, device="cuda")
            info["matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
            del x
            torch.cuda.empty_cache()
        except Exception as exc:
            info["status"] = FAIL
            info["matmul_ok"] = False
            info["detail"] = f"CUDA tensor op failed: {type(exc).__name__}: {exc}"
    else:
        info["status"] = FAIL
        info["detail"] = "torch installed but CUDA not available (CPU-only wheel?)"
    return info


def check_disk() -> dict[str, Any]:
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = round(usage.free / 1024**3, 1)
    return {
        "status": OK if free_gb >= MIN_FREE_DISK_GB else FAIL,
        "path": str(REPO_ROOT),
        "free_gb": free_gb,
        "total_gb": round(usage.total / 1024**3, 1),
        "required_gb": MIN_FREE_DISK_GB,
    }


def check_binaries() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, cmd, required in (
        ("ffmpeg", ["ffmpeg", "-version"], True),
        ("git", ["git", "--version"], False),
    ):
        rc, out = _run(cmd)
        first = out.splitlines()[0] if out else ""
        results[name] = {
            "status": OK if rc == 0 else (FAIL if required else WARN),
            "version": first,
        }
    worst = FAIL if any(v["status"] == FAIL for v in results.values()) else (
        WARN if any(v["status"] == WARN for v in results.values()) else OK
    )
    return {"status": worst, "binaries": results}


def check_packages() -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group, names in PACKAGE_GROUPS.items():
        found = {n: _pkg_version(n) for n in names}
        missing = [n for n, v in found.items() if v is None]
        groups[group] = {
            "status": OK if not missing else WARN,
            "installed": {n: v for n, v in found.items() if v},
            "missing": missing,
        }
    optional = {
        name: {"version": _pkg_version(name), "why": why}
        for name, why in OPTIONAL_PACKAGES.items()
    }
    return {"status": OK, "groups": groups, "optional": optional}


def check_audio() -> dict[str, Any]:
    try:
        import sounddevice as sd
    except Exception as exc:
        return {"status": WARN, "detail": f"sounddevice not usable: {type(exc).__name__}: {exc}"}

    try:
        devices = sd.query_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]
        default_in, default_out = sd.default.device
        return {
            "status": OK if inputs and outputs else FAIL,
            "input_device_count": len(inputs),
            "output_device_count": len(outputs),
            "default_input": devices[default_in]["name"] if default_in is not None and default_in >= 0 else None,
            "default_output": devices[default_out]["name"] if default_out is not None and default_out >= 0 else None,
            "hostapis": [a["name"] for a in sd.query_hostapis()],
        }
    except Exception as exc:
        return {"status": FAIL, "detail": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
_RESET = "\033[0m"


def _paint(status: str, use_color: bool) -> str:
    if not use_color:
        return f"[{status}]"
    return f"{_COLOR.get(status, '')}[{status}]{_RESET}"


def _summarise(name: str, result: dict[str, Any]) -> str:
    """One-line human summary for a check result."""
    status = result.get("status")
    if name == "python":
        return f"{result['version']}  ({result['detail']})"
    if name == "platform":
        return (
            f"{result['system']} {result['release']} | {result['cpu_count']} logical CPUs | "
            f"{result['ram_gb']} GB RAM"
        )
    if name == "gpu":
        if status == FAIL and "name" not in result:
            return result.get("detail", "unavailable")
        return (
            f"{result['name']} | driver {result['driver']} | cc {result['compute_capability']} | "
            f"VRAM {result['vram_free_mib']}/{result['vram_total_mib']} MiB free"
        )
    if name == "torch":
        if "version" not in result:
            return result.get("detail", "")
        return (
            f"torch {result['version']} | CUDA {result.get('cuda_version')} | "
            f"available={result['cuda_available']} | matmul_ok={result.get('matmul_ok')}"
        )
    if name == "disk":
        return f"{result['free_gb']} GB free of {result['total_gb']} GB at {result['path']}"
    if name == "binaries":
        return " | ".join(
            f"{k}: {v['version'] or 'MISSING'}" for k, v in result["binaries"].items()
        )
    if name == "audio":
        if "input_device_count" not in result:
            return result.get("detail", "")
        return (
            f"{result['input_device_count']} in / {result['output_device_count']} out | "
            f"default in: {result['default_input']}"
        )
    if name == "packages":
        lines = []
        for group, info in result["groups"].items():
            n_ok = len(info["installed"])
            n_all = n_ok + len(info["missing"])
            lines.append(f"{group} {n_ok}/{n_all}")
        return " | ".join(lines)
    return json.dumps(result, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the raw JSON report only")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the JSON report (default: runs/env/env-<timestamp>.json)",
    )
    args = parser.parse_args()

    checks = {
        "python": check_python,
        "platform": check_platform,
        "disk": check_disk,
        "gpu": check_gpu,
        "torch": check_torch,
        "binaries": check_binaries,
        "audio": check_audio,
        "packages": check_packages,
    }

    report: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "repo_root": str(REPO_ROOT),
        "checks": {},
    }
    for name, fn in checks.items():
        try:
            report["checks"][name] = fn()
        except Exception as exc:  # a broken check must not hide the others
            report["checks"][name] = {
                "status": FAIL,
                "detail": f"check raised {type(exc).__name__}: {exc}",
            }

    failures = [n for n, r in report["checks"].items() if r.get("status") == FAIL]
    warnings = [n for n, r in report["checks"].items() if r.get("status") == WARN]
    report["summary"] = {
        "ok": not failures,
        "failed": failures,
        "warned": warnings,
    }

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "runs" / "env" / f"env-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if not failures else 1

    use_color = not args.no_color and sys.stdout.isatty()
    print(f"\nparlIAmo environment check  -  {report['timestamp']}")
    print("=" * 78)
    for name, result in report["checks"].items():
        print(f"{_paint(result.get('status', '?'), use_color):>16}  {name:<10} {_summarise(name, result)}")
        for note in result.get("notes", []):
            print(f"{'':>18}  -> {note}")

    pkgs = report["checks"]["packages"]["groups"]
    missing_any = {g: i["missing"] for g, i in pkgs.items() if i["missing"]}
    if missing_any:
        print("\nMissing packages by stage:")
        for group, names in missing_any.items():
            print(f"  {group:<8} {', '.join(names)}")

    optional = report["checks"]["packages"].get("optional") or {}
    if optional:
        print("\nOptional (absent is fine):")
        for name, info in optional.items():
            state = info["version"] or "not installed"
            print(f"  {name:<16} {state:<12} {info['why']}")

    print("=" * 78)
    print(f"report written to {out_path}")
    if failures:
        print(f"\nFAILED checks: {', '.join(failures)}")
        return 1
    if warnings:
        print(f"\nall hard requirements met; warnings in: {', '.join(warnings)}")
    else:
        print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
