"""Repository-root discovery and the canonical directory layout.

Every other module asks this one where things live, so that the project can be
cloned anywhere and still find its config, models and run artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

# Marker files that identify the repository root. ``pyproject.toml`` alone is
# not enough because a user may install the package into site-packages.
_ROOT_MARKERS = ("pyproject.toml", "config/default.yaml")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards from *start* until a directory containing all root markers.

    Falls back to the environment variable ``PARLIAMO_ROOT`` and finally to the
    current working directory, so the package degrades gracefully when it is
    imported from an installed wheel rather than a checkout.
    """
    env_root = os.environ.get("PARLIAMO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and all((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    return Path.cwd().resolve()


REPO_ROOT: Path = find_repo_root()


def resolve(path_like: str | Path) -> Path:
    """Resolve *path_like* against the repository root unless it is absolute."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def ensure_dir(path_like: str | Path) -> Path:
    """Resolve a path and create it (and parents) if missing."""
    p = resolve(path_like)
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_model_cache(models_dir: str | Path = "models") -> Path:
    """Point every model downloader at one directory inside the repository.

    Weights otherwise scatter across ``~/.cache/huggingface``, ``~/.cache/torch``
    and wherever else a library decides, which makes "how much disk does this
    project need" unanswerable and "run it offline" unverifiable. Keeping them
    together also means the stage machine can be checked with one ``du``.

    Call this before importing any model library. Existing environment
    variables win, so a user who has already set ``HF_HOME`` is not overridden.
    """
    root = ensure_dir(models_dir)
    for var, value in (
        ("HF_HOME", root / "huggingface"),
        ("HUGGINGFACE_HUB_CACHE", root / "huggingface" / "hub"),
        ("TORCH_HOME", root / "torch"),
        ("XDG_CACHE_HOME", root / "cache"),
    ):
        os.environ.setdefault(var, str(value))
    return root
