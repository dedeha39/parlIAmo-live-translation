"""Every document this project points at must exist.

Six references were dangling when this was written: the architecture note, the
hardware budget, the adding-a-language guide, the ethics and consent note, the
model-licence table, and an ADR filename about microphone choice that never
existed - ADR 0003 turned out to be about the VRAM budget instead.

(Those paths are deliberately described rather than written out here. Naming
them would make this file itself a source of dangling references, which the
first version of this test duly reported.)

They were referenced from the README, from `config/default.yaml`, and from a
docstring in `mt/ctranslate2_nllb.py` telling the reader where to check the
licence before building anything commercial. A pointer to a document that does
not exist is worse than no pointer: it reads as though the answer has been
written down somewhere.

The repository is meant to be published and to serve as a guide for people
building the same thing in other languages, so this is checked rather than
remembered.
"""

from __future__ import annotations

import re

from parliamo.paths import REPO_ROOT

# Third-party checkouts and generated output are not ours to keep tidy.
SKIP_DIRS = {"external", "models", "runs", "checkpoints", ".git", "data", "__pycache__"}

# Markdown links, and bare docs/ paths mentioned in code comments and YAML.
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+\.md)[^)]*\)")
_BARE_DOC = re.compile(r"docs/[A-Za-z0-9_./-]+\.md")


def _project_files():
    for pattern in ("*.md", "*.py", "*.yaml", "*.txt"):
        for path in REPO_ROOT.rglob(pattern):
            if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
                continue
            yield path


def test_no_dangling_document_references() -> None:
    missing: list[str] = []
    for path in _project_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        targets = set(_MD_LINK.findall(text)) | set(_BARE_DOC.findall(text))
        for target in targets:
            if target.startswith(("http://", "https://")):
                continue
            candidate = (
                REPO_ROOT / target
                if target.startswith("docs/")
                else path.parent / target
            )
            if not candidate.exists():
                missing.append(f"{path.relative_to(REPO_ROOT)} -> {target}")

    assert not missing, "references to documents that do not exist:\n  " + "\n  ".join(
        sorted(set(missing))
    )


def test_every_adr_is_linked_from_somewhere() -> None:
    """An ADR nobody links to is an ADR nobody reads."""
    adr_dir = REPO_ROOT / "docs" / "adr"
    adrs = {p.name for p in adr_dir.glob("[0-9]*.md")}

    linked: set[str] = set()
    for path in _project_files():
        if path.parent == adr_dir:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in adrs:
            if name in text or name.split("-")[0] in _adr_numbers(text):
                linked.add(name)

    orphans = adrs - linked
    assert not orphans, f"ADRs referenced from nowhere: {sorted(orphans)}"


def _adr_numbers(text: str) -> set[str]:
    """ADRs are also cited as 'ADR 0004' in prose rather than by filename."""
    return {m.group(1) for m in re.finditer(r"ADR (\d{4})", text)}


def test_the_entry_point_exists_and_is_linked() -> None:
    """docs/00-state.md is what someone picking this up cold is told to read."""
    state = REPO_ROOT / "docs" / "00-state.md"
    assert state.exists()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/00-state.md" in readme
