#!/usr/bin/env python
"""Export the committed tree as a separate, anonymised repository to publish.

The working repository keeps everything: the presenter's name in the hotword
list and the tests built on it, where the talk was given, local paths, and a
history whose every commit carries the author's address. A public copy should
carry none of that, and rewriting this repository's history to get there would
destroy the record it exists to keep. So this copies what is committed at HEAD
into another directory, rewrites it there, refuses to finish if anything on
the forbidden list survives, and makes a fresh repository with one commit
under the name and address given.

The rewrites live in a JSON file that is never committed - it necessarily
names what it hides (``*.local.json`` is git-ignored)::

    {
      "replace": [["literal text", "replacement"], ...],     # applied in order
      "regex":   [["pattern", "replacement"], ...],           # then these
      "forbid":  ["text that must not appear anywhere", ...], # checked last
      "drop":    ["path/to/omit", ...]                        # optional
    }

Usage:

    python scripts/export_public.py --map config/public_export.local.json \\
        --to ../parlIAmo-public --author "parlIAmo" --email 1+me@users.noreply.github.com

Run the test suite in the export before publishing it (with PYTHONPATH set to
its ``src``, so the copy is what is tested, not this tree).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def rewrite(text: str, rules: dict) -> str:
    for old, new in rules.get("replace", []):
        text = text.replace(old, new)
    for pattern, new in rules.get("regex", []):
        text = re.sub(pattern, new, text, flags=re.MULTILINE)
    return text


def leftovers(text: str, forbid: list[str]) -> list[str]:
    folded = text.casefold()
    return [term for term in forbid if term.casefold() in folded]


def export(rules: dict, target: Path) -> list[str]:
    """Copy and rewrite; returns the problems found, empty when clean."""
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty - refusing to write into it")
    target.mkdir(parents=True, exist_ok=True)
    dropped = {d.rstrip("/") for d in rules.get("drop", [])}
    problems: list[str] = []
    for rel in tracked_files():
        if any(rel == d or rel.startswith(d + "/") for d in dropped):
            continue
        src = ROOT / rel
        dst_rel = rewrite(rel, rules)
        dst = target / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = src.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            shutil.copyfile(src, dst)
            problems += [f"{rel}: binary, copied unchecked"]
            continue
        new = rewrite(text, rules)
        dst.write_bytes(new.encode("utf-8"))
        for term in leftovers(new, rules.get("forbid", [])) + leftovers(dst_rel, rules.get("forbid", [])):
            problems.append(f"{dst_rel}: still contains {term!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", required=True, help="the rewrite rules (a git-ignored JSON file)")
    parser.add_argument("--to", required=True, help="an empty or new directory")
    parser.add_argument("--author", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--message", default="parlIAmo: live speech translation on one laptop")
    args = parser.parse_args()

    rules = json.loads(Path(args.map).read_text(encoding="utf-8"))
    target = Path(args.to).resolve()
    if ROOT in target.parents or target == ROOT:
        raise SystemExit("the export must be outside this repository")
    problems = export(rules, target)
    for line in problems:
        print("  " + line)
    if any("still contains" in p for p in problems):
        print("NOT committed: forbidden text survived the rewrite")
        return 1

    def git(*cmd: str) -> None:
        subprocess.run(["git", *cmd], cwd=target, check=True)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("-c", f"user.name={args.author}", "-c", f"user.email={args.email}",
        "commit", "-q", "-m", args.message)
    print(f"exported and committed in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
