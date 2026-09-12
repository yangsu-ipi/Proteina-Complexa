#!/usr/bin/env python3
# CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged.
# Hashes the campaign package, excluding everything a run produces.
# Validated by the CBLN1/5KC5 campaign, first complete run 2026-08-28.
# Campaign-independent: every input is an argument or derived from the package
# layout. If you find yourself editing this file per campaign, that is a bug
# in the template -- add an argument instead, so the next campaign inherits it.
#
# The manifest answers one question: has the package drifted from what was
# checked in. So it must cover inputs and nothing else -- a manifest that also
# hashes run products changes on every run, and a manifest that changes on every
# run answers nothing. Three kinds were sweeping in and are now named:
#   .msa_work   MSA retrieval scratch, hundreds of MB, rewritten per fetch
#   *.bak-*     campaign.env backups the editing helpers leave behind
#   .DS_Store   macOS directory noise that arrives with an scp from a laptop
#   slurm-*.out scheduler logs, which land at the package root rather than in
#               logs/ and so slipped past the directory exclusion beside it
import fnmatch
import hashlib
from pathlib import Path

EXCLUDED_DIRS = ("metadata", "inference", "evaluation_results", "logs", "__pycache__", ".msa_work")
EXCLUDED_NAMES = ("CHECKSUMS.sha256", ".DS_Store")
EXCLUDED_GLOBS = ("*.bak-*", "*.pyc", "slurm-*.out")


def included(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part in EXCLUDED_DIRS for part in path.parts):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    return not any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDED_GLOBS)


root = Path(__file__).resolve().parents[1]
paths = sorted(p for p in root.rglob("*") if included(p))
(root / "CHECKSUMS.sha256").write_text(
    "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in paths)
)
print(f"CHECKSUMS.sha256: {len(paths)} files under {root}")
