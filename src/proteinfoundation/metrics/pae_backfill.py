"""Recover PAE matrices that were stored beside a structure's other name.

Why this exists. The advisory store copies a harness's structure to the path it
owns, and until ``carry_sidecars`` it copied the structure alone. The matrix
stayed at the harness's path, under a name no cache entry records, so the reader
looked beside the recorded name and found nothing. The fold was paid for and the
re-read it bought was not available.

The two names do not encode each other. The advisory name is content-addressed
-- ``<sha256(binder_seq)[:12]>_model2.pdb`` -- and the harness name is positional
-- ``..._self_seq_0_model2.pdb``. Neither is derivable from the other: the
sequence index is not recorded and the digest does not invert. What does bridge
them is that the copy was a copy, so the two files are byte-identical and the
matrix can be matched to the structure it describes by content.

Not every matrix survives. A harness directory is rewritten per sequence while
the advisory directory accumulates across them, so the original for an early
sequence is overwritten long before the campaign ends. On EPHA3, a sample put
recovery at about a third. The rest are gone, and the only other way to obtain
them is to fold again -- which is what a missing matrix now asks for.

Idempotent and additive: it writes sidecars that are absent and never replaces
one that is present, so running it twice is running it once.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path

from loguru import logger

from proteinfoundation.metrics.pae_store import (
    CONFIDENCE_KEPT_SUFFIX,
    PAE_STORE_SUFFIX,
    carry_sidecars,
    has_stored_pae,
)

STRUCTURE_SUFFIXES = (".pdb", ".cif")
# A job directory is the unit of search. The copy that lost the sidecar and the
# harness output it came from are always siblings under one design's directory,
# so hashing never crosses designs -- which keeps this linear in campaign size
# rather than quadratic, and makes a false match across two designs impossible.
ADVISORY_DIR_SUFFIX = "_complex"


def _structure_files(directory: str) -> list[str]:
    found: list[str] = []
    for base, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(STRUCTURE_SUFFIXES) and not name.endswith(
                (PAE_STORE_SUFFIX, CONFIDENCE_KEPT_SUFFIX)
            ):
                found.append(os.path.join(base, name))
    return found


def _digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def backfill_job_dir(job_dir: str, dry_run: bool = False) -> tuple[int, int]:
    """Carry sidecars to the structures that are byte-identical copies of them.

    Returns ``(recovered, still missing)`` over the advisory structures in this
    directory -- the ones a cache entry points at, which is where the reader
    looks and therefore the only place a matrix counts.
    """
    wanted = [
        p
        for p in _structure_files(job_dir)
        if os.path.basename(os.path.dirname(p)).endswith(ADVISORY_DIR_SUFFIX) and not has_stored_pae(p)
    ]
    if not wanted:
        return 0, 0

    donors = [p for p in _structure_files(job_dir) if has_stored_pae(p)]
    # Size first: hashing every structure in a job directory to match a handful
    # would read hundreds of megabytes to answer a question most files cannot be
    # the answer to. Two files of different length are never the same copy.
    by_size: dict[int, list[str]] = defaultdict(list)
    for path in donors:
        by_size[os.path.getsize(path)].append(path)

    digests: dict[str, str] = {}
    recovered = 0
    for target in wanted:
        candidates = by_size.get(os.path.getsize(target), [])
        if not candidates:
            continue
        target_digest = _digest(target)
        for candidate in candidates:
            if candidate not in digests:
                digests[candidate] = _digest(candidate)
            if digests[candidate] != target_digest:
                continue
            if dry_run:
                recovered += 1
            elif carry_sidecars(candidate, target):
                recovered += 1
            break
    return recovered, len(wanted) - recovered


def backfill(root: str, dry_run: bool = False) -> dict[str, int]:
    """Every job directory under *root*, reported in one line each way."""
    job_dirs = sorted(
        {
            os.path.dirname(os.path.join(base, d))
            for base, dirs, _files in os.walk(root)
            for d in dirs
            if d.endswith(ADVISORY_DIR_SUFFIX)
        }
    )
    totals = {"job_dirs": len(job_dirs), "recovered": 0, "still_missing": 0}
    for job_dir in job_dirs:
        recovered, missing = backfill_job_dir(job_dir, dry_run=dry_run)
        totals["recovered"] += recovered
        totals["still_missing"] += missing
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="campaign directory, or its evaluation_results")
    parser.add_argument("--dry-run", action="store_true", help="report what would be carried, write nothing")
    args = parser.parse_args()

    root = str(args.root)
    if not os.path.isdir(root):
        logger.error(f"not a directory: {root}")
        return 2

    totals = backfill(root, dry_run=args.dry_run)
    verb = "would recover" if args.dry_run else "recovered"
    logger.info(
        f"{verb} {totals['recovered']} advisory PAE matrices across {totals['job_dirs']} job directories; "
        f"{totals['still_missing']} have no surviving copy and can only be obtained by folding again"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
