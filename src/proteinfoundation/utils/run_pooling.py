"""Treating several runs of one campaign as one pool.

A campaign's designs arrive over more than one run: an initial production run,
then follow-ups sized from its yield, because how many designs survive trimming,
dedup and the gate cannot be known until the first run has happened.

Dedup and analysis were both scoped to a single run's directory, which is right
for one run and wrong for a campaign. A follow-up could regenerate a design
production already had -- 32% of a single production run was duplicates, so the
model repeats itself readily -- and nothing would notice, leaving a pooled set
whose real size was smaller than its row count.

Kept free of hydra, torch and pandas so the pooling rules are reachable from a
test. The files being read are small: one row per retained design.
"""

import csv
import json
import os

# What filter writes for the designs that survived it: the run's contribution to
# the pool, already deduplicated within itself and past any reward threshold.
RETAINED_TEMPLATE = "top_samples_{config_name}.csv"

# The column filter deduplicates on. Integer residue indices joined by commas,
# not the letter sequence -- comparing the wrong representation would silently
# match nothing.
DEDUP_KEY = "aatype"


def retained_path(inference_dir: str, config_name: str) -> str:
    return os.path.join(inference_dir, RETAINED_TEMPLATE.format(config_name=config_name))


def retained_aatypes(inference_dir: str, config_name: str) -> set[str]:
    """The dedup keys one run contributed to the pool.

    Read from what filter retained rather than from what generation produced: a
    design that was dropped for a low reward is not in the pool, and treating it
    as taken would make a later run discard a design nothing else holds.

    A missing file is an error rather than an empty set. The caller names runs it
    believes are complete, and quietly contributing nothing is how a pool ends up
    with duplicates nobody can account for afterwards.
    """
    path = retained_path(inference_dir, config_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot pool against {inference_dir}: {os.path.basename(path)} is missing, so what that "
            f"run kept is unknown. Run its filter stage, or drop it from the pool."
        )
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and DEDUP_KEY not in rows[0]:
        raise KeyError(
            f"{path} has no '{DEDUP_KEY}' column, so it cannot say which sequences it kept. "
            f"Columns present: {sorted(rows[0])[:8]}"
        )
    return {row[DEDUP_KEY] for row in rows if row.get(DEDUP_KEY)}


def pooled_aatypes(inference_dirs: list[str], config_name: str) -> set[str]:
    """Every dedup key already taken by the runs named."""
    taken: set[str] = set()
    for directory in inference_dirs:
        taken |= retained_aatypes(directory, config_name)
    return taken


def read_pool_manifest(path: str) -> list[str]:
    """The inference directories forming a campaign's pool.

    A file rather than a list of overrides: paths reach the filter through Hydra,
    where a comma is list syntax, and a manifest is also the audit record of what
    a run was deduplicated against -- which cannot be recovered from the outputs.
    """
    with open(path) as handle:
        payload = json.load(handle)
    dirs = payload.get("inference_dirs") if isinstance(payload, dict) else payload
    if not isinstance(dirs, list) or not all(isinstance(d, str) for d in dirs):
        raise ValueError(f"{path} does not hold a list of inference directories")
    return dirs
