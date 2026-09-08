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


# A campaign's deliverable is a numbered sequence of runs, and a run is named for
# its position in it. Three spellings name the same sequence:
#
#   production      the first run of a campaign written before the kinds merged
#   followup{K}     the K-th run after that one, so index K+1
#   production{N}   the current spelling, N from 1
#
# The merge is a rename, not a renumbering: `production` and `followup1` were
# always runs 1 and 2, they were just named for how they came about rather than
# for where they sat. Seeds are derived as base + (index - 1) * stride, which is
# what the two old kinds already produced, so nothing on disk has to move.
#
# Matched rather than pattern-excluded, so a smoke variant nobody anticipated --
# `_smoke_bw8`, say -- is left out by default instead of by having been thought of.
LEGACY_FIRST_RUN_SUFFIX = "production"
RUN_SUFFIX_PREFIX = "production"
LEGACY_LATER_RUN_PREFIX = "followup"


def run_suffix(index: int) -> str:
    """The directory suffix naming run *index*, counting from 1."""
    if index < 1:
        raise ValueError(f"a campaign run is numbered from 1, not {index}")
    return f"{RUN_SUFFIX_PREFIX}{index}"


def run_index(suffix: str) -> int | None:
    """The run number a suffix names, or None if it names no pooled run.

    The one place the three spellings are reconciled. Everything that orders,
    deduplicates against, or reports over a campaign's runs asks this rather than
    matching names itself, because a second copy of the mapping would disagree
    silently in exactly the two directions that matter: a run pooled but not
    deduplicated against, or the reverse.
    """
    if suffix == LEGACY_FIRST_RUN_SUFFIX:
        return 1
    for prefix, offset in ((LEGACY_LATER_RUN_PREFIX, 1), (RUN_SUFFIX_PREFIX, 0)):
        rest = suffix[len(prefix) :]
        if suffix.startswith(prefix) and rest.isdigit():
            return int(rest) + offset
    return None


def is_pooled_run(dir_name: str, config_name: str, task_name: str, run_prefix: str) -> bool:
    """Whether a run directory belongs to the campaign's pooled deliverable.

    The one rule, shared by the run planner (which deduplicates against these)
    and the pooled analysis (which reports over them). Two copies would drift,
    and the failure would be silent in both directions: designs deduplicated
    against a run the analysis ignores, or counted from a run the dedup never saw.
    """
    stem = f"{config_name}_{task_name}_{run_prefix}_"
    if not dir_name.startswith(stem):
        return False
    return run_index(dir_name[len(stem) :]) is not None


def pooled_run_dirs(root: str, config_name: str, task_name: str, run_prefix: str) -> list[str]:
    """Pooled run directories under *root*, in run order.

    Ordered so a pooled report reads chronologically rather than however the
    filesystem happened to list them.

    Two directories claiming one run number is refused rather than ordered. It
    means a campaign holds the same run under two spellings -- `_production` and
    `_production1` -- and pooling both would count its designs twice while the
    duplicate check, which compares sequences across runs, would report them as
    duplicates of themselves.
    """
    if not os.path.isdir(root):
        return []
    stem = f"{config_name}_{task_name}_{run_prefix}_"
    names = [n for n in os.listdir(root) if is_pooled_run(n, config_name, task_name, run_prefix)]

    by_index: dict[int, list[str]] = {}
    for name in names:
        by_index.setdefault(run_index(name[len(stem) :]), []).append(name)
    collisions = {i: sorted(v) for i, v in by_index.items() if len(v) > 1}
    if collisions:
        raise ValueError(
            f"more than one run directory claims the same run number in {root}: {collisions}. "
            f"These are the same run under different spellings; keep one."
        )
    return [os.path.join(root, by_index[i][0]) for i in sorted(by_index)]
