#!/usr/bin/env python3
"""Derive a follow-up run's parameters from the production run that preceded it.

Yield cannot be predicted before a production run: how many raw designs survive
trimming, then global dedup, then the gate is a property of the target. So the
first production run is also the measurement, and a shortfall is the normal
outcome rather than a mistake.

A follow-up therefore takes one number -- how many more designs are wanted --
and reads everything else off what production actually produced:

    designs   = live_after_global_dedup       (run_outputs_<kind>.json)
    expansion = raw_generation_rows / seeds   (observed beam expansion)
    keep      = retained / generated          (observed trim ratio, per shard)

which invert to SEEDS, RAW, KEEP and EXPECT for the size requested. Asked for
production's own design count, it reproduces production's parameters exactly;
that is the arithmetic's own regression test, and `--check` runs it.

Every derived value is written to metadata/followup_<n>.json before anything
runs, so a campaign's history is auditable without re-deriving it -- and the
seed in particular, since it is the one parameter that cannot be recovered from
the outputs afterwards.
"""

# Annotations as strings, so `int | None` does not have to be evaluated by the
# interpreter that runs this. submit_campaign.sh calls this script with plain
# python3, before any conda environment exists, and that can be old.
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# The seed reaching generation is base + job_id, so consecutive runs must not sit
# within SHARDS of each other or a follow-up's shard 0 would redraw a previous
# run's shard 1. A stride far larger than any plausible shard count removes the
# arithmetic from the reader's head.
SEED_STRIDE = 1000


# A campaign's runs are one numbered sequence, and a run is named for its position
# in it. Three spellings name the same sequence:
#
#   production      the first run of a campaign written before the kinds merged
#   followup{K}     the K-th run after that one, so run number K + 1
#   production{N}   the current spelling, N from 1
#
# The merge is a rename, not a renumbering: seeds are base + (number - 1) * stride,
# which is exactly what the two old kinds produced, so nothing on disk moves.
#
# This mirrors proteinfoundation.utils.run_pooling. It is duplicated, not
# imported, because submit_campaign.sh runs this script with plain python3 before
# any conda environment exists -- the repo is not importable at that point.
# tests/test_followup_planning.py asserts the two agree on every spelling, which
# is the only thing that keeps a forced duplicate from drifting.
LEGACY_FIRST_RUN_SUFFIX = "production"
RUN_SUFFIX_PREFIX = "production"
LEGACY_LATER_RUN_PREFIX = "followup"


def run_suffix(number: int) -> str:
    """The run-name suffix for run *number*, counting from 1."""
    if number < 1:
        raise ValueError(f"a campaign run is numbered from 1, not {number}")
    return f"{RUN_SUFFIX_PREFIX}{number}"


def run_number(suffix: str) -> int | None:
    """The run number a suffix names, or None if it names no pooled run."""
    if suffix == LEGACY_FIRST_RUN_SUFFIX:
        return 1
    for prefix, offset in ((LEGACY_LATER_RUN_PREFIX, 1), (RUN_SUFFIX_PREFIX, 0)):
        rest = suffix[len(prefix) :]
        if suffix.startswith(prefix) and rest.isdigit():
            return int(rest) + offset
    return None


def legacy_followup_index(number: int) -> int:
    """The `followup_{K}.json` index that records run *number*.

    Run 1 has no follow-up record -- it is the run every other one was sized
    from -- so this is only meaningful from run 2 on.
    """
    return number - 1


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"Cannot plan a follow-up: {path} is missing. A follow-up derives its size from what "
            f"production actually produced, so a completed production run has to exist first."
        )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot plan a follow-up: {path} is unreadable ({exc}).") from exc


def observed_yield(campaign_dir: Path, kind: str, seeds: int) -> dict:
    """What the reference run got per seed, and how it got there."""
    outputs = _load(campaign_dir / "metadata" / f"run_outputs_{kind}.json")
    trim = _load(campaign_dir / "metadata" / f"shard_trim_{kind}.json")

    raw = int(outputs.get("raw_generation_rows") or 0)
    designs = int(outputs.get("live_after_global_dedup") or 0)
    if seeds < 1 or raw < 1 or designs < 1:
        raise SystemExit(
            f"Cannot plan a follow-up: {kind} recorded {raw} raw rows and {designs} designs from "
            f"{seeds} seeds. A run that produced nothing cannot say what a seed is worth."
        )

    per_shard = trim.get("shards") or {}
    generated = sum(int(s.get("generated_rows") or 0) for s in per_shard.values())
    retained = sum(int(s.get("retained") or 0) for s in per_shard.values())
    if generated < 1 or retained < 1:
        raise SystemExit(f"Cannot plan a follow-up: {kind}'s trim report records no retained designs.")

    return {
        "reference_kind": kind,
        "reference_seeds": seeds,
        "reference_raw": raw,
        "reference_retained": retained,
        "reference_designs": designs,
        # Kept as ratios so the numbers below are visibly derived, not guessed.
        "expansion_per_seed": raw / seeds,
        "trim_ratio": retained / generated,
        "designs_per_seed": designs / seeds,
    }


def metadata_tag(number: int, campaign_dir: Path) -> str:
    """The tag a run's metadata files are written under.

    Whichever spelling that run actually used: a campaign started before the
    kinds merged has `run_outputs_production.json` and `run_outputs_followup1
    .json`, and one started after has `run_outputs_production1.json`. Chosen by
    looking, because the campaign's own files are the only authority on what it
    called itself.
    """
    candidates = [run_suffix(number)]
    if number == 1:
        candidates.append(LEGACY_FIRST_RUN_SUFFIX)
    else:
        candidates.append(f"{LEGACY_LATER_RUN_PREFIX}{legacy_followup_index(number)}")
    for tag in candidates:
        if (campaign_dir / "metadata" / f"run_outputs_{tag}.json").exists():
            return tag
    return candidates[0]


def completed_runs(campaign_dir: Path, first_run_seeds: int) -> list[dict]:
    """Every finished run's seed count and what it produced, in run order.

    Run 1's seed count comes from campaign.env, because it is the run whose size
    was stated rather than derived. Every later run recorded its own.
    """
    runs = []
    for number in range(1, 1000):
        tag = metadata_tag(number, campaign_dir)
        outputs = campaign_dir / "metadata" / f"run_outputs_{tag}.json"
        if not outputs.exists():
            if number > 1:
                break
            continue
        if number == 1:
            seeds = first_run_seeds
        else:
            record = campaign_dir / "metadata" / f"{LEGACY_LATER_RUN_PREFIX}_{legacy_followup_index(number)}.json"
            if not record.exists():
                record = campaign_dir / "metadata" / f"run_{number}.json"
            if not record.exists():
                break
            seeds = int(json.loads(record.read_text())["seeds"])
        runs.append({"number": number, "tag": tag, "seeds": seeds})
    return runs


def observed_yield_pooled(campaign_dir: Path, first_run_seeds: int) -> dict:
    """What a seed is worth, measured over every completed run rather than one.

    Calibrating on the first run alone anchors a campaign permanently on its
    smallest and least representative one, and never updates: on CBLN1 the first
    run measured 5.31 designs per seed while the two runs sized from it went on
    to achieve 5.53 and 5.56, so every later run was planned ~4% short and simply
    over-delivered. Pooling is self-correcting and costs nothing -- the numbers
    are already on disk.
    """
    runs = completed_runs(campaign_dir, first_run_seeds)
    if not runs:
        raise SystemExit(
            f"Cannot size a run: no completed run found under {campaign_dir / 'metadata'}. "
            f"The first run of a campaign has nothing to calibrate on -- state its seed count."
        )
    totals = {"seeds": 0, "raw": 0, "designs": 0, "generated": 0, "retained": 0}
    for run in runs:
        outputs = _load(campaign_dir / "metadata" / f"run_outputs_{run['tag']}.json")
        trim = _load(campaign_dir / "metadata" / f"shard_trim_{run['tag']}.json")
        per_shard = trim.get("shards") or {}
        totals["seeds"] += run["seeds"]
        totals["raw"] += int(outputs.get("raw_generation_rows") or 0)
        totals["designs"] += int(outputs.get("live_after_global_dedup") or 0)
        totals["generated"] += sum(int(s.get("generated_rows") or 0) for s in per_shard.values())
        totals["retained"] += sum(int(s.get("retained") or 0) for s in per_shard.values())
    if min(totals["seeds"], totals["raw"], totals["designs"], totals["generated"], totals["retained"]) < 1:
        raise SystemExit(
            f"Cannot size a run: the completed runs {[r['tag'] for r in runs]} record "
            f"{totals['raw']} raw rows and {totals['designs']} designs from {totals['seeds']} seeds. "
            f"A campaign that produced nothing cannot say what a seed is worth."
        )
    return {
        "calibrated_on": [r["tag"] for r in runs],
        "reference_seeds": totals["seeds"],
        "reference_raw": totals["raw"],
        "reference_retained": totals["retained"],
        "reference_designs": totals["designs"],
        # Kept as ratios so the numbers below are visibly derived, not guessed.
        "expansion_per_seed": totals["raw"] / totals["seeds"],
        "trim_ratio": totals["retained"] / totals["generated"],
        "designs_per_seed": totals["designs"] / totals["seeds"],
    }


def plan(
    shards: int,
    base_seed: int,
    number: int,
    observed: dict,
    want_designs: int | None = None,
    seeds: int | None = None,
) -> dict:
    """SEEDS, RAW, KEEP and EXPECT for one run.

    Sized either by a seed count, which is what a run actually takes, or by a
    design target, which is converted using what a seed has been worth so far.
    Exactly one of the two: a run has one size, and accepting both would let a
    caller state a target and a count that do not correspond.
    """
    if (want_designs is None) == (seeds is None):
        raise SystemExit("A run is sized by --seeds or by --want-designs, not both and not neither.")
    if shards < 1:
        raise SystemExit("A run needs at least one shard.")

    if seeds is None:
        if want_designs < 1:
            raise SystemExit("A run needs a positive number of designs.")
        # Round up: asking for 700 and planning 699 is the failure mode this
        # exists to remove.
        seeds = math.ceil(want_designs / observed["designs_per_seed"])
    elif seeds < 1:
        raise SystemExit("A run needs a positive number of seeds.")
    # Seeds split evenly across shards, so round up to a whole number per shard
    # or the last shard silently gets a different size than the trim assumes.
    seeds = math.ceil(seeds / shards) * shards
    raw = round(seeds * observed["expansion_per_seed"])
    per_shard = raw // shards
    keep = max(1, math.floor(per_shard * observed["trim_ratio"]))
    return {
        # Both recorded: `number` is the run's position in the campaign, `index`
        # the legacy follow-up numbering that names its metadata files. They
        # differ by one, and writing only the new one would strand every record
        # a campaign already has.
        "number": number,
        "index": legacy_followup_index(number),
        "want_designs": want_designs,
        "seeds": seeds,
        "raw": raw,
        "keep": keep,
        "expect": keep * shards,
        "shards": shards,
        # base + (number - 1) * stride. Run 1 gets the base seed, which is what
        # the old `production` kind used, and run K+1 gets what `followup{K}`
        # used -- so the merge renames runs without redrawing any of them.
        "rng_seed": base_seed + (number - 1) * SEED_STRIDE,
        "projected_designs": round(seeds * observed["designs_per_seed"]),
        **observed,
    }


def pool_dirs(campaign_dir: Path, config_name: str, task_name: str, run_prefix: str, before_run: int) -> list[str]:
    """The inference directories a follow-up must not duplicate.

    Every pooled run numbered below *before_run* -- the runs whose designs are
    already part of the deliverable. The bound is a run number, not the legacy
    follow-up index it used to be: run 3 pools runs 1 and 2. Deliberately not the smoke run: those designs are a throwaway
    check, and letting one of them claim a sequence would make a production
    design disappear because a test happened to draw it first.

    A run whose filter output is missing is refused rather than skipped. Skipping
    would under-deduplicate silently, and the duplicates it let through could not
    be identified afterwards without re-deriving every sequence.
    """
    root = campaign_dir / "inference"
    # Every run before this one, whatever it is called. Built from the numbering
    # rather than from a list of spellings, so a campaign holding `production`,
    # `followup1` and `production4` pools all three in order.
    names = []
    if root.is_dir():
        stem = f"{config_name}_{task_name}_{run_prefix}_"
        found = {}
        for entry in root.iterdir():
            if not entry.name.startswith(stem):
                continue
            number = run_number(entry.name[len(stem) :])
            if number is not None and number < before_run:
                found.setdefault(number, entry.name[len(f"{config_name}_{task_name}_") :])
        names = [found[n] for n in sorted(found)]
    dirs = []
    for name in names:
        directory = root / f"{config_name}_{task_name}_{name}"
        if not directory.is_dir():
            continue
        retained = directory / f"top_samples_{config_name}.csv"
        if not retained.exists():
            raise SystemExit(
                f"Cannot pool against {directory.name}: {retained.name} is missing, so what that run "
                f"kept is unknown. Run its filter stage, or move the directory aside."
            )
        dirs.append(str(directory))
    if not dirs:
        raise SystemExit(
            f"Cannot plan a follow-up: no completed run found under {root}. A follow-up is sized "
            f"from a production run and deduplicated against it."
        )
    return dirs


def next_index(campaign_dir: Path) -> int:
    """One past the highest follow-up already recorded.

    Read from the audit records rather than kept in a config, so the campaign's
    own history is the only state -- and a follow-up that was planned but never
    run still consumes its index, which is what keeps seeds from being reused.
    """
    existing = sorted((campaign_dir / "metadata").glob("followup_*.json"))
    indices = []
    for path in existing:
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def resume_index(campaign_dir: Path, want_designs: int) -> int:
    """The index of the follow-up already planned for this many designs.

    A re-run that starts after generation has to reuse the follow-up that exists
    rather than allocate a new one: a fresh index names an inference directory
    nothing ever wrote, and consumes a seed no run will ever use -- and neither
    failure says so until the evaluate stage cannot find its inputs.

    Matched on ``want_designs`` because that is the one number the caller
    supplies and the one recorded verbatim; everything else in the record is
    derived from it. Ambiguity is refused rather than resolved by picking the
    newest: two follow-ups asking for the same count are two different runs, and
    guessing which one was meant would re-evaluate the wrong designs.
    """
    matches, known = [], []
    for path in sorted((campaign_dir / "metadata").glob("followup_*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        index, wanted = record.get("index"), record.get("want_designs")
        if index is None or wanted is None:
            continue
        known.append(f"#{index} wanted {wanted}")
        if int(wanted) == want_designs:
            matches.append(int(index))
    if not matches:
        raise SystemExit(
            f"Cannot resume a follow-up for {want_designs} designs: no record under "
            f"{campaign_dir / 'metadata'} asks for that many. Recorded follow-ups: "
            f"{', '.join(known) or 'none'}. Drop the stage argument to plan a new follow-up, "
            f"or name one outright with FOLLOWUP_INDEX=<n> (--index here)."
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Cannot resume a follow-up for {want_designs} designs: follow-ups "
            f"{matches} all asked for that many. Say which with FOLLOWUP_INDEX=<n> "
            f"(--index here) -- the design count cannot tell them apart."
        )
    return matches[0]


def record_path(campaign_dir: Path, number: int) -> Path:
    """Where run *number*'s planning record lives, in whichever spelling it used."""
    meta = campaign_dir / "metadata"
    if number > 1:
        legacy = meta / f"{LEGACY_LATER_RUN_PREFIX}_{legacy_followup_index(number)}.json"
        if legacy.exists():
            return legacy
        return meta / f"run_{number}.json"
    return meta / "run_1.json"


def next_run_number(campaign_dir: Path, first_run_seeds: int) -> int:
    """One past the highest run the campaign has recorded.

    Read from the audit records rather than kept in a config, so the campaign's
    own history is the only state -- and a run that was planned but never
    executed still consumes its number, which is what keeps seeds from being
    reused.
    """
    highest = 0
    meta = campaign_dir / "metadata"
    if meta.is_dir():
        for path in meta.glob("run_outputs_*.json"):
            number = run_number(path.stem[len("run_outputs_") :])
            if number:
                highest = max(highest, number)
        for path in meta.glob(f"{LEGACY_LATER_RUN_PREFIX}_*.json"):
            try:
                highest = max(highest, int(path.stem.split("_")[-1]) + 1)
            except ValueError:
                continue
        for path in meta.glob("run_*.json"):
            try:
                highest = max(highest, int(path.stem.split("_")[-1]))
            except ValueError:
                continue
    if highest == 0:
        # Nothing recorded: this is the campaign's first run, which is also the
        # only one whose size cannot be derived.
        return 1
    return highest + 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", type=Path, required=True)
    size = p.add_mutually_exclusive_group()
    size.add_argument("--seeds", type=int, default=None, help="run size, in beam-search roots")
    size.add_argument("--want-designs", type=int, default=None, help="design target, converted using observed yield")
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--reference-kind", default="production", help="ignored; calibration pools every completed run")
    p.add_argument("--reference-seeds", type=int, required=True, help="seed count of the campaign's first run")
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--task-name", required=True)
    p.add_argument("--run-number", type=int, default=None, help="name this run outright; defaults to the next unused")
    p.add_argument("--index", type=int, default=None, help="legacy follow-up index; --run-number is this plus one")
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse the existing run asking for --want-designs instead of allocating a new number",
    )
    p.add_argument("--check", action="store_true", help="verify the derivation reproduces the calibration set")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and report the plan without writing the record or the pool manifest",
    )
    args = p.parse_args()

    observed = observed_yield_pooled(args.campaign_dir, args.reference_seeds)

    if args.check:
        # Asked for what the calibration set produced, the arithmetic must return
        # its own seed count. If it does not, every run sized from it is skewed.
        back = plan(args.shards, args.base_seed, 1, observed, want_designs=observed["reference_designs"])
        if back["seeds"] != observed["reference_seeds"]:
            print(
                f"CHECK FAILED: {observed['reference_designs']} designs plans {back['seeds']} seeds, "
                f"but {observed['calibrated_on']} used {observed['reference_seeds']}",
                file=sys.stderr,
            )
            return 1
        print(f"CHECK OK: reproduces {observed['calibrated_on']} at {observed['reference_seeds']} seeds")
        return 0

    if args.run_number is not None:
        number = args.run_number
    elif args.index is not None:
        number = args.index + 1
    elif args.resume:
        number = resume_index(args.campaign_dir, args.want_designs) + 1
    else:
        number = next_run_number(args.campaign_dir, args.reference_seeds)

    # A run that already exists is replayed, not re-derived. Its size is history:
    # it decided what generation drew, what the trim kept, and what
    # verify_run_outputs expects. Re-deriving would now move it, because the
    # calibration pools every completed run and so changes as the campaign grows
    # -- re-planning followup1 today yields 164 seeds where it actually used 170,
    # and a re-run under that number would record a resolved config disagreeing
    # with the designs on disk and fail its own output check.
    record = record_path(args.campaign_dir, number)
    if record.exists():
        planned = json.loads(record.read_text())
        # A record that cannot reconstruct the run is worse than none: the runner
        # would size generation, trimming and the output check from whatever
        # happened to be present. Named here rather than surfacing as a KeyError
        # from the print loop at the bottom.
        missing = [k for k in ("seeds", "raw", "keep", "expect", "rng_seed") if k not in planned]
        if missing:
            raise SystemExit(
                f"{record} records run {number} but is missing {missing}, so the run cannot be "
                f"reconstructed from it. Delete the record to plan the run afresh, or restore the "
                f"fields from the run's resolved config."
            )
        asked = args.seeds if args.seeds is not None else args.want_designs
        field = "seeds" if args.seeds is not None else "want_designs"
        if asked is not None and planned.get(field) not in (None, asked):
            raise SystemExit(
                f"Run {number} is already recorded with {field}={planned.get(field)}, but {asked} was "
                f"asked for. A run's size is fixed once it has one; plan a new run, or pass the size "
                f"it was planned with."
            )
    else:
        if args.seeds is None and args.want_designs is None:
            raise SystemExit("A run is sized by --seeds or by --want-designs; neither was given.")
        planned = plan(
            args.shards, args.base_seed, number, observed, want_designs=args.want_designs, seeds=args.seeds
        )
    # The name a run already has wins over the name the current scheme would give
    # it, so re-planning an existing run points at the directory it wrote. Its own
    # record is the authority: a run is planned before it executes, so asking
    # which metadata files exist would rename a run that had been planned and not
    # yet run -- and rename it between the generate and evaluate stages of one
    # chain, which is precisely the split-run failure the index pinning exists to
    # prevent.
    recorded_name = planned.get("run_name")
    if recorded_name and recorded_name.startswith(f"{args.run_prefix}_"):
        tag = recorded_name[len(args.run_prefix) + 1 :]
    else:
        tag = metadata_tag(number, args.campaign_dir)
    planned["run_name"] = f"{args.run_prefix}_{tag}"

    # Written before the run, because what a run was deduplicated against cannot
    # be recovered from its outputs -- a design absent from the results looks the
    # same whether it was never drawn or dropped as a duplicate.
    pool = pool_dirs(args.campaign_dir, args.config_name, args.task_name, args.run_prefix, number)
    manifest = args.campaign_dir / "metadata" / f"pool_{tag}.json"
    planned["pool_manifest"] = str(manifest)
    planned["pooled_against"] = pool

    # Planning is a write, and that is the point: the parameters reach disk before
    # anything is queued, so a chain sitting in the queue for a day is already
    # auditable. A dry run queues nothing, so there is nothing to make auditable
    # -- and writing anyway rewrote the absolute paths of a finished campaign's
    # records to wherever the dry run happened to be pointed, and would have
    # burned a run number on a run nobody submitted.
    if not args.dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"for_run": planned["run_name"], "inference_dirs": pool}, indent=2) + "\n")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(planned, indent=2, sort_keys=True) + "\n")
    else:
        print(f"DRY_RUN_PLAN_NOT_WRITTEN={record}", file=sys.stderr)

    # Shell-evalable, so the runner needs no parsing of its own. Still named
    # FOLLOWUP_* because run_campaign.sh of every existing campaign package reads
    # those names; RUN_NUMBER is the addition.
    for key in ("run_name", "seeds", "raw", "keep", "expect", "rng_seed", "index", "pool_manifest"):
        print(f"FOLLOWUP_{key.upper()}={planned[key]}")
    print(f"FOLLOWUP_RECORD={record}")
    # The run's position and the name it actually goes by. One authority for
    # both, so the submitter's job names, the runner's metadata filenames and the
    # inference directory cannot disagree about what this run is called.
    print(f"RUN_NUMBER={number}")
    print(f"RUN_TAG={tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
