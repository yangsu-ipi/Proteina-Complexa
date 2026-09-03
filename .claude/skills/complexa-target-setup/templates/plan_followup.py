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


def plan(want_designs: int, shards: int, base_seed: int, index: int, observed: dict) -> dict:
    """SEEDS, RAW, KEEP and EXPECT for the requested number of designs."""
    if want_designs < 1:
        raise SystemExit("A follow-up needs a positive number of designs.")
    if shards < 1:
        raise SystemExit("A follow-up needs at least one shard.")

    # Round up: asking for 700 and planning 699 is the failure mode this exists
    # to remove.
    seeds = math.ceil(want_designs / observed["designs_per_seed"])
    # Seeds split evenly across shards, so round up to a whole number per shard
    # or the last shard silently gets a different size than the trim assumes.
    seeds = math.ceil(seeds / shards) * shards
    raw = round(seeds * observed["expansion_per_seed"])
    per_shard = raw // shards
    keep = max(1, math.floor(per_shard * observed["trim_ratio"]))
    return {
        "index": index,
        "want_designs": want_designs,
        "seeds": seeds,
        "raw": raw,
        "keep": keep,
        "expect": keep * shards,
        "shards": shards,
        "rng_seed": base_seed + index * SEED_STRIDE,
        "projected_designs": round(seeds * observed["designs_per_seed"]),
        **observed,
    }


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-dir", type=Path, required=True)
    p.add_argument("--want-designs", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--base-seed", type=int, required=True)
    p.add_argument("--reference-kind", default="production")
    p.add_argument("--reference-seeds", type=int, required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--index", type=int, default=None, help="override; defaults to the next unused")
    p.add_argument("--check", action="store_true", help="verify the derivation reproduces the reference")
    args = p.parse_args()

    observed = observed_yield(args.campaign_dir, args.reference_kind, args.reference_seeds)

    if args.check:
        # Asked for what the reference produced, the arithmetic must return the
        # reference's own parameters. If it does not, every follow-up is skewed.
        back = plan(observed["reference_designs"], args.shards, args.base_seed, 0, observed)
        if back["seeds"] != observed["reference_seeds"]:
            print(
                f"CHECK FAILED: {observed['reference_designs']} designs plans "
                f"{back['seeds']} seeds, but {args.reference_kind} used {observed['reference_seeds']}",
                file=sys.stderr,
            )
            return 1
        print(f"CHECK OK: reproduces {args.reference_kind} at {observed['reference_seeds']} seeds")
        return 0

    index = args.index if args.index is not None else next_index(args.campaign_dir)
    planned = plan(args.want_designs, args.shards, args.base_seed, index, observed)
    planned["run_name"] = f"{args.run_prefix}_followup{index}"

    record = args.campaign_dir / "metadata" / f"followup_{index}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(planned, indent=2, sort_keys=True) + "\n")

    # Shell-evalable, so the runner needs no parsing of its own.
    for key in ("run_name", "seeds", "raw", "keep", "expect", "rng_seed", "index"):
        print(f"FOLLOWUP_{key.upper()}={planned[key]}")
    print(f"FOLLOWUP_RECORD={record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
