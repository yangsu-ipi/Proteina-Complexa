#!/usr/bin/env python3
"""Structural diversity of a campaign's binders, per run and pooled.

Each run already reports its own FoldSeek clustering, but two clusterings cannot
be added together: a cluster in one run and a cluster in another are not the
same object, and a fold found by both would be counted twice. The question a
campaign actually asks -- does a follow-up explore new backbones, or resample
the ones production already found -- can only be answered by clustering every
run's binders at once.

So this reruns the clustering over the pooled set, with the settings analyze
uses for its own binder diversity (structure-only alignment, no sequence
identity floor), and then reports how the pooled clusters divide between runs.

Run it in the environment that has foldseek on PATH.
"""

import argparse
import ast
import json
import os
import sys

import pandas as pd
from loguru import logger

# What analyze uses for binder diversity (analyze.py:1432). Matched so pooled
# numbers can be compared against the per-run ones already on disk rather than
# being a second, differently-calibrated measurement.
MIN_SEQ_ID = 0.0
ALIGNMENT_TYPE = 1


def _passes(value) -> bool:
    """Whether a row's per-sequence verdict vector holds any pass."""
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return False
    if isinstance(value, (list, tuple)):
        return any(str(v) in ("1", "True") for v in value)
    return str(value) in ("1", "True")


def subsets(df: pd.DataFrame, seq_types: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """The design sets worth clustering.

    'all' answers what the generator explores; the per-sequence-type successful
    sets answer what survives the gate, which is the number that decides how
    many distinct backbones an order actually contains.
    """
    out = [("all_generated", df)]
    for seq_type in seq_types:
        column = f"{seq_type}_pass_all"
        if column in df.columns:
            keep = df[df[column].map(_passes)]
            if len(keep):
                out.append((f"successful_{seq_type}", keep))
    any_pass = df[[f"{s}_pass_all" for s in seq_types if f"{s}_pass_all" in df.columns]]
    if not any_pass.empty:
        mask = any_pass.apply(lambda row: any(_passes(v) for v in row), axis=1)
        if mask.any():
            out.append(("successful_any", df[mask]))
    return out


def cluster_split_by_run(assignments_csv: str, df: pd.DataFrame) -> dict:
    """How the pooled clusters divide between the runs that contributed them.

    The question the pooled clustering exists to answer. A follow-up that only
    resamples production's folds shows up as clusters shared by both runs and
    almost none of its own -- in which case more designs buy more sequences but
    no new backbones.
    """
    if not os.path.exists(assignments_csv):
        return {"available": False}
    assign = pd.read_csv(assignments_csv)
    if "cluster_index" not in assign or "sample_index" not in assign:
        return {"available": False}
    runs = df["pooled_run"].tolist() if "pooled_run" in df.columns else []
    if not runs:
        return {"available": False}
    assign = assign[assign["sample_index"].between(0, len(runs) - 1)]
    assign["run"] = [runs[i] for i in assign["sample_index"]]
    by_cluster = assign.groupby("cluster_index")["run"].agg(lambda s: frozenset(s))
    shared = sum(1 for v in by_cluster if len(v) > 1)
    only = {}
    for run in sorted(set(runs)):
        only[run] = int(sum(1 for v in by_cluster if v == frozenset({run})))
    return {"available": True, "clusters": len(by_cluster), "shared_by_runs": shared, "exclusive_to": only}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pooled-csv", required=True, help="written by run_campaign.sh pooled")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sequence-types", nargs="+", default=["self", "mpnn"])
    p.add_argument("--per-run", action="store_true", help="also cluster each run alone, under these settings")
    args = p.parse_args(argv)

    # Imported here rather than at module scope: it pulls in the structure stack,
    # and the subset and attribution logic above should stay reachable from a
    # test on a machine that has no foldseek to run anyway.
    from proteinfoundation.result_analysis.compute_diversity import compute_foldseek_diversity

    df = pd.read_csv(args.pooled_csv)
    logger.info(f"Loaded {len(df)} designs from {args.pooled_csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    report: dict = {"pooled": {}, "per_run": {}}
    frames = [("pooled", df)]
    if args.per_run and "pooled_run" in df.columns:
        # Clustered separately but with identical settings, so a per-run number
        # here is comparable with the pooled one. The numbers already on disk
        # were produced the same way, which is the point of matching them.
        frames += [(run, part) for run, part in df.groupby("pooled_run", sort=True)]

    for scope, frame in frames:
        for suffix, subset in subsets(frame, args.sequence_types):
            tag = f"{scope}_{suffix}"
            store = os.path.join(args.out_dir, tag)
            os.makedirs(store, exist_ok=True)
            try:
                result = compute_foldseek_diversity(
                    df=subset.reset_index(drop=True),
                    groupby_cols=[],
                    path_store_results=store,
                    tmp_path=os.path.join(store, "tmp"),
                    metric_suffix=tag,
                    min_seq_id=MIN_SEQ_ID,
                    alignment_type=ALIGNMENT_TYPE,
                    diversity_mode="binder",
                )
            except Exception as exc:
                logger.error(f"FoldSeek failed for {tag}: {exc}")
                continue
            column = next((c for c in result.columns if "diversity" in c), None)
            value = result.iloc[0][column] if column is not None and len(result) else None
            entry = {"structures": len(subset), "foldseek": str(value)}
            if scope == "pooled":
                entry["by_run"] = cluster_split_by_run(os.path.join(store, "cluster_assignments.csv"), subset)
                report["pooled"][suffix] = entry
            else:
                report["per_run"].setdefault(scope, {})[suffix] = entry
            logger.info(f"{tag}: {len(subset)} binders -> {value}")

    out = os.path.join(args.out_dir, "pooled_diversity.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    logger.info(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
