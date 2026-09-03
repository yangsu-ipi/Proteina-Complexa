"""Report a campaign's runs as one pool.

A campaign reaches its number over several runs, because yield cannot be known
before the first one. But analyze is scoped to a single ``results_dir``, so a
campaign with an initial run and two follow-ups had three separate answers and
no combined one -- and changing a threshold meant re-running analyze once per
run and adding the results up by hand.

This reads every pooled run, applies one threshold set to all of them, and
reports the campaign total. Verdicts are re-derived rather than read: a verdict
is a comparison against thresholds, so it changes whenever they do while every
metric behind it stays identical. That is why this needs no re-evaluation.

The duplicate audit is the other half. Cross-run dedup happens at filter time,
so a pooled set should hold no sequence twice; if it does, a run was filtered
without its pool manifest and the pooled count is an overcount.
"""

import argparse
import ast
import json
import os
import sys

import pandas as pd
from loguru import logger

from proteinfoundation.result_analysis.binder_analysis import refresh_per_sequence_verdicts
from proteinfoundation.result_analysis.binder_analysis_utils import (
    get_thresholds_for_result_type,
)
from proteinfoundation.utils.run_pooling import pooled_run_dirs

RESULTS_TEMPLATE = "RAW_{result_type}_results_{config_name}_combined.csv"


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return list(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return []
    return []


def load_pooled(
    evaluation_root: str, config_name: str, task_name: str, run_prefix: str, result_type: str
) -> pd.DataFrame:
    """Every pooled run's results, tagged with the run each row came from.

    The tag is what makes a pooled number auditable: without it a campaign total
    cannot be traced back to the run that contributed to it.
    """
    frames = []
    for directory in pooled_run_dirs(evaluation_root, config_name, task_name, run_prefix):
        path = os.path.join(directory, RESULTS_TEMPLATE.format(result_type=result_type, config_name=config_name))
        if not os.path.exists(path):
            logger.warning(
                f"Skipping {os.path.basename(directory)}: no {os.path.basename(path)}. Run its analyze stage."
            )
            continue
        frame = pd.read_csv(path)
        frame["pooled_run"] = os.path.basename(directory)
        frames.append(frame)
        logger.info(f"Pooled {len(frame)} designs from {os.path.basename(directory)}")
    if not frames:
        raise SystemExit(
            f"No pooled results under {evaluation_root}. A pooled report needs at least one run "
            f"that has been through analyze."
        )
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame, seq_types: list[str]) -> dict:
    """Sequences, passes and pass rate, per run and pooled."""

    def counts(frame: pd.DataFrame) -> dict:
        out = {"designs": len(frame)}
        total_slots = total_pass = 0
        for seq_type in seq_types:
            column = f"{seq_type}_pass_all"
            if column not in frame.columns:
                continue
            vectors = [_as_list(v) for v in frame[column]]
            slots = sum(len(v) for v in vectors)
            passes = sum(1 for v in vectors for x in v if str(x) in ("1", "True"))
            out[seq_type] = {"sequences": slots, "passed": passes}
            total_slots += slots
            total_pass += passes
        out["sequences"] = total_slots
        out["orderable_sequences"] = total_pass
        out["pass_rate"] = round(total_pass / total_slots, 4) if total_slots else None
        headline = [c for c in (f"{s}_pass" for s in seq_types) if c in frame.columns]
        if headline:
            any_pass = frame[headline].apply(lambda row: any(str(v) == "1" for v in row), axis=1)
            out["designs_with_a_passing_sequence"] = int(any_pass.sum())
        return out

    return {
        "pooled": counts(df),
        "per_run": {run: counts(part) for run, part in df.groupby("pooled_run", sort=True)},
    }


def duplicate_audit(df: pd.DataFrame) -> dict:
    """Sequences appearing in more than one run.

    Should be empty: cross-run dedup drops them at filter time. A non-empty
    result means a run was filtered without its pool manifest, and every pooled
    count above is an overcount by exactly this much.
    """
    if "binder_sequence" not in df.columns:
        return {"checked": False, "reason": "no binder_sequence column"}
    per_sequence = df.groupby("binder_sequence")["pooled_run"].nunique()
    repeated = per_sequence[per_sequence > 1]
    return {
        "checked": True,
        "unique_sequences": int(df["binder_sequence"].nunique()),
        "rows": len(df),
        "sequences_in_more_than_one_run": len(repeated),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--task-name", required=True)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--result-type", default="protein_binder")
    p.add_argument("--sequence-types", nargs="+", default=["self", "mpnn"])
    p.add_argument("--thresholds-json", default=None, help="override aggregation.success_thresholds")
    p.add_argument("--output", required=True, help="where the pooled summary is written")
    p.add_argument("--pooled-csv", default=None, help="optional: the concatenated rows")
    args = p.parse_args(argv)

    df = load_pooled(args.evaluation_root, args.config_name, args.task_name, args.run_prefix, args.result_type)

    override = None
    if args.thresholds_json:
        with open(args.thresholds_json) as handle:
            override = json.load(handle)
    thresholds = get_thresholds_for_result_type(override, is_ligand_binder=args.result_type == "ligand_binder")
    # One threshold set over every run, re-derived here rather than trusted from
    # whatever each run froze in. Changing a threshold changes no metric, so this
    # costs a comparison rather than a re-evaluation.
    df = refresh_per_sequence_verdicts(df, args.sequence_types, thresholds)

    report = {
        "runs": sorted(df["pooled_run"].unique().tolist()),
        "thresholds": {name: dict(spec) for name, spec in thresholds.items()},
        "duplicates": duplicate_audit(df),
        **summarise(df, args.sequence_types),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    if args.pooled_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.pooled_csv)), exist_ok=True)
        df.to_csv(args.pooled_csv, index=False)

    pooled = report["pooled"]
    logger.info(
        f"Pooled {pooled['designs']} designs over {len(report['runs'])} run(s): "
        f"{pooled['orderable_sequences']}/{pooled['sequences']} sequences pass "
        f"({pooled['pass_rate']}), {pooled.get('designs_with_a_passing_sequence')} designs with at least one"
    )
    dup = report["duplicates"]
    if dup.get("sequences_in_more_than_one_run"):
        logger.error(
            f"{dup['sequences_in_more_than_one_run']} sequence(s) appear in more than one run. "
            f"A run was filtered without its pool manifest, so these counts are an overcount."
        )
    logger.info(f"Pooled report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
