#!/usr/bin/env python3
"""Read the three things a first apo-gated run is for.

The apo gate went live at 2.0 A on a convention transferred from monomer scRMSD,
not on a measured distribution. Two other columns landed emitted-but-ungated for
the same reason. This reads all three off a results CSV so the numbers decide what
happens next instead of an argument doing it.

  1. Apo scRMSD distribution -- is 2.0 A doing anything? A gate everything clears
     is not selecting; a gate almost nothing clears is mis-calibrated. Reported
     against the holo scRMSD on the same sequences, since the two come from
     different predictors and their difference mixes target dependence with
     predictor disagreement.

  2. Target-aligned binder RMSD -- read against binder_scRMSD_ca, never alone. It
     measures fold *and* placement, so the pair is what separates them.

  3. Does the inverse folder's own score predict the gate? The harm check for
     ranking redesigns by it. A null result is the EXPECTED outcome, not a
     disqualification: the score tracks expressibility, which the gate does not
     measure. What would argue against ranking is a clear negative relationship.

Usage:
  python script_utils/bioinformatic/read_apo_smoke_test.py <binder_results_*.csv> [--seq-type mpnn]
"""

import argparse
import ast
import glob
import math
import sys


def parse_list(value):
    """A stringified python list from the CSV, or None."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip().startswith("["):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


def finite(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def quantiles(values, points=(0.0, 0.25, 0.5, 0.75, 0.9, 1.0)):
    if not values:
        return {}
    ordered = sorted(values)
    return {p: ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1) + 0.5))] for p in points}


def describe(label, values, threshold=None):
    values = finite(values)
    if not values:
        print(f"  {label}: no finite values")
        return
    qs = quantiles(values)
    line = "  ".join(f"p{int(p * 100)}={qs[p]:.2f}" for p in sorted(qs))
    print(f"  {label}  n={len(values)}  {line}")
    if threshold is not None:
        under = sum(1 for v in values if v < threshold)
        print(f"    below {threshold}: {under}/{len(values)} ({100 * under / len(values):.0f}%)")


def per_sequence(df, seq_type, suffix):
    """Flatten one _all column across designs, keeping only rows that have it."""
    column = f"{seq_type}_{suffix}_all"
    if column not in df.columns:
        return None
    out = []
    for value in df[column]:
        parsed = parse_list(value)
        if parsed is not None:
            out.extend(parsed)
    return out


def paired(df, seq_type, a_suffix, b_suffix):
    """Pairs from two _all columns, aligned by index within each design."""
    ca, cb = f"{seq_type}_{a_suffix}_all", f"{seq_type}_{b_suffix}_all"
    if ca not in df.columns or cb not in df.columns:
        return []
    pairs = []
    for va, vb in zip(df[ca], df[cb], strict=False):
        la, lb = parse_list(va), parse_list(vb)
        if la is None or lb is None:
            continue
        for x, y in zip(la, lb, strict=False):
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y)):
                pairs.append((x, y))
    return pairs


def spearman(pairs):
    """Rank correlation without scipy. Returns None when it would be meaningless."""
    if len(pairs) < 8:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    xs, ys = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def apo_columns(df, seq_type):
    prefix = f"{seq_type}_apo_scRMSD_"
    return sorted({c[: -len("_all")] for c in df.columns if c.startswith(prefix) and c.endswith("_all")})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="binder_results_*.csv (globs allowed)")
    ap.add_argument("--seq-type", default="mpnn", help="self, mpnn or mpnn_fixed (default: mpnn)")
    ap.add_argument("--apo-threshold", type=float, default=2.0)
    ap.add_argument("--holo-threshold", type=float, default=1.5)
    args = ap.parse_args()

    import pandas as pd

    paths = [p for pattern in args.csv for p in sorted(glob.glob(pattern))] or args.csv
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except (OSError, pd.errors.ParserError) as exc:
            print(f"Could not read {path}: {exc}")
    if not frames:
        print("No readable results CSV")
        return 1
    df = pd.concat(frames, ignore_index=True)
    seq = args.seq_type
    print(f"{len(df)} design row(s) from {len(frames)} file(s), sequence type '{seq}'")

    for column in ("redesign_model", "redesign_score_kind", "redesign_conditioning"):
        if column in df.columns:
            print(f"  {column}: {sorted(set(df[column].dropna().astype(str)))}")

    print("\n[1] apo vs holo scRMSD")
    holo = per_sequence(df, seq, "binder_scRMSD_ca")
    if holo is None:
        print("  no holo scRMSD column -- wrong seq-type?")
    else:
        describe("holo binder_scRMSD_ca", holo, args.holo_threshold)
    apo_cols = apo_columns(df, seq)
    if not apo_cols:
        print("  no apo columns -- was compute_apo_metrics on?")
    for column in apo_cols:
        model = column.rsplit("_", 1)[-1]
        describe(f"apo   [{model}]", per_sequence(df, seq, column[len(seq) + 1 :]), args.apo_threshold)
        pairs = paired(df, seq, "binder_scRMSD_ca", column[len(seq) + 1 :])
        if pairs:
            gaps = [b - a for a, b in pairs]
            describe(f"apo - holo [{model}]", gaps)
            print("    (positive = folds worse without the target; mixes that with predictor disagreement)")

    print("\n[2] target-aligned binder RMSD (fold AND placement)")
    ta = per_sequence(df, seq, "binder_scRMSD_target_aligned_ca")
    if ta is None:
        print("  column absent")
    else:
        describe("target-aligned", ta)
        pairs = paired(df, seq, "binder_scRMSD_ca", "binder_scRMSD_target_aligned_ca")
        if pairs:
            describe("target-aligned - binder-aligned", [b - a for a, b in pairs])
            print("    (large positive = folded as designed but docked elsewhere)")

    print("\n[3] does the redesign score predict the gate?")
    scores = f"{seq}_redesign_score_all"
    passes = f"{seq}_pass_all"
    if scores not in df.columns or passes not in df.columns:
        print(f"  need {scores} and {passes}")
    else:
        pairs = paired(df, seq, "redesign_score", "pass")
        if not pairs:
            print("  no paired values (self is never inverse-folded)")
        else:
            passed = [s for s, p in pairs if p]
            failed = [s for s, p in pairs if not p]
            kind = str(df.get("redesign_score_kind", pd.Series(["?"])).dropna().iloc[0])
            print(f"  score kind: {kind}   n={len(pairs)}  passing={len(passed)} failing={len(failed)}")
            for label, values in (("passing", passed), ("failing", failed)):
                if values:
                    describe(f"score of {label}", values)
            rho = spearman(pairs)
            if rho is None:
                print("  too few sequences for a rank correlation")
            else:
                print(f"  spearman(score, pass) = {rho:+.3f}")
                print("  A value near zero is the EXPECTED result: this score tracks expressibility,")
                print("  which the gate does not measure. Ranking by it is argued against only by a")
                print("  clear relationship in the direction that costs pass rate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
