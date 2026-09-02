"""Robust outlier flags for advisory metrics.

Advisory confidence scores have no transferable absolute scale, so the campaign's
own designs are the only baseline. These tests pin the two properties that makes
usable: that the estimate is not destroyed by the outliers it is looking for, and
that nothing here can reach a pass/fail decision.
"""

import math

import pandas as pd
import pytest

from proteinfoundation.result_analysis.binder_analysis_utils import (
    add_outlier_columns,
    advisory_per_chain_columns,
    low_outlier_threshold,
    robust_spread,
)

COL = "mpnn_esmfold2_binder_pLDDT"


def test_the_spread_is_not_inflated_by_the_outliers_it_looks_for():
    """The masking effect, which is the whole reason for median and MAD. On the
    CBLN1 production run a 3-sigma cut flagged one design where a 3-MAD cut
    flagged twelve."""
    clean = [0.88, 0.89, 0.90, 0.91, 0.92] * 8
    contaminated = clean + [0.20] * 6
    _, mad_sigma = robust_spread(contaminated)
    mean = sum(contaminated) / len(contaminated)
    std = (sum((x - mean) ** 2 for x in contaminated) / (len(contaminated) - 1)) ** 0.5
    assert mad_sigma < std, "the tail inflates the standard deviation and not the MAD"

    threshold = low_outlier_threshold(contaminated, k=3)
    assert all(x > threshold for x in clean), "no good design is caught"
    assert threshold > 0.20, "and every bad one is"


def test_no_spread_means_no_threshold():
    """Over half the designs sharing one value exactly gives a zero MAD. The
    median is then a cut that rejects everything at the most common value, which
    is not an outlier threshold -- there are no outliers to find."""
    assert math.isnan(low_outlier_threshold([0.9] * 40 + [0.2] * 6))


def test_a_low_design_is_flagged_and_a_normal_one_is_not():
    values = [0.90, 0.91, 0.89, 0.90, 0.92, 0.88, 0.30]
    df = add_outlier_columns(pd.DataFrame({COL: values}), k=3)
    assert list(df[f"{COL}_low_outlier"]) == [False] * 6 + [True]


def test_a_high_design_is_never_a_low_outlier():
    """Folding better than the campaign is not a defect."""
    values = [0.90, 0.91, 0.89, 0.90, 0.92, 0.88, 0.99]
    df = add_outlier_columns(pd.DataFrame({COL: values}), k=3)
    assert not any(df[f"{COL}_low_outlier"])
    assert df[f"{COL}_robust_z"].iloc[-1] > 0, "but it is visible in the signed score"


def test_a_campaign_with_no_spread_has_no_outliers():
    """Every design identical means sigma is zero, and dividing by it would
    report every design as infinitely unusual."""
    df = add_outlier_columns(pd.DataFrame({COL: [0.9] * 10}))
    assert not any(df[f"{COL}_low_outlier"])
    assert df[f"{COL}_robust_z"].isna().all()


def test_too_few_designs_to_say_anything():
    df = add_outlier_columns(pd.DataFrame({COL: [0.9, 0.2]}))
    assert not any(df[f"{COL}_low_outlier"])
    assert math.isnan(low_outlier_threshold([0.9, 0.2]))


def test_missing_values_are_not_outliers():
    """A design with no advisory number is unmeasured, not unusual."""
    df = add_outlier_columns(pd.DataFrame({COL: [0.9, 0.91, 0.89, 0.9, float("nan")]}))
    assert list(df[f"{COL}_low_outlier"]) == [False] * 5
    assert math.isnan(df[f"{COL}_robust_z"].iloc[-1])


def test_nonfinite_values_do_not_poison_the_estimate():
    median, sigma = robust_spread([0.9, 0.91, 0.89, float("nan"), float("inf")])
    assert median == pytest.approx(0.90)
    assert math.isfinite(sigma)


def test_gated_columns_are_left_alone():
    """The AF2 per-chain columns carry the reserved 'complex' segment. Flagging
    them here would put a batch-dependent column beside a gated one, which is
    how a verdict starts depending on its neighbours."""
    df = pd.DataFrame({"mpnn_complex_target_pLDDT": [0.9] * 5, COL: [0.9] * 5})
    assert advisory_per_chain_columns(df) == [COL]
    add_outlier_columns(df)
    assert "mpnn_complex_target_pLDDT_low_outlier" not in df.columns


def test_both_per_chain_metrics_are_picked_up():
    df = pd.DataFrame(
        {
            "mpnn_esmfold2_target_pLDDT": [0.9] * 5,
            "mpnn_esmfold2_binder_pLDDT": [0.9] * 5,
            "mpnn_esmfold2_pLDDT": [0.9] * 5,
        }
    )
    found = advisory_per_chain_columns(df)
    assert set(found) == {"mpnn_esmfold2_target_pLDDT", "mpnn_esmfold2_binder_pLDDT"}
    assert "mpnn_esmfold2_pLDDT" not in found, "the confounded complex mean is not per-chain"


def test_the_flags_cannot_reach_a_verdict():
    """These are batch-dependent by construction, so they must never be gated.
    A design's pass/fail must not change because a different design was run
    beside it."""
    from proteinfoundation.result_analysis.binder_analysis_utils import DEFAULT_PROTEIN_BINDER_THRESHOLDS

    df = add_outlier_columns(pd.DataFrame({COL: [0.9, 0.91, 0.2, 0.89, 0.9]}))
    added = [c for c in df.columns if c.endswith(("_low_outlier", "_robust_z"))]
    assert added, "the test is vacuous otherwise"
    for spec in DEFAULT_PROTEIN_BINDER_THRESHOLDS.values():
        metric = spec.get("metric") or ""
        assert not metric.endswith(("_low_outlier", "_robust_z"))
