"""Per-backend thresholds, nested inside the criterion rather than beside it.

A threshold calibrated on one folding model does not transfer to another -- the
protein-binder defaults already say so about ESMFold2's compressed pLDDT scale.
Expressing that needs one criterion to carry several rules, and the criterion
key names the *quantity*, so two sibling entries differing only in backend would
need two keys. The natural thing to write is the same key twice, which Python
resolves silently in favour of the last: the failure this file already carries a
scar from, where `binder` and `complex` scRMSD_ca collided on one key.
"""

import pandas as pd
import pytest

from proteinfoundation.result_analysis.binder_analysis_utils import (
    COMPLEX_BACKEND_COLUMN,
    ThresholdSpecError,
    complex_backend_of,
    resolve_backend_overrides,
)

BASE = {"kind": "complex", "metric": "i_pAE", "op": "<=", "scale": 31.0, "threshold": 7.0}


def test_a_criterion_without_overrides_is_untouched():
    got = resolve_backend_overrides({"complex_i_pAE": BASE}, "af2")
    assert got == {"complex_i_pAE": BASE}


def test_the_matching_backend_replaces_the_base_rule():
    """Replaces, not adds. Sibling entries would have been a conjunction, so
    tightening appeared to work and loosening silently did not."""
    spec = {**BASE, "by_backend": {"rf3": {"threshold": 12.0}}}
    assert resolve_backend_overrides({"c": spec}, "rf3")["c"]["threshold"] == 12.0
    assert resolve_backend_overrides({"c": spec}, "af2")["c"]["threshold"] == 7.0


@pytest.mark.parametrize("threshold", [3.0, 12.0])
def test_both_tightening_and_loosening_take_effect(threshold):
    spec = {**BASE, "by_backend": {"rf3": {"threshold": threshold}}}
    assert resolve_backend_overrides({"c": spec}, "rf3")["c"]["threshold"] == threshold


def test_an_override_is_a_partial_spec_not_a_bare_number():
    """scale is part of what does not transfer: AF2's i_pAE reaches gated units
    multiplied by 31, and another folder need not."""
    spec = {**BASE, "by_backend": {"rf3": {"threshold": 12.0, "scale": 1.0}}}
    got = resolve_backend_overrides({"c": spec}, "rf3")["c"]
    assert got["threshold"] == 12.0
    assert got["scale"] == 1.0
    assert got["op"] == "<=" and got["metric"] == "i_pAE", "unstated fields come from the base"


def test_by_backend_never_survives_resolution():
    """Downstream reads a plain spec; leaving the nested dict in would let a
    consumer read the base threshold while another read the override."""
    spec = {**BASE, "by_backend": {"rf3": {"threshold": 12.0}}}
    for backend in ("af2", "rf3", None):
        assert "by_backend" not in resolve_backend_overrides({"c": spec}, backend)["c"]


def test_a_criterion_with_no_applicable_rule_raises():
    """Dropping it would shrink the gate silently, and a design passing five
    criteria is indistinguishable from one passing the six it was meant to face."""
    spec = {"kind": "complex", "metric": "i_pAE", "op": "<=", "by_backend": {"af2": {"threshold": 7.0}}}
    assert resolve_backend_overrides({"c": spec}, "af2")["c"]["threshold"] == 7.0
    with pytest.raises(ThresholdSpecError, match="no base threshold"):
        resolve_backend_overrides({"c": spec}, "rf3")


def test_an_unknown_backend_falls_back_to_the_base_rule():
    spec = {**BASE, "by_backend": {"rf3": {"threshold": 12.0}}}
    assert resolve_backend_overrides({"c": spec}, "protenix")["c"]["threshold"] == 7.0


def test_results_without_the_provenance_column_use_base_rules():
    """Every result written before the column existed -- including the three
    completed CBLN1 runs."""
    frame = pd.DataFrame({"self_pass": [1, 0]})
    assert complex_backend_of(frame) is None
    spec = {**BASE, "by_backend": {"rf3": {"threshold": 12.0}}}
    assert resolve_backend_overrides({"c": spec}, None)["c"]["threshold"] == 7.0


def test_one_backend_across_rows_resolves_to_it():
    frame = pd.DataFrame({COMPLEX_BACKEND_COLUMN: ["af2", "af2", "af2"]})
    assert complex_backend_of(frame) == "af2"


def test_rows_disagreeing_about_the_backend_raise():
    """A pooled frame can hold runs that used different folders. Judging half the
    designs by the other half's thresholds is worse than refusing."""
    frame = pd.DataFrame({COMPLEX_BACKEND_COLUMN: ["af2", "rf3"]})
    with pytest.raises(ThresholdSpecError, match="more than one complex folding backend"):
        complex_backend_of(frame)


def test_an_all_nan_column_reads_as_absent():
    frame = pd.DataFrame({COMPLEX_BACKEND_COLUMN: [None, None]})
    assert complex_backend_of(frame) is None


# ---------------------------------------------------------------------------
# A frame that carries the provenance column twice.
#
# The row builder appended it once per sequence type, so every per-job CSV of a
# two-type run holds two identical copies. Nothing read it as a Series until the
# refolded-metrics path did, and then pandas raised
# `'DataFrame' object has no attribute 'unique'` from inside its own internals,
# naming neither the column nor the duplication. The evaluate stage died on it.
# ---------------------------------------------------------------------------


def test_a_duplicated_provenance_column_is_one_fact_recorded_twice():
    import pandas as pd

    from proteinfoundation.result_analysis.binder_analysis_utils import (
        COMPLEX_BACKEND_COLUMN,
        complex_backend_of,
    )

    frame = pd.DataFrame(
        [["af2", "af2", 1.0], ["af2", "af2", 2.0]],
        columns=[COMPLEX_BACKEND_COLUMN, COMPLEX_BACKEND_COLUMN, "x"],
    )
    assert frame[COMPLEX_BACKEND_COLUMN].shape[1] == 2, "the fixture really is duplicated"
    assert complex_backend_of(frame) == "af2"


def test_duplicated_copies_that_disagree_still_raise():
    """Tolerating a duplicate is not tolerating a contradiction: a frame whose
    copies disagree cannot be gated by one resolved set either way."""
    import pandas as pd
    import pytest

    from proteinfoundation.result_analysis.binder_analysis_utils import (
        COMPLEX_BACKEND_COLUMN,
        ThresholdSpecError,
        complex_backend_of,
    )

    frame = pd.DataFrame([["af2", "rf3"]], columns=[COMPLEX_BACKEND_COLUMN, COMPLEX_BACKEND_COLUMN])
    with pytest.raises(ThresholdSpecError):
        complex_backend_of(frame)


def test_nan_is_still_not_a_backend():
    import numpy as np
    import pandas as pd

    from proteinfoundation.result_analysis.binder_analysis_utils import (
        COMPLEX_BACKEND_COLUMN,
        complex_backend_of,
    )

    assert complex_backend_of(pd.DataFrame({COMPLEX_BACKEND_COLUMN: [np.nan, "af2"]})) == "af2"
    assert complex_backend_of(pd.DataFrame({COMPLEX_BACKEND_COLUMN: [np.nan]})) is None


def test_the_provenance_column_is_emitted_once_per_run_not_once_per_sequence_type():
    """It is a property of the run. The guard has to be membership, not
    `idx == 0`: that block runs once per sequence type, so first-design fires
    once per type."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/proteinfoundation/evaluation/binder_eval.py").read_text()
    block = source[source.index("row_dict[COMPLEX_BACKEND_COLUMN] = complex_backend") :][:800]
    assert "if COMPLEX_BACKEND_COLUMN not in all_columns:" in block
