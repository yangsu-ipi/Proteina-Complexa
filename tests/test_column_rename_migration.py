"""The rename must not move a single verdict.

134 columns change name. A criterion reading a column that no longer exists does
not fail loudly at the row level -- it produces no verdict, which upstream reads
as "cannot judge". So the way this goes wrong is silence, and the check has to be
that real data re-derives to the same answers it already had.

The fixture is 40 rows lifted verbatim from the CBLN1 pooled results, stratified
so both passing and failing designs are exercised, carrying the six criteria's
columns under their pre-rename names plus the verdicts recorded at the time.
"""

from pathlib import Path

import pandas as pd
import pytest

from proteinfoundation.metrics.column_names import migrate_frame
from proteinfoundation.result_analysis.binder_analysis import refresh_per_sequence_verdicts
from proteinfoundation.result_analysis.binder_analysis_utils import get_thresholds_for_result_type

FIXTURE = Path(__file__).parent / "data" / "cbln1_verdict_sample.csv"
SEQ_TYPES = ["self", "mpnn"]


@pytest.fixture
def legacy():
    return pd.read_csv(FIXTURE)


def thresholds():
    return get_thresholds_for_result_type(None, is_ligand_binder=False)


def test_the_fixture_holds_both_outcomes(legacy):
    """A fixture where everything fails would pass this file's tests while the
    gate did nothing."""
    assert (legacy["self_pass"] == 1).any() and (legacy["self_pass"] != 1).any()


def test_migration_renames_every_criterion_column(legacy):
    migrated = migrate_frame(legacy)
    for old in ("self_complex_i_pAE_all", "self_binder_scRMSD_ca_all", "self_apo_scRMSD_ca_esmfold2_all"):
        assert old not in migrated.columns, f"{old} should have been renamed"
    for new in (
        "self_complex_af2_i_pAE_all",
        "self_complex_af2_binder_scRMSD_ca_all",
        "self_apo_esmfold2_binder_scRMSD_ca_all",
    ):
        assert new in migrated.columns


def test_verdicts_are_actually_recomputed(legacy):
    """The check that makes the rest meaningful. Dropping the stored verdicts
    first means a criterion that matches nothing yields no columns at all, rather
    than leaving stale ones behind for the comparison to read."""
    migrated = migrate_frame(legacy)
    stripped = migrated.drop(columns=[c for c in migrated.columns if c.endswith(("_pass", "_pass_all"))])
    out = refresh_per_sequence_verdicts(stripped, SEQ_TYPES, thresholds())
    for seq in SEQ_TYPES:
        assert f"{seq}_pass_all" in out.columns, f"no {seq} verdicts were produced"


@pytest.mark.parametrize("seq_type", SEQ_TYPES)
def test_recomputed_verdicts_match_the_recorded_ones(legacy, seq_type):
    """Every column changed name; not one verdict may change with it."""
    migrated = migrate_frame(legacy)
    recorded = [list(eval(v)) if isinstance(v, str) else v for v in migrated[f"{seq_type}_pass_all"]]
    stripped = migrated.drop(columns=[c for c in migrated.columns if c.endswith(("_pass", "_pass_all"))])
    out = refresh_per_sequence_verdicts(stripped, SEQ_TYPES, thresholds())
    got = [list(v) for v in out[f"{seq_type}_pass_all"]]
    assert got == recorded


def test_migration_is_safe_to_apply_twice(legacy):
    """A pooled frame can gather runs from either side of the rename, so the
    migration meets already-current names routinely."""
    once = migrate_frame(legacy)
    twice = migrate_frame(once)
    assert list(once.columns) == list(twice.columns)


def test_a_frame_states_which_backend_it_was_gated_on(legacy):
    """Results predating the provenance column resolve to af2, because
    {seq}_complex_* was AF2 by circumstance -- from best_paths_dict, whatever
    refolder filled it."""
    assert "complex_folding_backend" not in legacy.columns
    assert "self_complex_af2_i_pAE_all" in migrate_frame(legacy).columns


def test_a_recorded_backend_wins_over_the_default(legacy):
    frame = legacy.copy()
    frame["complex_folding_backend"] = "rf3"
    migrated = migrate_frame(frame)
    assert "self_complex_rf3_i_pAE_all" in migrated.columns
    assert "self_complex_af2_i_pAE_all" not in migrated.columns
