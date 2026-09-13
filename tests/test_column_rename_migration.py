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


def test_colabfold_is_an_input_name_for_af2_and_never_a_produced_one():
    """`colabfold` is the CLI that runs AlphaFold2 for a monomer, the way
    `colabdesign` is the harness that runs it for a complex. Neither is a model,
    and the complex side has always called this one `af2`. Accepted on the way
    in, never emitted."""
    from proteinfoundation.metrics.column_names import BACKENDS, canonical_backend

    assert canonical_backend("colabfold") == "af2"
    assert canonical_backend("ColabFold ") == "af2", "trimmed and lowered like any other name"
    assert canonical_backend("af2") == "af2"
    assert canonical_backend(canonical_backend("colabfold")) == "af2", "idempotent"
    assert "colabfold" not in BACKENDS, "never a slot value"


def test_the_apo_columns_a_folder_rename_left_behind_are_migrated():
    """These sat in the RIGHT SHAPE with the WRONG NAME, so classify() fell
    through to UNCLASSIFIED and every migration passed over them. The apo builders
    mint names by f-string, which is how a value outside BACKENDS got into a slot
    in the first place."""
    from proteinfoundation.metrics.column_names import classify, rename

    for old, new in (
        ("mpnn_apo_colabfold_binder_pTM", "mpnn_apo_af2_binder_pTM"),
        ("self_apo_colabfold_binder_scRMSD_ca_all", "self_apo_af2_binder_scRMSD_ca_all"),
        ("mpnn_apo_colabfold_sasa_engine_all", "mpnn_apo_af2_sasa_engine_all"),
    ):
        rule, got = classify(old)
        assert got == new, f"{old} -> {got}"
        assert rule == "backend alias", f"and by a named rule, not by accident: {rule}"
        assert rename(got) == got, "applying the map twice must be a no-op"


def test_the_res_family_keeps_its_carve_out_except_for_the_renamed_folder():
    """_res_* is out of the slot scheme by decision. The one exception is
    suffix-anchored: these carry the FOLDER as a trailing token, and nowhere else
    records which model produced a _res_ number -- so leaving it is not
    'unchanged', it is a column naming a folder that no longer exists."""
    from proteinfoundation.metrics.column_names import classify

    assert classify("_res_scRMSD_ca_colabfold")[1] == "_res_scRMSD_ca_af2"
    assert classify("_res_co_scRMSD_all_atom_colabfold_all")[1] == "_res_co_scRMSD_all_atom_af2_all"
    assert classify("_res_scRMSD_single_ca_colabfold")[1] == "_res_scRMSD_single_ca_af2"

    # Everything else in the family is still left alone.
    for untouched in ("_res_scRMSD_ca_esmfold2", "_res_mpnn_best_sequence", "_res_co_seq_rec"):
        rule, got = classify(untouched)
        assert got == untouched and rule == "aggregate", f"{untouched} -> {rule}"


def test_a_mixed_vintage_header_migrates_and_settles(tmp_path):
    """A pooled frame holds runs from either side of a rename, so the map has to
    be applicable twice and to a header that is already half-migrated."""
    import pandas as pd

    from proteinfoundation.metrics.column_names import migrate_frame

    frame = pd.DataFrame(
        {
            "mpnn_apo_colabfold_binder_pTM": [0.5],
            "mpnn_apo_af2_binder_pLDDT": [0.8],
            "_res_scRMSD_ca_colabfold": [1.2],
            "_res_scRMSD_ca_esmfold2": [1.4],
            "self_complex_esmfold2_i_pAE": [0.3],
            "pdb_path": ["x.pdb"],
        }
    )
    once = migrate_frame(frame)
    twice = migrate_frame(once.copy())
    assert list(once.columns) == list(twice.columns), "idempotent over a real header"
    assert "mpnn_apo_af2_binder_pTM" in once.columns
    assert "_res_scRMSD_ca_af2" in once.columns
    assert not [c for c in once.columns if "colabfold" in c]
    # And nothing that was already right moved.
    for kept in ("mpnn_apo_af2_binder_pLDDT", "_res_scRMSD_ca_esmfold2", "self_complex_esmfold2_i_pAE"):
        assert kept in once.columns
