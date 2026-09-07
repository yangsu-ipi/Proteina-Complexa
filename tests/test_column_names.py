"""The result column naming scheme, and the migration off four older ones.

See docs/design-notes/column-naming.md. The scheme's whole point is that a
column says which structure and which part of it a number describes, and which
model produced that structure -- none of which the old names carried.
"""

from pathlib import Path

import pytest

from proteinfoundation.metrics.column_names import (
    BACKENDS,
    GRANDFATHERED_METRICS,
    KINDS,
    RETIRED_COLUMNS,
    SCOPES,
    SEQ_TYPES,
    ColumnNameError,
    classify,
    metric_column,
    rename,
    rename_map,
)

FIXTURE = Path(__file__).parent / "data" / "cbln1_pooled_columns.txt"


def real_columns():
    return [
        line.strip()
        for line in FIXTURE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_the_slots_render_in_order():
    assert (
        metric_column("i_pAE", kind="complex", backend="af2", seq_type="self")
        == "self_complex_af2_i_pAE"
    )
    assert (
        metric_column("ss_counts", kind="complex", backend="esmfold2", seq_type="mpnn", scope="binder_interface")
        == "mpnn_complex_esmfold2_binder_interface_ss_counts"
    )


def test_generated_takes_no_sequence_type():
    """One co-designed sequence exists per design, so there is no mpnn variant of
    a generated structure. Leaving the slot empty keeps the asymmetry visible in
    a generated-vs-mpnn comparison instead of implying a column that cannot
    exist."""
    assert (
        metric_column("ss_counts", kind="complex", backend="generated", scope="binder")
        == "complex_generated_binder_ss_counts"
    )
    with pytest.raises(ColumnNameError, match="cannot exist"):
        metric_column("ss_counts", kind="complex", backend="generated", seq_type="self", scope="binder")


def test_an_apo_structure_has_no_target_or_interface():
    """It holds only the binder, so those scopes describe nothing. Not a
    convention to remember -- a consequence of what the structure is."""
    assert (
        metric_column("ss_counts", kind="apo", backend="esmfold2", seq_type="self", scope="binder")
        == "self_apo_esmfold2_binder_ss_counts"
    )
    for scope in ("target", "interface", "binder_interface", "target_interface"):
        with pytest.raises(ColumnNameError, match="nothing to describe"):
            metric_column("ss_counts", kind="apo", backend="esmfold2", seq_type="self", scope=scope)


@pytest.mark.parametrize("metric", sorted(GRANDFATHERED_METRICS))
def test_a_grandfathered_metric_refuses_a_scope(metric):
    """i_pAE, i_pTM and the ipSAE family are standard field terms and inherently
    interfacial. Giving them a scope slot would rename the one set of metrics a
    reader already knows."""
    assert metric_column(metric, kind="complex", backend="af2", seq_type="self")
    with pytest.raises(ColumnNameError, match="inherently interfacial"):
        metric_column(metric, kind="complex", backend="af2", seq_type="self", scope="interface")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"kind": "binder", "backend": "af2", "seq_type": "self"}, "unknown kind"),
        ({"kind": "complex", "backend": "alphafold", "seq_type": "self"}, "unknown backend"),
        ({"kind": "complex", "backend": "af2", "seq_type": "mpnn2"}, "unknown seq_type"),
        ({"kind": "complex", "backend": "af2", "seq_type": "self", "scope": "epitope"}, "unknown scope"),
    ],
)
def test_an_unknown_slot_value_raises(kwargs, match):
    with pytest.raises(ColumnNameError, match=match):
        metric_column("pTM", **kwargs)


def test_binder_and_target_are_scopes_not_kinds():
    """Every {seq}_binder_* column was an RMSD of the binder within the refolded
    complex; target only ever appeared as target_pLDDT. Neither was ever a
    structure the pipeline produced."""
    assert set(KINDS) == {"complex", "apo"}
    assert {"binder", "target"} <= set(SCOPES)


def test_the_per_sequence_sibling_is_a_suffix():
    assert metric_column(
        "i_pAE", kind="complex", backend="af2", seq_type="self", per_sequence=True
    ) == "self_complex_af2_i_pAE_all"


def test_names_are_built_and_never_parsed():
    """`binder` + `interface_nres` and `binder_interface` + `nres` render to the
    same string, so a parser would be guessing. This pins that they collide, so
    nobody later writes one believing they do not."""
    a = metric_column("interface_nres", kind="complex", backend="af2", seq_type="self", scope="binder")
    b = metric_column("nres", kind="complex", backend="af2", seq_type="self", scope="binder_interface")
    assert a == b == "self_complex_af2_binder_interface_nres"

    from proteinfoundation.metrics import column_names

    assert not any(n.startswith("parse") for n in dir(column_names)), "construction only"


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "old,new",
    [
        ("self_complex_i_pAE", "self_complex_af2_i_pAE"),
        ("mpnn_complex_binder_pLDDT_all", "mpnn_complex_af2_binder_pLDDT_all"),
        ("self_binder_scRMSD_target_aligned_ca", "self_complex_af2_binder_scRMSD_target_aligned_ca"),
        ("self_apo_scRMSD_ca_esmfold2", "self_apo_esmfold2_binder_scRMSD_ca"),
        ("mpnn_apo_pLDDT_esmfold2_all", "mpnn_apo_esmfold2_binder_pLDDT_all"),
        ("self_esmfold2_binder_pLDDT", "self_complex_esmfold2_binder_pLDDT"),
        ("mpnn_fixed_complex_i_pAE", "mpnn_fixed_complex_af2_i_pAE"),
    ],
)
def test_old_names_map_as_documented(old, new):
    assert rename(old) == new


def test_mpnn_fixed_is_matched_before_mpnn():
    """It starts with "mpnn", so a shorter-first match reads
    mpnn_fixed_aa_counts as seq_type "mpnn" and metric "fixed_aa_counts" -- a
    silent misparse, not an error."""
    assert SEQ_TYPES.index("mpnn_fixed") < SEQ_TYPES.index("mpnn")
    assert rename("mpnn_fixed_sequence") == "mpnn_fixed_sequence"
    assert rename("mpnn_fixed_complex_pTM") == "mpnn_fixed_complex_af2_pTM"


def test_the_interface_composition_column_names_the_generated_structure():
    """Verified from binder_metrics: the interface positions come from the
    generated pdb, while the amino acids come from this sequence type. Both
    slots earn their place."""
    assert rename("self_aa_interface_counts") == "self_complex_generated_binder_interface_aa_counts"


@pytest.mark.parametrize("column", ["pdb_path", "binder_sequence", "L", "id_gen", "pooled_run", "task_name"])
def test_identity_columns_are_untouched(column):
    """pdb_path in particular: dedup, FoldSeek diversity and the pooled report
    all key on it, and it is not a metric."""
    assert rename(column) == column
    assert classify(column)[0] == "identity"


@pytest.mark.parametrize("column", ["self_sequence", "mpnn_pass_all", "self_aa_counts", "mpnn_esm_log_likelihood"])
def test_sequence_level_columns_are_untouched(column):
    assert rename(column) == column
    assert classify(column)[0] == "sequence-level"


def test_aggregates_keep_their_backend_suffix_by_decision():
    """_res_* are run-level rollups, not per-structure metrics. Out of scope
    deliberately -- including their backend-as-suffix form."""
    assert rename("_res_scRMSD_ca_esmfold2") == "_res_scRMSD_ca_esmfold2"
    assert classify("_res_scRMSD_ca_esmfold2")[0] == "aggregate"


@pytest.mark.parametrize("column", sorted(RETIRED_COLUMNS))
def test_the_p_sea_columns_are_retired_not_renamed(column):
    """P-SEA claimed 877 beta residues where mkdssp confirmed 178, so its sheet
    fraction was never safe to report."""
    assert rename(column) is None
    assert rename(column + "_all") is None


# --------------------------------------------------------------------------
# Coverage against a real header
# --------------------------------------------------------------------------


def test_every_real_column_resolves_to_a_named_rule():
    """The migration's own assertion. A missed rename otherwise looks exactly
    like a deliberate leave-alone, which is how 134 renames go wrong."""
    unclassified = [c for c in real_columns() if classify(c)[0] == "UNCLASSIFIED"]
    assert not unclassified, unclassified


def test_the_real_header_migrates_to_the_documented_totals():
    columns = real_columns()
    assert len(columns) == 217
    mapping = rename_map(columns)
    renamed = {k: v for k, v in mapping.items() if v is not None}
    retired = [k for k, v in mapping.items() if v is None]
    assert len(renamed) == 134
    assert len(retired) == 3
    assert len(columns) - len(mapping) == 80


def test_no_two_columns_rename_onto_one_name():
    """A collision would silently merge two metrics into one column."""
    columns = real_columns()
    renamed = {k: v for k, v in rename_map(columns).items() if v is not None}
    assert len(set(renamed.values())) == len(renamed)
    survivors = {c for c in columns if c not in rename_map(columns)}
    assert not (set(renamed.values()) & survivors), "a new name must not land on a column that stayed"


def test_renaming_is_idempotent():
    """Applying the map to already-migrated names must not move them again --
    the property that makes it safe to run over a mixed-vintage header."""
    for old in real_columns():
        new = rename(old)
        if new is not None:
            assert rename(new) == new, f"{old} -> {new} -> {rename(new)}"


def test_every_backend_the_scheme_knows_can_appear_in_a_name():
    for backend in BACKENDS:
        kwargs = {"kind": "complex", "backend": backend, "scope": "binder"}
        if backend != "generated":
            kwargs["seq_type"] = "self"
        assert backend in metric_column("ss_counts", **kwargs)
