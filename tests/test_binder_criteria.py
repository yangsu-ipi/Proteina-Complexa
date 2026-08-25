"""Success criteria: the pass vector, per-model expansion, and column alignment.

Three of the five findings in the a531c92..bbfc3ba review were in this area, and
all three were alignment or degradation bugs rather than arithmetic ones -- a
metric paired with the wrong sequence, a criterion silently dropped. Those are the
properties asserted here: not "does it compute 1.5 correctly" but "does index i
mean the same sequence everywhere, and does an unanswerable question stay
unanswered".
"""

import math

import pytest

from proteinfoundation.evaluation.binder_eval_utils import (
    check_thresholds_are_computable,
    per_sequence_pass,
    resolve_success_thresholds,
)
from proteinfoundation.result_analysis.analysis_utils import parse_threshold_spec
from proteinfoundation.result_analysis.binder_analysis_utils import (
    DEFAULT_LIGAND_BINDER_THRESHOLDS,
    DEFAULT_PROTEIN_BINDER_THRESHOLDS,
    MODEL_PLACEHOLDER,
    build_column_name,
    check_sample_has_passing_redesign,
    count_passing_redesigns,
    expand_model_criteria,
    normalize_threshold_dict,
    redesign_pass_vector,
)

PROTEIN = normalize_threshold_dict(DEFAULT_PROTEIN_BINDER_THRESHOLDS)

# redesign_pass_vector takes criteria that are already expanded -- the {model}
# template is resolved one layer up, against the columns. Feeding it the raw
# default set would silently fail every sequence on a criterion the caller never
# supplied, so the primitive's tests use the holo three explicitly.
HOLO = {name: spec for name, spec in PROTEIN.items() if MODEL_PLACEHOLDER not in name}
PARSED = {name: parse_threshold_spec(spec) for name, spec in HOLO.items()}


def metric_values(i_pae, plddt, scrmsd, apo=None):
    """One design's *_all lists, keyed the way redesign_pass_vector expects."""
    values = {"i_pAE": i_pae, "pLDDT": plddt, "scRMSD_ca": scrmsd}
    for model, vals in (apo or {}).items():
        values[f"scRMSD_ca_{model}"] = vals
    return values


def row(seq_type="mpnn", *, i_pae=(0.10,), plddt=(0.95,), scrmsd=(1.0,), apo=None):
    """One design's row_dict, keyed by real column names."""
    out = {
        f"{seq_type}_complex_i_pAE_all": list(i_pae),
        f"{seq_type}_complex_pLDDT_all": list(plddt),
        f"{seq_type}_binder_scRMSD_ca_all": list(scrmsd),
    }
    for model, vals in (apo or {}).items():
        out[f"{seq_type}_apo_scRMSD_ca_{model}_all"] = vals
    return out


# ---------------------------------------------------------------- pass vector


def test_vector_is_the_primitive_the_reductions_agree_with():
    """any() and sum() of the vector are what the two older helpers return.

    They were rewritten as reductions of it precisely so a redesign cannot be a
    failure in one place and a success in another.
    """
    values = metric_values([0.10, 0.30, 0.20], [0.95, 0.95, 0.80], [1.0, 1.0, 1.0])
    vector = redesign_pass_vector(values, PARSED)
    assert vector == [1, 0, 0]
    assert check_sample_has_passing_redesign(values, PARSED) is any(vector)
    assert count_passing_redesigns(values, PARSED) == sum(vector)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "n/a"])
def test_unusable_values_fail_rather_than_raise(bad):
    """A NaN i_pAE is not a pass. Nor is it a crash mid-campaign."""
    assert redesign_pass_vector(metric_values([bad], [0.95], [1.0]), PARSED) == [0]


def test_every_criterion_must_pass():
    """One failing metric fails the sequence, whichever it is."""
    assert redesign_pass_vector(metric_values([0.10], [0.95], [1.0]), PARSED) == [1]
    assert redesign_pass_vector(metric_values([0.90], [0.95], [1.0]), PARSED) == [0]
    assert redesign_pass_vector(metric_values([0.10], [0.10], [1.0]), PARSED) == [0]
    assert redesign_pass_vector(metric_values([0.10], [0.95], [9.0]), PARSED) == [0]


def test_ragged_metric_lists_judge_the_covered_prefix():
    """A misalignment costs the unjudgeable tail, not an exception from inside a
    metric computation."""
    values = metric_values([0.10, 0.10, 0.10], [0.95, 0.95], [1.0, 1.0, 1.0])
    assert redesign_pass_vector(values, PARSED) == [1, 1]


def test_empty_input_is_empty_not_a_pass():
    assert redesign_pass_vector({}, PARSED) == []
    assert count_passing_redesigns({}, PARSED) == 0


# ------------------------------------------------------- per-model expansion


@pytest.mark.parametrize("models", [["esmfold"], ["esmfold2"], ["esmfold", "esmfold2"], ["colabfold", "esmfold2"]])
def test_apo_criterion_follows_the_models_actually_produced(models):
    """The criterion is templated so no threshold override is needed when
    apo_folding_models changes."""
    columns = list(row(apo={m: [1.0] for m in models}))
    expanded = expand_model_criteria(PROTEIN, "mpnn", columns)
    apo = sorted(k for k in expanded if k.startswith("scRMSD_ca_"))
    assert apo == sorted(f"scRMSD_ca_{m}" for m in models)
    assert len(expanded) == 3 + len(models)


def test_two_apo_models_must_both_pass():
    """Conjunctive, like the three holo criteria: if two predictors disagree about
    whether the binder folds alone, that is not a pass."""
    both = row(
        i_pae=[0.1] * 3,
        plddt=[0.95] * 3,
        scrmsd=[1.0] * 3,
        apo={"esmfold": [1.0, 1.0, 3.0], "esmfold2": [1.0, 3.0, 1.0]},
    )
    assert per_sequence_pass(both, "mpnn", PROTEIN) == [1, 0, 0]


def test_expansion_does_not_leak_across_sequence_types():
    """self and mpnn can be folded by different model sets in one run."""
    columns = list(row("self", apo={"esmfold": [1.0]})) + list(row("mpnn", apo={"esmfold": [1.0], "esmfold2": [1.0]}))
    assert sorted(k for k in expand_model_criteria(PROTEIN, "self", columns) if k.startswith("scRMSD_ca_")) == [
        "scRMSD_ca_esmfold"
    ]
    assert sorted(k for k in expand_model_criteria(PROTEIN, "mpnn", columns) if k.startswith("scRMSD_ca_")) == [
        "scRMSD_ca_esmfold",
        "scRMSD_ca_esmfold2",
    ]


def test_holo_scrmsd_is_never_mistaken_for_an_apo_model():
    """binder_scRMSD_ca and apo_scRMSD_ca_<model> differ only in prefix."""
    expanded = expand_model_criteria(PROTEIN, "mpnn", list(row(apo={"esmfold": [1.0]})))
    assert "scRMSD_ca" in expanded
    assert parse_threshold_spec(expanded["scRMSD_ca"])["column_prefix"] == "binder"


def test_unmatched_template_is_kept_so_the_gate_cannot_quietly_shrink():
    """Dropping it would leave three criteria to be evaluated alone, and a design
    passing a three-criterion gate must not look like one that passed four."""
    columns = list(row())  # no apo columns
    expanded = expand_model_criteria(PROTEIN, "mpnn", columns)
    assert any(MODEL_PLACEHOLDER in name for name in expanded)


def test_missing_apo_column_yields_no_verdict_rather_than_a_pass():
    unjudgeable = row()  # would pass the holo three
    assert per_sequence_pass(unjudgeable, "mpnn", PROTEIN) is None


# ------------------------------------------------------------ column contract


@pytest.mark.parametrize("seq_type", ["self", "mpnn", "mpnn_fixed"])
def test_threshold_machinery_builds_the_column_apo_refolding_emits(seq_type):
    """The gate is a config entry only because these two agree exactly."""
    from proteinfoundation.evaluation.binder_eval_utils import apo_column

    assert build_column_name(seq_type, "apo", "scRMSD_ca_esmfold") == apo_column(seq_type, "ca", "esmfold") + "_all"


def test_per_sequence_pass_reads_the_columns_it_claims_to():
    """Index i of the verdict must describe index i of every other _all list."""
    r = row(i_pae=[0.10, 0.90], plddt=[0.95, 0.95], scrmsd=[1.0, 1.0], apo={"esmfold": [1.0, 1.0]})
    assert per_sequence_pass(r, "mpnn", PROTEIN) == [1, 0]
    r["mpnn_apo_scRMSD_ca_esmfold_all"] = [9.0, 1.0]
    assert per_sequence_pass(r, "mpnn", PROTEIN) == [0, 0]


# --------------------------------------------------------- criteria resolution


def test_ligand_defaults_carry_no_apo_criterion():
    """Apo folding is skipped for ligand targets, so gating on it would remove
    gating entirely."""
    assert not any("apo" in str(spec) for spec in DEFAULT_LIGAND_BINDER_THRESHOLDS.values())


def test_resolve_reads_the_key_analysis_reads():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"aggregation": {"success_thresholds": None}})
    assert set(resolve_success_thresholds(cfg, is_target_ligand=False)) == set(PROTEIN)
    assert set(resolve_success_thresholds(cfg, is_target_ligand=True)) == set(
        normalize_threshold_dict(DEFAULT_LIGAND_BINDER_THRESHOLDS)
    )
    assert resolve_success_thresholds(OmegaConf.create({}), False) == resolve_success_thresholds(cfg, False)


def test_custom_thresholds_replace_rather_than_merge():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {"aggregation": {"success_thresholds": {"iptm": {"threshold": 0.8, "op": ">=", "column_prefix": "complex"}}}}
    )
    assert list(resolve_success_thresholds(cfg, False)) == ["i_pTM"]  # normalised, and alone


@pytest.mark.parametrize(
    "compute_apo,modes,models,expect_error",
    [
        (True, ["ca"], ["esmfold"], False),
        (True, ["ca"], ["esmfold2"], False),
        (True, ["ca"], ["esmfold", "esmfold2"], False),
        (False, ["ca"], ["esmfold"], True),
        (True, ["bb3o"], ["esmfold"], True),
    ],
)
def test_startup_check_reports_criteria_this_run_cannot_evaluate(caplog, compute_apo, modes, models, expect_error):
    """The model half is filled in from the columns, so only apo being off or a
    mode mismatch can make the criterion unsatisfiable."""
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(m), level="ERROR")
    try:
        check_thresholds_are_computable(PROTEIN, compute_apo, apo_rmsd_modes=modes, apo_folding_models=models)
    finally:
        logger.remove(sink)
    assert bool(seen) is expect_error


def test_math_isfinite_is_what_the_vector_relies_on():
    """Guards the assumption behind the NaN cases above."""
    assert not math.isfinite(float("nan")) and not math.isfinite(float("inf"))
