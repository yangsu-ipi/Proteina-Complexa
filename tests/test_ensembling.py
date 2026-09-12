"""How several predictions collapse into one number.

AF2-Multimer has five parameter sets that disagree, and the disagreement is
information. These tests pin the reductions that turn those draws into the
values a gate later reads -- including the cases where a draw is missing or
unusable, which is where a reduction quietly invents a number if it can.
"""

import math

import pytest

from proteinfoundation.metrics.ensembling import (
    AF2_STAT_PRECISION,
    PLACEMENT_METRICS,
    af2_stats_from_metrics,
    average_af2_stats,
    mean_chain_plddt,
    mean_plddt_from_pdb,
    per_model_paths_from_first,
    pop_per_model_paths,
    reduce_rmsd_over_models,
    residue_weighted_mean,
)

RAW = {
    "plddt": 0.9,
    "ptm": 0.8,
    "i_ptm": 0.7,
    "pae": 6.0,
    "i_pae": 5.0,
    "min_ipae": 4.0,
    "min_ipsae": 0.3,
    "max_ipsae": 0.6,
    "avg_ipsae": 0.45,
}


def test_one_model_is_what_the_single_model_path_produced():
    """Ensembling must not move the numbers of a run that asked for one model."""
    stats = average_af2_stats([af2_stats_from_metrics(RAW)])
    assert stats["pTM"] == round(RAW["ptm"], 3)
    assert stats["i_pAE"] == round(RAW["i_pae"], 3)
    assert stats["avg_ipSAE"] == round(RAW["avg_ipsae"], 4)


def test_absent_ipsae_10_reads_as_zero_rather_than_raising():
    """RAW has no *_10 keys, matching older colabdesign builds."""
    assert af2_stats_from_metrics(RAW)["min_ipSAE_10"] == 0.0


def test_scores_are_meaned_not_best_of():
    """Taking the best model would discard the disagreement that is the whole
    reason for running five of them."""
    good = af2_stats_from_metrics({**RAW, "i_ptm": 0.95})
    bad = af2_stats_from_metrics({**RAW, "i_ptm": 0.55})
    assert average_af2_stats([good, bad])["i_pTM"] == pytest.approx(0.75)


def test_the_mean_is_taken_before_rounding():
    """Rounding each model first would average five rounding errors too."""
    models = [af2_stats_from_metrics({**RAW, "avg_ipsae": v}) for v in (0.111151, 0.111153)]
    assert average_af2_stats(models)["avg_ipSAE"] == round((0.111151 + 0.111153) / 2, 4)


def test_every_stat_keeps_the_precision_it_had():
    stats = average_af2_stats([af2_stats_from_metrics(RAW)])
    assert set(stats) <= set(AF2_STAT_PRECISION)
    assert set(AF2_STAT_PRECISION) - set(stats) == {"target_pLDDT", "binder_pLDDT"}, (
        "the per-chain means come from the per-residue array, not the log dict"
    )
    assert "pLDDT" not in AF2_STAT_PRECISION, "collapsed into binder_pLDDT, the number ColabDesign's log always held"


def test_a_backend_reporting_no_per_residue_confidence_still_averages():
    """average_af2_stats must not require the per-chain keys of every caller --
    a model that exposes no per-residue pLDDT should lose those two columns,
    not raise."""
    stats = average_af2_stats([af2_stats_from_metrics(RAW), af2_stats_from_metrics(RAW)])
    assert "target_pLDDT" not in stats
    assert "binder_pLDDT" not in stats
    assert stats["pTM"] == round(RAW["ptm"], 3)


def test_per_chain_means_average_over_models_like_any_other_score():
    per_model = [
        {**af2_stats_from_metrics(RAW), **mean_chain_plddt([0.9] * 4 + [0.8] * 2, 4)},
        {**af2_stats_from_metrics(RAW), **mean_chain_plddt([0.9] * 4 + [0.6] * 2, 4)},
    ]
    assert average_af2_stats(per_model)["binder_pLDDT"] == pytest.approx(0.7)


def test_the_target_and_binder_are_split_at_the_boundary():
    """ColabDesign orders the binder protocol target-first, which is what the
    binder-only losses already assume via _target_len."""
    split = mean_chain_plddt([1.0, 1.0, 1.0, 0.5, 0.3], target_len=3)
    assert split["target_pLDDT"] == pytest.approx(1.0)
    assert split["binder_pLDDT"] == pytest.approx(0.4)


def test_the_complex_mean_hides_a_bad_binder_that_the_split_shows():
    """The reason this exists. A CBLN1-shaped complex is mostly target, so the
    target folding as well as it always does carries the average past a
    threshold the binder comes nowhere near."""
    plddt = [0.97] * 136 + [0.55] * 44
    split = mean_chain_plddt(plddt, target_len=136)
    assert sum(plddt) / len(plddt) > 0.86, "complex mean is dragged up by the target"
    assert split["binder_pLDDT"] == pytest.approx(0.55), "the binder is plainly bad"


def test_a_0_to_100_array_is_read_as_fractions():
    """The gates compare against 0.9. An unnormalised array would clear every
    threshold for every design ever scored, and look like a great campaign."""
    split = mean_chain_plddt([97.0] * 3 + [55.0] * 2, target_len=3)
    assert split["target_pLDDT"] == pytest.approx(0.97)
    assert split["binder_pLDDT"] == pytest.approx(0.55)


def test_no_usable_boundary_emits_nothing():
    """Better an absent column than the whole complex labelled as one chain."""
    assert mean_chain_plddt([0.9, 0.8], target_len=None) == {}
    assert mean_chain_plddt(None, 3) == {}
    assert mean_chain_plddt([0.9, 0.8], target_len=0) == {}
    assert mean_chain_plddt([0.9, 0.8], target_len=2) == {}, "no binder residues left"
    assert mean_chain_plddt([0.9, 0.8], target_len=5) == {}


def test_a_nonfinite_residue_does_not_erase_its_chain():
    split = mean_chain_plddt([0.9, float("nan"), 0.7, 0.5], target_len=3)
    assert split["target_pLDDT"] == pytest.approx(0.8)


def test_fold_quality_is_meaned_over_the_models():
    """The spread between models is uncertainty about one structure."""
    per_model = [{"binder_scRMSD_ca": 1.0}, {"binder_scRMSD_ca": 3.0}]
    assert reduce_rmsd_over_models(per_model)["binder_scRMSD_ca"] == pytest.approx(2.0)


def test_placement_takes_the_worst_model_not_the_typical_one():
    """A binder that lands correctly in one model of five has not been placed
    correctly. Meaning it instead let 24 sequences cross from failing
    complex_scRMSD_ca to passing it on a real campaign, twelve of them from
    4-8 A on a single model -- the 2.0 A thresholds were calibrated against
    single-model geometry, and a mean pulls exactly that band to the cutoff."""
    per_model = [{"complex_scRMSD_ca": 0.5}, {"complex_scRMSD_ca": 0.6}, {"complex_scRMSD_ca": 9.0}]
    assert reduce_rmsd_over_models(per_model)["complex_scRMSD_ca"] == pytest.approx(9.0)

    aligned = [{"binder_scRMSD_target_aligned_ca": 1.0}, {"binder_scRMSD_target_aligned_ca": 4.0}]
    assert reduce_rmsd_over_models(aligned)["binder_scRMSD_target_aligned_ca"] == pytest.approx(4.0)


def test_the_legacy_alias_reduces_like_the_metric_it_aliases():
    """complex_scRMSD is the same number as complex_scRMSD_ca. Reducing them
    differently would put two answers to one question on the same row."""
    per_model = [
        {"complex_scRMSD": 0.5, "complex_scRMSD_ca": 0.5},
        {"complex_scRMSD": 9.0, "complex_scRMSD_ca": 9.0},
    ]
    out = reduce_rmsd_over_models(per_model)
    assert out["complex_scRMSD"] == out["complex_scRMSD_ca"] == pytest.approx(9.0)


def test_a_placement_reduction_never_flatters_a_single_bad_model():
    """Strictly harder to satisfy than the single model the gate used to read."""
    per_model = [{"complex_scRMSD_ca": v} for v in (0.4, 0.5, 0.6, 0.7, 2.4)]
    reduced = reduce_rmsd_over_models(per_model)["complex_scRMSD_ca"]
    assert reduced >= max(m["complex_scRMSD_ca"] for m in per_model[:1]), "not below model 1"
    assert reduced == pytest.approx(2.4), "and it fails the 2.0 gate the mean would have passed"


def test_one_unusable_model_does_not_erase_the_others():
    """A NaN placement is a missing measurement, not a bad one: dropping it
    keeps four good models answerable, while averaging it in would leave the
    design with no number and no reason visible for why."""
    per_model = [{"binder_scRMSD_ca": 1.0}, {"binder_scRMSD_ca": float("nan")}, {"binder_scRMSD_ca": 2.0}]
    assert reduce_rmsd_over_models(per_model)["binder_scRMSD_ca"] == pytest.approx(1.5)


def test_nothing_finite_stays_nan_rather_than_becoming_zero():
    """A zero RMSD passes every gate there is."""
    per_model = [{"complex_scRMSD_ca": float("nan")}, {"complex_scRMSD_ca": float("inf")}]
    assert math.isnan(reduce_rmsd_over_models(per_model)["complex_scRMSD_ca"])


def test_a_single_model_result_is_passed_through_untouched():
    """Backends that predict one structure must not be reshaped by a reduction
    they never asked for -- including any non-numeric fields they carry."""
    only = {"complex_scRMSD_ca": 1.25, "note": "rf3"}
    assert reduce_rmsd_over_models([only]) is only


def test_no_models_is_an_empty_result():
    assert reduce_rmsd_over_models([]) == {}


def test_per_model_paths_are_removed_as_they_are_read():
    """They must not reach the dataframe: a list-valued column survives no
    round-trip through CSV."""
    stats = {"seq_1": {"pLDDT": 0.9, "complex_pdb_paths": ["a.pdb", "b.pdb"]}}
    assert pop_per_model_paths([stats], 0) == ["a.pdb", "b.pdb"]
    assert "complex_pdb_paths" not in stats["seq_1"]
    assert pop_per_model_paths([stats], 0) is None, "already taken"


def test_a_backend_without_per_model_paths_reports_nothing():
    """RF3 and Boltz return the same shape without the key, and must fall back
    to their single structure rather than raise inside the geometry loop."""
    assert pop_per_model_paths([{"seq_1": {"pLDDT": 0.9}}], 0) is None
    assert pop_per_model_paths([], 0) is None
    assert pop_per_model_paths(["not a dict"], 0) is None
    assert pop_per_model_paths([{"seq_1": None}], 0) is None


# ---------------------------------------------------------------------------
# Wiring. A knob that only reaches some of its call sites is worse than no knob
# -- the seed count shipped that way once, and a default of 1 made every missed
# site look deliberate. These read source because the modules they check import
# torch and colabdesign.
# ---------------------------------------------------------------------------

import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (SRC / rel).read_text()


def test_the_campaign_config_asks_for_all_five_models():
    assert "n_af2_models: 5" in _read("configs/pipeline/binder/binder_evaluate.yaml")


def test_the_model_count_invalidates_the_binder_eval_cache():
    """Both the confidence scores and the geometry are means over this many
    models, so a cache written at one count cannot answer for another. Without
    this the count could be raised and every cached design would keep serving
    its single-model numbers."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    base = source[source.index("cache_fingerprint_base = {") : source.index("n_reused = 0")]
    assert '"n_af2_models": n_af2_models' in base


def test_the_count_reaches_the_folding_call():
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    assert 'cfg_metric.get("n_af2_models"' in source, "read from config"
    assert "n_af2_models=n_af2_models," in source, "passed to run_binder_eval"
    assert "get_af2_advanced_settings(num_af2_models=n_af2_models)" in _read(
        "src/proteinfoundation/metrics/binder_metrics.py"
    )


def test_the_prediction_loop_is_not_pinned_to_the_first_model():
    """It was ``model_num = 0`` with ``models=[model_num]``, which reads like a
    loop variable and is not one."""
    source = _read("src/proteinfoundation/utils/colabdesign_utils.py")
    body = source[source.index("def predict_binder_complex(") :]
    assert "for model_num in range(n_models)" in body
    assert "model_num = 0" not in body


def test_every_model_gets_its_own_structure_file():
    """Per-chain pLDDT is read back off these later, so five models overwriting
    one path would silently become one model measured five times."""
    body = _read("src/proteinfoundation/utils/colabdesign_utils.py")
    body = body[body.index("def predict_binder_complex(") :]
    assert "_model{model_num + 1}.pdb" in body
    assert "complex_pdb_paths.append(complex_pdb)" in body


def test_per_chain_plddt_is_computed_for_every_model():
    """Per-model, not once per design: the split has to average over the five
    models like every other score, which it cannot do from a single call."""
    source = _read("src/proteinfoundation/utils/colabdesign_utils.py")
    body = source[source.index("for model_num in range(n_models)") : source.index("stats = average_af2_stats")]
    assert "mean_chain_plddt(" in body
    assert 'aux.get("plddt")' in body
    assert '"_target_len"' in body


def test_the_column_a_gate_would_look_for_is_the_column_produced():
    """binder_eval names complex columns f"{seq_type}_complex_{metric}_all", and
    a threshold spec builds its column from column_prefix plus metric. These are
    two derivations of one name written in different files, which is how the
    last set of gate columns went missing."""
    from proteinfoundation.result_analysis.binder_analysis_utils import build_column_name

    for metric in ("target_pLDDT", "binder_pLDDT"):
        produced = f"mpnn_complex_af2_{metric}_all"
        assert build_column_name("mpnn", "complex", metric) == produced


# ---------------------------------------------------------------------------
# Reading confidence back off a structure. The folding backends write
# per-residue pLDDT into the B-factor column, which is where it lives once
# the model has been unloaded.
# ---------------------------------------------------------------------------


def pdb_with_plddt(path: pathlib.Path, values: list[float]) -> str:
    """A minimal CA-only PDB with pLDDT in the B-factor column, as the folding
    backends write it."""
    lines = []
    for i, value in enumerate(values, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    {0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{value:>6.2f}           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)


def test_plddt_is_read_out_of_the_b_factor_column(tmp_path):
    path = pdb_with_plddt(tmp_path / "f.pdb", [0.9, 0.8, 0.7])
    assert mean_plddt_from_pdb(path) == pytest.approx(0.8)


def test_a_0_to_100_structure_is_read_as_fractions(tmp_path):
    """ColabFold writes pLDDT on 0-100. A gate comparing to 0.9 would pass every
    design ever folded."""
    path = pdb_with_plddt(tmp_path / "f.pdb", [90.0, 80.0, 70.0])
    assert mean_plddt_from_pdb(path) == pytest.approx(0.8)


def test_an_unreadable_structure_is_nan_not_zero(tmp_path):
    """Zero would make every ratio against it infinite; a missing file must stay
    missing."""
    assert math.isnan(mean_plddt_from_pdb(str(tmp_path / "absent.pdb")))
    assert math.isnan(mean_plddt_from_pdb(pdb_with_plddt(tmp_path / "empty.pdb", [])))


def test_chains_are_weighted_by_their_length():
    """The target's pLDDT in complex is one mean over all its residues, so the
    reference has to be too -- otherwise the ratio measures the chain split."""
    assert residue_weighted_mean([1.0, 0.0], [3, 1]) == pytest.approx(0.75)
    assert residue_weighted_mean([1.0, 0.0], [1, 3]) == pytest.approx(0.25)


def test_the_advisory_scorer_splits_its_plddt_too():
    """mean_chain_plddt has two call sites now. The seed count shipped with one
    of four wired, so a second call site is worth a test of its own."""
    from proteinfoundation.metrics.consensus_folding import _esmfold2_metrics

    class Result:
        plddt = [0.94] * 4 + [0.60] * 2
        ptm = None
        iptm = None
        pae = None

    metrics = _esmfold2_metrics(Result(), target_len=4)
    assert metrics["target_pLDDT"] == pytest.approx(0.94)
    assert metrics["binder_pLDDT"] == pytest.approx(0.60)
    assert metrics["pLDDT"] == pytest.approx(sum(Result.plddt) / 6), "the complex mean still stands"


def test_the_per_chain_metrics_are_emitted_at_all():
    """Column emission filters on CONSENSUS_METRIC_SUFFIXES, so a metric the
    scorer computes but the tuple omits never reaches a column."""
    from proteinfoundation.metrics.consensus_folding import CONSENSUS_METRIC_SUFFIXES

    assert {"target_pLDDT", "binder_pLDDT"} <= set(CONSENSUS_METRIC_SUFFIXES)


def test_the_advisory_per_chain_columns_cannot_be_mistaken_for_gated_ones():
    """ESMFold2 runs on a compressed scale -- a native protein folds to ~0.65 --
    so these must stay out of any gate. The guard is what enforces that."""
    from proteinfoundation.metrics.consensus_folding import advisory_column, assert_columns_are_advisory

    columns = [advisory_column("mpnn", "esmfold2", m) for m in ("target_pLDDT", "binder_pLDDT")]
    gated = {"mpnn_complex_af2_target_pLDDT", "mpnn_complex_af2_binder_pLDDT", "mpnn_complex_af2_pLDDT"}
    assert_columns_are_advisory(columns, gated)


def _cbln1_row():
    """A row shaped like the CBLN1 campaign's: esmfold2 folds the apo structures
    the fourth criterion gates on, AND provides the advisory second opinion.

    Criteria read the ``*_all`` lists, and ``{model}`` expands against the columns
    a row actually carries, so the apo column has to be present for the apo
    criterion to resolve to anything.
    """
    return {
        "self_complex_af2_i_pAE_all": [0.1],
        "self_complex_af2_binder_pLDDT_all": [0.95],
        "self_complex_af2_binder_scRMSD_ca_all": [1.0],
        "self_complex_af2_scRMSD_ca_all": [1.0],
        "self_complex_af2_binder_scRMSD_target_aligned_ca_all": [1.0],
        "self_apo_esmfold2_binder_scRMSD_ca_all": [1.5],
    }


def test_a_model_can_serve_the_apo_gate_and_the_advisory_track_at_once():
    """The regression. esmfold2 is both `apo_folding_models` and
    `consensus_backends` on CBLN1, and the apo criterion is gated on purpose. The
    check used to scan gated columns for the substring `_esmfold2_`; once the
    rename moved the model into the backend slot, the deliberate, gated
    `self_apo_esmfold2_binder_scRMSD_ca` matched, and every evaluate run under
    that config died on its first design after loading its models."""
    from proteinfoundation.evaluation.binder_eval_utils import gated_columns
    from proteinfoundation.metrics.consensus_folding import (
        CONSENSUS_METRIC_SUFFIXES,
        advisory_column,
        assert_columns_are_advisory,
    )
    from proteinfoundation.result_analysis.binder_analysis_utils import DEFAULT_PROTEIN_BINDER_THRESHOLDS

    row = _cbln1_row()
    gated = gated_columns(row, "self", DEFAULT_PROTEIN_BINDER_THRESHOLDS)
    assert "self_apo_esmfold2_binder_scRMSD_ca_all" in gated, "the apo criterion is gated on purpose"

    advisory = [advisory_column("self", "esmfold2", m) for m in CONSENSUS_METRIC_SUFFIXES]
    advisory += [f"{c}_all" for c in advisory]
    advisory.append(advisory_column("self", "esmfold2", "pdb_path"))
    assert_columns_are_advisory(advisory, gated, set(row))


def test_a_gate_that_really_reads_an_advisory_column_still_fails():
    """The check has to keep refusing what it exists for, not merely stop
    refusing the apo case."""
    from proteinfoundation.metrics.consensus_folding import advisory_column, assert_columns_are_advisory

    col = advisory_column("self", "esmfold2", "i_pAE")
    with pytest.raises(ValueError, match="pass criterion reads advisory columns"):
        assert_columns_are_advisory([col], {col})


def test_an_advisory_column_may_not_overwrite_one_already_built():
    from proteinfoundation.metrics.consensus_folding import advisory_column, assert_columns_are_advisory

    col = advisory_column("self", "esmfold2", "i_pAE")
    with pytest.raises(ValueError, match="collide with columns already built"):
        assert_columns_are_advisory([col], set(), {col})


def test_complex_plddt_is_gone_and_still_addressable():
    """It was one number under two names: ColabDesign's log["plddt"] is the
    binder-only mean, matching the per-residue split to the last digit on a real
    run. The name that survives is the one that says so -- but an existing
    success_thresholds override naming the old one must keep gating what it
    meant, not silently drop a criterion."""
    from proteinfoundation.result_analysis.binder_analysis_utils import (
        DEFAULT_PROTEIN_BINDER_THRESHOLDS,
        normalize_threshold_dict,
    )

    assert "complex_pLDDT" not in DEFAULT_PROTEIN_BINDER_THRESHOLDS
    assert "complex_binder_pLDDT" in DEFAULT_PROTEIN_BINDER_THRESHOLDS

    for spec in DEFAULT_PROTEIN_BINDER_THRESHOLDS.values():
        assert spec.get("metric") != "pLDDT", "nothing gates the retired suffix"

    for alias in ("complex_pLDDT", "complex_plddt"):
        normalized = normalize_threshold_dict({alias: {"threshold": 0.8, "op": ">="}})
        assert "complex_binder_pLDDT" in normalized, f"{alias} still lands on a real criterion"
        assert normalized["complex_binder_pLDDT"]["threshold"] == 0.8, "and keeps its threshold"


def test_the_refolded_path_lookup_builds_its_column_rather_than_guessing():
    """This test used to assert that three metric columns in this file had been
    repointed. They had -- but they were inside a function nothing called, while
    the live path lookup a few lines above went on guessing among four candidate
    names, three of which never existed in any frame. The dead function is gone
    (it also carried its own copy of the success criteria, three criteria out of
    date); what is checked now is the code that runs."""
    source = (SRC / "src/proteinfoundation/utils/refolded_structure_utils.py").read_text()
    assert "get_successful_best_samples_with_paths" not in source
    assert "possible_columns" not in source, "no candidate list -- one name, built once"
    assert 'rename(f"{t}_complex_pdb_path", backend)' in source
    assert "complex_backend_of(df)" in source, "the backend comes from the frame's provenance"
    # And finding nothing is loud, because that is what made it invisible: a run
    # asking for refolded interface metrics got none, with only a debug line.
    assert "logger.error(" in source
    assert "if not any(paths.values()):" in source


def _frame_with_refold_paths(tmp_path, slots_by_type, scalar=False):
    """A frame shaped the way evaluate emits it: per-sequence path lists."""
    import pandas as pd

    from proteinfoundation.result_analysis.binder_analysis_utils import COMPLEX_BACKEND_COLUMN

    design = tmp_path / "design_0.pdb"
    design.write_text("")
    row = {"pdb_path": str(design), COMPLEX_BACKEND_COLUMN: "af2"}
    for seq_type, slots in slots_by_type.items():
        paths = []
        for i, present in enumerate(slots):
            if not present:
                paths.append(None)
                continue
            path = tmp_path / f"{seq_type}_seq{i}_model1.pdb"
            path.write_text("")
            paths.append(str(path))
        column = f"{seq_type}_complex_af2_pdb_path"
        row[f"{column}_all"] = paths
        if scalar:
            row[column] = next((p for p in paths if p), None)
    return pd.DataFrame([row])


def test_the_path_lookup_returns_every_redesign_not_the_ranked_one(tmp_path):
    """The asymmetry this removes: AF2 measured one redesign's interface while
    the advisory backend measured all of them, so the two backends' interface
    numbers could describe different sequences."""
    from proteinfoundation.utils.refolded_structure_utils import extract_refolded_structure_paths_from_df

    df = _frame_with_refold_paths(tmp_path, {"self": [True], "mpnn": [True, True, True]})
    got = extract_refolded_structure_paths_from_df(df, sequence_types=["self", "mpnn"])

    assert len(got["design_0"]["mpnn"]) == 3, "every redesign, not the best one"
    assert len(got["design_0"]["self"]) == 1


def test_the_path_lookup_does_not_need_the_headline_scalar(tmp_path):
    """The regression. Evaluate stopped writing {seq}_complex_{backend}_pdb_path
    when ranking moved to analyze; this lookup still read it, so every metric
    computed on a refolded structure vanished from the run -- loudly, but
    vanished."""
    from proteinfoundation.utils.refolded_structure_utils import extract_refolded_structure_paths_from_df

    df = _frame_with_refold_paths(tmp_path, {"self": [True], "mpnn": [True, True]}, scalar=False)
    assert not [c for c in df.columns if c.endswith("pdb_path") and not c.endswith("_all")][1:], (
        "the frame carries no headline path column beyond the design's own"
    )

    got = extract_refolded_structure_paths_from_df(df, sequence_types=["self", "mpnn"])
    assert got["design_0"]["self"] and got["design_0"]["mpnn"]


def test_a_redesign_with_no_structure_keeps_its_slot(tmp_path):
    """Dropping it would slide every later redesign's metrics onto the wrong
    sequence: slot i of these lists has to stay the sequence in
    {seq}_sequence_all[i]."""
    from proteinfoundation.utils.refolded_structure_utils import extract_refolded_structure_paths_from_df

    df = _frame_with_refold_paths(tmp_path, {"mpnn": [True, False, True]})
    got = extract_refolded_structure_paths_from_df(df, sequence_types=["mpnn"])

    slots = got["design_0"]["mpnn"]
    assert len(slots) == 3
    assert slots[1] is None
    assert slots[0] and slots[2]


def test_every_redesign_gets_its_own_interface_metrics(tmp_path, monkeypatch):
    """One list per metric, one entry per redesign, and no headline scalar --
    which redesign the row presents is analyze's call, like every other family."""
    from proteinfoundation.evaluation import binder_eval
    from proteinfoundation.utils.refolded_structure_utils import extract_refolded_structure_paths_from_df

    seen = []

    def fake_bio(pdb_path, binder_chain, target_chain):
        seen.append(pdb_path)
        return {"interface_sc": 0.5 + 0.01 * len(seen), "binder_ss_counts": [1.0] * 8}

    monkeypatch.setattr(binder_eval, "compute_bioinformatics_metrics_single", fake_bio)
    monkeypatch.setattr(
        binder_eval, "get_binder_chain_from_complex", lambda path, return_multi_target=False: ("B", ["A"], False)
    )
    monkeypatch.setattr(binder_eval, "parse_cfg_for_table", lambda cfg: ([], {}))

    df = _frame_with_refold_paths(tmp_path, {"mpnn": [True, False, True]})
    paths = extract_refolded_structure_paths_from_df(df, sequence_types=["mpnn"])
    out = binder_eval.compute_interface_metrics_on_refolded_structures(
        df=df,
        paths_dict=paths,
        cfg_metric={"sequence_types": ["mpnn"], "binder_folding_method": "colabdesign"},
        cfg={},
        compute_bioinformatics=True,
        n_af2_models=1,
    )

    assert len(seen) == 2, "the missing slot is not folded for"
    values = out.at[0, "mpnn_complex_af2_interface_sc_all"]
    assert len(values) == 3, "aligned with the sequence list, missing slot included"
    assert values[0] == pytest.approx(0.51) and values[2] == pytest.approx(0.52)
    assert math.isnan(values[1])
    # The packed counts survive as a list per redesign, not averaged into one.
    assert out.at[0, "mpnn_complex_af2_binder_ss_counts_all"][0] == [1.0] * 8
    assert "mpnn_complex_af2_interface_sc" not in out.columns, "the headline is analyze's to choose"


def test_the_reduction_rule_is_derivation_not_structure():
    """Which structures get predicted, and how numbers are read off them, are two
    questions. Keeping the reduction out of the structure hash is what lets a
    reduction change reuse the structures instead of refolding to recompute an
    arithmetic choice."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    base = source[source.index("cache_fingerprint_base = {") : source.index("derivation_fingerprint =")]
    assert "GEOMETRY_REDUCTION_VERSION" not in base, "not part of structure identity"
    derivation = source[source.index("derivation_fingerprint = binder_eval_fingerprint(") :][:400]
    assert "geometry_reduction=GEOMETRY_REDUCTION_VERSION" in derivation


def test_a_stale_derivation_recomputes_instead_of_refolding():
    """The whole point of the split. A cache whose structures match but whose
    numbers came from another rule must be refreshed, not thrown away."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    assert "if derivation_stale:" in source
    # Anchored on the import, because "if derivation_stale:" also appears in the
    # reader that computes it -- and matching there would test the wrong branch.
    stale_block = source[source.index("import recompute_derived") :][:1600]
    assert "recompute_derived(" in stale_block
    assert "cached = None" in stale_block, "missing structures fall back to a refold"
    assert "write_binder_eval_cache(" in stale_block, "the refreshed numbers are persisted"


def test_every_gated_placement_criterion_is_in_the_placement_set():
    """The two must not drift apart: a criterion gating at 2.0 A against a mean
    is reading a different quantity than the one that threshold was set for."""
    from proteinfoundation.result_analysis.binder_analysis_utils import DEFAULT_PROTEIN_BINDER_THRESHOLDS

    # A threshold spec splits the column into prefix + metric; the reduction sees
    # the joined name the geometry dict is keyed by. Two namespaces for one
    # quantity, which is why this is worth asserting rather than assuming.
    joined = {f"{spec['column_prefix']}_{spec['metric']}" for spec in DEFAULT_PROTEIN_BINDER_THRESHOLDS.values()}
    assert {"complex_scRMSD_ca", "binder_scRMSD_target_aligned_ca"} <= joined, "both placement criteria are still gated"
    assert {"complex_scRMSD_ca", "binder_scRMSD_target_aligned_ca"} <= PLACEMENT_METRICS, (
        "and both are reduced by worst case, not by mean"
    )
    assert "binder_scRMSD_ca" in joined and "binder_scRMSD_ca" not in PLACEMENT_METRICS


def test_fold_quality_criteria_stay_out_of_the_placement_set():
    """binder_scRMSD_ca asks whether the sequence folds as designed, not where
    it sits. Its 1.5 A threshold was calibrated on a mean."""
    assert "binder_scRMSD_ca" not in PLACEMENT_METRICS
    assert "apo_scRMSD_ca" not in PLACEMENT_METRICS


def test_sibling_structures_are_derived_from_the_one_path_the_stats_keep(tmp_path):
    """predict_binder_complex names them {design}_model{n}.pdb, and the cached
    stats keep only model 1 -- the rest are popped before they can reach a
    dataframe. Recovering them is what makes a geometry-only refresh possible."""
    for n in (1, 2, 3):
        (tmp_path / f"d_model{n}.pdb").write_text("ATOM\n")
    got = per_model_paths_from_first(str(tmp_path / "d_model1.pdb"), 3)
    assert got == [str(tmp_path / f"d_model{n}.pdb") for n in (1, 2, 3)]


def test_a_missing_sibling_gives_up_rather_than_reducing_over_what_is_left(tmp_path):
    """A worst case over three of five models reports a better number than the
    design earned. All-or-nothing, so the caller refolds instead."""
    for n in (1, 2):
        (tmp_path / f"d_model{n}.pdb").write_text("ATOM\n")
    assert per_model_paths_from_first(str(tmp_path / "d_model1.pdb"), 3) is None


def test_a_path_that_is_not_a_model_structure_recovers_nothing(tmp_path):
    """RF3 and Boltz write one structure under their own names. There are no
    siblings to find, and inventing some would point at another design's files."""
    (tmp_path / "d.pdb").write_text("ATOM\n")
    assert per_model_paths_from_first(str(tmp_path / "d.pdb"), 3) is None
    assert per_model_paths_from_first("", 3) is None


def test_a_single_model_run_recovers_just_its_own_structure(tmp_path):
    (tmp_path / "d_model1.pdb").write_text("ATOM\n")
    assert per_model_paths_from_first(str(tmp_path / "d_model1.pdb"), 1) == [str(tmp_path / "d_model1.pdb")]


# ---------------------------------------------------------------------------
# Interface metrics reduced over the models a refold produced
# ---------------------------------------------------------------------------


def test_interface_metrics_are_averaged_not_read_off_one_model():
    """best_paths_dict names the model-1 structure, and reporting it alone
    reports one draw of five as the design. On a real CBLN1 design the buried
    area ran 2219 to 2414 A^2 across five models."""
    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    rows = [{"interface_dSASA": v, "interface_sc": 0.5} for v in (2238.0, 2295.0, 2219.0, 2378.0, 2414.0)]
    got = mean_interface_metrics(rows)
    assert got["interface_dSASA"] == pytest.approx(2308.8)
    assert got["n_interface_models"] == 5.0


def test_the_number_of_contributing_models_is_recorded():
    """A design that fell back to one structure must be distinguishable from one
    that averaged five, or a mean of one reads as a mean of five."""
    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    assert mean_interface_metrics([{"interface_dSASA": 1.0}])["n_interface_models"] == 1.0
    assert mean_interface_metrics([{"interface_dSASA": 1.0}] * 5)["n_interface_models"] == 5.0


def test_provenance_is_taken_from_the_first_model_not_averaged():
    """The SASA engine and radii are identical across models by construction and
    meaningless as an average."""
    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    rows = [{"sasa_engine": "freesasa", "sasa_radii": "ProtOr", "interface_dSASA": v} for v in (10.0, 20.0)]
    got = mean_interface_metrics(rows)
    assert got["sasa_engine"] == "freesasa"
    assert got["sasa_radii"] == "ProtOr"
    assert got["interface_dSASA"] == 15.0


def test_a_nan_model_does_not_poison_the_mean():
    """One structure failing should not erase the metric for the other four --
    the same reason reduce_rmsd_over_models ignores non-finite entries."""
    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    rows = [{"interface_dSASA": 10.0}, {"interface_dSASA": float("nan")}, {"interface_dSASA": 20.0}]
    assert mean_interface_metrics(rows)["interface_dSASA"] == 15.0


def test_an_all_nan_metric_stays_not_measured():
    import math

    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    got = mean_interface_metrics([{"interface_dSASA": float("nan")}] * 3)
    assert math.isnan(got["interface_dSASA"])


def test_pdb_path_is_left_to_the_caller():
    """Which model names the design is the caller's decision; averaging paths is
    meaningless and picking one silently would hide that."""
    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    got = mean_interface_metrics([{"pdb_path": "a_model1.pdb", "interface_dSASA": 1.0}])
    assert "pdb_path" not in got


def test_no_rows_is_empty_rather_than_zero():
    from proteinfoundation.metrics.ensembling import mean_interface_metrics

    assert mean_interface_metrics([]) == {}


def _pae_result(pae):
    class Result:
        plddt = None
        ptm = None
        iptm = None

    result = Result()
    result.pae = pae
    return result


def test_the_advisory_pae_family_is_on_the_scale_every_other_backend_uses():
    """One name, 31x apart, in the column pair the advisory track exists to
    compare: self_complex_af2_i_pAE read 0.173 on a design whose
    self_complex_esmfold2_i_pAE read 3.855. AF2 divides by the top PAE bin inside
    its loss and RF3 divides on the way in; ESMFold2 was the one backend left in
    Angstroms."""
    pytest.importorskip("esm")
    import numpy as np

    from proteinfoundation.metrics.consensus_folding import _esmfold2_metrics
    from proteinfoundation.metrics.ensembling import PAE_MAX_BIN

    pae = np.random.default_rng(0).uniform(0.5, 30.0, size=(6, 6))
    metrics = _esmfold2_metrics(_pae_result(pae), target_len=4)

    target_binder, binder_target = pae[:4, 4:], pae[4:, :4]
    symmetric = (pae + pae.T) / 2

    # Each against the definition the vendored ColabDesign loss uses, not merely
    # against "something in [0, 1]": i_pae is the symmetrised cross-block mean,
    # pae is the binder's rows against everything, min_ipae is unsymmetrised.
    assert metrics["i_pAE"] == pytest.approx(
        ((target_binder.mean() + binder_target.mean()) / 2) / PAE_MAX_BIN
    )
    assert metrics["pAE"] == pytest.approx(symmetric[4:].mean() / PAE_MAX_BIN)
    assert metrics["min_ipAE"] == pytest.approx(binder_target.min() / PAE_MAX_BIN)
    for name in ("i_pAE", "pAE", "min_ipAE"):
        assert 0.0 <= metrics[name] <= 1.0, f"{name} is a fraction of the top bin"


def test_ipsae_is_not_divided_because_it_is_already_a_fraction():
    """It is a TM-like score computed FROM the PAE in Angstroms against a cutoff
    in Angstroms -- 15 A plain, 10 A for the _10 columns, matching ColabDesign.
    Dividing it would be applying the normalisation twice."""
    pytest.importorskip("esm")
    import numpy as np
    from esm.models.esmfold2.interface_metrics import ipsae

    from proteinfoundation.metrics.consensus_folding import _esmfold2_metrics

    pae = np.random.default_rng(1).uniform(0.5, 30.0, size=(8, 8))
    metrics = _esmfold2_metrics(_pae_result(pae), target_len=5)

    for cutoff, suffix in ((15.0, ""), (10.0, "_10")):
        scored = ipsae(pae, 5, cutoff)
        forward, reverse = scored["ipsae_target_binder"], scored["ipsae_binder_target"]
        assert metrics[f"min_ipSAE{suffix}"] == pytest.approx(min(forward, reverse))
        assert metrics[f"max_ipSAE{suffix}"] == pytest.approx(max(forward, reverse))
        assert metrics[f"avg_ipSAE{suffix}"] == pytest.approx((forward + reverse) / 2)


def test_the_advisory_backend_reports_the_whole_pae_family():
    """Emission filters on CONSENSUS_METRIC_SUFFIXES, so a metric the scorer
    computes but the tuple omits never reaches a column."""
    from proteinfoundation.metrics.consensus_folding import CONSENSUS_METRIC_SUFFIXES

    assert {
        "i_pAE",
        "pAE",
        "min_ipAE",
        "min_ipSAE",
        "max_ipSAE",
        "avg_ipSAE",
        "min_ipSAE_10",
        "max_ipSAE_10",
        "avg_ipSAE_10",
    } <= set(CONSENSUS_METRIC_SUFFIXES)


def test_the_pae_divisor_has_one_definition():
    """It was the same float typed out in a dozen places -- the two producers,
    the RF3 adapter and its reward, four threshold dicts -- with nothing tying
    any of them to a model's actual bin range."""
    import ast

    from proteinfoundation.metrics.ensembling import PAE_MAX_BIN

    assert PAE_MAX_BIN == 31.0

    # Parsed rather than grepped, so a docstring that spells out the number for a
    # reader -- a config example, which cannot import anything -- is not mistaken
    # for a second definition of it.
    home = SRC / "src/proteinfoundation/metrics/ensembling.py"
    offenders = []
    for path in sorted((SRC / "src" / "proteinfoundation").rglob("*.py")):
        if path == home:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == 31.0:
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"the PAE divisor retyped instead of imported: {offenders}"


def test_colabfold_is_pointed_at_the_parameters_the_build_already_fetched(tmp_path, monkeypatch):
    """build_blackwell.sh puts the 2022-12-06 release at community_models/ckpts/AF2,
    which is exactly the <dir>/params/params_model_*.npz layout colabfold_batch
    wants -- so AF2 apo folding needs no download, only an address."""
    from proteinfoundation.metrics.folding_models import colabfold_data_dir

    for name in ("COLABFOLD_DATA_DIR", "AF2_DIR", "CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)

    af2 = tmp_path / "community_models" / "ckpts" / "AF2"
    (af2 / "params").mkdir(parents=True)
    (af2 / "params" / "params_model_1_ptm.npz").write_text("")
    monkeypatch.setenv("AF2_DIR", str(af2))

    assert colabfold_data_dir() == str(af2)
    # An explicit argument still wins, and a directory with no parameters yet is
    # still a legitimate place to download into.
    empty = tmp_path / "downloads"
    empty.mkdir()
    assert colabfold_data_dir(str(empty)) == str(af2), "a populated tree beats an empty one"


def test_colabfold_refuses_to_download_into_a_directory_named_none(monkeypatch):
    """The bug this replaced: cache_dir = os.environ.get("CACHE_DIR") was assigned
    OVER the function's own argument, so with CACHE_DIR unset the command read
    `--data None` and colabfold downloaded four gigabytes into ./None, per
    design, on a compute node."""
    import pytest as _pytest

    from proteinfoundation.metrics.folding_models import colabfold_data_dir

    for name in ("COLABFOLD_DATA_DIR", "AF2_DIR", "CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    with _pytest.raises(RuntimeError, match="params_model"):
        colabfold_data_dir()


# ---------------------------------------------------------------------------
# Fold confidence sidecars. pLDDT survives a fold in the B-factor column; pTM
# and PAE exist only in the folder's output, so a monomer fold reported one of
# the three while the complex track reported the whole family.
# ---------------------------------------------------------------------------


def test_the_sidecar_stores_pae_on_the_scale_every_other_column_uses(tmp_path):
    """Divided by the top PAE bin, like ColabDesign's loss, the RF3 adapter and
    the advisory complex path. An apo PAE in Angstroms beside a holo one at /31
    is the same 31x mismatch that had to be corrected in the advisory track."""
    from proteinfoundation.metrics.ensembling import PAE_MAX_BIN
    from proteinfoundation.metrics.folding_models import read_fold_confidence, write_fold_confidence

    pdb = tmp_path / "esm_1_seed7.pdb_esm_apo_mpnn"
    pdb.write_text("")
    pae = [[0.0, 6.0], [4.0, 0.0]]
    write_fold_confidence(str(pdb), ptm=0.82, pae=pae)

    got = read_fold_confidence(str(pdb))
    assert got["pTM"] == pytest.approx(0.82)
    # Symmetrised first, exactly as the advisory pAE is: (0 + 5 + 5 + 0) / 4.
    assert got["pAE"] == pytest.approx(2.5 / PAE_MAX_BIN)
    assert got["pAE"] < 1.0


def test_a_fold_with_no_sidecar_is_unmeasured_not_zero(tmp_path):
    """Every fold cached before the backends wrote these down lands here, and a
    zero pTM is a claim about the structure that nobody made."""
    from proteinfoundation.metrics.folding_models import read_fold_confidence, write_fold_confidence

    pdb = tmp_path / "esm_1.pdb_esm_apo_mpnn"
    pdb.write_text("")
    assert read_fold_confidence(str(pdb)) == {}

    # A backend that reports neither writes no file at all, rather than a file
    # asserting nothing.
    write_fold_confidence(str(pdb), ptm=None, pae=None)
    assert read_fold_confidence(str(pdb)) == {}


def test_the_sidecar_sits_beside_a_structure_with_no_extension_to_replace(tmp_path):
    """These files are named `esm_1_seed7.pdb_esm_apo_mpnn`. Substituting an
    extension would have written the sidecar over a different fold's name."""
    from proteinfoundation.metrics.folding_models import confidence_sidecar_path

    a = confidence_sidecar_path("/d/esm_1_seed7.pdb_esm_apo_mpnn")
    b = confidence_sidecar_path("/d/esm_1_seed8.pdb_esm_apo_mpnn")
    assert a != b
    assert a.startswith("/d/esm_1_seed7.pdb_esm_apo_mpnn")


def test_a_confidence_is_never_attributed_to_the_wrong_sequence():
    """ESMFold's pTM may be per-batch or per-sequence depending on the head. A
    scalar for a batch of four says nothing about which of the four it
    describes, and a number on the wrong sequence is worse than no number."""
    from proteinfoundation.metrics.folding_models import _batch_confidences

    batched = _batch_confidences({"ptm": [0.8, 0.6, 0.7]}, 3)
    assert [c["ptm"] for c in batched] == [0.8, 0.6, 0.7]

    assert _batch_confidences({"ptm": 0.8}, 3) == [{}, {}, {}], "a scalar names no sequence"
    assert _batch_confidences({"ptm": [0.8, 0.6]}, 3) == [{}, {}, {}], "a short batch names no sequence"
    assert _batch_confidences({}, 2) == [{}, {}]


# ---------------------------------------------------------------------------
# The force field, on the advisory structures too.
# ---------------------------------------------------------------------------


def test_tmol_is_requested_rather_than_registered():
    """It needs a compiled extension not every box has, and the binder campaigns
    run with it off. Registered unconditionally, every cached advisory entry
    would look under-derived forever on a box that cannot compute it, re-reading
    every kept PDB on every run to produce nothing."""
    from proteinfoundation.metrics.consensus_folding import (
        CONSENSUS_TMOL_SUFFIXES,
        consensus_derived_suffixes,
    )

    off = consensus_derived_suffixes(False)
    on = consensus_derived_suffixes(True)
    assert not set(CONSENSUS_TMOL_SUFFIXES) & set(off)
    assert set(CONSENSUS_TMOL_SUFFIXES) <= set(on)
    assert set(off) < set(on), "asking for it adds, it never replaces"


def test_asking_for_less_is_not_staleness():
    """The request is in the derivation fingerprint, so turning the force field
    on re-reads the kept structures and leaving it off does not. The campaigns
    run with it off: their fingerprint must be the one they already have."""
    from proteinfoundation.metrics.consensus_folding import consensus_derivation_fingerprint

    assert consensus_derivation_fingerprint(False) == consensus_derivation_fingerprint()
    assert consensus_derivation_fingerprint(True) != consensus_derivation_fingerprint(False)


def test_the_advisory_and_generated_structures_answer_the_same_four():
    """One mapping from TMOL's reward keys to column names, in one module. Two
    lists would agree right up until one of them was edited."""
    from proteinfoundation.evaluation.binder_eval_utils import TMOL_METRIC_COLS as from_eval
    from proteinfoundation.metrics.consensus_folding import CONSENSUS_TMOL_SUFFIXES
    from proteinfoundation.metrics.tmol_interface import TMOL_METRIC_COLS, TMOL_METRICS

    assert from_eval is TMOL_METRIC_COLS, "re-exported, not restated"
    assert tuple(TMOL_METRIC_COLS) == CONSENSUS_TMOL_SUFFIXES
    assert all(col.endswith("_tmol") for col in TMOL_METRICS.values())
