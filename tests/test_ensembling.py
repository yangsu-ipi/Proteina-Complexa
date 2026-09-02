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
    af2_stats_from_metrics,
    average_af2_stats,
    average_rmsd_over_models,
    mean_chain_plddt,
    mean_plddt_from_pdb,
    pop_per_model_paths,
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
    assert stats["pLDDT"] == round(RAW["plddt"], 3)
    assert stats["i_pAE"] == round(RAW["i_pae"], 3)
    assert stats["avg_ipSAE"] == round(RAW["avg_ipsae"], 4)


def test_absent_ipsae_10_reads_as_zero_rather_than_raising():
    """RAW has no *_10 keys, matching older colabdesign builds."""
    assert af2_stats_from_metrics(RAW)["min_ipSAE_10"] == 0.0


def test_scores_are_meaned_not_best_of():
    """Taking the best model would discard the disagreement that is the whole
    reason for running five of them."""
    good = af2_stats_from_metrics({**RAW, "plddt": 0.95})
    bad = af2_stats_from_metrics({**RAW, "plddt": 0.55})
    assert average_af2_stats([good, bad])["pLDDT"] == pytest.approx(0.75)


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


def test_a_backend_reporting_no_per_residue_confidence_still_averages():
    """average_af2_stats must not require the per-chain keys of every caller --
    a model that exposes no per-residue pLDDT should lose those two columns,
    not raise."""
    stats = average_af2_stats([af2_stats_from_metrics(RAW), af2_stats_from_metrics(RAW)])
    assert "target_pLDDT" not in stats
    assert stats["pLDDT"] == round(RAW["plddt"], 3)


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


def test_rmsd_averages_over_the_models():
    per_model = [{"complex_scRMSD_ca": 1.0}, {"complex_scRMSD_ca": 3.0}]
    assert average_rmsd_over_models(per_model)["complex_scRMSD_ca"] == pytest.approx(2.0)


def test_one_unusable_model_does_not_erase_the_others():
    """A NaN placement is a missing measurement, not a bad one: dropping it
    keeps four good models answerable, while averaging it in would leave the
    design with no number and no reason visible for why."""
    per_model = [{"binder_scRMSD_ca": 1.0}, {"binder_scRMSD_ca": float("nan")}, {"binder_scRMSD_ca": 2.0}]
    assert average_rmsd_over_models(per_model)["binder_scRMSD_ca"] == pytest.approx(1.5)


def test_nothing_finite_stays_nan_rather_than_becoming_zero():
    """A zero RMSD passes every gate there is."""
    per_model = [{"complex_scRMSD_ca": float("nan")}, {"complex_scRMSD_ca": float("inf")}]
    assert math.isnan(average_rmsd_over_models(per_model)["complex_scRMSD_ca"])


def test_a_single_model_result_is_passed_through_untouched():
    """Backends that predict one structure must not be reshaped by a reduction
    they never asked for -- including any non-numeric fields they carry."""
    only = {"complex_scRMSD_ca": 1.25, "note": "rf3"}
    assert average_rmsd_over_models([only]) is only


def test_no_models_is_an_empty_result():
    assert average_rmsd_over_models([]) == {}


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
        produced = f"mpnn_complex_{metric}_all"
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
