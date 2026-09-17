"""Where a folder's draws get collapsed, and why it is here rather than there.

Evaluate records one value per prediction and reduces none. Whether five AF2
parameter sets are meaned, or their worst case taken, is a formulation over
recorded numbers -- exactly like the ranking that picks which sequence a row
presents, and like the thresholds that turn numbers into verdicts. All three
belong in analyze, where changing one costs a re-read.

It used to live in the folding code, and the cost was not theoretical: AF2's five
models were averaged inside the harness and the parts discarded, so asking a
different question of them meant predicting the campaign again -- and because
only the first model's structure stayed in reach, anything re-read off "the"
structure answered for one model while the number beside it answered for five.

The invariant that matters most here is that moving the reduction did not move
any number: analyze reducing the recorded draws must agree with what evaluate
used to freeze into the artifact.
"""

import ast
import math

import pandas as pd
import pytest

from proteinfoundation.metrics.consensus_folding import draws_by_metric, reduce_over_draws
from proteinfoundation.result_analysis.analysis_utils import literal_eval_with_infinities
from proteinfoundation.result_analysis.binder_analysis_utils import (
    reduce_draws,
    reduce_draws_in_frame,
)


# ---------------------------------------------------------------------------
# Which rule applies
# ---------------------------------------------------------------------------


def test_every_metric_reduces_by_mean_including_placement():
    """Placement used to take the worst draw while fold quality took the mean, so
    the same column reduced two ways depending on its name. One rule now, and the
    placement names are the ones that moved."""
    assert reduce_draws([0.5, 8.0]) == pytest.approx(4.25)  # was 8.0
    assert reduce_draws([1.0, 2.0, 9.0]) == pytest.approx(4.0)  # was 9.0


def test_a_column_name_no_longer_selects_a_reduction():
    """The rule used to be chosen by matching the column name against the
    placement family, which meant binder_scRMSD_ca -- ending in the placement
    name scRMSD_ca -- had to be excluded by longest-match. Nothing selects any
    more, so the trap is gone with the mechanism."""
    assert reduce_draws([1.0, 3.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Collapsing one sequence's draws
# ---------------------------------------------------------------------------


def test_confidence_means_and_placement_takes_the_worst_draw():
    assert reduce_draws([0.9, 0.5]) == pytest.approx(0.7)
    assert reduce_draws([0.5, 8.0]) == pytest.approx(4.25)


def test_a_failed_draw_is_dropped_not_folded_in():
    """One NaN must not cost four good measurements."""
    assert reduce_draws([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert reduce_draws([1.0, float("inf"), 3.0]) == pytest.approx(2.0)


def test_nothing_finite_stays_nan_rather_than_becoming_zero():
    """A metric with nothing behind it must not become a plausible-looking number
    that clears a threshold. That holds for the worst case too: a failed draw
    leaves it unknown, not zero."""
    assert math.isnan(reduce_draws([float("nan"), float("inf")]))
    assert math.isnan(reduce_draws([float("nan")]))
    assert math.isnan(reduce_draws([]))


def test_a_structure_path_takes_the_first_draw_rather_than_averaging():
    assert reduce_draws(["/a.pdb", "/b.pdb"]) == "/a.pdb"


def test_an_already_reduced_value_passes_straight_through():
    """What a frame written before evaluate recorded draws holds, and what the
    primary backend's columns still hold until every folder is reached the same
    way. Both must survive this untouched."""
    assert reduce_draws(0.83) == 0.83
    assert reduce_draws("af2") == "af2"


# ---------------------------------------------------------------------------
# The invariant: moving the reduction moved no number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "by_draw",
    [
        {"model1": {"i_pTM": 0.9, "scRMSD_ca": 0.5}, "model2": {"i_pTM": 0.5, "scRMSD_ca": 8.0}},
        {f"model{k}": {"avg_ipSAE": 0.1 * k, "binder_scRMSD_ca": float(k)} for k in range(1, 6)},
        {"seed11": {"i_pAE": 0.2}, "seed22": {"i_pAE": 0.4}, "seed33": {"i_pAE": 0.6}},
    ],
)
def test_analyze_reducing_the_draws_agrees_with_what_evaluate_used_to_freeze(by_draw):
    """The whole point of the move. If these disagreed, every finished campaign's
    numbers would shift for no reason anyone could name."""
    was = reduce_over_draws(by_draw)
    per_draw = draws_by_metric(by_draw)
    for metric, values in per_draw.items():
        if metric == "n_predictions":
            continue
        assert reduce_draws(values) == pytest.approx(was[metric]), metric


def test_a_draw_missing_a_metric_holds_its_place_in_the_list():
    """Position k must be draw k in every list, or two metrics of one sequence
    describe different predictions -- the pairing failure sequences_for_type
    exists to stop, one axis down."""
    per_draw = draws_by_metric({"model1": {"a": 1.0, "b": 2.0}, "model2": {"a": 3.0}})
    assert per_draw["a"] == [1.0, 3.0]
    assert len(per_draw["b"]) == 2 and math.isnan(per_draw["b"][1])


# ---------------------------------------------------------------------------
# Over a frame
# ---------------------------------------------------------------------------


def _frame():
    return pd.DataFrame({
        # two sequences, each with three draws
        "mpnn_complex_af2_i_pTM_all": [[[0.9, 0.5, 0.7], [0.6, 0.6, 0.6]]],
        "mpnn_complex_af2_scRMSD_ca_all": [[[0.5, 8.0, 0.4], [1.0, 1.0, 1.0]]],
        # already reduced, as a pre-draw frame holds it
        "mpnn_complex_af2_binder_dSASA_all": [[900.0, 950.0]],
        "mpnn_sequence_all": [["AAAA", "CCCC"]],
    })


def test_a_per_draw_cell_becomes_the_per_sequence_list_analyze_reads():
    """The contract downstream is unchanged: X_all is a list over sequences and
    X is X_all[best_idx]. Only the number at each position is now computed here."""
    out = reduce_draws_in_frame(_frame())
    assert out["mpnn_complex_af2_i_pTM_all"][0] == pytest.approx([0.7, 0.6])
    assert out["mpnn_complex_af2_scRMSD_ca_all"][0] == pytest.approx([8.9 / 3, 1.0]), "placement by mean"


def test_an_already_reduced_column_is_left_exactly_as_it_is():
    """Mixed frames are the normal case while the primary backend still reduces
    in evaluate, and a pooled frame can hold runs from both eras."""
    out = reduce_draws_in_frame(_frame())
    assert out["mpnn_complex_af2_binder_dSASA_all"][0] == [900.0, 950.0]
    assert out["mpnn_sequence_all"][0] == ["AAAA", "CCCC"]


def test_reducing_twice_changes_nothing():
    """Analyze re-runs over its own output, and a second pass must not mean the
    means again."""
    once = reduce_draws_in_frame(_frame())
    first = list(once["mpnn_complex_af2_i_pTM_all"][0])
    twice = reduce_draws_in_frame(once)
    assert list(twice["mpnn_complex_af2_i_pTM_all"][0]) == pytest.approx(first)


def test_per_draw_cells_survive_the_csv_round_trip():
    """The frame reaches analyze as the repr of a Python list, parsed back by
    literal_eval_with_infinities. Nesting one list inside another must survive it
    -- including the NaN a failed draw leaves, which is a bare name that plain
    ast.literal_eval refuses."""
    cell = [[0.9, float("nan"), 0.7], [0.6, 0.6, 0.6]]
    parsed = literal_eval_with_infinities(repr(cell))
    assert len(parsed) == 2 and len(parsed[0]) == 3
    assert math.isnan(parsed[0][1])
    assert reduce_draws(parsed[0]) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        ast.literal_eval(repr(cell))


def test_placement_and_fold_quality_now_reduce_alike():
    """These two used to reduce differently -- scRMSD_ca by worst case,
    binder_scRMSD_ca by mean -- which is why the name had to be matched by
    longest suffix. Whatever the column, the draws behind it now mean the same
    thing."""
    draws = [1.0, 5.0]
    assert reduce_draws(draws) == pytest.approx(3.0)
    frame = pd.DataFrame({
        "self_complex_esmfold2_scRMSD_ca_all": [[draws]],
        "self_complex_esmfold2_binder_scRMSD_ca_all": [[draws]],
    })
    out = reduce_draws_in_frame(frame)
    assert out["self_complex_esmfold2_scRMSD_ca_all"][0] == [pytest.approx(3.0)]
    assert out["self_complex_esmfold2_binder_scRMSD_ca_all"][0] == [pytest.approx(3.0)]


# ---------------------------------------------------------------------------
# The frame analyze actually gets
# ---------------------------------------------------------------------------
#
# Every test above hands reduce_draws_in_frame a frame built in memory. Analyze
# never sees one. It sees pd.concat(pd.read_csv(f) for f in per_shard_files), so
# each list-valued cell is the REPR of a list, and `isinstance(cell, list)` --
# which is what both readers below branch on -- is False for all of them.
#
# Nothing raised. The nesting survived reduction untouched and every selected
# scalar came out NaN, across two EFNB3 production runs, while a suite of 965
# tests stayed green and a campaign gate that checks column PRESENCE passed
# twice. These tests go through a file.


def _round_trip(df, tmp_path):
    """The frame as analyze receives it: through a CSV, one shard per file."""
    path = tmp_path / "binder_results_pipeline_0.csv"
    df.to_csv(path, index=False)
    return pd.read_csv(path)


def test_nested_draws_are_reduced_after_a_csv_round_trip(tmp_path):
    """The defect, at its smallest. Two redesigns, three ESMFold2 seeds each."""
    df = pd.DataFrame({
        "mpnn_complex_esmfold2_i_pAE_all": [[[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]],
    })
    out = reduce_draws_in_frame(_round_trip(df, tmp_path))
    got = out["mpnn_complex_esmfold2_i_pAE_all"][0]
    assert not any(isinstance(v, (list, tuple)) for v in got), (
        "the draws stayed nested, which is what shipped [[0.13, 0.14, 0.13]] to analyze"
    )
    assert list(got) == pytest.approx([0.20, 0.50])


def test_the_selected_scalar_is_a_number_after_a_csv_round_trip(tmp_path):
    """`_best` was NaN in every column of every row of two production runs. Not
    because the selection was wrong -- because it indexed a string."""
    from proteinfoundation.result_analysis.binder_analysis import pick_headline_sequence

    df = pd.DataFrame({
        "complex_folding_backend": ["af2"],
        "mpnn_complex_af2_i_pAE_all": [[[9.0, 9.0], [1.0, 1.0]]],
    })
    out = pick_headline_sequence(
        reduce_draws_in_frame(_round_trip(df, tmp_path)),
        ["mpnn"],
        {"i_pAE": {"scale": 1.0, "direction": "minimize"}},
    )
    assert list(out["mpnn_best_idx"]) == [1]
    best = out["mpnn_complex_af2_i_pAE_best"][0]
    assert not math.isnan(best), "a selected scalar that is NaN is the whole defect"
    assert best == pytest.approx(1.0)


def test_a_failed_fold_survives_the_round_trip_as_nan_not_as_a_parse_failure(tmp_path):
    """`nan` and `inf` are bare names that ast.literal_eval refuses, and an
    all-NaN column is exactly what an adopted-but-underived metric looks like.
    It must read back as NaN rather than being left as an unparsed string."""
    df = pd.DataFrame({
        "mpnn_complex_af2_i_pAE_all": [[float("nan")]],
        "self_complex_af2_i_pAE_all": [[[0.5, float("inf")]]],
    })
    out = reduce_draws_in_frame(_round_trip(df, tmp_path))
    assert math.isnan(out["mpnn_complex_af2_i_pAE_all"][0][0])
    assert out["self_complex_af2_i_pAE_all"][0] == pytest.approx([0.5]), (
        "inf is the absence of a measurement, not a large one"
    )


def test_a_column_of_equal_length_lists_is_not_flattened_into_the_frame(tmp_path):
    """Assigning a list of equal-length lists to a DataFrame column reads as a
    2-D array. Equal redesign counts across designs are the normal case, so this
    is the shape production always has."""
    df = pd.DataFrame({
        "mpnn_complex_af2_i_pAE_all": [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]],
    })
    out = reduce_draws_in_frame(_round_trip(df, tmp_path))
    assert len(out) == 2
    assert list(out["mpnn_complex_af2_i_pAE_all"][0]) == pytest.approx([2.0, 6.0])
    assert list(out["mpnn_complex_af2_i_pAE_all"][1]) == pytest.approx([3.0, 7.0])
