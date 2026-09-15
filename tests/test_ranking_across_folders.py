"""What "best" means, and which folders get a vote in deciding it.

Two separate things happen to a row's numbers, and until now only one of them
was named. Draws -- repeat predictions of one sequence -- are pooled by a
statistic. Sequences are different molecules, and one of them is SELECTED to be
the row's headline. A selected value is not a measurement of the design: for the
ranking criterion's own metric it is an order statistic of itself, the minimum
i_pAE over redesigns published under the name i_pAE. So the selected scalars are
``X_best`` and nothing is bare.

And the selection itself used to take one folder's word. complex_backend_of(df)
resolved a single backend for every criterion, so "best" meant whatever the
frame called primary thought was best, with the other folders' columns reported
at that index and no vote in choosing it. A criterion can now name its folder.
"""

import pandas as pd
import pytest

from proteinfoundation.result_analysis.binder_analysis import pick_headline_sequence
from proteinfoundation.result_analysis.binder_analysis_utils import (
    ThresholdSpecError,
    ranking_criterion_backend,
    resolve_ranking_columns,
    split_ranking_key,
)

MIN = {"scale": 1.0, "direction": "minimize"}


def _frame(af2, esm, seq_type="mpnn"):
    """Two redesigns, with each folder's opinion of both."""
    return pd.DataFrame({
        "complex_folding_backend": ["af2"],
        f"{seq_type}_complex_af2_i_pAE_all": [af2],
        f"{seq_type}_complex_esmfold2_i_pAE_all": [esm],
    })


# ---------------------------------------------------------------------------
# Naming a folder in a criterion
# ---------------------------------------------------------------------------


def test_a_bare_criterion_still_means_the_frames_own_folder():
    assert split_ranking_key("i_pAE") == (None, "i_pAE")
    assert ranking_criterion_backend("i_pAE", MIN, "af2") == ("af2", "i_pAE", False)


def test_a_qualified_criterion_names_its_folder():
    assert split_ranking_key("esmfold2:i_pAE") == ("esmfold2", "i_pAE")
    assert ranking_criterion_backend("esmfold2:i_pAE", MIN, "af2") == ("esmfold2", "i_pAE", True)


def test_the_folder_lives_in_the_key_so_two_folders_cannot_collide():
    """The criteria dict is keyed by quantity. Two criteria on i_pAE from two
    folders would otherwise need the same key twice, which Python resolves
    silently in favour of the last -- the failure resolve_backend_overrides
    carries a scar from, where binder and complex scRMSD_ca collided and one of
    them simply stopped being applied."""
    criteria = {"af2:i_pAE": MIN, "esmfold2:i_pAE": MIN}
    assert len(criteria) == 2
    columns = resolve_ranking_columns("mpnn", criteria, "af2")
    assert columns["af2:i_pAE"] == "mpnn_complex_af2_i_pAE_all"
    assert columns["esmfold2:i_pAE"] == "mpnn_complex_esmfold2_i_pAE_all"
    assert len(set(columns.values())) == 2, "two criteria must not resolve to one column"


def test_the_key_wins_over_a_spec_field():
    """One fact, one spelling. The key is the half that has to be unique anyway."""
    got = ranking_criterion_backend("esmfold2:i_pAE", {**MIN, "backend": "af2"}, "rf3")
    assert got[0] == "esmfold2"


def test_a_spec_field_works_where_the_key_is_unambiguous():
    assert ranking_criterion_backend("i_pAE", {**MIN, "backend": "esmfold2"}, "af2")[0] == "esmfold2"


# ---------------------------------------------------------------------------
# Ranking across folders
# ---------------------------------------------------------------------------


def test_one_folder_alone_picks_its_own_favourite():
    out = pick_headline_sequence(_frame([9.0, 1.0], [1.0, 9.0]), ["mpnn"], {"i_pAE": MIN})
    assert list(out["mpnn_best_idx"]) == [1], "af2 prefers redesign 1"


def test_two_folders_together_pick_the_one_they_agree_on():
    """The case that motivated this: a redesign that is good in AF2 AND in
    ESMFold2, rather than excellent in one and poor in the other. Redesign 0
    wins on AF2 alone and redesign 1 on ESMFold2 alone; redesign 2 is second-best
    in both and is what agreement selects."""
    frame = _frame([1.0, 9.0, 2.0], [9.0, 1.0, 2.0])
    alone = pick_headline_sequence(frame.copy(), ["mpnn"], {"i_pAE": MIN})
    assert list(alone["mpnn_best_idx"]) == [0]

    together = pick_headline_sequence(
        frame.copy(), ["mpnn"], {"af2:i_pAE": MIN, "esmfold2:i_pAE": MIN}
    )
    assert list(together["mpnn_best_idx"]) == [2], "agreement, not either folder's favourite"


def test_a_folders_vote_can_be_weighted():
    """scale already means this for a metric; it means it for a folder too, so a
    folder can advise without deciding."""
    frame = _frame([1.0, 9.0], [9.0, 1.0])
    out = pick_headline_sequence(
        frame, ["mpnn"], {"af2:i_pAE": MIN, "esmfold2:i_pAE": {"scale": 0.1, "direction": "minimize"}}
    )
    assert list(out["mpnn_best_idx"]) == [0], "af2 outweighs a tenth-weighted esmfold2"


def test_naming_a_folder_the_frame_does_not_have_is_an_error():
    """A criterion that asked for ESMFold2 on a campaign that never ran it is a
    question nothing can answer. Ranking by the remainder would silently rank by
    something else -- the same shrinking-gate failure resolve_backend_overrides
    refuses for thresholds."""
    frame = pd.DataFrame({
        "complex_folding_backend": ["af2"],
        "mpnn_complex_af2_i_pAE_all": [[1.0, 2.0]],
    })
    with pytest.raises(ThresholdSpecError, match="esmfold2"):
        pick_headline_sequence(frame, ["mpnn"], {"esmfold2:i_pAE": MIN})


def test_an_unqualified_criterion_that_is_missing_still_only_warns():
    """It named no folder, so the frame's own answer is the honest one and index
    0 is recorded as the fallback. Only an explicit request is an error."""
    frame = pd.DataFrame({"complex_folding_backend": ["af2"], "mpnn_other_all": [[1.0]]})
    out = pick_headline_sequence(frame, ["mpnn"], {"i_pAE": MIN})
    assert list(out["mpnn_best_idx"]) == [0]


# ---------------------------------------------------------------------------
# X_best
# ---------------------------------------------------------------------------


def test_every_headline_scalar_carries_best_and_none_is_bare():
    out = pick_headline_sequence(_frame([9.0, 1.0], [1.0, 9.0]), ["mpnn"], {"i_pAE": MIN})
    assert out["mpnn_complex_af2_i_pAE_best"].iloc[0] == 1.0
    assert "mpnn_complex_af2_i_pAE" not in out.columns
    assert "mpnn_complex_esmfold2_i_pAE" not in out.columns


def test_every_folders_scalar_is_taken_at_the_one_chosen_index():
    """A row presents one molecule. Taking each folder's own favourite would give
    a row describing no actual redesign -- which is what
    assert_headline_indices_agree exists to refuse."""
    out = pick_headline_sequence(_frame([9.0, 1.0], [1.0, 9.0]), ["mpnn"], {"i_pAE": MIN})
    assert out["mpnn_complex_af2_i_pAE_best"].iloc[0] == 1.0
    assert out["mpnn_complex_esmfold2_i_pAE_best"].iloc[0] == 9.0, (
        "esmfold2 reports on the redesign af2 chose, not on its own favourite"
    )


def test_a_stale_pre_rename_scalar_is_dropped_rather_than_left_beside_the_new_one():
    """Its value came from THAT run's ranking. Leaving it would put a stale
    number under the more inviting name."""
    frame = _frame([9.0, 1.0], [1.0, 9.0])
    frame["mpnn_complex_af2_i_pAE"] = [9.0]
    out = pick_headline_sequence(frame, ["mpnn"], {"i_pAE": MIN})
    assert "mpnn_complex_af2_i_pAE" not in out.columns
    assert out["mpnn_complex_af2_i_pAE_best"].iloc[0] == 1.0
