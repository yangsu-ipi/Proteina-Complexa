"""The advisory headline must describe the same sequence as the primary headline.

The regression this guards: advisory scalars were taken from ``advisory[0]`` while
every primary scalar used ``seq_best_idx``, so whenever the ranked-best sequence
was not the first, ``{seq}_esmfold2_i_pAE`` and ``{seq}_complex_i_pAE`` described
different redesigns -- and nothing in either number said so.

Asserted on the row rather than at the point of assignment. The bug was choosing
the wrong index at a call site, so a test of the index expression would pass while
a different call site made the same mistake; a test of the row catches it however
it arises. Production runs the same check on the first design of every run.
"""

import math
import pathlib

import pytest

from proteinfoundation.metrics.column_names import rename
from proteinfoundation.metrics.consensus_folding import (
    CONSENSUS_METRIC_SUFFIXES,
    advisory_column,
    assert_headline_indices_agree,
)
from proteinfoundation.result_analysis.binder_analysis_utils import COMPLEX_BACKEND_COLUMN

SEQ = "mpnn"
BACKEND = "esmfold2"
COMPLEX_BACKEND = "af2"

# Three redesigns. The ranked best is index 1, which is what made the original bug
# invisible in any test using a single sequence or a best-of-first ordering.
# Every metric in CONSENSUS_METRIC_SUFFIXES, since build_row iterates it: a metric
# added there without a fixture entry fails loudly here rather than going
# unchecked. The per-chain pair moves with the complex mean, as it would on a
# real row -- the binder is what varies between redesigns, the target barely.
PRIMARY = {
    "i_pAE": [0.30, 0.10, 0.25],
    "pAE": [0.25, 0.09, 0.20],
    "min_ipAE": [0.05, 0.02, 0.04],
    "min_ipSAE": [0.40, 0.74, 0.55],
    "max_ipSAE": [0.50, 0.82, 0.63],
    "avg_ipSAE": [0.45, 0.78, 0.59],
    "min_ipSAE_10": [0.35, 0.70, 0.50],
    "max_ipSAE_10": [0.45, 0.78, 0.58],
    "avg_ipSAE_10": [0.40, 0.74, 0.54],
    "i_pTM": [0.5, 0.9, 0.6],
    "pTM": [0.6, 0.8, 0.7],
    "pLDDT": [0.80, 0.95, 0.88],
    "target_pLDDT": [0.94, 0.96, 0.95],
    "binder_pLDDT": [0.55, 0.92, 0.70],
}
ADVISORY = {
    "i_pAE": [0.40, 0.15, 0.35],
    "pAE": [0.32, 0.14, 0.28],
    "min_ipAE": [0.07, 0.03, 0.05],
    "min_ipSAE": [0.32, 0.66, 0.47],
    "max_ipSAE": [0.42, 0.74, 0.55],
    "avg_ipSAE": [0.37, 0.70, 0.51],
    "min_ipSAE_10": [0.28, 0.62, 0.43],
    "max_ipSAE_10": [0.38, 0.70, 0.50],
    "avg_ipSAE_10": [0.33, 0.66, 0.46],
    "i_pTM": [0.4, 0.8, 0.5],
    "pTM": [0.5, 0.7, 0.6],
    "pLDDT": [0.70, 0.93, 0.82],
    "target_pLDDT": [0.80, 0.84, 0.82],
    "binder_pLDDT": [0.45, 0.80, 0.60],
}
BEST = 1


def build_row(primary_idx=BEST, advisory_idx=BEST, *, advisory_all=True, pass_idx=None):
    # Keys exactly as binder_eval writes them -- through rename(), with the backend
    # slot. Built the pre-rename way, every test here passed against a guard that
    # could not see a single real row.
    row = {COMPLEX_BACKEND_COLUMN: COMPLEX_BACKEND}
    for suffix in CONSENSUS_METRIC_SUFFIXES:
        primary = rename(f"{SEQ}_complex_{suffix}", COMPLEX_BACKEND)
        row[primary] = PRIMARY[suffix][primary_idx]
        row[f"{primary}_all"] = list(PRIMARY[suffix])
        column = advisory_column(SEQ, BACKEND, suffix)
        row[column] = ADVISORY[suffix][advisory_idx]
        if advisory_all:
            row[f"{column}_all"] = list(ADVISORY[suffix])
    if pass_idx is not None:
        verdicts = [0, 1, 0]
        row[f"{SEQ}_pass_all"] = list(verdicts)
        row[f"{SEQ}_pass"] = verdicts[pass_idx]
    return row


def test_a_consistent_row_passes():
    assert_headline_indices_agree(build_row(), SEQ, BACKEND)


def test_the_regression_is_caught():
    """advisory[0] while the primary headline is index 1 -- the exact bug."""
    with pytest.raises(ValueError, match="different sequence"):
        assert_headline_indices_agree(build_row(advisory_idx=0), SEQ, BACKEND)


@pytest.mark.parametrize("primary_idx,advisory_idx", [(0, 2), (2, 0), (1, 2), (2, 1)])
def test_any_disagreement_is_caught(primary_idx, advisory_idx):
    with pytest.raises(ValueError, match="different sequence"):
        assert_headline_indices_agree(build_row(primary_idx, advisory_idx), SEQ, BACKEND)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_agreement_at_any_shared_index_passes(index):
    """The contract is that one index explains both, not that it is a particular
    one -- best-of ranking may pick any of them."""
    assert_headline_indices_agree(build_row(index, index), SEQ, BACKEND)


def test_a_row_without_advisory_all_columns_is_not_flagged():
    """No current run produces one -- consensus_best_only is gone and every
    sequence is folded. But rows written under it still exist on disk, and
    re-analysing an old campaign must not fail on a row that simply has nothing
    to compare."""
    assert_headline_indices_agree(build_row(advisory_all=False), SEQ, BACKEND)


def test_a_row_with_lists_but_no_headline_is_not_flagged():
    """What evaluate produces, and what took down the EphA3 smoke run.

    Evaluate emits per-sequence lists and NO scalars -- analyze builds the
    headline -- so on an evaluate-time row every scalar is absent. Reading those
    absences as None made the check report that no index explained the headline.
    It was right that none did: there was no headline. A column with no scalar
    makes no claim and cannot disagree with one."""
    row = build_row()
    lists_only = {k: v for k, v in row.items() if k.endswith("_all")}
    assert lists_only, "the fixture must actually carry per-sequence lists"
    assert not any(not k.endswith("_all") for k in lists_only)

    assert_headline_indices_agree(lists_only, SEQ, BACKEND)


def test_a_partly_built_headline_is_still_checked():
    """The skip must be per column, not a blanket escape. One scalar present and
    wrong is exactly the drift this exists to catch, and it must not be excused by
    the scalars beside it being absent."""
    row = build_row()
    lists_only = {k: v for k, v in row.items() if k.endswith("_all")}
    primary = rename(f"{SEQ}_complex_i_pAE", "af2")
    # Present, and equal to no entry of its own list.
    lists_only[primary] = 99.0
    lists_only[COMPLEX_BACKEND_COLUMN] = "af2"
    with pytest.raises(ValueError, match="No single sequence"):
        assert_headline_indices_agree(lists_only, SEQ, BACKEND)


def test_the_check_runs_where_the_headline_exists():
    """A guard at a call site that can never exercise it is the failure this
    function's own docstring records having had. It is called from analyze, after
    the headline is chosen and the verdicts refreshed -- not from evaluate."""
    analyze = pathlib.Path("src/proteinfoundation/analyze.py").read_text()
    evaluate = pathlib.Path("src/proteinfoundation/evaluation/binder_eval.py").read_text()

    assert "assert_frame_headline_indices_agree(" in analyze
    assert analyze.index("refresh_per_sequence_verdicts(") < analyze.index(
        "assert_frame_headline_indices_agree("
    ), "the verdict is part of what is checked, so it must be written first"
    assert "assert_headline_indices_agree(row_dict" not in evaluate, (
        "evaluate has no headline to check"
    )


def test_advisory_backends_are_read_off_the_columns():
    """A pooled frame can hold runs that asked for different backends, so the
    frame's contents decide what is checked, not the config's request."""
    from proteinfoundation.metrics.consensus_folding import advisory_backends_in

    columns = [
        "mpnn_complex_af2_i_pAE",
        "mpnn_complex_esmfold2_i_pAE",
        "mpnn_complex_esmfold2_min_ipAE",
        "mpnn_complex_af2_min_ipAE",
        "self_complex_esmfold2_i_pAE",
    ]
    assert advisory_backends_in(columns, "mpnn", "af2") == ["esmfold2"]
    assert advisory_backends_in(columns, "self", "af2") == ["esmfold2"]
    assert advisory_backends_in(columns, "mpnn", "esmfold2") == ["af2"], "whoever is primary is excluded"
    assert advisory_backends_in([], "mpnn", "af2") == []


def test_a_failed_backend_writing_nan_is_consistent_not_mismatched():
    """A backend that failed writes NaN to the scalar and the list. NaN != NaN, so
    a naive equality check would report a mismatch and kill the run."""
    row = build_row()
    for suffix in CONSENSUS_METRIC_SUFFIXES:
        column = advisory_column(SEQ, BACKEND, suffix)
        row[column] = float("nan")
        row[f"{column}_all"] = [float("nan")] * 3
    assert_headline_indices_agree(row, SEQ, BACKEND)


def test_a_row_without_advisory_columns_is_not_flagged():
    """consensus_backends is empty by default."""
    row = {k: v for k, v in build_row().items() if BACKEND not in k}
    assert_headline_indices_agree(row, SEQ, BACKEND)


def test_one_sequence_cannot_hide_the_bug_but_is_still_accepted():
    """A single redesign makes every index 0, so this shape proves nothing -- it is
    here to document why the fixtures above use three."""
    row = {}
    for suffix in CONSENSUS_METRIC_SUFFIXES:
        row[f"{SEQ}_complex_{suffix}"] = PRIMARY[suffix][0]
        row[f"{SEQ}_complex_{suffix}_all"] = [PRIMARY[suffix][0]]
        column = advisory_column(SEQ, BACKEND, suffix)
        row[column] = ADVISORY[suffix][0]
        row[f"{column}_all"] = [ADVISORY[suffix][0]]
    assert_headline_indices_agree(row, SEQ, BACKEND)


def test_nan_helper_assumption():
    assert float("nan") != float("nan") and math.isnan(float("nan"))


def test_the_guard_sees_production_key_shapes():
    """It did not. binder_eval writes `mpnn_complex_af2_i_pAE` through rename();
    this module looked up `mpnn_complex_i_pAE`, found no _all lists, and returned
    before checking anything -- dead on every real row since the slot scheme
    landed, and green the whole time because these tests built the old shape."""
    row = build_row(primary_idx=1, advisory_idx=0)
    assert any("_complex_af2_" in k for k in row), "the fixture must use the real keys"
    with pytest.raises(ValueError, match="different sequence"):
        assert_headline_indices_agree(row, SEQ, BACKEND)


def test_a_verdict_from_another_sequence_is_caught():
    """The verdict is a headline column, and the one that actually drifted: analyze
    re-derives it, so it is the one able to end up describing a different sequence
    than the metrics beside it."""
    with pytest.raises(ValueError, match="No single sequence explains"):
        assert_headline_indices_agree(build_row(pass_idx=0), SEQ, BACKEND)


def test_a_verdict_on_the_headline_sequence_passes():
    assert_headline_indices_agree(build_row(pass_idx=BEST), SEQ, BACKEND)


def test_a_fresh_advisory_fold_is_derived_on_the_run_that_made_it(tmp_path, monkeypatch):
    """The advisory scorer reports what the FOLDER knows -- pTM, PAE, the ipSAE
    family -- and the entry filter keeps only those. Everything read off the
    structure (buried area, shape complementarity, secondary structure, geometry
    against the design) arrived absent, and the derivation pass ran only over
    entries read from CACHE. So a fresh fold shipped NaN in all of them and
    healed on the next run: the cache ended up correct while the CSV that run
    wrote did not, and a campaign had to be evaluated twice to fill columns whose
    structures were on disk the first time.

    Verified on the EphA3 box before the fix: consensus_fold_cache_esmfold2.json
    held binder_dSASA=812.928 and interface_sc=0.373 while the run's own
    binder_results CSV carried [nan, nan] for both."""
    import inspect

    from proteinfoundation.metrics import consensus_folding

    source = inspect.getsource(consensus_folding.score_binders)
    derive_calls = source.count("_derive_into_scores(")
    assert derive_calls >= 3, (
        "one definition and two callers -- cached entries AND fresh folds; "
        f"found {derive_calls} occurrences"
    )
    fresh_block = source.split("if pending:", 1)[-1]
    assert "_derive_into_scores(fresh" in fresh_block, (
        "the fresh folds must be derived before they are cached or returned"
    )
    cached_at = source.index("_derive_into_scores(scores")
    fresh_at = source.index("_derive_into_scores(fresh")
    assert cached_at < fresh_at, "cached entries first, then the folds this run just made"
