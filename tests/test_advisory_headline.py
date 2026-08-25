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

import pytest

from proteinfoundation.metrics.consensus_folding import (
    CONSENSUS_METRIC_SUFFIXES,
    advisory_column,
    assert_headline_indices_agree,
)

SEQ = "mpnn"
BACKEND = "esmfold2"

# Three redesigns. The ranked best is index 1, which is what made the original bug
# invisible in any test using a single sequence or a best-of-first ordering.
PRIMARY = {"i_pAE": [0.30, 0.10, 0.25], "i_pTM": [0.5, 0.9, 0.6], "pTM": [0.6, 0.8, 0.7], "pLDDT": [0.80, 0.95, 0.88]}
ADVISORY = {"i_pAE": [0.40, 0.15, 0.35], "i_pTM": [0.4, 0.8, 0.5], "pTM": [0.5, 0.7, 0.6], "pLDDT": [0.70, 0.93, 0.82]}
BEST = 1


def build_row(primary_idx=BEST, advisory_idx=BEST, *, advisory_all=True):
    row = {}
    for suffix in CONSENSUS_METRIC_SUFFIXES:
        row[f"{SEQ}_complex_{suffix}"] = PRIMARY[suffix][primary_idx]
        row[f"{SEQ}_complex_{suffix}_all"] = list(PRIMARY[suffix])
        column = advisory_column(SEQ, BACKEND, suffix)
        row[column] = ADVISORY[suffix][advisory_idx]
        if advisory_all:
            row[f"{column}_all"] = list(ADVISORY[suffix])
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


def test_best_only_mode_has_no_all_columns_and_is_not_flagged():
    """With consensus_best_only=true a single sequence is folded and no advisory
    _all columns are written; there is nothing to compare and that is not a fault."""
    assert_headline_indices_agree(build_row(advisory_all=False), SEQ, BACKEND)


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
