"""What the binder refolding cache accepts, and what it refuses.

Every rule here decides whether hours of GPU time are spent or skipped, so the
failure modes are asymmetric: refusing a good cache costs a refold, while
accepting a bad one puts numbers in a results CSV that no column names as
suspect. These tests exist because the reader grew a third acceptance path --
structure fingerprints a run declares reusable -- and nothing could reach it
while the cache lived next to an ``atomworks`` import.

The cutoff case is the one that motivated the split. ``interface_cutoff``
selects which residues are *counted*; it does not decide which structures were
*predicted*, except under ``mpnn_fixed`` where the interface residues are the
positions ProteinMPNN holds fixed. Held in the structure hash unconditionally,
moving it from 8.0 to 5.0 invalidated every cache on the CBLN1 campaign and
would have cost ~42 GPU-hours of refolding to recompute a distance query.
"""

import json
import pathlib

from proteinfoundation.evaluation.binder_eval_cache import (
    binder_eval_fingerprint,
    read_binder_eval_cache,
    write_binder_eval_cache,
)

SRC = pathlib.Path(__file__).resolve().parents[1]

STATS = {"self": {"complex_stats": [{}], "rmsd_stats": [{}], "aa_stats": [{"sequence": "ACDE"}]}}
SEQS = {"self": [{"seq": "ACDE"}]}


def store(tmp_path, fingerprint, derivation="deriv"):
    write_binder_eval_cache(str(tmp_path), fingerprint, STATS, SEQS, derivation)
    return str(tmp_path)


# ----------------------------------------------------------------- acceptance


def test_the_same_request_is_served_without_recomputation(tmp_path):
    root = store(tmp_path, "fp")
    stats, seqs, stale = read_binder_eval_cache(root, "fp", ["self"], "deriv")
    assert (stats, seqs) == (STATS, SEQS)
    assert stale is False


def test_a_different_structure_request_is_refused(tmp_path):
    """The case that matters most: a cache written by another folding backend
    must not be served for this one, however expensive refolding is."""
    root = store(tmp_path, "fp")
    assert read_binder_eval_cache(root, "other", ["self"], "deriv") is None


def test_a_cache_missing_a_requested_sequence_type_is_refused(tmp_path):
    root = store(tmp_path, "fp")
    assert read_binder_eval_cache(root, "fp", ["self", "mpnn"], "deriv") is None


def test_a_changed_derivation_keeps_the_structures_and_flags_the_numbers(tmp_path):
    root = store(tmp_path, "fp", derivation="old")
    stats, _, stale = read_binder_eval_cache(root, "fp", ["self"], "new")
    assert stats == STATS, "the structures are still the ones this run wants"
    assert stale is True, "but the numbers read off them are not"


def test_a_cache_predating_the_derivation_split_is_stale(tmp_path):
    """It carries no derivation fingerprint, so it cannot say which rule produced
    its numbers -- and guessing 'the current one' would serve means as maxima."""
    root = store(tmp_path, "fp", derivation=None)
    _, _, stale = read_binder_eval_cache(root, "fp", ["self"], "deriv")
    assert stale is True


# ------------------------------------------------------- declared-reusable folds


def test_a_declared_reusable_fingerprint_is_accepted_and_always_stale(tmp_path):
    """Reuse across a cutoff change is a claim about the folds, never about the
    numbers: whatever made the fingerprint differ has to be re-derived."""
    root = store(tmp_path, "old_fp", derivation="deriv")
    result = read_binder_eval_cache(root, "new_fp", ["self"], "deriv", ["old_fp"])
    assert result is not None, "the declared fingerprint should be accepted"
    stats, _, stale = result
    assert stats == STATS
    assert stale is True, "matching derivation fingerprints must not mask the change"


def test_an_undeclared_fingerprint_stays_refused(tmp_path):
    """The list is an allow-list, not a switch that disables the check."""
    root = store(tmp_path, "old_fp")
    assert read_binder_eval_cache(root, "new_fp", ["self"], "deriv", ["unrelated_fp"]) is None


def test_no_declared_fingerprints_behaves_as_before(tmp_path):
    root = store(tmp_path, "old_fp")
    assert read_binder_eval_cache(root, "new_fp", ["self"], "deriv", []) is None
    assert read_binder_eval_cache(root, "new_fp", ["self"], "deriv", None) is None


def test_a_reused_cache_is_rewritten_under_the_current_fingerprint(tmp_path):
    """So the migration happens once per design, on first use, rather than on
    every later run -- and so the next run sees an ordinary cache hit."""
    root = store(tmp_path, "old_fp")
    assert read_binder_eval_cache(root, "new_fp", ["self"], "deriv", ["old_fp"]) is not None
    write_binder_eval_cache(root, "new_fp", STATS, SEQS, "deriv")
    _, _, stale = read_binder_eval_cache(root, "new_fp", ["self"], "deriv")
    assert stale is False


# ----------------------------------------------------------------- the key itself


def test_the_interface_cutoff_is_part_of_the_derivation_not_the_structure():
    """The split this file exists for, asserted where it is written."""
    source = (SRC / "src/proteinfoundation/evaluation/binder_eval.py").read_text()
    opened = source.index("cache_fingerprint_base = {")
    # The dict literal alone. The comment that follows it explains the exclusion
    # and names the key, so slicing to the guard below would match the prose.
    base = source[opened : source.index("\n    }\n", opened)]
    assert "interface_cutoff" not in base, "a cutoff change must not invalidate folds"
    derivation = source[source.index("derivation_fingerprint = binder_eval_fingerprint(") :][:400]
    assert "interface_cutoff=interface_cutoff" in derivation


def test_mpnn_fixed_puts_the_cutoff_back_into_structure_identity():
    """There the interface residues are the positions ProteinMPNN holds fixed, so
    the cutoff decides which sequences get folded, not only which get counted."""
    source = (SRC / "src/proteinfoundation/evaluation/binder_eval.py").read_text()
    guard = source[source.index('if "mpnn_fixed" in sequence_types:') :][:200]
    assert 'cache_fingerprint_base["interface_cutoff"] = interface_cutoff' in guard


def test_declaring_reusable_cutoffs_under_mpnn_fixed_is_refused():
    """Silently reusing there would serve folds of different sequences."""
    source = (SRC / "src/proteinfoundation/evaluation/binder_eval.py").read_text()
    block = source[source.index("reusable_interface_cutoffs = ") :][:900]
    assert 'if reusable_interface_cutoffs and "mpnn_fixed" in sequence_types:' in block
    assert "raise ValueError(" in block


def test_two_cutoffs_hash_apart_in_the_derivation():
    """The mechanism the refresh relies on: a cutoff change has to be visible as
    staleness, or the cached counts would be served for the wrong cutoff."""
    a = binder_eval_fingerprint(geometry_reduction=2, interface_cutoff=8.0)
    b = binder_eval_fingerprint(geometry_reduction=2, interface_cutoff=5.0)
    assert a != b


def test_an_unwritable_payload_leaves_no_half_written_cache(tmp_path):
    """A cache is an optimisation; a truncated one is a landmine for the next run."""
    write_binder_eval_cache(str(tmp_path), "fp", {"self": {object(): 1}}, SEQS, "deriv")
    path = tmp_path / "binder_eval_cache.json"
    assert not path.exists() or json.loads(path.read_text())


# --------------------------------------------------- what the refresh guarantees
#
# ``binder_metrics`` pulls in the folding stack, so these read the source. The
# properties are the ones a partial refresh would violate quietly.


def _binder_metrics():
    return (SRC / "src/proteinfoundation/metrics/binder_metrics.py").read_text()


def test_the_refresh_measures_the_interface_on_the_file_the_fold_did():
    """A refreshed count taken from the all-atom design while the original was
    taken from the C-alpha ``_updated`` view would differ for reasons that have
    nothing to do with the cutoff, and nothing downstream would say so."""
    source = _binder_metrics()
    assert source.count("updated_structure_path(") >= 3, "one definition, used by both paths"
    refresh = source[source.index("def recompute_derived(") : source.index("def run_binder_eval(")]
    assert "updated_structure_path(pdb_file_path, is_target_ligand)" in refresh
    assert "interface_positions(" in refresh


def test_a_sequence_the_cache_never_recorded_forces_a_refold():
    """Recovering it by indexing sequences_dict in parallel is the silent
    mispairing that recording the sequence exists to remove."""
    refresh = _binder_metrics()
    refresh = refresh[refresh.index("def recompute_derived(") : refresh.index("def run_binder_eval(")]
    guard = refresh[refresh.index('sequence = aa_stat.get("sequence")') :][:900]
    assert "if sequence is None:" in guard
    assert "return False" in guard


def test_out_of_range_interface_positions_force_a_refold():
    """Dropping them would emit a composition that looks measured."""
    refresh = _binder_metrics()
    refresh = refresh[refresh.index("def recompute_derived(") : refresh.index("def run_binder_eval(")]
    assert "if any(j >= len(sequence) for j in interface_seq_indices):" in refresh


def test_counts_are_applied_only_after_every_sequence_succeeds():
    """A refusal halfway through must leave the cache as it was found, not half
    at the new cutoff and half at the old."""
    refresh = _binder_metrics()
    refresh = refresh[refresh.index("def recompute_derived(") : refresh.index("def run_binder_eval(")]
    loop = refresh.index("refreshed_counts = {}")
    collect = refresh.index("refreshed_counts[(seq_type, i)] = (")
    apply = refresh.index("for (seq_type, i), counts in refreshed_counts.items():")
    assert loop < collect < apply, "collected into a holding dict first, applied second"
    # Both refusals live inside the collecting loop, so a later sequence can veto
    # counts an earlier one already produced -- which is only safe because
    # nothing has been written to sequence_type_stats by then.
    assert refresh[loop:collect].count("return False") == 2
    assert "interface_counts" not in refresh[loop:collect], "nothing is mutated while collecting"
