"""Cache keys and cache-write guards.

Two review findings were here, and both were the same shape: a cache answering a
question it was not asked. One cached a wholly failed refold because the guard
tested list non-emptiness under a variable named ``finite``; the other returned
scores when the caller had asked for structures.

A cache key is only as good as what it refuses. These tests are mostly
*separation* assertions -- that two different requests do not collide -- because a
key that is too coarse fails silently and looks like a speedup.
"""

import os

import pytest

from proteinfoundation.evaluation.binder_eval_utils import apo_fold_fingerprint
from proteinfoundation.evaluation.monomer_eval_utils import (
    DesignabilityResult,
    monomer_fold_cache_path,
    monomer_fold_fingerprint,
    read_monomer_fold_cache,
    write_monomer_fold_cache,
)
from proteinfoundation.metrics.seeding import mpnn_seed

BASE = dict(  # noqa: C408
    reference_pdb_path="/d/x_binder.pdb",
    suffix="mpnn",
    folding_models=["esmfold"],
    model_identities={"esmfold": "facebook/esmfold_v1"},
    num_seq_per_target=8,
    pmpnn_sampling_temp=0.1,
    binder_chain="B",
)


def result(values, sequences=("AAA",)):
    return DesignabilityResult(
        rmsd_values={"ca": {"esmfold": list(values)}},
        best_rmsd={"ca": {"esmfold": min(values) if values else float("inf")}},
        folded_paths=[],
        sequences=list(sequences),
    )


# ------------------------------------------------- writing a useless result


@pytest.mark.parametrize("values", [[float("inf")], [float("nan")], [float("inf"), float("nan")]])
def test_a_wholly_failed_refold_is_not_cached(tmp_path, values):
    """Caching it makes one bad run permanent: every later resume serves the
    failure instead of retrying a transient GPU or model fault."""
    write_monomer_fold_cache(str(tmp_path), "mpnn", "fp", result(values), keep_outputs=False)
    assert not os.path.exists(monomer_fold_cache_path(str(tmp_path), "mpnn"))


def test_one_usable_value_among_failures_is_worth_keeping(tmp_path):
    """Partial failure is normal -- a single sequence failing to fold should not
    discard the ones that worked."""
    write_monomer_fold_cache(str(tmp_path), "mpnn", "fp", result([float("inf"), 1.2], ("A", "B")), keep_outputs=False)
    cached = read_monomer_fold_cache(str(tmp_path), "mpnn", "fp")
    assert cached is not None and cached["rmsd_values"]["ca"]["esmfold"] == [float("inf"), 1.2]


def test_a_different_request_does_not_read_this_entry(tmp_path):
    write_monomer_fold_cache(str(tmp_path), "mpnn", "fp", result([1.2]), keep_outputs=False)
    assert read_monomer_fold_cache(str(tmp_path), "mpnn", "other-fp") is None


# ------------------------------------------------ monomer fold key separation


def test_redesign_conditioning_separates_entries():
    """The sequences are stored, not keyed on, so the key must cover what
    produced them. Without this, an entry written when designability redesigned
    the binder alone would be served for a request that redesigns it in the
    target's context -- the old metric under the new name."""
    binder_only = monomer_fold_fingerprint(
        **BASE, mpnn_context_chains=["B"], mpnn_seed_value=mpnn_seed("x", ["B"], ["B"])
    )
    complexed = monomer_fold_fingerprint(
        **BASE, mpnn_context_chains=["A", "B"], mpnn_seed_value=mpnn_seed("x", ["A", "B"], ["B"])
    )
    assert binder_only != complexed


def test_inverse_folder_separates_entries():
    """metric.inverse_folding_model governs both tracks now; flipping it must not
    serve the previous model's redesigns."""
    keys = {
        monomer_fold_fingerprint(**BASE, mpnn_context_chains=["A", "B"], mpnn_seed_value=1, inverse_folding_model=m)
        for m in ("protein_mpnn", "soluble_mpnn", "ligand_mpnn")
    }
    assert len(keys) == 3


def test_chain_order_does_not_change_the_key():
    """Chain lists are a set in meaning; ordering them differently is the same
    request and must hit."""
    a = monomer_fold_fingerprint(**BASE, mpnn_context_chains=["A", "B"], mpnn_seed_value=1)
    b = monomer_fold_fingerprint(**BASE, mpnn_context_chains=["B", "A"], mpnn_seed_value=1)
    assert a == b


def test_codesignability_key_is_untouched_by_redesign_fields():
    """It runs no inverse folder, so invalidating its folds -- potentially a
    diffusion sampler over every design -- would cost compute for information
    that does not apply to it."""
    codes = dict(BASE, suffix="pdb")
    assert monomer_fold_fingerprint(**codes) == monomer_fold_fingerprint(
        **codes, mpnn_context_chains=None, mpnn_seed_value=None, inverse_folding_model=None
    )


# ---------------------------------------------------------- apo fold key


def test_apo_key_covers_the_sequences_because_they_are_an_input():
    """Unlike designability, these come from the complex track rather than being
    generated here, so they belong in the key rather than being stood in for."""
    base = dict(binder_pdb_path="/d/x_binder.pdb", folding_models=["esmfold"], model_identities={"esmfold": "v1"})  # noqa: C408
    same = apo_fold_fingerprint(sequences=["AAA", "BBB"], **base)
    assert same == apo_fold_fingerprint(sequences=["AAA", "BBB"], **base)
    assert same != apo_fold_fingerprint(sequences=["AAA", "CCC"], **base)


def test_apo_key_is_order_sensitive_because_index_i_is_a_sequence():
    """The apo values are positionally aligned with the holo ones; reordering them
    would pair each sequence with another's verdict."""
    base = dict(binder_pdb_path="/d/x_binder.pdb", folding_models=["esmfold"], model_identities={"esmfold": "v1"})  # noqa: C408
    assert apo_fold_fingerprint(sequences=["AAA", "BBB"], **base) != apo_fold_fingerprint(
        sequences=["BBB", "AAA"], **base
    )


def test_apo_key_separates_folding_models():
    assert apo_fold_fingerprint(
        binder_pdb_path="/p", sequences=["A"], folding_models=["esmfold"], model_identities={"esmfold": "v1"}
    ) != apo_fold_fingerprint(
        binder_pdb_path="/p", sequences=["A"], folding_models=["esmfold2"], model_identities={"esmfold2": "fast"}
    )


# ------------------------------------------- advisory: scores vs structures


def _stub_backend(calls):
    def scorer(target_seqs, seq, cfg, out_pdb):
        calls.append((seq, out_pdb))
        if out_pdb:
            os.makedirs(os.path.dirname(out_pdb), exist_ok=True)
            with open(out_pdb, "w") as handle:
                handle.write("PDB\n")
        return {"i_pAE": 0.1, "pdb_path": out_pdb}

    return scorer


def test_cached_scores_do_not_satisfy_a_request_for_structures(tmp_path, monkeypatch):
    """An earlier run with keep_folding_outputs=false cached metrics and wrote no
    PDB. Enabling retention later must produce the file: the request was for a
    file and the cache can only answer about a number."""
    from proteinfoundation.metrics import consensus_folding as cf

    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", _stub_backend(calls))
    args = dict(target_seqs=["TGT"], binder_seqs=["AAA", "BBB"], cfg={}, cache_dir=str(tmp_path), reuse_cache=True)  # noqa: C408

    cf.score_binders("esmfold2", keep_structures=False, **args)
    assert len(calls) == 2  # cold cache

    calls.clear()
    cf.score_binders("esmfold2", keep_structures=False, **args)
    assert calls == []  # scores suffice when no structure is wanted

    calls.clear()
    cf.score_binders("esmfold2", keep_structures=True, **args)
    assert len(calls) == 2  # structures wanted and absent -> refold

    calls.clear()
    cf.score_binders("esmfold2", keep_structures=True, **args)
    assert calls == []  # now present -> no refold


def test_a_deleted_structure_is_regenerated(tmp_path, monkeypatch):
    from proteinfoundation.metrics import consensus_folding as cf

    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", _stub_backend(calls))
    args = dict(target_seqs=["TGT"], binder_seqs=["AAA"], cfg={}, cache_dir=str(tmp_path), reuse_cache=True)  # noqa: C408
    cf.score_binders("esmfold2", keep_structures=True, **args)
    path = calls[0][1]
    os.remove(path)

    calls.clear()
    cf.score_binders("esmfold2", keep_structures=True, **args)
    assert len(calls) == 1 and os.path.exists(path)
