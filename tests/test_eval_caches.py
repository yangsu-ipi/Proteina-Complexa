"""Cache keys and cache-write guards.

Two review findings were here, and both were the same shape: a cache answering a
question it was not asked. One cached a wholly failed refold because the guard
tested list non-emptiness under a variable named ``finite``; the other returned
scores when the caller had asked for structures.

A cache key is only as good as what it refuses. These tests are mostly
*separation* assertions -- that two different requests do not collide -- because a
key that is too coarse fails silently and looks like a speedup.
"""

import json
import os
import pathlib

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
    # Takes `seed`: score_binders derives seeds and passes one per fold, so a
    # backend is now a pure function of (target, binder, cfg, out_pdb, seed).
    def scorer(target_seqs, seq, cfg, out_pdb, seed=0, context=None):
        calls.append((seq, out_pdb, seed))
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


def test_the_self_mismatch_guard_returns_what_the_caller_unpacks(monkeypatch, tmp_path):
    """The guard that drops apo columns must not drop them by raising.

    ``apo_refold`` for ``self`` delegates to codesignability and then checks that
    the sequence read off the binder PDB is the one whose holo metrics sit on the
    row. When it is not, it returns empties. It returned two of them while every
    other exit returned three, and the caller's unpack sits outside the try that
    turns a failed refold into NaN -- so the mismatch guard, whose whole purpose
    is to drop the columns gracefully, would instead take the design's row down
    with a ValueError. The one path nobody folds a structure to reach is the one
    path a test has to cover.
    """
    from proteinfoundation.evaluation import binder_eval, monomer_eval
    from proteinfoundation.evaluation.monomer_eval_utils import DesignabilityResult

    monkeypatch.setattr(
        monomer_eval,
        "evaluate_self_consistency",
        lambda **kwargs: DesignabilityResult(
            rmsd_values={"ca": {"esmfold2": [1.0]}},
            sequences=["WHAT THE PDB SAYS"],
            plddt={"esmfold2": [0.8]},
        ),
        raising=False,
    )

    returned = binder_eval.apo_refold(
        seq_type="self",
        sequences=["WHAT THE ROW SAYS"],
        binder_pdb_path=str(tmp_path / "d_binder.pdb"),
        sample_root_path=str(tmp_path),
        folding_models=["esmfold2"],
        rmsd_modes=["ca"],
        keep_outputs=False,
        reuse_cache=False,
    )

    assert len(returned) == 3, "the caller unpacks three"
    rmsds, plddt, derived = returned
    assert not rmsds and not plddt and not derived, "a mispaired fold contributes nothing"


def _cached_fold(tmp_path, paths, modes=("ca",)):
    kept = str(tmp_path / "kept.pdb")
    open(kept, "w").close()
    return {
        "sequences": ["AAA", "CCC"],
        "rmsd_values": {m: {"esmfold2": [1.0, 2.0]} for m in modes},
        "best_rmsd": {m: {"esmfold2": 1.0} for m in modes},
        "folded_paths": {"esmfold2": [kept if p else None for p in paths]},
        "structures_kept": True,
        "plddt": {"esmfold2": [0.7, 0.6]},
    }


def test_adding_a_mode_reads_the_kept_structures_instead_of_refolding(monkeypatch, tmp_path):
    """The payoff for keeping structures: apo geometry went from one mode to four,
    and that must cost a re-read. Modes are deliberately absent from the fold
    fingerprint so this path is reachable at all."""
    from proteinfoundation.evaluation import monomer_eval

    seen = {}

    def fake(reference_pdb_path, folding_results, rmsd_modes):
        seen["modes"] = list(rmsd_modes)
        seen["results"] = folding_results
        return DesignabilityResult(
            rmsd_values={m: {"esmfold2": [3.0, 4.0]} for m in rmsd_modes},
        )

    monkeypatch.setattr(monomer_eval, "compute_scrmsd_from_folded", fake)
    out = monomer_eval._result_from_cache(
        _cached_fold(tmp_path, [True, True]), ["ca", "bb3", "all_atom"], "/design.pdb"
    )

    assert out is not None, "the kept structures answer this"
    assert seen["modes"] == ["bb3", "all_atom"], "only the modes the cache lacks"
    assert out.rmsd_values["ca"]["esmfold2"] == [1.0, 2.0], "the cached mode is not recomputed"


def test_one_failed_fold_does_not_force_a_refold_of_the_rest(monkeypatch, tmp_path):
    """A None slot is a fold that failed, not a structure that went missing -- its
    RMSD is inf in every mode, so a new mode costs nothing for it. Requiring every
    slot to be a readable file sent the whole design back through the folder the
    first time a mode was added, which is the one moment refolding is the thing
    being avoided."""
    from proteinfoundation.evaluation import monomer_eval

    seen = {}

    def fake(reference_pdb_path, folding_results, rmsd_modes):
        seen["results"] = folding_results
        return DesignabilityResult(
            rmsd_values={m: {"esmfold2": [3.0, float("inf")]} for m in rmsd_modes},
        )

    monkeypatch.setattr(monomer_eval, "compute_scrmsd_from_folded", fake)
    out = monomer_eval._result_from_cache(
        _cached_fold(tmp_path, [True, False]), ["ca", "all_atom"], "/design.pdb"
    )

    assert out is not None, "one failed sequence must not refold the design"
    success = [r.success for r in seen["results"]["esmfold2"]]
    assert success == [True, False], "the failed slot is handed on as failed, not as a path"


def test_a_structure_deleted_since_the_fold_still_refolds(tmp_path):
    """The relaxation above must not swallow the case it was carved out of: a path
    that names a file which is no longer there cannot answer for a new mode."""
    from proteinfoundation.evaluation import monomer_eval

    cached = _cached_fold(tmp_path, [True, True])
    os.remove(cached["folded_paths"]["esmfold2"][0])
    assert monomer_eval._result_from_cache(cached, ["ca", "all_atom"], "/design.pdb") is None


def test_a_new_mode_is_averaged_over_seeds_like_the_cached_one(monkeypatch, tmp_path):
    """Found on the real CBLN1 apo caches. average_folds reduces the RMSDs over
    seeds to one value per sequence but CONCATENATES the structures, so a
    three-seed entry holds two RMSDs and six paths. Measuring a new mode on the
    averaged entry produced six values against two sequences -- a list three
    times too long, aligned with nothing, in a column whose whole contract is
    that position i is sequence i. Modes are filled per seed, then averaged."""
    from proteinfoundation.evaluation import monomer_eval

    kept = str(tmp_path / "kept.pdb")
    open(kept, "w").close()

    def fold(seed, ca):
        return {
            "sequences": ["AAA", "CCC"],
            "rmsd_values": {"ca": {"esmfold2": list(ca)}},
            "best_rmsd": {"ca": {"esmfold2": min(ca)}},
            "folded_paths": {"esmfold2": [kept, kept]},
            "structures_kept": True,
        }

    per_seed = {1: fold(1, [1.0, 2.0]), 2: fold(2, [3.0, 4.0]), 3: fold(3, [5.0, 6.0])}

    def fake(reference_pdb_path, folding_results, rmsd_modes):
        n = len(folding_results["esmfold2"])
        return DesignabilityResult(
            rmsd_values={m: {"esmfold2": [7.0] * n} for m in rmsd_modes},
        )

    monkeypatch.setattr(monomer_eval, "compute_scrmsd_from_folded", fake)
    out = monomer_eval._result_from_folds(per_seed, ["ca", "all_atom"], "/design.pdb")

    assert out is not None
    assert out.rmsd_values["ca"]["esmfold2"] == pytest.approx([3.0, 4.0]), "the seeds averaged"
    assert len(out.rmsd_values["all_atom"]["esmfold2"]) == len(out.rmsd_values["ca"]["esmfold2"]) == 2, (
        "one value per sequence, whatever the seed count"
    )


def test_an_averaged_entry_is_never_measured_directly(monkeypatch, tmp_path):
    """The guard that keeps the shape above honest: more structures than
    sequences means the entry is an average of several seeds, and measuring it
    would produce a list of the wrong length rather than a wrong number, which no
    downstream check looks for."""
    from proteinfoundation.evaluation import monomer_eval

    kept = str(tmp_path / "kept.pdb")
    open(kept, "w").close()
    averaged = {
        "sequences": ["AAA", "CCC"],
        "rmsd_values": {"ca": {"esmfold2": [1.0, 2.0]}},
        "best_rmsd": {"ca": {"esmfold2": 1.0}},
        "folded_paths": {"esmfold2": [kept] * 6},
        "structures_kept": True,
    }

    def fake(**kwargs):
        raise AssertionError("an averaged entry must not be measured")

    monkeypatch.setattr(monomer_eval, "compute_scrmsd_from_folded", fake)
    assert monomer_eval._fill_missing_modes(averaged, ["ca", "bb3"], "/design.pdb") is averaged
    assert monomer_eval._result_from_cache(averaged, ["ca", "bb3"], "/design.pdb") is None


# ---------------------------------------------------------------------------
# One cache per backend. A shared fingerprint made enabling a second folder
# discard the first one's folds for a reason that had nothing to do with how it
# folded them -- ~13,000 cached ESMFold2 apo folds per campaign, thrown away by
# adding ColabFold beside them.
# ---------------------------------------------------------------------------


def test_a_backends_fingerprint_does_not_mention_the_backends_beside_it():
    """The property the split turns on: what identifies an ESMFold2 fold cannot
    change because ColabFold was enabled. Nothing about ColabFold went into
    producing that structure."""
    from proteinfoundation.evaluation.binder_eval_utils import apo_fold_fingerprint

    def fp(models):
        return apo_fold_fingerprint(
            binder_pdb_path="/d/x_binder.pdb",
            sequences=["AAA", "CCC"],
            folding_models=models,
            model_identities={m: f"id-{m}" for m in models},
        )

    assert fp(["esmfold2"]) != fp(["esmfold2", "colabfold"]), "the old, shared key"
    assert fp(["esmfold2"]) == fp(["esmfold2"]), "and the per-model one is stable"
    assert fp(["colabfold"]) != fp(["esmfold2"]), "two backends are two caches"


def test_a_single_backend_cache_is_adopted_by_that_backend(tmp_path):
    """What makes the split free rather than a migration: the per-model
    fingerprint is the same function with a one-element list, so a campaign that
    ran with one backend already wrote exactly the bytes that backend now asks
    for. Its folds are reused, not recomputed."""
    from proteinfoundation.evaluation.monomer_eval_utils import (
        monomer_fold_cache_path,
        read_monomer_folds,
    )

    # A cache as a pre-split run left it: the legacy shared filename.
    legacy = monomer_fold_cache_path(str(tmp_path), "apo_mpnn")
    assert legacy.endswith("monomer_fold_cache_apo_mpnn.json")
    per_model = monomer_fold_cache_path(str(tmp_path), "apo_mpnn", "esmfold2")
    assert per_model.endswith("monomer_fold_cache_apo_mpnn_esmfold2.json")

    with open(legacy, "w") as handle:
        json.dump(
            {
                "fingerprint": "FP",
                "schema": 3,
                "folds": {"7": {"sequences": ["AAA"], "rmsd_values": {"ca": {"esmfold2": [1.0]}}}},
            },
            handle,
        )

    adopted = read_monomer_folds(str(tmp_path), "apo_mpnn", "FP", model="esmfold2")
    assert adopted and 7 in adopted, "the legacy file answers for the backend that wrote it"
    assert read_monomer_folds(str(tmp_path), "apo_mpnn", "OTHER", model="colabfold") is None


def test_the_first_write_after_the_split_keeps_what_it_reused(tmp_path):
    """Otherwise the adoption undoes itself: the folds are reused in memory,
    written to a file that does not contain them, and refolded on the next
    resume -- every run, forever."""
    from proteinfoundation.evaluation.monomer_eval_utils import (
        monomer_fold_cache_path,
        read_monomer_folds,
        write_monomer_fold_cache,
    )

    legacy = monomer_fold_cache_path(str(tmp_path), "apo_mpnn")
    with open(legacy, "w") as handle:
        json.dump(
            {
                "fingerprint": "FP",
                "schema": 3,
                "folds": {"7": {"sequences": ["AAA"], "rmsd_values": {"ca": {"esmfold2": [1.0]}}}},
            },
            handle,
        )

    write_monomer_fold_cache(
        str(tmp_path), "apo_mpnn", "FP", result([2.0]), False, seed=9, seed_index=1, model="esmfold2"
    )

    got = read_monomer_folds(str(tmp_path), "apo_mpnn", "FP", model="esmfold2")
    assert got is not None and set(got) == {7, 9}, "the reused seed and the new one"
    assert os.path.exists(monomer_fold_cache_path(str(tmp_path), "apo_mpnn", "esmfold2"))


def test_a_deterministic_folder_is_not_asked_for_three_seeds():
    """Seeds are a property of a backend. Derived over the whole model list, a
    deterministic folder got however many a sampler beside it wanted -- three
    identical ColabFold structures per design, averaged with themselves, at the
    cost of three folds."""
    from proteinfoundation.evaluation.monomer_eval_utils import _fold_seeds

    seqs = ["AAA", "CCC"]
    assert len(_fold_seeds("d", "apo_mpnn", seqs, ["esmfold2"], 3)) == 3
    assert len(_fold_seeds("d", "apo_mpnn", seqs, ["colabfold"], 3)) == 1
    # The shared list is what made the deterministic one expensive.
    assert len(_fold_seeds("d", "apo_mpnn", seqs, ["esmfold2", "colabfold"], 3)) == 3


def test_two_backends_merge_into_the_shape_one_backend_gives():
    """Every field below the top level is already keyed by model, so combining
    two backends is a union -- and nothing downstream can tell that it did not
    come from one file."""
    from proteinfoundation.evaluation.monomer_eval_utils import merge_model_folds

    merged = merge_model_folds(
        {
            "esmfold2": {
                "sequences": ["AAA"],
                "rmsd_values": {"ca": {"esmfold2": [1.0]}},
                "best_rmsd": {"ca": {"esmfold2": 1.0}},
                "plddt": {"esmfold2": [0.7]},
                "folded_paths": {"esmfold2": ["/a.pdb"]},
                "structures_kept": True,
            },
            "colabfold": {
                "sequences": ["AAA"],
                "rmsd_values": {"ca": {"colabfold": [2.0]}},
                "best_rmsd": {"ca": {"colabfold": 2.0}},
                "plddt": {"colabfold": [0.9]},
                "folded_paths": {"colabfold": ["/b.pdb"]},
                "structures_kept": True,
            },
        }
    )
    assert merged["rmsd_values"]["ca"] == {"esmfold2": [1.0], "colabfold": [2.0]}
    assert set(merged["plddt"]) == {"esmfold2", "colabfold"}
    assert set(merged["folded_paths"]) == {"esmfold2", "colabfold"}


def test_backends_that_disagree_about_the_sequences_are_not_merged():
    """Two caches describing different sequences describe different designs, and
    pairing one model's folds with another's sequences is the one failure the
    apo columns must not introduce."""
    from proteinfoundation.evaluation.monomer_eval_utils import merge_model_folds

    merged = merge_model_folds(
        {
            "esmfold2": {"sequences": ["AAA"], "rmsd_values": {"ca": {"esmfold2": [1.0]}}},
            "colabfold": {"sequences": ["DIFFERENT"], "rmsd_values": {"ca": {"colabfold": [2.0]}}},
        }
    )
    assert set(merged["rmsd_values"]["ca"]) == {"esmfold2"}, "the disagreeing backend is dropped"


def test_a_cache_written_before_best_rmsd_was_dropped_still_reads(tmp_path):
    """best_rmsd was computed, cached, merged and reconstructed, and nothing ever
    read its value -- compute_monomer_metrics takes its own min() over
    rmsd_values for the _res_scRMSD_* columns. Worse than dead: _result_from_cache
    indexed it per requested mode, so a value nobody used could KeyError a cache
    read for a mode it happened to lack.

    Removing it must not invalidate a campaign's caches. Every entry on disk
    carries the key; it is now simply ignored."""
    from proteinfoundation.evaluation.monomer_eval import _result_from_cache

    legacy = {
        "sequences": ["MKV"],
        "rmsd_values": {"ca": {"esmfold2": [1.25]}},
        # Written by a version that still recorded it.
        "best_rmsd": {"ca": {"esmfold2": 1.25}},
        "folded_paths": {"esmfold2": [None]},
        "plddt": {"esmfold2": [0.8]},
        "confidence": {"esmfold2": {"pTM": [0.5], "pAE": [0.2]}},
        "structures_kept": False,
    }
    result = _result_from_cache(legacy, ["ca"], str(tmp_path / "ref.pdb"))
    assert result is not None, "an old entry must still rebuild"
    assert result.rmsd_values["ca"]["esmfold2"] == [1.25]
    assert not hasattr(result, "best_rmsd"), "and the field is gone from the result"

    # The mode that used to raise: present in rmsd_values, absent from best_rmsd.
    lopsided = dict(legacy, rmsd_values={"ca": {"esmfold2": [1.25]}, "bb3": {"esmfold2": [1.4]}})
    rebuilt = _result_from_cache(lopsided, ["ca", "bb3"], str(tmp_path / "ref.pdb"))
    assert rebuilt is not None and rebuilt.rmsd_values["bb3"]["esmfold2"] == [1.4]


def test_an_af2_request_adopts_the_folds_cached_under_the_old_folder_name(tmp_path):
    """The folder name rides in BOTH the cache filename and the fingerprint, so
    renaming colabfold to af2 would otherwise refold every structure that folder
    ever produced -- to learn what is already on disk. Adopted only against the
    fingerprint the OLD name would have written, so a hit is a fold this run
    would have produced, not merely a file that happens to be there."""
    from proteinfoundation.evaluation.monomer_eval_utils import (
        legacy_fold_fingerprints,
        monomer_fold_cache_path,
        read_monomer_folds,
    )

    def fingerprint_for(backend: str) -> str:
        return f"fp-for-{backend}"

    assert legacy_fold_fingerprints("af2", fingerprint_for) == {"colabfold": "fp-for-colabfold"}
    assert legacy_fold_fingerprints("esmfold2", fingerprint_for) == {}, "only a renamed folder has one"

    legacy_path = monomer_fold_cache_path(str(tmp_path), "apo_mpnn", "colabfold")
    pathlib.Path(legacy_path).write_text(
        json.dumps(
            {
                "fingerprint": "fp-for-colabfold",
                "schema": 3,
                "folds": {
                    "77": {
                        "sequences": ["MKV"],
                        "rmsd_values": {"ca": {"af2": [1.5]}},
                        "folded_paths": {"af2": [None]},
                        "plddt": {"af2": [0.9]},
                        "confidence": {},
                        "structures_kept": False,
                    }
                },
            }
        )
    )

    adopted = read_monomer_folds(
        str(tmp_path),
        "apo_mpnn",
        "fp-for-af2",
        model="af2",
        also_accept=legacy_fold_fingerprints("af2", fingerprint_for),
    )
    assert adopted is not None, "the old name's folds are this folder's folds"
    assert adopted[77]["rmsd_values"]["ca"]["af2"] == [1.5]

    # A file under the old name whose fingerprint does not match is still refused.
    pathlib.Path(legacy_path).write_text(json.dumps({"fingerprint": "something-else", "folds": {}}))
    assert (
        read_monomer_folds(
            str(tmp_path),
            "apo_mpnn",
            "fp-for-af2",
            model="af2",
            also_accept=legacy_fold_fingerprints("af2", fingerprint_for),
        )
        is None
    ), "adoption is by key, not by filename"
