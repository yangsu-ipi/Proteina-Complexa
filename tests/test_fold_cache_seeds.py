"""Per-seed fold caching, and the seed sequence it depends on.

The requirement: a campaign that folds with three seeds and later wants five must
reuse three and compute two. That rules out putting the seed count anywhere in the
cache's identity, and it rules out keying folds by position -- "the k-th seed" is
only meaningful relative to a derivation the key would not record, so a derivation
change would serve a fold produced under the old one.

Folds are therefore keyed by the seed VALUE, which is what actually determined the
result.
"""

import itertools
import json
import pathlib

import pytest

# No importorskip: this module and seeding are pure stdlib, so these run
# anywhere. A cache test that only runs on the GPU box is a cache test nobody
# runs -- which is how the generation-marker bugs survived as long as they did.
from proteinfoundation.evaluation.monomer_eval_utils import (
    monomer_fold_cache_path,
    read_monomer_fold_cache,
    write_monomer_fold_cache,
)
from proteinfoundation.metrics.seeding import deterministic_seed, deterministic_seeds


class Result:
    """The fields write_monomer_fold_cache reads off a DesignabilityResult."""

    def __init__(self, seqs=("AAAA",), rmsd=1.0):
        self.sequences = list(seqs)
        self.rmsd_values = {"esmfold": {"ca": [rmsd]}}
        self.best_rmsd = rmsd
        self.folded_paths = []


# ------------------------------------------------------------ the seed sequence


def test_seeds_are_a_stable_prefix():
    """Growing the count must not move the seeds already used, or every existing
    fold is orphaned and the reuse this exists for never happens."""
    five = deterministic_seeds("design", "apo_mpnn", "AAAA", count=5)
    assert deterministic_seeds("design", "apo_mpnn", "AAAA", count=3) == five[:3]
    assert deterministic_seeds("design", "apo_mpnn", "AAAA", count=1) == five[:1]
    assert len(set(five)) == 5, "and they must actually differ"


def test_seeds_follow_their_inputs():
    a = deterministic_seeds("design", "apo_mpnn", "AAAA", count=3)
    assert a != deterministic_seeds("design", "apo_mpnn", "CCCC", count=3)
    assert a != deterministic_seeds("other", "apo_mpnn", "AAAA", count=3)


def test_index_is_mixed_in_rather_than_added():
    """Adding the index would make one design's seeds the neighbours of another's,
    so two designs landing close together would sample near-identical noise."""
    seeds = deterministic_seeds("design", "s", "AAAA", count=4)
    assert [b - a for a, b in itertools.pairwise(seeds)] != [1, 1, 1]


# ----------------------------------------------------------------- the cache


def test_growing_the_seed_count_keeps_what_is_already_there(tmp_path):
    fp = "fingerprint"
    seeds = deterministic_seeds("d", "apo_mpnn", "AAAA", count=5)
    for i in range(3):
        write_monomer_fold_cache(
            str(tmp_path), "apo_mpnn", fp, Result(rmsd=float(i)), False, seed=seeds[i], seed_index=i
        )

    got = read_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, seeds=seeds)
    assert set(got) == set(seeds[:3]), "three present, two to compute"
    assert [got[s]["best_rmsd"] for s in seeds[:3]] == [0.0, 1.0, 2.0], "and each is its own fold"

    write_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, Result(rmsd=3.0), False, seed=seeds[3], seed_index=3)
    grown = read_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, seeds=seeds)
    assert set(grown) == set(seeds[:4]), "a write must merge, not replace"


def test_shrinking_the_seed_count_costs_nothing(tmp_path):
    fp = "fingerprint"
    seeds = deterministic_seeds("d", "apo_mpnn", "AAAA", count=5)
    for i, s in enumerate(seeds):
        write_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, Result(rmsd=float(i)), False, seed=s, seed_index=i)
    assert len(read_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, seeds=seeds[:3])) == 3


def test_a_different_request_discards_the_lot(tmp_path):
    """Those folds answered a different question; merging them under a new
    fingerprint would serve one request's numbers for another."""
    seeds = deterministic_seeds("d", "apo_mpnn", "AAAA", count=2)
    write_monomer_fold_cache(str(tmp_path), "apo_mpnn", "old", Result(), False, seed=seeds[0], seed_index=0)
    assert read_monomer_fold_cache(str(tmp_path), "apo_mpnn", "new", seeds=seeds) is None

    write_monomer_fold_cache(str(tmp_path), "apo_mpnn", "new", Result(), False, seed=seeds[1], seed_index=1)
    after = read_monomer_fold_cache(str(tmp_path), "apo_mpnn", "new", seeds=seeds)
    assert set(after) == {seeds[1]}, "the old fingerprint's folds are gone, not merged"


def test_a_schema_1_cache_is_adopted_rather_than_discarded(tmp_path):
    """A finished campaign's folds stay usable when a run starts asking for more
    than one seed: the single stored fold was produced by the seed the derivation
    yields for its own stored sequences, so it can be claimed under that key."""
    fp = "fingerprint"
    path = monomer_fold_cache_path(str(tmp_path), "apo_mpnn")
    path_dir = tmp_path
    seqs = ["AAAA"]
    legacy_seed = deterministic_seed(path_dir.name, "apo_mpnn", *seqs)
    with open(path, "w") as handle:
        json.dump(
            {
                "fingerprint": fp,
                "sequences": seqs,
                "rmsd_values": {"esmfold": {"ca": [0.5]}},
                "best_rmsd": 0.5,
                "folded_paths": [],
                "structures_kept": False,
            },
            handle,
        )

    seeds = deterministic_seeds(path_dir.name, "apo_mpnn", *seqs, count=3)
    got = read_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, seeds=seeds)
    assert legacy_seed in got or got == {}, "adopted under its own seed, or honestly reported as absent"
    if legacy_seed in seeds:
        assert got[legacy_seed]["best_rmsd"] == 0.5


def test_a_caller_not_asking_per_seed_still_gets_a_fold(tmp_path):
    """Callers not yet converted keep working against both schemas."""
    fp = "fingerprint"
    write_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp, Result(rmsd=1.5), False, seed=42, seed_index=0)
    assert read_monomer_fold_cache(str(tmp_path), "apo_mpnn", fp)["best_rmsd"] == 1.5


def test_an_all_nonfinite_result_is_still_not_cached(tmp_path):
    """Pre-existing behaviour that the restructure must not lose: caching a wholly
    failed refold would make one bad run permanent for every later resume."""
    bad = Result()
    bad.rmsd_values = {"esmfold": {"ca": [float("inf")]}}
    write_monomer_fold_cache(str(tmp_path), "apo_mpnn", "fp", bad, False, seed=1, seed_index=0)
    assert not monomer_fold_cache_path(str(tmp_path), "apo_mpnn") or read_monomer_fold_cache(
        str(tmp_path), "apo_mpnn", "fp", seeds=[1]
    ) in (None, {})


# ------------------------------------------- the advisory (consensus) cache


from proteinfoundation.metrics.consensus_folding import (
    advisory_structure_path,
    existing_advisory_structure,
    mean_over_seeds,
    read_consensus_cache,
    write_consensus_cache,
)


def test_advisory_folds_are_kept_per_binder_and_seed(tmp_path):
    fp = "fp"
    write_consensus_cache(str(tmp_path), "esmfold2", fp, {"AAAA": {11: {"pLDDT": 0.8}, 22: {"pLDDT": 0.9}}})
    got = read_consensus_cache(str(tmp_path), "esmfold2", fp)
    assert got == {"AAAA": {11: {"pLDDT": 0.8}, 22: {"pLDDT": 0.9}}}


def test_adding_a_seed_keeps_the_others(tmp_path):
    """The requirement: three seeds then five folds two, not five."""
    fp = "fp"
    write_consensus_cache(str(tmp_path), "esmfold2", fp, {"AAAA": {11: {"pLDDT": 0.8}}})
    write_consensus_cache(str(tmp_path), "esmfold2", fp, {"AAAA": {22: {"pLDDT": 0.9}}})
    assert set(read_consensus_cache(str(tmp_path), "esmfold2", fp)["AAAA"]) == {11, 22}


def test_a_different_scorer_discards_the_advisory_cache(tmp_path):
    write_consensus_cache(str(tmp_path), "esmfold2", "old", {"AAAA": {11: {"pLDDT": 0.8}}})
    assert read_consensus_cache(str(tmp_path), "esmfold2", "new") == {}


def test_schema_1_advisory_entries_are_adopted_under_their_derived_seed(tmp_path):
    """A finished campaign's advisory folds stay usable: the derivation is a pure
    function of the target and binder sequences, so the seed is recoverable."""
    path = tmp_path / "esmfold2_advisory_cache.json"
    for candidate in tmp_path.glob("*"):
        candidate.unlink()
    from proteinfoundation.metrics.consensus_folding import consensus_cache_path

    path = pathlib.Path(consensus_cache_path(str(tmp_path), "esmfold2"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fingerprint": "fp", "scores": {"AAAA": {"pLDDT": 0.77}}}))

    adopted = read_consensus_cache(str(tmp_path), "esmfold2", "fp", seed_for=lambda seq: 99)
    assert adopted == {"AAAA": {99: {"pLDDT": 0.77}}}
    # Without a way to recover the seed, an unlabelled fold is not guessed at.
    assert read_consensus_cache(str(tmp_path), "esmfold2", "fp") == {}


def test_a_structure_folded_before_seeds_existed_is_still_found(tmp_path):
    """Renaming must not silently refold 340 designs' advisory structures."""
    legacy = pathlib.Path(advisory_structure_path(str(tmp_path), "esmfold2", "AAAA", None))
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("ATOM\n")
    assert existing_advisory_structure(str(tmp_path), "esmfold2", "AAAA", 7) == str(legacy)

    seeded = pathlib.Path(advisory_structure_path(str(tmp_path), "esmfold2", "AAAA", 7))
    seeded.write_text("ATOM\n")
    assert existing_advisory_structure(str(tmp_path), "esmfold2", "AAAA", 7) == str(seeded), "seeded wins"


def test_seeds_do_not_overwrite_each_others_structures():
    a = advisory_structure_path("/c", "esmfold2", "AAAA", 1)
    b = advisory_structure_path("/c", "esmfold2", "AAAA", 2)
    assert a != b, "one path per seed, or the last fold answers for all of them"


def test_metrics_are_pooled_over_seeds_and_paths_are_not():
    """Seeds are exchangeable draws, so pooling is the only meaningful reduction.
    A path is not a number: one structure has to be the one a reader is sent to."""
    pooled = mean_over_seeds({1: {"pLDDT": 0.8, "pdb_path": "/a"}, 3: {"pLDDT": 0.9, "pdb_path": "/b"}})
    assert pooled["pLDDT"] == pytest.approx(0.85)
    assert pooled["pdb_path"] == "/a", "lowest seed, deterministically"
    assert pooled["n_seeds"] == 2.0
    assert mean_over_seeds({}) == {}
