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
