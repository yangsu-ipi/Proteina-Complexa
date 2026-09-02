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


# ------------------------------------------ which folders actually need seeds


def test_only_esmfold2_gets_more_than_one_seed():
    """ESMFold v1 and ColabFold are deterministic given their inputs, so asking
    them for three seeds would fold one structure three times and average it with
    itself -- three times the cost for no variance reduction."""
    from proteinfoundation.evaluation.monomer_eval_utils import _fold_seeds

    assert len(_fold_seeds("d", "apo_mpnn", ["AAAA"], ["esmfold2"], 3)) == 3
    assert len(_fold_seeds("d", "apo_mpnn", ["AAAA"], ["esmfold"], 3)) == 1
    assert len(_fold_seeds("d", "apo_mpnn", ["AAAA"], ["colabfold"], 3)) == 1
    assert len(_fold_seeds("d", "apo_mpnn", ["AAAA"], ["esmfold", "esmfold2"], 3)) == 3, "any esmfold2 is enough"
    assert len(_fold_seeds("d", "apo_mpnn", ["AAAA"], ["esmfold2"], 0)) == 1, "a count below one is still one fold"


def test_seeds_are_prefix_stable_through_the_fold_layer():
    """Raising the count must fold only the new ones."""
    from proteinfoundation.evaluation.monomer_eval_utils import _fold_seeds

    five = _fold_seeds("d", "apo_mpnn", ["AAAA"], ["esmfold2"], 5)
    assert _fold_seeds("d", "apo_mpnn", ["AAAA"], ["esmfold2"], 3) == five[:3]


def test_per_seed_rmsds_average_and_keep_their_positions():
    """Averaging must stay per sequence, per mode, per model: every downstream
    column reads these lists positionally."""
    from proteinfoundation.evaluation.monomer_eval_utils import average_folds

    folds = {
        1: {
            "sequences": ["A", "B"],
            "rmsd_values": {"ca": {"esmfold2": [1.0, 3.0]}},
            "best_rmsd": 1.0,
            "folded_paths": ["/1"],
        },
        2: {
            "sequences": ["A", "B"],
            "rmsd_values": {"ca": {"esmfold2": [2.0, 5.0]}},
            "best_rmsd": 2.0,
            "folded_paths": ["/2"],
        },
    }
    got = average_folds(folds)
    assert got["rmsd_values"]["ca"]["esmfold2"] == [1.5, 4.0]
    assert set(got["folded_paths"]) == {"/1", "/2"}, "paths are kept, not averaged -- each is a real structure"


def test_one_bad_seed_does_not_average_into_a_plausible_number():
    """An infinite RMSD means that fold failed. Averaging it as a finite value
    would turn one failure into a slightly worse success."""
    from proteinfoundation.evaluation.monomer_eval_utils import average_folds

    folds = {
        1: {"sequences": ["A"], "rmsd_values": {"ca": {"esmfold2": [1.0]}}, "best_rmsd": 1.0, "folded_paths": []},
        2: {
            "sequences": ["A"],
            "rmsd_values": {"ca": {"esmfold2": [float("inf")]}},
            "best_rmsd": 0.0,
            "folded_paths": [],
        },
    }
    assert average_folds(folds)["rmsd_values"]["ca"]["esmfold2"] == [float("inf")]


def test_no_usable_folds_is_reported_as_nothing_rather_than_zero():
    from proteinfoundation.evaluation.monomer_eval_utils import average_folds

    assert average_folds({}) is None
    assert average_folds({1: {"sequences": ["A"], "rmsd_values": {}}}) is None


def test_the_first_seed_is_the_one_single_seed_runs_used():
    """Otherwise every fold a finished campaign holds is orphaned, and the cache
    adoption written to preserve them never matches a requested seed -- the
    adoption path would be dead code and the saving it exists for imaginary."""
    parts = ("design", "apo_mpnn", "AAAA")
    assert deterministic_seeds(*parts, count=3)[0] == deterministic_seed(*parts)
    assert deterministic_seeds(*parts, count=1) == [deterministic_seed(*parts)]


def test_a_schema_1_fold_is_actually_reused_end_to_end(tmp_path):
    """The adoption and the derivation have to agree, which is the thing the two
    were written apart from each other and did not."""
    from proteinfoundation.evaluation.monomer_eval_utils import _fold_seeds, read_monomer_folds

    fp = "fp"
    seqs = ["AAAA"]
    path = monomer_fold_cache_path(str(tmp_path), "apo_mpnn")
    with open(path, "w") as handle:
        json.dump(
            {
                "fingerprint": fp,
                "sequences": seqs,
                "rmsd_values": {"esmfold2": {"ca": [0.4]}},
                "best_rmsd": 0.4,
                "folded_paths": [],
            },
            handle,
        )
    stored = read_monomer_folds(str(tmp_path), "apo_mpnn", fp)
    wanted = _fold_seeds(tmp_path.name, "apo_mpnn", seqs, ["esmfold2"], 3)
    assert set(stored) & set(wanted), "the stored fold must satisfy one of the requested seeds"
    assert wanted[0] in stored, "and specifically the first, which is what it was folded as"


def test_adoption_survives_the_first_write(tmp_path):
    """Reading adopted a schema-1 fold and writing dropped it, so the fold was
    reused in memory and refolded on the next resume -- adoption undone every run,
    invisibly, because the run that did it looked correct."""
    from proteinfoundation.evaluation.monomer_eval_utils import read_monomer_folds

    fp = "fp"
    seqs = ["AAAA"]
    with open(monomer_fold_cache_path(str(tmp_path), "pdb"), "w") as handle:
        json.dump(
            {"fingerprint": fp, "sequences": seqs, "rmsd_values": {"esmfold2": {"ca": [0.4]}}, "best_rmsd": 0.4},
            handle,
        )
    legacy_seed = deterministic_seed(tmp_path.name, "pdb", *seqs)

    write_monomer_fold_cache(str(tmp_path), "pdb", fp, Result(seqs=seqs, rmsd=0.6), False, seed=999, seed_index=1)

    after = read_monomer_folds(str(tmp_path), "pdb", fp)
    assert set(after) == {legacy_seed, 999}, "the legacy fold must survive the merge, not be replaced by it"


def test_every_evaluate_self_consistency_call_passes_a_seed_count():
    """The smoke run threaded only one of four folding paths, so `pdb` folds got
    three seeds while `mpnn` and `apo_mpnn` silently stayed at one -- visible only
    by counting files on disk afterwards. A signature default of 1 makes a missed
    call site look like a deliberate choice."""
    import re

    src = pathlib.Path("src/proteinfoundation/evaluation/monomer_eval.py").read_text()
    calls = [
        m.start()
        for m in re.finditer(r"evaluate_self_consistency\(", src)
        if "def " not in src[max(0, m.start() - 40) : m.start()]
    ]
    for start in calls:
        depth, i = 0, src.index("(", start)
        for j in range(i, len(src)):
            depth += src[j] == "("
            depth -= src[j] == ")"
            if depth == 0:
                break
        assert "n_esmfold2_seeds" in src[i:j], f"a call at offset {start} does not pass n_esmfold2_seeds"


def test_the_apo_path_folds_every_seed_too():
    """apo_refold's redesign branch has its own fold-and-cache flow rather than
    delegating, so threading the codesignability path left it at one seed."""
    src = pathlib.Path("src/proteinfoundation/evaluation/binder_eval.py").read_text()
    body = src[src.index("def apo_refold(") : src.index("def ", src.index("def apo_refold(") + 10)]
    assert "_fold_seeds(" in body, "the apo branch must derive its seeds"
    assert "average_folds(" in body, "and average across them"


def test_the_advisory_path_inherits_the_seed_count():
    """consensus_cfg is a separate dict, so metric.n_esmfold2_seeds did not reach
    it -- three seeds everywhere except the complex folds, which are the expensive
    ones. An explicit consensus_cfg.n_seeds still wins."""
    src = pathlib.Path("src/proteinfoundation/evaluation/binder_eval.py").read_text()
    i = src.index("consensus_cfg = dict(")
    window = src[i : i + 700]
    assert 'consensus_cfg.setdefault("n_seeds", n_esmfold2_seeds)' in window
    assert src.index("n_esmfold2_seeds = max(") < i, "the count must be resolved before it is inherited"


def test_a_pinned_advisory_seed_means_one_fold():
    """A pinned seed names a specific sample; folding it three times would be one
    draw counted three times, and the mean of a value with itself."""
    src = pathlib.Path("src/proteinfoundation/metrics/consensus_folding.py").read_text()
    body = src[src.index("def seeds_for(") : src.index("def first_seed_for(")]
    assert "if pinned is not None:" in body and "return [int(pinned)]" in body
