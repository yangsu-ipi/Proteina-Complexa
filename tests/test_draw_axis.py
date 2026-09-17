"""One prediction, one entry — however a folder repeats itself.

A draw is one prediction of one complex. ESMFold2's draws differ by the seed it
sampled at; AF2's differ by which parameter set produced them. Those are the same
kind of thing, and the only reason they were treated differently is that AF2's
were averaged inside the folding harness before anything else could see them.

That collapse was not cosmetic. Only the first model's structure was named, so
anything reading a metric back off "the" structure read one model while the
numbers beside it were five models' mean -- measured at up to 0.55 on avg_ipSAE
over 120 EFNB3 complexes, on a 0-to-1 metric that gets thresholded. Worse, the
thing that decided whether that re-read happened was ``missing_pae_cutoffs``, a
question about DISTANCE. A question about distance was deciding ensemble size.

With one entry per draw, each carrying its own structure, the question cannot
arise: what is read off a structure is read off the model it belongs to.
"""

import json
import pathlib

import pytest

from proteinfoundation.metrics import consensus_folding as cf
from proteinfoundation.metrics.consensus_folding import (
    CONSENSUS_CACHE_SCHEMA,
    CONSENSUS_PLACEMENT_SUFFIXES,
    advisory_structure_path,
    consensus_cache_path,
    draw_ids_for,
    read_consensus_cache,
    reduce_over_draws,
    seed_of_draw,
    write_consensus_cache,
)

TARGET = ["MTARGET"]
SEQ = "AAAA"


# ---------------------------------------------------------------------------
# What a draw is
# ---------------------------------------------------------------------------


def test_a_sampler_draws_by_seed_and_af2_draws_by_parameter_set():
    """The axis is per backend because the repetition is. Both are draws."""
    assert draw_ids_for("esmfold2", {"n_seeds": 3}, TARGET, SEQ) == [
        f"seed{s}" for s in cf.fold_seeds_for("esmfold2", {"n_seeds": 3}, TARGET, SEQ)
    ]
    assert draw_ids_for("af2", {"n_af2_models": 5}, TARGET, SEQ) == [f"model{k}" for k in range(1, 6)]


def test_af2_draw_count_follows_the_campaigns_model_count_not_its_seed_count():
    """n_seeds is a sampler's knob. Reading it for AF2 would fold one model where
    the campaign folded five, and put a one-model ensemble in the same frame."""
    assert len(draw_ids_for("af2", {"n_seeds": 3, "n_af2_models": 5}, TARGET, SEQ)) == 5


def test_a_seed_and_a_model_of_the_same_number_are_different_draws():
    """Why the key is a string. Schema 2 keyed by bare integer, which cannot tell
    seed 3 from model 3 -- and a cache that confuses them serves one folder's
    prediction for another's."""
    assert seed_of_draw("seed3") == 3
    assert "seed3" != "model3"


def test_a_bare_seed_still_decodes_for_callers_that_speak_in_seeds():
    assert seed_of_draw(7) == 7
    assert seed_of_draw("seed7") == 7


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_schema_2_entries_are_adopted_onto_the_draw_axis(tmp_path):
    """A finished campaign's ESMFold2 folds keep answering. The seed axis of the
    draw key IS what schema 2 stored, so the migration is a rename."""
    path = pathlib.Path(consensus_cache_path(str(tmp_path), "esmfold2"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fingerprint": "fp", "schema": 2, "derivation": "d",
        "scores": {SEQ: {"11": {"pLDDT": 0.8}, "22": {"pLDDT": 0.9}}},
    }))
    got, stale = read_consensus_cache(str(tmp_path), "esmfold2", "fp", derivation="d")
    assert got == {SEQ: {"seed11": {"pLDDT": 0.8}, "seed22": {"pLDDT": 0.9}}}
    assert stale is False


def test_the_migration_moves_no_structure_on_disk(tmp_path):
    """The point of spelling the seed axis ``seed{n}``: the names schema 2 already
    wrote are the names the draw axis wants. A rename that moved 22k PDBs would
    be a refold wearing a migration's clothes."""
    old_style = tmp_path / "esmfold2_complex"
    old_style.mkdir(parents=True)
    written = pathlib.Path(advisory_structure_path(str(tmp_path), "esmfold2", SEQ, "seed11"))
    written.write_text("ATOM")
    assert written.name.endswith("_seed11.pdb"), written.name
    assert cf.existing_advisory_structure(str(tmp_path), "esmfold2", SEQ, "seed11") == str(written)


def test_the_schema_is_stamped_so_a_later_reader_knows_what_it_holds(tmp_path):
    write_consensus_cache(str(tmp_path), "esmfold2", "fp", {SEQ: {"seed1": {"pLDDT": 0.5}}})
    blob = json.loads(pathlib.Path(consensus_cache_path(str(tmp_path), "esmfold2")).read_text())
    assert blob["schema"] == CONSENSUS_CACHE_SCHEMA == 3


# ---------------------------------------------------------------------------
# One call, several draws
# ---------------------------------------------------------------------------


def _multi_draw_backend(calls, n=5):
    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, **_):
        calls.append(draw)
        return {"draws": {f"model{k}": {"i_pTM": 0.1 * k, "pTM": 0.5} for k in range(1, n + 1)}}

    return scorer


def test_a_folder_that_answers_for_every_draw_is_called_once(tmp_path, monkeypatch):
    """AF2 predicts with every parameter set per invocation. Asking it once per
    draw would do five times the work to keep four answers it already had."""
    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "af2", _multi_draw_backend(calls))
    out = cf.score_binders(
        "af2", TARGET, [SEQ], cfg={"n_af2_models": 5}, cache_dir=str(tmp_path), keep_structures=False
    )
    assert len(calls) == 1, f"one call should answer all five draws, got {calls}"
    assert out[0]["n_predictions"] == 5.0


def test_every_draw_it_answered_for_is_cached_separately(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "af2", _multi_draw_backend(calls))
    cf.score_binders(
        "af2", TARGET, [SEQ], cfg={"n_af2_models": 5}, cache_dir=str(tmp_path), keep_structures=False
    )
    blob = json.loads(pathlib.Path(consensus_cache_path(str(tmp_path), "af2")).read_text())
    assert sorted(blob["scores"][SEQ]) == [f"model{k}" for k in range(1, 6)]
    assert blob["scores"][SEQ]["model3"]["i_pTM"] == pytest.approx(0.3)


def test_a_second_run_folds_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "af2", _multi_draw_backend(calls))
    args = dict(cfg={"n_af2_models": 5}, cache_dir=str(tmp_path), keep_structures=False)  # noqa: C408
    cf.score_binders("af2", TARGET, [SEQ], **args)
    cf.score_binders("af2", TARGET, [SEQ], **args)
    assert len(calls) == 1, "the cache must cover every draw the one call answered"


def test_a_single_draw_backend_still_answers_flatly(tmp_path, monkeypatch):
    """ESMFold2 folds one seed per call and returns a plain dict. The unpacking
    must not require every backend to speak in draws."""
    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, **_):
        return {"i_pTM": 0.4}

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", scorer)
    out = cf.score_binders(
        "esmfold2", TARGET, [SEQ], cfg={"n_seeds": 3}, cache_dir=str(tmp_path), keep_structures=False
    )
    assert out[0]["n_predictions"] == 3.0


def test_each_draw_gets_its_own_structure_path(tmp_path, monkeypatch):
    """Pointing several draws at one file is the collapse, rebuilt. A multi-draw
    folder is handed a path per draw so all of its structures land."""
    seen = {}

    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, **_):
        for k in range(1, 4):
            seen[f"model{k}"] = out_path_for(f"model{k}")
        return {"draws": {f"model{k}": {"i_pTM": 0.1} for k in range(1, 4)}}

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "af2", scorer)
    cf.score_binders(
        "af2", TARGET, [SEQ], cfg={"n_af2_models": 3}, cache_dir=str(tmp_path), keep_structures=True
    )
    assert len(set(seen.values())) == 3, seen
    assert all(p and p.endswith(f"_{d}.pdb") for d, p in seen.items()), seen


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def test_everything_numeric_pools_by_mean_including_placement():
    """The rule reduce_rmsd_over_models gives, applied to every draw of every
    folder. Placement used to take the worst draw here; this side is where that
    change is felt, because ESMFold2's seeds disagree about where the binder went
    far more than AF2's models do.
    """
    assert "scRMSD_ca" in CONSENSUS_PLACEMENT_SUFFIXES
    got = reduce_over_draws({
        "model1": {"i_pTM": 0.9, "scRMSD_ca": 0.5},
        "model2": {"i_pTM": 0.5, "scRMSD_ca": 8.0},
    })
    assert got["i_pTM"] == pytest.approx(0.7)
    assert got["scRMSD_ca"] == pytest.approx(4.25), "placement takes the mean too"


def test_the_placement_set_is_mapped_from_the_primary_sides_not_retyped():
    """One definition of 'this metric is about placement'. The primary side calls
    it complex_scRMSD_ca and the advisory slot calls it scRMSD_ca, so a second
    hardcoded list would agree only by inspection and drift silently. The set no
    longer selects a reduction; it still names the family."""
    from proteinfoundation.metrics.ensembling import PLACEMENT_METRICS

    expected = {s for k, s in cf.CONSENSUS_RMSD_SUFFIXES.items() if k in PLACEMENT_METRICS}
    assert set(CONSENSUS_PLACEMENT_SUFFIXES) == expected and expected


def test_fold_quality_is_not_placement():
    """binder_scRMSD_ca asks how well the binder folded, not where it landed.
    Both reduce by mean now, so this pins the value rather than the distinction."""
    got = reduce_over_draws({"a": {"binder_scRMSD_ca": 1.0}, "b": {"binder_scRMSD_ca": 3.0}})
    assert got["binder_scRMSD_ca"] == pytest.approx(2.0)


def test_reduction_is_over_repeat_predictions_of_one_sequence_only():
    """Choosing AMONG sequences is analyze's and stays there. This reduces the
    draws of one binder and nothing else."""
    assert reduce_over_draws({})== {}
    one = reduce_over_draws({"model1": {"i_pTM": 0.4}})
    assert one["i_pTM"] == pytest.approx(0.4) and one["n_predictions"] == 1.0


def test_provenance_lists_survive_the_reduction_instead_of_being_averaged():
    """Packed eight-state counts are a vector of numbers and average elementwise.
    A list of metric NAMES -- which keys an adopted entry carries as the folder's
    own reduction rather than this draw's -- is provenance, identical on every
    draw, and summing it raised on the first real campaign this met."""
    got = reduce_over_draws({
        "model1": {"binder_ss_counts": [10.0, 0.0], "reduced_over_models": ["i_pTM", "pTM"]},
        "model2": {"binder_ss_counts": [0.0, 10.0], "reduced_over_models": ["i_pTM", "pTM"]},
    })
    assert got["binder_ss_counts"] == [5.0, 5.0]
    assert got["reduced_over_models"] == ["i_pTM", "pTM"]


# ---------------------------------------------------------------------------
# Every complex folder is reachable the same way
# ---------------------------------------------------------------------------


def test_every_folder_declared_complex_capable_is_registered_as_a_backend():
    """The invariant behind "no folder is special". _FOLDER_CAPABILITIES is what
    a config is validated against, and CONSENSUS_BACKENDS is what can actually be
    folded through the shared path -- so a folder in the first and not the second
    is one a campaign may name and only the gated mechanism can reach. RF3 was
    exactly that: declaring it complex-capable let a config ask for it, while
    being absent here meant it could only ever be folders.complex[0], so a
    campaign naming it could have no second folder beside it and one naming it
    second could not use it at all.
    """
    from proteinfoundation.metrics.column_names import _FOLDER_CAPABILITIES

    declared = {name for name, tracks in _FOLDER_CAPABILITIES.items() if "complex" in tracks}
    assert declared <= set(cf.CONSENSUS_BACKENDS), (
        f"{sorted(declared - set(cf.CONSENSUS_BACKENDS))} can fold a complex by _FOLDER_CAPABILITIES "
        f"but cannot be reached through score_binders"
    )


def test_rf3_refuses_a_context_with_no_runner_rather_than_folding_nothing():
    """It is an object holding weights, not a function of its inputs. A caller
    that has only sequences has to be told so, not handed empty metrics that look
    like a fold which found nothing."""
    with pytest.raises(ValueError, match="runner"):
        cf.CONSENSUS_BACKENDS["rf3"](["MTGT"], "AAAA", {}, None, "seed1", None)


def test_rf3_draws_once():
    """Its ensemble is not a sampler's. One prediction per binder, like AF2 per
    parameter set and unlike ESMFold2 per seed."""
    assert draw_ids_for("rf3", {"n_seeds": 3}, TARGET, SEQ) == draw_ids_for("rf3", {}, TARGET, SEQ)
    assert len(draw_ids_for("rf3", {"n_seeds": 3}, TARGET, SEQ)) == 1


def test_a_harness_envelope_is_unwrapped_for_every_backend_that_uses_one():
    """Both complex harnesses return {"seq_N": stats}. Reading the envelope as
    the statistics is what made _score_af2 return a path and no numbers."""
    assert cf._unwrap_sequence_envelope({"seq_1": {"i_pTM": 0.8}}) == {"i_pTM": 0.8}
    assert cf._unwrap_sequence_envelope({"i_pTM": 0.8}) == {"i_pTM": 0.8}, "a bare dict is already the stats"
    assert cf._unwrap_sequence_envelope(None) == {}


def test_constructing_a_folder_refuses_nothing_the_backends_can_fold():
    """There is one registry of what can fold a complex, and it is not this one.

    initialize_folding_model used to raise for any folder it could not build,
    which made it a second registry free to disagree with CONSENSUS_BACKENDS --
    and it did. A pass naming esmfold2 as its only complex folder died there, on
    a folder the campaign had configured and this registry folds perfectly well.
    It only builds things now: RF3's runner, and nothing else.
    """
    from proteinfoundation.evaluation.binder_eval import initialize_folding_model

    for backend in cf.CONSENSUS_BACKENDS:
        if backend == "rf3":
            continue  # constructing it loads weights; its own test covers the contract
        specs = initialize_folding_model(backend, ["A"], "TARGET", is_target_ligand=False)
        assert isinstance(specs, dict) and specs.get("runner") is None, (
            f"{backend} folds from sequences and a context; nothing to construct"
        )


def test_one_backends_knob_cannot_invalidate_anothers_cache():
    """n_af2_models is AF2's draw count. It lives in the shared consensus_cfg so
    AF2 folds with the campaign's parameter sets, and the whole cfg is hashed
    into every backend's fold fingerprint -- so adding it changed ESMFOLD2's
    fingerprint, discarded every cached ESMFold2 complex on EFNB3, and the refold
    came back NaN on an SVD that does not always converge.

    A count says how MANY predictions to make, not what any one of them is, and
    the cache is keyed per draw and merges: asking for more must fold only what
    is new.
    """
    target = ["MKVTARGET"]
    base = cf.consensus_fingerprint("esmfold2", {"n_seeds": 3}, target)
    with_af2 = cf.consensus_fingerprint("esmfold2", {"n_seeds": 3, "n_af2_models": 5}, target)
    assert base == with_af2, "an AF2 knob must be invisible to ESMFold2's cache"

    one = cf.consensus_fingerprint("af2", {"n_af2_models": 1}, target)
    five = cf.consensus_fingerprint("af2", {"n_af2_models": 5}, target)
    assert one == five, "raising the draw count folds what is new, it does not discard what is held"


def test_a_cache_written_when_the_count_was_hashed_is_adopted_not_discarded(tmp_path):
    """Removing a key from a hash changes every fingerprint, which would discard
    the very caches taking it out is meant to keep.

    So the old value is reconstructed: a cache holding N draws was written by a
    run asking for N, and putting N back reproduces exactly what that run stored.
    EFNB3's retry is the case -- three seeds of good ESMFold2 folds on disk, a
    campaign now asking for one, and a fingerprint that called it a different
    scorer.
    """
    cfg, target = {"n_seeds": 3}, ["MKVTARGET"]
    old_fp = cf.legacy_count_fingerprint("esmfold2", cfg, target, 3)
    new_fp = cf.consensus_fingerprint("esmfold2", cfg, target)
    assert old_fp != new_fp, "the count used to be in the identity"

    path = pathlib.Path(consensus_cache_path(str(tmp_path), "esmfold2"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fingerprint": old_fp, "schema": 3, "derivation": "d",
        "scores": {SEQ: {f"seed{k}": {"i_pTM": 0.1 * k} for k in (11, 22, 33)}},
    }))

    got, _ = read_consensus_cache(
        str(tmp_path), "esmfold2", new_fp, derivation="d",
        legacy_fingerprint_for=lambda n: cf.legacy_count_fingerprint("esmfold2", cfg, target, n),
    )
    assert sorted(got[SEQ]) == ["seed11", "seed22", "seed33"], "three good folds, kept"


def test_a_genuinely_different_scorer_is_still_refused(tmp_path):
    """The reconstruction must not become a way for any cache to be accepted."""
    cfg, target = {"n_seeds": 3}, ["MKVTARGET"]
    path = pathlib.Path(consensus_cache_path(str(tmp_path), "esmfold2"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "fingerprint": "written-by-something-else", "schema": 3,
        "scores": {SEQ: {"seed1": {"i_pTM": 0.5}}},
    }))
    got, _ = read_consensus_cache(
        str(tmp_path), "esmfold2", cf.consensus_fingerprint("esmfold2", cfg, target),
        legacy_fingerprint_for=lambda n: cf.legacy_count_fingerprint("esmfold2", cfg, target, n),
    )
    assert got == {}


def test_changing_the_seed_count_no_longer_discards_the_folds_already_made():
    """deterministic_seeds is prefix-stable so three seeds then five folds two,
    not five -- which was only ever true if the count stayed out of the identity."""
    target = ["MKVTARGET"]
    assert cf.consensus_fingerprint("esmfold2", {"n_seeds": 3}, target) == cf.consensus_fingerprint(
        "esmfold2", {"n_seeds": 5}, target
    )


def test_the_other_axis_count_does_not_leak_into_a_reconstructed_identity():
    """binder_eval sets n_af2_models on the shared consensus_cfg for EVERY
    backend. The live fingerprint drops all counts so it does not care; the
    legacy reconstruction hashes counts on purpose, so AF2's model count was
    landing in ESMFold2's reconstructed identity. The stored hash then missed and
    EFNB3 refolded all 5913 ESMFold2 complexes it already had on disk.

    What a run stored is what a run WITH ONLY ITS OWN COUNT would have computed.
    """
    from proteinfoundation.metrics.consensus_folding import legacy_count_fingerprint

    target = ["MTARGET"]
    base = {"num_loops": 20, "num_sampling_steps": 200}

    # What the old run hashed: its own axis count and nothing else.
    stored = legacy_count_fingerprint("esmfold2", {**base, "n_seeds": 3}, target, 3)

    for polluted in (
        {**base, "n_af2_models": 5},
        {**base, "n_af2_models": 5, "n_seeds": 3},
        {**base, "n_af2_models": 5, "n_esmfold2_seeds": 3},
    ):
        assert legacy_count_fingerprint("esmfold2", polluted, target, 3) == stored, (
            f"a count belonging to another axis changed the reconstruction: {sorted(polluted)}"
        )

    # Symmetric: ESMFold2's seed count must not reach AF2's reconstruction.
    af2 = legacy_count_fingerprint("af2", {**base, "n_af2_models": 5}, target, 5)
    assert legacy_count_fingerprint("af2", {**base, "n_af2_models": 5, "n_seeds": 3}, target, 5) == af2

    # And the two backends still key differently from one another.
    assert stored != af2
