"""Adopting complex folds made by the old mechanism into the advisory cache.

The two mechanisms cached one kind of result in two shapes: ``binder_eval_cache``
per design, written by the folder named ``folders.complex[0]``, and
``consensus_fold_cache_{backend}`` per (sequence, seed), written by every other
complex folder. Unifying them on the second shape -- which is the one that can
re-read a kept structure instead of refolding -- would otherwise make every
finished campaign's AF2 complexes unreadable: ~616 designs x 3 sequences x 5
models, about 42 GPU-hours per campaign, to reproduce structures already on disk.

What a bridge must get right is narrow and unforgiving. The key is a binder
sequence and a seed VALUE, so a seed derived even slightly differently files the
fold where nothing looks for it and the refold happens anyway. And an adopted
entry must be recognisably INCOMPLETE in exactly the ways that make
``score_binders`` re-read the structure rather than trust history -- the file it
comes from does not record which ipSAE cutoffs produced its numbers.
"""

import ast
import json
import math
import pathlib

from proteinfoundation.evaluation.binder_eval_cache import (
    REDUCED_FROM_MODELS_KEY,
    _resolve_structure,
    adopt_binder_eval_folds,
    consensus_entries_from_complex_stats,
)
from proteinfoundation.metrics.consensus_folding import (
    CONSENSUS_METRIC_SUFFIXES,
    PAE_CUTOFF_KEY,
    consensus_cache_path,
    consensus_derivation_fingerprint,
    consensus_fingerprint,
    draw_ids_for,
    missing_pae_cutoffs,
    read_consensus_cache,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

TARGET = ["MTARGETSEQ"]
SEQ_A = "SGGMIGEGQLYANG"
SEQ_B = "SGGMIGSVQLFADG"

# The keys a real pre-migration cache holds, from a finished EFNB3 design.
REAL_COMPLEX_STATS_KEYS = (
    "avg_ipSAE", "avg_ipSAE_10", "binder_pLDDT", "complex_pdb_path", "i_pAE", "i_pTM",
    "max_ipSAE", "max_ipSAE_10", "min_ipAE", "min_ipSAE", "min_ipSAE_10", "pAE", "pTM",
    "target_pLDDT",
)


def _stats(seq_index: int, pdb: str | None = None, **over) -> dict:
    entry = {k: float(seq_index + i) for i, k in enumerate(REAL_COMPLEX_STATS_KEYS) if k != "complex_pdb_path"}
    if pdb is not None:
        entry["complex_pdb_path"] = pdb
    entry.update(over)
    return entry


def _draws(seq, n_models: int = 1):
    return draw_ids_for("af2", {"n_af2_models": n_models}, TARGET, seq)


_seeds = _draws


# ---------------------------------------------------------------------------
# Finding the structures again
# ---------------------------------------------------------------------------


def test_a_relative_structure_path_is_found_from_the_design_directory(tmp_path):
    """The recorded path resolves only from the directory the run was launched
    in. Every later reader -- this migration included -- is somewhere else, and
    would conclude the structure is gone and refold it."""
    af2 = tmp_path / "AF2"
    af2.mkdir()
    (af2 / "d_mpnn_seq_0_model1.pdb").write_text("ATOM")
    stored = "./evaluation_results/pipeline_X/d/AF2/d_mpnn_seq_0_model1.pdb"
    assert _resolve_structure(stored, str(tmp_path)) == str(af2 / "d_mpnn_seq_0_model1.pdb")


def test_resolution_carries_no_knowledge_of_which_folder_wrote_the_structure(tmp_path):
    """The rule is the stored path's tail under the design directory, so a folder
    whose subdirectory is not called AF2 is found by the same code."""
    sub = tmp_path / "rf3_complex"
    sub.mkdir()
    (sub / "x.pdb").write_text("ATOM")
    assert _resolve_structure("./a/b/c/rf3_complex/x.pdb", str(tmp_path)) == str(sub / "x.pdb")


def test_a_structure_that_is_really_gone_resolves_to_nothing(tmp_path):
    assert _resolve_structure("./a/b/AF2/missing.pdb", str(tmp_path)) is None
    assert _resolve_structure(None, str(tmp_path)) is None
    assert _resolve_structure("", str(tmp_path)) is None


def test_an_absolute_path_that_exists_is_taken_as_it_stands(tmp_path):
    real = tmp_path / "somewhere.pdb"
    real.write_text("ATOM")
    assert _resolve_structure(str(real), str(tmp_path / "elsewhere")) == str(real)


# ---------------------------------------------------------------------------
# What an adopted entry holds
# ---------------------------------------------------------------------------


def test_entries_are_keyed_by_the_sequence_the_stats_describe(tmp_path):
    entries = consensus_entries_from_complex_stats(
        [_stats(0), _stats(1)], [SEQ_A, SEQ_B], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    assert sorted(entries) == sorted([SEQ_A, SEQ_B])
    assert entries[SEQ_A] != entries[SEQ_B], "two sequences must not share one fold's numbers"


def test_a_draw_whose_structure_is_missing_is_not_pointed_at_another_models(tmp_path):
    """Five draws are only adoptable where five structures are. A draw with no
    structure of its own must be left to fold rather than handed model 1's --
    which is the collapse this axis exists to end, rebuilt one level down."""
    entries = consensus_entries_from_complex_stats(
        [_stats(0, pdb="./d/AF2/x_model1.pdb")], [SEQ_A], lambda s: _draws(s, 5),
        str(tmp_path), CONSENSUS_METRIC_SUFFIXES,
    )
    assert list(entries[SEQ_A]) == ["model1"], "no structures on disk, so only the named one"


def test_every_model_with_a_structure_becomes_its_own_draw(tmp_path):
    """The recorded path names model 1; the rest sit beside it under the harness's
    own naming. Each gets its own entry pointing at its own file."""
    af2 = tmp_path / "AF2"
    af2.mkdir()
    for k in range(1, 6):
        (af2 / f"x_model{k}.pdb").write_text("ATOM")
    entries = consensus_entries_from_complex_stats(
        [_stats(0, pdb="./d/AF2/x_model1.pdb")], [SEQ_A], lambda s: _draws(s, 5),
        str(tmp_path), CONSENSUS_METRIC_SUFFIXES,
    )
    assert list(entries[SEQ_A]) == [f"model{k}" for k in range(1, 6)]
    paths = {d: m["pdb_path"] for d, m in entries[SEQ_A].items()}
    assert len(set(paths.values())) == 5, "five draws must not share one structure"
    assert paths["model3"] == str(af2 / "x_model3.pdb")


def test_the_folders_mean_is_carried_but_labelled_as_a_mean(tmp_path):
    """pLDDT, pTM and i_pTM were averaged before anything saw the parts and no
    re-read reproduces them, so the mean rides on each draw rather than leaving a
    gating column NaN across every finished campaign -- and says that it is one.
    Everything the stored matrices CAN answer per model is dropped instead, so it
    gets recomputed from that model's own matrix."""
    af2 = tmp_path / "AF2"
    af2.mkdir()
    for k in range(1, 6):
        (af2 / f"x_model{k}.pdb").write_text("ATOM")
    entries = consensus_entries_from_complex_stats(
        [_stats(0, pdb="./d/AF2/x_model1.pdb")], [SEQ_A], lambda s: _draws(s, 5),
        str(tmp_path), CONSENSUS_METRIC_SUFFIXES,
    )
    one = entries[SEQ_A]["model2"]
    assert one[REDUCED_FROM_MODELS_KEY] == ["binder_pLDDT", "i_pTM", "pTM", "target_pLDDT"]
    assert not [k for k in one if "ipSAE" in k], "per-model ipSAE is recoverable, so the mean must not stand in"
    assert missing_pae_cutoffs(one), "and the recomputation must be asked for"


def test_a_single_model_campaign_keeps_its_numbers_unlabelled(tmp_path):
    """With one model there is no reduction to disown: the entry's numbers ARE
    that draw's, cutoffs included."""
    entries = consensus_entries_from_complex_stats(
        [_stats(0)], [SEQ_A], lambda s: _draws(s, 1), str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    one = entries[SEQ_A]["model1"]
    assert REDUCED_FROM_MODELS_KEY not in one
    assert not missing_pae_cutoffs(one)


def test_the_adopted_draw_is_the_one_score_binders_would_look_under(tmp_path):
    """The whole point of adoption. A key derived even slightly differently files
    the fold where nothing looks for it, and the refold happens anyway."""
    entries = consensus_entries_from_complex_stats(
        [_stats(0)], [SEQ_A], _draws, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    assert list(entries[SEQ_A]) == draw_ids_for("af2", {}, TARGET, SEQ_A)


def test_a_failed_fold_is_not_adopted(tmp_path):
    """inf and NaN are the absence of a measurement. Adopting one would retire
    the refold that would replace it -- the EFNB3 case, where two designs came
    back as inf and only a rerun could fill them."""
    dead = {k: float("inf") for k in REAL_COMPLEX_STATS_KEYS if k != "complex_pdb_path"}
    dead["pTM"] = float("nan")
    assert consensus_entries_from_complex_stats(
        [dead], [SEQ_A], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    ) == {}


def test_nothing_read_off_a_structure_is_adopted(tmp_path):
    """Only what the folder reported. Buried area, shape complementarity and
    geometry against the design are re-read from the kept PDB by the same code
    that reads every other folder's structures, rather than mapped across from a
    second set of names that could disagree."""
    entries = consensus_entries_from_complex_stats(
        [_stats(0, binder_dSASA=900.0, scRMSD_ca=1.2, interface_sc=0.7)],
        [SEQ_A], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES,
    )
    held = set(next(iter(entries[SEQ_A].values())))
    assert not held & {"binder_dSASA", "scRMSD_ca", "interface_sc"}
    assert held <= set(CONSENSUS_METRIC_SUFFIXES) | {"pdb_path", PAE_CUTOFF_KEY}


def test_an_evidenced_ipsae_cutoff_is_recorded_as_the_fact_it_is(tmp_path):
    """The entry holds values produced at these distances, so it says so.

    Not, as an earlier version of this test claimed, because leaving them absent
    triggers a single-model recomputation of a five-model mean. That does happen
    and it does move avg_ipSAE by up to 0.55 -- but it is a defect in the
    derivation, where a question about distance decides ensemble size, and the
    cutoff record only masks it. It is fixed by per-model entries instead.
    """
    entries = consensus_entries_from_complex_stats(
        [_stats(0)], [SEQ_A], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    assert not missing_pae_cutoffs(next(iter(entries[SEQ_A].values()))), (
        "a cutoff the entry holds all three ipSAE kinds for must be recorded, or "
        "the folder's multi-model mean is silently replaced by one model's value"
    )


def test_a_cutoff_the_entry_cannot_evidence_stays_missing(tmp_path):
    """Recorded per cutoff, not per entry. A distance this run asks for that the
    campaign never scored must still be computed from the stored matrix -- which
    is what makes adding one cost a re-read rather than a refold."""
    partial = _stats(0)
    for kind in ("min_", "max_", "avg_"):
        partial.pop(f"{kind}ipSAE_10", None)
    entries = consensus_entries_from_complex_stats(
        [partial], [SEQ_A], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    missing = missing_pae_cutoffs(next(iter(entries[SEQ_A].values())))
    assert [s for _, s in missing] == ["_10"], missing


def test_a_resolvable_structure_rides_along_so_the_derived_metrics_can_heal(tmp_path):
    af2 = tmp_path / "AF2"
    af2.mkdir()
    (af2 / "m.pdb").write_text("ATOM")
    entries = consensus_entries_from_complex_stats(
        [_stats(0, pdb="./x/y/AF2/m.pdb")], [SEQ_A], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    assert next(iter(entries[SEQ_A].values()))["pdb_path"] == str(af2 / "m.pdb")


def test_more_stats_than_sequences_pairs_none_of_the_extras(tmp_path):
    """Pairing by position is only safe as far as the shorter list. A stats entry
    with no sequence to name it has no cache key at all."""
    entries = consensus_entries_from_complex_stats(
        [_stats(0), _stats(1)], [SEQ_A], _seeds, str(tmp_path), CONSENSUS_METRIC_SUFFIXES
    )
    assert list(entries) == [SEQ_A]


# ---------------------------------------------------------------------------
# The migration as score_binders will meet it
# ---------------------------------------------------------------------------


STATS_BY_TYPE = {
    "mpnn": {"complex_stats": [_stats(0), _stats(1)]},
    "self": {"complex_stats": [_stats(2)]},
}
SEQS_BY_TYPE = {"mpnn": [SEQ_A, SEQ_B], "self": ["MSELFSEQ"]}


def test_an_adopted_cache_reads_back_under_the_fingerprint_score_binders_computes(tmp_path):
    """The round trip that decides whether a campaign refolds. Both fingerprints
    have to be the ones score_binders will ask with, or the file is discarded on
    sight and the GPU hours are spent anyway."""
    n = adopt_binder_eval_folds(str(tmp_path), "af2", TARGET, {}, STATS_BY_TYPE, SEQS_BY_TYPE)
    assert n == 3

    scores, stale = read_consensus_cache(
        str(tmp_path), "af2", consensus_fingerprint("af2", {}, TARGET),
        derivation=consensus_derivation_fingerprint(False),
    )
    assert sorted(scores) == sorted([SEQ_A, SEQ_B, "MSELFSEQ"])
    assert not stale, "an adopted entry is not stale -- it is incomplete, which heals per key"
    assert list(scores[SEQ_A]) == draw_ids_for("af2", {}, TARGET, SEQ_A)
    assert math.isfinite(scores[SEQ_A]["model1"]["i_pTM"])


def test_adoption_never_overwrites_a_cache_the_folder_itself_wrote(tmp_path):
    """An existing advisory cache is authoritative: it was written by the folder,
    while this only ever reconstructs."""
    path = consensus_cache_path(str(tmp_path), "af2")
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps({"fingerprint": "real", "schema": 2, "scores": {}}))
    assert adopt_binder_eval_folds(str(tmp_path), "af2", TARGET, {}, STATS_BY_TYPE, SEQS_BY_TYPE) == 0
    assert json.loads(path_obj.read_text())["fingerprint"] == "real"


def test_adoption_is_idempotent(tmp_path):
    """It runs on every design of every run rather than as a one-off script
    somebody has to remember to point at each campaign, so the second call must
    cost nothing and change nothing."""
    assert adopt_binder_eval_folds(str(tmp_path), "af2", TARGET, {}, STATS_BY_TYPE, SEQS_BY_TYPE) == 3
    before = pathlib.Path(consensus_cache_path(str(tmp_path), "af2")).read_text()
    assert adopt_binder_eval_folds(str(tmp_path), "af2", TARGET, {}, STATS_BY_TYPE, SEQS_BY_TYPE) == 0
    assert pathlib.Path(consensus_cache_path(str(tmp_path), "af2")).read_text() == before


def test_nothing_adoptable_writes_no_cache(tmp_path):
    """A design whose folds all failed must leave no file behind. One would read
    as a cache with no entry for the sequence, which is the same as none -- but
    it would also stop a later run from adopting the folds that replace them."""
    assert adopt_binder_eval_folds(str(tmp_path), "af2", TARGET, {}, {}, {}) == 0
    assert not pathlib.Path(consensus_cache_path(str(tmp_path), "af2")).exists()


def test_a_broken_source_costs_a_refold_and_not_the_run(tmp_path):
    """Failing to adopt is expensive; failing the evaluation is worse."""
    assert adopt_binder_eval_folds(
        str(tmp_path), "af2", TARGET, {}, {"mpnn": {"complex_stats": "not a list"}}, SEQS_BY_TYPE
    ) == 0


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_the_seed_rule_has_exactly_one_definition():
    """A cache is keyed by seed VALUE. Two copies of the rule is two ways for one
    fold to be filed under two keys, which reads as a miss and refolds -- so
    score_binders must call the shared helper rather than derive its own.
    """
    tree = ast.parse((SRC / "proteinfoundation/metrics/consensus_folding.py").read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    body = ast.dump(fns["score_binders"])
    assert "deterministic_seeds" not in body and "deterministic_seed" not in body, (
        "score_binders derives its own draw keys again; the bridge would key differently"
    )
    assert "draw_ids_for" in body


def test_the_bridge_is_actually_called_by_the_pipeline():
    """A migration nothing invokes is a migration that did not happen.

    This one was written, tested, and left unwired: every finished campaign would
    have refolded its AF2 complexes the first time the unified path ran -- 657
    designs x 3 sequences x 5 models on EFNB3, about 42 GPU-hours, to reproduce
    predictions already on disk. The tests all passed, because they called it
    themselves.
    """
    source = (SRC / "proteinfoundation/evaluation/binder_eval.py").read_text()
    assert "adopt_binder_eval_folds(" in source, "the bridge must be reached from the per-design loop"
    # Before the folds it exists to make unnecessary.
    assert source.index("adopt_binder_eval_folds(") < source.index("for backend_name in complex_folders:")
