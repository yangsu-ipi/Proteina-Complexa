"""The advisory complex's chain IDs, and what happens when its structure
cannot be written.

The bug these pin: target chains were labelled T0/T1, PDB gives the chain ID one
column, so every advisory structure write failed. A missing structure is what
asks for a refold, so the folds recurred on every run -- roughly nine complexes
per design, indefinitely, producing nothing. The write failure was a warning.
"""

import pytest

from proteinfoundation.metrics.consensus_folding import (
    AdvisoryStructureWriteError,
    advisory_chain_ids,
    score_binders,
)


def test_every_chain_id_fits_the_column_pdb_gives_it():
    """The whole cause: T0 is two characters and a PDB chain column is one."""
    for n in (1, 2, 5, 25):
        ids = advisory_chain_ids(n)
        assert all(len(i) == 1 for i in ids), f"{n} target chains produced a multi-character id"


def test_the_binder_is_last_and_the_targets_keep_their_order():
    """Only the order carries meaning -- the metrics index the complex by
    target_len, not by name -- so target-first is the property to hold."""
    assert advisory_chain_ids(1) == ["A", "B"]
    assert advisory_chain_ids(3) == ["A", "B", "C", "D"]
    for n in (1, 4):
        assert len(advisory_chain_ids(n)) == n + 1, "one id per target, plus the binder"


def test_a_single_chain_target_matches_the_generated_complex_convention():
    assert advisory_chain_ids(1) == ["A", "B"]


def test_more_chains_than_the_alphabet_is_refused_rather_than_wrapped():
    """Wrapping would give two chains the same id, which is worse than failing:
    a complex that cannot be written as PDB should say so."""
    with pytest.raises(ValueError, match="single-character"):
        advisory_chain_ids(26)
    with pytest.raises(ValueError):
        advisory_chain_ids(0)


def test_a_failed_structure_write_stops_the_run(tmp_path, monkeypatch):
    """keep_folding_outputs asked for the file. Degrading that to 'metrics only'
    is the pipeline overriding the request -- and since a missing structure is
    what triggers the refold, it means folding again on every future run."""
    import proteinfoundation.metrics.consensus_folding as cf

    def exploding_scorer(target_seqs, binder_seq, cfg, out_pdb_path, seed):
        raise AdvisoryStructureWriteError("cannot write")

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", exploding_scorer)
    with pytest.raises(AdvisoryStructureWriteError):
        score_binders(
            "esmfold2",
            ["MKV"],
            ["AAAA"],
            cfg={"n_seeds": 1},
            cache_dir=str(tmp_path),
            keep_structures=True,
        )


def test_a_fold_that_merely_fails_is_still_survivable(tmp_path, monkeypatch):
    """Per-design failures stay tolerated: one bad binder must not end a run of
    340. Only the systematic failure is fatal."""
    import proteinfoundation.metrics.consensus_folding as cf

    def flaky_scorer(target_seqs, binder_seq, cfg, out_pdb_path, seed):
        raise RuntimeError("this one binder exploded")

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", flaky_scorer)
    out = score_binders(
        "esmfold2", ["MKV"], ["AAAA"], cfg={"n_seeds": 1}, cache_dir=str(tmp_path), keep_structures=True
    )
    assert out == [{}], "no metrics, but the run continues"


def test_the_scorer_uses_the_derived_ids():
    """Structural: the ids have to reach ProteinInput, or the helper is decoration."""
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src/proteinfoundation/metrics/consensus_folding.py"
    ).read_text()
    body = source[source.index("def _score_esmfold2(") : source.index("def _esmfold2_metrics(")]
    assert "advisory_chain_ids(len(target_seqs))" in body
    assert 'f"T{i}"' not in body, "the label that could not be written"
