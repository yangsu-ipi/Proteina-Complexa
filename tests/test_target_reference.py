"""The target folded on its own, once per campaign.

This value is a denominator. Every property tested here is about what happens
when it is wrong, stale, or missing -- because a ratio against a bad reference
still looks like a number, and a gate reading it cannot tell.
"""

import json
import math
import pathlib
from dataclasses import dataclass

import pytest

from proteinfoundation.evaluation.target_reference import (
    TARGET_REFERENCE_DIR,
    read_target_reference,
    target_alone_plddt,
    target_reference_cache_path,
    target_reference_fingerprint,
    write_target_reference,
)
from proteinfoundation.metrics.ensembling import mean_plddt_from_pdb, residue_weighted_mean

TARGET = ["MKV", "AAAAAA"]


@dataclass
class Fold:
    pdb_path: str
    sequence: str


def pdb_with_plddt(path: pathlib.Path, values: list[float]) -> str:
    """A minimal CA-only PDB with pLDDT in the B-factor column, as the folding
    backends write it."""
    lines = []
    for i, value in enumerate(values, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    {0.0:>8.3f}{0.0:>8.3f}{0.0:>8.3f}{1.0:>6.2f}{value:>6.2f}           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")
    return str(path)


def make_fold_fn(tmp_path, per_seed: dict | None = None, fail_on=None):
    """A stand-in for fold_sequences that writes real PDBs with known pLDDT."""
    calls = []

    def fold_fn(sequences, output_dir, name, folding_models, suffix, cache_dir, keep_outputs, seed):
        model = folding_models[0]
        calls.append((model, seed))
        if fail_on is not None and model == fail_on:
            raise RuntimeError("folding backend exploded")
        value = (per_seed or {}).get(seed, 0.90)
        results = []
        for i, seq in enumerate(sequences):
            path = tmp_path / f"{model}_{seed}_{i}.pdb"
            pdb_with_plddt(path, [value] * len(seq))
            results.append(Fold(pdb_path=str(path), sequence=seq))
        return {model: results}

    fold_fn.calls = calls
    return fold_fn


def test_plddt_is_read_out_of_the_b_factor_column(tmp_path):
    path = pdb_with_plddt(tmp_path / "f.pdb", [0.9, 0.8, 0.7])
    assert mean_plddt_from_pdb(path) == pytest.approx(0.8)


def test_a_0_to_100_structure_is_read_as_fractions(tmp_path):
    """ColabFold writes pLDDT on 0-100. A gate comparing to 0.9 would pass every
    design ever folded."""
    path = pdb_with_plddt(tmp_path / "f.pdb", [90.0, 80.0, 70.0])
    assert mean_plddt_from_pdb(path) == pytest.approx(0.8)


def test_an_unreadable_structure_is_nan_not_zero(tmp_path):
    """Zero would make every ratio against it infinite; a missing file must stay
    missing."""
    assert math.isnan(mean_plddt_from_pdb(str(tmp_path / "absent.pdb")))
    assert math.isnan(mean_plddt_from_pdb(pdb_with_plddt(tmp_path / "empty.pdb", [])))


def test_chains_are_weighted_by_their_length():
    """The target's pLDDT in complex is one mean over all its residues, so the
    reference has to be too -- otherwise the ratio measures the chain split."""
    assert residue_weighted_mean([1.0, 0.0], [3, 1]) == pytest.approx(0.75)
    assert residue_weighted_mean([1.0, 0.0], [1, 3]) == pytest.approx(0.25)


def test_the_reference_is_computed_and_cached(tmp_path):
    fold_fn = make_fold_fn(tmp_path)
    got = target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=2, fold_fn=fold_fn)
    assert got["esmfold2"] == pytest.approx(0.90)
    assert pathlib.Path(target_reference_cache_path(str(tmp_path))).exists()


def test_a_second_shard_reuses_the_first_shards_fold(tmp_path):
    """Two shards of one campaign share this directory. The second must not
    fold the target again and land on a different number."""
    fold_fn = make_fold_fn(tmp_path)
    first = target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=3, fold_fn=fold_fn)
    folds_after_first = len(fold_fn.calls)
    second = target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=3, fold_fn=fold_fn)
    assert second == first
    assert len(fold_fn.calls) == folds_after_first, "no refolding on the second call"


def test_esmfold2_pools_over_the_seeds_it_was_asked_for(tmp_path):
    """A reference drawn once would carry the sampler's own spread into every
    ratio measured against it."""
    fold_fn = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=3, fold_fn=fold_fn)
    seeds = [seed for model, seed in fold_fn.calls if model == "esmfold2"]
    assert len(seeds) == 3
    assert len(set(seeds)) == 3, "three distinct seeds, not one seed three times"


def test_the_seeds_are_prefix_stable(tmp_path):
    """Raising the seed count must extend the sequence, not replace it -- the
    same promise the design-side folds make."""
    three = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path / "a"), ["esmfold2"], n_esmfold2_seeds=3, fold_fn=three)
    five = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path / "b"), ["esmfold2"], n_esmfold2_seeds=5, fold_fn=five)
    assert [s for _, s in five.calls][:3] == [s for _, s in three.calls]


def test_a_deterministic_model_is_folded_once(tmp_path):
    """Only esmfold2 samples. Seeding esmfold would average a structure with
    itself at full cost."""
    fold_fn = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path), ["esmfold"], n_esmfold2_seeds=3, fold_fn=fold_fn)
    assert fold_fn.calls == [("esmfold", None)]


def test_seeds_are_pooled_by_mean(tmp_path):
    seeds_seen = []

    def recording(**kwargs):
        seeds_seen.append(kwargs["seed"])
        return make_fold_fn(tmp_path, per_seed={kwargs["seed"]: 0.6 if len(seeds_seen) == 1 else 1.0})(**kwargs)

    got = target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=2, fold_fn=recording)
    assert got["esmfold2"] == pytest.approx(0.8)


def test_a_changed_target_invalidates_the_cache(tmp_path):
    """The whole point of the fingerprint: a different target is a different
    reference, and serving the old one would rescore every design against a
    molecule it was never measured against."""
    fold_fn = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=1, fold_fn=fold_fn)
    before = len(fold_fn.calls)
    target_alone_plddt(["WWWW"], str(tmp_path), ["esmfold2"], n_esmfold2_seeds=1, fold_fn=fold_fn)
    assert len(fold_fn.calls) > before


def test_more_seeds_invalidates_the_cache(tmp_path):
    """A three-seed reference is not a five-seed reference."""
    fold_fn = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=1, fold_fn=fold_fn)
    before = len(fold_fn.calls)
    target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], n_esmfold2_seeds=2, fold_fn=fold_fn)
    assert len(fold_fn.calls) > before


def test_a_failing_backend_leaves_its_key_out(tmp_path):
    """Absent, not NaN: a ratio computed against NaN propagates silently, and a
    gate reading it cannot tell a broken reference from a bad design."""
    fold_fn = make_fold_fn(tmp_path, fail_on="esmfold2")
    assert target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], fold_fn=fold_fn) == {}
    assert not pathlib.Path(target_reference_cache_path(str(tmp_path))).exists()


def test_one_failing_backend_does_not_take_the_others_down(tmp_path):
    fold_fn = make_fold_fn(tmp_path, fail_on="esmfold")
    got = target_alone_plddt(TARGET, str(tmp_path), ["esmfold", "esmfold2"], fold_fn=fold_fn)
    assert set(got) == {"esmfold2"}


def test_nothing_to_fold_is_not_an_error(tmp_path):
    assert target_alone_plddt([], str(tmp_path), ["esmfold2"]) == {}
    assert target_alone_plddt(TARGET, str(tmp_path), []) == {}


def test_a_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    path = pathlib.Path(target_reference_cache_path(str(tmp_path)))
    path.write_text("{not json")
    assert read_target_reference(str(tmp_path), "any") is None


def test_the_cache_is_written_whole(tmp_path):
    """Shards write this concurrently; a reader must never see half a file."""
    fingerprint = target_reference_fingerprint(TARGET, ["esmfold2"], 3)
    write_target_reference(str(tmp_path), fingerprint, {"esmfold2": 0.91})
    assert read_target_reference(str(tmp_path), fingerprint) == {"esmfold2": 0.91}
    assert not list(tmp_path.glob("*.tmp")), "no temporary file left behind"
    stored = json.loads(pathlib.Path(target_reference_cache_path(str(tmp_path))).read_text())
    assert stored["fingerprint"] == fingerprint


def test_structures_are_kept_beside_the_campaign(tmp_path):
    fold_fn = make_fold_fn(tmp_path)
    target_alone_plddt(TARGET, str(tmp_path), ["esmfold2"], fold_fn=fold_fn)
    assert (tmp_path / TARGET_REFERENCE_DIR).is_dir()


# ---------------------------------------------------------------------------
# Wiring. Read as source: binder_eval imports torch.
# ---------------------------------------------------------------------------

SRC = pathlib.Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (SRC / rel).read_text()


def test_the_campaign_config_asks_for_a_reference():
    config = _read("configs/pipeline/binder/binder_evaluate.yaml")
    assert "target_reference_models: [esmfold2]" in config
    assert "reuse_cached_target_reference: true" in config


def test_the_reference_is_folded_once_not_once_per_design():
    """Inside the sample loop it would be folded once per design -- the cache
    would hide the cost but not the mistake."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    assert source.index("target_alone_plddt(") < source.index("for idx, sample_root_path in enumerate(")


def test_the_cache_is_campaign_scoped_not_shard_scoped():
    """sample_root_path is one design. Its parent is the directory both shards
    of a campaign write into, which is the scope this reference has."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    call = source[source.index("target_alone_plddt(") : source.index("# Setup columns")]
    assert "os.path.dirname(os.path.abspath(sample_root_paths[0]))" in call


def test_the_reference_reaches_the_table():
    """The denominator belongs next to the ratio: without it in the table there
    is no way to tell a damaged target from a target that never folded well."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    assert 'f"target_alone_pLDDT_{model}": value' in source
    assert 'reference_columns = [f"target_alone_pLDDT_{model}"' in source
    assert "all_columns += reference_columns" in source


def test_a_ligand_target_is_not_folded():
    """A ligand has no sequence, and the folding backends would be handed
    nothing to fold."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    guard = source[source.index("target_reference_models = ") : source.index("# Setup columns")]
    assert "not is_target_ligand" in guard


def test_the_target_sequence_read_is_all_or_nothing():
    """A target read as two chains when it has three is a different molecule,
    and every ratio against it would be wrong with nothing to catch it."""
    source = _read("src/proteinfoundation/evaluation/binder_eval.py")
    body = source[source.index("def _target_chain_sequences(") : source.index("def compute_binder_metrics(")]
    assert "return []" in body, "a failed chain read abandons the whole target"
