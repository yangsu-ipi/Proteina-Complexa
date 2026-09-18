"""The PAE matrix, kept beside the structure it describes.

Everything in the PAE family is computed from one L x L matrix that the folder
returns and nothing on disk recorded, which is why expanding the advisory metric
list from 6 entries to 14 cost ~20,000 ESMFold2 complex refolds on CBLN1 and
~5,900 on EFNB3. These tests pin the two properties that make the store worth
having: it is small, and it is faithful enough that a metric recomputed from it
is the metric.
"""

import json
import os

import numpy as np
import pytest

from proteinfoundation.metrics.pae_store import (
    PAE_QUANT_MAX,
    PAE_QUANT_STEP,
    load_pae,
    pae_sidecar_path,
    save_pae,
)


def realistic_pae(target_len=136, binder_len=40, seed=0):
    """A PAE with the structure a real one has: confident within each chain,
    less so across the interface, smooth in between. Smoothness is what the
    compression exploits, so a white-noise matrix would understate the cost and
    a constant one would flatter it."""
    rng = np.random.default_rng(seed)
    n = target_len + binder_len
    idx = np.arange(n)
    same_chain = (idx[:, None] < target_len) == (idx[None, :] < target_len)
    separation = np.abs(idx[:, None] - idx[None, :])
    pae = 2.0 + 0.06 * separation + np.where(same_chain, 0.0, 12.0)
    pae += rng.normal(0, 0.4, size=(n, n))
    return np.clip(pae, 0.2, 31.5).astype(np.float32)


def test_a_stored_matrix_round_trips_within_the_models_own_resolution(tmp_path):
    """The case for uint8. Every backend here bins PAE at 0.5 A and reports a
    weighted mean over those bin centres, so a 1/8 A step is four times finer
    than the number is produced at -- the round-trip error cannot reach the
    resolution of the thing being stored."""
    pae = realistic_pae()
    structure = str(tmp_path / "complex_model1.pdb")

    assert save_pae(structure, pae, chain_lengths=[136, 40], backend="af2", model="model1")
    got = load_pae(structure)

    assert got is not None
    assert got["pae"].shape == pae.shape
    assert np.abs(got["pae"] - pae).max() <= PAE_QUANT_STEP / 2 + 1e-6
    assert got["units"] == "angstrom", "ipSAE compares against a cutoff in Angstroms"
    assert got["chain_lengths"] == [136, 40], "the interface split, without reopening the PDB"
    assert got["backend"] == "af2" and got["model"] == "model1"


def test_the_metrics_recomputed_from_it_are_the_metrics(tmp_path):
    """What the store is for. A stored matrix has to reproduce the columns, not
    merely resemble the matrix."""
    pae = realistic_pae()
    target_len = 136
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae, chain_lengths=[target_len, 40])
    back = load_pae(structure)["pae"]

    def i_pae(m):
        return float(np.concatenate([m[target_len:, :target_len].ravel(), m[:target_len, target_len:].ravel()]).mean())

    def mean_pae(m):
        return float(((m + m.T) / 2)[target_len:].mean())

    # Averages over thousands of entries: the quantisation error cancels.
    assert i_pae(back) == pytest.approx(i_pae(pae), abs=1e-3)
    assert mean_pae(back) == pytest.approx(mean_pae(pae), abs=1e-3)
    # A single-element min takes the full step, and 1/16 A is still far below
    # what a 0.5 A bin width can distinguish.
    single = float(back[target_len:, :target_len].min()) - float(pae[target_len:, :target_len].min())
    assert abs(single) <= PAE_QUANT_STEP / 2 + 1e-6


def test_it_is_smaller_than_every_obvious_alternative(tmp_path):
    """The reason for compressing quantised bytes rather than storing floats or
    the JSON the RF3 runner already writes."""
    pae = realistic_pae()
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae, chain_lengths=[136, 40])
    stored = os.path.getsize(pae_sidecar_path(structure))

    as_float32 = pae.astype(np.float32).nbytes
    as_json = len(json.dumps(pae.round(2).tolist()))

    assert stored < as_float32 / 3, f"{stored} vs {as_float32} bytes of float32"
    assert stored < as_json / 8, f"{stored} vs {as_json} bytes of JSON"


def test_the_asymmetry_survives(tmp_path):
    """PAE is not symmetric -- pae[i][j] is the error at j when aligned on i --
    and ipSAE reads both directions and takes their min and max. A store that
    symmetrised would quietly collapse two of the six ipSAE columns into one."""
    pae = realistic_pae()
    pae[10, 150] = 3.0
    pae[150, 10] = 28.0
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae, chain_lengths=[136, 40])
    back = load_pae(structure)["pae"]

    assert back[10, 150] == pytest.approx(3.0, abs=PAE_QUANT_STEP)
    assert back[150, 10] == pytest.approx(28.0, abs=PAE_QUANT_STEP)
    assert back[10, 150] != back[150, 10]


def test_a_fold_with_no_stored_pae_is_unmeasured_not_zero(tmp_path):
    """Every fold from before the store lands here, and a zero PAE is a claim of
    perfect confidence that nobody made."""
    assert load_pae(str(tmp_path / "never_folded.pdb")) is None
    # A backend that reports no matrix writes no file, rather than a file
    # asserting nothing.
    assert save_pae(str(tmp_path / "c.pdb"), None) is None
    assert load_pae(str(tmp_path / "c.pdb")) is None


def test_the_sidecar_sits_beside_a_structure_with_no_extension_to_replace():
    """Monomer folds are named `esm_1_seed7.pdb_esm_apo_mpnn`. Substituting an
    extension would put two seeds' matrices at one path."""
    a = pae_sidecar_path("/d/esm_1_seed7.pdb_esm_apo_mpnn")
    b = pae_sidecar_path("/d/esm_1_seed8.pdb_esm_apo_mpnn")
    assert a != b and a.startswith("/d/esm_1_seed7.pdb_esm_apo_mpnn")


def test_a_matrix_beyond_the_representable_range_is_clipped_not_wrapped(tmp_path):
    """uint8 arithmetic wraps. A PAE above the cap must come back as the cap --
    "at least this uncertain" -- rather than as near-zero, which would read as
    the most confident pair in the complex."""
    pae = np.full((8, 8), 40.0, dtype=np.float32)
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae)
    back = load_pae(structure)["pae"]
    assert back.max() == pytest.approx(PAE_QUANT_MAX)


def test_a_corrupt_sidecar_is_ignored_rather_than_raising(tmp_path):
    """A fold costs minutes and this costs milliseconds; a damaged sidecar must
    not take an evaluation down."""
    structure = str(tmp_path / "c.pdb")
    with open(pae_sidecar_path(structure), "wb") as handle:
        handle.write(b"not an npz")
    assert load_pae(structure) is None


def test_a_non_square_matrix_is_refused(tmp_path):
    """PAE is L x L. Anything else is a different quantity and would be stored
    under a name that promises this one."""
    structure = str(tmp_path / "c.pdb")
    assert save_pae(structure, np.zeros((4, 7), dtype=np.float32)) is None
    assert load_pae(structure) is None


# ---------------------------------------------------------------------------
# The point of storing it: changing an ipSAE cutoff must cost a re-read.
# ---------------------------------------------------------------------------


def test_a_cutoff_is_a_request_not_an_identity():
    """The cutoffs were a literal inside _esmfold2_metrics, covered by nothing:
    changing 15 to 12 renamed nothing and invalidated nothing. They are in
    neither fingerprint now either -- but for the opposite reason. The fold
    fingerprint must not see them because the structure does not depend on them,
    and the derivation fingerprint must not either because hashing them made
    every cutoff change re-derive every structure-read metric for every cutoff.
    Reuse is per cutoff, recorded per entry."""
    from proteinfoundation.metrics import consensus_folding as cf

    original = cf.IPSAE_CUTOFFS
    fold = cf.consensus_fingerprint("esmfold2", {}, ["AAA"])
    derivation = cf.consensus_derivation_fingerprint()
    try:
        cf.IPSAE_CUTOFFS = ((12.0, "_12"),)
        assert cf.consensus_fingerprint("esmfold2", {}, ["AAA"]) == fold
        assert cf.consensus_derivation_fingerprint() == derivation
    finally:
        cf.IPSAE_CUTOFFS = original

    # What IS in the fold fingerprint is whether the family is reported at all:
    # a cache written before ipSAE existed cannot produce it from anything.
    base = cf.PAE_BASE_METRICS
    try:
        cf.PAE_BASE_METRICS = tuple(x for x in base if x != "min_ipSAE")
        assert cf.consensus_fingerprint("esmfold2", {}, ["AAA"]) != fold
    finally:
        cf.PAE_BASE_METRICS = base


def test_the_family_is_recomputed_from_the_stored_matrix(tmp_path, monkeypatch):
    """What the store buys. The derivation reads the matrix beside the structure
    and produces the columns again, at whatever cutoffs are current."""
    from proteinfoundation.metrics import consensus_folding as cf

    pae = realistic_pae(target_len=30, binder_len=12)
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae, chain_lengths=[30, 12], backend="esmfold2")

    seen = {}

    def fake_family(matrix, target_len, cutoffs=None, include_base=True):
        seen["target_len"] = target_len
        seen["matrix"] = np.asarray(matrix)
        seen["cutoffs"] = cutoffs
        return {"i_pAE": 0.123, "min_ipSAE": 0.4}

    monkeypatch.setattr(cf, "pae_family", fake_family)
    got = cf.pae_family_from_store(structure, n_target_chains=1)

    assert got["i_pAE"] == 0.123 and got["min_ipSAE"] == 0.4
    assert seen["target_len"] == 30, "the chain lengths in the sidecar place the interface"
    assert np.abs(seen["matrix"] - pae).max() <= PAE_QUANT_STEP / 2 + 1e-6


def test_a_fold_with_no_stored_matrix_yields_nothing_rather_than_a_guess(tmp_path):
    """Every fold from before the store. Placing the interface by guessing would
    produce plausible, wrong numbers -- worse than the absence."""
    from proteinfoundation.metrics.consensus_folding import pae_family_from_store

    assert pae_family_from_store(str(tmp_path / "never.pdb"), 1) == {}

    # A matrix stored without chain lengths cannot say where the binder starts.
    bare = str(tmp_path / "bare.pdb")
    save_pae(bare, realistic_pae(20, 8), chain_lengths=None)
    assert pae_family_from_store(bare, 1) == {}

    # And one describing a different number of target chains is not this complex.
    two = str(tmp_path / "two.pdb")
    save_pae(two, realistic_pae(20, 8), chain_lengths=[10, 10, 8])
    assert pae_family_from_store(two, n_target_chains=1) == {}


def test_one_definition_of_the_family():
    """The fold-time path and the re-derivation must be the same arithmetic, or
    a column recomputed from the store would differ from the one it replaces."""
    from proteinfoundation.metrics import consensus_folding as cf

    source = (
        __import__("pathlib").Path("src/proteinfoundation/metrics/consensus_folding.py").read_text()
    )
    body = source.split("def _esmfold2_metrics")[1].split("\ndef ")[0]
    assert "pae_family(" in body, "the backend delegates rather than reimplementing"
    assert "ipsae(" not in body, "and holds no second copy of the arithmetic"
    assert set(cf.ipsae_suffixes()) <= set(cf.CONSENSUS_METRIC_SUFFIXES), (
        "every cutoff the request names produces columns the backend declares"
    )


# ---------------------------------------------------------------------------
# Rounds of comparison with overlapping cutoffs. The question is not whether a
# cutoff can be changed -- it is whether changing it twice costs twice.
# ---------------------------------------------------------------------------


def test_a_cutoff_already_scored_is_not_scored_again():
    """Round 1 asks for 15 and 10, round 2 for 15 and 12, round 3 for all three.
    Only what is new is computed each time; the overlap is reused."""
    from proteinfoundation.metrics.consensus_folding import PAE_CUTOFF_KEY, missing_pae_cutoffs

    entry = {}
    round1 = ((15.0, ""), (10.0, "_10"))
    assert missing_pae_cutoffs(entry, round1) == round1, "nothing cached yet"

    # After round 1 the entry holds both, and records what produced them.
    entry.update({f"{k}ipSAE{s}": 0.5 for _, s in round1 for k in ("min_", "max_", "avg_")})
    entry[PAE_CUTOFF_KEY] = {"": 15.0, "_10": 10.0}
    assert missing_pae_cutoffs(entry, round1) == ()

    round2 = ((15.0, ""), (12.0, "_12"))
    assert missing_pae_cutoffs(entry, round2) == ((12.0, "_12"),), "15 is reused, 12 is new"

    entry.update({f"{k}ipSAE_12": 0.4 for k in ("min_", "max_", "avg_")})
    entry[PAE_CUTOFF_KEY]["_12"] = 12.0
    round3 = ((15.0, ""), (10.0, "_10"), (12.0, "_12"))
    assert missing_pae_cutoffs(entry, round3) == (), "a round that unions earlier ones is free"


def test_a_suffix_whose_cutoff_moved_is_recomputed():
    """The reason reuse is keyed on the distance and not the column name. A round
    that redefines the plain suffix from 15 A to 12 must not keep the 15 A
    numbers under a name that has come to mean something else."""
    from proteinfoundation.metrics.consensus_folding import PAE_CUTOFF_KEY, missing_pae_cutoffs

    entry = {f"{k}ipSAE": 0.5 for k in ("min_", "max_", "avg_")}
    entry[PAE_CUTOFF_KEY] = {"": 15.0}
    assert missing_pae_cutoffs(entry, ((15.0, ""),)) == ()
    assert missing_pae_cutoffs(entry, ((12.0, ""),)) == ((12.0, ""),)


def test_columns_present_without_a_record_are_not_trusted():
    """An entry folded before the record existed has the columns but cannot say
    at what distance. Recomputing is cheap; assuming is how a comparison of two
    cutoffs quietly becomes a comparison of one with itself."""
    from proteinfoundation.metrics.consensus_folding import missing_pae_cutoffs

    entry = {f"{k}ipSAE": 0.5 for k in ("min_", "max_", "avg_")}
    assert missing_pae_cutoffs(entry, ((15.0, ""),)) == ((15.0, ""),)


def test_adding_a_cutoff_does_not_refold_and_does_not_reread_the_structure(tmp_path, monkeypatch):
    """The two costs a cutoff change must not pay. Refolding is minutes per
    complex; re-reading the structure to rescore its interface is ~0.3 s against
    the ~1 ms the cutoff itself takes, which over CBLN1's 22,000 advisory
    structures is two hours against half a minute."""
    from proteinfoundation.metrics import consensus_folding as cf

    fold_before = cf.consensus_fingerprint("esmfold2", {}, ["AAA"])
    derivation_before = cf.consensus_derivation_fingerprint()
    original = cf.IPSAE_CUTOFFS
    try:
        cf.IPSAE_CUTOFFS = ((15.0, ""), (10.0, "_10"), (12.0, "_12"))
        assert cf.consensus_fingerprint("esmfold2", {}, ["AAA"]) == fold_before, "no refold"
        assert cf.consensus_derivation_fingerprint() == derivation_before, (
            "and no re-derivation of SASA, shape complementarity or the RMSDs either"
        )
    finally:
        cf.IPSAE_CUTOFFS = original

    # And the structure is not opened when only a cutoff is missing.
    pae = realistic_pae(target_len=30, binder_len=12)
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae, chain_lengths=[30, 12])

    def explode(*args, **kwargs):
        raise AssertionError("the structure must not be rescored to add a cutoff")

    monkeypatch.setattr(
        "proteinfoundation.utils.pr_alternative_utils.pr_alternative_score_interface", explode
    )
    held = dict.fromkeys(cf.consensus_derived_suffixes(False), 0.0)
    held.update({"i_pAE": 0.1, "pAE": 0.2, "min_ipAE": 0.3})
    held.update({f"{k}ipSAE": 0.5 for k in ("min_", "max_", "avg_")})
    held[cf.PAE_CUTOFF_KEY] = {"": 15.0}

    try:
        cf.IPSAE_CUTOFFS = ((15.0, ""), (10.0, "_10"))
        got = cf.derive_from_structure(structure, 1, reference_pdb_path=None, have=held)
    finally:
        cf.IPSAE_CUTOFFS = original

    assert "min_ipSAE_10" in got, "the new cutoff was computed"
    assert "min_ipSAE" not in got, "and the one already held was not"
    assert got[cf.PAE_CUTOFF_KEY] == {"": 15.0, "_10": 10.0}, "the record accumulates"


def test_the_ipsae_cutoffs_drive_the_column_list():
    """Adding a cutoff has to add columns. The names were a hardcoded tuple
    beside the cutoffs, so a third cutoff would have been computed and then never
    emitted."""
    from proteinfoundation.metrics import consensus_folding as cf

    original = cf.IPSAE_CUTOFFS
    try:
        cf.IPSAE_CUTOFFS = ((15.0, ""), (12.0, "_12"))
        names = cf.consensus_metric_suffixes()
        assert "min_ipSAE_12" in names and "avg_ipSAE_12" in names
        assert "min_ipSAE_10" not in names
    finally:
        cf.IPSAE_CUTOFFS = original


def test_a_stored_matrix_names_the_checkpoint_that_made_it():
    """The advisory sidecar recorded an empty model string on every real run: it
    read consensus_cfg["model_id"], which campaigns only set when overriding the
    checkpoint. A stored PAE that cannot say which model produced it is a matrix
    nobody can compare against another run's."""
    from proteinfoundation.metrics import consensus_folding as cf
    from proteinfoundation.metrics.esmfold2_loader import complex_model_id

    assert cf._esmfold2_model_id({}) == complex_model_id(), "the default, not an empty string"
    assert cf._esmfold2_model_id({"model_id": "someone/else"}) == "someone/else"
    assert cf._esmfold2_model_id({}), "never empty"


def test_dropping_structures_keeps_what_cannot_be_recovered(tmp_path):
    """keep_folding_outputs: false means "I do not need the PDBs", not "discard
    everything derived from them". The two are not the same trade: a structure is
    large and reproducible by refolding, while the PAE matrix is small and is the
    only reason a later ipSAE cutoff costs a re-read instead of a campaign.

    The monomer cleanup used to rmtree the whole directory, which threw away the
    cheap irreplaceable artifact to save the expensive reproducible one."""
    from proteinfoundation.metrics.pae_store import drop_structures_keeping_sidecars

    root = tmp_path / "esmfold2_output" / "apo_mpnn"
    root.mkdir(parents=True)
    (root / "esm_1_seed7.pdb_esm_apo_mpnn").write_text("ATOM\n")
    (root / "esm_1_seed7.pdb_esm_apo_mpnn.pae.npz").write_bytes(b"\x00")
    (root / "esm_1_seed7.pdb_esm_apo_mpnn.confidence.json").write_text("{}")
    (root / "scratch.a3m").write_text(">x\nAAAA\n")

    removed, kept = drop_structures_keeping_sidecars(str(root))
    assert (removed, kept) == (2, 2), "the structure and the scratch file go; both sidecars stay"
    assert not (root / "esm_1_seed7.pdb_esm_apo_mpnn").exists()
    assert (root / "esm_1_seed7.pdb_esm_apo_mpnn.pae.npz").exists()
    assert (root / "esm_1_seed7.pdb_esm_apo_mpnn.confidence.json").exists()


def test_dropping_structures_never_raises(tmp_path):
    """Cleanup that fails must not fail a run whose metrics are already
    computed."""
    from proteinfoundation.metrics.pae_store import drop_structures_keeping_sidecars

    assert drop_structures_keeping_sidecars(str(tmp_path / "never_existed")) == (0, 0)


def test_every_complex_folder_honours_the_retention_flag():
    """The primary complex structures -- n_af2_models per sequence per design,
    the largest set a run produces -- were written unconditionally and never
    cleaned up, while the apo folds and the second complex folder's structures
    honoured keep_folding_outputs. One flag, every folder."""
    import inspect

    from proteinfoundation.evaluation import binder_eval, monomer_eval

    for module, where in ((binder_eval, "the complex tracks"), (monomer_eval, "the monomer tracks")):
        source = inspect.getsource(module)
        assert "drop_structures_keeping_sidecars(" in source, f"{where} ignore keep_folding_outputs"
        assert "shutil.rmtree(model_dir)" not in source, f"{where} still delete sidecars wholesale"


def test_the_esmc_pin_narrows_the_name_and_never_breaks_scoring(monkeypatch):
    """pinned_esmc_location turns the ESMC repo id into a fixed snapshot path, so
    a moved refs/main cannot redirect it. Reaching the pin imports the fork's
    esmfold2 package, whose init pulls in triton's CUDA kernels -- on a box with
    no GPU driver that raises RuntimeError, and ESM scoring needs no GPU. So an
    unreachable pin costs the pin, never the scoring."""
    import builtins

    from proteinfoundation.evaluation.esm_eval import pinned_esmc_location

    # A name that is not the ESMC repo is never touched.
    assert pinned_esmc_location("facebook/esm2_t33_650M_UR50D") == "facebook/esm2_t33_650M_UR50D"

    real_import = builtins.__import__

    def no_driver(name, *args, **kwargs):
        if name.startswith("esm.models.esmfold2"):
            raise RuntimeError("0 active drivers ([]). There should only be one.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_driver)
    assert pinned_esmc_location("biohub/ESMC-6B") == "biohub/ESMC-6B", (
        "an unreachable pin must fall back to the name, not raise"
    )


def test_a_relocated_structure_takes_its_sidecars_with_it(tmp_path):
    """The advisory store copies a harness's structure to the path it owns, and
    the cache entry records that path. A sidecar left at the harness's path is a
    matrix under a name no reader looks up -- the fold is paid for and the
    re-read it was supposed to buy is gone. This is what left every af2 advisory
    fold on CBLN1 with 770 structures and zero findable matrices."""
    from proteinfoundation.metrics.pae_store import CONFIDENCE_KEPT_SUFFIX, carry_sidecars

    produced = tmp_path / "harness" / "model1.pdb"
    produced.parent.mkdir(parents=True)
    produced.write_text("ATOM\n")
    pae = realistic_pae(target_len=8, binder_len=4)
    save_pae(str(produced), pae, chain_lengths=[8, 4], backend="af2", model="model1")
    (tmp_path / "harness" / f"model1.pdb{CONFIDENCE_KEPT_SUFFIX}").write_text('{"pTM": 0.5}')

    wanted = tmp_path / "store" / "af2_complex" / "abc123_model1.pdb"
    wanted.parent.mkdir(parents=True)
    assert carry_sidecars(str(produced), str(wanted)) == 2

    # The reader addresses the matrix by the path the cache entry records.
    reread = load_pae(str(wanted))
    assert reread is not None, "the matrix must be findable beside the recorded path"
    assert np.abs(reread["pae"] - pae).max() <= PAE_QUANT_STEP / 2 + 1e-6
    assert reread["chain_lengths"] == [8, 4]
    assert os.path.exists(str(wanted) + CONFIDENCE_KEPT_SUFFIX)


def test_carrying_sidecars_never_fails_a_fold_and_never_copies_onto_itself(tmp_path):
    """A structure already at its final path has nothing to carry, and a
    companion that cannot be copied costs a re-read, not the fold that just
    produced it."""
    from proteinfoundation.metrics.pae_store import carry_sidecars

    produced = tmp_path / "model1.pdb"
    produced.write_text("ATOM\n")
    save_pae(str(produced), realistic_pae(target_len=6, binder_len=3), chain_lengths=[6, 3], backend="af2")

    # Same path in and out: copying a file onto itself would truncate it.
    assert carry_sidecars(str(produced), str(produced)) == 0
    assert load_pae(str(produced)) is not None

    # An unwritable destination is a warning, not an exception.
    assert carry_sidecars(str(produced), "/proc/nonexistent-dir/model1.pdb") == 0


def test_place_structure_carries_the_sidecars(tmp_path):
    """The wiring, not just the helper: _place_structure is the single point
    every harness-produced advisory structure passes through."""
    from proteinfoundation.metrics.consensus_folding import _place_structure

    produced = tmp_path / "harness" / "out.pdb"
    produced.parent.mkdir(parents=True)
    produced.write_text("ATOM\n")
    save_pae(str(produced), realistic_pae(target_len=5, binder_len=3), chain_lengths=[5, 3], backend="af2")

    wanted = tmp_path / "store" / "af2_complex" / "deadbeef_model2.pdb"
    placed = _place_structure(str(produced), str(wanted))

    assert placed == str(wanted)
    assert load_pae(placed) is not None, "_place_structure dropped the matrix it was copying past"


def test_the_matrix_survives_keep_folding_outputs_being_off():
    """The two artefacts are not the same trade. Everything read off a structure
    is stable once computed, so discarding the structure costs a refold only to
    someone who wants to look at it. The PAE family is folder-reported: a later
    ipSAE cutoff has no source but the matrix, or predicting the complex again.
    So the matrix must not ride on the structure's retention flag."""
    import inspect

    from proteinfoundation.metrics import consensus_folding

    source = inspect.getsource(consensus_folding._score_esmfold2)
    anchor_at = source.index("anchor = out_pdb_path or pae_path")
    structure_at = source.index("if out_pdb_path:")
    assert anchor_at < structure_at, (
        "save_pae must be reached on a path that does not require out_pdb_path; "
        "inside `if out_pdb_path:` is what tied the matrix to keep_folding_outputs"
    )

    # The caller must offer a name even when it is keeping no structures.
    caller = inspect.getsource(consensus_folding.score_binders)
    assert "out_pdb = anchor if keep_structures else None" in caller
    assert "pae_path=anchor" in caller, "the anchor never reaches the folder"


def test_a_stored_matrix_is_read_back_with_no_structure_beside_it(tmp_path):
    """Storing the matrix without a structure is pointless if the reader still
    demands the structure. The PAE family is not read off the structure -- it is
    read off the matrix beside the structure's name."""
    from proteinfoundation.metrics.consensus_folding import pae_family_from_store

    anchor = tmp_path / "esmfold2_complex" / "abc123_seed7.pdb"
    anchor.parent.mkdir(parents=True)
    pae = realistic_pae(target_len=30, binder_len=12)
    save_pae(str(anchor), pae, chain_lengths=[30, 12], backend="esmfold2", seed=7)
    assert not anchor.exists(), "this test is meaningless if a structure is present"

    family = pae_family_from_store(str(anchor), n_target_chains=1)
    assert family, "no structure must not mean no PAE family"
    assert "i_pAE" in family and family["i_pAE"] == family["i_pAE"]


def test_the_cache_keeps_folder_values_when_only_a_structure_is_missing(tmp_path):
    """Adoption discards the folder-reported PAE columns only when a stored
    matrix can reproduce them. An entry from a run that kept no structures has
    the matrix and no pdb_path, so asking about pdb_path alone would throw away
    values nothing could replace."""
    from proteinfoundation.evaluation.binder_eval_cache import _has_stored_pae

    anchor = tmp_path / "esmfold2_complex" / "deadbeef_seed1.pdb"
    anchor.parent.mkdir(parents=True)
    save_pae(str(anchor), realistic_pae(target_len=9, binder_len=4), chain_lengths=[9, 4], backend="esmfold2")

    assert _has_stored_pae(str(anchor)) is True, "the matrix is there; the structure never was"
    assert _has_stored_pae(None) is False
    assert _has_stored_pae(str(tmp_path / "never_folded.pdb")) is False


def test_a_new_cutoff_is_answered_from_the_matrix_with_no_structure_kept(tmp_path, monkeypatch):
    """The whole point, end to end. A run that kept no structures still has the
    matrix, so widening the ipSAE request must cost a re-read of that file and
    not a refold. Before this, the derivation demanded a structure and skipped
    the entry, so the only way to answer was to predict the complex again."""
    from proteinfoundation.metrics import consensus_folding as cf

    target, binder = ["MKVTARGETSEQ"], "AAAACCCCGGGG"
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []

    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, pae_path=None, **_):
        calls.append(draw)
        # A folder that keeps no structure still stores its matrix, under the
        # name the structure would have had.
        save_pae(
            pae_path,
            realistic_pae(target_len=len(target_seqs[0]), binder_len=len(seq)),
            chain_lengths=[len(target_seqs[0]), len(seq)],
            backend="esmfold2",
        )
        return {"i_pTM": 0.5, "pTM": 0.6, "pae_path": pae_path}

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", scorer)

    first = cf.score_binders(
        "esmfold2", target, [binder], cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=False
    )
    assert len(calls) == 1, "the first pass has to fold once"
    assert first and first[0], "the fold produced no metrics"
    assert not list(cache.glob("**/*.pdb")), "keep_structures=False must keep no structures"
    assert list(cache.glob("**/*.pae.npz")), "the matrix must survive keep_structures=False"

    # Second pass: the cache answers, and the PAE family comes off the matrix.
    second = cf.score_binders(
        "esmfold2", target, [binder], cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=False
    )
    assert len(calls) == 1, "a stored matrix must not be re-earned by folding again"
    # Presence first: `x == x` alone passes when the key is absent, because
    # None == None. Absent is exactly the failure this test exists to catch.
    reread = second[0].get("i_pAE")
    assert isinstance(reread, float) and reread == reread, (
        f"i_pAE came back {reread!r}; the PAE family was not read off the stored matrix"
    )


def _matrix_trigger_scorer(calls, store_matrix=True):
    """A folder that records every call and optionally stores its matrix."""

    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, pae_path=None, **_):
        calls.append((seq, draw))
        if store_matrix:
            save_pae(
                pae_path,
                realistic_pae(target_len=len(target_seqs[0]), binder_len=len(seq)),
                chain_lengths=[len(target_seqs[0]), len(seq)],
                backend="esmfold2",
            )
        return {"i_pTM": 0.5, "pTM": 0.6, "pae_path": pae_path}

    return scorer


def test_a_missing_matrix_asks_for_the_fold_again(tmp_path, monkeypatch):
    """A cached entry whose matrix is gone can only answer the cutoffs it was
    folded at, and no re-read can change that. So a missing matrix triggers the
    fold the same way a missing structure does when one was asked for."""
    from proteinfoundation.metrics import consensus_folding as cf
    from proteinfoundation.metrics.pae_store import pae_sidecar_path

    target, binder = ["MKVTARGETSEQ"], "AAAACCCCGGGG"
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", _matrix_trigger_scorer(calls))
    kw = dict(cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=False)

    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1

    # Cached and matrixed: nothing to do.
    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1, "a complete entry must not refold"

    # Delete the matrix. The scores are still cached, so only the missing matrix
    # can ask for this fold.
    matrices = list(cache.glob("**/*.pae.npz"))
    assert matrices, "the first fold stored nothing to delete"
    for m in matrices:
        m.unlink()

    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 2, "a missing matrix must trigger the fold"
    assert list(cache.glob("**/*.pae.npz")), "the refold must restore the matrix"
    assert pae_sidecar_path  # imported for the reader's addressing, asserted by use above


def test_a_folder_that_reports_no_matrix_is_not_refolded_forever(tmp_path, monkeypatch):
    """The trigger must not become an indefinite refold. A folder that answers
    without a PAE matrix will answer without one again, so the entry records
    that and stops asking -- the failure AdvisoryStructureWriteError exists to
    make loud, arrived at from the other side."""
    from proteinfoundation.metrics import consensus_folding as cf
    from proteinfoundation.metrics.pae_store import PAE_UNAVAILABLE_KEY

    target, binder = ["MKVTARGETSEQ"], "AAAACCCCGGGG"
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []
    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", _matrix_trigger_scorer(calls, store_matrix=False))
    kw = dict(cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=False)

    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1
    assert not list(cache.glob("**/*.pae.npz")), "this folder stores no matrix, by construction"

    cached = json.loads((cache / "consensus_fold_cache_esmfold2.json").read_text())
    entry = next(iter(next(iter(cached["scores"].values())).values()))
    assert entry.get(PAE_UNAVAILABLE_KEY) is True, "a matrixless fold must say so in its entry"

    for _ in range(3):
        cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1, "a folder that reports no matrix must not be refolded on every run"


# ---------------------------------------------------------------------------
# A cached metric must describe the structure actually beside it.
# ---------------------------------------------------------------------------


def test_a_structure_is_only_visible_once_it_is_complete(tmp_path):
    """A run killed mid-write must not leave a truncated structure in place.
    Existence is what suppresses the refold, so a half-written file is worse
    than none: it is reused forever and never repaired."""
    from proteinfoundation.metrics.consensus_folding import write_structure_atomically

    target = tmp_path / "deep" / "complex.pdb"

    def explode(tmp):
        open(tmp, "w").write("ATOM      1  N   ALA A   1\n")
        raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        write_structure_atomically(explode, str(target))
    assert not target.exists(), "a failed write must leave nothing at the final path"
    assert not list(target.parent.glob("*.tmp-*")), "the temporary must not survive"

    write_structure_atomically(lambda tmp: open(tmp, "w").write("ATOM\n"), str(target))
    assert target.read_text() == "ATOM\n"
    assert not list(target.parent.glob("*.tmp-*"))


def test_an_entry_records_which_structure_it_measured(tmp_path, monkeypatch):
    """The digest is of the bytes the fold wrote, and it reaches the cache."""
    from proteinfoundation.metrics import consensus_folding as cf
    from proteinfoundation.metrics.consensus_folding import STRUCTURE_DIGEST_KEY, structure_digest

    target, binder = ["MKVTARGETSEQ"], "AAAACCCCGGGG"
    cache = tmp_path / "cache"
    cache.mkdir()

    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, pae_path=None, **_):
        cf.write_structure_atomically(lambda t: open(t, "w").write("ATOM      1  N   ALA A   1\n"), out_pdb)
        return {"i_pTM": 0.5, "pTM": 0.6, "pdb_path": out_pdb,
                STRUCTURE_DIGEST_KEY: structure_digest(out_pdb)}

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", scorer)
    kw = dict(cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=True)
    cf.score_binders("esmfold2", target, [binder], **kw)

    cached = json.loads((cache / "consensus_fold_cache_esmfold2.json").read_text())
    entry = next(iter(next(iter(cached["scores"].values())).values()))
    pdb = entry["pdb_path"]
    assert entry[STRUCTURE_DIGEST_KEY] == structure_digest(pdb), "the entry must name the file it measured"


def test_a_structure_that_changed_under_its_entry_is_refolded(tmp_path, monkeypatch):
    """The crash window: the structure is written inside the fold, the entry
    after the whole batch. Killed in between, disk holds a new prediction under
    the previous run's numbers -- and since a fold is not reproducible from its
    seed, those numbers describe a structure that is gone."""
    from proteinfoundation.metrics import consensus_folding as cf
    from proteinfoundation.metrics.consensus_folding import STRUCTURE_DIGEST_KEY, structure_digest

    target, binder = ["MKVTARGETSEQ"], "AAAACCCCGGGG"
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []

    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, pae_path=None, **_):
        calls.append(draw)
        cf.write_structure_atomically(
            lambda t: open(t, "w").write(f"ATOM      1  N   ALA A   1  fold{len(calls)}\n"), out_pdb)
        return {"i_pTM": 0.5, "pTM": 0.6, "pdb_path": out_pdb,
                STRUCTURE_DIGEST_KEY: structure_digest(out_pdb)}

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", scorer)
    kw = dict(cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=True)

    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1
    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1, "an entry that matches its structure must not refold"

    # Simulate the crash: the structure on disk moves on, the entry does not.
    pdb = next(cache.glob("**/esmfold2_complex/*.pdb"))
    pdb.write_text("ATOM      1  N   ALA A   1  interrupted\n")

    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 2, "a structure that changed under its entry must be refolded"

    cached = json.loads((cache / "consensus_fold_cache_esmfold2.json").read_text())
    entry = next(iter(next(iter(cached["scores"].values())).values()))
    assert entry[STRUCTURE_DIGEST_KEY] == structure_digest(entry["pdb_path"]), "must end consistent"


def test_an_entry_without_a_digest_is_left_alone(tmp_path, monkeypatch):
    """Every campaign cached before this existed has no digest. Absent is not
    mismatched: reading it as such would refold every one of them once."""
    from proteinfoundation.metrics import consensus_folding as cf
    from proteinfoundation.metrics.consensus_folding import STRUCTURE_DIGEST_KEY

    target, binder = ["MKVTARGETSEQ"], "AAAACCCCGGGG"
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []

    def scorer(target_seqs, seq, cfg, out_pdb, draw, context=None, out_path_for=None, pae_path=None, **_):
        calls.append(draw)
        cf.write_structure_atomically(lambda t: open(t, "w").write("ATOM      1  N   ALA A   1\n"), out_pdb)
        return {"i_pTM": 0.5, "pTM": 0.6, "pdb_path": out_pdb}   # pre-digest entry

    monkeypatch.setitem(cf.CONSENSUS_BACKENDS, "esmfold2", scorer)
    kw = dict(cfg={"n_seeds": 1}, cache_dir=str(cache), keep_structures=True)

    cf.score_binders("esmfold2", target, [binder], **kw)
    cached = json.loads((cache / "consensus_fold_cache_esmfold2.json").read_text())
    entry = next(iter(next(iter(cached["scores"].values())).values()))
    assert STRUCTURE_DIGEST_KEY not in entry

    pdb = next(cache.glob("**/esmfold2_complex/*.pdb"))
    pdb.write_text("ATOM      1  N   ALA A   1  changed\n")
    cf.score_binders("esmfold2", target, [binder], **kw)
    assert len(calls) == 1, "a legacy entry with no digest must not be refolded"
