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
