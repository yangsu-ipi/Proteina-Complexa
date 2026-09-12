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


def test_the_ipsae_cutoffs_are_in_the_derivation_fingerprint():
    """They were a literal inside _esmfold2_metrics, covered by nothing. Changing
    15 to 12 would have left every cached column untouched and renamed nothing,
    so the next run would have served ipSAE-at-15 under a name that had come to
    mean ipSAE-at-12 -- and, because the cutoffs are not in the FOLD fingerprint
    either, would never have refolded to find out."""
    from proteinfoundation.metrics import consensus_folding as cf

    before = cf.consensus_derivation_fingerprint()
    original = cf.IPSAE_CUTOFFS
    try:
        cf.IPSAE_CUTOFFS = ((12.0, ""), (10.0, "_10"))
        assert cf.consensus_derivation_fingerprint() != before, "a cutoff change must re-derive"
    finally:
        cf.IPSAE_CUTOFFS = original
    assert cf.consensus_derivation_fingerprint() == before

    # And it must NOT re-fold: the folder returns the same structure either way.
    fold_before = cf.consensus_fingerprint("esmfold2", {}, ["AAA"])
    try:
        cf.IPSAE_CUTOFFS = ((12.0, ""), (10.0, "_10"))
        assert cf.consensus_fingerprint("esmfold2", {}, ["AAA"]) == fold_before, (
            "the structure is unchanged, so nothing may refold"
        )
    finally:
        cf.IPSAE_CUTOFFS = original


def test_the_family_is_recomputed_from_the_stored_matrix(tmp_path, monkeypatch):
    """What the store buys. The derivation reads the matrix beside the structure
    and produces the columns again, at whatever cutoffs are current."""
    from proteinfoundation.metrics import consensus_folding as cf

    pae = realistic_pae(target_len=30, binder_len=12)
    structure = str(tmp_path / "c.pdb")
    save_pae(structure, pae, chain_lengths=[30, 12], backend="esmfold2")

    seen = {}

    def fake_family(matrix, target_len):
        seen["target_len"] = target_len
        seen["matrix"] = np.asarray(matrix)
        return {"i_pAE": 0.123, "min_ipSAE": 0.4}

    monkeypatch.setattr(cf, "pae_family", fake_family)
    got = cf.pae_family_from_store(structure, n_target_chains=1)

    assert got == {"i_pAE": 0.123, "min_ipSAE": 0.4}
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
    assert set(cf.PAE_FAMILY_SUFFIXES) <= set(cf.CONSENSUS_METRIC_SUFFIXES), (
        "everything the family produces is a column the backend already declares"
    )
