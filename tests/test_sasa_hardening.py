"""The SASA engine: pinned, recorded, and unable to invent a value.

Three fabrications lived here. The Biopython engine returned 0.30 surface
hydrophobicity and 0.0 for every area when it failed. The FreeSASA engine fell
back to the Biopython one, which carries different radii, so the same input
silently produced numbers ~3% apart with nothing recording which ran. And a
failed chain selection left the binder's in-complex area at 0.0, which turns
interface dSASA into the binder's entire surface -- large, plausible, wrong.
"""

import pytest

pytest.importorskip("Bio", reason="pr_alternative_utils imports Bio.PDB")
freesasa = pytest.importorskip("freesasa", reason="the pinned SASA engine")

from proteinfoundation.utils import pr_alternative_utils as pau
from proteinfoundation.utils.pr_alternative_utils import (
    SASA_STRUCTURE_OPTIONS,
    SasaError,
    pr_alternative_score_interface,
    sasa_provenance,
)

# A blunt two-chain poly-alanine complex. Geometry only has to be legal enough
# for the engines to run -- these tests are about failure handling, not areas.
ATOMS = [("N", 0.0, 0.0, 0.0), ("CA", 1.46, 0.0, 0.0), ("C", 2.0, 1.42, 0.0),
         ("O", 1.3, 2.4, 0.0), ("CB", 2.0, -0.77, -1.2)]


def complex_pdb(tmp_path, name="complex.pdb", chains=("A", "B")):
    lines, serial = [], 1
    for ci, chain in enumerate(chains):
        for res in range(1, 4):
            for atom, x, y, z in ATOMS:
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} ALA {chain}{res:4d}    "
                    f"{x + res * 3.6 + ci * 0.5:8.3f}{y + ci * 5.0:8.3f}{z:8.3f}"
                    f"  1.00 50.00          {atom[0]:>2s}  "
                )
                serial += 1
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\nTER\nEND\n")
    return str(path)


@pytest.fixture
def sc_stub(tmp_path, monkeypatch):
    """sc-rs is a Linux binary; these tests are not about it."""
    stub = tmp_path / "sc"
    stub.write_text('#!/bin/sh\necho \'{"sc": 0.7}\'\n')
    stub.chmod(0o755)
    monkeypatch.setenv("SC_EXEC", str(stub))
    return str(stub)


def test_the_engine_and_radii_are_pinned_not_discovered():
    got = sasa_provenance()
    assert got == {
        "engine": "freesasa",
        "algorithm": "Lee-Richards",
        "radii": "ProtOr",
        "probe_radius": 1.4,
        "n_slices": 20,
    }


def test_hetatm_is_included_and_unknown_atoms_halt():
    """A ligand target is HETATM: dropping it computes the binder against nothing.
    An unknown atom must stop the calculation, not be given a guessed radius."""
    assert SASA_STRUCTURE_OPTIONS["hetatm"] is True
    assert SASA_STRUCTURE_OPTIONS["halt-at-unknown"] is True


def test_the_radii_constant_is_named_for_the_set_it_holds():
    """It was called R_CHOTHIA while holding Bondi values; Chothia's set is
    united-atom and materially different."""
    assert pau.R_BONDI == {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
    assert not hasattr(pau, "R_CHOTHIA")


def test_auto_resolves_to_freesasa_and_says_so(tmp_path, sc_stub):
    scores, _, _ = pr_alternative_score_interface(
        complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="auto"
    )
    assert scores["sasa_engine"] == "freesasa"
    assert scores["sasa_radii"] == "ProtOr"


def test_the_biopython_engine_records_its_own_radii(tmp_path, sc_stub):
    """Still selectable, but never silently: it carries Bondi, not ProtOr."""
    scores, _, _ = pr_alternative_score_interface(
        complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="biopython"
    )
    assert scores["sasa_engine"] == "biopython"
    assert scores["sasa_radii"] == "Bondi"


def test_an_unknown_engine_raises(tmp_path, sc_stub):
    with pytest.raises(SasaError, match="unknown sasa_engine"):
        pr_alternative_score_interface(
            complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="rosetta"
        )


def test_freesasa_failure_raises_instead_of_switching_engines(tmp_path, monkeypatch, sc_stub):
    """The fallback that made this quiet: a FreeSASA error used to return
    Biopython's numbers, computed with different radii, under no new label."""
    called = []
    monkeypatch.setattr(pau, "_compute_sasa_metrics",
                        lambda *a, **k: called.append(1) or (0.3, 0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(pau.freesasa, "calc", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SasaError, match="FreeSASA failed"):
        pr_alternative_score_interface(
            complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="freesasa"
        )
    assert not called, "the other engine must not be used as a fallback"


def test_biopython_failure_raises_instead_of_returning_0_30(tmp_path, monkeypatch, sc_stub):
    monkeypatch.setattr(pau, "ShrakeRupley", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SasaError, match="Biopython SASA failed"):
        pr_alternative_score_interface(
            complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="biopython"
        )


def test_a_failed_chain_selection_raises(tmp_path, monkeypatch, sc_stub):
    """This used to leave the in-complex area at 0.0, making interface dSASA the
    binder's whole surface."""
    monkeypatch.setattr(pau.freesasa, "selectArea",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no such chain")))
    with pytest.raises(SasaError, match="could not select chains"):
        pr_alternative_score_interface(
            complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="freesasa"
        )


def test_impossible_burial_raises_rather_than_clamping(tmp_path, monkeypatch, sc_stub):
    """Burial cannot be negative. Clamping rounding noise to zero is right;
    clamping a real disagreement turns it into a confident 0.0."""
    monkeypatch.setattr(
        pau, "_compute_sasa_metrics_with_freesasa",
        lambda *a, **k: (0.3, 900.0, 100.0, 500.0, 500.0),  # binder buried = -800
    )
    with pytest.raises(SasaError, match="buried area"):
        pr_alternative_score_interface(
            complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="freesasa"
        )


def test_the_two_engines_disagree_which_is_why_it_must_be_recorded(tmp_path, sc_stub):
    """Not a defect -- different radii conventions. It is the reason a dSASA
    without its engine is not reproducible."""
    pdb = complex_pdb(tmp_path)
    a, _, _ = pr_alternative_score_interface(pdb, binder_chain="B", target_chain="A", sasa_engine="freesasa")
    b, _, _ = pr_alternative_score_interface(pdb, binder_chain="B", target_chain="A", sasa_engine="biopython")
    assert a["sasa_engine"] != b["sasa_engine"]
    assert a["sasa_radii"] != b["sasa_radii"]


def test_an_engulfed_binder_is_not_reported_as_having_no_interface(tmp_path, monkeypatch, sc_stub):
    """binder_sasa_in_complex == 0 is the largest interface there is. Reporting
    interface_fraction 0.0 for it was backwards, not merely missing."""
    monkeypatch.setattr(
        pau, "_compute_sasa_metrics_with_freesasa",
        lambda *a, **k: (0.3, 0.0, 1200.0, 500.0, 900.0),  # binder fully buried
    )
    with pytest.raises(SasaError, match="no exposed surface"):
        pr_alternative_score_interface(
            complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="freesasa"
        )


def test_no_interface_residues_gives_nan_hydrophobicity_not_zero(tmp_path, monkeypatch, sc_stub):
    """A binder that misses its target has no interface composition. 0.0 says
    'entirely non-hydrophobic', which a threshold would read as a real value."""
    import math

    monkeypatch.setattr(pau, "hotspot_residues", lambda *a, **k: {})
    scores, _, _ = pr_alternative_score_interface(
        complex_pdb(tmp_path), binder_chain="B", target_chain="A", sasa_engine="freesasa"
    )
    assert scores["interface_nres"] == 0
    assert math.isnan(scores["interface_hydrophobicity"])
