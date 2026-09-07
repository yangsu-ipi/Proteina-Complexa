"""One interface definition, replacing three that disagreed.

`{seq}_aa_interface_counts` used CA atoms at 8 A; `interface_nres` and
`interface_hydrophobicity` used all atoms at 4 A. Over 12 CBLN1 complexes those
selected 15.9 and 17.9 binder residues, mean Jaccard 0.664, and the 4 A set was
never a subset of the 8 A one.
"""

import pytest

pytest.importorskip("protein_interface", reason="the interface engine")

from proteinfoundation.metrics.interface import (
    DEFAULT_CONTACT_CUTOFF,
    INTERFACE_DSASA_THRESHOLD,
    INTERFACE_MODE,
    InterfaceError,
    InterfaceResidue,
    interface_provenance,
    interface_residues,
    resseqs,
    sequence_indices,
)

# Two chains far enough apart to have no interface, close enough to make one when
# translated. Geometry only has to be legal; the assertions are about contracts.
ATOMS = [("N", 0.0, 0.0, 0.0), ("CA", 1.46, 0.0, 0.0), ("C", 2.0, 1.42, 0.0),
         ("O", 1.3, 2.4, 0.0), ("CB", 2.0, -0.77, -1.2)]


def complex_pdb(tmp_path, gap=5.0, name="c.pdb", first_resseq=1, resname="ALA"):
    lines, serial = [], 1
    for ci, chain in enumerate(("A", "B")):
        for res in range(3):
            for atom, x, y, z in ATOMS:
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} {resname} {chain}"
                    f"{first_resseq + res if chain == 'B' else 1 + res:4d}    "
                    f"{x + res * 3.6:8.3f}{y + ci * gap:8.3f}{z:8.3f}  1.00 50.00"
                    f"          {atom[0]:>2s}  "
                )
                serial += 1
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\nTER\nEND\n")
    return str(path)


def test_the_mode_is_burial_aware_not_just_distance():
    """The reason for the change. A purely geometric cutoff misses residues that
    bury area without a close contact; strict takes the union of both."""
    assert INTERFACE_MODE == "strict"
    assert INTERFACE_DSASA_THRESHOLD == 3.0
    assert DEFAULT_CONTACT_CUTOFF == 5.0


def test_the_provenance_names_everything_that_moves_the_set():
    got = interface_provenance()
    assert got["engine"] == "protein-interface"
    assert got["engine_version"] == "0.1.3", "pinned: the radii are compiled in and unreadable"
    assert got["mode"] == "strict"
    assert got["dsasa_threshold"] == 3.0
    assert got["contact_cutoff"] == 5.0
    assert got["n_points"] == 960
    assert "version" in got


def test_a_cutoff_change_is_visible_in_the_provenance():
    assert interface_provenance(8.0)["contact_cutoff"] == 8.0


def test_both_sides_are_returned(tmp_path):
    """The old all-atom function returned binder residues only, so the target's
    interface had to be obtained by swapping its arguments."""
    binder, target = interface_residues(complex_pdb(tmp_path, gap=4.0), ["B"], ["A"])
    assert binder and target
    assert {r.chain for r in binder} == {"B"}
    assert {r.chain for r in target} == {"A"}
    assert all(isinstance(r, InterfaceResidue) for r in binder + target)


def test_records_carry_the_burial_that_decided_them(tmp_path):
    binder, _ = interface_residues(complex_pdb(tmp_path, gap=4.0), ["B"], ["A"])
    assert all(r.dsasa >= 0.0 for r in binder)
    assert any(r.dsasa > 0.0 for r in binder), "an interface residue buries something"
    assert all(r.min_interchain_dist >= 0.0 for r in binder)


def test_sequence_indices_do_not_assume_contiguous_numbering(tmp_path):
    """The old form computed `res_id - min(res_id)` and used it to index a
    sequence string, so a gap in numbering shifted every count. seq_index comes
    from the classifier and is a real position within the chain."""
    from_one = complex_pdb(tmp_path, gap=4.0, name="a.pdb", first_resseq=1)
    from_501 = complex_pdb(tmp_path, gap=4.0, name="b.pdb", first_resseq=501)
    a, _ = interface_residues(from_one, ["B"], ["A"])
    b, _ = interface_residues(from_501, ["B"], ["A"])
    assert sequence_indices(a) == sequence_indices(b), "numbering offset must not move sequence positions"
    assert resseqs(a) != resseqs(b), "but the PDB numbers themselves differ"
    assert min(sequence_indices(a)) >= 0, "0-based, for indexing a sequence string"


def test_a_chain_named_as_both_sides_raises(tmp_path):
    with pytest.raises(InterfaceError, match="both binder and target"):
        interface_residues(complex_pdb(tmp_path), ["B"], ["B"])


@pytest.mark.parametrize("binder,target", [([], ["A"]), (["B"], [])])
def test_a_missing_side_raises(tmp_path, binder, target):
    with pytest.raises(InterfaceError, match="need both binder and target"):
        interface_residues(complex_pdb(tmp_path), binder, target)


def test_an_unreadable_structure_raises_rather_than_reporting_no_interface(tmp_path):
    """No interface residues is a real measurement about a binder that missed;
    it must not also be what a failure looks like."""
    bad = tmp_path / "nope.pdb"
    with pytest.raises(InterfaceError):
        interface_residues(str(bad), ["B"], ["A"])


def test_an_unfamiliar_residue_of_standard_atoms_is_fine(tmp_path):
    """The radius table matches on atom name with an element fallback, so a
    nonstandard residue name is not by itself a problem."""
    p = complex_pdb(tmp_path, gap=4.0, resname="XYZ")
    binder, _ = interface_residues(p, ["B"], ["A"])
    assert binder


def test_atoms_without_a_radius_raise(tmp_path):
    """protein-interface gives an unrecognised atom radius 0.0 and returns early,
    so its burial silently vanishes -- and burial is half of the strict rule.
    Measured to fire on exactly the cases that matter: FE in HEM, and every
    nucleic-acid atom (C1', O5')."""
    import protein_interface as pi

    assert pi.unknown_sasa_radius_atoms(["FE"], ["HEM"]), "a metal cofactor has no radius"
    assert pi.unknown_sasa_radius_atoms(["C1'", "O5'"], ["DA", "DA"]), "nor does a nucleotide"

    lines, serial = [], 1
    for ci, chain in enumerate(("A", "B")):
        for res in range(3):
            for atom, x, y, z in ATOMS:
                name = "ZZ1" if (chain == "B" and atom == "CB") else atom
                lines.append(
                    f"ATOM  {serial:5d}  {name:<3s} ALA {chain}{1 + res:4d}    "
                    f"{x + res * 3.6:8.3f}{y + ci * 4.0:8.3f}{z:8.3f}  1.00 50.00"
                    f"          {name[0]:>2s}  "
                )
                serial += 1
    p = tmp_path / "unknown_atom.pdb"
    p.write_text("\n".join(lines) + "\nTER\nEND\n")
    with pytest.raises(InterfaceError, match="no radius"):
        interface_residues(str(p), ["B"], ["A"])


def test_a_multi_chain_target_needs_no_special_case(tmp_path):
    """The all-atom implementation took one chain id and raised KeyError on the
    comma-joined form its own callers built."""
    lines, serial = [], 1
    for ci, chain in enumerate(("A", "C", "B")):
        for res in range(3):
            for atom, x, y, z in ATOMS:
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} ALA {chain}{1 + res:4d}    "
                    f"{x + res * 3.6:8.3f}{y + ci * 4.0:8.3f}{z:8.3f}  1.00 50.00"
                    f"          {atom[0]:>2s}  "
                )
                serial += 1
    p = tmp_path / "multi.pdb"
    p.write_text("\n".join(lines) + "\nTER\nEND\n")
    binder, target = interface_residues(str(p), ["B"], ["A", "C"])
    assert {r.chain for r in target} <= {"A", "C"}
    assert {r.chain for r in binder} == {"B"}
