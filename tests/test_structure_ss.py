"""Eight-state secondary structure, packed and collapsed late.

The retired _res_ss_* columns came from biotite's P-SEA, which over-called beta
by four times on this campaign's binders (877 residues claimed, 178 confirmed by
mkdssp). These cover the replacement's packing, its collapse rule, and the
reduction choice at the interface, where the denominator varies per model.
"""

import pytest

from proteinfoundation.metrics.structure_ss import (
    SS_COARSE,
    SS_STATE_ORDER,
    collapse_counts,
    counts_from_states,
    ss_provenance,
    unpack_counts,
)


def test_the_state_order_is_the_data_format():
    """A packed column is unpacked by position, so the order is not cosmetic."""
    assert SS_STATE_ORDER == ("H", "G", "I", "E", "B", "T", "S", "-")


def test_counts_are_packed_in_that_order():
    counts = counts_from_states(["H", "H", "E", "T", "-"])
    assert counts == [2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


@pytest.mark.parametrize("coil", [" ", "", "  "])
def test_every_spelling_of_coil_counts_as_coil(coil):
    """mdtraj writes a space; the biotite-era code wrote an empty string."""
    assert counts_from_states([coil]) == counts_from_states(["-"])


def test_an_unknown_state_is_coil_rather_than_dropped():
    """DSSP 4 emits P for polyproline-II, which classic DSSP had no state for.
    Counting it as coil keeps the total equal to the residue count, so a reader
    can trust ss_total."""
    counts = counts_from_states(["H", "P", "Z"])
    assert sum(counts) == 3
    assert counts[SS_STATE_ORDER.index("-")] == 2


def test_the_collapse_rule_puts_an_isolated_bridge_in_loop():
    """calc_ss_percentage's rule, which this repo has always used: H/G/I helix,
    only E sheet. B is a bridge, not a strand."""
    assert SS_COARSE["H"] == SS_COARSE["G"] == SS_COARSE["I"] == "helix"
    assert SS_COARSE["E"] == "sheet"
    assert "B" not in SS_COARSE, "B falls through to loop"
    got = collapse_counts(counts_from_states(["B", "B", "E", "H"]))
    assert got == {"helix": 0.25, "sheet": 0.25, "loop": 0.5}


def test_fractions_are_derived_not_stored():
    got = collapse_counts([3, 1, 0, 4, 0, 1, 0, 1])
    assert got["helix"] == pytest.approx(4 / 10)
    assert got["sheet"] == pytest.approx(4 / 10)
    assert got["loop"] == pytest.approx(2 / 10)
    assert sum(got.values()) == pytest.approx(1.0)


def test_counts_survive_the_csv_round_trip():
    """The column is written by pandas as the text of a list; counting its
    characters is a bug this codebase has already shipped once."""
    packed = counts_from_states(["H"] * 34 + ["T"] * 6 + ["S"] * 2 + ["-"] * 5)
    assert unpack_counts(str(packed)) == packed
    assert collapse_counts(str(packed)) == collapse_counts(packed)


@pytest.mark.parametrize("bad", ["", "not a list", "[1, 2, 3]", None, 7, "[]"])
def test_a_malformed_count_reads_as_not_measured(bad):
    """Zeros would look like a structure with no secondary structure at all."""
    assert unpack_counts(bad) is None
    assert collapse_counts(bad) == {}


def test_an_empty_interface_has_no_composition():
    """A binder that touches nothing has no interface SS. Zero fractions would
    read as a measured all-loop interface."""
    assert collapse_counts(counts_from_states([])) == {}


def test_the_interface_reduction_is_pooled_not_paired():
    """Two models, one presenting 4 interface residues and one 16. Pooled weights
    by interface size; paired would give the 4-residue model equal say on a
    denominator small enough to be noise. Pooled is what collapsing mean counts
    does, and this pins that it is not accidentally the other one."""
    model_a = counts_from_states(["H"] * 4)                  # 4 residues, all helix
    model_b = counts_from_states(["-"] * 16)                 # 16 residues, all loop
    mean_counts = [(a + b) / 2 for a, b in zip(model_a, model_b, strict=True)]
    pooled = collapse_counts(mean_counts)
    paired = {
        k: (collapse_counts(model_a)[k] + collapse_counts(model_b)[k]) / 2
        for k in ("helix", "sheet", "loop")
    }
    assert pooled["helix"] == pytest.approx(4 / 20)
    assert paired["helix"] == pytest.approx(0.5)
    assert pooled != paired, "the two reductions differ, so the choice has to be deliberate"


def test_the_provenance_names_everything_that_moves_a_number():
    got = ss_provenance()
    assert got["engine"] == "mdtraj"
    assert got["states"] == list(SS_STATE_ORDER)
    assert got["collapse"]["B"] == "loop"
    assert "version" in got


def test_the_interface_is_decided_elsewhere():
    """This module does not define an interface; it reports SS over whichever
    residues metrics/interface.py named. Two definitions here is what the
    unification removed."""
    import inspect

    from proteinfoundation.metrics import structure_ss

    assert not hasattr(structure_ss, "SS_INTERFACE_CUTOFF")
    assert "interface_cutoff" not in ss_provenance()
    assert "interface_resseqs" in inspect.signature(structure_ss.structure_ss).parameters


def test_states_are_read_from_the_complex_not_the_isolated_chain(tmp_path):
    """DSSP's hydrogen bonds cross the interface: 39 of 150 campaign complexes
    hold binder strands that exist only in the target's presence. chain_states
    takes a chain out of a folded complex rather than folding a chain."""
    pytest.importorskip("mdtraj", reason="the SS engine")
    import inspect

    from proteinfoundation.metrics import structure_ss

    src = inspect.getsource(structure_ss.chain_states)
    assert "md.load(pdb_path)" in src and "compute_dssp" in src
    assert "chain_id" in src, "the chain is selected after folding, not before"
