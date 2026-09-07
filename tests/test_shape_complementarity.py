"""Shape complementarity, in process.

This used to shell out to an sc-rs binary located through SC_EXEC, and returned
a hardcoded 0.70 whenever that binary was missing, failing, timing out or
producing something unparseable. 0.70 reads as a well-packed interface, so a
binary that never ran wrote a column of plausible passes across a whole
campaign.

It is now the same sc-rs -- the copy vendored in protein-interface -- called in
process. Same algorithm, same CCP4-sc radii, no path to resolve and no value
that can be returned when nothing was measured.
"""

import pytest

pytest.importorskip("Bio", reason="pr_alternative_utils imports Bio.PDB")
pytest.importorskip("protein_interface", reason="the shape complementarity engine")

from proteinfoundation.utils import pr_alternative_utils as pau
from proteinfoundation.utils.pr_alternative_utils import (
    ShapeComplementarityError,
    _calculate_shape_complementarity,
)

ATOMS = [("N", 0.0, 0.0, 0.0), ("CA", 1.46, 0.0, 0.0), ("C", 2.0, 1.42, 0.0),
         ("O", 1.3, 2.4, 0.0), ("CB", 2.0, -0.77, -1.2)]


@pytest.fixture
def pdb(tmp_path):
    lines, serial = [], 1
    for ci, chain in enumerate(("A", "B")):
        for res in range(3):
            for atom, x, y, z in ATOMS:
                lines.append(
                    f"ATOM  {serial:5d}  {atom:<3s} ALA {chain}{1 + res:4d}    "
                    f"{x + res * 3.6:8.3f}{y + ci * 4.0:8.3f}{z:8.3f}  1.00 50.00"
                    f"          {atom[0]:>2s}  "
                )
                serial += 1
    path = tmp_path / "complex.pdb"
    path.write_text("\n".join(lines) + "\nTER\nEND\n")
    return str(path)


def test_a_measured_value_comes_back(pdb):
    got = _calculate_shape_complementarity(pdb, "B", "A")
    assert 0.0 <= got <= 1.0
    assert got != 0.70, "the placeholder this replaced"


def test_there_is_no_binary_to_locate():
    """SC_EXEC, resolve_sc_bin and DEFAULT_SC_EXEC existed only to find a
    subprocess. Their absence is the point of the change."""
    for gone in ("resolve_sc_bin", "DEFAULT_SC_EXEC"):
        assert not hasattr(pau, gone), f"{gone} should have gone with the subprocess"
    import inspect

    assert "sc_bin" not in inspect.signature(pau.pr_alternative_score_interface).parameters


def test_a_missing_structure_raises(tmp_path):
    with pytest.raises(ShapeComplementarityError, match="failed"):
        _calculate_shape_complementarity(str(tmp_path / "nope.pdb"), "B", "A")


def test_a_missing_chain_raises(pdb):
    with pytest.raises(ShapeComplementarityError, match="failed"):
        _calculate_shape_complementarity(pdb, "Z", "A")


@pytest.mark.parametrize("binder,target", [("B", ""), ("", "A")])
def test_a_missing_side_raises(pdb, binder, target):
    with pytest.raises(ShapeComplementarityError, match="need both binder and target"):
        _calculate_shape_complementarity(pdb, binder, target)


def test_a_multi_chain_target_is_split(pdb, monkeypatch):
    """target_chain arrives comma-joined from binder_eval."""
    seen = {}

    class _R:
        sc = 0.5

    def fake(path, chains_a, chains_b, **kw):
        seen["a"], seen["b"] = chains_a, chains_b
        return _R()

    import protein_interface as pi

    monkeypatch.setattr(pi, "from_pdb", fake)
    _calculate_shape_complementarity(pdb, "B", "A,C")
    assert seen["a"] == ["A", "C"] and seen["b"] == ["B"]


def test_an_out_of_range_value_raises(pdb, monkeypatch):
    """SC is a fraction. Out of range means the engine reported something that is
    not one, which must not reach a column."""
    import protein_interface as pi

    class _R:
        sc = 2.0

    monkeypatch.setattr(pi, "from_pdb", lambda *a, **k: _R())
    with pytest.raises(ShapeComplementarityError, match=r"outside \[0, 1\]"):
        _calculate_shape_complementarity(pdb, "B", "A")


def test_no_failure_path_returns_a_number(pdb, tmp_path, monkeypatch):
    """The point of the change, as one assertion: nothing that goes wrong yields
    a value, least of all a good-looking one."""
    import protein_interface as pi

    cases = [
        lambda: _calculate_shape_complementarity(str(tmp_path / "gone.pdb"), "B", "A"),
        lambda: _calculate_shape_complementarity(pdb, "Z", "A"),
        lambda: _calculate_shape_complementarity(pdb, "B", ""),
    ]
    for case in cases:
        with pytest.raises(ShapeComplementarityError):
            case()

    monkeypatch.setattr(pi, "from_pdb", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(ShapeComplementarityError):
        _calculate_shape_complementarity(pdb, "B", "A")
