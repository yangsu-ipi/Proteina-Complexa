"""The shape complementarity fallback.

_calculate_shape_complementarity used to return a hardcoded 0.70 whenever sc-rs
was missing, timed out, or produced something unparseable, announcing it only on
stdout. 0.70 is a good SC -- a well-packed interface -- so a broken binary wrote
a column of plausible passes across a whole campaign. These pin that every
failure path raises instead.
"""

import json
import stat

import pytest

Bio = pytest.importorskip("Bio", reason="pr_alternative_utils imports Bio.PDB")

from proteinfoundation.utils.pr_alternative_utils import (
    ShapeComplementarityError,
    _calculate_shape_complementarity,
)


def fake_sc(tmp_path, body: str, name: str = "sc"):
    """A stand-in sc-rs binary, so the failure modes are exercised for real
    rather than through a mock of subprocess."""
    script = tmp_path / name
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def pdb(tmp_path):
    path = tmp_path / "complex.pdb"
    path.write_text("ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00\n")
    return str(path)


def test_a_measured_value_is_returned(tmp_path, pdb):
    """The success path still works, and is what the raises are protecting."""
    sc_bin = fake_sc(tmp_path, 'echo \'{"sc": 0.612}\'')
    assert _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin) == pytest.approx(0.612)


def test_a_value_embedded_in_chatter_is_still_read(tmp_path, pdb):
    """sc-rs prints progress before its JSON; the lenient parse is deliberate."""
    sc_bin = fake_sc(tmp_path, 'echo "loading..."; echo \'{"sc_value": 0.5}\'')
    assert _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin) == pytest.approx(0.5)


def test_no_binary_configured_raises(pdb):
    with pytest.raises(ShapeComplementarityError, match="SC_EXEC"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=None)


def test_a_missing_binary_raises(tmp_path, pdb):
    with pytest.raises(ShapeComplementarityError, match="could not run"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=str(tmp_path / "not_here"))


def test_a_failing_binary_raises(tmp_path, pdb):
    sc_bin = fake_sc(tmp_path, 'echo "boom" >&2; exit 3')
    with pytest.raises(ShapeComplementarityError, match="exited 3"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


def test_empty_output_raises(tmp_path, pdb):
    sc_bin = fake_sc(tmp_path, "exit 0")
    with pytest.raises(ShapeComplementarityError, match="no output"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


def test_unparseable_output_raises(tmp_path, pdb):
    sc_bin = fake_sc(tmp_path, 'echo "segmentation fault"')
    with pytest.raises(ShapeComplementarityError, match="not JSON"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


def test_json_without_an_sc_field_raises(tmp_path, pdb):
    sc_bin = fake_sc(tmp_path, 'echo \'{"area": 812.0}\'')
    with pytest.raises(ShapeComplementarityError, match="no SC field"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


def test_a_non_numeric_sc_raises(tmp_path, pdb):
    sc_bin = fake_sc(tmp_path, 'echo \'{"sc": "n/a"}\'')
    with pytest.raises(ShapeComplementarityError, match="non-numeric"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_an_out_of_range_sc_raises(tmp_path, pdb, value):
    """SC is a fraction. Out of range means sc-rs reported something that is not
    one, which previously fell through to the placeholder."""
    sc_bin = fake_sc(tmp_path, f"echo '{json.dumps({'sc': value})}'")
    with pytest.raises(ShapeComplementarityError, match=r"outside \[0, 1\]"):
        _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


def test_no_failure_path_returns_the_old_placeholder(tmp_path, pdb):
    """The point of the change, stated as one assertion: nothing that goes wrong
    yields a number, least of all a good-looking one."""
    broken = [
        "exit 0",                       # empty
        'echo "boom" >&2; exit 3',      # failure
        'echo "not json"',              # unparseable
        'echo \'{"area": 1.0}\'',       # wrong field
        'echo \'{"sc": 2.0}\'',         # out of range
    ]
    for i, body in enumerate(broken):
        sc_bin = fake_sc(tmp_path, body, name=f"sc_{i}")
        with pytest.raises(ShapeComplementarityError):
            _calculate_shape_complementarity(pdb, "B", "A", sc_bin=sc_bin)


def test_the_evaluation_caller_records_nan_rather_than_a_number(tmp_path, pdb, monkeypatch):
    """Evaluation should survive a missing sc-rs, but the column must say
    'not measured' -- which is what NaN says and 0.70 did not."""
    np = pytest.importorskip("numpy")
    binder_eval = pytest.importorskip(
        "proteinfoundation.evaluation.binder_eval",
        reason="the evaluation stack is not importable here",
    )
    got = binder_eval.compute_bioinformatics_metrics_single(
        pdb, binder_chain="B", target_chain="A", sc_bin=str(tmp_path / "not_here")
    )
    assert all(np.isnan(v) for v in got.values()), got
