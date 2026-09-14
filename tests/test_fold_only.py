"""A fold-only pass: what it turns off, and what it refuses to write.

These live apart from the folding stack on purpose -- see the module docstring
of ``proteinfoundation.evaluation.fold_only``. Every question here is about a
dict or a DataFrame, and none of them should need a GPU to answer.
"""

import pandas as pd
import pytest

from proteinfoundation.evaluation.fold_only import (
    FOLD_ONLY_DISABLED_METRICS,
    apply_fold_only,
    save_results_csv,
)


def test_a_fold_only_pass_turns_esm_off():
    """ESMC-6B is the one thing in evaluate that would sit on the card beside
    the folder the pass exists to give it to."""
    cfg = {"compute_esm_metrics": True, "folding_models": ["af2"]}
    assert apply_fold_only(cfg) == ["compute_esm_metrics"]
    assert cfg["compute_esm_metrics"] is False


def test_it_reports_what_it_changed_not_what_it_asked_for():
    """The returned list is logged as the pass's record of itself. A key that was
    already off, or absent, did not change and must not be claimed."""
    assert apply_fold_only({"compute_esm_metrics": False}) == []
    assert apply_fold_only({}) == []


def test_it_leaves_the_structure_derived_metrics_alone():
    """Turning these off would change the consensus DERIVATION fingerprint, and
    the cached structures would read as under-derived to the pass that wants
    them -- a re-read rather than a refold, but an unexplained one."""
    cfg = {
        "compute_esm_metrics": True,
        "compute_pre_refolding_metrics": True,
        "compute_refolded_structure_metrics": True,
    }
    apply_fold_only(cfg)
    assert cfg["compute_pre_refolding_metrics"] is True
    assert cfg["compute_refolded_structure_metrics"] is True
    assert "compute_refolded_structure_metrics" not in FOLD_ONLY_DISABLED_METRICS


def test_it_does_not_add_keys_a_struct_config_would_refuse():
    """OmegaConf under struct mode raises on an unknown key. Only keys already
    present and truthy are written, so this is safe on a real Hydra config."""
    cfg = {}
    apply_fold_only(cfg)
    assert cfg == {}


FRAME = pd.DataFrame({"id_gen": [0, 1], "mpnn_complex_af2_binder_pae": [3.1, 4.2], "dryrun": [False, False]})


def test_a_fold_only_pass_writes_no_csv(tmp_path):
    """A one-folder CSV carries a subset of the columns, and nothing downstream
    -- analyze, verify_run_outputs, analyze_pooled -- can tell it from a
    finished one."""
    save_results_csv(FRAME, str(tmp_path), "binder", "cfg", 0, fold_only=True)
    assert list(tmp_path.iterdir()) == []


def test_a_normal_pass_writes_one(tmp_path):
    save_results_csv(FRAME, str(tmp_path), "binder", "cfg", 0, fold_only=False)
    written = list(tmp_path.iterdir())
    assert [p.name for p in written] == ["binder_results_cfg_0.csv"]
    assert len(pd.read_csv(written[0])) == 2


@pytest.mark.parametrize("fold_only", [True, False])
def test_the_rows_come_back_either_way(tmp_path, fold_only):
    """The caller counts samples off this frame for the run summary, and a
    fold-only pass still built its rows -- reading its own caches back is what
    proves they are readable before the final pass depends on them."""
    out = save_results_csv(FRAME, str(tmp_path), "binder", "cfg", 0, fold_only=fold_only)
    assert len(out) == 2
    # Config/metadata columns are dropped on both paths, so the discarded frame
    # is the same frame the written one would have been.
    assert "dryrun" not in out.columns
