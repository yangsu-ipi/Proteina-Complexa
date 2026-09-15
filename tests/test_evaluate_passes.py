"""What each evaluate pass turns off, and what it refuses to write.

These live apart from the folding stack on purpose -- see the module docstring
of ``proteinfoundation.evaluation.evaluate_passes``. Every question here is about
a dict or a DataFrame, and none of them should need a GPU to answer.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from proteinfoundation.evaluation.evaluate_passes import (
    PASS_ESM,
    PASS_FINAL,
    PASS_FOLD,
    SUPPRESSED_BY_PASS,
    UnknownEvaluatePass,
    apply_pass,
    resolve_pass,
    save_results_csv,
    writes_run_level_output,
)


def test_a_fold_pass_turns_off_esm_and_both_derived_families():
    """ESMC-6B because it would sit on the card beside the folder. The derived
    metrics because a fold pass throws their answers away and the final pass
    recomputes them -- there is no cache behind either family."""
    cfg = {
        "compute_esm_metrics": True,
        "compute_pre_refolding_metrics": True,
        "compute_refolded_structure_metrics": True,
        "folding_models": ["af2"],
    }
    assert sorted(apply_pass(cfg, PASS_FOLD)) == [
        "compute_esm_metrics",
        "compute_pre_refolding_metrics",
        "compute_refolded_structure_metrics",
    ]
    assert not any(cfg[k] for k in SUPPRESSED_BY_PASS[PASS_FOLD])


def test_the_esm_pass_keeps_esm_and_drops_the_derived_families():
    """Its whole reason to exist: run ESMC-6B with nothing else on the card, so
    the final pass has TMOL to itself."""
    cfg = {
        "compute_esm_metrics": True,
        "compute_pre_refolding_metrics": True,
        "compute_refolded_structure_metrics": True,
    }
    turned_off = apply_pass(cfg, PASS_ESM)
    assert "compute_esm_metrics" not in turned_off
    assert cfg["compute_esm_metrics"] is True
    assert cfg["compute_pre_refolding_metrics"] is False
    assert cfg["compute_refolded_structure_metrics"] is False


def test_the_final_pass_turns_nothing_off():
    cfg = {"compute_esm_metrics": True, "compute_refolded_structure_metrics": True}
    assert apply_pass(cfg, PASS_FINAL) == []
    assert cfg == {"compute_esm_metrics": True, "compute_refolded_structure_metrics": True}


def test_suppressing_the_derived_families_is_what_suppresses_tmol():
    """TmolRewardModel defaults to torch.device("cuda"), so TMOL is a GPU tenant
    and must not be in a pass that exists to leave the card to something else.
    It needs no entry of its own: derive_consensus_tmol is
    `compute_refolded_structure_metrics and refolded.tmol`, and the other TMOL
    site is inside the pre-refolding metrics."""
    for family in ("compute_pre_refolding_metrics", "compute_refolded_structure_metrics"):
        assert family in SUPPRESSED_BY_PASS[PASS_FOLD]
        assert family in SUPPRESSED_BY_PASS[PASS_ESM]


def test_it_reports_what_it_changed_not_what_it_asked_for():
    """The returned list is logged as the pass's record of itself. A key already
    off, or absent, did not change and must not be claimed."""
    assert apply_pass({"compute_esm_metrics": False}, PASS_FOLD) == []
    assert apply_pass({}, PASS_FOLD) == []


def test_it_does_not_add_keys_a_struct_config_would_refuse():
    """OmegaConf under struct mode raises on an unknown key. Only keys already
    present and truthy are written, so this is safe on a real Hydra config."""
    cfg = {}
    apply_pass(cfg, PASS_FOLD)
    assert cfg == {}


def test_an_unknown_pass_name_is_refused_not_defaulted():
    """Falling back to 'final' would put a one-folder CSV in the output directory
    under the name the finished run uses."""
    assert resolve_pass({}) == PASS_FINAL
    assert resolve_pass({"evaluate_pass": "fold"}) == PASS_FOLD
    with pytest.raises(UnknownEvaluatePass):
        resolve_pass({"evaluate_pass": "folds"})


# ----------------------------- run-level artifacts


def test_only_the_final_pass_writes_run_level_output():
    assert writes_run_level_output(PASS_FINAL, "anything") is True
    assert writes_run_level_output(PASS_FOLD, "anything") is False
    assert writes_run_level_output(PASS_ESM, "anything") is False


FRAME = pd.DataFrame({"id_gen": [0, 1], "mpnn_complex_af2_binder_pae": [3.1, 4.2], "dryrun": [False, False]})


def test_a_non_final_pass_writes_no_csv(tmp_path):
    """A pass's rows carry a subset of the columns, and nothing downstream --
    analyze, verify_run_outputs, analyze_pooled -- can tell that from a finished
    run."""
    for kind in (PASS_FOLD, PASS_ESM):
        save_results_csv(FRAME, str(tmp_path), "binder", "cfg", 0, evaluate_pass=kind)
    assert list(tmp_path.iterdir()) == []


def test_the_final_pass_writes_one(tmp_path):
    save_results_csv(FRAME, str(tmp_path), "binder", "cfg", 0, evaluate_pass=PASS_FINAL)
    written = list(tmp_path.iterdir())
    assert [p.name for p in written] == ["binder_results_cfg_0.csv"]
    assert len(pd.read_csv(written[0])) == 2


@pytest.mark.parametrize("kind", [PASS_FOLD, PASS_ESM, PASS_FINAL])
def test_the_rows_come_back_either_way(tmp_path, kind):
    """The caller counts samples off this frame for the run summary, and every
    pass still built its rows -- reading its own caches back is what proves they
    are readable before the final pass depends on them."""
    out = save_results_csv(FRAME, str(tmp_path), "binder", "cfg", 0, evaluate_pass=kind)
    assert len(out) == 2
    assert "dryrun" not in out.columns


EVALUATE = Path(__file__).resolve().parents[1] / "src" / "proteinfoundation" / "evaluate.py"

# Gates itself, so it is reached unconditionally on purpose: it still builds and
# returns the rows, because reading its own caches back is what proves they are
# readable before the final pass depends on them.
SELF_GATING = {"save_results_csv"}


def _guards_of(tree):
    """Every node, paired with the `if` tests enclosing it."""
    out = {}

    def walk(node, guards):
        for child in ast.iter_child_nodes(node):
            inner = guards
            if isinstance(node, ast.If) and child in node.body:
                inner = guards + [ast.unparse(node.test)]
            out[child] = inner
            walk(child, inner)

    walk(tree, [])
    return out


def test_every_run_level_write_in_evaluate_is_gated():
    """The structural check, and the reason this file exists.

    Three artifacts are written once per evaluate PROCESS rather than once per
    design -- the results CSVs, the success-criteria JSON, the timing rows. With
    one fused evaluate that distinction did not exist. With six passes, five of
    which know about a subset of the work, an ungated one means the last pass to
    touch it wins: the run's recorded evaluation time becomes the final pass's
    minutes, and its success criteria name one backend's columns.

    Two of them were ungated in the first version of this change. A fourth
    run-level write added later would be ungated the same way, and this is what
    should notice.
    """
    tree = ast.parse(EVALUATE.read_text())
    guards = _guards_of(tree)

    unguarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        writes = name == "open" and any(
            isinstance(a, ast.Constant) and a.value == "w" for a in node.args
        )
        writes = writes or (name or "").startswith("save_")
        if not writes or name in SELF_GATING:
            continue
        if not any("writes_run_level_output" in g for g in guards.get(node, [])):
            unguarded.append(f"{name} at line {node.lineno}")

    assert not unguarded, (
        "run-level writes reachable from a non-final pass: "
        + ", ".join(unguarded)
        + " -- wrap each in `if writes_run_level_output(evaluate_pass, ...)`"
    )


def test_the_redesign_pass_emits_no_rows():
    """It short-circuits BEFORE the row is built, not after.

    After was the bug, and it survived every test here because no test and no
    campaign had ever run this pass end to end: EFNB3 finished on the fused path.
    Appending a row and then skipping the loop left the frame with one row per
    design and every metric list empty, so `df[metric] = metrics[metric]` raised
    "Length of values (0) does not match length of index (2)" -- the first time
    the pass ran on a real campaign, in its first minute.

    Emitting nothing is correct rather than a workaround: this pass writes no
    run-level output, and the rows a fold pass builds exist to prove its caches
    read back. There is no fold here to read.
    """
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src/proteinfoundation/evaluation/monomer_eval.py"
    ).read_text()
    loop = source[source.index("def compute_monomer_metrics(") :]
    short_circuit = loop.index('if str(cfg_metric.get("evaluate_pass", "final")) == "redesign":')
    row_append = loop.index("results.append(row_dict)")
    assert short_circuit < row_append, (
        "the redesign pass must skip the design before a row is appended for it, "
        "or the frame carries rows whose metric columns were never filled"
    )


def test_the_redesign_pass_turns_off_the_folding_tracks():
    """It runs the inverse folder and nothing else, and has to SAY so.

    With no track it inherited the campaign's own metric flags, so the binder
    loop ran in full beside it: EFNB3's first split run put 73.5 GB of AF2
    weights on each card during the one pass whose point is that no folder is
    resident. The folds were not wrong -- a later pass would have found them
    cached -- they were just paid for in the pass built to avoid paying.
    """
    from proteinfoundation.evaluation.evaluate_passes import (
        PASS_REDESIGN,
        evaluate_pass_plan,
        pass_overrides,
    )

    plan = evaluate_pass_plan(["af2", "esmfold2"])
    redesign = next(p for p in plan if p["kind"] == PASS_REDESIGN)
    overrides = pass_overrides(redesign)
    assert "++metric.compute_binder_metrics=false" in overrides, (
        "the binder loop folds complexes; the redesign pass must not run it"
    )
    assert "++metric.compute_monomer_metrics=true" in overrides, (
        "the redesign short-circuit lives in the monomer loop, so that loop has to run"
    )
    assert not any("folding_models" in o for o in overrides), (
        "it names no folder, because it loads none"
    )
