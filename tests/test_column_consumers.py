"""Nothing may build a pre-rename column name.

The rename moved every emitted column into the slot scheme
(``{seq}_{kind}_{backend}_{metric}``). Producers and threshold builders were
migrated with it; four consumers that built their column names as f-string
literals were not, and none of them failed loudly:

  * ``refolded_structure_utils`` looked for ``{seq}_complex_pdb_path`` among four
    candidate names, three of which never existed in any frame. Finding nothing
    was logged at debug level and looked exactly like "this design has no
    refold", so a campaign that asked for interface metrics on its refolded
    structures got no such columns at all, silently;
  * ``binder_analysis`` aggregated the same absent column behind
    ``if col in df.columns`` and then reported ``_res_*_original_samples`` and
    ``_res_*_total_redesigns`` as ``0`` for every run;
  * the two motif equivalents would have done the same on a motif run.

The suite passed throughout. A test per consumer would not have helped -- the
next consumer written the same way is the one that breaks -- so this checks the
property directly, across the whole package, by asking the naming module whether
each literal is already migrated.
"""

import ast
import pathlib
import re

from proteinfoundation.metrics.column_names import rename

SRC = pathlib.Path(__file__).resolve().parents[1] / "src/proteinfoundation"

# A literal like "{seq_type}_complex_pdb_path": a sequence-type placeholder, then
# the rest of a column name.
TEMPLATED = re.compile(r"^\{(?:seq_type|seq|t|sequence_type)\}_([A-Za-z0-9_]+)$")

# Emitted per-run under their pre-slot names on purpose: the row builder renames
# the metric columns at the emission boundary and leaves these two, and
# `migrate_frame` maps them when the pooled frame is read. Consumers of a
# *per-run* CSV therefore correctly ask for the legacy name. Listed rather than
# pattern-matched so adding a third such column is a deliberate act.
DELIBERATELY_LEGACY = {"aa_counts", "aa_interface_counts"}


def _templated_literals(tree: ast.AST):
    """Every f-string in the tree that reads as a templated column name.

    Skips literals that are an argument to ``rename`` -- there the pre-rename
    name is the input, which is the correct way to build one -- and literals
    ending in ``_``, which are prefix tests rather than column names.
    """
    renamed_args = {
        id(arg)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "rename"
        for arg in node.args
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr) or id(node) in renamed_args:
            continue
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                parts.append("{" + (value.value.id if isinstance(value.value, ast.Name) else "?") + "}")
        literal = "".join(parts)
        if literal.endswith("_"):
            continue
        match = TEMPLATED.match(literal)
        if match and match.group(1) not in DELIBERATELY_LEGACY:
            yield node.lineno, literal, match.group(1)


def test_no_source_file_builds_a_pre_rename_column_name():
    stale = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for lineno, literal, suffix in _templated_literals(tree):
            probe = f"self_{suffix}"
            if rename(probe) != probe:
                stale.append(f"{path.relative_to(SRC.parent.parent)}:{lineno} builds {literal!r}, which the "
                             f"naming module maps to {rename(probe).replace('self_', '{seq_type}_', 1)!r}")
    assert not stale, "pre-rename column names:\n  " + "\n  ".join(stale)


def test_the_check_can_actually_see_a_stale_name():
    """A guard that passes because its detector is broken is worse than none."""
    tree = ast.parse('col = f"{seq_type}_complex_pdb_path"')
    found = list(_templated_literals(tree))
    assert found and found[0][2] == "complex_pdb_path"
    assert rename("self_complex_pdb_path") == "self_complex_af2_pdb_path"


def test_a_name_built_through_rename_is_not_flagged():
    tree = ast.parse('col = rename(f"{seq_type}_complex_pdb_path", backend)')
    assert not list(_templated_literals(tree))


# ---------------------------------------------------------------------------
# A column emitted more than once.
#
# `complex_folding_backend` was appended to the column list inside the
# per-sequence-type loop under `if idx == 0` -- true once per type, not once per
# run -- so every per-job CSV of a two-type run carried it twice. It stayed inert
# for months because the analyze stage reads it from CSV, and pandas mangles
# duplicate headers on read into `name` and `name.1`. The first code to select
# the duplicated label on an in-memory frame got a DataFrame instead of a Series
# and died inside pandas, naming neither the column nor the duplication, taking
# a GPU evaluate stage with it.
#
# The literal-name guard above cannot see this class: every name involved is
# correctly built. What is wrong is how many times it is emitted.
# ---------------------------------------------------------------------------


def test_duplicates_are_dropped_in_order_and_reported(caplog):
    from proteinfoundation.evaluation.binder_eval_utils import dedupe_columns

    assert dedupe_columns(["a", "b", "a", "c", "b"]) == ["a", "b", "c"], "first occurrence wins"
    assert dedupe_columns(["a", "b"]) == ["a", "b"], "an already-clean list is untouched"


def test_a_clean_list_reports_nothing(capsys):
    """The error is the point of the helper; it must not cry wolf."""
    import io

    from loguru import logger

    from proteinfoundation.evaluation.binder_eval_utils import dedupe_columns

    sink = io.StringIO()
    handle = logger.add(sink, level="ERROR")
    try:
        dedupe_columns(["a", "b", "c"])
        assert sink.getvalue() == ""
        dedupe_columns(["a", "a"], "Binder results")
        message = sink.getvalue()
    finally:
        logger.remove(handle)
    assert "Binder results" in message and "'a'" in message
    assert "DataFrame rather than a Series" in message, "the message says why it matters"


def test_the_real_scenario_survives_the_helper():
    """The exact frame that crashed: a provenance column named once per sequence
    type, reindexed onto, then read as a Series."""
    import pandas as pd

    from proteinfoundation.evaluation.binder_eval_utils import dedupe_columns
    from proteinfoundation.result_analysis.binder_analysis_utils import (
        COMPLEX_BACKEND_COLUMN,
        complex_backend_of,
    )

    rows = [{"x": 1, COMPLEX_BACKEND_COLUMN: "af2"}]
    naive = pd.DataFrame(rows).reindex(columns=["x", COMPLEX_BACKEND_COLUMN, COMPLEX_BACKEND_COLUMN])
    assert naive[COMPLEX_BACKEND_COLUMN].ndim == 2, "the fixture reproduces the duplication"

    guarded = pd.DataFrame(rows).reindex(
        columns=dedupe_columns(["x", COMPLEX_BACKEND_COLUMN, COMPLEX_BACKEND_COLUMN])
    )
    assert guarded[COMPLEX_BACKEND_COLUMN].ndim == 1
    assert complex_backend_of(guarded) == "af2"


def test_every_result_frame_is_built_through_the_guard():
    """The class-level check. Any future `reindex(columns=...)` that accumulates
    its column list is the same bug waiting to happen, and there are five such
    frames across the evaluation modules."""
    guarded, unguarded = [], []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "reindex"):
                continue
            for kw in node.keywords:
                if kw.arg != "columns":
                    continue
                where = f"{path.relative_to(SRC.parent.parent)}:{node.lineno}"
                through = isinstance(kw.value, ast.Call) and getattr(kw.value.func, "id", None) == "dedupe_columns"
                (guarded if through else unguarded).append(where)
    assert not unguarded, "reindex(columns=...) not routed through dedupe_columns:\n  " + "\n  ".join(unguarded)
    assert len(guarded) >= 5, f"expected the five known result frames, found {guarded}"
