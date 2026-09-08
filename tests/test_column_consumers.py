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
