"""Result column names, built from slots and never parsed back.

Four schemes coexisted, and none of them said which folding model produced a
number. `{seq}_complex_*` was AF2 only by circumstance; `{seq}_esmfold2_*` put
the backend in the middle; `{seq}_apo_*_{model}` put it at the end; and
`generated_*` / `refolded_{seq}_*` were blanket renames applied in evaluate.py
and binder_eval.py with no slot for a model at all. `refolded_` was the sharpest
case: the structures come from best_paths_dict, whatever refolder filled those
columns, so refolded_self_binder_interface_dSASA is an AF2 number that neither
its name nor the code says is one.

One scheme replaces them::

    {seq_type}_{kind}_{backend}_{scope}_{metric}

See docs/design-notes/column-naming.md for the full migration -- 134 renamed,
3 retired, 141 new, 80 unchanged.

CONSTRUCTION ONLY. These names cannot be parsed back into slots, and nothing
here tries: `binder` + `interface_nres` and `binder_interface` + `nres` render
to the same string. Callers build names from slots and ask the registry what a
column means; anything that regexes a column name to recover its scope is
reading a coincidence. The registry asserts uniqueness at import, so two
definitions colliding on one string is a startup error rather than two metrics
silently sharing a column.
"""

from __future__ import annotations

# Longest-first. "mpnn_fixed" starts with "mpnn", so a shorter-first match reads
# mpnn_fixed_aa_counts as seq_type "mpnn" with metric "fixed_aa_counts".
# analyze.py already has to know this (it lists "mpnn_fixed_" as its own prefix).
SEQ_TYPES: tuple[str, ...] = ("mpnn_fixed", "mpnn", "self")

# The structures a run actually produces. `binder` and `target` were never kinds:
# every {seq}_binder_* column is an RMSD of the binder *within the refolded
# complex*, and target only ever appeared as target_pLDDT. Both are scopes.
KINDS: tuple[str, ...] = ("complex", "apo")

# What produced the structure. `generated` is a backend, not a prefix -- the
# generated structure is a complex too, so it takes the same slots and differs
# only in this one, which is what makes generated-vs-refolded a one-slot diff.
BACKENDS: tuple[str, ...] = ("af2", "esmfold2", "esmfold", "rf3", "protenix", "boltz2", "generated")

# Which part of the structure. Compound values are deliberate: `interface` alone
# means both sides, which is right for an additive quantity like dSASA and wrong
# for a composition, where you want each side separately. Omitted scope means
# the whole structure -- on a complex that is target and binder together.
SCOPES: tuple[str, ...] = ("binder", "target", "interface", "binder_interface", "target_interface")

# Standard field terms, kept as they are. i_pAE, i_pTM and the ipSAE family are
# inherently interfacial and are what every binder paper calls them; giving them
# an explicit scope slot would rename the one set of metrics a reader already
# knows. They take no scope.
GRANDFATHERED_METRICS: frozenset[str] = frozenset(
    {
        "i_pAE",
        "i_pTM",
        "min_ipAE",
        "min_ipSAE",
        "max_ipSAE",
        "avg_ipSAE",
        "min_ipSAE_10",
        "max_ipSAE_10",
        "avg_ipSAE_10",
    }
)

# Bumped when the scheme itself changes, not when a column is added.
COLUMN_SCHEME_VERSION = 1


class ColumnNameError(ValueError):
    """A column name was asked for that the scheme cannot express."""


# metric.binder_folding_method names a model and often a version --
# "rf3_latest", "protenix_v0.4.0". The backend slot takes the family: a column
# name that changed when weights were upgraded would make every campaign
# incomparable with the last, and the resolved config already records exactly
# which version ran. "colabdesign" is the odd one out -- it is the harness, and
# what it runs is AF2.
_FOLDING_METHOD_FAMILIES: tuple[tuple[str, str], ...] = (
    ("colabdesign", "af2"),
    ("af2", "af2"),
    ("rf3", "rf3"),
    ("protenix", "protenix"),
    ("boltz2", "boltz2"),
    ("esmfold2", "esmfold2"),
    ("esmfold", "esmfold"),
)


def backend_for_folding_method(folding_method: str) -> str:
    """The backend slot for a configured folding method.

    Refuses an unrecognised one rather than passing it through: a column named
    after a raw config string would claim provenance the scheme cannot check,
    and the failure would be a new column silently appearing beside the old.
    """
    name = str(folding_method or "").strip().lower()
    for prefix, family in _FOLDING_METHOD_FAMILIES:
        if name == prefix or name.startswith(prefix + "_"):
            return family
    raise ColumnNameError(
        f"no backend slot for folding method {folding_method!r}; add it to "
        f"_FOLDING_METHOD_FAMILIES so its columns say what produced them"
    )


def metric_column(
    metric: str,
    *,
    kind: str,
    backend: str,
    seq_type: str | None = None,
    scope: str | None = None,
    per_sequence: bool = False,
) -> str:
    """Build one result column name from its slots.

    *seq_type* is omitted for ``generated``: one co-designed sequence exists per
    design, so there is no mpnn variant of a generated structure. Leaving the
    slot empty keeps that asymmetry visible -- a generated-vs-mpnn comparison
    pairs ``complex_generated_*`` with ``mpnn_complex_af2_*``, and hiding the
    mismatch is what sequences_for_type exists to prevent.

    *scope* is omitted for a metric describing the whole structure, and refused
    for a grandfathered interfacial one.

    *per_sequence* appends ``_all``, the sibling column holding one entry per
    redesign sequence.
    """
    if not metric:
        raise ColumnNameError("metric is required")
    if kind not in KINDS:
        raise ColumnNameError(f"unknown kind {kind!r}; expected one of {KINDS}")
    if backend not in BACKENDS:
        raise ColumnNameError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "generated":
        if seq_type is not None:
            raise ColumnNameError(
                f"generated structures carry one co-designed sequence, so seq_type {seq_type!r} "
                f"would name a variant that cannot exist"
            )
    elif seq_type not in SEQ_TYPES:
        raise ColumnNameError(f"unknown seq_type {seq_type!r}; expected one of {SEQ_TYPES}")
    if scope is not None and scope not in SCOPES:
        raise ColumnNameError(f"unknown scope {scope!r}; expected one of {SCOPES}")
    if scope is not None and metric in GRANDFATHERED_METRICS:
        raise ColumnNameError(
            f"{metric} is inherently interfacial and takes no scope; it kept its i_/ip name "
            f"rather than gaining a slot"
        )
    if kind == "apo" and scope in ("target", "interface", "binder_interface", "target_interface"):
        raise ColumnNameError(f"an apo structure holds only the binder, so scope {scope!r} has nothing to describe")

    parts = [p for p in (seq_type, kind, backend, scope, metric) if p]
    return "_".join(parts) + ("_all" if per_sequence else "")


# =============================================================================
# Migration
# =============================================================================

# Columns whose names do not change, by the rule that decides it. Everything a
# reader might expect to be renamed and is not should be explainable from here.
#
#   sequence-level   describes the sequence, not a structure, so no kind,
#                    backend or scope applies
#   aggregate        run-level rollups (_res_*), out of scope by decision: they
#                    are not per-structure metrics and folding them in triples
#                    the surface for no clarity. Includes their own
#                    backend-as-suffix form, deliberately left alone
#   identity         config, provenance and the design's own identity, including
#                    pdb_path -- used by dedup, FoldSeek diversity and the
#                    pooled report, and not a metric
_SEQUENCE_LEVEL_METRICS: tuple[str, ...] = (
    "sequence",
    "aa_counts",
    "esm_log_likelihood",
    "esm_pseudo_perplexity",
    "redesign_score",
    "pass",
)

# Retired rather than renamed: biotite's P-SEA over-called beta by four times on
# this campaign's binders (877 residues claimed against 178 mkdssp confirmed), so
# its sheet fraction was never safe to report. Secondary structure comes from
# metrics/structure_ss.py now.
RETIRED_COLUMNS: frozenset[str] = frozenset({"_res_ss_alpha", "_res_ss_beta", "_res_ss_coil"})


def _split_seq_type(name: str) -> tuple[str | None, str]:
    """Peel a sequence-type prefix, longest-first. See SEQ_TYPES."""
    for seq in SEQ_TYPES:
        if name.startswith(seq + "_"):
            return seq, name[len(seq) + 1 :]
    return None, name


def classify(old: str, complex_backend: str = "af2") -> tuple[str, str | None]:
    """``(rule, new_name)`` for one old column, where *rule* names why.

    *complex_backend* fills the slot the old names never had. It defaults to
    ``af2`` because ``{seq}_complex_*`` was AF2 by circumstance -- the columns
    came from best_paths_dict, whatever refolder filled them, and every run
    written before this scheme used colabdesign. A pre-rename run that used RF3
    cannot be told apart from its column names alone, so it needs the override.
    The producer passes its own resolved backend, which is what makes emission
    and migration one mapping rather than two that must agree by inspection.

    Every column resolves to a named rule, so "unchanged" can be told apart from
    "fell through the bottom of the function". A migration checked against a real
    header wants that distinction: silence is what lets a missed rename read as a
    deliberate one.
    """
    if old in RETIRED_COLUMNS or old.removesuffix("_all") in RETIRED_COLUMNS:
        return "retired", None
    tail = "_all" if old.endswith("_all") else ""
    base = old[: -len(tail)] if tail else old

    if base.startswith("_res_"):
        return "aggregate", old
    seq, rest = _split_seq_type(base)
    if seq is None:
        return "identity", old
    if rest in _SEQUENCE_LEVEL_METRICS:
        return "sequence-level", old
    # Already in the new scheme. Without this, self_complex_af2_pTM re-enters the
    # complex_ branch and becomes self_complex_af2_af2_pTM -- so the map could not
    # be applied twice, nor to a header of mixed vintage, which is exactly what a
    # partially migrated campaign has.
    for kind in KINDS:
        if rest.startswith(kind + "_"):
            after_kind = rest[len(kind) + 1 :]
            if any(after_kind.startswith(backend + "_") for backend in BACKENDS):
                return "already migrated", old
    if rest == "aa_interface_counts":
        return "generated-defined interface", f"{seq}_complex_generated_binder_interface_aa_counts{tail}"
    if rest.startswith("complex_"):
        return "backend named explicitly", f"{seq}_complex_{complex_backend}_{rest[len('complex_'):]}{tail}"
    if rest.startswith("binder_"):
        return "binder was a scope", f"{seq}_complex_{complex_backend}_{rest}{tail}"
    if rest.startswith("apo_"):
        inner = rest[len("apo_") :]
        for backend in BACKENDS:
            if inner.endswith("_" + backend):
                return "backend suffix to slot", f"{seq}_apo_{backend}_binder_{inner[: -len(backend) - 1]}{tail}"
        return "UNCLASSIFIED", old
    for backend in BACKENDS:
        if rest.startswith(backend + "_"):
            return "gains kind slot", f"{seq}_complex_{backend}_{rest[len(backend) + 1:]}{tail}"
    return "UNCLASSIFIED", old


def rename(old: str, complex_backend: str = "af2") -> str | None:
    """The new name for an old column, or None if it is retired.

    Returns *old* unchanged for anything the scheme does not govern. Pure and
    table-free so it can be applied to a CSV header without loading a campaign.
    """
    return classify(old, complex_backend)[1]


def rename_map(columns, complex_backend: str = "af2") -> dict[str, str | None]:
    """``{old: new}`` for the columns that change, with None for retired ones."""
    out: dict[str, str | None] = {}
    for column in columns:
        new = rename(column, complex_backend)
        if new != column:
            out[column] = new
    return out


def migrate_frame(frame, complex_backend: str | None = None):
    """Rename a results frame's columns into the current scheme, in place.

    Applied where a CSV enters, so everything downstream sees one vocabulary.
    Idempotent, because a frame may already be current or hold a mix -- a pooled
    frame can gather runs written on either side of the rename.

    *complex_backend* defaults to the frame's own ``complex_folding_backend``
    column when it has one, and to ``af2`` when it does not: results predating
    that column came from ``{seq}_complex_*``, which was AF2 by circumstance.
    Retired columns are dropped rather than renamed.
    """
    columns = list(getattr(frame, "columns", ()))
    if not columns:
        return frame
    if complex_backend is None:
        complex_backend = "af2"
        if "complex_folding_backend" in columns:
            present = {v for v in frame["complex_folding_backend"].dropna().unique()}
            if len(present) == 1:
                complex_backend = str(next(iter(present)))
    mapping = rename_map(columns, complex_backend)
    dropped = [old for old, new in mapping.items() if new is None]
    renamed = {old: new for old, new in mapping.items() if new is not None}
    if dropped:
        frame = frame.drop(columns=dropped)
    if renamed:
        frame = frame.rename(columns=renamed)
    return frame
