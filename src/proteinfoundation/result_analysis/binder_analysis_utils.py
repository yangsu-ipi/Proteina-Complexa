"""
Binder analysis utilities and default criteria.

This module contains:
- Default success thresholds for protein and ligand binders
- Metric column name mappings and normalization
- Threshold check helpers for binder-specific success criteria
"""

import math
import statistics
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # annotations only -- this module stays free of pandas at runtime
    import pandas as pd

from proteinfoundation.metrics.ensembling import PAE_MAX_BIN
from proteinfoundation.result_analysis.analysis_utils import (
    evaluate_threshold,
    literal_eval_with_infinities,
)

# =============================================================================
# Metric Name Mapping
# =============================================================================

# Mapping from lowercase/alternative metric names to canonical column name suffixes
# This allows users to specify "plddt" instead of "pLDDT" etc.
METRIC_CASE_MAPPING = {
    # pLDDT variations
    "plddt": "pLDDT",
    # complex_pLDDT collapsed into complex_binder_pLDDT, which is the number it
    # always held. Aliased rather than dropped so an existing
    # aggregation.success_thresholds override keeps gating what it meant to.
    "complex_plddt": "complex_binder_pLDDT",
    "complex_pLDDT": "complex_binder_pLDDT",
    "complex_binder_plddt": "complex_binder_pLDDT",
    # ipAE variations
    "ipae": "i_pAE",
    "i_pae": "i_pAE",
    "complex_ipae": "complex_i_pAE",
    "complex_i_pae": "complex_i_pAE",
    # iPTM variations
    "iptm": "i_pTM",
    "i_ptm": "i_pTM",
    "complex_iptm": "complex_i_pTM",
    "complex_i_ptm": "complex_i_pTM",
    # min_ipAE variations
    # The interface and secondary-structure metrics. Registered so a threshold on
    # any of them is a config entry rather than a code change, and none is active
    # by default: their distributions have not been measured on a campaign yet,
    # and a guessed bar silently moves the orderable count.
    "binder_dsasa": "binder_dSASA",
    "target_dsasa": "target_dSASA",
    "interface_dsasa": "interface_dSASA",
    "binder_buried_frac": "binder_buried_fraction",
    "interface_shape_complementarity": "interface_sc",
    "binder_iface_nres": "binder_interface_nres",
    "target_iface_nres": "target_interface_nres",
    "min_ipae": "min_ipAE",
    "min_i_pae": "min_ipAE",
    "complex_min_ipae": "complex_min_ipAE",
    "complex_min_i_pae": "complex_min_ipAE",
    # avg_ipSAE variations
    "avg_ipsae": "avg_ipSAE",
    "avg_i_psae": "avg_ipSAE",
    "complex_avg_ipsae": "complex_avg_ipSAE",
    # scRMSD variations
    "scrmsd": "scRMSD",
    "binder_scrmsd": "binder_scRMSD",
    "binder_scrmsd_ca": "binder_scRMSD_ca",
    "binder_scrmsd_allatom": "binder_scRMSD_allatom",
    "ligand_scrmsd": "ligand_scRMSD",
    "ligand_scrmsd_aligned_allatom": "ligand_scRMSD_aligned_allatom",
    "ligand_scrmsd_aligned_ca": "ligand_scRMSD_aligned_ca",
    "complex_scrmsd": "complex_scRMSD",
    # pTM variations
    "ptm": "pTM",
    "binder_ptm": "binder_pTM",
}


# =============================================================================
# Default Success Thresholds
# =============================================================================

# Default threshold specification structure:
# {
#     "metric_suffix": {
#         "threshold": float,           # The threshold value
#         "op": str,                    # Comparison operator: "<=", "<", ">=", ">", "=="
#         "scale": float,               # Scale factor applied to value before comparison (default 1.0)
#         "column_prefix": str,         # "complex", "binder", "ligand" - what comes before metric name
#     }
# }

# Default protein binder success thresholds (AlphaProteo criteria)
# Keys are NAMES; the column suffix is the `metric` field. Two criteria that differ
# only by prefix -- binder and complex scRMSD_ca -- cannot both exist while the key
# doubles as the suffix, because Python keeps whichever literal came last with no
# error at all. Every entry states its metric, so nothing here depends on the key.
# (The ligand and motif dicts still use the implicit form; threshold_column falls
# back to the key for them.)
DEFAULT_PROTEIN_BINDER_THRESHOLDS = {
    "complex_i_pAE": {
        "threshold": 7.0,
        "op": "<=",
        "scale": PAE_MAX_BIN,  # ipae * 31 <= 7
        "column_prefix": "complex",
        "metric": "i_pAE",
    },
    # What complex_pLDDT always was. ColabDesign's binder protocol computes
    # log["plddt"] over the binder alone, so the old name described a whole-
    # complex mean it never held -- confirmed numerically: the two columns
    # matched to the last digit on every row of a real run. Same 0.9 bar, same
    # meaning, honest name. AF2 pLDDT is on its native scale; this threshold
    # does not transfer to ESMFold2, which runs compressed (see
    # consensus_folding), and nothing there is gated.
    "complex_binder_pLDDT": {
        "threshold": 0.9,
        "op": ">=",
        "scale": 1.0,
        "column_prefix": "complex",
        "metric": "binder_pLDDT",
    },
    "binder_scRMSD_ca": {
        "threshold": 1.5,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "binder",
        "metric": "scRMSD_ca",
    },
    # Apo: the same sequence folded WITHOUT its target. BoltzGen reports that
    # requiring a binder to fold as designed both with and without the target
    # improves experimental success, and the holo criteria above cannot see it.
    #
    # 2.0 A was a convention when written. It is now measured: on 340 production
    # designs it rejects ~50% of sequences and is the most selective criterion in
    # the gate (docs/EVALUATION_METRICS.md). The smoke test said the opposite only
    # because it measured designs a reward had already selected.
    #
    # The {model} placeholder lives in the METRIC, because the emitted columns are
    # per-model -- {seq}_apo_esmfold2_binder_scRMSD_ca_all, with no unsuffixed
    # form. The model occupies the backend slot, so the placeholder leads. It is
    # expanded against the columns a run produced, so one criterion covers whatever
    # apo_folding_models asks for, and [esmfold, esmfold2] gates BOTH.
    "apo_scRMSD_ca": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "apo",
        "metric": "{model}_binder_scRMSD_ca",
    },
    # Placement, not fold. Both catch designs that fold correctly and sit somewhere
    # other than the interface they were designed for -- invisible to
    # binder_scRMSD_ca, which aligns on the binder and so cannot see where it went.
    #
    # Measured on the same 340 designs: five sequences across four designs passed
    # all four criteria above while sitting 11-27 A from their designed placement,
    # with i_pAE as good as 4.84 scaled, so no tightening of the existing gate
    # reaches them.
    #
    # complex_scRMSD_ca at 2.0 removes exactly those five and nothing else --
    # well-placed passers top out at 1.54. target_aligned at 2.0 is stricter: it
    # also drops sequences 2-5 A off, about 9% of passers. Both are kept because
    # they fail differently as targets change -- complex RMSD dilutes a binder
    # displacement across the stationary target residues (a 27 A shift reads as
    # 11 A here, with 136 target vs 59 binder), while target_aligned does not.
    "complex_scRMSD_ca": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "complex",
        "metric": "scRMSD_ca",
    },
    "binder_scRMSD_target_aligned_ca": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "binder",
        "metric": "scRMSD_target_aligned_ca",
    },
}

# Default ligand binder success thresholds
DEFAULT_LIGAND_BINDER_THRESHOLDS = {
    "min_ipAE": {
        "threshold": 2.0,
        "op": "<",
        "scale": PAE_MAX_BIN,  # min_ipae * 31 < 2
        "column_prefix": "complex",
    },
    "scRMSD_ca": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "binder",
    },
    "scRMSD_aligned_allatom": {
        "threshold": 5.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "ligand",
    },
}


# =============================================================================
# Metric Name Utilities
# =============================================================================


# =============================================================================
# Reducing a folder's draws
# =============================================================================
#
# A draw is one prediction: ESMFold2 repeats by seed, AF2 by parameter set.
# Evaluate records every one of them and reduces none, so an ``_all`` cell in a
# frame written that way is a list over SEQUENCES whose entries are lists over
# DRAWS. Collapsing them is a formulation over recorded values -- which is why it
# lives here, beside the thresholds, rather than in the folding code.
#
# It used to live there, and the cost was not theoretical. AF2's five models were
# averaged inside the folding harness and the parts discarded, so the only way to
# ask a different question of them was to predict the campaign again; and because
# only the first model's structure was kept in reach, anything re-read off "the"
# structure silently answered for one model while the number beside it answered
# for five. Recording the draws is what makes changing this rule a re-read.


def reduce_draws(values: list) -> Any:
    """Collapse one sequence's draws to the scalar a row reports, by mean.

    One rule for every metric, including the placement RMSDs that used to take
    the worst draw -- see ensembling.reduce_rmsd_over_models for what that
    exception bought and what dropping it gives up.

    A draw that produced no usable number is dropped rather than folded in, so
    one NaN cannot cost four good measurements, and a metric with nothing finite
    behind it stays NaN rather than becoming a plausible-looking zero.

    Non-numeric draws (a structure path, the SASA engine) take the first, which
    is what a reader pointed at "the" structure gets; and a value that is not a
    list at all is already reduced and passes straight through, which is how a
    frame written before evaluate recorded draws still reads.
    """
    if not isinstance(values, (list, tuple)):
        return values
    if not values:
        return float("nan")
    numeric = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    ]
    if not numeric:
        non_numeric = [v for v in values if not isinstance(v, (int, float, bool))]
        return non_numeric[0] if non_numeric else float("nan")
    return sum(numeric) / len(numeric)


def _object_column(values: list) -> "np.ndarray":
    """A 1-D object array holding *values*, whatever shape they are.

    Assigning a plain list of lists to a DataFrame column is a trap: when the
    inner lists happen to be the same length, both pandas and numpy read them as
    a 2-D array and refuse -- or worse, succeed and spread one row's draws across
    the frame. Every row here holds a list, and equal redesign counts are the
    normal case, so the column is built element by element where no shape can be
    inferred.
    """
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


def parse_all_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Turn every ``_all`` cell back into the list it was before the CSV.

    Analyze does not receive evaluate's frame. It receives
    ``pd.concat(pd.read_csv(f) for f in per_shard_files)``, so every list-valued
    cell arrives as the *repr* of a list -- ``"[[0.13, 0.14], [0.15, 0.16]]"`` --
    and a ``str`` satisfies none of the ``isinstance(cell, list)`` tests the
    readers below are written against.

    Nothing raised. :func:`reduce_draws_in_frame` passed the strings through as
    already-reduced, so the per-draw lists stayed nested; ``pick_headline_sequence``
    then indexed a string and produced NaN for every selected scalar in the frame.
    Both symptoms are this one missing step, and neither was visible to the tests,
    which build frames in memory and never go through a file.

    ``literal_eval_with_infinities`` rather than ``ast.literal_eval`` because a
    failed fold is ``inf`` and an absent metric is ``nan``, and the plain parser
    refuses both -- which is the whole reason that helper exists.

    A cell that is already a list (a frame passed in memory, as the tests do) and
    one that does not look like a list at all are left exactly as they are, so
    this is safe to run on any frame and is not a second contract.
    """
    for column in [c for c in df.columns if c.endswith("_all")]:
        if not any(isinstance(cell, str) for cell in df[column]):
            continue
        parsed: list = []
        failures = 0
        for cell in df[column]:
            if not isinstance(cell, str) or not cell.strip().startswith("["):
                parsed.append(cell)
                continue
            try:
                parsed.append(literal_eval_with_infinities(cell))
            except (ValueError, SyntaxError, TypeError, RecursionError):
                # Left as the string it was: a cell this cannot read is a cell
                # nothing downstream should pretend to have understood.
                failures += 1
                parsed.append(cell)
        df[column] = _object_column(parsed)
        if failures:
            logger.warning(f"Could not parse {failures} cell(s) of {column}; they stay unread")
    return df


def reduce_draws_in_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    """Collapse every per-draw ``_all`` cell into the per-sequence list analyze reads.

    Applied once, on the way in, so that everything downstream keeps the contract
    it already has: ``X_all`` is a list over sequences and ``X`` is
    ``X_all[best_idx]``. What changes is only that the number at each position is
    now computed here from the draws behind it, rather than having been computed
    in evaluate and frozen into the artifact.

    Tolerant by design. A cell whose entries are already scalars is left exactly
    as it is, which is what a frame written before evaluate recorded draws holds,
    and what the primary backend's columns still hold until every complex folder
    is reached the same way. Mixed frames are therefore fine, and a pooled frame
    holding both is fine.
    """
    # The frame reaches analyze through CSV, so the lists are reprs until this
    # runs. Done here rather than at the call site because every reader below
    # depends on it, and a caller that forgot would see no error -- only NaN.
    df = parse_all_columns(df)
    for column in [c for c in df.columns if c.endswith("_all")]:
        reduced = []
        touched = False
        for cell in df[column]:
            if not isinstance(cell, (list, tuple)) or not any(
                isinstance(v, (list, tuple)) for v in cell
            ):
                reduced.append(cell)
                continue
            touched = True
            reduced.append([reduce_draws(v) for v in cell])
        if touched:
            df[column] = _object_column(reduced)
            logger.debug(f"Reduced per-draw values in {column} (mean)")
    return df


def normalize_metric_name(metric_name: str) -> str:
    """Normalize a metric name to its canonical form using METRIC_CASE_MAPPING.

    Args:
        metric_name: The metric name (potentially lowercase or alternative form)

    Returns:
        The canonical metric name
    """
    # Check if it's in the mapping (case-insensitive lookup)
    lower_name = metric_name.lower()
    if lower_name in METRIC_CASE_MAPPING:
        return METRIC_CASE_MAPPING[lower_name]
    # Also check the original name in case it's already correct
    if metric_name in METRIC_CASE_MAPPING:
        return METRIC_CASE_MAPPING[metric_name]
    # Return as-is if not in mapping
    return metric_name


def normalize_threshold_dict(thresholds: dict) -> dict:
    """Normalize all metric names in a threshold dictionary.

    Args:
        thresholds: Dictionary with metric names as keys

    Returns:
        Dictionary with normalized metric names
    """
    normalized = {}
    for metric_name, spec in thresholds.items():
        normalized_name = normalize_metric_name(metric_name)
        normalized[normalized_name] = spec
    return normalized


# A ranking criterion may name the folder whose opinion it reads, so "best" can
# be defined across folders -- low i_pAE in AF2 AND in ESMFold2 -- rather than by
# whichever one the frame happens to call primary.
#
# Spelled into the KEY rather than only into the spec, because the criteria dict
# is keyed by quantity and two criteria on i_pAE from two folders would otherwise
# need the same key twice. Python resolves that silently in favour of the last,
# which is the failure resolve_backend_overrides carries a scar from: `binder`
# and `complex` scRMSD_ca collided on one key and one of them simply stopped
# being applied. A folder-qualified key cannot collide.
RANKING_BACKEND_SEPARATOR = ":"


def split_ranking_key(name: str) -> tuple[str | None, str]:
    """``"esmfold2:i_pAE"`` -> ``("esmfold2", "i_pAE")``; a bare name -> ``(None, name)``.

    None means "whichever folder this frame says produced its complexes", which
    is what every criterion meant before folders could be named and is what an
    unqualified criterion still means.
    """
    backend, separator, metric = name.partition(RANKING_BACKEND_SEPARATOR)
    if not separator:
        return None, name
    return (backend or None), metric


def ranking_criterion_backend(name: str, spec, default_backend: str) -> tuple[str, str, bool]:
    """Which folder and metric a ranking criterion reads, and whether it said so.

    The key wins over the spec: it is the half that has to be unique anyway, so
    letting a spec field contradict it would make two spellings of one fact.
    The third element says the folder was NAMED, which is what lets a missing
    column be an error rather than a shrug -- a criterion that asked for
    ESMFold2 on a campaign that never ran it is a question nothing can answer,
    and ranking by the remainder would silently rank by something else.
    """
    backend, metric = split_ranking_key(name)
    if backend is None and isinstance(spec, Mapping):
        backend = spec.get("backend")
    return (backend or default_backend), metric, backend is not None


def resolve_ranking_columns(
    seq_type: str, ranking_criteria: dict, default_backend: str, available=None
) -> dict[str, str]:
    """``{criterion key: the per-sequence column it ranks on}``.

    One column per criterion rather than one backend for all of them, which is
    the whole generalisation: the folder is resolved per criterion, so a run can
    ask for agreement between folders instead of taking one folder's word.

    *available* is the frame's columns. Given it, a criterion that NAMED a folder
    whose column is absent raises rather than being skipped -- see
    :func:`ranking_criterion_backend` for why that asymmetry is deliberate.
    """
    from proteinfoundation.result_analysis.analysis_utils import parse_threshold_spec

    out: dict[str, str] = {}
    unanswerable: list[str] = []
    for name, spec in ranking_criteria.items():
        backend, metric, named = ranking_criterion_backend(name, spec, default_backend)
        column = threshold_column(seq_type, metric, parse_threshold_spec(spec), backend)
        if named and available is not None and column not in available:
            unanswerable.append(f"{name} -> {column}")
        out[name] = column
    if unanswerable:
        raise ThresholdSpecError(
            f"Ranking criteria {sorted(unanswerable)} name a folder this frame has no columns for. "
            f"Ranking by the rest would choose a sequence by criteria nobody asked for, silently. "
            f"Drop the criterion, or run that folder."
        )
    return out


def build_column_name(
    seq_type: str, column_prefix: str, metric_suffix: str, complex_backend: str = "af2"
) -> str:
    """Build the full column name for a metric.

    Args:
        seq_type: Sequence type ("self", "mpnn", "mpnn_fixed")
        column_prefix: Prefix like "complex", "binder", "ligand"
        metric_suffix: The metric suffix like "i_pAE", "pLDDT", "scRMSD"

    Returns:
        Full column name like "self_complex_af2_i_pAE_all"

    The column_prefix vocabulary predates the slot scheme and is kept because
    threshold dictionaries are user-facing config. It maps onto slots: "complex"
    is the whole refolded complex, "binder" the binder within it, and both live
    under kind=complex with the run's own backend. "apo" is a different
    structure and is left alone here.
    """
    from proteinfoundation.metrics.column_names import rename

    if column_prefix in ("complex", "binder"):
        legacy = f"{seq_type}_{column_prefix}_{metric_suffix}"
        return f"{rename(legacy, complex_backend)}_all"
    return f"{seq_type}_{column_prefix}_{metric_suffix}_all"


def threshold_column(seq_type: str, metric_name: str, spec: dict, complex_backend: str = "af2") -> str:
    """The column a criterion reads.

    A criterion's key doubles as the column suffix unless the spec says otherwise.
    That default is why ``binder`` and ``complex`` ``scRMSD_ca`` could not both
    exist: one key, and Python keeps whichever literal came last -- no error, and
    the only symptom a pass rate that moved for no stated reason.

    ``metric`` frees the key to be a name. The protein-binder defaults now state it
    on every entry, so nothing there depends on the key at all; the ligand and
    motif dicts still use the implicit form, which is why the fallback stays.
    """
    return build_column_name(
        seq_type,
        spec.get("column_prefix", "complex"),
        spec.get("metric") or metric_name,
        complex_backend,
    )


# The provenance column naming the folding model that produced the complex
# refolds a run gated on. Recorded per row rather than read from config: analyze
# and analyze_pooled re-derive verdicts from a CSV alone, and a pooled frame can
# hold runs that used different folders.
COMPLEX_BACKEND_COLUMN = "complex_folding_backend"


class ThresholdSpecError(ValueError):
    """A threshold dictionary cannot be applied to the run it was given."""


# Packed eight-state counts. A threshold cannot be compared against a list, and
# the three-state fractions analyze derives from them are what a gate reads.
PACKED_COUNT_SUFFIX = "_ss_counts"


def reject_thresholds_on_packed_columns(thresholds: dict, seq_type: str, complex_backend: str = "af2") -> None:
    """Refuse a criterion that would compare a threshold against a list.

    The eight-state counts are the record; the helix/sheet/loop fractions
    derived from them in analyze are the numbers. Without this, a threshold on
    ``..._ss_counts`` reaches the comparison as ``[34.0, 0.0, ...] < 0.5`` --
    a TypeError deep in the verdict loop at best, and a criterion that never
    matches at worst.
    """
    from proteinfoundation.result_analysis.analysis_utils import parse_threshold_spec

    offenders = sorted(
        name
        for name, spec in thresholds.items()
        if threshold_column(seq_type, name, parse_threshold_spec(spec), complex_backend).endswith(
            PACKED_COUNT_SUFFIX + "_all"
        )
    )
    if offenders:
        raise ThresholdSpecError(
            f"Criteria {offenders} read packed eight-state counts, which cannot be compared against a "
            f"threshold. Gate on the derived fractions instead -- ..._ss_helix, _ss_sheet, _ss_loop."
        )


def resolve_backend_overrides(thresholds: dict, backend: str | None) -> dict:
    """Apply each criterion's ``by_backend`` overrides for one folding backend.

    A criterion states a base rule and, optionally, per-backend replacements::

        "complex_i_pAE": {
            "kind": "complex", "metric": "i_pAE", "op": "<=",
            "scale": 31.0, "threshold": 7.0,
            "by_backend": {"rf3": {"threshold": 12.0, "scale": 1.0}},
        }

    Nested rather than written as sibling entries because the key names the
    *quantity*: two entries differing only in backend would need two keys, and
    the natural thing to write is the same key twice, which Python resolves
    silently in favour of the last -- the failure this file already carries a
    scar from, where `binder` and `complex` scRMSD_ca collided on one key.

    Overrides are partial specs merged over the base, not bare numbers, because
    ``scale`` is part of what does not transfer: AF2's i_pAE reaches gated units
    multiplied by 31, and another folder need not.

    A criterion with no applicable rule raises. Dropping it would shrink the gate
    silently, and a design passing five criteria is indistinguishable from one
    passing the six it was meant to face.
    """
    out: dict = {}
    for name, spec in thresholds.items():
        # Mapping, not dict: a spec from a campaign's pipeline.yaml is an
        # omegaconf DictConfig, and testing for the builtin sent every one of
        # them down the passthrough branch. That failure is the quiet twin of the
        # one parse_threshold_spec raised -- by_backend would simply not apply,
        # and the base threshold would gate every backend as though the overrides
        # had never been written.
        if not isinstance(spec, Mapping) or "by_backend" not in spec:
            out[name] = spec
            continue
        base = {k: v for k, v in spec.items() if k != "by_backend"}
        by_backend = spec.get("by_backend") or {}
        override = by_backend.get(backend) if backend is not None else None
        if override is not None:
            out[name] = {**base, **dict(override)}
            continue
        if "threshold" not in base:
            raise ThresholdSpecError(
                f"Criterion '{name}' has no base threshold and no override for backend {backend!r} "
                f"(has {sorted(by_backend)}). Add a base threshold, or a rule for this backend -- "
                f"dropping the criterion would shrink the gate without saying so."
            )
        out[name] = base
    return out


def complex_backend_of(frame) -> str | None:
    """The one complex folding backend a frame's rows share, or None.

    None when the column is absent -- results written before it existed -- which
    leaves every criterion on its base rule. Raises when rows disagree: a pooled
    frame mixing folders cannot be gated by one resolved set, and silently using
    either would judge half the designs by the other's thresholds.
    """
    if COMPLEX_BACKEND_COLUMN not in getattr(frame, "columns", ()):
        return None
    # Flattened rather than assumed to be a Series. Per-job CSVs written while the
    # row builder appended this column once per sequence type carry two identical
    # copies, and selecting by name then returns a DataFrame -- whose .unique()
    # raised an AttributeError from inside pandas, naming neither this column nor
    # the duplication. Copies carrying the same value are one fact recorded twice;
    # copies that disagree still raise below, which is the case worth stopping for.
    values = {v for v in np.ravel(frame[COMPLEX_BACKEND_COLUMN].to_numpy()) if v is not None and v == v}
    if not values:
        return None
    if len(values) > 1:
        raise ThresholdSpecError(
            f"Rows report more than one complex folding backend ({sorted(values)}); "
            f"resolve thresholds per backend rather than for the frame as a whole."
        )
    return str(next(iter(values)))


MODEL_PLACEHOLDER = "{model}"


def expand_model_criteria(
    thresholds: dict,
    seq_type: str,
    available_columns,
    complex_backend: str = "af2",
    report_gaps: bool = True,
) -> dict:
    """Resolve ``{model}``-templated criteria against the columns a run produced.

    A criterion like ``scRMSD_ca_{model}`` with ``column_prefix: apo`` names the
    apo fold of the folder this run is gated on -- the one in
    ``complex_folding_backend`` -- and is checked against the *columns* rather
    than against config, so the evaluation stage and the analysis stage cannot
    disagree about what is gated.

    It used to stand for "every apo folder this run used", conjunctively, which
    meant the gate tightened itself whenever a folder was added: see the comment
    at the resolution below for what that cost on EFNB3. A folder's opinion is
    worth recording either way, and every advisory column still is; what changed
    is that it no longer silently becomes a pass/fail criterion.

    Exactly one folder answers. The gating one when it folded *prefix* at all;
    otherwise the sole folder that did, loudly -- that is a real configuration
    (CBLN1 gates on AF2 and folds apo with ESMFold2 deliberately), and refusing
    it would leave that campaign with no verdicts. Several candidates and none of
    them the gating folder is ambiguous, and ambiguity is reported rather than
    resolved by sort order.

    Criteria without the placeholder pass through untouched.

    *report_gaps* is what a caller knows and this function cannot: whether the
    column set it was handed is the whole run. Analyze passes the finished frame,
    where a column a criterion needs and cannot find is a real problem and is
    reported. Evaluate asks the same question of a row still being built -- the
    pass plan fills it one folder at a time, so most of the columns are legitimately
    absent most of the time -- and 657 designs x 7 criteria x 5 passes of that is
    tens of thousands of lines saying nothing, loud enough to bury the errors that
    do matter. There the gaps go to debug, and per_sequence_pass reports the one
    fact that follows from them: no verdict.

    Args:
        thresholds: Normalised threshold dictionary, possibly templated.
        seq_type: Sequence type whose columns to match against.
        available_columns: Column names present (any iterable).

    Returns:
        A new dictionary with each templated entry replaced by one entry per
        matching model, in sorted order for a stable gate.
    """
    from proteinfoundation.result_analysis.analysis_utils import parse_threshold_spec

    columns = set(available_columns)
    # Demoted, never dropped: a gap is still worth seeing under -v when a caller
    # has said the column set is partial.
    say_error = logger.error if report_gaps else logger.debug
    say_warning = logger.warning if report_gaps else logger.debug
    out: dict = {}
    for name, spec in thresholds.items():
        parsed = parse_threshold_spec(spec)
        # The placeholder lives wherever the column suffix comes from -- the
        # `metric` field when a spec sets one, the key otherwise. Partitioning the
        # key unconditionally would leave `apo_scRMSD_ca` with
        # `metric: scRMSD_ca_{model}` unexpanded, gating a column no run emits,
        # which per_sequence_pass turns into no verdict at all.
        effective = parsed.get("metric") or name
        if MODEL_PLACEHOLDER not in effective:
            # The guard: a criterion naming a column this run did not emit does not
            # weaken the gate, it removes it -- per_sequence_pass returns None and
            # no verdict is produced for any sequence. This is the only place with
            # both the criteria and the actual columns, so it is where that gets
            # said. Kept non-fatal and kept in the set, matching how an unmatched
            # {model} criterion is handled below: naming a missing column reads
            # downstream as "cannot judge", which is the honest outcome.
            col = build_column_name(seq_type, parsed.get("column_prefix", "complex"), effective, complex_backend)
            if col not in columns:
                say_error(
                    f"Criterion '{name}' reads column '{col}', which this run did not produce, so no "
                    f"pass verdict will be emitted for '{seq_type}'. Enable the metric that produces "
                    f"it, or override aggregation.success_thresholds to drop the criterion."
                )
            out[name] = spec
            continue
        prefix = parsed.get("column_prefix", "complex")
        head, _, tail = effective.partition(MODEL_PLACEHOLDER)
        lead = build_column_name(seq_type, prefix, head, complex_backend)[: -len("_all")]
        present = sorted(
            col[len(lead) : -len(tail + "_all")] if tail else col[len(lead) : -len("_all")]
            for col in columns
            if col.startswith(lead) and col.endswith(tail + "_all")
        )
        # The gating folder answers this criterion, and only it. Expanding over
        # every folder present made the expansions CONJUNCTIVE, so a run that
        # gained a folder silently gained a mandatory criterion: EFNB3's apo gate
        # went from "ESMFold2 agrees the binder folds the same unbound" to
        # "ESMFold2 and AF2 both agree" the moment the migration produced an AF2
        # apo fold, and the pass rate fell 16.1% -> 3.0% with no design changing.
        # Nobody chose that, and it made two campaigns with different folder sets
        # incomparable. Every other criterion resolves to the one backend named in
        # complex_folding_backend; this one now does too.
        if complex_backend in present:
            models = [complex_backend]
        elif len(present) == 1:
            # The gating folder did not fold apo, and exactly one folder did, so
            # there is no choice to make and no ambiguity to hide. This is the
            # CBLN1 shape: gated on AF2, apo folded by ESMFold2 on purpose.
            # Said out loud, because which folder answered a criterion is not
            # something a reader should have to infer from the column list.
            models = list(present)
            say_warning(
                f"Criterion '{name}' is answered by '{present[0]}' for '{seq_type}': the gating folder "
                f"'{complex_backend}' produced no {prefix} fold, and '{present[0]}' is the only one that did."
            )
        else:
            models = []
            if present:
                say_error(
                    f"Criterion '{name}' cannot be answered for '{seq_type}': the gating folder "
                    f"'{complex_backend}' produced no {prefix} fold and {present} disagree about who "
                    f"should stand in. No verdict will be produced. Name the folder in "
                    f"aggregation.success_thresholds, or fold {prefix} with '{complex_backend}'."
                )
                out[name] = spec
                continue
        if not models:
            # Kept, not dropped. Dropping it would leave the remaining criteria to
            # be evaluated on their own, and a design passing a three-criterion
            # gate is indistinguishable from one passing the four it was supposed
            # to face. Left in place, the template names a column that does not
            # exist, which the consumers already treat as "cannot judge" rather
            # than "passed".
            say_error(
                f"Criterion '{name}' matched no {prefix} column for '{seq_type}', so it cannot be "
                f"applied and no verdict will be produced. Expected columns like "
                f"'{lead}<model>{tail}_all'. Check that the metric producing them is enabled."
            )
            out[name] = spec
            continue
        for model in models:
            # The resolved METRIC has to travel with the resolved key, or the
            # consumer reads the column from a spec still holding "{model}".
            # Keyed by criterion-and-folder so the emitted criteria JSON says
            # which folder answered, rather than leaving it to be inferred.
            out[f"{name}_{model}"] = {**parsed, "metric": f"{head}{model}{tail}"}
    return out


def get_thresholds_for_result_type(
    success_thresholds: dict | None,
    is_ligand_binder: bool = False,
) -> dict:
    """Get appropriate thresholds based on result type.

    Args:
        success_thresholds: User-provided thresholds (may be None)
        is_ligand_binder: Whether this is a ligand binder

    Returns:
        Threshold dictionary to use
    """
    if success_thresholds is not None:
        return success_thresholds

    if is_ligand_binder:
        return DEFAULT_LIGAND_BINDER_THRESHOLDS.copy()
    return DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()


# =============================================================================
# Threshold Check Helpers
# =============================================================================


def check_redesign_passes_all_thresholds(
    metric_values: dict[str, Any],
    parsed_thresholds: dict,
) -> bool:
    """Check if a single redesign passes all threshold criteria.

    This is the shared evaluation logic used by both filter_by_success_thresholds
    and compute_filter_pass_rate.

    Args:
        metric_values: Dictionary mapping metric names to values for one redesign
                       e.g., {"i_pAE": 0.15, "pLDDT": 0.92, "scRMSD": 1.2}
        parsed_thresholds: Dictionary of parsed threshold specs (output of parse_threshold_spec)
                          e.g., {"i_pAE": {"threshold": 7.0, "op": "<=", "scale": 31.0, "column_prefix": "complex"}}

    Returns:
        True if all criteria pass, False otherwise
    """
    for metric_name, spec in parsed_thresholds.items():
        if metric_name not in metric_values:
            return False

        value = metric_values[metric_name]

        # Handle non-float values (e.g., strings, None)
        if not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return False

        if not evaluate_threshold(value, spec["threshold"], spec["op"], spec["scale"]):
            return False

    return True


def as_redesign_list(value) -> list:
    """One metric's per-redesign values, however they arrived.

    analyze reads its inputs with ``pd.read_csv``, which hands back the *text* of
    a list column -- ``"[0.146]"`` -- not a list. Indexing that by redesign
    position walks characters instead: a design with one redesign looked like
    seven, each judged against a ``'['`` or a digit, and every verdict came out
    0. The lengths disagreeing is what made it visible at all.

    Coerced here rather than at the call site because this function is what
    defines a per-redesign list. Another caller reading the same CSV would
    otherwise reintroduce it, and the symptom -- every design failing -- looks
    far more like a strict gate than like a parsing bug.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            # Not ast.literal_eval: a fold that failed is `inf` in this list, and
            # `inf` is a Name rather than a literal, so the repr of any list
            # holding one raised here and became []. Downstream that reads as
            # "no redesigns", not as "could not parse" -- see the docstring on
            # literal_eval_with_infinities for what that cost.
            parsed = literal_eval_with_infinities(text)
        except (ValueError, SyntaxError):
            return []
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):  # numpy array from a parsed column
        listed = value.tolist()
        return listed if isinstance(listed, list) else [listed]
    if value is None:
        return []
    # A bare scalar is one redesign, unless it is the NaN a missing cell becomes.
    if isinstance(value, float) and math.isnan(value):
        return []
    return [value]


def redesign_pass_vector(
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> list[int]:
    """Per-redesign verdicts for one sample: 1 if that redesign passes ALL criteria.

    The primitive behind every pass/fail statement about a sample. Both the
    "did any redesign pass" question and the "how many passed" question are
    reductions of this vector, and the per-sequence column emitted at
    evaluation time is the vector itself -- so a redesign cannot be a failure
    in one place and a success in another.

    Args:
        sample_metric_values: Dictionary mapping metric names to lists of values (one per redesign)
                             e.g., {"i_pAE": [0.15, 0.18], "pLDDT": [0.92, 0.88]}
        parsed_thresholds: Dictionary of parsed threshold specs

    Returns:
        List of 1/0, one per redesign, in the order the redesigns appear.
        Empty if there is nothing to judge.
    """
    if not sample_metric_values:
        return []

    sample_metric_values = {name: as_redesign_list(v) for name, v in sample_metric_values.items()}

    # The criteria's metric lists are built in append order and are parallel by
    # construction. Judge only the prefix all of them cover, rather than indexing
    # off the end of a short one: a misalignment should cost the unjudgeable tail,
    # not raise from inside a metric computation. It is loud because a silent
    # truncation here would read downstream as "these redesigns did not exist".
    lengths = {metric: len(values) for metric, values in sample_metric_values.items()}
    n_redesigns = min(lengths.values())
    if len(set(lengths.values())) > 1:
        logger.error(f"Metric lists disagree on redesign count {lengths}; judging the first {n_redesigns}")

    verdicts = []
    for i in range(n_redesigns):
        # Build metric values for this redesign
        redesign_values = {}
        for metric_name in parsed_thresholds:
            if metric_name in sample_metric_values:
                redesign_values[metric_name] = sample_metric_values[metric_name][i]

        verdicts.append(1 if check_redesign_passes_all_thresholds(redesign_values, parsed_thresholds) else 0)

    return verdicts


def check_sample_has_passing_redesign(
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> bool:
    """Check if ANY redesign in a sample passes ALL threshold criteria.

    Args:
        sample_metric_values: Dictionary mapping metric names to lists of values (one per redesign)
                             e.g., {"i_pAE": [0.15, 0.18], "pLDDT": [0.92, 0.88]}
        parsed_thresholds: Dictionary of parsed threshold specs

    Returns:
        True if at least one redesign passes all criteria
    """
    return any(redesign_pass_vector(sample_metric_values, parsed_thresholds))


def count_passing_redesigns(
    sample_metric_values: dict[str, list],
    parsed_thresholds: dict,
) -> int:
    """Count how many redesigns in a sample pass ALL threshold criteria.

    Args:
        sample_metric_values: Dictionary mapping metric names to lists of values (one per redesign)
        parsed_thresholds: Dictionary of parsed threshold specs

    Returns:
        Number of redesigns that pass all criteria
    """
    return sum(redesign_pass_vector(sample_metric_values, parsed_thresholds))


# =============================================================================
# Robust outlier flags
# =============================================================================
#
# Advisory metrics have no transferable absolute scale -- ESMFold2 in particular
# runs compressed, where a native protein folds to ~0.65 -- so the only baseline
# available is the campaign's own designs. That baseline is a good one here: the
# target sequence is identical in every design and the advisory fold is not
# templated, so the spread in target pLDDT across a campaign already is the
# effect of the binder on the target.

OUTLIER_K = 3.0

# Makes the MAD comparable to a standard deviation for normally distributed data,
# so k keeps its usual meaning: k=3 is roughly "three sigma".
_MAD_TO_SIGMA = 1.4826


def robust_spread(values: list[float]) -> tuple[float, float]:
    """Median and MAD-derived sigma, ignoring anything non-finite.

    Median and MAD rather than mean and standard deviation because what we are
    looking for is *in* the data we would estimate from: a handful of designs
    that wreck the target inflate the standard deviation, which pushes a
    mean-based threshold down until those same designs sit inside it and hide
    themselves. On the CBLN1 production run a 3-sigma cut flagged one design
    where a 3-MAD cut flagged twelve.
    """
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(finite) < 3:
        # Two points have a MAD, but not one that means anything.
        return (float("nan"), float("nan"))
    median = statistics.median(finite)
    mad = statistics.median([abs(v - median) for v in finite])
    return (median, mad * _MAD_TO_SIGMA)


def low_outlier_threshold(values: list[float], k: float = OUTLIER_K) -> float:
    """``median - k * sigma``, or NaN when there is not enough data to say."""
    median, sigma = robust_spread(values)
    if not math.isfinite(median) or not math.isfinite(sigma) or sigma == 0:
        # A zero MAD means over half the designs share one value exactly. The
        # median is then a threshold that rejects every design at or below the
        # most common number, which is not what a caller asking for an outlier
        # cut wants -- there are no outliers to find.
        return float("nan")
    return median - k * sigma


def advisory_per_chain_columns(df: "pd.DataFrame") -> list[str]:
    """Advisory per-chain pLDDT columns present on the frame.

    Gated columns are excluded by their reserved ``complex`` segment, which is
    the same marker :func:`report_gated_and_reported_columns` enforces on the way in.
    """
    return [
        column
        for column in df.columns
        if (column.endswith("_target_pLDDT") or column.endswith("_binder_pLDDT")) and "_complex_" not in column
    ]


def add_outlier_columns(df: "pd.DataFrame", columns: list[str] | None = None, k: float = OUTLIER_K):
    """Add ``_robust_z`` and ``_low_outlier`` beside each named column.

    ``_robust_z`` is signed, so a design well *above* the campaign is visible
    too; ``_low_outlier`` marks only the low tail, which is the direction that
    means the binder hurt something.

    These describe a design relative to the campaign it was run in, so a row's
    flag depends on which other rows are present. That is acceptable precisely
    because these are advisory: they change nothing about pass or fail, and the
    reproducibility a verdict needs is not a property this has to carry. A gate
    built on this reduction would have to freeze its threshold to a constant
    first.
    """
    for column in columns if columns is not None else advisory_per_chain_columns(df):
        if column not in df.columns:
            continue
        values = df[column].tolist()
        median, sigma = robust_spread(values)
        if not math.isfinite(median) or not math.isfinite(sigma) or sigma == 0:
            # A campaign with no spread has no outliers, and dividing by its
            # sigma would report every design as infinitely unusual.
            df[f"{column}_robust_z"] = float("nan")
            df[f"{column}_low_outlier"] = False
            continue
        z = [(float(v) - median) / sigma if v is not None and math.isfinite(float(v)) else float("nan") for v in values]
        df[f"{column}_robust_z"] = z
        df[f"{column}_low_outlier"] = [bool(score < -k) if math.isfinite(score) else False for score in z]
    return df
