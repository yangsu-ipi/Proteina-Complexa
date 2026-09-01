"""
Binder analysis utilities and default criteria.

This module contains:
- Default success thresholds for protein and ligand binders
- Metric column name mappings and normalization
- Threshold check helpers for binder-specific success criteria
"""

from typing import Any

import numpy as np
from loguru import logger

from proteinfoundation.result_analysis.analysis_utils import evaluate_threshold

# =============================================================================
# Metric Name Mapping
# =============================================================================

# Mapping from lowercase/alternative metric names to canonical column name suffixes
# This allows users to specify "plddt" instead of "pLDDT" etc.
METRIC_CASE_MAPPING = {
    # pLDDT variations
    "plddt": "pLDDT",
    "complex_plddt": "complex_pLDDT",
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
        "scale": 31.0,  # ipae * 31 <= 7
        "column_prefix": "complex",
        "metric": "i_pAE",
    },
    "complex_pLDDT": {
        "threshold": 0.9,
        "op": ">=",
        "scale": 1.0,
        "column_prefix": "complex",
        "metric": "pLDDT",
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
    # per-model -- {seq}_apo_scRMSD_ca_esmfold2_all, with no unsuffixed form. It is
    # expanded against the columns a run produced, so one criterion covers whatever
    # apo_folding_models asks for, and [esmfold, esmfold2] gates BOTH.
    "apo_scRMSD_ca": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "apo",
        "metric": "scRMSD_ca_{model}",
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
        "scale": 31.0,  # min_ipae * 31 < 2
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


def build_column_name(seq_type: str, column_prefix: str, metric_suffix: str) -> str:
    """Build the full column name for a metric.

    Args:
        seq_type: Sequence type ("self", "mpnn", "mpnn_fixed")
        column_prefix: Prefix like "complex", "binder", "ligand"
        metric_suffix: The metric suffix like "i_pAE", "pLDDT", "scRMSD"

    Returns:
        Full column name like "self_complex_i_pAE_all"
    """
    return f"{seq_type}_{column_prefix}_{metric_suffix}_all"


def threshold_column(seq_type: str, metric_name: str, spec: dict) -> str:
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
    )


MODEL_PLACEHOLDER = "{model}"


def expand_model_criteria(thresholds: dict, seq_type: str, available_columns) -> dict:
    """Expand ``{model}``-templated criteria against the columns a run produced.

    A criterion like ``scRMSD_ca_{model}`` with ``column_prefix: apo`` stands for
    "every apo folding model this run used". Expanding against the *columns*
    rather than against config means the evaluation stage and the analysis stage
    cannot disagree about which models are gated -- they read the same frame --
    and it works whether the run asked for one model or several.

    Criteria without the placeholder pass through untouched.

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
            col = build_column_name(seq_type, parsed.get("column_prefix", "complex"), effective)
            if col not in columns:
                logger.error(
                    f"Criterion '{name}' reads column '{col}', which this run did not produce, so no "
                    f"pass verdict will be emitted for '{seq_type}'. Enable the metric that produces "
                    f"it, or override aggregation.success_thresholds to drop the criterion."
                )
            out[name] = spec
            continue
        prefix = parsed.get("column_prefix", "complex")
        head, _, tail = effective.partition(MODEL_PLACEHOLDER)
        lead = build_column_name(seq_type, prefix, head)[: -len("_all")]
        models = sorted(
            col[len(lead) : -len(tail + "_all")] if tail else col[len(lead) : -len("_all")]
            for col in columns
            if col.startswith(lead) and col.endswith(tail + "_all")
        )
        if not models:
            # Kept, not dropped. Dropping it would leave the remaining criteria to
            # be evaluated on their own, and a design passing a three-criterion
            # gate is indistinguishable from one passing the four it was supposed
            # to face. Left in place, the template names a column that does not
            # exist, which the consumers already treat as "cannot judge" rather
            # than "passed".
            logger.error(
                f"Criterion '{name}' matched no {prefix} column for '{seq_type}', so it cannot be "
                f"applied and no verdict will be produced. Expected columns like "
                f"'{lead}<model>{tail}_all'. Check that the metric producing them is enabled."
            )
            out[name] = spec
            continue
        for model in models:
            # The expanded METRIC has to travel with the expanded key, or the
            # consumer resolves the column from a spec still holding "{model}".
            # Keyed by criterion-and-model so the four apo columns of a two-model
            # run stay distinguishable in logs and in the emitted criteria JSON.
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
