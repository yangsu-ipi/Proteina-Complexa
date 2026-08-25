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
DEFAULT_PROTEIN_BINDER_THRESHOLDS = {
    "i_pAE": {
        "threshold": 7.0,
        "op": "<=",
        "scale": 31.0,  # ipae * 31 <= 7
        "column_prefix": "complex",
    },
    "pLDDT": {
        "threshold": 0.9,
        "op": ">=",
        "scale": 1.0,
        "column_prefix": "complex",
    },
    "scRMSD_ca": {
        "threshold": 1.5,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "binder",
    },
    # Apo: the same sequence folded WITHOUT its target. BoltzGen reports that
    # requiring a binder to fold as designed both with and without the target
    # improves experimental success, and the holo criteria above cannot see it.
    #
    # 2.0 A follows the monomer scRMSD convention rather than a measured
    # distribution -- scRMSD is geometric, so the number transfers in principle,
    # but no run has produced the apo distribution for binders. Expect to revisit
    # it once one has.
    #
    # Requires metric.compute_apo_metrics. The ``{model}`` placeholder is expanded
    # against the apo columns a run actually produced, so one criterion covers
    # whatever apo_folding_models asks for: [esmfold] gates the esmfold column,
    # [esmfold2] the esmfold2 one, and [esmfold, esmfold2] gates BOTH -- the
    # binder must fold apo under every predictor asked, which composes the same
    # way the three holo criteria do. See expand_model_criteria.
    "scRMSD_ca_{model}": {
        "threshold": 2.0,
        "op": "<",
        "scale": 1.0,
        "column_prefix": "apo",
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
        if MODEL_PLACEHOLDER not in name:
            out[name] = spec
            continue
        parsed = parse_threshold_spec(spec)
        prefix = parsed.get("column_prefix", "complex")
        head, _, tail = name.partition(MODEL_PLACEHOLDER)
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
            out[f"{head}{model}{tail}"] = spec
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
