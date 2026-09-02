"""Reductions over an ensemble of predictions.

AF2-Multimer has five parameter sets and ESMFold2 is a diffusion sampler, so
in both cases "the prediction" is a draw rather than an answer. These helpers
hold the rules for collapsing several draws into one number.

Kept free of torch, JAX and colabdesign on purpose. The reduction rules are the
part most worth testing -- whether a NaN poisons a mean, whether averaging
happens before or after rounding -- and none of that needs a GPU, or the
dependencies that would keep it out of a unit test.
"""

import math

# The rounding the single-model path used, kept so a one-model run's numbers
# stay byte-identical to what it produced before ensembling existed.
AF2_STAT_PRECISION = {
    "pLDDT": 3,
    "pTM": 3,
    "i_pTM": 3,
    "pAE": 3,
    "i_pAE": 3,
    "min_ipAE": 4,
    "min_ipSAE": 4,
    "max_ipSAE": 4,
    "avg_ipSAE": 4,
    "min_ipSAE_10": 4,
    "max_ipSAE_10": 4,
    "avg_ipSAE_10": 4,
}


def af2_stats_from_metrics(prediction_metrics: dict) -> dict:
    """The confidence scores for one AF2 model, unrounded.

    Rounding is deferred to :func:`average_af2_stats` so the mean is taken at
    full precision -- rounding each model first would average five rounding
    errors along with the scores.
    """
    return {
        "pLDDT": prediction_metrics["plddt"],
        "pTM": prediction_metrics["ptm"],
        "i_pTM": prediction_metrics["i_ptm"],
        "pAE": prediction_metrics["pae"],
        "i_pAE": prediction_metrics["i_pae"],
        "min_ipAE": prediction_metrics["min_ipae"],
        "min_ipSAE": prediction_metrics["min_ipsae"],
        "max_ipSAE": prediction_metrics["max_ipsae"],
        "avg_ipSAE": prediction_metrics["avg_ipsae"],
        "min_ipSAE_10": prediction_metrics.get("min_ipsae_10", 0.0),
        "max_ipSAE_10": prediction_metrics.get("max_ipsae_10", 0.0),
        "avg_ipSAE_10": prediction_metrics.get("avg_ipsae_10", 0.0),
    }


def average_af2_stats(per_model: list[dict]) -> dict:
    """Mean each confidence score over the AF2 models that produced it.

    A plain mean, not a best-of: the point of running five models is that their
    disagreement is information about how confident the prediction really is,
    and taking the best would discard exactly that.
    """
    return {
        key: round(sum(model[key] for model in per_model) / len(per_model), precision)
        for key, precision in AF2_STAT_PRECISION.items()
    }


def average_rmsd_over_models(per_model: list[dict]) -> dict:
    """Mean each RMSD over the models that produced a finite value.

    A model that failed to produce a usable number is dropped rather than
    averaged in, so one NaN cannot turn four good placements into no answer --
    and a metric with nothing finite behind it stays NaN rather than becoming a
    plausible-looking zero.
    """
    if not per_model:
        return {}
    if len(per_model) == 1:
        return per_model[0]
    averaged = {}
    for key in per_model[0]:
        values = [
            model[key] for model in per_model if isinstance(model.get(key), (int, float)) and math.isfinite(model[key])
        ]
        averaged[key] = sum(values) / len(values) if values else float("nan")
    return averaged


def pop_per_model_paths(complex_statistics: list, seq_num: int) -> list[str] | None:
    """The per-model structure paths for one sequence, removed as they are read.

    Popped rather than read because these stats become dataframe columns
    downstream, and a list-valued column survives nothing.

    Shaped defensively because only the colabdesign backend sets the key: the
    others return the same ``[{"seq_N": stats}]`` shape without it, and a
    backend that someday returns something else should degrade to the single
    structure rather than raise inside the geometry loop.
    """
    if seq_num >= len(complex_statistics):
        return None
    entry = complex_statistics[seq_num]
    if not isinstance(entry, dict):
        return None
    stats = entry.get(f"seq_{seq_num + 1}")
    if not isinstance(stats, dict):
        return None
    paths = stats.pop("complex_pdb_paths", None)
    return list(paths) if paths else None
