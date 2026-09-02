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
# No "pLDDT": ColabDesign's binder protocol reports log["plddt"] over the binder
# alone, so complex_pLDDT was never the whole-complex mean its name implied --
# it equalled the binder half exactly on every row of a real run, to the last
# digit the CSV carried. Keeping both would be one number under two names, and
# the misleading name is the one a threshold could be pointed at by mistake.
AF2_STAT_PRECISION = {
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
    # Per-chain means, from the same per-residue array the scalar pLDDT is the
    # mean of. Absent from backends that report no per-residue confidence, so
    # average_af2_stats skips what it is not given rather than requiring these.
    "target_pLDDT": 3,
    "binder_pLDDT": 3,
}


# A complex pLDDT below this is not plausible as a fraction, so the array is
# almost certainly on AlphaFold's 0-100 scale. Worth guarding: the gates read
# these as fractions, and a 0-100 array would clear a >= 0.9 threshold for
# every design ever scored, silently, while looking like a great campaign.
_PLDDT_FRACTION_CEILING = 1.5


def af2_stats_from_metrics(prediction_metrics: dict) -> dict:
    """The confidence scores for one AF2 model, unrounded.

    Rounding is deferred to :func:`average_af2_stats` so the mean is taken at
    full precision -- rounding each model first would average five rounding
    errors along with the scores.

    ``plddt`` is deliberately not read from here. It is the binder-only mean,
    and :func:`mean_chain_plddt` derives the same number from the per-residue
    array along with the target's, so taking it twice would only create a second
    name for it.
    """
    return {
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
        if all(key in model for model in per_model)
    }


# Placement asks whether the binder landed where it was designed to. A binder
# that lands correctly in one model of five has not been placed correctly, so the
# reduction is the worst model rather than the typical one.
#
# Measured, not assumed. Meaning these over five models compressed the
# distribution asymmetrically: designs already under 2 A barely moved (median
# +0.01 A, none crossing into failure), while 24 sequences crossed from failing
# complex_scRMSD_ca to passing it -- twelve of those from 4-8 A on a single
# model. The 2.0 A thresholds were calibrated against single-model geometry
# where misplaced designs sat at 4.6-11.6 A, which is exactly the band a mean
# pulls toward the cutoff. The max restores that discrimination, and is strictly
# harder to satisfy than the single model ever was.
#
# complex_scRMSD is the legacy alias of complex_scRMSD_ca and has to reduce the
# same way, or one row would carry two different answers to one question.
PLACEMENT_METRICS = frozenset(
    {
        "complex_scRMSD_ca",
        "complex_scRMSD",
        "binder_scRMSD_target_aligned_ca",
    }
)

# Bumped when a reduction rule changes what a cached number means. Rides in the
# binder eval fingerprint: the structures on disk stay valid, the numbers derived
# from them do not.
GEOMETRY_REDUCTION_VERSION = 2


def reduce_rmsd_over_models(per_model: list[dict]) -> dict:
    """Collapse per-model RMSDs, by mean or by worst case depending on the metric.

    Placement metrics (:data:`PLACEMENT_METRICS`) take the max: every model has
    to agree the binder is where it belongs. Everything else -- fold quality
    against the designed backbone -- takes the mean, where the spread between
    models is uncertainty about one structure rather than disagreement about a
    location.

    A model that produced no usable number is dropped rather than folded in, so
    one NaN cannot cost four good measurements, and a metric with nothing finite
    behind it stays NaN rather than becoming a plausible-looking zero. That holds
    for the max too: a failed model leaves the worst case unknown, not zero --
    and a NaN placement fails its threshold regardless.
    """
    if not per_model:
        return {}
    if len(per_model) == 1:
        return per_model[0]
    reduced = {}
    for key in per_model[0]:
        values = [
            model[key] for model in per_model if isinstance(model.get(key), (int, float)) and math.isfinite(model[key])
        ]
        if not values:
            reduced[key] = float("nan")
        elif key in PLACEMENT_METRICS:
            reduced[key] = max(values)
        else:
            reduced[key] = sum(values) / len(values)
    return reduced


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


def _mean(values) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def mean_chain_plddt(plddt, target_len: int | None) -> dict:
    """Split a complex's per-residue pLDDT into a target mean and a binder mean.

    A whole-complex mean confounds the two. A CBLN1 complex is 136 target
    residues against a 40-59 residue binder, so roughly three quarters of such a
    mean is the target folding as well as it always does, and a binder can be
    modelled badly and still clear a threshold on the average. That is the case
    on the ESMFold2 advisory side, where the reported pLDDT does cover the whole
    complex: measured over a real campaign its median tracked the target's to
    three decimals while the binder ranged 0.71 to 0.92.

    It was *not* the case for AF2, whose binder protocol reports its scalar over
    the binder alone -- the split reproduced that number exactly, which is how
    complex_pLDDT came to be retired as a second name for binder_pLDDT rather
    than kept as a coarser one. The target mean is the new information there.

    ColabDesign's binder protocol orders residues target-first, which is the
    same assumption the binder-only losses make via ``_target_len``.
    """
    if plddt is None or target_len is None:
        return {}
    total = len(plddt)
    if not 0 < target_len < total:
        # No boundary means no split worth trusting -- better to emit nothing
        # and leave the columns absent than to label the whole complex as one
        # chain or the other.
        return {}
    values = [float(v) for v in plddt]
    if _mean(values) > _PLDDT_FRACTION_CEILING:
        values = [v / 100.0 for v in values]
    return {
        "target_pLDDT": _mean(values[:target_len]),
        "binder_pLDDT": _mean(values[target_len:]),
    }


def mean_plddt_from_pdb(pdb_path: str) -> float:
    """Mean pLDDT over the CA atoms of a folded structure, as a fraction.

    The folding backends write per-residue pLDDT into the B-factor column --
    ESMFold does it explicitly, ColabFold by convention -- so this is where the
    confidence of a monomer fold lives once the model has been unloaded.

    Parsed by column offset rather than through a structure library: the fixed
    PDB columns are the one thing about the format that is genuinely stable,
    and it keeps this reachable from a test that cannot import atomworks.

    Returns NaN when the file holds nothing usable, so a missing reference
    stays visibly missing instead of becoming a number a ratio would divide by.
    """
    values = []
    try:
        with open(pdb_path) as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    except OSError:
        return float("nan")
    if not values:
        return float("nan")
    mean = _mean(values)
    return mean / 100.0 if mean > _PLDDT_FRACTION_CEILING else mean


def residue_weighted_mean(values: list[float], weights: list[int]) -> float:
    """Mean of per-chain values weighted by how many residues each covers.

    A target's chains are folded separately but its pLDDT in complex is one
    mean over all its residues. Averaging the chain means unweighted would let
    a ten-residue chain count as much as a three-hundred-residue one, and the
    ratio between the two numbers would then measure the chain split rather
    than the target.
    """
    pairs = [
        (float(v), int(w))
        for v, w in zip(values, weights, strict=False)
        if w > 0 and isinstance(v, (int, float)) and math.isfinite(v)
    ]
    if not pairs:
        return float("nan")
    total = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total
