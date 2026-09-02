"""The target folded on its own, once per campaign.

``complex_pLDDT`` conflates a target that folds well with a binder that does
not. Splitting it gives a target-in-complex number, but that number is only
interpretable against what the target scores by itself: a target that folds to
0.82 alone has not been damaged by a design that leaves it at 0.80.

The reference depends on the target and the folding model, not on any design,
so it is computed once and cached beside the campaign's evaluation results --
where every shard of that campaign can find it.
"""

import hashlib
import json
import os
import tempfile

from loguru import logger

from proteinfoundation.metrics.ensembling import mean_plddt_from_pdb, residue_weighted_mean
from proteinfoundation.metrics.seeding import deterministic_seeds

TARGET_REFERENCE_CACHE_SCHEMA = 1
TARGET_REFERENCE_CACHE_NAME = "target_alone_fold_cache.json"
TARGET_REFERENCE_DIR = "target_alone"


def target_reference_cache_path(campaign_dir: str) -> str:
    return os.path.join(campaign_dir, TARGET_REFERENCE_CACHE_NAME)


def target_reference_fingerprint(target_seqs: list[str], folding_models: list[str], n_esmfold2_seeds: int) -> str:
    """What the reference depends on, and nothing else.

    Not the design, not the shard, not the campaign name: two campaigns against
    the same target with the same folding settings should reuse one reference
    rather than fold it twice and disagree in the last decimal.
    """
    canonical = json.dumps(
        {
            "schema": TARGET_REFERENCE_CACHE_SCHEMA,
            "sequences": list(target_seqs),
            "folding_models": sorted(folding_models),
            "n_esmfold2_seeds": int(n_esmfold2_seeds),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_target_reference(campaign_dir: str, fingerprint: str) -> dict[str, float] | None:
    path = target_reference_cache_path(campaign_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") != fingerprint:
            return None
        values = cached.get("plddt")
        return {str(k): float(v) for k, v in values.items()} if values else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable target reference cache {path}: {exc}")
        return None


def write_target_reference(campaign_dir: str, fingerprint: str, plddt: dict[str, float]) -> None:
    """Write the reference so a concurrent shard cannot read a half-written one.

    Shards of one campaign run as separate jobs against a shared directory, so
    two of them can reach this at once. Both computing the same reference wastes
    one fold; both writing the same file through a rename costs nothing, since
    the value does not depend on which shard produced it.
    """
    path = target_reference_cache_path(campaign_dir)
    payload = {
        "schema": TARGET_REFERENCE_CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "plddt": plddt,
    }
    try:
        os.makedirs(campaign_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=campaign_dir, delete=False, suffix=".tmp") as handle:
            json.dump(payload, handle)
            tmp = handle.name
        os.replace(tmp, path)
    except OSError as exc:
        # A reference that cannot be cached is still a usable reference. Losing
        # the cache costs one fold per shard, not the campaign.
        logger.warning(f"Could not cache target reference at {path}: {exc}")


def _fold_once(fold_fn, target_seqs, out_dir, model, seed):
    """Residue-weighted mean pLDDT over the target's chains for one fold."""
    folded = fold_fn(
        sequences=target_seqs,
        output_dir=out_dir,
        name=TARGET_REFERENCE_DIR,
        folding_models=[model],
        suffix=TARGET_REFERENCE_DIR,
        cache_dir=None,
        keep_outputs=True,
        seed=seed,
    )
    results = folded.get(model) or []
    values = [mean_plddt_from_pdb(r.pdb_path) for r in results if getattr(r, "pdb_path", None)]
    weights = [len(r.sequence) for r in results if getattr(r, "pdb_path", None)]
    return residue_weighted_mean(values, weights)


def target_alone_plddt(
    target_seqs: list[str],
    campaign_dir: str,
    folding_models: list[str],
    n_esmfold2_seeds: int = 1,
    reuse_cache: bool = True,
    fold_fn=None,
) -> dict[str, float]:
    """Mean pLDDT of the target folded without its binder, per folding model.

    ESMFold2 is a diffusion sampler, so its reference is pooled over the same
    number of seeds the designs are folded with -- a reference drawn once would
    put the sampler's own spread into every ratio measured against it.

    Returns whatever it could compute. A model that fails leaves its key out
    rather than contributing a NaN that a later ratio would silently propagate.
    """
    if not target_seqs or not folding_models:
        return {}

    fingerprint = target_reference_fingerprint(target_seqs, folding_models, n_esmfold2_seeds)
    if reuse_cache:
        cached = read_target_reference(campaign_dir, fingerprint)
        if cached is not None:
            logger.info(f"Target-alone reference reused from cache: {cached}")
            return cached

    if fold_fn is None:  # imported lazily: pulls in torch and the folding stacks
        from proteinfoundation.evaluation.monomer_eval import fold_sequences

        fold_fn = fold_sequences

    out_dir = os.path.join(campaign_dir, TARGET_REFERENCE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    plddt: dict[str, float] = {}
    for model in folding_models:
        seeds = (
            deterministic_seeds(TARGET_REFERENCE_DIR, model, *target_seqs, count=n_esmfold2_seeds)
            if model == "esmfold2"
            else [None]
        )
        per_seed = []
        for seed in seeds:
            try:
                value = _fold_once(fold_fn, target_seqs, out_dir, model, seed)
            except Exception as exc:
                # The reference is a denominator. A model that cannot produce
                # one must leave the ratio absent, not make one up.
                logger.error(f"Target-alone refold failed for {model} (seed={seed}): {exc}")
                continue
            if value == value:  # not NaN
                per_seed.append(value)
        if per_seed:
            plddt[model] = sum(per_seed) / len(per_seed)
        else:
            logger.error(f"No usable target-alone reference for {model}; ratios against it will be absent")

    if plddt:
        logger.info(f"Target-alone reference pLDDT: {plddt}")
        write_target_reference(campaign_dir, fingerprint, plddt)
    return plddt
