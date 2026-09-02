"""
Monomer evaluation utilities: data classes and default configuration.

Data classes:
  - FoldingResult:        output of a single structure prediction run
  - DesignabilityResult:  full-structure scRMSD values from fold-and-compare

Default constants:
  - RMSD modes, folding models, ProteinMPNN parameters

Column name patterns (written by monomer_eval.compute_monomer_metrics):
  Designability (ProteinMPNN + refold):
    _res_scRMSD_{mode}_{model}           best scRMSD (min over sequences)
    _res_scRMSD_{mode}_{model}_all       all scRMSD values (list)
    _res_scRMSD_single_{mode}_{model}    first ProteinMPNN sequence only
  Codesignability (PDB seq + refold):
    _res_co_scRMSD_{mode}_{model}        best scRMSD
    _res_co_scRMSD_{mode}_{model}_all    all scRMSD values (list)

Note: Thresholds for filtering/analysis are in monomer_analysis_utils.py
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

from loguru import logger

# =============================================================================
# Folding Configuration Constants
# =============================================================================

VALID_RMSD_MODES = ["ca", "bb3o", "all_atom"]
# esmfold2 folds single-chain single-sequence via the Fast checkpoint; see
# folding_models.run_esmfold2 and metrics.esmfold2_loader.
VALID_FOLDING_MODELS = ["esmfold", "esmfold2", "colabfold"]

# Default folding configuration
DEFAULT_DESIGNABILITY_MODES = ["ca"]
DEFAULT_DESIGNABILITY_FOLDING_MODELS = ["esmfold"]
DEFAULT_CODESIGNABILITY_MODES = ["ca", "all_atom"]
DEFAULT_CODESIGNABILITY_FOLDING_MODELS = ["esmfold"]

# ProteinMPNN default parameters
DEFAULT_NUM_SEQ_PER_TARGET = 8
DEFAULT_PMPNN_SAMPLING_TEMP = 0.1


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FoldingResult:
    """Result from a single structure prediction run."""

    pdb_path: str | None  # Path to folded structure, None if failed
    sequence: str
    model_name: str
    success: bool = True
    error: str | None = None


@dataclass
class DesignabilityResult:
    """Full-structure scRMSD values from fold-and-compare (monomer evaluation)."""

    rmsd_values: dict[str, dict[str, list[float]]]  # mode -> model -> list of rmsds
    best_rmsd: dict[str, dict[str, float]]  # mode -> model -> best rmsd
    folded_paths: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)


# =============================================================================
# Refold cache
# =============================================================================
#
# Monomer refolding had no cache: a resumed evaluation refolded every sequence
# even though the binder-complex path beside it reuses everything. That was
# tolerable when the only backend was a single ESMFold forward, and is not once
# esmfold2 -- a diffusion sampler -- can be selected, with num_seq_per_target
# sequences to fold per design.
#
# The sequences are *stored*, not part of the key. They are an output of the
# request, so keying on them would mean running ProteinMPNN to find out whether
# the ProteinMPNN run could be skipped. What stands in for them is everything
# that determines them -- the design, the chains conditioning it, the seed, the
# count and the temperature -- so a hit implies the same sequences without
# generating them. This mirrors binder_eval_cache, which stores its
# sequences_dict for the same reason.

MONOMER_FOLD_CACHE_TEMPLATE = "monomer_fold_cache_{suffix}.json"


def monomer_fold_cache_path(output_dir: str, suffix: str) -> str:
    return os.path.join(output_dir, MONOMER_FOLD_CACHE_TEMPLATE.format(suffix=suffix))


def monomer_fold_fingerprint(
    reference_pdb_path: str,
    suffix: str,
    folding_models: list[str],
    model_identities: dict[str, str],
    num_seq_per_target: int,
    pmpnn_sampling_temp: float,
    binder_chain: str | None,
    mpnn_context_chains: list[str] | None = None,
    mpnn_seed_value: int | None = None,
    inverse_folding_model: str | None = None,
) -> str:
    """Everything that determines the refolds, excluding the sequences themselves.

    ``model_identities`` carries each backend's checkpoint, so switching
    esmfold2's Fast checkpoint for the full one recomputes rather than serving
    structures from the other model. rmsd_modes is deliberately absent: modes are
    recorded per entry so a newly requested mode is a partial miss rather than a
    full invalidation.

    ``mpnn_context_chains``, ``mpnn_seed_value`` and ``inverse_folding_model``
    cover the redesigns, which are
    stored in the cache but are not otherwise keyed on anything. Without them a
    cache written when designability redesigned the binder alone would be served
    for a request that now redesigns it in the target's context -- same design,
    same folding model, entirely different sequences, and the served numbers
    would silently be the old metric under the new name.
    """
    from proteinfoundation.metrics.seeding import SEED_DERIVATION_VERSION

    key = {
        "reference_pdb_path": reference_pdb_path,
        "suffix": suffix,
        "folding_models": sorted(folding_models),
        "model_identities": {k: model_identities[k] for k in sorted(model_identities)},
        "num_seq_per_target": num_seq_per_target,
        "pmpnn_sampling_temp": pmpnn_sampling_temp,
        "binder_chain": binder_chain,
        # The folding seed derives from the stored sequences, which are an output
        # rather than a key, so what needs covering here is the derivation
        # itself: change it and a cached entry would disagree with a fresh
        # computation while the fingerprint stayed put.
        "seed_derivation": SEED_DERIVATION_VERSION,
    }
    # Added only when ProteinMPNN is involved, which keeps the codesignability
    # key byte-identical to what it was before redesign conditioning existed.
    # Codesignability reads the sequence off the PDB, so nothing about it
    # changed, and invalidating its folds -- potentially a diffusion sampler over
    # every design -- to record a fact that does not apply to it would be a real
    # cost for no information.
    if mpnn_context_chains is not None:
        key["mpnn_context_chains"] = sorted(mpnn_context_chains)
    if mpnn_seed_value is not None:
        key["mpnn_seed"] = mpnn_seed_value
    if inverse_folding_model is not None:
        key["inverse_folding_model"] = inverse_folding_model

    canonical = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


MONOMER_CACHE_SCHEMA = 2  # 1 held a single fold; 2 holds one per seed


def _fold_payload(entry: dict) -> dict | None:
    """One seed's stored fold, or None if it holds nothing usable."""
    if not isinstance(entry, dict) or not entry.get("sequences") or not entry.get("rmsd_values"):
        return None
    return entry


def read_monomer_fold_cache(
    output_dir: str, suffix: str, fingerprint: str, seeds: list[int] | None = None
) -> dict | None:
    """Cached refold results for this design, or None. Never raises.

    With *seeds*, returns ``{seed: fold}`` for the seeds present -- possibly a
    subset, possibly empty. Folds are keyed by the SEED VALUE, not by position:
    a seed is what actually determined a result, while "the k-th seed" is only
    meaningful relative to a derivation the key would not record. Keying by value
    also means a pinned seed is an ordinary entry rather than a special case, and
    a derivation change simply misses rather than serving a fold produced under
    the old one.

    Entries from a superseded derivation are never served and are not deleted:
    making a read destructive to reclaim a few hundred bytes is a worse trade than
    letting these files grow slowly.

    Without *seeds*, returns the single fold a schema-1 cache holds, for callers
    not yet asking per seed.
    """
    path = monomer_fold_cache_path(output_dir, suffix)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") != fingerprint:
            logger.info(
                f"Monomer refold cache at {path} was produced by a different request "
                f"({str(cached.get('fingerprint'))[:12]} != {fingerprint[:12]}); recomputing"
            )
            return None
        folds = cached.get("folds")
        if folds is None:
            # Schema 1: one unlabelled fold. It was produced by whatever seed the
            # derivation yields for its own stored sequences, so it can be adopted
            # under that key rather than discarded -- which is what keeps a
            # finished campaign's folds usable when a run starts asking for more
            # than one seed.
            single = _fold_payload(cached)
            if single is None:
                return None
            if seeds is None:
                return single
            from proteinfoundation.metrics.seeding import deterministic_seed

            legacy = deterministic_seed(os.path.basename(output_dir), suffix, *single["sequences"])
            return {legacy: single} if legacy in seeds else {}
        if seeds is None:
            first = next((v for v in folds.values() if _fold_payload(v)), None)
            return first
        present = {}
        for seed in seeds:
            entry = _fold_payload(folds.get(str(seed)))
            if entry is not None:
                present[seed] = entry
        return present
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable monomer refold cache {path}: {exc}")
        return None


def write_monomer_fold_cache(
    output_dir: str,
    suffix: str,
    fingerprint: str,
    result: "DesignabilityResult",
    keep_outputs: bool,
    seed: int | None = None,
    seed_index: int | None = None,
) -> None:
    """Persist refold results. Never raises.

    Values are always stored -- they are a few floats per sequence, so they cost
    nothing even when outputs are being reclaimed. Structure paths are stored only
    when ``keep_outputs`` is set, because that flag exists to free disk and a path
    to a deleted file is worse than no path: with the structures kept, a later run
    asking for an RMSD mode this entry lacks can recompute it from them instead of
    refolding.
    """
    # any() over a dict of {model: [values]} iterates the *lists*, so a non-empty
    # list of infinities was truthy and a wholly failed refold got cached
    # permanently -- under a variable named `finite`. Test the values.
    finite = any(
        math.isfinite(value)
        for by_model in result.rmsd_values.values()
        for values in by_model.values()
        for value in values
    )
    if not finite:
        # Nothing usable was produced. Caching that would make one bad run
        # permanent for every later resume.
        return
    entry = {
        "sequences": list(result.sequences),
        "rmsd_values": result.rmsd_values,
        "best_rmsd": result.best_rmsd,
        "folded_paths": list(result.folded_paths) if keep_outputs else [],
        "structures_kept": bool(keep_outputs),
    }
    from proteinfoundation.metrics.seeding import SEED_DERIVATION_VERSION, deterministic_seed

    path = monomer_fold_cache_path(output_dir, suffix)
    try:
        # MERGE, never replace. Growing three seeds to five must add two entries
        # and keep three, so a write has to read what is there first. A stale
        # fingerprint discards the lot: those folds answered a different request.
        folds = {}
        if os.path.exists(path):
            try:
                with open(path) as handle:
                    existing = json.load(handle)
                if existing.get("fingerprint") == fingerprint:
                    folds = dict(existing.get("folds") or {})
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                folds = {}
        if seed is None:
            seed = deterministic_seed(os.path.basename(output_dir), suffix, *result.sequences)
        entry["seed_index"] = seed_index
        entry["seed_derivation"] = SEED_DERIVATION_VERSION
        folds[str(seed)] = entry
        blob = json.dumps({"fingerprint": fingerprint, "schema": MONOMER_CACHE_SCHEMA, "folds": folds})
        with open(path, "w") as handle:
            handle.write(blob)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"Could not write monomer refold cache for {output_dir}: {exc}")
