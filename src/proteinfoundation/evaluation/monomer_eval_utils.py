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

# Only needs math, so importing it here keeps this module as light as it was.
from proteinfoundation.metrics.ensembling import mean_plddt_from_pdb

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
    # {model: [path or None per sequence]}. Keyed, because a flat list cannot say
    # which structure belongs to which sequence or which model, and anything read
    # off these structures needs both.
    folded_paths: dict[str, list[str | None]] = field(default_factory=dict)
    sequences: list[str] = field(default_factory=list)
    # model -> per-sequence mean pLDDT of the fold, positionally aligned with
    # sequences like everything else here. Empty for folds cached before it was
    # recorded; a reader must treat absence as unmeasured, not as zero.
    plddt: dict[str, list[float]] = field(default_factory=dict)


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


MONOMER_CACHE_SCHEMA = 3  # 1 held a single fold; 2 one per seed; 3 keys folded_paths by model


# =============================================================================
# Derived metrics
# =============================================================================
#
# Read off a kept apo structure rather than reported by the folder, and so not
# part of a fold's identity: the same structure answers for any of them, which
# means changing which are computed must re-read the PDBs already on disk rather
# than refold. The split mirrors the advisory one in metrics/consensus_folding.py
# and exists for the same arithmetic -- on CBLN1 that is the difference between
# re-reading 22k structures and predicting them again.
#
# An apo fold is one chain, so everything defined across an interface is absent
# here on purpose: dSASA, shape complementarity, interface composition. What a
# monomer can say about itself is its own surface and its secondary structure.
MONOMER_DERIVED_SUFFIXES: tuple[str, ...] = (
    "sasa_engine",
    "sasa_radii",
    "binder_sasa",
    "binder_surface_hydrophobicity",
    "binder_ss_counts",
    "binder_ss_total",
)
# Bumped when the derivation of a registered metric changes without its name
# changing, which the name alone cannot express.
MONOMER_DERIVATION_VERSION = 1


def monomer_derivation_fingerprint() -> str:
    """Identity of what is read OFF an apo structure, not of the structure."""
    canonical = json.dumps(
        {"derived": sorted(MONOMER_DERIVED_SUFFIXES), "version": MONOMER_DERIVATION_VERSION},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def derive_from_monomer_structure(pdb_path: str) -> dict:
    """The registered metrics, read off one apo structure.

    Returns ``{}`` while nothing is registered, which keeps the split inert until
    a caller opts in. Raises nothing of its own: a caller treats a failure as "not
    derivable for this structure" and leaves the columns absent, so one unreadable
    PDB does not cost a refold of everything.
    """
    if not MONOMER_DERIVED_SUFFIXES:
        return {}
    from proteinfoundation.utils.pr_alternative_utils import monomer_structure_metrics

    metrics = monomer_structure_metrics(pdb_path)
    return {name: metrics[name] for name in MONOMER_DERIVED_SUFFIXES if name in metrics}


def _entry_needs_derivation(entry: dict, stale: bool) -> bool:
    if stale:
        return True
    derived = entry.get("derived") or {}
    return any(
        any(name not in (derived.get(model) or {}) for name in MONOMER_DERIVED_SUFFIXES)
        for model in folded_paths_by_model(entry)
    )


def derive_for_result(result) -> dict[str, dict[str, list]]:
    """Derived metrics for one in-memory result, with no cache to stamp.

    For the apo ``self`` path, which shares the codesignability fold: its
    structures live under that track's cache and fingerprint, so there is nothing
    here to mark as derived. Re-reading one sequence's structures each run is
    cheaper than reaching into another track's cache file with a fingerprint this
    caller would have to reconstruct -- and reconstructing it wrongly would write
    derived values against folds they did not come from.
    """
    if not MONOMER_DERIVED_SUFFIXES:
        return {}
    entry = {
        "sequences": list(getattr(result, "sequences", []) or []),
        "rmsd_values": getattr(result, "rmsd_values", {}) or {},
        "folded_paths": getattr(result, "folded_paths", {}) or {},
    }
    folds = {0: entry}
    _derive_into(folds, stale=True)
    return (folds[0].get("derived") or {}) if folds else {}


def refresh_monomer_derivation(
    output_dir: str, suffix: str, fingerprint: str, folds: dict[int, dict]
) -> dict[int, dict]:
    """Fill in what is read off the kept apo structures, re-reading not refolding.

    Runs when the derivation fingerprint moved AND when an entry simply lacks a
    key -- so a fold cached before its structure existed heals itself once the PDB
    is there, instead of staying blank forever behind a fingerprint that already
    matches. Fresh folds are filled on the run that produced them rather than the
    one after, which is the difference between a column being there and a campaign
    having to be evaluated twice to populate it.

    Returns *folds* with ``derived`` attached per seed, as
    ``{model: {metric: [value per sequence]}}``, positionally aligned with the
    sequences like everything else here.
    """
    if not MONOMER_DERIVED_SUFFIXES or not folds:
        return folds
    path = monomer_fold_cache_path(output_dir, suffix)
    current = monomer_derivation_fingerprint()
    stored = None
    if os.path.exists(path):
        try:
            with open(path) as handle:
                stored = json.load(handle).get("derivation")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stored = None
    stale = stored != current

    changed = _derive_into(folds, stale)
    if changed or stale:
        _rewrite_monomer_derivation(path, fingerprint, folds, current)
    return folds


def _derive_into(folds: dict, stale: bool) -> bool:
    """Attach ``derived`` to every entry that needs it. True if anything changed."""
    changed, failed = False, 0
    for entry in folds.values():
        if not _entry_needs_derivation(entry, stale):
            continue
        sequences = list(entry.get("sequences") or [])
        derived = {str(m): dict(v) for m, v in (entry.get("derived") or {}).items()}
        for model, paths in folded_paths_by_model(entry).items():
            if len(paths) != len(sequences):
                continue
            per_metric: dict[str, list] = {name: [] for name in MONOMER_DERIVED_SUFFIXES}
            usable = False
            for pdb in paths:
                one = {}
                if pdb and os.path.exists(pdb):
                    try:
                        one = derive_from_monomer_structure(pdb)
                        usable = usable or bool(one)
                    except Exception as exc:
                        failed += 1
                        logger.warning(f"Could not derive apo metrics from {pdb}: {exc}")
                for name in MONOMER_DERIVED_SUFFIXES:
                    per_metric[name].append(one.get(name, math.nan))
            if usable:
                derived[model] = per_metric
                changed = True
        if derived:
            entry["derived"] = derived
    if failed:
        logger.warning(f"Apo derivation failed for {failed} structures; their columns stay absent")
    return changed


def _rewrite_monomer_derivation(path: str, fingerprint: str, folds: dict[int, dict], derivation: str) -> None:
    """Persist the derived values beside the folds they were read from."""
    try:
        existing = {}
        if os.path.exists(path):
            with open(path) as handle:
                blob = json.load(handle)
            if blob.get("fingerprint") == fingerprint:
                existing = dict(blob.get("folds") or {})
        for seed, entry in folds.items():
            stored = existing.get(str(seed))
            if isinstance(stored, dict) and entry.get("derived"):
                stored["derived"] = entry["derived"]
            elif entry.get("derived"):
                existing[str(seed)] = entry
        with open(path, "w") as handle:
            handle.write(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "schema": MONOMER_CACHE_SCHEMA,
                        "derivation": derivation,
                        "folds": existing,
                    }
                )
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(f"Could not persist apo derived metrics for {path}: {exc}")


def folded_paths_by_model(entry: dict) -> dict[str, list[str | None]]:
    """Kept structures as ``{model: [path or None per sequence]}``.

    Schema-3 entries store exactly this. Schema-2 stored one flat list, appended
    model by model with failed folds skipped, which is recoverable only where
    there was one model and one path per sequence -- the same condition the pLDDT
    recovery already refused to guess past. Anything else yields ``{}``, which a
    caller reads as "no structures it can attribute", not as "no structures".
    """
    paths = entry.get("folded_paths")
    if isinstance(paths, dict):
        return {str(model): list(values) for model, values in paths.items()}
    paths = list(paths or [])
    sequences = list(entry.get("sequences") or [])
    models = sorted({m for by_model in (entry.get("rmsd_values") or {}).values() for m in by_model})
    if len(models) == 1 and paths and len(paths) == len(sequences):
        return {models[0]: paths}
    return {}


def _plddt_from_kept_structures(entry: dict) -> dict[str, list[float]]:
    """pLDDT for a fold cached before it was recorded, read back off the
    structures that fold kept.

    Attributable now for any number of models, because the paths say which model
    and which sequence each belongs to. Under the flat list this worked only for a
    single model, and a confidence attributed to the wrong sequence is worse than
    an absent one -- so those folds stayed unmeasured, and would have gone on
    staying unmeasured the moment a second apo folder was enabled.

    This is what makes the column available without refolding: whenever
    keep_folding_outputs held, the structures are already on disk and their
    B-factor column already holds the number.
    """
    sequences = list(entry.get("sequences") or [])
    recovered: dict[str, list[float]] = {}
    for model, paths in folded_paths_by_model(entry).items():
        if len(paths) != len(sequences):
            continue
        values = [mean_plddt_from_pdb(path) if path else math.nan for path in paths]
        # All NaN means the structures are gone or unreadable. Recording that is no
        # better than recording nothing, and nothing is what the caller expects.
        if not all(math.isnan(v) for v in values):
            recovered[model] = values
    return recovered


def _fold_payload(entry: dict) -> dict | None:
    """One seed's stored fold, or None if it holds nothing usable.

    Also the single funnel every cache read goes through, which is why the
    pLDDT recovery hangs here rather than at each of them.
    """
    if not isinstance(entry, dict) or not entry.get("sequences") or not entry.get("rmsd_values"):
        return None
    if not entry.get("plddt"):
        recovered = _plddt_from_kept_structures(entry)
        if recovered:
            # A copy: the caller's cached dict is not ours to edit.
            return {**entry, "plddt": recovered}
    return entry


def _fold_seeds(name: str, suffix: str, sequences: list[str], folding_models: list[str], count: int) -> list[int]:
    """The seeds this design's folds are identified by.

    Only ESMFold2 is a sampler; ESMFold v1 and ColabFold are deterministic given
    their inputs, so asking them for several seeds would fold the same structure
    repeatedly and average it with itself. A run without esmfold2 therefore gets
    exactly one seed however many are configured.
    """
    from proteinfoundation.metrics.seeding import deterministic_seeds

    n = max(1, int(count)) if "esmfold2" in (folding_models or []) else 1
    return deterministic_seeds(name, suffix, *sequences, count=n)


def per_model_plddt(plddt: dict | None, folding_models: list[str], n: int) -> dict[str, list[float]]:
    """Per-model apo pLDDT, padded to the sequence count.

    Folds cached before pLDDT was recorded have none, and a model that produced
    no readable confidence has none either. Both come back as NaN rather than as
    a short list, because every apo column is positionally aligned with the holo
    columns beside it and a short list would silently shift that alignment.
    """
    stored = plddt or {}
    out = {}
    for model in folding_models:
        values = list(stored.get(model) or [])
        out[model] = (values + [float("nan")] * n)[:n]
    return out


def _mean_derived(per_seed: list):
    """Average one derived metric over the seeds that produced it.

    Each entry is a list over sequences whose elements may be floats, packed
    eight-state counts (lists), or provenance strings. Strings come from the first
    seed -- averaging "freesasa" is not a thing -- and a NaN in any seed makes the
    result NaN, the same rule the RMSDs use for infinity: a seed that produced no
    usable value did not produce a slightly worse one.
    """
    usable = [v for v in per_seed if isinstance(v, list)]
    if not usable:
        return per_seed[0] if per_seed else None
    width = min(len(v) for v in usable)
    out = []
    for i in range(width):
        values = [v[i] for v in usable]
        first = values[0]
        if isinstance(first, str):
            out.append(first)
        elif isinstance(first, list):
            if any(not isinstance(v, list) or len(v) != len(first) for v in values):
                out.append(first)
            else:
                out.append([sum(v[j] for v in values) / len(values) for j in range(len(first))])
        else:
            numbers = [float(v) for v in values if isinstance(v, (int, float))]
            out.append(
                sum(numbers) / len(numbers)
                if len(numbers) == len(values) and all(math.isfinite(n) for n in numbers)
                else math.nan
            )
    return out


def average_folds(folds: dict[int, dict]) -> dict | None:
    """One fold's worth of numbers, averaged across the seeds that produced them.

    Seeds are exchangeable draws, so the reduction is a mean -- per sequence, per
    mode, per model, keeping the positional alignment every downstream column
    depends on.

    A non-finite value in any seed makes the average non-finite. That fold failed,
    and averaging a failure into a finite number would turn one failure into a
    slightly worse success -- which is exactly the kind of quiet degradation this
    whole gate is meant to catch rather than produce.

    ``folded_paths`` are concatenated rather than averaged: a path is not a
    measurement, and each is a real structure a reader may want.

    ``plddt`` follows the same rule as the RMSDs, with NaN standing in for the
    RMSDs' infinity: a seed that produced no usable confidence did not produce a
    slightly less confident fold.
    """
    usable = [f for f in (folds or {}).values() if f and f.get("rmsd_values")]
    if not usable:
        return None
    if len(usable) == 1:
        return usable[0]

    merged: dict[str, dict[str, list[float]]] = {}
    for mode in usable[0]["rmsd_values"]:
        merged[mode] = {}
        for model in usable[0]["rmsd_values"][mode]:
            per_seed = [f["rmsd_values"].get(mode, {}).get(model, []) for f in usable]
            width = min((len(v) for v in per_seed), default=0)
            merged[mode][model] = [
                sum(v[i] for v in per_seed) / len(per_seed)
                if all(math.isfinite(v[i]) for v in per_seed)
                else float("inf")
                for i in range(width)
            ]
    merged_plddt: dict[str, list[float]] = {}
    for model in usable[0].get("plddt") or {}:
        per_seed = [f.get("plddt", {}).get(model, []) for f in usable]
        width = min((len(v) for v in per_seed), default=0)
        merged_plddt[model] = [
            sum(v[i] for v in per_seed) / len(per_seed) if all(math.isfinite(v[i]) for v in per_seed) else float("nan")
            for i in range(width)
        ]

    averaged = dict(usable[0])
    averaged["rmsd_values"] = merged
    if merged_plddt:
        averaged["plddt"] = merged_plddt
    # Concatenated per model rather than across models: a path is not a
    # measurement, and each is a real structure a reader may want -- but which
    # model produced it is part of what makes it readable.
    merged_paths: dict[str, list[str | None]] = {}
    for fold in usable:
        for model, paths in folded_paths_by_model(fold).items():
            merged_paths.setdefault(model, []).extend(paths)
    averaged["folded_paths"] = merged_paths
    # Derived metrics average over seeds exactly as the RMSDs and pLDDTs do: three
    # seeds are three structures, and reporting the first one's secondary
    # structure beside confidences that are means of three would be two different
    # claims on one row. Provenance strings are taken, not averaged.
    merged_derived: dict[str, dict[str, list]] = {}
    for model in {m for fold in usable for m in (fold.get("derived") or {})}:
        per_seed = [fold["derived"][model] for fold in usable if model in (fold.get("derived") or {})]
        merged_derived[model] = {
            name: _mean_derived([seed.get(name) for seed in per_seed])
            for name in {n for seed in per_seed for n in seed}
        }
    if merged_derived:
        averaged["derived"] = merged_derived
    averaged["n_seeds"] = len(usable)
    return averaged


def read_monomer_folds(
    output_dir: str, suffix: str, fingerprint: str, name: str | None = None
) -> dict[int, dict] | None:
    """Every stored fold for this design, as ``{seed: fold}``, or None.

    Separate from :func:`read_monomer_fold_cache` because a caller cannot always
    name its seeds up front: the seed derives from the redesigned sequences, and
    those are an *output* of the expensive step this cache exists to skip. So the
    order has to be read-then-derive -- take the sequences from any stored fold,
    derive the seeds from them, and only run ProteinMPNN if nothing is stored.
    Deriving first would re-run the inverse folder on every resume.

    Schema-1 caches yield their single fold under the seed the derivation gives
    for their own stored sequences, so a finished campaign keeps its folds.

    *name* must match what the caller passes to ``_fold_seeds``. It defaults to
    the output directory's basename, which is what the codesignability path uses
    -- but the apo path folds under ``<binder>_apo_<seq_type>`` while caching
    beside the sample, so the default adopted its legacy fold under a seed nothing
    would ever ask for. The symptom was four entries for three seeds: three folded
    fresh, one orphan unreachable.
    """
    path = monomer_fold_cache_path(output_dir, suffix)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") != fingerprint:
            return None
        folds = cached.get("folds")
        if folds is None:
            single = _fold_payload(cached)
            if single is None:
                return None
            from proteinfoundation.metrics.seeding import deterministic_seed

            legacy = deterministic_seed(name or os.path.basename(output_dir), suffix, *single["sequences"])
            return {legacy: single}
        out = {}
        for key, entry in folds.items():
            payload = _fold_payload(entry)
            if payload is not None:
                out[int(key)] = payload
        return out or None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable monomer refold cache {path}: {exc}")
        return None


def read_monomer_fold_cache(
    output_dir: str, suffix: str, fingerprint: str, seeds: list[int] | None = None, name: str | None = None
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

            legacy = deterministic_seed(name or os.path.basename(output_dir), suffix, *single["sequences"])
            return {legacy: single} if legacy in seeds else {}
        if seeds is None:
            # Take what _fold_payload returns, not the entry it was asked
            # about: it normalises now, so discarding its result would throw
            # away the pLDDT it just recovered.
            return next((p for p in (_fold_payload(v) for v in folds.values()) if p), None)
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
    name: str | None = None,
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
        "folded_paths": {m: list(v) for m, v in (result.folded_paths or {}).items()} if keep_outputs else {},
        "plddt": result.plddt,
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
                    if not folds:
                        # A schema-1 file about to be overwritten. Reading adopts
                        # its single fold under the seed that produced it; without
                        # doing the same here the first write drops it, and the
                        # adoption is undone every run -- the fold is reused in
                        # memory and refolded on the next resume, forever.
                        legacy = _fold_payload(existing)
                        if legacy is not None:
                            legacy_seed = deterministic_seed(
                                name or os.path.basename(output_dir), suffix, *legacy["sequences"]
                            )
                            folds[str(legacy_seed)] = legacy
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                folds = {}
        if seed is None:
            seed = deterministic_seed(name or os.path.basename(output_dir), suffix, *result.sequences)
        entry["seed_index"] = seed_index
        entry["seed_derivation"] = SEED_DERIVATION_VERSION
        folds[str(seed)] = entry
        blob = json.dumps(
            {
                "fingerprint": fingerprint,
                "schema": MONOMER_CACHE_SCHEMA,
                # Stamped even though this write carries no derived values: the
                # refresh pass runs right after and compares against it, and a
                # file with no stamp reads as stale on every run forever.
                "derivation": monomer_derivation_fingerprint(),
                "folds": folds,
            }
        )
        with open(path, "w") as handle:
            handle.write(blob)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"Could not write monomer refold cache for {output_dir}: {exc}")
