"""
Monomer evaluation: designability, codesignability, novelty, sequence recovery.

Pipeline per sample:
  1. (Optional) Extract binder chain from complex PDB
  2. Designability:    ProteinMPNN -> fold -> full-structure scRMSD
  3. Codesignability:  PDB sequence -> fold -> full-structure scRMSD
  4. Sequence recovery, secondary structure, novelty

Key design choices:
  - Folding (fold_sequences) is separated from RMSD (compute_scrmsd_from_folded)
    so motif_eval can reuse fold_sequences and add motif-region RMSD on top.
  - Binder complexes: binder chain is extracted to a monomer PDB for evaluation.

See monomer_eval_utils.py for data classes and column name patterns.
"""

import math
import os
import shutil
from typing import Literal

import numpy as np
import pandas as pd
import torch
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from omegaconf import DictConfig

from proteinfoundation.evaluation.binder_eval_utils import (
    dedupe_columns,
    extract_binder_chain_to_pdb,
    get_binder_chain_from_complex,
)
from proteinfoundation.evaluation.monomer_eval_utils import (
    MONOMER_CONFIDENCE_SUFFIXES,
    DesignabilityResult,
    FoldingResult,
    _derive_into,
    _fold_seeds,
    average_folds,
    folded_paths_by_model,
    legacy_fold_fingerprints,
    merge_model_folds,
    monomer_fold_fingerprint,
    read_monomer_folds,
    write_monomer_fold_cache,
)
from proteinfoundation.evaluation.motif_eval_utils import compute_and_store_ss
from proteinfoundation.evaluation.utils import maybe_tqdm, parse_cfg_for_table, redesign_conditioning
from proteinfoundation.metrics.ensembling import mean_plddt_from_pdb
from proteinfoundation.metrics.folding_models import colabfold_model_siblings, read_fold_confidence
from proteinfoundation.metrics.inverse_folding_models import (
    DEFAULT_INVERSE_FOLDING_MODEL,
    inverse_fold,
    resolve_inverse_folding_model,
)
from proteinfoundation.metrics.metric_utils import rmsd_metric
from proteinfoundation.metrics.novelty import novelty_from_list
from proteinfoundation.metrics.seeding import MPNN_OMIT_AAS, mpnn_seed
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb, load_pdb, pdb_name_from_path

# =============================================================================
# Structure Prediction
# =============================================================================


def _chain_ids(pdb_path: str) -> list[str]:
    """Chain IDs present in a PDB, or [] if it cannot be read.

    Empty on failure rather than raising: this backs a provenance check, and a
    check that cannot run must not take the evaluation down with it.
    """
    try:
        return sorted(set(load_any(pdb_path)[0].chain_id.tolist()))
    except Exception as exc:
        logger.warning(f"Could not read chains from {pdb_path} to verify redesign context: {exc}")
        return []


def designability_mpnn_chains(binder_chain: str | None, target_chains: list[str] | None = None) -> list[str]:
    """The chains ProteinMPNN *sees* when redesigning a binder for designability.

    Its context, not the set it redesigns: only the binder is ever redesigned,
    while the target is present so the redesign is interface-aware.

    A single definition so the redesigns and the ``redesign_conditioning`` column
    that describes them cannot disagree -- the column is derived from this list
    rather than asserted alongside it. For a plain monomer, or a binder whose
    target chains could not be determined, there is no context to add and this
    is the binder alone.

    The question a binder backbone has to answer is not "is this redesignable in
    general" but "is this redesignable *for binding this target*", which is why
    the target is here at all. It does change what designability measures; see
    ``docs/design-notes/apo-holo-redesign-sharing.md``.
    """
    chain_to_design = binder_chain if binder_chain is not None else "A"
    return list(target_chains or []) + [chain_to_design]


def get_sequences_for_evaluation(
    pdb_path: str,
    use_pdb_seq: bool = True,
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    tmp_path: str | None = None,
    binder_chain: str | None = None,
    mpnn_pdb_path: str | None = None,
    target_chains: list[str] | None = None,
    inverse_folding_model: str = DEFAULT_INVERSE_FOLDING_MODEL,
) -> list[str]:
    """
    Get sequences for structure prediction evaluation.

    Args:
        pdb_path: Path to the PDB file
        use_pdb_seq: If True, use sequence from PDB; if False, run ProteinMPNN
        num_seq_per_target: Number of sequences to generate with ProteinMPNN
        pmpnn_sampling_temp: ProteinMPNN sampling temperature
        tmp_path: Temporary directory for ProteinMPNN output
        binder_chain: Chain ID of the binder (for ProteinMPNN). Defaults to "A" if None.
        mpnn_pdb_path: Structure ProteinMPNN reads. Defaults to *pdb_path*. For a
            binder these differ: the redesign is conditioned on the complex while
            everything downstream -- folding, RMSD -- is about the binder alone.
        target_chains: Target chain IDs present in *mpnn_pdb_path*, giving the
            redesign its context. None for a plain monomer.
        inverse_folding_model: Which inverse folder redesigns the binder. The
            same ``metric.inverse_folding_model`` the complex track uses, so both
            tracks judge the same sequences rather than two models' opinions of
            the same backbone.

    Returns:
        List of sequences to evaluate
    """
    if use_pdb_seq:
        logger.debug("Using sequence from PDB file")
        return [extract_seq_from_pdb(pdb_path)]
    else:
        logger.debug("Running ProteinMPNN for sequence design")
        if tmp_path is None:
            tmp_path = os.path.dirname(pdb_path)

        # Which chain to design (default "A" for monomers), and which chains
        # ProteinMPNN gets to see while designing it. Only the binder is ever
        # redesigned; the target is context.
        chain_to_design = binder_chain if binder_chain is not None else "A"
        context_chains = designability_mpnn_chains(binder_chain, target_chains)
        design_pdb = mpnn_pdb_path or pdb_path

        # What ProteinMPNN actually conditions on is the *file* -- every chain in
        # it is context, and --pdb_path_chains only says which to redesign.
        # all_chains is bookkeeping for fixed positions. So the reported
        # conditioning is a claim about design_pdb, and checking it against the
        # file costs one load against a subprocess launch. Without the check,
        # passing the complex while describing it as the binder alone would label
        # the metric wrongly and nothing would notice.
        # Reported, never repaired. The cache fingerprint and the provenance
        # column both derive from the declared chains, so silently substituting
        # the observed ones here would desync the seed from the key that is
        # supposed to describe it. A mismatch means the caller is wrong; say so
        # and let it be fixed there.
        actual = set(_chain_ids(design_pdb))
        if actual and actual != set(context_chains):
            logger.error(
                f"Redesign context mismatch for {design_pdb}: ProteinMPNN sees chains {sorted(actual)} "
                f"but this run is described as conditioned on {sorted(context_chains)}. "
                f"The redesigns are conditioned on the file; redesign_conditioning and the fold "
                f"cache key are not describing them correctly."
            )

        # Seeded so a resumed run reproduces the redesigns rather than drawing
        # new ones, and so the numbers computed from them are a property of the
        # design rather than of when the job happened to run.
        seed = mpnn_seed(pdb_name_from_path(design_pdb), context_chains, [chain_to_design])

        gen_seqs = inverse_fold(
            model_type=inverse_folding_model,
            pdb_file_path=design_pdb,
            out_dir_root=tmp_path,
            all_chains=context_chains,
            pdb_path_chains=[chain_to_design],
            num_seq_per_target=num_seq_per_target,
            omit_AAs=MPNN_OMIT_AAS,
            sampling_temp=pmpnn_sampling_temp,
            seed=seed,
        )
        return [v["seq"] for v in gen_seqs]


def fold_sequences(
    sequences: list[str],
    output_dir: str,
    name: str,
    folding_models: list[Literal["af2", "esmfold2", "esmfold"]] = ["esmfold"],
    suffix: str = "fold",
    cache_dir: str | None = None,
    keep_outputs: bool = False,
    seed: int | None = None,
) -> dict[str, list[FoldingResult]]:
    """
    Fold sequences using specified structure prediction models.

    This function separates the folding step from RMSD calculation,
    allowing for more flexible evaluation pipelines.

    Args:
        sequences: List of sequences to fold
        output_dir: Directory for folding outputs
        name: Base name for output files
        folding_models: List of folding models to use
        suffix: Suffix for output files
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep output files after evaluation

    Returns:
        Dictionary mapping model names to lists of FoldingResults
    """
    from proteinfoundation.metrics.folding_models import fold_monomer

    # Set cache directory (expand ~ to home directory)
    if os.getenv("CACHE_DIR"):
        cache_dir = os.path.expanduser(os.getenv("CACHE_DIR"))
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.environ["TORCH_HOME"] = cache_dir

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for model in folding_models:
        logger.info(f"Running {model} on {len(sequences)} sequences")

        # Scoped by track, not just by backend. Two tracks share one output_dir --
        # designability and codesignability both pass tmp_dir, the apo track passes
        # sample_root_path for both apo_self and apo_mpnn -- and ESMFold/ESMFold2
        # survive that because they receive `name` and `suffix` and encode both in
        # their filenames. ColabFold does not: it names queries positionally
        # (seq_1..seq_N) and collects by that prefix, so a second track's kept
        # copies sit in the same directory answering to the same names, and the
        # collector can return another track's structure for another sequence.
        model_output_dir = os.path.join(output_dir, f"{model}_output", suffix or "default")
        os.makedirs(model_output_dir, exist_ok=True)

        try:
            out_paths = fold_monomer(
                model,
                sequences,
                model_output_dir,
                name,
                suffix=suffix,
                cache_dir=cache_dir,
                keep_outputs=keep_outputs,
                seed=seed,
            )

            # Convert paths to FoldingResults
            model_results = []
            for i, path in enumerate(out_paths):
                if path is None:
                    model_results.append(
                        FoldingResult(
                            pdb_path=None,
                            sequence=sequences[i],
                            model_name=model,
                            success=False,
                            error="Folding failed",
                        )
                    )
                else:
                    model_results.append(
                        FoldingResult(
                            pdb_path=path,
                            sequence=sequences[i],
                            model_name=model,
                            success=True,
                        )
                    )
            results[model] = model_results

        except Exception as e:
            logger.error(f"Error running {model}: {e}")
            results[model] = [
                FoldingResult(
                    pdb_path=None,
                    sequence=seq,
                    model_name=model,
                    success=False,
                    error=str(e),
                )
                for seq in sequences
            ]

    return results


def _mean_over_models(values) -> float:
    """Mean of the models that produced a usable number, NaN if none did.

    A model that reported nothing did not report a slightly worse value, so it is
    dropped rather than folded in as a zero -- and a metric with nothing finite
    behind it stays NaN, which every reader already treats as unmeasured. Same
    rule reduce_rmsd_over_models applies on the holo side.
    """
    usable = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(usable) / len(usable) if usable else float("nan")


def _rmsd_over_models(paths: list[str], ref_coors, ref_mask, rmsd_modes: list[str]) -> dict[str, float]:
    """Each mode's RMSD, meaned over the predictions of one sequence.

    Mean rather than the worst case, which is what the holo track reserves for
    PLACEMENT metrics: there is no target here and nothing to be placed against,
    so the spread between models is uncertainty about one structure rather than
    disagreement about a location. It is also the reduction ESMFold2's seeds
    already get from average_folds, which keeps the two apo backends comparable.

    A structure that could not be read is skipped; inf only when none could, which
    is the value the caller already uses for a fold that did not happen.
    """
    per_model: list[dict[str, float]] = []
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        folded_prot = load_pdb(path)
        folded_coors = torch.tensor(folded_prot.atom_positions, dtype=torch.float32)
        folded_mask = torch.tensor(folded_prot.atom_mask, dtype=torch.bool)
        mask = ref_mask * folded_mask
        per_model.append(
            {
                mode: rmsd_metric(
                    coors_1_atom37=ref_coors,
                    coors_2_atom37=folded_coors,
                    mask_atom_37=mask,
                    mode=mode,
                )
                for mode in rmsd_modes
            }
        )
    if not per_model:
        return {mode: float("inf") for mode in rmsd_modes}
    return {mode: _mean_over_models(m[mode] for m in per_model) for mode in rmsd_modes}


def compute_scrmsd_from_folded(
    reference_pdb_path: str,
    folding_results: dict[str, list[FoldingResult]],
    rmsd_modes: list[Literal["ca", "bb3", "bb3o", "all_atom"]] = ["ca"],
) -> DesignabilityResult:
    """
    Compute scRMSD from pre-folded structures.

    This function computes RMSD between the reference structure and
    folded structures, separated from the folding step for cleaner architecture.

    Args:
        reference_pdb_path: Path to the reference PDB structure
        folding_results: Dictionary of folding results from fold_sequences()
        rmsd_modes: Which atoms to use for RMSD calculation

    Returns:
        DesignabilityResult with RMSD values for each mode and model
    """
    # Load reference structure
    ref_prot = load_pdb(reference_pdb_path)
    ref_coors = torch.tensor(ref_prot.atom_positions, dtype=torch.float32)
    ref_mask = torch.tensor(ref_prot.atom_mask, dtype=torch.bool)

    rmsd_values = {mode: {} for mode in rmsd_modes}
    plddt: dict[str, list[float]] = {}
    # Folder-reported, so unlike pLDDT they cannot be recovered from the
    # structure: they come from the sidecar the backend wrote beside it, and are
    # NaN wherever there is none.
    confidence: dict[str, dict[str, list[float]]] = {}
    # Keyed by model, one slot per sequence, None where a fold failed. It used to
    # be one flat list appended model by model with failures skipped, which threw
    # away the only thing that says which structure belongs to which sequence and
    # which model -- so anything read OFF these structures had to refuse to guess,
    # and did.
    folded_paths: dict[str, list[str | None]] = {}

    for model_name, results in folding_results.items():
        for mode in rmsd_modes:
            rmsd_values[mode][model_name] = []
        plddt[model_name] = []
        confidence[model_name] = {name: [] for name in MONOMER_CONFIDENCE_SUFFIXES}
        folded_paths[model_name] = []

        for result in results:
            if not result.success or result.pdb_path is None:
                for mode in rmsd_modes:
                    rmsd_values[mode][model_name].append(float("inf"))
                plddt[model_name].append(float("nan"))
                for name in MONOMER_CONFIDENCE_SUFFIXES:
                    confidence[model_name][name].append(float("nan"))
                folded_paths[model_name].append(None)
                continue

            folded_paths[model_name].append(result.pdb_path)

            # One entry per prediction of this sequence. For ESMFold and ESMFold2
            # that is the one structure returned; for ColabFold it is all five AF2
            # parameter sets, which ran anyway and whose disagreement is the point
            # of having run them. Reducing over the set here rather than reporting
            # the top-ranked one is the same rule the holo track applies in
            # average_af2_stats -- a mean, never a best-of, because a best-of
            # discards exactly the uncertainty five models were run to measure.
            ensemble = colabfold_model_siblings(result.pdb_path)
            for name in MONOMER_CONFIDENCE_SUFFIXES:
                confidence[model_name][name].append(
                    _mean_over_models(read_fold_confidence(path).get(name) for path in ensemble)
                )
            # Read here because this is already the one place that opens every
            # folded structure. The backends write per-residue pLDDT into the
            # B-factor column, so it costs a parse of a file being parsed anyway.
            plddt[model_name].append(_mean_over_models(mean_plddt_from_pdb(p) for p in ensemble))

            try:
                per_mode = _rmsd_over_models(ensemble, ref_coors, ref_mask, rmsd_modes)
                for mode in rmsd_modes:
                    rmsd_values[mode][model_name].append(per_mode[mode])
            except Exception as e:
                logger.error(f"Error computing RMSD for {result.pdb_path}: {e}")
                for mode in rmsd_modes:
                    rmsd_values[mode][model_name].append(float("inf"))


    return DesignabilityResult(
        rmsd_values=rmsd_values,
        folded_paths=folded_paths,
        plddt=plddt,
        confidence=confidence,
    )


def _result_from_folds(folds: dict[int, dict], rmsd_modes: list[str], pdb_path: str):
    """Average per-seed folds into one result, or None if none are usable.

    Missing modes are filled in per SEED and then averaged, never the other way
    round. ``average_folds`` reduces the RMSDs over seeds to one value per
    sequence but concatenates the structures, so an averaged three-seed entry
    holds two RMSDs and six paths: measuring a new mode there produced six
    values against two sequences, and the list that reached the row was three
    times too long and aligned with nothing. Filling first keeps every list one
    entry per sequence, and the new mode is averaged over seeds exactly as the
    cached one was.
    """
    averaged = _averaged_entry(folds, rmsd_modes, pdb_path)
    return _result_from_cache(averaged, rmsd_modes, pdb_path) if averaged else None


def _averaged_entry(
    folds: dict[int, dict], rmsd_modes: list[str], pdb_path: str, derive: bool = False
) -> dict:
    """One backend's seeds, each filled in and then averaged into one entry.

    *derive* reads the registered structure metrics off each seed's structures
    before averaging, which is the only moment they can be read: ``average_folds``
    concatenates the paths, so a three-seed entry holds one sequence and three
    structures and the one-path-per-sequence alignment every reader checks is
    gone. ``average_folds`` then averages the derived values over seeds exactly
    as it does the confidences.

    Off by default because it costs a PDB re-read per seed per backend, and the
    designability track that shares this function has no use for the columns.
    """
    filled = {seed: _fill_missing_modes(entry, rmsd_modes, pdb_path) for seed, entry in (folds or {}).items()}
    if derive and filled:
        _derive_into(filled, stale=False)
    return average_folds(filled) or {}


def _result_from_model_folds(
    per_model: dict[str, dict[int, dict]], rmsd_modes: list[str], pdb_path: str, derive: bool = False
):
    """One result from several backends, each averaged over its own seeds first.

    Seeds are a property of a backend -- a sampler wants several, a deterministic
    folder wants one -- so averaging has to happen per backend and the merge
    after it. Doing it the other way round would average a three-seed ESMFold2
    mean with a single ColabFold value as though they were two draws of one
    thing.
    """
    averaged = {}
    for model, folds in (per_model or {}).items():
        one = _averaged_entry(folds, rmsd_modes, pdb_path, derive=derive)
        if one:
            averaged[model] = one
    merged = merge_model_folds(averaged)
    return _result_from_cache(merged, rmsd_modes, pdb_path) if merged else None


def _fill_missing_modes(entry: dict, rmsd_modes: list[str], reference_pdb_path: str) -> dict:
    """One fold's missing RMSD modes, measured from the structures it kept.

    This is the payoff for keeping them: adding a mode costs a reload rather than
    a refold. Returns the entry unchanged when its structures cannot answer --
    they were not kept, they are gone from disk, or there is not exactly one per
    sequence -- and the caller then decides to refold. The last of those is what
    keeps an averaged multi-seed entry out of here; see :func:`_result_from_folds`.

    A None slot is a fold that failed, not a structure that went missing: its
    RMSD is inf in every mode, so a new mode costs nothing for it and refolding
    it would only fail again. Requiring every slot to be a readable file meant
    one failed sequence forced a full refold of its design the first time a mode
    was added -- which is exactly when refolding is the thing being avoided.
    """
    have = entry.get("rmsd_values") or {}
    missing = [m for m in rmsd_modes if m not in have]
    if not missing:
        return entry

    sequences = list(entry.get("sequences") or [])
    paths = folded_paths_by_model(entry)
    present = [pth for by_model in paths.values() for pth in by_model if pth]
    aligned = bool(paths) and all(len(by_model) == len(sequences) for by_model in paths.values())
    if not (entry.get("structures_kept") and paths and aligned and all(os.path.exists(p) for p in present)):
        return entry

    synthetic = {
        model: [
            FoldingResult(
                pdb_path=pth,
                sequence=seq,
                model_name=model,
                success=pth is not None,
                error=None if pth else "Folding failed",
            )
            for pth, seq in zip(by_model, sequences, strict=True)
        ]
        for model, by_model in paths.items()
    }
    try:
        extra = compute_scrmsd_from_folded(
            reference_pdb_path=reference_pdb_path,
            folding_results=synthetic,
            rmsd_modes=missing,
        )
    except Exception as exc:
        logger.warning(f"Could not measure mode(s) {missing} from {len(present)} kept structure(s): {exc}")
        return entry

    logger.info(f"Cached refold lacked mode(s) {missing}; measured from {len(present)} kept structure(s)")
    filled = dict(entry)
    filled["rmsd_values"] = {**have, **extra.rmsd_values}
    # The re-read opened every structure, so it also read the confidence sidecars:
    # a fold kept before those existed gains its pTM and PAE here without being
    # folded again, wherever the backend has since written one.
    if extra.confidence and not entry.get("confidence"):
        filled["confidence"] = extra.confidence
    return filled


def _result_from_cache(
    cached: dict,
    rmsd_modes: list[str],
    reference_pdb_path: str,
) -> DesignabilityResult | None:
    """Rebuild a result from cache, or None to recompute.

    Two outcomes once :func:`_fill_missing_modes` has had its turn: every
    requested mode is present and the entry is rebuilt, or one is still missing
    and the caller refolds.

    The partial path used to be limited to a single folding model, because
    folded_paths was flat across models and could not be split back apart -- so
    with two models a partial miss refolded. Schema 3 keys the paths by model, and
    the restriction is gone with it.
    """
    # A single-fold entry is already one structure per sequence, so it can be
    # filled here; a multi-seed average cannot, and _result_from_folds has
    # filled its seeds before averaging them.
    cached = _fill_missing_modes(cached, rmsd_modes, reference_pdb_path)
    have = cached["rmsd_values"]
    missing = [m for m in rmsd_modes if m not in have]
    sequences = list(cached.get("sequences") or [])
    paths = folded_paths_by_model(cached)

    if not missing:
        logger.info(f"Reusing cached refold for {len(sequences)} sequence(s), modes {rmsd_modes}")
        return DesignabilityResult(
            rmsd_values={m: have[m] for m in rmsd_modes},
            folded_paths=paths,
            sequences=sequences,
            plddt=dict(cached.get("plddt") or {}),
            confidence=dict(cached.get("confidence") or {}),
            # Carried across this boundary, not recomputed beyond it: the entry
            # is an average whose paths are a seed-major concatenation, so this
            # is the last place the per-seed derivation survives.
            derived={str(m): dict(v) for m, v in (cached.get("derived") or {}).items()},
        )

    why = (
        "structures were not kept"
        if not cached.get("structures_kept")
        else "structures are missing from disk or do not match the sequences one for one"
        if paths
        else "no kept structures could be attributed to a model"
    )
    logger.info(f"Cached refold lacks mode(s) {missing} and {why}; refolding")
    return None


def fold_and_measure_seeds(
    *,
    sequences: list[str],
    reference_pdb_path: str,
    output_dir: str,
    name: str,
    suffix: str,
    model: str,
    fingerprint: str,
    seeds: list[int],
    rmsd_modes: list[str],
    keep_outputs: bool,
    stored: dict[int, dict] | None = None,
    fold_cache_dir: str | None = None,
    what: str = "seeds",
) -> dict[int, dict]:
    """One backend's per-seed folds for one design: reuse, fold, measure, cache.

    The measurement half of both refold tracks. Codesignability/designability
    (:func:`evaluate_self_consistency`) and the apo track
    (:func:`~proteinfoundation.evaluation.binder_eval.apo_refold`) ask different
    questions of different sequences, but they answer them the same way -- same
    seeds, same folder, same RMSD against the same designed backbone, same cache
    -- and they used to do it through two copies of this loop.

    The copies had already drifted twice. One derived structure metrics per seed
    and the other after averaging, which cost self_apo_esmfold2 six columns until
    it was found; and only one recorded ``structures_kept``, so a fresh apo entry
    claimed its structures were gone and could never fill a newly requested RMSD
    mode from them. Both are the same bug: an entry that means one thing on one
    path and another thing on the other. One builder, one meaning.

    Returns ``{seed: entry}``. *stored* is what a cache read already produced;
    seeds found there are passed through untouched.
    """
    per_seed: dict[int, dict] = {seed: stored[seed] for seed in seeds if stored and seed in stored}
    if per_seed and len(per_seed) < len(seeds):
        logger.info(f"{len(per_seed)}/{len(seeds)} {what} cached for {name} ({model}); folding the rest")

    for seed in seeds:
        if seed in per_seed:
            continue
        folded = fold_sequences(
            sequences=sequences,
            output_dir=output_dir,
            name=name,
            folding_models=[model],
            suffix=suffix,
            cache_dir=fold_cache_dir,
            keep_outputs=keep_outputs,
            seed=seed,
        )
        scored = compute_scrmsd_from_folded(
            reference_pdb_path=reference_pdb_path,
            folding_results=folded,
            rmsd_modes=rmsd_modes,
        )
        scored.sequences = sequences
        write_monomer_fold_cache(
            output_dir,
            suffix,
            fingerprint,
            scored,
            keep_outputs,
            seed=seed,
            seed_index=seeds.index(seed),
            name=name,
            model=model,
        )
        per_seed[seed] = _fold_entry(scored, keep_outputs)
    return per_seed


def _fold_entry(scored, keep_outputs: bool) -> dict:
    """One seed's fold as the caches and reducers expect it.

    The same shape :func:`write_monomer_fold_cache` persists, built in one place
    so an in-memory entry and the one read back from disk cannot disagree about
    what a fold recorded -- which is how an apo entry came to lack
    ``structures_kept`` while its own cache file carried it.
    """
    return {
        "sequences": list(scored.sequences),
        "rmsd_values": scored.rmsd_values,
        "folded_paths": {m: list(v) for m, v in (scored.folded_paths or {}).items()},
        "plddt": scored.plddt,
        "confidence": scored.confidence,
        "structures_kept": bool(keep_outputs),
    }


def evaluate_self_consistency(
    pdb_path: str,
    output_dir: str,
    use_pdb_seq: bool = False,
    rmsd_modes: list[Literal["ca", "bb3", "bb3o", "all_atom"]] = ["ca"],
    folding_models: list[Literal["af2", "esmfold2", "esmfold"]] = ["esmfold"],
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
    binder_chain: str | None = None,
    reuse_cache: bool = True,
    mpnn_pdb_path: str | None = None,
    target_chains: list[str] | None = None,
    inverse_folding_model: str = DEFAULT_INVERSE_FOLDING_MODEL,
    n_esmfold2_seeds: int = 1,
    derive_structure_metrics: bool = False,
) -> DesignabilityResult:
    """
    Unified function to evaluate designability/codesignability.

    This is the main entry point that combines:
    1. Sequence generation (ProteinMPNN or PDB sequence)
    2. Structure prediction (folding)
    3. RMSD calculation

    Args:
        pdb_path: Path to the reference PDB structure
        output_dir: Directory for output files
        use_pdb_seq: If True, use PDB sequence (codesignability); if False, use ProteinMPNN
        rmsd_modes: Which atoms to use for RMSD
        folding_models: Folding models to use
        num_seq_per_target: Number of ProteinMPNN sequences (only if use_pdb_seq=False)
        pmpnn_sampling_temp: ProteinMPNN temperature (only if use_pdb_seq=False)
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep folding outputs
        binder_chain: Chain ID of the binder for ProteinMPNN (only if use_pdb_seq=False)
        reuse_cache: Reuse a cached refold of this design when the request matches.
            Values are always cached; folded structures only when keep_outputs is
            set, in which case a newly requested RMSD mode can be computed from
            them instead of refolding.
        mpnn_pdb_path: Structure ProteinMPNN reads, if not *pdb_path*. For a
            binder this is the complex, so the redesign is interface-aware, while
            *pdb_path* stays the binder alone -- what gets folded and what the
            RMSD is measured against. Ignored when use_pdb_seq is True.
        target_chains: Target chain IDs in *mpnn_pdb_path*.
        inverse_folding_model: Which inverse folder produces the redesigns.
            Ignored when use_pdb_seq is True.
        derive_structure_metrics: Also read the registered metrics off the folded
            structures (SASA, secondary structure, surface hydrophobicity). Done
            per seed, before averaging, which is the only point at which there is
            one structure per sequence to read. Off by default: it costs a PDB
            re-read per seed per backend, and the designability track does not
            report those columns. The apo track does, which is why it asks.

    Returns:
        DesignabilityResult with all RMSD values
    """
    name = pdb_name_from_path(pdb_path)
    os.makedirs(output_dir, exist_ok=True)

    suffix = "pdb" if use_pdb_seq else "mpnn"

    # Reuse a previous refold of this design when nothing that determines it has
    # changed. Sequences are stored rather than keyed on -- they are an output of
    # the request -- so the fingerprint has to cover everything that produces
    # them, the redesign context and seed included.
    from proteinfoundation.metrics.folding_models import folding_model_identity

    # Codesignability reads the sequence off the PDB, so no ProteinMPNN runs and
    # there is no conditioning or seed to key on.
    if use_pdb_seq:
        context_chains = None
        seed_value = None
    else:
        context_chains = designability_mpnn_chains(binder_chain, target_chains)
        seed_value = mpnn_seed(
            pdb_name_from_path(mpnn_pdb_path or pdb_path),
            context_chains,
            [binder_chain if binder_chain is not None else "A"],
        )

    # One fingerprint and one cache file per backend, so enabling a second one
    # does not discard the first one's folds for a reason that has nothing to do
    # with how it folded them. The per-model fingerprint is this same function
    # called with a one-element list, which is byte-identical to what a
    # single-backend run has already written.
    def fingerprint_for(model: str) -> str:
        return monomer_fold_fingerprint(
            reference_pdb_path=pdb_path,
            suffix=suffix,
            folding_models=[model],
            model_identities={model: folding_model_identity(model)},
            num_seq_per_target=num_seq_per_target,
            pmpnn_sampling_temp=pmpnn_sampling_temp,
            binder_chain=binder_chain,
            mpnn_context_chains=context_chains,
            mpnn_seed_value=seed_value,
            # Without this, flipping inverse_folding_model would serve designability
            # numbers computed from the previous model's redesigns: same design, same
            # folding backend, sequences from a different inverse folder entirely.
            inverse_folding_model=None if use_pdb_seq else inverse_folding_model,
        )

    fingerprints = {m: fingerprint_for(m) for m in folding_models}
    # Folds are stored per seed, so this reads the whole map and derives the
    # seeds from the sequences it holds. Deriving first is impossible: the seed
    # comes from the redesigned sequences, which are an output of the inverse
    # folder -- the expensive step the cache exists to skip. Read-then-derive
    # keeps that skip; derive-then-read would re-run ProteinMPNN on every resume.
    stored_by_model = {
        m: (
            read_monomer_folds(
                output_dir,
                suffix,
                fingerprints[m],
                model=m,
                # A folder that was renamed still has its folds on disk under the
                # old name and the old key; adopt them rather than refold.
                also_accept=legacy_fold_fingerprints(m, fingerprint_for),
            )
            if reuse_cache
            else None
        )
        for m in folding_models
    }
    sequences = None
    for by_seed in stored_by_model.values():
        if by_seed:
            sequences = next(iter(by_seed.values())).get("sequences")
            if sequences:
                break
    if sequences:
        # A full hit needs every backend complete, not just the first one: a run
        # that added a backend has the sequences already and still has folding to
        # do.
        complete = True
        for model in folding_models:
            wanted = _fold_seeds(name, suffix, sequences, [model], n_esmfold2_seeds)
            have = {s for s in wanted if (stored_by_model.get(model) or {}).get(s)}
            if len(have) != len(wanted):
                complete = False
                logger.info(
                    f"{len(have)}/{len(wanted)} seeds cached for {name} ({suffix}, {model}); folding the rest"
                )
        if complete:
            reused = _result_from_model_folds(
                {m: stored_by_model[m] or {} for m in folding_models},
                rmsd_modes,
                pdb_path,
                derive=derive_structure_metrics,
            )
            if reused is not None:
                return reused

    # Step 1: Get sequences -- unless the cache already holds them, in which case
    # the inverse folder does not need running to add a seed.
    if sequences is None:
        sequences = get_sequences_for_evaluation(
            pdb_path=pdb_path,
            use_pdb_seq=use_pdb_seq,
            num_seq_per_target=num_seq_per_target,
            pmpnn_sampling_temp=pmpnn_sampling_temp,
            tmp_path=output_dir,
            binder_chain=binder_chain,
            mpnn_pdb_path=mpnn_pdb_path,
            target_chains=target_chains,
            inverse_folding_model=inverse_folding_model,
        )

    # Steps 2 and 3, once per seed: fold, then score. A seed is a whole unit of
    # work rather than an extra structure to average at the end, because that is
    # what the cache stores -- so adding a seed later folds and scores only the
    # ones that are new.
    per_model_seeds: dict[str, dict[int, dict]] = {}
    for model in folding_models:
        stored = stored_by_model.get(model) or {}
        # A deterministic folder gets one seed however many a sampler beside it
        # wants -- which is only expressible once the seeds are derived per model.
        seeds = _fold_seeds(name, suffix, sequences, [model], n_esmfold2_seeds)
        per_model_seeds[model] = fold_and_measure_seeds(
            sequences=sequences,
            reference_pdb_path=pdb_path,
            output_dir=output_dir,
            name=name,
            suffix=suffix,
            model=model,
            fingerprint=fingerprints[model],
            seeds=seeds,
            rmsd_modes=rmsd_modes,
            keep_outputs=keep_outputs,
            stored=stored,
            fold_cache_dir=cache_dir,
        )

    result = _result_from_model_folds(
        per_model_seeds, rmsd_modes, pdb_path, derive=derive_structure_metrics
    )

    # Cleanup if not keeping outputs
    if not keep_outputs:
        for model in folding_models:
            model_dir = os.path.join(output_dir, f"{model}_output")
            if os.path.exists(model_dir):
                try:
                    shutil.rmtree(model_dir)
                except Exception as e:
                    logger.warning(f"Could not clean up {model_dir}: {e}")

    return result


# =============================================================================
# Main Metrics Computation
# =============================================================================

_COMPLEX_PROTEIN_TYPES = {"binder", "motif_binder"}


def _shadow_folder_resolution(cfg_metric, track: str, in_use) -> None:
    """Report what one folding_models list would resolve to, without obeying it.

    A migration step, not a feature: it makes the resolver's answer observable on
    real configs before anything depends on it, so a disagreement is found in a
    log line rather than in a pass rate.
    """
    from proteinfoundation.metrics.column_names import folders_for_track
    from proteinfoundation.metrics.folder_selection import resolve_folding_models

    try:
        resolved = resolve_folding_models(cfg_metric).for_track(track)
    except Exception as exc:  # never fail a run for a shadow check
        logger.warning(f"Folder resolution (shadow, {track}) could not resolve this config: {exc}")
        return
    current = folders_for_track(in_use, track)
    if resolved != current:
        logger.warning(
            f"Folder resolution (shadow, {track}): one metric.folding_models list would use "
            f"{resolved}, this run uses {current}. Nothing has changed -- the per-track keys are "
            f"still what folds."
        )
    else:
        logger.info(f"Folder resolution (shadow, {track}): agrees with this run -- {current}")


def _is_complex(protein_type: str) -> bool:
    """Check whether the protein type represents a complex requiring binder chain extraction.

    Args:
        protein_type: Protein type string (e.g. ``"monomer"``, ``"binder"``).

    Returns:
        True if *protein_type* is in :data:`_COMPLEX_PROTEIN_TYPES`.
    """
    return protein_type in _COMPLEX_PROTEIN_TYPES


def prepare_pdb_for_monomer_eval(
    pdb_path: str,
    protein_type: str,
    binder_chain: str | None = None,
) -> tuple[str, str]:
    """
    Prepare PDB file for monomer evaluation.

    For binder complexes, extracts the binder chain to a separate file.

    Args:
        pdb_path: Path to the PDB file
        protein_type: "monomer" or "binder"
        binder_chain: Chain ID of the binder (auto-detected if None)

    Returns:
        Tuple of (path to use for evaluation, original complex path)
    """
    if not _is_complex(protein_type):
        return pdb_path, pdb_path

    # For binder: extract binder chain
    pdb_dir = os.path.dirname(pdb_path)
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    binder_only_path = os.path.join(pdb_dir, f"{pdb_name}_binder.pdb")

    try:
        extract_binder_chain_to_pdb(
            complex_pdb_path=pdb_path,
            output_pdb_path=binder_only_path,
            binder_chain=binder_chain,
        )
        logger.debug(f"Extracted binder chain to {binder_only_path}")
        return binder_only_path, pdb_path
    except Exception as e:
        logger.error(f"Failed to extract binder chain from {pdb_path}: {e}")
        raise


def compute_monomer_metrics(
    cfg: DictConfig,
    cfg_metric: DictConfig,
    samples_paths: list[str],
    job_id: int,
    ncpus: int,
    root_path: str,
    protein_type: str = "monomer",
    show_progress: bool = False,
    is_target_ligand: bool = False,
) -> pd.DataFrame:
    """
    Compute monomer metrics: designability, codesignability, novelty, sequence recovery.

    Args:
        cfg: Full configuration
        cfg_metric: Metric configuration
        samples_paths: List of PDB file paths to evaluate
        job_id: Job ID for this evaluation
        ncpus: Number of CPUs for parallel processing
        root_path: Root path for temporary files
        protein_type: Type of protein ("monomer" or "binder")
            - "monomer": Evaluate the entire structure
            - "binder": Extract binder chain from complex and evaluate only that
        show_progress: Whether to show progress bar (default: False)

    Returns:
        DataFrame with computed metrics
    """
    columns, flat_dict = parse_cfg_for_table(cfg)
    columns += ["id_gen", "pdb_path", "L", "task_name"]

    if _is_complex(protein_type):
        columns.append("complex_pdb_path")

    # Resolve task_name from config when available (binder/motif pipelines).
    # Pure monomer generation may not have one — default to None.
    task_name = None
    for cfg_key in ["dataset", "generation"]:
        sub = cfg.get(cfg_key, {})
        if hasattr(sub, "get"):
            candidate = sub.get("task_name", None)
            if candidate is None and cfg_key == "generation":
                candidate = sub.get("dataloader", {}).get("dataset", {}).get("task_name", None)
            if candidate is not None:
                task_name = candidate
                break

    # Configure evaluation modes and models
    # monomer_folding_models is the shared default; per-metric keys override if set.
    shared_models = cfg_metric.get("monomer_folding_models", ["esmfold"])
    designability_modes = cfg_metric.get("designability_modes", ["ca"])
    designability_folding_models = cfg_metric.get("designability_folding_models", shared_models)

    codesignability_modes = cfg_metric.get("codesignability_modes", ["ca", "all_atom"])
    codesignability_folding_models = cfg_metric.get("codesignability_folding_models", shared_models)

    # Shadow mode: resolved and reported, not yet obeyed. The four keys above are
    # still what folds. This runs beside them so a real campaign says, in its own
    # log, whether one list would fold what four keys currently do -- which is the
    # evidence worth having before a change that moves verdicts.
    _shadow_folder_resolution(cfg_metric, "monomer", designability_folding_models + codesignability_folding_models)

    # Resolve metric flags once.  compute_monomer_metrics=true cascades to all
    # sub-flags unless they are explicitly set to false.
    monomer_on = cfg_metric.get("compute_monomer_metrics", False)
    do_des = cfg_metric.get("compute_designability", monomer_on)
    do_codes = cfg_metric.get("compute_codesignability", monomer_on)
    do_seq_rec = cfg_metric.get("compute_co_sequence_recovery", monomer_on)
    do_ss = cfg_metric.get("compute_ss", True)

    # The same key resolved the same way the complex track resolves it, ligand
    # override included, so the two tracks cannot end up redesigning with
    # different models. redesign_model records the resolved value.
    inverse_folding_model = resolve_inverse_folding_model(
        cfg_metric.get("inverse_folding_model", DEFAULT_INVERSE_FOLDING_MODEL), is_target_ligand
    )

    # Provenance for the designability numbers. Declared with the columns because
    # the frame is built with reindex(columns=columns) and anything absent here
    # is dropped from every row.
    if do_des:
        columns.extend(["redesign_conditioning", "redesign_model"])

    metrics = {}

    # Initialize metric columns
    if do_des:
        for model in designability_folding_models:
            for mode in designability_modes:
                metrics[f"_res_scRMSD_{mode}_{model}"] = []
                metrics[f"_res_scRMSD_{mode}_{model}_all"] = []
                # Single MPNN designability: use only the first ProteinMPNN sequence
                metrics[f"_res_scRMSD_single_{mode}_{model}"] = []
        # Store MPNN sequences used for designability
        metrics["_res_mpnn_sequences"] = []
        # _res_mpnn_best_sequence is not written here any more: which redesign is
        # "best" is a formulation over _res_mpnn_sequences and a per-redesign
        # metric, not a measurement, so analyze chooses it and the metric is
        # configurable -- it was hardcoded to the first folding model's scRMSD.

    if do_codes:
        for model in codesignability_folding_models:
            for mode in codesignability_modes:
                metrics[f"_res_co_scRMSD_{mode}_{model}"] = []
                metrics[f"_res_co_scRMSD_{mode}_{model}_all"] = []

    if do_seq_rec:
        metrics["_res_co_seq_rec"] = []
        metrics["_res_co_seq_rec_all"] = []

    if do_ss:
        metrics["_res_ss_alpha"] = []
        metrics["_res_ss_beta"] = []
        metrics["_res_ss_coil"] = []

    # Log enabled metrics
    enabled_metrics = []
    if do_des:
        enabled_metrics.append(f"designability (models={designability_folding_models}, modes={designability_modes})")
    if do_codes:
        enabled_metrics.append(
            f"codesignability (models={codesignability_folding_models}, modes={codesignability_modes})"
        )
    if do_seq_rec:
        enabled_metrics.append("sequence_recovery")
    if do_ss:
        enabled_metrics.append("secondary_structure")
    if cfg_metric.get("compute_novelty_pdb", False):
        enabled_metrics.append("novelty_pdb")
    if cfg_metric.get("compute_novelty_afdb", False):
        enabled_metrics.append("novelty_afdb")
    logger.info(f"Enabled monomer metrics: {enabled_metrics}")

    results = []

    # Determine binder and target chains once if protein_type is binder. The
    # target chains are what gives the designability redesigns their context; a
    # plain monomer has none and keeps redesigning in isolation.
    binder_chain = None
    target_chains: list[str] | None = None
    if _is_complex(protein_type) and len(samples_paths) > 0:
        first_sample = samples_paths[0]
        binder_chain, target_chains = get_binder_chain_from_complex(first_sample)
        logger.info(
            f"Detected binder chain: {binder_chain}, target chain(s): {target_chains or 'none'}; "
            f"designability redesigns by {inverse_folding_model}, conditioned on "
            f"{redesign_conditioning(designability_mpnn_chains(binder_chain, target_chains))}"
        )

    for i, pdb_path in enumerate(maybe_tqdm(samples_paths, "Monomer evaluation", show_progress)):
        # Validate PDB file exists
        if not os.path.exists(pdb_path):
            logger.warning(f"PDB file not found: {pdb_path}, skipping")
            continue

        # Prepare PDB for evaluation (extract binder if needed)
        try:
            eval_pdb_path, complex_pdb_path = prepare_pdb_for_monomer_eval(
                pdb_path=pdb_path,
                protein_type=protein_type,
                binder_chain=binder_chain,
            )
        except Exception as e:
            logger.error(f"Skipping {pdb_path}: {e}")
            continue

        # Extract sequence
        try:
            seq = extract_seq_from_pdb(eval_pdb_path)
        except Exception as e:
            logger.error(f"Failed to extract sequence from {eval_pdb_path}: {e}")
            continue

        n = len(seq)

        row_dict = {
            **flat_dict,
            "id_gen": i,
            "pdb_path": pdb_path,
            "L": n,
            "task_name": task_name,
        }
        if _is_complex(protein_type):
            row_dict["complex_pdb_path"] = complex_pdb_path
        if do_des:
            # Which conditioning the designability numbers on this row carry.
            # Groupby-eligible on purpose: concatenating results from before and
            # after the conditioning change then splits into separate rows
            # instead of averaging two different metrics into one.
            row_dict["redesign_conditioning"] = redesign_conditioning(
                designability_mpnn_chains(binder_chain, target_chains)
            )
            row_dict["redesign_model"] = inverse_folding_model
        results.append(row_dict)

        # Create tmp_dir for this sample
        tmp_dir = os.path.splitext(eval_pdb_path)[0]
        os.makedirs(tmp_dir, exist_ok=True)
        des_result = None

        try:
            # Designability evaluation (ProteinMPNN + folding)
            if do_des:
                des_result = evaluate_self_consistency(
                    pdb_path=eval_pdb_path,
                    output_dir=tmp_dir,
                    use_pdb_seq=False,  # Use ProteinMPNN
                    rmsd_modes=designability_modes,
                    folding_models=designability_folding_models,
                    num_seq_per_target=cfg_metric.get("designability_num_seq", 8),
                    keep_outputs=cfg_metric.get("keep_folding_outputs", True),
                    binder_chain=binder_chain,
                    reuse_cache=cfg_metric.get("reuse_cached_monomer_folds", True),
                    # ProteinMPNN reads the complex; everything downstream --
                    # folding, RMSD -- stays on the binder alone. That asymmetry
                    # is the point: the redesign is judged apo, but it was made
                    # knowing what it has to bind.
                    mpnn_pdb_path=complex_pdb_path if _is_complex(protein_type) else None,
                    target_chains=target_chains,
                    inverse_folding_model=inverse_folding_model,
                    n_esmfold2_seeds=max(1, int(cfg_metric.get("n_esmfold2_seeds", 1))),
                )

                for model in designability_folding_models:
                    for mode in designability_modes:
                        values = des_result.rmsd_values[mode].get(model, [float("inf")])
                        best_val = min(values) if values else float("inf")
                        metrics[f"_res_scRMSD_{mode}_{model}"].append(best_val)
                        metrics[f"_res_scRMSD_{mode}_{model}_all"].append(values if values else [float("inf")])
                        metrics[f"_res_scRMSD_single_{mode}_{model}"].append(values[0] if values else float("inf"))
                        logger.debug(
                            f"Des {os.path.basename(eval_pdb_path)} [{mode}/{model}]: "
                            f"best={best_val:.3f}, all={[f'{v:.3f}' for v in values]}"
                        )

                metrics["_res_mpnn_sequences"].append(des_result.sequences)


            # Codesignability evaluation (PDB sequence + folding)
            if do_codes:
                codes_result = evaluate_self_consistency(
                    pdb_path=eval_pdb_path,
                    output_dir=tmp_dir,
                    use_pdb_seq=True,  # Use PDB sequence
                    rmsd_modes=codesignability_modes,
                    folding_models=codesignability_folding_models,
                    keep_outputs=cfg_metric.get("keep_folding_outputs", True),
                    reuse_cache=cfg_metric.get("reuse_cached_monomer_folds", True),
                    n_esmfold2_seeds=max(1, int(cfg_metric.get("n_esmfold2_seeds", 1))),
                )

                for model in codesignability_folding_models:
                    for mode in codesignability_modes:
                        values = codes_result.rmsd_values[mode].get(model, [float("inf")])
                        best_val = min(values) if values else float("inf")
                        metrics[f"_res_co_scRMSD_{mode}_{model}"].append(best_val)
                        metrics[f"_res_co_scRMSD_{mode}_{model}_all"].append(values if values else [float("inf")])
                        logger.debug(f"Codes {os.path.basename(eval_pdb_path)} [{mode}/{model}]: best={best_val:.3f}")

            # Sequence recovery (reuses MPNN sequences from designability if available)
            if do_seq_rec:
                mpnn_seqs = getattr(des_result, "sequences", None) if do_des else None
                if mpnn_seqs is None:
                    # Same conditioning as designability: this branch only runs
                    # when designability is off, and recovery computed against a
                    # differently-conditioned redesign set would not be
                    # comparable with recovery from a run where it was on.
                    mpnn_seqs = get_sequences_for_evaluation(
                        pdb_path=eval_pdb_path,
                        use_pdb_seq=False,
                        num_seq_per_target=cfg_metric.get("designability_num_seq", 8),
                        tmp_path=tmp_dir,
                        binder_chain=binder_chain,
                        mpnn_pdb_path=complex_pdb_path if _is_complex(protein_type) else None,
                        target_chains=target_chains,
                        inverse_folding_model=inverse_folding_model,
                    )
                rec_rates = [sum(a == b for a, b in zip(seq, s, strict=False)) / len(seq) for s in mpnn_seqs]
                metrics["_res_co_seq_rec"].append(max(rec_rates) if rec_rates else 0.0)
                metrics["_res_co_seq_rec_all"].append(rec_rates)

            if do_ss:
                compute_and_store_ss(metrics, eval_pdb_path)

        except Exception as e:
            logger.error(f"Metric computation failed for {pdb_path}: {e}")
            for key in metrics:
                if len(metrics[key]) < len(results):
                    metrics[key].append(float("nan"))

    # --- Post-loop summaries ---
    if do_des:
        for model in designability_folding_models:
            for mode in designability_modes:
                vals = metrics.get(f"_res_scRMSD_{mode}_{model}", [])
                if vals:
                    logger.info(
                        f"Designability [{mode}/{model}] over {len(vals)} samples: "
                        f"mean={np.nanmean(vals):.3f}, min={np.nanmin(vals):.3f}, median={np.nanmedian(vals):.3f}"
                    )

    if do_codes:
        for model in codesignability_folding_models:
            for mode in codesignability_modes:
                vals = metrics.get(f"_res_co_scRMSD_{mode}_{model}", [])
                if vals:
                    logger.info(
                        f"Codesignability [{mode}/{model}] over {len(vals)} samples: "
                        f"mean={np.nanmean(vals):.3f}, min={np.nanmin(vals):.3f}, median={np.nanmedian(vals):.3f}"
                    )

    if do_seq_rec and metrics.get("_res_co_seq_rec"):
        vals = metrics["_res_co_seq_rec"]
        logger.info(
            f"Sequence recovery over {len(vals)} samples: mean={np.nanmean(vals):.3f}, max={np.nanmax(vals):.3f}"
        )

    if do_ss and metrics.get("_res_ss_alpha"):
        n_ss = len(metrics["_res_ss_alpha"])
        logger.info(
            f"Secondary structure over {n_ss} samples: "
            f"mean alpha={np.nanmean(metrics['_res_ss_alpha']):.3f}, "
            f"beta={np.nanmean(metrics['_res_ss_beta']):.3f}, "
            f"coil={np.nanmean(metrics['_res_ss_coil']):.3f}"
        )

    df = pd.DataFrame(results).reindex(columns=dedupe_columns(columns, "Monomer results"))
    for metric in metrics:
        df[metric] = metrics[metric]

    # Novelty metrics - use binder-only PDBs for complex types
    if _is_complex(protein_type):
        novelty_pdb_list = []
        for pdb_path in df["pdb_path"].tolist():
            pdb_dir = os.path.dirname(pdb_path)
            pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
            binder_only_path = os.path.join(pdb_dir, f"{pdb_name}_binder.pdb")
            if os.path.exists(binder_only_path):
                novelty_pdb_list.append(binder_only_path)
            else:
                novelty_pdb_list.append(pdb_path)
    else:
        novelty_pdb_list = df["pdb_path"].tolist()

    novelty_configs = [
        ("compute_novelty_pdb", "pdb", "_res_novelty_pdb_tm"),
        ("compute_novelty_afdb", "genie2", "_res_novelty_afdb_tm"),
        ("compute_novelty_afdb_rep_v4", "afdb_rep_v4", "_res_novelty_afdb_rep_v4_tm"),
        (
            "compute_novelty_afdb_rep_v4_geniefilters_maxlen512",
            "afdb_rep_v4_geniefilters_maxlen512",
            "_res_novelty_afdb_rep_v4_geniefilters_maxlen512_tm",
        ),
    ]
    for config_key, db_type, col_name in novelty_configs:
        if cfg_metric.get(config_key, False):
            df[col_name] = novelty_from_list(
                query_pdb_list=novelty_pdb_list,
                db_type=db_type,
                tmp_path=os.path.join(root_path, f"tmp_{job_id}"),
                num_workers=ncpus,
            )

    return df
