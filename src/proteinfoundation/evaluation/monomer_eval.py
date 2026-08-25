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

import os
import shutil
from typing import Literal

import numpy as np
import pandas as pd
import torch
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from omegaconf import DictConfig

from proteinfoundation.evaluation.binder_eval_utils import extract_binder_chain_to_pdb, get_binder_chain_from_complex
from proteinfoundation.evaluation.monomer_eval_utils import (
    DesignabilityResult,
    FoldingResult,
    monomer_fold_fingerprint,
    read_monomer_fold_cache,
    write_monomer_fold_cache,
)
from proteinfoundation.evaluation.motif_eval_utils import compute_and_store_ss
from proteinfoundation.evaluation.utils import maybe_tqdm, parse_cfg_for_table, redesign_conditioning
from proteinfoundation.metrics.inverse_folding_models import inverse_fold
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
    inverse_folding_model: str = "protein_mpnn",
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
    folding_models: list[Literal["esmfold", "colabfold"]] = ["esmfold"],
    suffix: str = "fold",
    cache_dir: str | None = None,
    keep_outputs: bool = False,
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
    from proteinfoundation.metrics.folding_models import run_colabfold, run_esmfold, run_esmfold2

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

        model_output_dir = os.path.join(output_dir, f"{model}_output")
        os.makedirs(model_output_dir, exist_ok=True)

        try:
            if model == "esmfold":
                out_paths = run_esmfold(
                    sequences,
                    model_output_dir,
                    name,
                    suffix=suffix,
                    cache_dir=cache_dir,
                    keep_outputs=True,
                )
            elif model == "esmfold2":
                out_paths = run_esmfold2(
                    sequences,
                    model_output_dir,
                    name,
                    suffix=suffix,
                    cache_dir=cache_dir,
                    keep_outputs=True,
                )
            elif model == "colabfold":
                out_paths = run_colabfold(
                    sequences,
                    model_output_dir,
                    suffix=suffix,
                    cache_dir=cache_dir,
                    keep_outputs=keep_outputs,
                )
            else:
                raise ValueError(f"Unsupported folding model: {model}")

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


def compute_scrmsd_from_folded(
    reference_pdb_path: str,
    folding_results: dict[str, list[FoldingResult]],
    rmsd_modes: list[Literal["ca", "bb3o", "all_atom"]] = ["ca"],
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
    folded_paths = []

    for model_name, results in folding_results.items():
        for mode in rmsd_modes:
            rmsd_values[mode][model_name] = []

        for result in results:
            if not result.success or result.pdb_path is None:
                for mode in rmsd_modes:
                    rmsd_values[mode][model_name].append(float("inf"))
                continue

            folded_paths.append(result.pdb_path)

            try:
                folded_prot = load_pdb(result.pdb_path)
                folded_coors = torch.tensor(folded_prot.atom_positions, dtype=torch.float32)
                folded_mask = torch.tensor(folded_prot.atom_mask, dtype=torch.bool)
                mask = ref_mask * folded_mask

                for mode in rmsd_modes:
                    rmsd = rmsd_metric(
                        coors_1_atom37=ref_coors,
                        coors_2_atom37=folded_coors,
                        mask_atom_37=mask,
                        mode=mode,
                    )
                    rmsd_values[mode][model_name].append(rmsd)

            except Exception as e:
                logger.error(f"Error computing RMSD for {result.pdb_path}: {e}")
                for mode in rmsd_modes:
                    rmsd_values[mode][model_name].append(float("inf"))

    # Compute best RMSD for each mode/model
    best_rmsd = {}
    for mode in rmsd_modes:
        best_rmsd[mode] = {}
        for model_name in folding_results:
            values = rmsd_values[mode][model_name]
            best_rmsd[mode][model_name] = min(values) if values else float("inf")

    return DesignabilityResult(
        rmsd_values=rmsd_values,
        best_rmsd=best_rmsd,
        folded_paths=folded_paths,
    )


def _result_from_cache(
    cached: dict,
    rmsd_modes: list[str],
    reference_pdb_path: str,
) -> DesignabilityResult | None:
    """Rebuild a result from cache, or None to recompute.

    Three outcomes. A full hit needs every requested mode present. A partial hit
    -- a mode is missing but the folded structures were kept and still exist --
    computes only the missing modes from those structures, which is the payoff for
    caching them: adding an RMSD mode costs a reload rather than a refold. Anything
    else recomputes.

    The partial path is limited to a single folding model, because
    DesignabilityResult.folded_paths is flat across models and cannot be split
    back apart; with two models a partial miss simply refolds.
    """
    have = cached["rmsd_values"]
    missing = [m for m in rmsd_modes if m not in have]
    sequences = list(cached.get("sequences") or [])
    paths = list(cached.get("folded_paths") or [])

    if not missing:
        logger.info(f"Reusing cached refold for {len(sequences)} sequence(s), modes {rmsd_modes}")
        return DesignabilityResult(
            rmsd_values={m: have[m] for m in rmsd_modes},
            best_rmsd={m: cached["best_rmsd"][m] for m in rmsd_modes},
            folded_paths=paths,
            sequences=sequences,
        )

    models = sorted({model for by_model in have.values() for model in by_model})
    reusable = (
        cached.get("structures_kept") and len(models) == 1 and paths and all(os.path.exists(pth) for pth in paths)
    )
    if not reusable:
        why = (
            "structures were not kept"
            if not cached.get("structures_kept")
            else "structures are missing from disk"
            if paths and not all(os.path.exists(pth) for pth in paths)
            else f"{len(models)} folding models cannot be separated"
        )
        logger.info(f"Cached refold lacks mode(s) {missing} and {why}; refolding")
        return None

    model = models[0]
    logger.info(f"Cached refold lacks mode(s) {missing}; computing them from {len(paths)} kept structure(s)")
    synthetic = {
        model: [
            FoldingResult(pdb_path=pth, sequence=seq, model_name=model, success=True)
            for pth, seq in zip(paths, sequences + [""] * len(paths), strict=False)
        ]
    }
    extra = compute_scrmsd_from_folded(
        reference_pdb_path=reference_pdb_path,
        folding_results=synthetic,
        rmsd_modes=missing,
    )
    merged_values = {**{m: have[m] for m in rmsd_modes if m in have}, **extra.rmsd_values}
    merged_best = {**{m: cached["best_rmsd"][m] for m in rmsd_modes if m in have}, **extra.best_rmsd}
    return DesignabilityResult(
        rmsd_values={m: merged_values[m] for m in rmsd_modes},
        best_rmsd={m: merged_best[m] for m in rmsd_modes},
        folded_paths=paths,
        sequences=sequences,
    )


def evaluate_self_consistency(
    pdb_path: str,
    output_dir: str,
    use_pdb_seq: bool = False,
    rmsd_modes: list[Literal["ca", "bb3o", "all_atom"]] = ["ca"],
    folding_models: list[Literal["esmfold", "colabfold"]] = ["esmfold"],
    num_seq_per_target: int = 8,
    pmpnn_sampling_temp: float = 0.1,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
    binder_chain: str | None = None,
    reuse_cache: bool = True,
    mpnn_pdb_path: str | None = None,
    target_chains: list[str] | None = None,
    inverse_folding_model: str = "protein_mpnn",
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

    fingerprint = monomer_fold_fingerprint(
        reference_pdb_path=pdb_path,
        suffix=suffix,
        folding_models=list(folding_models),
        model_identities={m: folding_model_identity(m) for m in folding_models},
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
    cached = read_monomer_fold_cache(output_dir, suffix, fingerprint) if reuse_cache else None
    if cached is not None:
        reused = _result_from_cache(cached, rmsd_modes, pdb_path)
        if reused is not None:
            return reused

    # Step 1: Get sequences
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

    # Step 2: Fold sequences
    folding_results = fold_sequences(
        sequences=sequences,
        output_dir=output_dir,
        name=name,
        folding_models=folding_models,
        suffix=suffix,
        cache_dir=cache_dir,
        keep_outputs=keep_outputs,
    )

    # Step 3: Compute RMSD
    result = compute_scrmsd_from_folded(
        reference_pdb_path=pdb_path,
        folding_results=folding_results,
        rmsd_modes=rmsd_modes,
    )

    # Add sequences to result (for saving alongside RMSD values)
    result.sequences = sequences
    write_monomer_fold_cache(output_dir, suffix, fingerprint, result, keep_outputs)

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

    # Resolve metric flags once.  compute_monomer_metrics=true cascades to all
    # sub-flags unless they are explicitly set to false.
    monomer_on = cfg_metric.get("compute_monomer_metrics", False)
    do_des = cfg_metric.get("compute_designability", monomer_on)
    do_codes = cfg_metric.get("compute_codesignability", monomer_on)
    do_seq_rec = cfg_metric.get("compute_co_sequence_recovery", monomer_on)
    do_ss = cfg_metric.get("compute_ss", True)

    # The same key the complex track reads, so both tracks redesign with one
    # model. Ligand targets are the one place they can still diverge: the complex
    # track forces ligand_mpnn for them regardless of config, and nothing here
    # knows whether the target is a ligand. Both shipped ligand configs already
    # set ligand_mpnn, so the two agree in practice -- and redesign_model records
    # what was actually used, so a disagreement shows up in the data rather than
    # only in a reader's assumptions.
    inverse_folding_model = cfg_metric.get("inverse_folding_model", "protein_mpnn")

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
        metrics["_res_mpnn_best_sequence"] = []  # Best sequence (lowest scRMSD)

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

                first_model = designability_folding_models[0]
                first_mode = designability_modes[0]
                rmsd_values = des_result.rmsd_values[first_mode].get(first_model, [])
                if rmsd_values and des_result.sequences:
                    best_idx = rmsd_values.index(min(rmsd_values)) if rmsd_values else 0
                    best_idx = min(best_idx, len(des_result.sequences) - 1)
                    metrics["_res_mpnn_best_sequence"].append(des_result.sequences[best_idx])
                else:
                    metrics["_res_mpnn_best_sequence"].append("")

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

    df = pd.DataFrame(results).reindex(columns=columns)
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
