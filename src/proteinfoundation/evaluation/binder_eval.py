"""
Binder-specific evaluation metrics.

This module provides functions for evaluating protein binder designs:
- Refolding with structure prediction models (ColabDesign, Boltz2, RF3, Protenix)
- Interface analysis (bioinformatics metrics)
- Force field metrics (hydrogen bonds, electrostatics)
"""

import functools
import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from atomworks.io.utils.io_utils import load_any
from loguru import logger
from omegaconf import DictConfig
from openfold.np.residue_constants import restypes as OF_RESTYPES

from proteinfoundation.evaluation.binder_eval_cache import (
    adopt_binder_eval_folds,
    binder_eval_fingerprint,
    digest_file,
    legacy_complex_folds,
    read_binder_eval_cache,
    write_binder_eval_cache,
)
from proteinfoundation.evaluation.binder_eval_utils import (
    BIOINFORMATICS_METRIC_COLS,
    DEFAULT_INTERFACE_CUTOFF_LIGAND,
    DEFAULT_INTERFACE_CUTOFF_PROTEIN,
    DEFAULT_NUM_REDESIGN_SEQS_LIGAND,
    DEFAULT_NUM_REDESIGN_SEQS_PROTEIN,
    TMOL_METRIC_COLS,
    apo_column,
    apo_confidence_column,
    apo_derived_column,
    apo_fold_fingerprint,
    check_thresholds_are_computable,
    dedupe_columns,
    extract_binder_chain_to_pdb,
    gated_columns,
    get_binder_chain_from_complex,
    get_metric_columns,
    per_sequence_pass,
    resolve_success_thresholds,
    shared_redesign_indices,
)
from proteinfoundation.evaluation.esm_eval import (
    DEFAULT_ESM_BATCH_TOKENS,
    ESM_AVAILABLE,
    compute_esm_ppl_for_sequences,
)
from proteinfoundation.evaluation.monomer_eval_utils import (
    APO_CONFIDENCE_SUFFIXES,
    MONOMER_DERIVED_SUFFIXES,
    derive_for_result,
    per_model_confidence,
    refresh_monomer_derivation,
)
from proteinfoundation.evaluation.utils import maybe_tqdm, parse_cfg_for_table, redesign_conditioning
from proteinfoundation.metrics.binder_metrics import assemble_binder_sequences, complex_mpnn_chains
from proteinfoundation.metrics.column_names import backend_for_folding_method, folder_family
from proteinfoundation.metrics.consensus_folding import (
    CONSENSUS_METRIC_SUFFIXES,
    CONSENSUS_PROVENANCE_SUFFIXES,
    STRUCTURE_DIGEST_KEY,
    ComplexFoldContext,
    advisory_column,
    available_backends,
    consensus_derived_suffixes,
    report_gated_and_reported_columns,
    score_binders,
)
from proteinfoundation.metrics.ensembling import GEOMETRY_REDUCTION_VERSION
from proteinfoundation.metrics.folder_selection import resolve_folding_models
from proteinfoundation.metrics.interface import DEFAULT_CONTACT_CUTOFF, INTERFACE_DERIVATION_VERSION
from proteinfoundation.metrics.inverse_folding_models import (
    DEFAULT_INVERSE_FOLDING_MODEL,
    REDESIGN_SCORE_KIND,
    resolve_inverse_folding_model,
)
from proteinfoundation.metrics.pae_store import drop_structures_keeping_sidecars
from proteinfoundation.metrics.redesign_set import redesign_set_size
from proteinfoundation.metrics.seeding import SEED_DERIVATION_VERSION
from proteinfoundation.metrics.tmol_interface import tmol_interface_metrics
from proteinfoundation.result_analysis.binder_analysis_utils import COMPLEX_BACKEND_COLUMN
from proteinfoundation.utils.colabdesign_utils import AF2_SAVE_LOCATION

# =============================================================================
# Safe Imports with Availability Flags
# =============================================================================

# TMOL (force field metrics) - may not be available in all environments
TMOL_AVAILABLE = False
try:
    from proteinfoundation.rewards.tmol_reward import TmolRewardModel

    TMOL_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    logger.warning(f"TMOL import failed: {e}. TMOL metrics will return NaN.")

# PR Alternative (bioinformatics interface scoring) - may have missing dependencies
PR_ALTERNATIVE_AVAILABLE = False
try:
    from proteinfoundation.utils.pr_alternative_utils import pr_alternative_score_interface

    PR_ALTERNATIVE_AVAILABLE = True
except (ImportError, RuntimeError, OSError) as e:
    logger.warning(f"PR Alternative import failed: {e}. Bioinformatics metrics will return NaN.")


# =============================================================================
# Folding Model Initialization
# =============================================================================


def initialize_folding_model(
    folding_model: str,
    target_pdb_chain: list[str],
    target_task_name: str,
    is_target_ligand: bool,
) -> dict[str, Any]:
    """What a complex folder needs CONSTRUCTED before it can be called.

    Almost nothing, now. Every complex folder is reached through
    CONSENSUS_BACKENDS, which takes sequences and a ComplexFoldContext, so the
    only thing left to build is a folder that is an object rather than a function
    of its inputs -- RF3 and its weights.

    It used to decide which folders could fold a complex at all, raising for
    anything it could not construct. That made it a second registry beside
    CONSENSUS_BACKENDS, free to disagree with it, and it did: a pass naming
    esmfold2 as its only complex folder died here, on a folder the campaign had
    configured and the other registry folds perfectly well. Whether a folder can
    fold a complex is _FOLDER_CAPABILITIES' answer and CONSENSUS_BACKENDS'
    mechanism; this function only builds things.

    The ligand refusal stays, because it is about the MODEL rather than about
    this function's vocabulary: ColabDesign cannot template a small molecule, and
    a run that discovers that after folding has wasted the folds.
    """
    target_pdb_chain = sorted(target_pdb_chain)
    family = folder_family(folding_model)

    # `af2` is the model; `colabdesign` is the harness that runs it for a complex,
    # the way `colabfold` is the CLI that runs it for a monomer. Both names reach
    # here because the config vocabulary is the model's.
    if family == "af2":
        if is_target_ligand:
            raise ValueError("ColabDesign does not support ligand-protein complex folding")
        return {"model_name": "colabdesign"}

    if family == "rf3":
        from proteinfoundation.rewards.rf3_reward import get_default_rf3_runner

        logger.info(f"Initializing RF3 model: {folding_model}")
        runner = get_default_rf3_runner(
            ckpt_path=os.environ.get(
                "RF3_CKPT_PATH",
                os.path.join(os.environ.get("DATA_PATH", ""), "rf3/rf3_latest.pt"),
            ),
            dump_dir=None,
            rf3_path=os.environ.get("RF3_EXEC_PATH", None),
        )
        return {"model_name": "RF3", "runner": runner}

    # Everything else folds from sequences and a context, with nothing to build
    # -- but only if it can fold a complex at all. That question has one answer,
    # and it is the backend registry's, so a typo still fails here rather than
    # becoming a run with a folder nothing will ever call.
    from proteinfoundation.metrics.consensus_folding import CONSENSUS_BACKENDS

    if folding_model in CONSENSUS_BACKENDS or family in CONSENSUS_BACKENDS:
        return {"model_name": folding_model}
    raise ValueError(
        f"Folding model '{folding_model}' not supported: it is neither a folder this builds "
        f"nor one {sorted(CONSENSUS_BACKENDS)} can fold."
    )


# =============================================================================
# Binder Metrics
# =============================================================================


def sequences_for_type(
    seq_type: str,
    sequences_dict: dict,
    sequence_type_stats: dict,
) -> list[str]:
    """The sequences whose metrics this row reports, in stats order.

    ``sequences_dict[type]`` and ``sequence_type_stats[type]["complex_stats"]``
    are parallel lists built in append order, and the row's headline values index
    the second while its sequences used to come from the first. Nothing joined
    them, so a filter or reorder applied to one and not the other would silently
    pair a sequence with another sequence's structural metrics.

    ``aa_stats`` now records the sequence each row was computed from, which makes
    it the source of truth: it is appended in the same iteration as the complex
    and RMSD stats, so the lists stay parallel by construction -- which is what
    lets analyze index them all at one chosen position.
    Caches written before that field existed have no sequences, so fall back to
    the old behaviour there, and report a mismatch loudly rather than emitting
    misaligned metrics.
    """
    dict_seqs = [s["seq"] for s in sequences_dict.get(seq_type, [])]
    aa_stats = sequence_type_stats.get(seq_type, {}).get("aa_stats", [])
    stats_seqs = [a.get("sequence") for a in aa_stats]

    if not stats_seqs or any(s is None for s in stats_seqs):
        # Pre-existing cache entry: no recorded sequences to check against.
        return dict_seqs

    if stats_seqs != dict_seqs:
        logger.error(
            f"Sequence/stats misalignment for '{seq_type}': "
            f"{len(dict_seqs)} sequences vs {len(stats_seqs)} stats rows"
            + ("" if len(stats_seqs) != len(dict_seqs) else " (same length, different order or content)")
            + ". Using the sequences recorded with the stats, so per-sequence "
            "metrics stay paired with the structures they came from."
        )
    return stats_seqs


def redesign_scores_for_type(
    seq_type: str,
    sequences: list[str],
    sequences_dict: dict,
) -> list[float]:
    """ProteinMPNN's own score for each sequence, in the row's order.

    Looked up *by sequence* rather than by position. ``sequences`` is ordered by
    the stats (see ``sequences_for_type``) while the scores live in
    ``sequences_dict``, and pairing two lists by index when only one of them is
    known to be aligned is the mistake ``sequences_for_type`` exists to prevent.

    NaN where there is no score: ``self`` is read off the PDB and was never
    scored by an inverse folder, and a duplicate sequence with disagreeing scores
    cannot be attributed to one of them.
    """
    by_seq: dict[str, float] = {}
    ambiguous: set[str] = set()
    for entry in sequences_dict.get(seq_type, []):
        seq, score = entry.get("seq"), entry.get("score")
        if seq is None or score is None:
            continue
        if seq in by_seq and by_seq[seq] != score:
            ambiguous.add(seq)
        by_seq[seq] = score
    if ambiguous:
        logger.warning(f"{len(ambiguous)} duplicate '{seq_type}' sequence(s) with differing scores; reporting NaN")

    return [np.nan if s in ambiguous else by_seq.get(s, np.nan) for s in sequences]


def packed_aa_counts(counts_by_residue: dict[str, int]) -> list[int]:
    """One redesign's amino-acid composition, packed into OpenFold's residue order.

    Positional rather than a mapping because that is what the columns have always
    carried and what the distribution code in analyze unpacks; the order is the
    contract.
    """
    packed = [0] * len(OF_RESTYPES)
    for residue, count in counts_by_residue.items():
        if residue in OF_RESTYPES:
            packed[OF_RESTYPES.index(residue)] += count
    return packed


def _at_indices(values, indices):
    """Pick *indices* out of *values*, inf where the fold is missing.

    Module level rather than nested: a nested def inside apo_refold truncates the
    source-scanning tests that read its body, and this is the third time that has
    bitten. inf rather than dropping the slot, because the apo lists stay
    positionally aligned with the holo ones -- a short list shifts every later
    column rather than announcing itself.
    """
    return [values[i] if i < len(values) else float("inf") for i in indices]


def apo_fingerprint_for(backend: str, binder_pdb_path: str, sequences: list[str]) -> str:
    """One backend's apo fold key. Module level so the same call produces the key
    a fold is written under and the key an older name's fold is looked up by --
    two spellings of one fingerprint is how a cache silently splits."""
    from proteinfoundation.metrics.folding_models import folding_model_identity

    return apo_fold_fingerprint(
        binder_pdb_path=binder_pdb_path,
        sequences=sequences,
        folding_models=[backend],
        model_identities={backend: folding_model_identity(backend)},
    )


def apo_refold(
    seq_type: str,
    sequences: list[str],
    binder_pdb_path: str,
    sample_root_path: str,
    folding_models: list[str],
    rmsd_modes: list[str],
    keep_outputs: bool,
    reuse_cache: bool,
    n_esmfold2_seeds: int = 1,
    *,
    share_with_designability: bool = False,
    shared_count: int | None = None,
    binder_chain: str | None = None,
    complex_pdb_path: str | None = None,
    target_chains: list[str] | None = None,
    inverse_folding_model: str = DEFAULT_INVERSE_FOLDING_MODEL,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[str, dict[str, list[float]]],
    dict[str, dict[str, list]],
]:
    """Fold each sequence alone and measure it against the designed backbone.

    The holo track asks whether a sequence folds as designed *with* its target.
    This asks whether the same sequence folds as designed *without* it. BoltzGen
    reports that requiring both improves experimental success, and expressing
    that needs one sequence carrying both verdicts -- which is why this is
    per-sequence and positionally aligned with the holo columns, rather than the
    design-level ``min()`` the designability track reports.

    ``self`` is special: its apo fold is exactly what codesignability already
    computes, so it is delegated rather than repeated, and
    ``self_apo_scRMSD_{mode}_{model}`` is an alias of
    ``_res_co_scRMSD_{mode}_{model}`` sharing one fold.

    Returns ``({(mode, model): [rmsd per sequence]}, {model: {metric: [value per
    sequence]}}, {model: {metric: [value per sequence]}})`` -- the second being
    the confidence the folder reported (pLDDT, pTM, mean PAE), the third what is
    read OFF the kept structures rather than reported by the folder. Empty
    when nothing could be folded; a failed fold is ``inf``
    for that sequence, not a missing row, so the lists stay aligned with the
    sequences they describe. A fold with no readable confidence is NaN, which is
    the same distinction: unmeasured rather than bad.
    """
    from proteinfoundation.evaluation.monomer_eval import (
        evaluate_self_consistency,
        fold_and_measure_seeds,
    )
    from proteinfoundation.utils.pdb_utils import pdb_name_from_path

    # The apo fold of the co-designed sequence *is* codesignability: same sequence
    # read off the same binder PDB, same folder, same reference. Delegating to the
    # function that owns that computation shares the fold through its cache rather
    # than doing it twice -- and shares it by construction, instead of by
    # replicating a fingerprint that depends on defaults unrelated to this call.
    # It also keeps _res_co_scRMSD_* as the canonical upstream name, with
    # self_apo_scRMSD_* an alias for it.
    if seq_type == "self":
        result = evaluate_self_consistency(
            pdb_path=binder_pdb_path,
            output_dir=os.path.splitext(binder_pdb_path)[0],
            use_pdb_seq=True,
            rmsd_modes=rmsd_modes,
            folding_models=folding_models,
            keep_outputs=keep_outputs,
            reuse_cache=reuse_cache,
            n_esmfold2_seeds=n_esmfold2_seeds,
            # The apo track reports what is read off the structures, and the
            # designability track it shares this fold with does not. Asking here
            # gets it derived per seed, before averaging concatenates the paths
            # and leaves nothing aligned one-per-sequence to read.
            derive_structure_metrics=True,
        )
        # The sequence read off the PDB should be the one whose holo metrics sit on
        # this row. If it is not, the apo and holo columns would describe different
        # sequences, which is the one failure this alias must not introduce.
        if list(result.sequences) != list(sequences):
            logger.error(
                f"Apo/holo sequence mismatch for 'self': the binder PDB reads "
                f"{result.sequences} but the row reports {sequences}. Dropping the apo columns for "
                f"this design rather than pairing a fold with another sequence's metrics."
            )
            # Three empties, like every other return here: the caller unpacks
            # three, and a guard that dropped columns by raising would be worse
            # than the mispairing it exists to prevent.
            return {}, {}, {}
        return (
            {
                (mode, m): result.rmsd_values.get(mode, {}).get(m, [float("inf")] * len(sequences))
                for mode in rmsd_modes
                for m in folding_models
            },
            per_model_confidence(result.plddt, result.confidence, folding_models, len(sequences)),
            # Read off the same structures, without a cache of its own to stamp:
            # this path shares the codesignability fold, whose cache lives under
            # another track's fingerprint. One sequence's worth of re-reading per
            # run beats reconstructing that fingerprint here and writing into it.
            derive_for_result(result),
        )

    # The redesigns' apo fold IS designability: the same sequences, folded alone,
    # against the same binder backbone, by the same folders. Delegated for the
    # same reason the `self` branch delegates to codesignability -- one fold
    # shared by construction, rather than two caches that agree only while two
    # fingerprints are kept in step by hand.
    #
    # Exact equality, never startswith: mpnn_fixed is a DIFFERENT draw (fixed
    # interface positions, variant="fixed", its own seed), and pairing its
    # sequences with unfixed-draw folds is one character away from here.
    if seq_type == "mpnn" and share_with_designability:
        result = evaluate_self_consistency(
            pdb_path=binder_pdb_path,
            output_dir=os.path.splitext(binder_pdb_path)[0],
            use_pdb_seq=False,
            rmsd_modes=rmsd_modes,
            folding_models=folding_models,
            num_seq_per_target=shared_count or len(sequences),
            keep_outputs=keep_outputs,
            reuse_cache=reuse_cache,
            n_esmfold2_seeds=n_esmfold2_seeds,
            binder_chain=binder_chain,
            mpnn_pdb_path=complex_pdb_path,
            target_chains=target_chains,
            inverse_folding_model=inverse_folding_model,
            redesign_cache_dir=sample_root_path,
            derive_structure_metrics=True,
        )
        # By sequence, not by position. The binder set is a seeded prefix of the
        # shared set today; a slice would mispair silently the first time it is
        # not -- reordered by score ranking, or resumed at a different size.
        indices = shared_redesign_indices(sequences, result.sequences)
        if indices is None:
            logger.error(
                f"Apo/holo sequence mismatch for 'mpnn': this row reports {len(sequences)} "
                f"redesigns that are not all present in the shared set of {len(result.sequences)}. "
                f"Dropping the apo columns for this design rather than pairing a fold with another "
                f"sequence's metrics."
            )
            return {}, {}, {}

        take = functools.partial(_at_indices, indices=indices)
        rmsds = {
            (mode, m): take(result.rmsd_values.get(mode, {}).get(m, []))
            for mode in rmsd_modes
            for m in folding_models
        }
        confidence = per_model_confidence(
            result.plddt, result.confidence, folding_models, len(result.sequences)
        )
        sliced_confidence = {
            m: {metric: take(values) for metric, values in by_metric.items()}
            for m, by_metric in confidence.items()
        }
        derived = derive_for_result(result)
        sliced_derived = {
            m: {metric: take(values) for metric, values in by_metric.items()}
            for m, by_metric in derived.items()
        }
        return rmsds, sliced_confidence, sliced_derived

    suffix = f"apo_{seq_type}"
    from proteinfoundation.evaluation.monomer_eval_utils import (
        _fold_seeds,
        average_folds,
        legacy_fold_fingerprints,
        merge_model_folds,
        read_monomer_folds,
    )

    name = f"{pdb_name_from_path(binder_pdb_path)}_{suffix}"

    # One backend at a time, each against its own cache, its own fingerprint and
    # its own seeds. A shared fingerprint made a second backend discard the
    # first one's folds for a reason that had nothing to do with how it folded
    # them, and a shared seed list gave a deterministic folder three seeds
    # because a sampler beside it wanted three -- three identical structures,
    # averaged with themselves.
    per_model: dict[str, dict] = {}
    for model in folding_models:
        fingerprint = apo_fingerprint_for(model, binder_pdb_path, sequences)
        # Sequences are an argument here rather than an output, so the seeds can
        # be derived up front -- unlike the codesignability path, where they come
        # out of the inverse folder and the cache has to be read first.
        seeds = _fold_seeds(name, suffix, list(sequences), [model], n_esmfold2_seeds)
        stored = (
            read_monomer_folds(
                sample_root_path,
                suffix,
                fingerprint,
                name=name,
                model=model,
                also_accept=legacy_fold_fingerprints(
                    model, lambda b: apo_fingerprint_for(b, binder_pdb_path, sequences)
                ),
            )
            if reuse_cache
            else None
        )
        # The same fold-and-measure the codesignability track runs, through the
        # same function: these two ask different questions of different sequences
        # and answer them identically, and two copies of this loop had already
        # drifted twice.
        per_seed = fold_and_measure_seeds(
            sequences=sequences,
            reference_pdb_path=binder_pdb_path,
            output_dir=sample_root_path,
            name=name,
            suffix=suffix,
            model=model,
            fingerprint=fingerprint,
            seeds=seeds,
            rmsd_modes=rmsd_modes,
            keep_outputs=keep_outputs,
            stored=stored,
            # The apo track never steered the weights cache; fold_sequences takes
            # it for the backends that do.
            fold_cache_dir=None,
            what="apo seeds",
        )

        # What the kept structures say about themselves, filled in on the run that
        # produced them rather than the one after. Re-reads PDBs, never refolds.
        per_seed = refresh_monomer_derivation(sample_root_path, suffix, fingerprint, per_seed, model=model)
        per_model[model] = average_folds(per_seed) or {}

    averaged = merge_model_folds(per_model)
    values = averaged.get("rmsd_values", {})
    rmsds = {
        (mode, m): values.get(mode, {}).get(m, [float("inf")] * len(sequences))
        for mode in rmsd_modes
        for m in folding_models
    }
    return (
        rmsds,
        per_model_confidence(
            averaged.get("plddt"), averaged.get("confidence"), folding_models, len(sequences)
        ),
        averaged.get("derived") or {},
    )


def _target_chain_sequences(target_pdb_path: str, target_pdb_chain: list[str]) -> list[str]:
    """The target's chain sequences, or nothing if any of them cannot be read.

    All-or-nothing on purpose: a target read as two chains when it has three is
    a different molecule, and every number measured against it would be wrong
    in a way no downstream check would catch.
    """
    from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb

    sequences = []
    for chain in target_pdb_chain:
        try:
            seq = extract_seq_from_pdb(target_pdb_path, chain_id=chain)
        except Exception as exc:  # advisory features must not stop evaluation
            logger.error(f"Cannot read target chain {chain}: {exc}")
            return []
        if seq:
            sequences.append(seq)
    return sequences


def compute_binder_metrics(
    eval_config: DictConfig,
    sample_root_paths: list[str],
    target_pdb_path: str,
    target_pdb_chain: list[str],
    is_target_ligand: bool,
    get_fixed_residues_fn: Callable[[str, str], list[str] | None] | None = None,
) -> pd.DataFrame:
    """Compute comprehensive metrics for protein binders.

    Runs refolding evaluation and extracts metrics for each sequence type.

    Args:
        eval_config: Top-level evaluation config.
        sample_root_paths: Directories containing generated PDBs.
        target_pdb_path: Path to the target PDB.
        target_pdb_chain: Target chain IDs.
        is_target_ligand: Whether the target is a ligand.
        get_fixed_residues_fn: Optional per-sample callback for ``mpnn_fixed``
            residues.  Signature: ``(pdb_path, binder_chain) -> Optional[List[str]]``.
            When provided, called for each sample to get fixed positions
            (e.g. motif residues) instead of interface-based detection.
            Return ``None`` to fall back to interface residues for that sample.

    Returns:
        DataFrame with binder metrics, one row per sample.
    """
    logger.info("Starting binder evaluation")
    cfg_metric = eval_config.metric

    # Get dataset config - fallback to generation config only if dataset doesn't exist
    if "dataset" in eval_config:
        cfg_dataset = eval_config.dataset
    else:
        cfg_dataset = eval_config.generation.dataloader.dataset
    target_task_name = cfg_dataset.task_name

    # Which folders refold, for every track this function drives. One resolver,
    # so apo and the complex cross-check cannot end up describing different sets
    # by reading different keys.
    folders = resolve_folding_models(cfg_metric, is_target_ligand=is_target_ligand)

    # Which complex folder is the one whose columns the thresholds are written
    # against. The FIRST member that can fold a complex, so a config orders its
    # list and says nothing else; binder_folding_method is the legacy way to name
    # it and still wins where a campaign still sets it.
    #
    # "Primary" is now only this: the folder a criterion happens to name. It is
    # no longer a different mechanism, and every other complex folder produces
    # the same columns in its own backend slot.
    folding_model = cfg_metric.get("binder_folding_method") or (
        folders.complex[0] if folders.complex else "colabdesign"
    )
    folding_model_specs = initialize_folding_model(folding_model, target_pdb_chain, target_task_name, is_target_ligand)
    # The backend slot for every complex column this run emits, and the value the
    # provenance column records. Resolved once: a column that says af2 while an
    # rf3 model produced it is the mislabelling this scheme exists to remove.
    complex_backend = backend_for_folding_method(folding_model)

    # Evaluation parameters
    sequence_types = cfg_metric.get("sequence_types", ["self"])
    interface_cutoff = cfg_metric.get(
        "interface_cutoff",
        (DEFAULT_INTERFACE_CUTOFF_LIGAND if is_target_ligand else DEFAULT_INTERFACE_CUTOFF_PROTEIN),
    )
    num_redesign_seqs = cfg_metric.get(
        "num_redesign_seqs",
        (DEFAULT_NUM_REDESIGN_SEQS_LIGAND if is_target_ligand else DEFAULT_NUM_REDESIGN_SEQS_PROTEIN),
    )
    # Resolved here, not just read: a ligand target overrides the configured
    # model, and the columns below record which model ran and which score
    # convention to read them by. Recording the configured value would name the
    # wrong model and, worse, the wrong sort direction.
    inverse_folding_model = resolve_inverse_folding_model(
        cfg_metric.get("inverse_folding_model", DEFAULT_INVERSE_FOLDING_MODEL), is_target_ligand
    )

    # No ranking here. metric.ranking_criteria moved to aggregation.ranking_criteria,
    # where analyze applies it: choosing which redesign a row presents is a
    # formulation over the per-sequence lists this stage emits, and reweighting it
    # must not cost a refold. A config still setting metric.ranking_criteria is
    # named rather than ignored -- silently dropping a knob someone tuned is how a
    # run ends up ranked by something other than what its config says.
    if cfg_metric.get("ranking_criteria") is not None:
        logger.warning(
            "metric.ranking_criteria is no longer read here; move it to "
            "aggregation.ranking_criteria, which analyze applies when it chooses "
            "the headline sequence."
        )

    # Success criteria, applied per sequence rather than left to the analysis
    # stage. Analysis reports a design as passing when *any* of its sequences
    # does, so the verdict on an individual sequence exists nowhere on disk: a
    # consumer wanting the failures has to ast.literal_eval the *_all columns and
    # re-apply the thresholds by hand, and two consumers doing that can disagree.
    # Emitting the vector here settles it once, from the same
    # aggregation.success_thresholds analysis reads.
    success_thresholds = resolve_success_thresholds(eval_config, is_target_ligand)
    check_thresholds_are_computable(
        success_thresholds,
        cfg_metric.get("compute_apo_metrics", False),
        apo_rmsd_modes=list(cfg_metric.get("apo_rmsd_modes", ["ca"]) or []),
        apo_folding_models=list(cfg_metric.get("apo_folding_models", ["esmfold"]) or []),
    )
    logger.info(
        "Per-sequence pass criteria: " + ", ".join(f"{name}{spec}" for name, spec in success_thresholds.items())
    )

    # Apo refolding: does each sequence fold as designed *without* its target?
    # Off by default -- it is a monomer fold per sequence per design on top of the
    # complex fold, and nothing gates on it yet. Emitted ungated on purpose: the
    # threshold should be set after seeing the distribution, not before, or
    # "the gate works" and "the gate is mis-calibrated" look the same.
    compute_apo = cfg_metric.get("compute_apo_metrics", False)
    apo_folding_models = list(folders.apo)
    apo_rmsd_modes = list(cfg_metric.get("apo_rmsd_modes", ["ca"]) or [])
    reuse_cached_apo = cfg_metric.get("reuse_cached_apo_folds", True)
    # ESMFold2 is a diffusion sampler, so one fold is one draw. More seeds trade
    # evaluation time for a less noisy number; the other folders are deterministic
    # and ignore this. Prefix-stable, so raising it later folds only what is new.
    n_esmfold2_seeds = max(1, int(cfg_metric.get("n_esmfold2_seeds", 1)))
    n_af2_models = max(1, int(cfg_metric.get("n_af2_models", 1)))
    if compute_apo and is_target_ligand:
        logger.info("Apo refolding skipped: these folders fold a single protein chain, target is a ligand")
        compute_apo = False
    if compute_apo:
        logger.info(f"Apo refolding enabled: models={apo_folding_models}, modes={apo_rmsd_modes}")

    # Progress bar setting
    show_progress = eval_config.get("show_progress", False)

    # Reuse refolding results already on disk for a design, so an interrupted or
    # partially-failed evaluate does not repeat the expensive folding pass. Only
    # caches carrying an identical fingerprint are honoured, so this cannot serve
    # results from a different folding backend or target.
    reuse_cached_folding = cfg_metric.get("reuse_cached_folding", True)
    cache_fingerprint_base = {
        "folding_model": folding_model,
        "inverse_folding_model": inverse_folding_model,
        "num_redesign_seqs": num_redesign_seqs,
        "sequence_types": sorted(sequence_types),
        "is_target_ligand": bool(is_target_ligand),
        # The target's IDENTITY, not its location. This was `target_pdb_path`,
        # which meant moving a campaign package -- or reaching the same package
        # through a different mount -- changed the hash and silently discarded
        # every refold inside it. On CBLN1 that would have been ~42 GPU-hours to
        # recompute structures that had not changed at all. A fingerprint is
        # supposed to answer "same structure request?", and the file's contents
        # answer that; its path answers "same string?". Existing caches keep
        # working -- see the legacy fingerprint at the read site.
        "target_structure": digest_file(target_pdb_path),
        "target_pdb_chain": target_pdb_chain,
        "target_task_name": target_task_name,
        # Redesigns are now drawn from a derived seed rather than whatever
        # ProteinMPNN picked at run time. Caches written before that hold
        # unseeded draws: still valid sequences, but not the ones a fresh run
        # would produce, and not the ones the designability track draws from the
        # same design. Reusing them would quietly reintroduce the un-joinable
        # state this work exists to remove, so a derivation change invalidates.
        "mpnn_seed_derivation": SEED_DERIVATION_VERSION,
        # AF2 confidence scores and the geometry measured off its structures are
        # both means over this many models, so a cache written at one count
        # cannot answer for another. Without this the count could be raised and
        # every cached design would keep serving its single-model numbers --
        # the silent-stale-cache case this fingerprint exists to prevent.
        "n_af2_models": n_af2_models,
    }
    # interface_cutoff reaches the structures through exactly one route: the
    # interface residues are the positions mpnn_fixed holds fixed, so under that
    # sequence type the cutoff decides which sequences get folded. For every
    # other sequence type it decides only which residues are counted into
    # aa_interface_counts, which is read off structures the cutoff never
    # influenced. Held unconditionally in the structure fingerprint it made a
    # cutoff change cost a full refold of a campaign -- measured at ~42 GPU-hours
    # on CBLN1 -- to recompute a distance query. Present here only when it is
    # actually structural, which also leaves the hash of an mpnn_fixed run
    # byte-identical to the one it had before this split.
    if "mpnn_fixed" in sequence_types:
        cache_fingerprint_base["interface_cutoff"] = interface_cutoff
        cache_fingerprint_base["interface_derivation"] = INTERFACE_DERIVATION_VERSION
    # Split from the above on purpose. Everything in cache_fingerprint_base
    # decides which structures get predicted; this decides only how the numbers
    # are read off them. Keeping them apart is what lets a reduction change reuse
    # the structures instead of refolding to recompute an arithmetic choice.
    derivation_fingerprint = binder_eval_fingerprint(
        geometry_reduction=GEOMETRY_REDUCTION_VERSION,
        interface_cutoff=interface_cutoff,
        # The cutoff alone does not say which structure the interface was measured
        # on, or under which rule. Taken as the version constant rather than the
        # whole interface_provenance() dict so computing a fingerprint does not
        # require protein-interface to be installed -- a ligand run never calls it.
        interface_derivation=INTERFACE_DERIVATION_VERSION,
    )
    # Structure fingerprints a previous cutoff would have produced, which this run
    # is willing to reuse. Empty by default: reuse across a cutoff is a claim
    # about what the cutoff touched, and the run has to make it rather than
    # inherit it. A matching cache is accepted as structure-valid, refreshed
    # against the current cutoff, and rewritten under the current fingerprint --
    # so the migration happens once, on first use, per design.
    reusable_interface_cutoffs = [float(c) for c in (cfg_metric.get("reusable_interface_cutoffs", []) or [])]
    if reusable_interface_cutoffs and "mpnn_fixed" in sequence_types:
        raise ValueError(
            "metric.reusable_interface_cutoffs cannot be used with the 'mpnn_fixed' sequence type: "
            "there the cutoff picks the positions ProteinMPNN holds fixed, so caches written at "
            f"{reusable_interface_cutoffs} hold different sequences and different structures, not the "
            "same structures read differently. Refold instead."
        )
    # What the fingerprint looked like when the target was keyed by path. Caches
    # on disk carry this, and they are the same structures -- so they are accepted
    # as legacy rather than refolded, and the reader reports them stale so only the
    # derived numbers get recomputed. Without this, fixing the path dependency
    # would itself cost the refold it exists to prevent.
    legacy_target_base = {
        key: value for key, value in cache_fingerprint_base.items() if key != "target_structure"
    }
    legacy_target_base["target_pdb_path"] = target_pdb_path

    n_reused = 0
    failed_designs: list[tuple[str, str]] = []
    # (sequence, draw) entries carried over from a pre-unification campaign.
    n_adopted = 0

    # Every complex folder the resolver named, with nothing removed.
    #
    # This used to exclude folders.complex[0], because that one folded through
    # run_binder_eval and the rest through CONSENSUS_BACKENDS -- two mechanisms
    # for one job, so the list was split by which mechanism ran a member. It is
    # one mechanism now, so there is nothing to split and no first folder. The
    # columns are identical either way: advisory_column produces exactly the
    # names the old primary emission built through `rename`, checked metric by
    # metric with zero mismatches, so unifying the mechanism leaves the artifact
    # alone.
    #
    # These folders predict protein-protein complexes, so a ligand target has no
    # target sequence to fold against and complex refolding is skipped entirely.
    complex_folders = list(folders.complex)
    consensus_cfg = dict(cfg_metric.get("consensus_cfg", {}) or {})
    # The advisory folds are ESMFold2 too, so they answer to the same knob --
    # otherwise metric.n_esmfold2_seeds means "three seeds, except for the
    # expensive folds", which is not what it says. consensus_cfg.n_seeds still
    # wins if a config sets it, since the complex fold is several times the cost
    # of a monomer one and is the sensible place to want a different count.
    consensus_cfg.setdefault("n_seeds", n_esmfold2_seeds)
    # AF2's draw count, for the same reason and read from the same place. Its
    # draws are parameter sets rather than seeds, so metric.n_af2_models is what
    # says how many predictions one complex gets; without this default it would
    # fall to 1 here and an af2 fold reached through score_binders would be a
    # one-model ensemble beside a five-model one from the same campaign.
    consensus_cfg.setdefault("n_af2_models", n_af2_models)
    reuse_cached_consensus = cfg_metric.get("reuse_cached_consensus", True)
    # The force field, on the advisory structures too, when the run computes it on
    # the primary backend's refolds. Read from the same two config keys rather
    # than a knob of its own: "TMOL is on" should not mean "on for one of the two
    # models that folded this complex". It is off in the binder campaigns, and the
    # derivation fingerprint carries the request -- so a run that does not want it
    # is not a run whose cached structures look under-derived.
    _refolded_cfg = cfg_metric.get("refolded", {}) or {}
    derive_consensus_tmol = bool(cfg_metric.get("compute_refolded_structure_metrics", False)) and bool(
        _refolded_cfg.get("tmol", True)
    )
    consensus_target_seqs: list[str] = []
    if complex_folders:
        unknown = [b for b in complex_folders if b not in available_backends()]
        if unknown:
            logger.error(f"Unknown complex folders {unknown}; known: {available_backends()}. Skipping those.")
            complex_folders = [b for b in complex_folders if b not in unknown]
    if complex_folders and is_target_ligand:
        logger.info("Complex refolding skipped: these folders fold protein complexes, target is a ligand")
        complex_folders = []
    if complex_folders:
        consensus_target_seqs = _target_chain_sequences(target_pdb_path, target_pdb_chain)
        if not consensus_target_seqs:
            complex_folders = []
        else:
            logger.info(
                f"Complex refolding: {complex_folders}, target "
                f"{len(consensus_target_seqs)} chain(s)/{sum(len(s) for s in consensus_target_seqs)} residues, "
                "all sequences"
            )

    # Setup columns
    columns, flat_dict = parse_cfg_for_table(eval_config)
    all_columns = columns + ["id_gen", "pdb_path", "L", "task_name"]

    results = []
    binder_chain = None
    gen_target_chain = None

    for idx, sample_root_path in enumerate(maybe_tqdm(sample_root_paths, "Binder evaluation", show_progress)):
        pdb_path = os.path.join(sample_root_path, os.path.basename(sample_root_path) + ".pdb")

        # Validate PDB file exists
        if not os.path.exists(pdb_path):
            logger.warning(f"PDB file not found: {pdb_path}, skipping")
            continue

        # Detect chains on first sample
        if binder_chain is None:
            chains = sorted(set(load_any(pdb_path)[0].chain_id.tolist()))
            binder_chain = chains[-1]
            gen_target_chain = chains[:-1]
            logger.info(f"Detected chains - binder: {binder_chain}, target: {gen_target_chain}")

        row_dict = {
            **flat_dict,
            "id_gen": idx,
            "pdb_path": pdb_path,
            "task_name": target_task_name,
        }

        if cfg_metric.get("compute_binder_metrics", True):
            # Per-sample fixed residue override (e.g. motif residues)
            fixed_residues_override = None
            if get_fixed_residues_fn is not None:
                fixed_residues_override = get_fixed_residues_fn(pdb_path, binder_chain)

            fingerprint = binder_eval_fingerprint(
                **cache_fingerprint_base,
                binder_chain=binder_chain,
                gen_target_chain=gen_target_chain,
                fixed_residues_override=fixed_residues_override,
            )
            # Built per design because the two per-design keys below are part of
            # the hash, so a legacy fingerprint cannot be computed once for the run.
            legacy_fingerprints = [
                binder_eval_fingerprint(
                    **legacy_target_base,
                    binder_chain=binder_chain,
                    gen_target_chain=gen_target_chain,
                    fixed_residues_override=fixed_residues_override,
                )
            ] + [
                binder_eval_fingerprint(
                    **base,
                    interface_cutoff=cutoff,
                    binder_chain=binder_chain,
                    gen_target_chain=gen_target_chain,
                    fixed_residues_override=fixed_residues_override,
                )
                # Both bases, so a package that moved AND changed cutoff still
                # reuses. Neither base carries interface_cutoff here: pairing it
                # with reusable_interface_cutoffs is refused above, for
                # mpnn_fixed, which is the only case that sets it.
                for base in (cache_fingerprint_base, legacy_target_base)
                for cutoff in reusable_interface_cutoffs
            ]
            # What this design IS, not what any folder says about it: the
            # sequences to evaluate, where the interface sits, and what each
            # sequence is made of. Assembled once per design and answering for
            # every folder below -- which is the point. While this work lived
            # inside the folding dispatch, a second complex folder could only be
            # reached by a mechanism that assembled its own.
            #
            # The cache is still keyed on the old fingerprint, which carries the
            # folding model. That now over-invalidates rather than under: a
            # changed folder re-runs the inverse folder, which has a cache of its
            # own (redesign_set_{variant}.json), so the cost is a re-read.
            cached = (
                read_binder_eval_cache(
                    sample_root_path,
                    fingerprint,
                    sequence_types,
                    derivation_fingerprint,
                    legacy_fingerprints,
                    backend=complex_backend,
                )
                if reuse_cached_folding
                else None
            )
            assembled = None
            if cached is not None:
                sequence_type_stats, sequences_dict, _ = cached
                # A cache written before the split carries complex_stats and
                # rmsd_stats too. They are ignored rather than migrated: every
                # number they hold is now produced by score_binders from the same
                # structures, under the same column names, and reading them here
                # would be the gated path surviving inside the unified one.
                if not all((sequence_type_stats.get(t) or {}).get("aa_stats") for t in sequence_types):
                    cached = None
            # A campaign that ran before the unification has its complex folds in
            # the pre-unification cache and nowhere else. Adopt them into the
            # cache score_binders reads, per model, from the structures already
            # on disk -- otherwise every finished campaign refolds: on EFNB3 that
            # is 657 designs x 3 sequences x 5 models, about 42 GPU-hours, to
            # reproduce predictions that are sitting there.
            #
            # Deliberately NOT conditioned on `cached`. That asks whether the
            # file's SEQUENCES are the ones this run wants, and its fingerprint
            # carries the interface cutoff, the redesign count and the folding
            # method -- so a campaign that has moved any of them is told no. It
            # told EFNB3 no on every design, and the run refolded everything.
            # The folds are not the sequences; see legacy_complex_folds.
            #
            # A no-op once the advisory cache exists, so later runs pay one
            # file-exists check per design.
            if consensus_target_seqs:
                legacy = legacy_complex_folds(sample_root_path, complex_backend)
                if legacy is not None:
                    legacy_stats, legacy_sequences = legacy
                    n_adopted += adopt_binder_eval_folds(
                        sample_root_path,
                        complex_backend,
                        consensus_target_seqs,
                        consensus_cfg,
                        legacy_stats,
                        {
                            t: sequences_for_type(t, legacy_sequences, legacy_stats)
                            for t in legacy_stats
                        },
                        derive_tmol=derive_consensus_tmol,
                    )

            if cached is not None:
                n_reused += 1
            else:
                try:
                    assembled = assemble_binder_sequences(
                        pdb_file_path=pdb_path,
                        target_pdb_path=target_pdb_path,
                        tmp_path=sample_root_path,
                        target_pdb_chain=target_pdb_chain,
                        sequence_types=sequence_types,
                        inverse_folding_model=inverse_folding_model,
                        is_target_ligand=is_target_ligand,
                        interface_cutoff=interface_cutoff,
                        gen_target_chain=gen_target_chain,
                        binder_chain=binder_chain,
                        num_redesign_seqs=num_redesign_seqs,
                        shared_redesign_count=redesign_set_size(cfg_metric),
                        fixed_residues_override=fixed_residues_override,
                    )
                except Exception as exc:
                    # One design, not the campaign. The inverse folder runs in a
                    # subprocess and can decline a backbone; nothing above this
                    # caught it, so a single design took the whole evaluate job
                    # down. IL1R1 died exactly this way after 2 of 774 designs.
                    #
                    # Dropped rather than emitted half-filled: apo and ESM below
                    # read these sequences, so without them there is nothing to
                    # carry. Nothing is cached, so the next run rebuilds it.
                    logger.error(
                        f"Binder evaluation failed for {os.path.basename(sample_root_path)}: "
                        f"{type(exc).__name__}: {exc}. Skipping this design; the run continues."
                    )
                    failed_designs.append((os.path.basename(sample_root_path), f"{type(exc).__name__}: {exc}"))
                    continue

                sequences_dict = assembled.sequences_dict
                # Shaped as the row builder already expects, holding only what a
                # design owns. The folding-derived groups are simply absent now.
                sequence_type_stats = {t: {"aa_stats": rows} for t, rows in assembled.aa_stats.items()}

                with open(os.path.join(sample_root_path, "sequence_type_stats.json"), "w") as f:
                    json.dump(sequence_type_stats, f, indent=4)

                write_binder_eval_cache(
                    sample_root_path,
                    fingerprint,
                    sequence_type_stats,
                    sequences_dict,
                    derivation_fingerprint,
                    backend=complex_backend,
                )

            # Extract metrics for each sequence type
            for seq_type in sequence_types:
                if not (sequence_type_stats.get(seq_type) or {}).get("aa_stats"):
                    logger.debug(f"No sequences for {seq_type} at sample {idx}, skipping")
                    continue

                # Find best sample using composite ranking score
                # No ranking here any more. Which redesign a row's scalars describe is
                # a formulation over the per-sequence lists, not a measurement, so
                # analyze chooses it (pick_headline_sequence) and can re-choose it
                # without a refold. Evaluate emits the lists and nothing else.
                #
                # The only per-row scalars left are those equal for every redesign:
                # ProteinMPNN is fixed-length, so binder_length is the backbone's.
                aa_stats_all = sequence_type_stats[seq_type]["aa_stats"]

                row_dict["L"] = aa_stats_all[0]["binder_length"]
                # Which folder produced the complex columns below. analyze and
                # analyze_pooled re-derive verdicts from a CSV with no config in
                # reach, and a pooled frame can hold runs that used different
                # folders, so it travels with the rows like redesign_model does.
                row_dict[COMPLEX_BACKEND_COLUMN] = complex_backend
                # Guarded on membership, not on idx == 0: this block runs once per
                # sequence type, so "first design" fires once per type and the
                # frame ended up carrying the column as many times as there were
                # types. Selecting it by name then yields a DataFrame rather than
                # a Series, and the first caller to treat it as one died several
                # frames below anything that mentioned the column.
                if COMPLEX_BACKEND_COLUMN not in all_columns:
                    all_columns.append(COMPLEX_BACKEND_COLUMN)

                # The complex metrics are emitted below, once per folder, by the
                # loop over folders.complex. There is no separate emission for a
                # first folder any more: advisory_column produces exactly the
                # names this block used to build through `rename` -- verified
                # metric by metric, zero mismatches -- so the artifact is
                # unchanged and only the mechanism behind it is.

                # AA composition, per redesign. ProteinMPNN changes the sequence,
                # so both vectors differ from one redesign to the next; taking
                # aa_stats[0] reported the first redesign's composition for the
                # whole row. It also made {col}_all the twenty counts rather than
                # a list over redesigns, so once the headline moved to analyze the
                # scalar became count number best_idx OUT OF the twenty.
                row_dict[f"{seq_type}_aa_counts_all"] = [
                    packed_aa_counts(a["residue_counts"]) for a in aa_stats_all
                ]
                row_dict[f"{seq_type}_aa_interface_counts_all"] = [
                    packed_aa_counts(a["interface_counts"]) for a in aa_stats_all
                ]
                if idx == 0:
                    all_columns.extend(
                        [f"{seq_type}_aa_counts_all", f"{seq_type}_aa_interface_counts_all"]
                    )

                # Store sequences (best and all). Taken from the stats rather
                # than from sequences_dict so they stay paired with the metrics
                # on this row -- see sequences_for_type.
                seqs = sequences_for_type(seq_type, sequences_dict, sequence_type_stats)
                if seqs:
                    row_dict[f"{seq_type}_sequence_all"] = seqs
                    if idx == 0:
                        all_columns.extend([f"{seq_type}_sequence", f"{seq_type}_sequence_all"])

                    # The inverse folder's own per-sequence quality number.
                    # Computed on every run and discarded until now. Recorded
                    # because it is the leading candidate for ranking which
                    # redesigns become candidate binders, and that choice needs
                    # evidence -- which needs the scores sitting beside the
                    # verdicts. See docs/design-notes/apo-holo-redesign-sharing.md.
                    #
                    # Which way it points depends on the model, so
                    # redesign_score_kind travels with it. Reading the column
                    # without that is how a ranking comes to select the worst
                    # sequences while looking like it works.
                    #
                    # Named for the thing scored, not the scorer: the scorer is
                    # whatever inverse_folding_model selects, and under the
                    # shipped binder config that is SolubleMPNN reporting a
                    # confidence -- so "mpnn_score" would name the wrong model
                    # and imply the wrong convention. It also spared the CSV a
                    # column called mpnn_mpnn_score.
                    scores = redesign_scores_for_type(seq_type, seqs, sequences_dict)
                    if not all(np.isnan(v) for v in scores):
                        row_dict[f"{seq_type}_redesign_score_all"] = scores
                        row_dict["redesign_score_kind"] = REDESIGN_SCORE_KIND.get(inverse_folding_model, "unknown")
                        # Named the same as the monomer track's column so the two
                        # CSVs can be compared on it, which is the only way to
                        # see the ligand-target case where they can diverge.
                        row_dict["redesign_model"] = inverse_folding_model
                        for col in (
                            f"{seq_type}_redesign_score",
                            f"{seq_type}_redesign_score_all",
                            "redesign_score_kind",
                            "redesign_model",
                        ):
                            if col not in all_columns:
                                all_columns.append(col)

                # Apo refolding (optional, gates nothing yet).
                #
                # Placed after the holo verdict so the two sit together on the
                # row: {seq_type}_pass_all[i] says whether sequence i passed with
                # its target, and these say how well it folded without one. The
                # column name is built so that a threshold spec with
                # column_prefix "apo" finds it -- turning this into a joint
                # criterion is a config entry, not a code change.
                if compute_apo and seqs:
                    binder_pdb_path = os.path.join(sample_root_path, f"{os.path.basename(sample_root_path)}_binder.pdb")
                    try:
                        if not os.path.exists(binder_pdb_path):
                            extract_binder_chain_to_pdb(pdb_path, binder_pdb_path, binder_chain)
                        apo_values = apo_refold(
                            seq_type=seq_type,
                            sequences=seqs,
                            binder_pdb_path=binder_pdb_path,
                            sample_root_path=sample_root_path,
                            folding_models=apo_folding_models,
                            rmsd_modes=apo_rmsd_modes,
                            keep_outputs=cfg_metric.get("keep_folding_outputs", True),
                            reuse_cache=reuse_cached_apo,
                            n_esmfold2_seeds=n_esmfold2_seeds,
                            # The redesigns' apo fold and designability are the
                            # same computation on the same sequences. Shared
                            # through the function that owns it, not through a
                            # second cache keyed to agree with the first.
                            share_with_designability=True,
                            shared_count=redesign_set_size(cfg_metric),
                            binder_chain=binder_chain,
                            complex_pdb_path=pdb_path,
                            target_chains=gen_target_chain,
                            inverse_folding_model=inverse_folding_model,
                        )
                    except Exception as exc:
                        logger.error(f"Apo refolding failed for {seq_type} at sample {idx}: {exc}")
                        apo_values = ({}, {}, {})

                    apo_values, apo_confidence, apo_derived = apo_values
                    for model, by_metric in (apo_confidence or {}).items():
                        # Advisory. The campaign folds apo with esmfold2, which
                        # runs on a compressed scale -- a native protein reaches
                        # ~0.65 there -- so an AF2-calibrated floor would reject
                        # nearly everything. Emitted for looking at, and picked
                        # up by the outlier flags in analyze.
                        #
                        # pTM and PAE join pLDDT here: the complex track reports
                        # the whole confidence family, and an apo fold that is
                        # confident residue by residue while its domains float
                        # apart is exactly what a pTM says and a mean pLDDT does
                        # not. Absent for folds cached before the backends wrote
                        # them down -- the folder is the only thing that knows.
                        for metric in APO_CONFIDENCE_SUFFIXES:
                            if metric not in by_metric:
                                continue
                            col = apo_confidence_column(seq_type, model, metric)
                            row_dict[f"{col}_all"] = by_metric[metric]
                            for name in (col, f"{col}_all"):
                                if name not in all_columns:
                                    all_columns.append(name)

                    for (mode, model), values in apo_values.items():
                        col = apo_column(seq_type, mode, model)
                        row_dict[f"{col}_all"] = values
                        for name in (col, f"{col}_all"):
                            if name not in all_columns:
                                all_columns.append(name)

                    # Read off the kept apo structures, not reported by the folder:
                    # the binder's own surface and its secondary structure, by the
                    # same engines and the same eight-state counts the complex side
                    # uses. Advisory like the pLDDT above -- nothing gates on them.
                    for model, by_metric in (apo_derived or {}).items():
                        for metric in MONOMER_DERIVED_SUFFIXES:
                            if metric not in by_metric:
                                continue
                            col = apo_derived_column(seq_type, model, metric)
                            row_dict[f"{col}_all"] = by_metric[metric]
                            for name in (col, f"{col}_all"):
                                if name not in all_columns:
                                    all_columns.append(name)

                # ESM pseudo-perplexity metrics (optional).
                #
                # Cached separately from the refolding above, in
                # esm_eval_cache.json beside binder_eval_cache.json. Adding the
                # sequence model to binder_eval_fingerprint would change every
                # fingerprint and invalidate the refolding caches already on
                # disk, forcing a full refold to gain ESM caching; a separate
                # cache keyed on (model, backend, sequence) avoids that and
                # reuses per sequence, so growing sequence_types keeps what it
                # already has. It also cannot serve ESM2 numbers for an ESMC run.
                #
                # Not an optimisation to skip: a 6B scorer costs ~15s per
                # 140-residue sequence, so a resumed evaluation would repay hours
                # of scoring while the folding it accompanies is free.
                # Per-sequence verdict against the success criteria, positionally
                # aligned with every other *_all column on this row, so
                # {seq_type}_sequence_all[i] and its metrics and its refolded
                # structure path all describe the sequence judged by
                # {seq_type}_pass_all[i]. None when a criterion's column is
                # missing -- unjudged, not failed.
                #
                # AFTER the apo block, not before it. The apo criterion is part of
                # the gate (DEFAULT_PROTEIN_BINDER_THRESHOLDS), so its column has to
                # exist on the row before the verdict is taken. Computing the verdict
                # first meant one criterion's column was always absent, so
                # per_sequence_pass returned None every time and no {seq_type}_pass
                # column was ever written -- a whole run of correct apo numbers with
                # no verdict beside them. The original order was chosen for how the
                # row reads, which stopped being the only consideration the moment
                # apo became a criterion rather than a decoration.
                pass_vector = per_sequence_pass(row_dict, seq_type, success_thresholds)
                if pass_vector is not None:
                    row_dict[f"{seq_type}_pass_all"] = pass_vector
                    # Not gated on idx == 0: the criteria columns can be absent
                    # for the first design and present later, and reindex would
                    # then drop the column for every design that had it.
                    for col in (f"{seq_type}_pass", f"{seq_type}_pass_all"):
                        if col not in all_columns:
                            all_columns.append(col)

                if cfg_metric.get("compute_esm_metrics", False) and ESM_AVAILABLE and seqs:
                    esm_model = cfg_metric.get("esm_model", "facebook/esm2_t33_650M_UR50D")
                    esm_df = compute_esm_ppl_for_sequences(
                        seqs,
                        model_name=esm_model,
                        backend=cfg_metric.get("esm_backend", "auto"),
                        # bfloat16 for ESMC unless overridden: it is scored in
                        # bfloat16 regardless, and float32 weights cost ~12 GiB on
                        # a card that also holds ESMFold2 and JAX.
                        dtype=cfg_metric.get("esm_dtype", "auto"),
                        max_batch_tokens=cfg_metric.get("esm_batch_tokens", DEFAULT_ESM_BATCH_TOKENS),
                        cache_dir=sample_root_path,
                        reuse_cache=cfg_metric.get("reuse_cached_esm", True),
                    )

                    row_dict[f"{seq_type}_esm_pseudo_perplexity_all"] = esm_df["esm_pseudo_perplexity"].tolist()
                    row_dict[f"{seq_type}_esm_log_likelihood_all"] = esm_df["esm_log_likelihood"].tolist()

                    if idx == 0:
                        all_columns.extend(
                            [
                                f"{seq_type}_esm_pseudo_perplexity",
                                f"{seq_type}_esm_log_likelihood",
                                f"{seq_type}_esm_pseudo_perplexity_all",
                                f"{seq_type}_esm_log_likelihood_all",
                            ]
                        )

                # Advisory second-opinion refolding (optional, gates nothing).
                #
                # Scored per design and cached like the ESM scores, because a
                # diffusion folder costs minutes per complex -- far more than the
                # primary refold -- and a resumed evaluation must not repay it.
                #
                # Every sequence, always. Folding only the ranked-best one
                # conditions the advisory sample on the primary backend's ranking,
                # which is the opposite of what calibration needs: it makes rank
                # disagreement unmeasurable (whether this backend would pick a
                # different winner), estimates any fit on the primary's upper tail
                # only, and never folds the sequences the primary rejected -- the
                # interesting failures. Since the point of these columns is to
                # decide whether the backend could replace the primary one,
                # best-only defeats it.
                #
                # There used to be a consensus_best_only knob for "cheap monitoring
                # of a characterised backend". It is gone: the saving is one fold
                # per extra redesign, and the cost is a run whose advisory _all
                # lists hold a single entry -- which freezes the primary's ranking
                # into the artifact, so no later stage can re-rank or re-calibrate
                # from it. A cheaper run that cannot answer the question it was
                # written to answer is not cheaper.
                for backend_name in complex_folders:
                    to_score = seqs
                    advisory = score_binders(
                        backend_name,
                        consensus_target_seqs,
                        to_score,
                        cfg=consensus_cfg,
                        cache_dir=sample_root_path,
                        reuse_cache=reuse_cached_consensus,
                        keep_structures=cfg_metric.get("keep_folding_outputs", True),
                        # The designed complex the geometry family is measured
                        # against -- the same structure the primary backend's
                        # scRMSD columns compare to.
                        reference_pdb_path=pdb_path,
                        derive_tmol=derive_consensus_tmol,
                        # The draws themselves, not a scalar per metric. Which
                        # reduction is right -- a mean today, for every metric --
                        # is a formulation over recorded values, and analyze
                        # re-asks it on every run for the cost of a re-read.
                        # Freezing it here is what made AF2's five models
                        # unrecoverable without predicting them again.
                        reduce=False,
                        # What a folder needs beyond the sequences. ESMFold2
                        # ignores every field and folds from sequence; AF2
                        # templates on the design and reads the target. Passing
                        # the same context to both is what lets either serve
                        # this track -- which is the whole of the old
                        # primary/advisory distinction, now gone.
                        context=ComplexFoldContext(
                            design_pdb=pdb_path,
                            target_pdb=target_pdb_path,
                            target_chains=tuple(gen_target_chain),
                            binder_chain=binder_chain,
                            design_name=os.path.basename(sample_root_path),
                            output_dir=sample_root_path,
                            # RF3 is an object holding weights rather than a
                            # function of its inputs, and it reaches this loop
                            # like every other folder now.
                            runner=(folding_model_specs or {}).get("runner"),
                            is_target_ligand=is_target_ligand,
                        ),
                    )
                    # `advisory` is parallel to `seqs`, so the headline must be the
                    # same sequence the primary columns describe. Using 0 here made
                    # {seq}_esmfold2_i_pAE and {seq}_complex_i_pAE describe
                    # different redesigns whenever the best was not the first --
                    # the exact pairing failure sequences_for_type exists to stop.
                    new_cols = []
                    for suffix in (
                        *CONSENSUS_METRIC_SUFFIXES,
                        *consensus_derived_suffixes(derive_consensus_tmol),
                        *CONSENSUS_PROVENANCE_SUFFIXES,
                    ):
                        col = advisory_column(seq_type, backend_name, suffix)
                        # Always, now that best-only is gone. These lists are what
                        # make the advisory numbers re-rankable and calibratable
                        # later; under best-only they held one entry and the
                        # primary's ranking was baked into the artifact.
                        col_all = f"{col}_all"
                        row_dict[col_all] = [m.get(suffix, np.nan) for m in advisory]
                        new_cols.append(col_all)
                    # Where the advisory structure landed, when keep_folding_outputs
                    # kept it. Not a metric, so emitted explicitly, and built through
                    # advisory_column like the rest -- the slot scheme puts the backend
                    # in a slot of its own, so nothing here has to avoid a substring.
                    path_col = advisory_column(seq_type, backend_name, "pdb_path")
                    row_dict[f"{path_col}_all"] = [m.get("pdb_path") for m in advisory]
                    new_cols.append(f"{path_col}_all")
                    # Which structure, not just which path. A refold writes the
                    # same name -- the path is addressed on the binder sequence,
                    # not on the prediction -- and a fold is not reproducible from
                    # its seed, so a frame naming only the path cannot say whether
                    # the file beside it today is the one these numbers came from.
                    # With the digest it can, and a mismatch is detectable instead
                    # of silent.
                    sha_col = advisory_column(seq_type, backend_name, STRUCTURE_DIGEST_KEY)
                    row_dict[f"{sha_col}_all"] = [m.get(STRUCTURE_DIGEST_KEY) for m in advisory]
                    new_cols.append(f"{sha_col}_all")
                    if idx == 0:
                        # The contract of these columns is that they cannot change a
                        # pass/fail decision. Checked against the columns the criteria
                        # actually resolve to -- not against every column in the row,
                        # and not against any property of their names. A model can
                        # serve both tracks: esmfold2 here is an advisory backend AND
                        # the apo folding model, and the apo criterion is gated on
                        # purpose.
                        report_gated_and_reported_columns(
                            new_cols,
                            gated_columns(row_dict, seq_type, success_thresholds),
                            set(all_columns),
                        )
                        # The headline-agreement check does NOT belong here.
                        # Evaluate emits per-sequence lists and no scalars at all,
                        # so there is no headline yet for the advisory columns to
                        # agree or disagree with; asserting it here compared every
                        # scalar against None and reported, correctly but
                        # uselessly, that no index explained a headline that did
                        # not exist. analyze runs it once the headline is chosen
                        # and the verdicts refreshed -- see
                        # assert_frame_headline_indices_agree.
                        all_columns.extend(new_cols)

        # The complex structures were never cleaned up at all. keep_folding_outputs
        # governed the apo folds and the second complex folder's structures while
        # the primary's -- n_af2_models per sequence per design, the largest set a
        # run produces -- were written unconditionally and kept forever. One flag,
        # honoured by every folder, and the PAE matrices survive either way.
        if not cfg_metric.get("keep_folding_outputs", True):
            for folder_dir in (AF2_SAVE_LOCATION, f"{complex_backend}_complex"):
                removed, kept = drop_structures_keeping_sidecars(
                    os.path.join(sample_root_path, folder_dir)
                )
                if removed or kept:
                    logger.debug(
                        f"{sample_root_path}/{folder_dir}: removed {removed} complex structures, "
                        f"kept {kept} sidecars"
                    )

        results.append(row_dict)

    if reuse_cached_folding:
        logger.info(f"Binder evaluation reused cached refolding for {n_reused}/{len(results)} designs")
    if n_adopted:
        logger.info(
            f"Adopted {n_adopted} (sequence, draw) complex folds from this campaign's pre-unification "
            f"cache into the per-folder cache, per model, without refolding any of them"
        )
    if failed_designs:
        # Loud and enumerated. A design skipped quietly is a row missing from the
        # frame with nothing saying why, and the counts downstream would simply
        # be smaller than the campaign expected.
        logger.warning(
            f"Binder evaluation skipped {len(failed_designs)} design(s) whose complex refold failed; "
            f"they carry no binder metrics and will refold on the next run. "
            + "; ".join(f"{name}: {why}" for name, why in failed_designs[:5])
            + (f" (and {len(failed_designs) - 5} more)" if len(failed_designs) > 5 else "")
        )

    df = pd.DataFrame(results).reindex(columns=dedupe_columns(all_columns, "Binder results"))
    # Carried out-of-band rather than as columns: both are properties of the run,
    # constant across every row, and the caller writes them to a sidecar beside
    # the CSV. Attached to the frame instead of recomputed by the caller so the
    # record cannot describe a different resolution than the one actually applied.
    df.attrs["success_thresholds"] = success_thresholds
    if binder_chain is not None and any(t.startswith("mpnn") for t in sequence_types):
        df.attrs["redesign_conditioning"] = redesign_conditioning(complex_mpnn_chains(gen_target_chain, binder_chain))
    return df


# =============================================================================
# Interface Metrics - Single PDB Functions (Core Building Blocks)
# =============================================================================


def compute_bioinformatics_metrics_single(
    pdb_path: str,
    binder_chain: str,
    target_chain: str,
    interface_cutoff: float = DEFAULT_CONTACT_CUTOFF,
) -> dict[str, Any]:
    """
    Compute bioinformatics interface metrics for a single PDB.

    Args:
        pdb_path: Path to PDB file
        binder_chain: Chain ID of the binder
        target_chain: Chain ID(s) of the target (comma-separated if multiple)

    Returns:
        Dictionary of metric names to values. Returns NaN if dependencies unavailable.
    """
    if not PR_ALTERNATIVE_AVAILABLE:
        return dict.fromkeys(BIOINFORMATICS_METRIC_COLS, np.nan)

    try:
        scores, _, _ = pr_alternative_score_interface(
            pdb_path,
            binder_chain=binder_chain,
            target_chain=target_chain,
            sasa_engine="auto",
            interface_cutoff=interface_cutoff,
        )
        # The scorer already names each value's scope, so nothing is renamed
        # here; taking the declared set rather than the whole dict keeps a new
        # score from silently becoming a column nobody declared.
        return {name: scores[name] for name in BIOINFORMATICS_METRIC_COLS if name in scores}
    except Exception as e:
        logger.error(f"Bioinformatics metrics failed for {pdb_path}: {e}")
        return dict.fromkeys(BIOINFORMATICS_METRIC_COLS, np.nan)


def compute_tmol_metrics_single(
    pdb_path: str,
    tmol_model: Any | None = None,
) -> dict[str, Any]:
    """
    Compute TMOL force field metrics for a single PDB.

    Args:
        pdb_path: Path to PDB file
        tmol_model: Initialized TmolRewardModel instance

    Returns:
        Dictionary of metric names to values. Returns NaN if TMOL unavailable.
    """
    if not TMOL_AVAILABLE or tmol_model is None:
        return dict.fromkeys(TMOL_METRIC_COLS, np.nan)

    # The reward-key mapping lives in metrics.tmol_interface, which the advisory
    # track reads the same four metrics through. NaN rather than absent here,
    # because this function's callers build a fixed column set per PDB and a
    # missing key would shorten one design's row.
    metrics = tmol_interface_metrics(pdb_path, model=tmol_model)
    return {name: metrics.get(name, np.nan) for name in TMOL_METRIC_COLS}


# =============================================================================
# Interface Metrics - Unified Computation
# =============================================================================


def compute_interface_metrics(
    pdb_paths: list[str],
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """
    Unified function to compute interface metrics for a batch of PDB files.

    This is the main entry point for computing any combination of interface metrics.
    Each metric type can be independently enabled/disabled for full control.

    If a dependency is not available (e.g., TMOL not installed), the corresponding
    metrics will be set to NaN and a warning will be logged.

    Args:
        pdb_paths: List of PDB file paths
        compute_bioinformatics: Whether to compute bioinformatics metrics (SC, SASA, hydrophobicity)
        compute_tmol: Whether to compute TMOL force field metrics
        show_progress: Whether to show progress bar

    Returns:
        List of dictionaries containing metrics for each PDB
    """
    if not pdb_paths:
        logger.debug("compute_interface_metrics called with empty pdb_paths")
        return []

    # Log what metrics will be computed
    enabled_metrics = []
    if compute_bioinformatics:
        enabled_metrics.append("bioinformatics")
    if compute_tmol:
        enabled_metrics.append("tmol")

    logger.info(f"Computing interface metrics for {len(pdb_paths)} PDBs: {enabled_metrics}")

    # Check if any metrics are requested
    if not enabled_metrics:
        logger.warning("No metrics requested in compute_interface_metrics")
        return [{"pdb_path": p} for p in pdb_paths]

    # Log availability warnings upfront
    if compute_bioinformatics and not PR_ALTERNATIVE_AVAILABLE:
        logger.warning("Bioinformatics metrics requested but PR Alternative not available. Metrics will be NaN.")

    if compute_tmol and not TMOL_AVAILABLE:
        logger.warning("TMOL metrics requested but TMOL not available. Metrics will be NaN.")

    # Initialize TMOL model lazily if needed and available
    tmol_model = None
    if compute_tmol and TMOL_AVAILABLE:
        try:
            tmol_model = TmolRewardModel(enable_hbond=True, enable_elec=True)
        except Exception as e:
            logger.warning(f"Failed to initialize TMOL model: {e}. TMOL metrics will be NaN.")

    # Detect chains from first PDB (assume consistent across batch)
    binder_chain = None
    target_chain = None
    multi_target = False

    results = []

    for pdb_path in maybe_tqdm(pdb_paths, "Interface metrics", show_progress):
        metrics = {"pdb_path": pdb_path}

        # Detect chains on first iteration
        if binder_chain is None:
            binder_chain, target_chains, multi_target = get_binder_chain_from_complex(
                pdb_path, return_multi_target=True
            )
            target_chain = ",".join(target_chains)
            logger.debug(f"Detected chains - binder: {binder_chain}, target: {target_chain}")

        # Bioinformatics metrics (return 0 for multi-target complexes)
        if compute_bioinformatics:
            if multi_target:
                # NaN, not 0: a zero dSASA and a zero shape complementarity read
                # as a measured non-interface. The interface definition handles
                # multi-chain targets now, so this guard is about the rest of the
                # block, and what it cannot compute it must not invent.
                logger.info("Multi-target complex detected. Bioinformatics metrics will be NaN.")
                bio_metrics = dict.fromkeys(BIOINFORMATICS_METRIC_COLS, np.nan)
            else:
                logger.info("Computing bioinformatics metrics...")
                bio_metrics = compute_bioinformatics_metrics_single(pdb_path, binder_chain, target_chain)
            metrics.update(bio_metrics)

        # TMOL metrics
        if compute_tmol:
            logger.info("Computing TMOL metrics...")
            metrics.update(compute_tmol_metrics_single(pdb_path, tmol_model))

        results.append(metrics)

    logger.info(f"Interface metrics complete: {len(results)} PDBs processed")
    return results


def compute_interface_metrics_df(
    cfg: DictConfig,
    pdb_paths: list[str],
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Compute interface metrics and return as a DataFrame with config columns.

    This is a convenience wrapper around compute_interface_metrics that adds
    the config-derived columns needed for merging with other evaluation results.

    Args:
        cfg: Configuration (used to extract columns for DataFrame)
        pdb_paths: List of PDB file paths
        compute_bioinformatics: Whether to compute bioinformatics metrics
        compute_tmol: Whether to compute TMOL metrics
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with config columns + metric columns
    """
    columns, flat_dict = parse_cfg_for_table(cfg)

    # Compute metrics
    metrics_list = compute_interface_metrics(
        pdb_paths=pdb_paths,
        compute_bioinformatics=compute_bioinformatics,
        compute_tmol=compute_tmol,
        show_progress=show_progress,
    )

    # Build result rows with config columns
    results = []
    for i, metrics in enumerate(metrics_list):
        row = {**flat_dict, "id_gen": i, **metrics}
        results.append(row)

    # Build column order
    metric_cols = get_metric_columns(
        compute_bioinformatics=compute_bioinformatics,
        compute_tmol=compute_tmol,
    )

    all_columns = columns + ["id_gen", "pdb_path"] + metric_cols

    return pd.DataFrame(results).reindex(columns=dedupe_columns(all_columns, "Interface metrics"))


# =============================================================================
# Refolded Structure Metrics
# =============================================================================


