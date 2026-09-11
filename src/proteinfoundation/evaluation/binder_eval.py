"""
Binder-specific evaluation metrics.

This module provides functions for evaluating protein binder designs:
- Refolding with structure prediction models (ColabDesign, Boltz2, RF3, Protenix)
- Interface analysis (bioinformatics metrics)
- Force field metrics (hydrogen bonds, electrostatics)
"""

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
    binder_eval_fingerprint,
    digest_file,
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
    apo_fold_fingerprint,
    apo_plddt_column,
    check_thresholds_are_computable,
    dedupe_columns,
    extract_binder_chain_to_pdb,
    gated_columns,
    get_binder_chain_from_complex,
    get_metric_columns,
    per_sequence_pass,
    resolve_success_thresholds,
)
from proteinfoundation.evaluation.esm_eval import (
    DEFAULT_ESM_BATCH_TOKENS,
    ESM_AVAILABLE,
    compute_esm_ppl_for_sequences,
)
from proteinfoundation.evaluation.monomer_eval_utils import (
    per_model_plddt,
    write_monomer_fold_cache,
)
from proteinfoundation.evaluation.utils import maybe_tqdm, parse_cfg_for_table, redesign_conditioning
from proteinfoundation.metrics.binder_metrics import complex_mpnn_chains, run_binder_eval
from proteinfoundation.metrics.column_names import backend_for_folding_method, rename
from proteinfoundation.metrics.consensus_folding import (
    CONSENSUS_DERIVED_SUFFIXES,
    CONSENSUS_METRIC_SUFFIXES,
    advisory_column,
    assert_columns_are_advisory,
    assert_headline_indices_agree,
    available_backends,
    score_binders,
)
from proteinfoundation.metrics.ensembling import GEOMETRY_REDUCTION_VERSION
from proteinfoundation.metrics.interface import DEFAULT_CONTACT_CUTOFF, INTERFACE_DERIVATION_VERSION
from proteinfoundation.metrics.inverse_folding_models import REDESIGN_SCORE_KIND, resolve_inverse_folding_model
from proteinfoundation.metrics.seeding import SEED_DERIVATION_VERSION
from proteinfoundation.result_analysis.analysis_utils import SEQUENCE_TYPES
from proteinfoundation.result_analysis.binder_analysis_utils import COMPLEX_BACKEND_COLUMN
from proteinfoundation.rewards.base_reward import REWARD_KEY

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
    """Initialize folding model specs for binder evaluation.

    Supported models: colabdesign, protenix_*, rf3_*, boltz2_*.

    Args:
        folding_model: Name of the folding model (e.g. ``"colabdesign"``,
            ``"protenix_v0.4.0"``, ``"rf3_latest"``, ``"boltz2_v1"``).
        target_pdb_chain: Sorted chain IDs of the target structure.
        target_task_name: Task name used to resolve MSA / template paths.
        is_target_ligand: Whether the target is a small-molecule ligand.

    Returns:
        Dictionary with ``"model_name"`` and model-specific runner / path keys.

    Raises:
        ValueError: If the model name is unsupported or incompatible with the
            target type.
    """
    target_pdb_chain = sorted(target_pdb_chain)

    if folding_model == "colabdesign":
        if is_target_ligand:
            raise ValueError("ColabDesign does not support ligand-protein complex folding")
        return {"model_name": "colabdesign"}

    elif "rf3" in folding_model:
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

    else:
        raise ValueError(f"Folding model '{folding_model}' not supported")


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
) -> dict[tuple[str, str], list[float]]:
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

    Returns ``({(mode, model): [rmsd per sequence]}, {model: [pLDDT per
    sequence]})``. Empty when nothing could be folded; a failed fold is ``inf``
    for that sequence, not a missing row, so the lists stay aligned with the
    sequences they describe. A fold with no readable confidence is NaN, which is
    the same distinction: unmeasured rather than bad.
    """
    from proteinfoundation.evaluation.monomer_eval import (
        compute_scrmsd_from_folded,
        evaluate_self_consistency,
        fold_sequences,
    )
    from proteinfoundation.metrics.folding_models import folding_model_identity
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
            return {}, {}
        return (
            {
                (mode, m): result.rmsd_values.get(mode, {}).get(m, [float("inf")] * len(sequences))
                for mode in rmsd_modes
                for m in folding_models
            },
            per_model_plddt(result.plddt, folding_models, len(sequences)),
        )

    suffix = f"apo_{seq_type}"
    fingerprint = apo_fold_fingerprint(
        binder_pdb_path=binder_pdb_path,
        sequences=sequences,
        folding_models=list(folding_models),
        model_identities={m: folding_model_identity(m) for m in folding_models},
    )
    from proteinfoundation.evaluation.monomer_eval_utils import _fold_seeds, average_folds, read_monomer_folds

    name = f"{pdb_name_from_path(binder_pdb_path)}_{suffix}"
    # Sequences are an argument here rather than an output, so the seeds can be
    # derived up front -- unlike the codesignability path, where they come out of
    # the inverse folder and the cache has to be read first.
    seeds = _fold_seeds(name, suffix, list(sequences), folding_models, n_esmfold2_seeds)
    stored = read_monomer_folds(sample_root_path, suffix, fingerprint, name=name) if reuse_cache else None
    per_seed = {seed: stored[seed] for seed in seeds if stored and seed in stored}
    if len(per_seed) < len(seeds):
        logger.info(f"{len(per_seed)}/{len(seeds)} apo seeds cached for {name}; folding the rest")

    for seed in seeds:
        if seed in per_seed:
            continue
        folded = fold_sequences(
            sequences=sequences,
            output_dir=sample_root_path,
            name=name,
            folding_models=folding_models,
            suffix=suffix,
            cache_dir=None,
            keep_outputs=keep_outputs,
            seed=seed,
        )
        scored = compute_scrmsd_from_folded(
            reference_pdb_path=binder_pdb_path,
            folding_results=folded,
            rmsd_modes=rmsd_modes,
        )
        scored.sequences = sequences
        write_monomer_fold_cache(
            sample_root_path,
            suffix,
            fingerprint,
            scored,
            keep_outputs,
            seed=seed,
            seed_index=seeds.index(seed),
            name=name,
        )
        per_seed[seed] = {
            "sequences": list(scored.sequences),
            "rmsd_values": scored.rmsd_values,
            "best_rmsd": scored.best_rmsd,
            "folded_paths": list(scored.folded_paths),
            "plddt": scored.plddt,
        }

    averaged = average_folds(per_seed) or {}
    values = averaged.get("rmsd_values", {})
    rmsds = {
        (mode, m): values.get(mode, {}).get(m, [float("inf")] * len(sequences))
        for mode in rmsd_modes
        for m in folding_models
    }
    return rmsds, per_model_plddt(averaged.get("plddt"), folding_models, len(sequences))


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

    # Initialize folding model
    folding_model = cfg_metric.get("binder_folding_method", "colabdesign")
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
        cfg_metric.get("inverse_folding_model", "protein_mpnn"), is_target_ligand
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
    apo_folding_models = list(cfg_metric.get("apo_folding_models", ["esmfold"]) or [])
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

    # Advisory second-opinion refolding. Off unless metric.consensus_backends is
    # set; emits {seq_type}_{backend}_{metric} columns and gates nothing. These
    # backends fold protein-protein complexes, so a ligand target has no target
    # sequence to fold against and the whole feature is skipped.
    consensus_backends = list(cfg_metric.get("consensus_backends", []) or [])
    consensus_cfg = dict(cfg_metric.get("consensus_cfg", {}) or {})
    # The advisory folds are ESMFold2 too, so they answer to the same knob --
    # otherwise metric.n_esmfold2_seeds means "three seeds, except for the
    # expensive folds", which is not what it says. consensus_cfg.n_seeds still
    # wins if a config sets it, since the complex fold is several times the cost
    # of a monomer one and is the sensible place to want a different count.
    consensus_cfg.setdefault("n_seeds", n_esmfold2_seeds)
    reuse_cached_consensus = cfg_metric.get("reuse_cached_consensus", True)
    consensus_target_seqs: list[str] = []
    if consensus_backends:
        unknown = [b for b in consensus_backends if b not in available_backends()]
        if unknown:
            logger.error(f"Unknown consensus_backends {unknown}; known: {available_backends()}. Skipping those.")
            consensus_backends = [b for b in consensus_backends if b not in unknown]
    if consensus_backends and is_target_ligand:
        logger.info("Advisory refolding skipped: these backends fold protein complexes, target is a ligand")
        consensus_backends = []
    if consensus_backends:
        consensus_target_seqs = _target_chain_sequences(target_pdb_path, target_pdb_chain)
        if not consensus_target_seqs:
            consensus_backends = []
        else:
            logger.info(
                f"Advisory refolding enabled: {consensus_backends}, target "
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
            cached = (
                read_binder_eval_cache(
                    sample_root_path,
                    fingerprint,
                    sequence_types,
                    derivation_fingerprint,
                    legacy_fingerprints,
                )
                if reuse_cached_folding
                else None
            )
            if cached is not None:
                sequence_type_stats, sequences_dict, derivation_stale = cached
                if derivation_stale:
                    from proteinfoundation.metrics.binder_metrics import recompute_derived

                    # The structures this run wants are already on disk; only the
                    # rules for reading numbers off them changed. Refolding to
                    # learn that a max is not a mean, or that a cutoff moved,
                    # would spend hours recomputing arithmetic.
                    if recompute_derived(
                        sequence_type_stats,
                        pdb_path,
                        binder_chain,
                        gen_target_chain,
                        is_target_ligand,
                        n_af2_models,
                        interface_cutoff,
                    ):
                        write_binder_eval_cache(
                            sample_root_path,
                            fingerprint,
                            sequence_type_stats,
                            sequences_dict,
                            derivation_fingerprint,
                        )
                    else:
                        # Something needed is missing or unrecorded. A row where
                        # some sequences answer to the new rules and some to the
                        # old is worse than a refold, so drop the cache rather
                        # than patch part of it.
                        cached = None
            if cached is not None:
                n_reused += 1
            else:
                _, _, sequence_type_stats, sequences_dict = run_binder_eval(
                    pdb_file_path=pdb_path,
                    target_pdb_path=target_pdb_path,
                    folding_model_specs=folding_model_specs,
                    tmp_path=sample_root_path,
                    target_pdb_chain=target_pdb_chain,
                    sequence_types=sequence_types,
                    inverse_folding_model=inverse_folding_model,
                    gen_target_chain=gen_target_chain,
                    binder_chain=binder_chain,
                    interface_cutoff=interface_cutoff,
                    is_target_ligand=is_target_ligand,
                    num_redesign_seqs=num_redesign_seqs,
                    fixed_residues_override=fixed_residues_override,
                    n_af2_models=n_af2_models,
                )

                # Save raw stats
                with open(os.path.join(sample_root_path, "sequence_type_stats.json"), "w") as f:
                    json.dump(sequence_type_stats, f, indent=4)

                write_binder_eval_cache(
                    sample_root_path, fingerprint, sequence_type_stats, sequences_dict, derivation_fingerprint
                )

            # Extract metrics for each sequence type
            for seq_type in sequence_types:
                seq_stats = sequence_type_stats[seq_type]["complex_stats"]
                if not seq_stats:
                    logger.debug(f"No complex stats for {seq_type} at sample {idx}, skipping")
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
                aa_stats = aa_stats_all[0]

                row_dict["L"] = aa_stats["binder_length"]
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

                # Complex metrics (best and all). Named through the same mapping
                # the migration uses, so emission and rename cannot drift into
                # agreeing only by inspection.
                for metric in seq_stats[0]:
                    col = rename(f"{seq_type}_complex_{metric.removeprefix('complex_')}", complex_backend)
                    row_dict[f"{col}_all"] = [s[metric] for s in seq_stats]
                    if idx == 0:
                        all_columns.append(f"{col}_all")

                # RMSD metrics (best and all). The keys already carry their scope
                # -- complex_scRMSD_ca is the whole complex, binder_scRMSD_ca the
                # binder within it -- so the mapping places them.
                for metric in sequence_type_stats[seq_type]["rmsd_stats"][0]:
                    col = rename(f"{seq_type}_{metric}", complex_backend)
                    row_dict[f"{col}_all"] = [s[metric] for s in sequence_type_stats[seq_type]["rmsd_stats"]]
                    if idx == 0:
                        all_columns.append(f"{col}_all")

                # AA composition
                res_count = [0] * len(OF_RESTYPES)
                interface_count = [0] * len(OF_RESTYPES)
                for aa, count in aa_stats["residue_counts"].items():
                    if aa in OF_RESTYPES:
                        res_count[OF_RESTYPES.index(aa)] += count
                for aa, count in aa_stats["interface_counts"].items():
                    if aa in OF_RESTYPES:
                        interface_count[OF_RESTYPES.index(aa)] += count

                row_dict[f"{seq_type}_aa_counts_all"] = res_count
                row_dict[f"{seq_type}_aa_interface_counts_all"] = interface_count
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
                        )
                    except Exception as exc:
                        logger.error(f"Apo refolding failed for {seq_type} at sample {idx}: {exc}")
                        apo_values = ({}, {})

                    apo_values, apo_plddt = apo_values
                    for model, values in (apo_plddt or {}).items():
                        # Advisory. The campaign folds apo with esmfold2, which
                        # runs on a compressed scale -- a native protein reaches
                        # ~0.65 there -- so an AF2-calibrated floor would reject
                        # nearly everything. Emitted for looking at, and picked
                        # up by the outlier flags in analyze.
                        col = apo_plddt_column(seq_type, model)
                        row_dict[f"{col}_all"] = values
                        for name in (col, f"{col}_all"):
                            if name not in all_columns:
                                all_columns.append(name)

                    for (mode, model), values in apo_values.items():
                        col = apo_column(seq_type, mode, model)
                        row_dict[f"{col}_all"] = values
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
                for backend_name in consensus_backends:
                    to_score = seqs
                    advisory = score_binders(
                        backend_name,
                        consensus_target_seqs,
                        to_score,
                        cfg=consensus_cfg,
                        cache_dir=sample_root_path,
                        reuse_cache=reuse_cached_consensus,
                        keep_structures=cfg_metric.get("keep_folding_outputs", True),
                    )
                    # `advisory` is parallel to `seqs`, so the headline must be the
                    # same sequence the primary columns describe. Using 0 here made
                    # {seq}_esmfold2_i_pAE and {seq}_complex_i_pAE describe
                    # different redesigns whenever the best was not the first --
                    # the exact pairing failure sequences_for_type exists to stop.
                    new_cols = []
                    for suffix in (*CONSENSUS_METRIC_SUFFIXES, *CONSENSUS_DERIVED_SUFFIXES):
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
                    if idx == 0:
                        # The contract of these columns is that they cannot change a
                        # pass/fail decision. Checked against the columns the criteria
                        # actually resolve to -- not against every column in the row,
                        # and not against any property of their names. A model can
                        # serve both tracks: esmfold2 here is an advisory backend AND
                        # the apo folding model, and the apo criterion is gated on
                        # purpose.
                        assert_columns_are_advisory(
                            new_cols,
                            gated_columns(row_dict, seq_type, success_thresholds),
                            set(all_columns),
                        )
                        # And that the headline they carry is the same sequence the
                        # primary headline describes. Checked on the row, so a
                        # future call site cannot reintroduce the mismatch quietly.
                        assert_headline_indices_agree(row_dict, seq_type, backend_name)
                        all_columns.extend(new_cols)

        results.append(row_dict)

    if reuse_cached_folding:
        logger.info(f"Binder evaluation reused cached refolding for {n_reused}/{len(results)} designs")

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

    try:
        result = tmol_model.score(pdb_path=pdb_path, requires_grad=False)
        return {
            "n_interface_hbonds_tmol": result[REWARD_KEY]["n_interface_hbonds"].item(),
            "total_interface_hbond_energy_tmol": result[REWARD_KEY]["total_interface_hbond_energy"].item(),
            "total_interface_elec_energy_tmol": result[REWARD_KEY]["total_interface_elec_energy"].item(),
            "n_interface_elec_interactions_tmol": result[REWARD_KEY]["n_interface_elec_interactions"].item(),
        }
    except Exception as e:
        logger.error(f"TMOL error for {pdb_path}: {e}")
        return dict.fromkeys(TMOL_METRIC_COLS, np.nan)


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


def merge_metrics_to_df(
    df: pd.DataFrame,
    metrics_list: list[dict[str, Any]],
    sample_names: list[str],
    column_prefix: str,
    skip_cols: set | None = None,
) -> pd.DataFrame:
    """
    Merge computed metrics back into the original DataFrame.

    Args:
        df: Original DataFrame
        metrics_list: List of metric dictionaries
        sample_names: List of sample names to match against pdb_path
        column_prefix: Prefix for new column names
        skip_cols: Columns to skip when merging

    Returns:
        Updated DataFrame with merged metrics
    """
    if skip_cols is None:
        skip_cols = {"pdb_path"}

    updated_df = df.copy()

    for sample_name, metrics in zip(sample_names, metrics_list, strict=False):
        sample_mask = updated_df["pdb_path"].str.contains(sample_name, regex=False)
        if not sample_mask.any():
            continue

        for col, value in metrics.items():
            if col in skip_cols:
                continue

            col_name = f"{column_prefix}{col}"
            if col_name not in updated_df.columns:
                updated_df[col_name] = None

            if isinstance(value, list):
                for idx in updated_df[sample_mask].index:
                    updated_df.at[idx, col_name] = value
            else:
                updated_df.loc[sample_mask, col_name] = value

    return updated_df


def compute_interface_metrics_over_models(
    pdb_paths: list[str],
    n_af2_models: int = 1,
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    show_progress: bool = False,
) -> list[dict]:
    """Interface metrics per structure, averaged over the models a refold produced.

    Each entry of *pdb_paths* names one model -- ``{design}_model1.pdb`` -- and
    its siblings are derived from it. All-or-nothing per design: averaging three
    of five would report a number the design did not earn, and silently, since
    nothing downstream records how many models a value came from. A design whose
    siblings are absent, or a backend that does not produce them, falls back to
    the single structure it has.

    The reduction itself is :func:`mean_interface_metrics`, which lives with the
    other per-model reductions and is unit-testable without this module's stack.
    """
    from proteinfoundation.metrics.ensembling import mean_interface_metrics, per_model_paths_from_first

    expanded: list[list[str]] = []
    for path in pdb_paths:
        siblings = per_model_paths_from_first(path, n_af2_models) if n_af2_models > 1 else None
        expanded.append(siblings or [path])

    flat = [p for group in expanded for p in group]
    computed = compute_interface_metrics(
        pdb_paths=flat,
        compute_bioinformatics=compute_bioinformatics,
        compute_tmol=compute_tmol,
        show_progress=show_progress,
    )
    by_path = dict(zip(flat, computed, strict=True))

    return [
        {"pdb_path": path, **mean_interface_metrics([by_path[p] for p in group])}
        for path, group in zip(pdb_paths, expanded, strict=True)
    ]


def compute_interface_metrics_on_refolded_structures(
    df: pd.DataFrame,
    best_paths_dict: dict[str, dict[str, str]],
    cfg_metric: DictConfig,
    cfg: DictConfig,
    compute_bioinformatics: bool = False,
    compute_tmol: bool = False,
    show_progress: bool = False,
    n_af2_models: int = 1,
) -> pd.DataFrame:
    """
    Compute force field and bioinformatics metrics on successful refolded structures.

    Args:
        df: DataFrame with evaluation results.
        best_paths_dict: Dictionary of best refolded structure paths.
        cfg_metric: Metric configuration.
        cfg: Full configuration.
        compute_bioinformatics: Whether to compute bioinformatics metrics (SC, SASA, hydrophobicity).
        compute_tmol: Whether to compute TMOL force field metrics.
        show_progress: Whether to show progress bar (default: False).
        n_af2_models: How many models the complex refold produced. Above one, the
            interface metrics are computed on every model and averaged, the way
            the confidence and geometry families already are.

    Returns:
        DataFrame with added refolded structure metrics.
    """
    # Check if any metrics requested
    if not any([compute_bioinformatics, compute_tmol]):
        return df

    successful_samples = []
    sequence_types = cfg_metric.get("sequence_types", SEQUENCE_TYPES)
    for _, row in df.iterrows():
        # Extract sample name from pdb_path
        pdb_path = row["pdb_path"]
        sample_name = os.path.basename(pdb_path).replace(".pdb", "").replace("tmp_", "")
        if sample_name not in best_paths_dict:
            continue
        for seq_type in sequence_types:
            structure_path = best_paths_dict[sample_name].get(seq_type)
            if structure_path and os.path.exists(structure_path):
                successful_samples.append((sample_name, seq_type, structure_path))
                logger.debug(f"Found successful best sample: {sample_name} {seq_type}")
            else:
                logger.debug(f"Structure path not found for successful sample: {sample_name} {seq_type}")

    logger.info(f"Found {len(successful_samples)} successful best samples with refolded structures")
    if not successful_samples:
        logger.warning("No successful refolded structures found")
        return df

    # Log which metrics are being computed
    enabled_metrics = []
    if compute_bioinformatics:
        enabled_metrics.append("bioinformatics")
    if compute_tmol:
        enabled_metrics.append("TMOL")

    logger.info(f"Computing metrics [{', '.join(enabled_metrics)}] on {len(successful_samples)} refolded structures")

    # Group samples by sequence type for efficient processing
    samples_by_seq_type: dict[str, list[tuple[str, str]]] = {}
    for sample_name, seq_type, path in successful_samples:
        if seq_type not in samples_by_seq_type:
            samples_by_seq_type[seq_type] = []
        samples_by_seq_type[seq_type].append((sample_name, path))

    # The slots the columns below carry. Resolved here rather than passed, so a
    # caller cannot label these with a different model than the one that folded.
    complex_backend = backend_for_folding_method(cfg_metric.get("binder_folding_method", "colabdesign"))

    # Compute and merge metrics for each sequence type
    updated_df = df.copy()
    _, flat_dict = parse_cfg_for_table(cfg)
    skip_cols = {"pdb_path"} | set(flat_dict.keys())

    for seq_type, samples in samples_by_seq_type.items():
        sample_names = [s[0] for s in samples]
        structure_paths = [s[1] for s in samples]

        # Averaged over the models the refold produced, not read off one of them.
        # best_paths_dict names the model-1 structure, and computing an interface
        # from it alone reports one draw of five as though it were the design:
        # across five AF2 models of one design the interface itself moves, 13 to
        # 16 residues being typical. The confidence and geometry families already
        # reduce over models; this brings the interface family into line.
        metrics_list = compute_interface_metrics_over_models(
            pdb_paths=structure_paths,
            n_af2_models=n_af2_models,
            compute_bioinformatics=compute_bioinformatics,
            compute_tmol=compute_tmol,
            show_progress=show_progress,
        )

        # Slots, not a "refolded" prefix that never said which model refolded.
        prefix = f"{seq_type}_complex_{complex_backend}_"
        updated_df = merge_metrics_to_df(
            df=updated_df,
            metrics_list=metrics_list,
            sample_names=sample_names,
            column_prefix=prefix,
            skip_cols=skip_cols,
        )

    logger.info("Successfully merged refolded structure metrics")
    return updated_df
