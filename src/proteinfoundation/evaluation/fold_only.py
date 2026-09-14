"""What a fold-only pass turns off, and what it refuses to write.

Separated from ``evaluate`` for the reason ``binder_eval_cache`` is: the rules
here decide whether a campaign's results CSV is the run's output or a quarter of
it, and a rule that can only be checked by reading it is a rule that drifts.
``evaluate`` imports torch, JAX and the folding stack at module scope, so a test
that exercised these two functions there would need a GPU box to answer a
question about a dict and a DataFrame.

A fold-only pass exists so that one folder has the GPU to itself. The campaign
runner (``evaluate_split`` in ``run_campaign.sh``) runs evaluate once per folder
with ``metric.fold_only=true`` and a one-element ``metric.folding_models``, then
once more with everything, which finds every fold cached. The passes need no
cache of their own: folds were already stored per backend, and no fold
fingerprint carries the folder LIST.
"""

from __future__ import annotations

import os

from loguru import logger

from proteinfoundation.result_analysis.analysis_utils import filter_columns_for_csv

# What a fold-only pass turns off. Exactly one entry, on purpose.
#
# The pass exists so one folder has the card to itself, so the only thing worth
# turning off is whatever else would occupy it. ESMC-6B is the only one evaluate
# has. Everything else it computes -- the pre-refolding and refolded-structure
# metrics -- is in-memory CPU that writes nothing, so a pass that redoes it
# wastes time and changes no state.
#
# Splitting evaluate by folder does not touch any fingerprint. Not the fold
# fingerprints, which hash a one-element folder list either way, and not the
# consensus DERIVATION fingerprint, which hashes the list of suffixes read off a
# structure and a version constant -- neither carries the folder list, the pass
# structure, or the order folders run in. That is why the split needs no new
# cache.
#
# The one way a pass COULD move the derivation fingerprint is by asking for less
# than the final pass: adding compute_refolded_structure_metrics here would flip
# consensus include_tmol for a campaign with refolded.tmol true, and the final
# pass would then re-read every kept structure (a re-read, never a refold --
# see _derive_into_scores). That is a reason not to extend this tuple casually,
# not a consequence of the split. It is also inert in the binder campaigns,
# where refolded.tmol is false and include_tmol is already False either way.
FOLD_ONLY_DISABLED_METRICS: tuple[str, ...] = ("compute_esm_metrics",)


def apply_fold_only(cfg_metric) -> list[str]:
    """Strip a pass down to folding, and report what that turned off.

    Mutates *cfg_metric* rather than returning a copy: everything downstream
    reads ``cfg.metric`` through its own reference, and a second config object
    would mean two answers to "is ESM on?" inside one process.

    Only keys already present and truthy are touched, which keeps this safe
    under OmegaConf's struct mode and makes the returned list a record of what
    actually changed rather than of what was asked for.
    """
    turned_off = []
    for key in FOLD_ONLY_DISABLED_METRICS:
        if cfg_metric.get(key, False):
            cfg_metric[key] = False
            turned_off.append(key)
    return turned_off


def writes_run_level_output(fold_only: bool, what: str) -> bool:
    """Whether this pass may write *what* into the run's output directory.

    A run-level artifact is written once per evaluate PROCESS rather than once
    per design: the results CSVs, the success-criteria JSON, the timing row.
    Under one fused evaluate that distinction did not exist, because there was
    one process and it knew about every folder. Under split passes there are
    five, four of which know about a subset -- so each of those would overwrite
    the run's record with a fraction of it, and the last writer would win.

    Design-level artifacts are deliberately NOT gated here. Fold caches, PAE
    matrices, kept structures and ``sequence_type_stats.json`` are written per
    design under the folder that produced them, and a pass folding af2 writes
    exactly what a fused run's af2 half wrote. The same is true of the derived
    interface and structure metrics, which measure one folder's predictions and
    return columns without writing anything at all.
    """
    if fold_only:
        logger.info(f"Fold-only pass: not writing {what}; that is the final pass's to write")
        return False
    return True


def save_results_csv(df, output_dir: str, kind: str, config_name: str, job_id, *, fold_only: bool):
    """Write one evaluation's rows, unless this is a fold-only pass.

    A fold-only pass leaves no CSV at all rather than a partial one. It runs with
    a subset of ``metric.folding_models``, so its rows carry a subset of the
    columns -- and nothing that reads this directory afterwards (``analyze``,
    ``verify_run_outputs``, ``analyze_pooled``) can tell a one-folder CSV from a
    finished one. ``verify_run_outputs`` would report missing columns for a run
    that is going to have them twenty minutes later, and ``analyze`` would build
    verdicts from whichever folder happened to run last.

    The rows are still built, because building them is what proves the caches are
    readable -- a pass that folded and never read its own cache back would defer
    every fingerprint mismatch to the final pass, which is the expensive place to
    find one.

    Returns the filtered frame either way, so the caller's sample count and
    summary do not have to know which pass this was.
    """
    filtered = filter_columns_for_csv(df)
    if not writes_run_level_output(fold_only, f"the {kind} CSV ({len(filtered)} row(s) built and discarded)"):
        return filtered
    csv_path = os.path.join(output_dir, f"{kind}_results_{config_name}_{job_id}.csv")
    filtered.to_csv(csv_path, index=False)
    logger.info(f"{kind.replace('_', ' ').capitalize()} results saved to {csv_path}")
    return filtered
