"""Which work each evaluate pass does, and what it is allowed to write.

Separated from ``evaluate`` for the reason ``binder_eval_cache`` is: the rules
here decide whether a campaign's results CSV is the run's output or a fraction
of it, and a rule that can only be checked by reading it is a rule that drifts.
``evaluate`` imports torch, JAX and the folding stack at module scope, so a test
that exercised these functions there would need a GPU box to answer a question
about a dict and a DataFrame.

The campaign runner (``evaluate_split`` in ``run_campaign.sh``) runs evaluate six
times per shard so that no pass has two models on its card at once:

    1-4   fold     one folder each -- af2 monomer, esmfold2 monomer,
                   af2 complex, esmfold2 complex
    5     esm      ESMC-6B, whose per-design results are cached like a fold's
    6     final    reads every cache, derives the structure metrics, writes

Splitting by folder needs no new cache and no new fingerprint. Folds were
already stored per backend, and no fold fingerprint carries the folder LIST.

WHY THE DERIVED METRICS ARE NOT IN A FOLD PASS, and why ESM is on its own:

* The pre-refolding metrics measure the GENERATED structure -- ``{sample}.pdb``,
  straight out of generation. No folder influences them and the answer is
  identical in every pass, so they belong in exactly one.

* The refolded-structure metrics are read off each folder's own structures by
  ``score_binders`` now, into that folder's own cache, so they are computed in
  the pass that folded them and re-read rather than recomputed afterwards. The
  separate pass that used to compute them for one folder is retired: it wrote
  the same columns and ran later, so it overwrote a per-draw reduction with an
  unconditional mean.

* Both of those can run TMOL, and ``TmolRewardModel`` defaults to
  ``torch.device("cuda")`` -- it is a force field on the GPU, not a CPU metric.
  Since it cannot be given its own pass (no cache to put its answer in), the
  next best thing is to make sure it is ALONE in the pass it does run in. That
  is what pass 5 is for: ESM is cached per design, so lifting it out leaves the
  final pass with TMOL as its only tenant.

The consequence for a campaign with tmol enabled: the fold passes write the
consensus cache with ``include_tmol=False``, and the final pass reads it under a
different derivation fingerprint and re-derives from the kept structures. That
is a re-read, never a refold, and it happens in the pass that wants the numbers.
Campaigns with tmol off -- which is all the binder campaigns -- see no
fingerprint movement at all, and the final pass skips the derivation entirely.
"""

from __future__ import annotations

import os

from loguru import logger

from proteinfoundation.result_analysis.analysis_utils import filter_columns_for_csv

PASS_REDESIGN = "redesign"
PASS_FOLD = "fold"
PASS_ESM = "esm"
PASS_FINAL = "final"
PASS_KINDS: tuple[str, ...] = (PASS_REDESIGN, PASS_FOLD, PASS_ESM, PASS_FINAL)

EVALUATE_PASS_KEY = "evaluate_pass"

# Metric flags each pass forces off. Only these three are ever touched: every
# other flag decides what gets FOLDED, which is the pass plan's business and not
# this table's.
#
# The tmol sub-flags need no entry of their own. derive_consensus_tmol is
# `compute_refolded_structure_metrics and refolded.tmol`, and the other TMOL site
# is inside the pre-refolding metrics -- so suppressing those two keys suppresses
# every route to the force field.
SUPPRESSED_BY_PASS: dict[str, tuple[str, ...]] = {
    # The redesign pass runs the inverse folder and nothing else, so it suppresses
    # what a fold pass does and folds nothing on top.
    PASS_REDESIGN: (
        "compute_esm_metrics",
        "compute_pre_refolding_metrics",
        "compute_refolded_structure_metrics",
    ),
    PASS_FOLD: (
        "compute_esm_metrics",
        "compute_pre_refolding_metrics",
        "compute_refolded_structure_metrics",
    ),
    PASS_ESM: (
        "compute_pre_refolding_metrics",
        "compute_refolded_structure_metrics",
    ),
    PASS_FINAL: (),
}


class UnknownEvaluatePass(ValueError):
    """A pass name nothing knows how to run."""


def resolve_pass(cfg_metric) -> str:
    """Which pass this process is, from ``metric.evaluate_pass``.

    Refuses an unknown name rather than falling back to ``final``. A typo that
    silently became the writing pass would put a one-folder CSV in the output
    directory under the name the finished run uses.
    """
    kind = str((cfg_metric or {}).get(EVALUATE_PASS_KEY, PASS_FINAL) or PASS_FINAL)
    if kind not in PASS_KINDS:
        raise UnknownEvaluatePass(
            f"metric.{EVALUATE_PASS_KEY}={kind!r} is not one of {list(PASS_KINDS)}. "
            f"'{PASS_FOLD}' folds one backend, '{PASS_ESM}' runs ESM alone, "
            f"'{PASS_FINAL}' derives and writes."
        )
    return kind


def apply_pass(cfg_metric, kind: str) -> list[str]:
    """Turn off what this pass has no use for, and report what changed.

    Mutates *cfg_metric* rather than returning a copy: everything downstream
    reads ``cfg.metric`` through its own reference, and a second config object
    would mean two answers to "is ESM on?" inside one process.

    Only keys already present and truthy are touched, which keeps this safe
    under OmegaConf's struct mode and makes the returned list a record of what
    actually changed rather than of what was asked for.
    """
    turned_off = []
    for key in SUPPRESSED_BY_PASS[kind]:
        if cfg_metric.get(key, False):
            cfg_metric[key] = False
            turned_off.append(key)
    return turned_off


def writes_run_level_output(kind: str, what: str) -> bool:
    """Whether this pass may write *what* into the run's output directory.

    A run-level artifact is written once per evaluate PROCESS rather than once
    per design: the results CSVs, the success-criteria JSON, the timing row.
    With one fused evaluate that distinction did not exist, because there was one
    process and it knew about every folder. With six, five of them know about a
    subset -- so each of those would overwrite the run's record with a fraction
    of it, and the last writer would win.

    Design-level artifacts are deliberately NOT gated here. Fold caches, PAE
    matrices, kept structures, the ESM cache and ``sequence_type_stats.json`` are
    written per design under the folder that produced them, and a pass folding
    af2 writes exactly what a fused run's af2 half wrote.
    """
    if kind != PASS_FINAL:
        logger.info(f"{kind} pass: not writing {what}; that is the final pass's to write")
        return False
    return True


def save_results_csv(df, output_dir: str, track: str, config_name: str, job_id, *, evaluate_pass: str):
    """Write one evaluation's rows, unless this pass is not the one that writes.

    A non-final pass leaves no CSV at all rather than a partial one. It runs with
    a subset of ``metric.folding_models`` and with the derived metrics off, so
    its rows carry a subset of the columns -- and nothing that reads this
    directory afterwards (``analyze``, ``verify_run_outputs``, ``analyze_pooled``)
    can tell a one-folder CSV from a finished one.

    The rows are still built, because building them is what proves the caches are
    readable -- a pass that folded and never read its own cache back would defer
    every fingerprint mismatch to the final pass, which is the expensive place to
    find one.

    Returns the filtered frame either way, so the caller's sample count and
    summary do not have to know which pass this was.
    """
    filtered = filter_columns_for_csv(df)
    if not writes_run_level_output(evaluate_pass, f"the {track} CSV ({len(filtered)} row(s) built and discarded)"):
        return filtered
    csv_path = os.path.join(output_dir, f"{track}_results_{config_name}_{job_id}.csv")
    filtered.to_csv(csv_path, index=False)
    logger.info(f"{track.replace('_', ' ').capitalize()} results saved to {csv_path}")
    return filtered


# =============================================================================
# The pass plan
# =============================================================================


def evaluate_pass_plan(folding_models) -> list[dict]:
    """The passes a campaign's folder list implies, in the order they must run.

    Derived rather than written down. The plan used to name af2 and esmfold2
    literally, which was wrong for every other list the config accepts: a
    campaign naming rf3 (complex-only) got no pass for it and folded it in the
    final pass beside ESMC-6B and TMOL, and one naming esmfold (monomer-only)
    got a pass for a folder it had not configured. Both silently.

    It is also what decouples the inverse folder from the folders. ProteinMPNN
    used to ride in "pass 1", defined as af2-monomer -- so the redesign sets were
    owned by a folder the campaign might not have configured, and adding a folder
    moved their owner. The redesign pass owns them instead, and depends on no
    folder at all.

    Every complex folder now gets a pass of its own, naming only itself. It used
    to name the first folder alongside each of the others, because
    ``run_binder_eval`` built the complex through ColabDesign or RF3 and raised
    for anything else, while every other folder was reached through
    ``score_binders``: a second folder could only be asked for with the first
    one present, so a pass meant to fold ESMFold2 loaded AF2 too. One mechanism
    reaches all of them now, so a pass costs one folder's VRAM instead of two,
    and no folder is a precondition for another.
    """
    from proteinfoundation.metrics.column_names import folders_for_track

    monomer = folders_for_track(folding_models, "monomer")
    complex_ = folders_for_track(folding_models, "complex")

    # The redesign pass runs the inverse folder and NOTHING else, which means it
    # has to say so on both tracks. With no track it inherited the campaign's own
    # flags, so the binder loop ran in full beside it -- measured on EFNB3's
    # first split run at 73.5 GB per card of AF2 weights, in the pass whose whole
    # point is that no folder is resident. The folds were not wrong, just paid
    # for in the pass built to avoid paying for them.
    #
    # "monomer" rather than a track of its own because that is where this pass
    # does its work: the short-circuit lives in the monomer loop, and the set it
    # writes is shared with the binder track (shared_redesign_set), so generating
    # it once on either side serves both.
    passes: list[dict] = [{"kind": PASS_REDESIGN, "models": None, "track": "monomer"}]
    passes += [{"kind": PASS_FOLD, "models": [m], "track": "monomer"} for m in monomer]
    passes += [{"kind": PASS_FOLD, "models": [m], "track": "complex"} for m in complex_]
    passes.append({"kind": PASS_ESM, "models": None, "track": "complex"})
    passes.append({"kind": PASS_FINAL, "models": None, "track": None})
    return passes


# Folders whose weights live in JAX. Everything else folds in torch.
#
# This matters because JAX PREALLOCATES a fraction of the card when it is
# imported, and evaluate imports it unconditionally -- so the fraction is
# claimed in every pass, including the ones that fold with torch and never call
# it. XLA_SOLE_TENANT=0.9 was written on the premise that a pass has one folder
# and may therefore take the card. True, and backwards for a torch folder: the
# ESMFold2 complex pass gave JAX 73.5 GB of an 80 GB card, left ESMFold2 with
# about 8, and every one of its folds failed. The fused run that worked gave JAX
# 0.3 and left torch 57 GB.
#
# So the fraction follows the pass's folder rather than the split's premise.
_JAX_BACKED = ("af2",)

# What JAX gets in a pass that does not fold with it. Not zero: it is imported,
# so it will claim something, and a hard floor is better than whatever it decides
# to grow to beside a torch model that wants the rest.
XLA_IDLE_FRACTION = "0.05"


def pass_folds_with_jax(step: dict) -> bool:
    """Whether this pass's folder is the one that preallocates the card."""
    from proteinfoundation.metrics.column_names import folder_family

    return any(folder_family(m) in _JAX_BACKED for m in (step.get("models") or ()))


def xla_fraction_for(step: dict, sole_tenant: str) -> str:
    """How much of the card JAX may take in this pass.

    *sole_tenant* when the pass folds with JAX, because then it is the tenant and
    the split exists so it can have the card. The idle floor otherwise -- a pass
    folding in torch, or folding nothing at all, still imports JAX and still has
    to leave the card to whatever does the work.
    """
    return sole_tenant if pass_folds_with_jax(step) else XLA_IDLE_FRACTION


def pass_overrides(step: dict) -> list[str]:
    """One pass's Hydra overrides, as the campaign runner passes them."""
    out = [f"++metric.{EVALUATE_PASS_KEY}={step['kind']}"] if step["kind"] != PASS_FINAL else []
    if step["models"]:
        out.append("++metric.folding_models=[" + ",".join(step["models"]) + "]")
    if step["track"] == "monomer":
        out += ["++metric.compute_monomer_metrics=true", "++metric.compute_binder_metrics=false"]
    elif step["track"] == "complex":
        out += ["++metric.compute_monomer_metrics=false", "++metric.compute_binder_metrics=true"]
    return out
