"""The binder-evaluation refolding cache: its key, and what it refuses.

Separated from ``binder_eval`` so the cache contract can be tested without the
folding stack. Every acceptance rule here decides whether hours of GPU time are
spent or skipped, and a rule that can only be checked by reading it is a rule
that drifts -- the reader below has three distinct acceptance paths and had no
test able to reach any of them while it lived next to an ``atomworks`` import.
"""

import hashlib
import json
import math
import os
import re
from typing import Any

from loguru import logger

# One cache file per complex backend, for the reason
# metrics/consensus_folding.py gives for doing the same, and that
# evaluation/monomer_eval_utils.py gives again: a single shared file carries a
# single fingerprint, so the moment a second backend is named as primary it
# writes its own and the first one's entries are discarded on every design.
#
# That was not hypothetical here. The backend was inside the fingerprint rather
# than in the filename, which is what made the ORDER of metric.folding_models
# load-bearing: a pass configured [esmfold2] made ESMFold2 primary, recomputed a
# different fingerprint for the same path, and discarded every AF2 complex --
# measured at ~42 GPU-hours on CBLN1. The pass plan had to carry a guard
# refusing any list that did not start with af2, to protect a filename.
#
# The advisory side and the monomer side had both already made this migration.
# This is the third and last place the backend was hidden in a hash.
BINDER_EVAL_CACHE_TEMPLATE = "binder_eval_cache_{backend}.json"
LEGACY_BINDER_EVAL_CACHE_FILENAME = "binder_eval_cache.json"
# Kept as the old name for anything importing it; it is the legacy path.
BINDER_EVAL_CACHE_FILENAME = LEGACY_BINDER_EVAL_CACHE_FILENAME


def binder_eval_cache_path(sample_root_path: str, backend: str | None = None) -> str:
    """Where one complex backend's refolds for this design live.

    *backend* omitted gives the pre-split path, which is where a campaign that
    ran before this still has its results. Readers try the per-backend path
    first and fall back; writers only ever write the per-backend one.
    """
    if backend is None:
        return os.path.join(sample_root_path, LEGACY_BINDER_EVAL_CACHE_FILENAME)
    return os.path.join(sample_root_path, BINDER_EVAL_CACHE_TEMPLATE.format(backend=backend))


def binder_eval_cache_paths(sample_root_path: str, backend: str | None = None) -> list[str]:
    """The paths a reader should try, in order: this backend's, then the legacy
    shared one.

    The legacy file is only ever accepted when its fingerprint matches the
    request being made -- and the fingerprint contains the folding model, so it
    matches exactly when it was written by this same backend. A campaign that
    ran af2 before this keeps its results; one that switches primary to another
    folder reads the legacy file, finds a different fingerprint, and refolds
    into a file of its own rather than overwriting af2's.
    """
    if backend is None:
        return [binder_eval_cache_path(sample_root_path)]
    return [
        binder_eval_cache_path(sample_root_path, backend),
        binder_eval_cache_path(sample_root_path),
    ]


def binder_eval_fingerprint(**inputs: Any) -> str:
    """Digest of every input that determines a refolding result.

    A cached result is only reusable if it was produced by the same request.
    Switching ``binder_folding_method`` from ``colabdesign`` to an ``rf3`` model
    is the case that matters most: the numbers are not comparable, and without a
    fingerprint the cache would silently serve AF2 results for an RF3 run.
    """
    canonical = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def unmeasured_complex_metrics(sequence_type_stats: dict) -> list[str]:
    """Metrics in *sequence_type_stats* that hold no usable number.

    A NaN or an infinity here is not a measurement, it is the absence of one
    wearing a measurement's clothes -- the same confusion that let a declined
    monomer fold be cached as a value and answer for that sequence forever.

    This cache is stricter than the monomer one, which tolerates a partial entry
    because MAX_FOLD_ATTEMPTS will retry it. There is no attempt counter here and
    no per-sequence retry: whatever this file records is what every later resume
    reads. So a single unusable value is enough to refuse the write, and the
    design refolds next run instead.

    Only complex_stats and rmsd_stats are checked. aa_stats are composition
    counts, not folder output, and nothing about a fold makes them unusable.
    """
    bad: list[str] = []
    for seq_type, payload in (sequence_type_stats or {}).items():
        for group in ("complex_stats", "rmsd_stats"):
            entries = (payload or {}).get(group)
            if not isinstance(entries, list):
                continue
            for i, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                for key, value in entry.items():
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    if not math.isfinite(value):
                        bad.append(f"{seq_type}.{group}[{i}].{key}={value}")
    return bad


def write_binder_eval_cache(
    sample_root_path: str,
    fingerprint: str,
    sequence_type_stats: dict,
    sequences_dict: dict,
    derivation_fingerprint: str | None = None,
    backend: str | None = None,
) -> None:
    """Persist everything the row-building code needs from ``run_binder_eval``.

    Written alongside — not instead of — ``sequence_type_stats.json``, whose
    schema stays as it was so existing consumers are unaffected.
    """
    unusable = unmeasured_complex_metrics(sequence_type_stats)
    if unusable:
        # Refuse rather than persist. Today run_af_eval raises on any failure, so
        # nothing partial reaches here -- but the per-design try in binder_eval.py
        # exists to keep one bad design from killing a run, and the moment a
        # complex fold can fail per SEQUENCE rather than per design, this is the
        # path that would make that failure permanent. The guard is cheap and the
        # cost of being wrong is a campaign's worth of NaNs nobody can retry.
        logger.warning(
            f"Not caching binder eval for {sample_root_path}: "
            f"{len(unusable)} metric(s) hold no usable number "
            f"({', '.join(unusable[:4])}{', ...' if len(unusable) > 4 else ''}). "
            f"This design will refold on the next run rather than serve these forever."
        )
        return
    try:
        # Serialise first so a non-encodable payload leaves no half-written file
        # behind for the next run to trip over.
        blob = json.dumps(
            {
                "fingerprint": fingerprint,
                "derivation_fingerprint": derivation_fingerprint,
                "sequence_type_stats": sequence_type_stats,
                "sequences_dict": sequences_dict,
            }
        )
        with open(binder_eval_cache_path(sample_root_path, backend), "w") as handle:
            handle.write(blob)
    except (OSError, TypeError, ValueError) as exc:
        # A cache is an optimisation; failing to write one must not fail evaluation.
        logger.warning(f"Could not write binder eval cache for {sample_root_path}: {exc}")


def digest_file(path: str) -> str:
    """A file's contents as a fingerprint component, never its location.

    Lives here rather than beside either caller because both the refolding
    fingerprint and ``consensus_folding.cfg_for_fingerprint`` need it, and two
    copies of a hash function is two ways for the same cache to key differently.

    An unreadable file keeps the path in the value: whatever is about to fail
    will say so, and until then the key still changes if it is later pointed
    somewhere else.
    """
    try:
        with open(path, "rb") as handle:
            return "sha256:" + hashlib.sha256(handle.read()).hexdigest()[:32]
    except OSError:
        return f"unreadable:{path}"


def read_binder_eval_cache(
    sample_root_path: str,
    fingerprint: str,
    sequence_types: list[str],
    derivation_fingerprint: str | None = None,
    legacy_fingerprints: list[str] | None = None,
    backend: str | None = None,
) -> tuple[dict, dict, bool] | None:
    """Cached refolding results for this design, or None to recompute.

    Returns ``(stats, sequences, derivation_stale)``. None unless the cache exists,
    parses, was produced by the same *structure* request, and covers every
    requested sequence type — a cache built for ``["self"]`` must not be reused
    for a run asking for ``["self", "mpnn"]``.

    ``derivation_stale`` says the structures are the ones this run wants but the
    numbers read off them are not: the reduction or the interface cutoff changed.
    The caller can then recompute rather than refold. A cache written before the
    split carries no derivation fingerprint and is treated as stale, which is
    correct -- it cannot say which rule produced its numbers.

    ``legacy_fingerprints`` are structure fingerprints the caller has declared
    equivalent to the current one. A cache matching one of those is accepted and
    always reported stale: whatever made the fingerprint differ is by definition
    something this run has to re-derive.
    """
    # This backend's file first, then the pre-split shared one. The shared file
    # only survives the fingerprint check when this same backend wrote it, so a
    # campaign keeps its results and a different primary refolds into its own
    # file rather than overwriting them.
    cache_path = next(
        (p for p in binder_eval_cache_paths(sample_root_path, backend) if os.path.exists(p)),
        None,
    )
    if cache_path is None:
        return None
    try:
        with open(cache_path) as handle:
            cached = json.load(handle)
        stats = cached["sequence_type_stats"]
        sequences = cached["sequences_dict"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(f"Ignoring unusable binder eval cache {cache_path}: {exc}")
        return None
    stored = cached.get("fingerprint")
    accepted_as_legacy = stored != fingerprint and stored in set(legacy_fingerprints or [])
    if stored != fingerprint and not accepted_as_legacy:
        logger.info(
            f"Binder eval cache at {cache_path} was produced by a different request "
            f"({str(stored)[:12]} != {fingerprint[:12]}); recomputing"
        )
        return None
    missing = [t for t in sequence_types if t not in stats or t not in sequences]
    if missing:
        logger.info(f"Binder eval cache at {cache_path} lacks sequence types {missing}; recomputing")
        return None
    derivation_stale = accepted_as_legacy or cached.get("derivation_fingerprint") != derivation_fingerprint
    if accepted_as_legacy:
        logger.info(
            f"Binder eval cache at {cache_path} was written under a structure fingerprint this run "
            f"declared reusable ({str(stored)[:12]}); re-deriving its numbers and rewriting it under "
            f"{fingerprint[:12]}"
        )
    elif derivation_stale:
        logger.info(
            f"Binder eval cache at {cache_path} holds the structures this run wants but numbers "
            f"from a different derivation; recomputing them rather than refolding"
        )
    return stats, sequences, derivation_stale


# =============================================================================
# Bridge to the advisory fold cache
# =============================================================================
#
# The complex folds in this file were made by the mechanism that used to be
# called "primary": one folder, named by folders.complex[0], reached through
# run_binder_eval. Every OTHER complex folder is reached through
# consensus_folding.score_binders and caches into consensus_fold_cache_{backend}.
#
# Those are two cache shapes for one kind of result, and the second is the better
# one: keyed per (sequence, seed) rather than per design, carrying its own
# derivation fingerprint, and able to re-read a kept structure instead of
# refolding when only the numbers read off it changed. Unifying the two means
# every complex folder writes the second shape.
#
# A unification alone would make every finished campaign's AF2 complexes
# unreadable -- ~616 designs x 3 sequences x 5 models each, ~42 GPU-hours per
# campaign, to reproduce structures already on disk. This adopts them instead.


def _resolve_structure(stored: str | None, sample_root_path: str) -> str | None:
    """A structure this design recorded, found from where it is being read now.

    ``complex_pdb_path`` was written relative to the directory the run was
    launched from -- ``./evaluation_results/<campaign>/<design>/AF2/<file>.pdb``
    -- which resolves only from that one working directory. Anything reading the
    cache from anywhere else, this migration included, sees a path that does not
    exist and concludes the structure is gone.

    So the stored path is treated as a name rather than a location: its last two
    components are rejoined to the design directory, which is known. That carries
    no knowledge of which folder wrote it or what its subdirectory is called --
    only that a folder puts its structures in one place under the design.

    Returns None when nothing is found, which a caller reads as "adopt the fold,
    leave the metrics read off it to be filled in when the file reappears".
    """
    if not stored:
        return None
    if os.path.isabs(stored) and os.path.exists(stored):
        return stored
    parts = [p for p in stored.replace("\\", "/").split("/") if p not in ("", ".")]
    for tail in (parts[-2:], parts[-1:]):
        candidate = os.path.join(sample_root_path, *tail)
        if os.path.exists(candidate):
            return candidate
    return stored if os.path.exists(stored) else None


# Which keys in an adopted entry are the FOLDER'S reduction over its models
# rather than this draw's own measurement.
#
# The legacy file recorded one mean per sequence and threw the per-model
# confidence values away, so pLDDT, pTM and i_pTM cannot be recovered per model
# by any amount of re-reading -- the matrices on disk hold expected error in
# Angstroms, not the logits pTM comes from. The mean is still the right number
# for the sequence, and dropping it would put NaN in a gating column across every
# finished campaign, so it is carried on each draw and labelled as what it is.
# Everything not listed here was measured from that draw's own structure.
REDUCED_FROM_MODELS_KEY = "reduced_over_models"


# The confidence scalars a folder reports and no structure re-read reproduces.
# Everything else in an adopted entry is recomputed per draw.
_FOLDER_REDUCED_SUFFIXES = ("pLDDT", "target_pLDDT", "binder_pLDDT", "pTM", "i_pTM")


def _sibling_structure(stored: str | None, index: int, sample_root_path: str) -> str | None:
    """The *index*-th model's structure, given the path the file recorded.

    Only the first model's path was ever stored, and the others sit beside it
    under the name the folding harness gave them -- ``..._model1.pdb`` through
    ``..._model5.pdb``, constructed in colabdesign_utils. Substituting the model
    number is therefore reading the harness's own naming, not inventing one; a
    path that does not carry it resolves only for index 0, so nothing is guessed
    into existence.
    """
    if not stored:
        return None
    if index == 0:
        return _resolve_structure(stored, sample_root_path)
    swapped, n = re.subn(r"_model\d+\.pdb$", f"_model{index + 1}.pdb", stored)
    if not n:
        return None
    return _resolve_structure(swapped, sample_root_path)


def _has_stored_pae(structure_path: str | None) -> bool:
    """Whether this draw's own PAE matrix is on disk beside its structure.

    Folds made since the store exists write one; the pre-unification AF2 runs
    wrote them for some sequences and not others. Adoption asks because what it
    may safely discard is exactly what a re-read can reproduce.
    """
    if not structure_path:
        return False
    from proteinfoundation.metrics.pae_store import pae_sidecar_path

    return os.path.exists(pae_sidecar_path(structure_path))


def consensus_entries_from_complex_stats(
    complex_stats: list[dict],
    sequences: list[str],
    draws_for,
    sample_root_path: str,
    metric_suffixes,
) -> dict[str, dict[str, dict]]:
    """``{binder_seq: {seed: metrics}}`` for folds recorded in this file's shape.

    One entry per DRAW, where AF2's draws are its parameter sets. The legacy file
    holds one mean per sequence and names only the first model's structure, but
    every model's structure and PAE matrix is still on disk beside it -- so the
    PAE family, and everything else read off a structure, is adopted per model
    exactly rather than reconstructed from one model standing in for five.

    *sequences* must be the row's own pairing of sequence to stats entry --
    ``binder_eval.sequences_for_type``, which reads it off ``aa_stats`` rather
    than assuming two lists appended separately stayed parallel. Passed in rather
    than derived here so that rule keeps living in one place.

    Only what the FOLDER reported is carried, filtered to *metric_suffixes* and
    to values that are actually numbers. Everything read off the structure --
    buried area, shape complementarity, secondary structure, geometry against the
    design -- is deliberately left out even where this file happens to hold it:
    ``score_binders`` re-reads the kept PDB for anything absent, by the same code
    on every folder's structures, which is cheaper than a second mapping between
    two sets of names that could disagree.

    The ipSAE cutoffs are the exception, and they are recorded -- for a cutoff
    the entry actually evidences, meaning all three of its min_/max_/avg_ keys
    are present. The suffix IS the distance by construction (``ipsae_suffixes``
    builds the names from the cutoff list), so this records a fact rather than
    guessing at history. A cutoff the entry cannot evidence stays missing, which
    is what lets a later round add a distance for the cost of a re-read.

    That is the entire reason, and it is worth saying what is NOT a reason,
    because the first version of this said it was. Leaving the cutoffs absent
    makes ``_derive_into_scores`` recompute the PAE family -- from the ONE
    structure the entry names, while the numbers beside it are the folder's mean
    over five models. Over 120 EFNB3 complexes that moved avg_ipSAE by 5% on
    average and 0.55 at worst, on a metric that runs 0 to 1 and gets thresholded.

    The measurement is real; the inference from it was not. A question about
    DISTANCE was deciding ENSEMBLE SIZE, which is a defect in the derivation and
    has nothing to do with whether a cutoff is recorded. Recording the truth here
    happens to mask it, and masking is not fixing: the same collapse still meets
    any entry whose cutoffs genuinely cannot be evidenced. It is fixed where it
    lives, by giving each model its own entry and its own structure, so that what
    is read off a structure is read off the model it belongs to.

    ``pLDDT`` -- the whole-complex mean -- is absent from files written before it
    was emitted, and no structure re-read produces it, so those entries carry
    target_ and binder_pLDDT and not the complex mean. That is one advisory
    column reading NaN on adopted designs, against a campaign-scale refold.
    """
    from proteinfoundation.metrics.consensus_folding import IPSAE_CUTOFFS, PAE_CUTOFF_KEY

    wanted = set(metric_suffixes)
    entries: dict[str, dict[int, dict]] = {}
    for i, stats in enumerate(complex_stats or []):
        if not isinstance(stats, dict) or i >= len(sequences):
            continue
        seq = sequences[i]
        if not seq:
            continue
        metrics: dict = {
            k: float(v)
            for k, v in stats.items()
            if k in wanted and not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
        }
        if not metrics:
            # A design whose fold failed: inf and NaN are the absence of a
            # measurement, and adopting one would retire the refold that would
            # replace it. Left out, the sequence simply has no entry and folds.
            continue
        evidenced = {
            suffix: float(cutoff)
            for cutoff, suffix in IPSAE_CUTOFFS
            if all(f"{kind}ipSAE{suffix}" in metrics for kind in ("min_", "max_", "avg_"))
        }
        if evidenced:
            metrics[PAE_CUTOFF_KEY] = evidenced
        pdb = _resolve_structure(stats.get("complex_pdb_path"), sample_root_path)
        if pdb:
            metrics["pdb_path"] = pdb
        draws = draws_for(seq)
        if not draws:
            continue
        stored = stats.get("complex_pdb_path")
        for index, draw in enumerate(draws):
            # This draw's OWN structure. The legacy file names only the first,
            # and the rest sit beside it under the name the folder gave them, so
            # the first is asked to produce its siblings; a draw whose structure
            # cannot be found is skipped rather than pointed at another model's.
            own = _sibling_structure(stored, index, sample_root_path)
            if own is None and index > 0:
                continue
            entry = dict(metrics)
            if own:
                entry["pdb_path"] = own
            elif "pdb_path" in entry:
                del entry["pdb_path"]
            # Say which of these numbers are the folder's mean over its models
            # rather than this draw's own. Only the confidence scalars are: the
            # PAE family and everything read off a structure get recomputed from
            # this draw's own files, which is the whole point of adopting per
            # model instead of adopting one entry five times.
            reduced = sorted(k for k in entry if k in _FOLDER_REDUCED_SUFFIXES)
            if reduced and len(draws) > 1:
                entry[REDUCED_FROM_MODELS_KEY] = reduced
                # The PAE family is dropped so the per-draw recomputation that
                # the stored matrices CAN answer exactly is not suppressed -- and
                # CAN is the load-bearing word. Dropping it unconditionally was
                # wrong: the pre-unification AF2 runs kept a matrix for only some
                # of their sequences, so on EFNB3 this deleted the folder's i_pAE
                # and then nothing could recompute it. Every adopted design read
                # NaN in the one column the campaign gates on, and the adoption
                # lost a number the file it adopted from actually held.
                #
                # So it is kept when no matrix can replace it -- but only on the
                # draw the legacy file named. Unlike the confidence scalars, the
                # PAE family there was never a mean over the models: it was read
                # off the ONE structure the entry pointed at, which is this index.
                # Copying it onto the siblings would invent four measurements;
                # dropping it everywhere threw away the one that exists. Draws
                # with no evidence for it simply carry no entry for it, which is
                # what reduce_draws already reads correctly.
                keep_pae = not _has_stored_pae(entry.get("pdb_path")) and index == 0
                if not keep_pae:
                    entry.pop(PAE_CUTOFF_KEY, None)
                    for key in list(entry):
                        if "ipSAE" in key or key in ("i_pAE", "pAE", "min_ipAE"):
                            del entry[key]
            entries.setdefault(seq, {})[str(draw)] = entry
    return entries


# Where each folder's harness puts its structures, which is how a pre-unification
# cache says who wrote it. The shared file records no folder name -- the folding
# model lives inside its fingerprint, as a hash -- but every entry records a
# structure path, and the harness that produced it chose the directory.
_STRUCTURE_DIR_BACKENDS = {"AF2": "af2", "rf3_outputs": "rf3"}


def legacy_complex_folds(sample_root_path: str, backend: str) -> tuple[dict, dict] | None:
    """This design's pre-unification complex folds, if *backend* is what made them.

    Read directly rather than through :func:`read_binder_eval_cache`, and this is
    the whole point of the function. That reader answers "are this file's
    SEQUENCES the ones this run wants", and its fingerprint carries the interface
    cutoff, the redesign count, the folding method -- so a campaign that has moved
    any of them since is told no. On EFNB3 it said no to every design, the
    adoption never fired, and the run refolded 15 AF2 complexes per design that
    were sitting on disk.

    The folds are not the sequences. A prediction of a named binder against a
    named target stays that prediction whatever the assembly request has since
    become, and whether it is reusable is the CONSENSUS fingerprint's question,
    asked when the adopted cache is read back. So this checks only the two things
    that would make an adoption wrong: that the file holds complex folds at all,
    and that this backend is the folder that made them.

    Who made them is read off where the structures went, because the shared file
    records the folding model only inside a hash. A design whose structures sit
    somewhere unrecognised is skipped rather than guessed at.
    """
    for path in binder_eval_cache_paths(sample_root_path, backend):
        if not os.path.exists(path):
            continue
        try:
            with open(path) as handle:
                cached = json.load(handle)
            stats = cached["sequence_type_stats"]
            sequences = cached["sequences_dict"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(f"Ignoring unusable binder eval cache {path}: {exc}")
            continue
        entries = [
            entry
            for payload in (stats or {}).values()
            for entry in ((payload or {}).get("complex_stats") or [])
            if isinstance(entry, dict)
        ]
        if not entries:
            continue
        wrote_it = {
            _STRUCTURE_DIR_BACKENDS.get(os.path.basename(os.path.dirname(str(e.get("complex_pdb_path") or ""))))
            for e in entries
        }
        if wrote_it != {backend}:
            logger.info(
                f"Not adopting {path} for '{backend}': its structures were written by "
                f"{sorted(w for w in wrote_it if w) or 'an unrecognised folder'}"
            )
            continue
        return stats, sequences
    return None


def adopt_binder_eval_folds(
    sample_root_path: str,
    backend: str,
    target_seqs: list[str],
    consensus_cfg: dict,
    sequence_type_stats: dict,
    sequences_by_type: dict[str, list[str]],
    derive_tmol: bool = False,
) -> int:
    """Migrate this design's complex folds into the advisory cache. Returns how many.

    A no-op -- and cheap -- once there is an advisory cache for *backend* on this
    design, which is what makes it safe to call on every design of every run
    rather than as a one-off script somebody has to remember to run against each
    campaign. An existing cache is authoritative and is never overwritten: it was
    written by the folder itself, and this only ever reconstructs.

    *sequences_by_type* maps each sequence type to the sequences its
    ``complex_stats`` describe, in stats order -- see
    ``binder_eval.sequences_for_type``. Returns the number of (sequence, draw)
    entries written, which for AF2 is one per parameter set.

    Never raises. Failing to adopt costs a refold, which is expensive; failing
    the evaluation costs the run.
    """
    from proteinfoundation.metrics.consensus_folding import (
        CONSENSUS_METRIC_SUFFIXES,
        consensus_cache_path,
        consensus_derivation_fingerprint,
        consensus_derived_suffixes,
        consensus_fingerprint,
        draw_ids_for,
        write_consensus_cache,
    )

    if os.path.exists(consensus_cache_path(sample_root_path, backend)):
        return 0
    try:
        entries: dict[str, dict[str, dict]] = {}
        for seq_type, payload in (sequence_type_stats or {}).items():
            adopted = consensus_entries_from_complex_stats(
                (payload or {}).get("complex_stats") or [],
                sequences_by_type.get(seq_type) or [],
                lambda seq: draw_ids_for(backend, consensus_cfg, target_seqs, seq),
                sample_root_path,
                CONSENSUS_METRIC_SUFFIXES,
            )
            for seq, by_seed in adopted.items():
                entries.setdefault(seq, {}).update(by_seed)
        if not entries:
            return 0
        # Stamped with the derivation this run wants even though nothing derived
        # was adopted. The alternative -- a mismatched stamp -- reads as "these
        # numbers were read off under a different rule", which would re-derive
        # every entry unconditionally on every later run. Absent keys are the
        # right signal instead: score_binders fills in what an entry LACKS from
        # the kept structure, which is the same path that heals an entry cached
        # before its PDB existed.
        derivation = (
            consensus_derivation_fingerprint(derive_tmol) if consensus_derived_suffixes(derive_tmol) else None
        )
        write_consensus_cache(
            sample_root_path,
            backend,
            consensus_fingerprint(backend, consensus_cfg, target_seqs),
            entries,
            derivation=derivation,
        )
        return sum(len(v) for v in entries.values())
    except Exception as exc:
        logger.warning(f"Could not adopt complex folds for {sample_root_path} into the '{backend}' cache: {exc}")
        return 0
