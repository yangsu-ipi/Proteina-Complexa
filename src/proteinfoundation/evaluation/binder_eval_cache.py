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
from typing import Any

from loguru import logger

BINDER_EVAL_CACHE_FILENAME = "binder_eval_cache.json"


def _binder_cache_path(sample_root_path: str) -> str:
    return os.path.join(sample_root_path, BINDER_EVAL_CACHE_FILENAME)


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
        with open(_binder_cache_path(sample_root_path), "w") as handle:
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
    cache_path = _binder_cache_path(sample_root_path)
    if not os.path.exists(cache_path):
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
