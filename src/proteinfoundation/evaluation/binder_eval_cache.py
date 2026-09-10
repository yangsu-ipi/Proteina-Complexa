"""The binder-evaluation refolding cache: its key, and what it refuses.

Separated from ``binder_eval`` so the cache contract can be tested without the
folding stack. Every acceptance rule here decides whether hours of GPU time are
spent or skipped, and a rule that can only be checked by reading it is a rule
that drifts -- the reader below has three distinct acceptance paths and had no
test able to reach any of them while it lived next to an ``atomworks`` import.
"""

import hashlib
import json
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
