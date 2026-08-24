"""Advisory second-opinion refolding. Emits metrics; gates nothing.

Complexa's binder gates come from one folding backend -- ColabDesign (AF2) or
RF3 -- selected by ``metric.binder_folding_method``. When generation also uses
an AF2 reward, designs are chosen with AF2 and then graded with AF2. A second
model folding the same complexes does not remove that circularity, but it makes
it visible: a design the primary backend likes and a second model cannot fold at
all is worth a look.

Everything here is deliberately advisory. The columns are named
``{seq_type}_{backend}_{metric}``, which cannot collide with the gated
``{seq_type}_complex_{metric}_all`` that
``binder_analysis_utils.build_column_name`` produces, and no threshold in
``DEFAULT_PROTEIN_BINDER_THRESHOLDS`` / ``DEFAULT_LIGAND_BINDER_THRESHOLDS``
refers to a backend prefix. ``assert_columns_are_advisory`` enforces that rather
than trusting it.

Why non-gating is not a temporary stage. Absolute confidence cutoffs do not
transfer between folding models. ESMFold2 in particular "runs on a compressed
scale" -- a native DKK1 folds to only ~0.65 pLDDT -- so applying AF2-tuned
filters (i_pTM>=0.5, i_pAE<0.35, i_pLDDT>=80) "rejects almost everything and is
NOT comparable" (esmfold2 deploy/useful_binders.py). Turning any of this into a
gate requires re-deriving thresholds against designs of known outcome, and until
that exists these columns are for looking at, not filtering on.

Adding a backend
----------------
A backend is a callable::

    (target_seqs: list[str], binder_seq: str, cfg: dict) -> dict[str, float]

keyed by the suffixes in :data:`CONSENSUS_METRIC_SUFFIXES`, with missing metrics
simply absent. Register it in :data:`CONSENSUS_BACKENDS`. Imports must be lazy
so an uninstalled backend costs nothing, and failures must raise -- the caller
converts them to NaN so one bad design never fails a campaign.

For OpenDDE (github.com/aurekaresearch/OpenDDE), the mapping is known but the
adapter is not written: its summary confidence JSON carries ``plddt``, ``ptm``,
``iptm`` and ``ranking_score`` directly, and ``i_pAE`` is derivable from
``full_data["token_pair_pae"]`` (requires ``--need_atom_confidence true``) using
the same cross-chain block reduction as ESMFold2's ``pae_interaction``. Two
cautions recorded while surveying it: it declares itself a preview whose
predictions are "not guaranteed to be reproducible across releases", and its
exact dependency pins (torch==2.7.1, numpy==2.4.1) will not co-install with this
environment, so it wants a subprocess adapter rather than an in-process one.
``pb_ranking_score`` -- per-chain, interface-specific -- is the better selector
for binder work than the global ``ranking_score``.
"""

import hashlib
import json
import os
from collections.abc import Callable

import numpy as np
from loguru import logger

# Metrics a backend may report. Named to mirror the primary backend's metrics so
# a column-to-column comparison reads naturally, without reusing its prefix.
CONSENSUS_METRIC_SUFFIXES = ("i_pAE", "i_pTM", "pTM", "pLDDT")

# One cache file per backend. A single shared file would thrash the moment two
# backends are enabled together: each writes its own fingerprint, and the other's
# entries are discarded on every design.
CONSENSUS_CACHE_TEMPLATE = "consensus_fold_cache_{backend}.json"


def consensus_cache_path(cache_dir: str, backend: str) -> str:
    return os.path.join(cache_dir, CONSENSUS_CACHE_TEMPLATE.format(backend=backend))


# Prefix that would make a column look gated. Reserved.
_GATED_PREFIX = "complex"


# =============================================================================
# Backends
# =============================================================================


def _score_esmfold2(target_seqs: list[str], binder_seq: str, cfg: dict) -> dict[str, float]:
    """Fold target+binder with ESMFold2 and reduce to interface metrics.

    Uses the same input shape as the fork's own reference adapter
    (``oracle/backends/local_esmfold2.py``): one ProteinInput per chain, target
    chains first so the binder is the last asym id, and ``msa=None``. Note the
    fork's CLI calls a target MSA the "validated production path"; this runs
    single-sequence like that reference adapter, which is a difference worth
    remembering when comparing against published ESMFold2 numbers.

    Untested against real weights: they live in a private repo and were not
    available where this was written. Treat the first real run as the test.
    """
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput
    from esm.models.esmfold2.interface_metrics import pae_interaction

    model = _esmfold2_model(cfg)
    chains = [ProteinInput(id=f"T{i}", sequence=s, msa=None) for i, s in enumerate(target_seqs)]
    chains.append(ProteinInput(id="B", sequence=binder_seq, msa=None))
    request = StructurePredictionInput(sequences=chains)

    from esm.models.esmfold2.processor import ESMFold2InputBuilder

    builder = ESMFold2InputBuilder()
    result = builder.fold(
        model,
        request,
        num_loops=int(cfg.get("num_loops", 20)),
        num_sampling_steps=int(cfg.get("num_sampling_steps", 200)),
        num_diffusion_samples=int(cfg.get("num_diffusion_samples", 1)),
    )

    metrics: dict[str, float] = {}
    if getattr(result, "iptm", None) is not None:
        metrics["i_pTM"] = float(result.iptm)
    if getattr(result, "ptm", None) is not None:
        metrics["pTM"] = float(result.ptm)
    plddt = getattr(result, "plddt", None)
    if plddt is not None:
        metrics["pLDDT"] = float(np.asarray(plddt.detach().cpu() if hasattr(plddt, "detach") else plddt).mean())
    pae = getattr(result, "pae", None)
    if pae is not None:
        pae_np = np.asarray(pae.detach().cpu() if hasattr(pae, "detach") else pae)
        target_len = sum(len(s) for s in target_seqs)
        metrics["i_pAE"] = float(pae_interaction(pae_np, target_len))
    return metrics


_ESMFOLD2_MODEL_CACHE: dict[str, object] = {}


def _esmfold2_model(cfg: dict):
    """Load ESMFold2 once per process, keyed on the checkpoint id.

    Without this the model reloads per design, which is the mistake ``run_esmfold``
    still makes and which dominates the cost of anything called per-design.
    """
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model_id = str(cfg.get("model_id", "biohub/ESMFold2"))
    if model_id not in _ESMFOLD2_MODEL_CACHE:
        logger.info(f"Loading ESMFold2 for advisory refolding: {model_id}")
        model = ESMFold2Model.from_pretrained(model_id)
        if cfg.get("cuda", True):
            model = model.cuda()
        _ESMFOLD2_MODEL_CACHE[model_id] = model.eval()
    return _ESMFOLD2_MODEL_CACHE[model_id]


def clear_consensus_model_cache() -> None:
    """Release cached advisory models (tests, or before switching checkpoints)."""
    _ESMFOLD2_MODEL_CACHE.clear()


CONSENSUS_BACKENDS: dict[str, Callable[[list[str], str, dict], dict[str, float]]] = {
    "esmfold2": _score_esmfold2,
}


def available_backends() -> list[str]:
    return sorted(CONSENSUS_BACKENDS)


# =============================================================================
# Column naming
# =============================================================================


def advisory_column(seq_type: str, backend: str, metric_suffix: str) -> str:
    return f"{seq_type}_{backend}_{metric_suffix}"


def assert_columns_are_advisory(columns: list[str], gated_columns: set[str]) -> None:
    """Fail loudly if an advisory column could be read as a gated one.

    Cheap insurance against a future backend named "complex", or a threshold
    gaining a backend prefix: the whole contract of this module is that nothing
    it emits can change a pass/fail decision.
    """
    collisions = sorted(set(columns) & gated_columns)
    if collisions:
        raise ValueError(f"Advisory columns collide with gated columns: {collisions}")
    suspicious = sorted(c for c in columns if f"_{_GATED_PREFIX}_" in c)
    if suspicious:
        raise ValueError(
            f"Advisory columns use the reserved '{_GATED_PREFIX}' prefix and could be "
            f"mistaken for gated metrics: {suspicious}"
        )


# =============================================================================
# Cache
# =============================================================================


def consensus_fingerprint(backend: str, cfg: dict, target_seqs: list[str]) -> str:
    """Identity of an advisory scorer: backend, its settings, and the target.

    The target is part of the key because these are complex metrics -- the same
    binder against a different target is a different number.
    """
    canonical = json.dumps(
        {"backend": backend, "cfg": {k: cfg[k] for k in sorted(cfg)}, "target_seqs": list(target_seqs)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_consensus_cache(cache_dir: str, backend: str, fingerprint: str) -> dict[str, dict[str, float]]:
    path = consensus_cache_path(cache_dir, backend)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") != fingerprint:
            logger.info(
                f"Advisory fold cache at {path} was produced by a different scorer "
                f"({str(cached.get('fingerprint'))[:12]} != {fingerprint[:12]}); recomputing"
            )
            return {}
        return {k: v for k, v in cached["scores"].items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable advisory fold cache {path}: {exc}")
        return {}


def write_consensus_cache(cache_dir: str, backend: str, fingerprint: str, scores: dict[str, dict[str, float]]) -> None:
    try:
        blob = json.dumps({"fingerprint": fingerprint, "scores": scores})
        with open(consensus_cache_path(cache_dir, backend), "w") as handle:
            handle.write(blob)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"Could not write advisory fold cache for {cache_dir}: {exc}")


# =============================================================================
# Public API
# =============================================================================


def score_binders(
    backend: str,
    target_seqs: list[str],
    binder_seqs: list[str],
    cfg: dict | None = None,
    cache_dir: str | None = None,
    reuse_cache: bool = True,
) -> list[dict[str, float]]:
    """Advisory metrics for each binder against the target, in input order.

    Never raises and never blocks a campaign: an unavailable backend or a failed
    fold yields empty dicts, and the caller writes NaN columns. Failures are not
    cached, so a transient one does not become permanent across resumes.

    A diffusion-based folder costs minutes per complex, so callers are expected
    to pass only the sequences they actually want scored -- see
    ``metric.consensus_best_only``.
    """
    cfg = dict(cfg or {})
    if backend not in CONSENSUS_BACKENDS:
        logger.error(f"Unknown advisory folding backend '{backend}'. Known: {available_backends()}")
        return [{} for _ in binder_seqs]
    if not target_seqs or not binder_seqs:
        return [{} for _ in binder_seqs]

    fingerprint = consensus_fingerprint(backend, cfg, target_seqs)
    scores: dict[str, dict[str, float]] = {}
    if cache_dir and reuse_cache:
        scores = read_consensus_cache(cache_dir, backend, fingerprint)

    pending = [s for s in dict.fromkeys(binder_seqs) if s and s not in scores]
    if pending:
        scorer = CONSENSUS_BACKENDS[backend]
        fresh: dict[str, dict[str, float]] = {}
        for seq in pending:
            try:
                metrics = scorer(target_seqs, seq, cfg)
            except Exception as exc:
                logger.warning(f"Advisory backend '{backend}' failed on a {len(seq)}-residue binder: {exc}")
                continue
            usable = {k: float(v) for k, v in metrics.items() if k in CONSENSUS_METRIC_SUFFIXES and v == v}
            if usable:
                fresh[seq] = usable
        scores.update(fresh)
        if cache_dir and fresh:
            write_consensus_cache(cache_dir, backend, fingerprint, scores)
        logger.info(f"Advisory backend '{backend}' scored {len(fresh)}/{len(pending)} binders")

    return [scores.get(seq, {}) for seq in binder_seqs]
