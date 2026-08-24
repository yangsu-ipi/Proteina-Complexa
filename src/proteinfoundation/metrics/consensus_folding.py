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
simply absent. ``cfg`` is the ``metric.consensus_cfg`` mapping; if a backend reads
file paths from it, add those keys to :data:`_PATH_VALUED_CFG_KEYS` so the cache
keys on contents rather than filenames. Register it in :data:`CONSENSUS_BACKENDS`. Imports must be lazy
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

    Same input shape as the fork's own reference adapter
    (``oracle/backends/local_esmfold2.py``): one ProteinInput per chain, target
    chains first so the binder is the last asym id.

    The target may carry an MSA (``consensus_cfg.target_msa`` or
    ``target_msa_paths``), which is what the fork's CLI calls the "validated
    production path"; without one this runs single-sequence like that reference
    adapter, worth remembering when comparing against published ESMFold2 numbers.
    The binder never carries one -- see :func:`_target_msas`.

    Note ``msa_trunk_depth`` is a ProductionFoldConfig field consumed by
    ``fold_complex_production``, not by ``builder.fold``, so the depth control
    available here is ``msa_max_sequences`` at load time.

    Untested against real weights: they live in a private repo and were not
    available where this was written. Treat the first real run as the test.
    """
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

    model = _esmfold2_model(cfg)
    target_msas = _target_msas(target_seqs, cfg)
    chains = [
        ProteinInput(id=f"T{i}", sequence=s, msa=m)
        for i, (s, m) in enumerate(zip(target_seqs, target_msas, strict=True))
    ]
    # msa=None for the binder, always. A de novo miniprotein has no meaningful
    # alignment, and this is not a knob for that reason.
    chains.append(ProteinInput(id="B", sequence=binder_seq, msa=None))
    request = StructurePredictionInput(sequences=chains)

    builder = ESMFold2InputBuilder()
    folded = builder.fold(
        model,
        request,
        num_loops=int(cfg.get("num_loops", 20)),
        num_sampling_steps=int(cfg.get("num_sampling_steps", 200)),
        num_diffusion_samples=int(cfg.get("num_diffusion_samples", 1)),
    )

    # fold() returns a bare MolecularComplexResult only when
    # num_diffusion_samples == 1; otherwise a list (processor.py, "if
    # num_diffusion_samples == 1 and len(results) == 1"). Since
    # num_diffusion_samples is a documented consensus_cfg knob, reading fields off
    # the return value directly would silently yield no metrics the moment anyone
    # raised it.
    results = folded if isinstance(folded, list) else [folded]
    scored = [_esmfold2_metrics(r, sum(len(s) for s in target_seqs)) for r in results]
    scored = [m for m in scored if m]
    if not scored:
        return {}
    # Best-of-N by interface PAE, matching how the primary backend picks a
    # representative refold (select_best_sample_idx on i_pAE, lower is better).
    # Falls back to i_pTM, then to the first sample.
    if all("i_pAE" in m for m in scored):
        return min(scored, key=lambda m: m["i_pAE"])
    if all("i_pTM" in m for m in scored):
        return max(scored, key=lambda m: m["i_pTM"])
    return scored[0]


def _esmfold2_metrics(result, target_len: int) -> dict[str, float]:
    """Reduce one MolecularComplexResult to advisory metrics.

    Field names are as declared on the dataclass (plddt, ptm, iptm, pae); each is
    optional there, so every one is guarded.
    """
    from esm.models.esmfold2.interface_metrics import pae_interaction

    metrics: dict[str, float] = {}
    if getattr(result, "iptm", None) is not None:
        metrics["i_pTM"] = float(result.iptm)
    if getattr(result, "ptm", None) is not None:
        metrics["pTM"] = float(result.ptm)
    plddt = getattr(result, "plddt", None)
    if plddt is not None:
        metrics["pLDDT"] = float(_np(plddt).mean())
    pae = getattr(result, "pae", None)
    if pae is not None:
        metrics["i_pAE"] = float(pae_interaction(_np(pae), target_len))
    return metrics


def _np(x) -> np.ndarray:
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


# Loaded MSAs, keyed on (path, max_sequences). Reading and validating an a3m per
# design would be wasteful and would repeat the same error message per design.
_MSA_CACHE: dict[tuple[str, int], object] = {}


def _load_msa(path: str, max_sequences: int):
    """Load and validate one a3m, or raise with a message naming the problem.

    Applies the same two checks the fork's own CLI applies before folding: the
    alignment's query must be the sequence being folded, and the alignment must
    have depth >= 2. A silently-ignored or mismatched MSA is worse than none,
    because the run would look like it used one.

    Do not edit an MSA while a run is in flight. The parsed alignment is cached
    here for the process while the cache fingerprint is recomputed from disk per
    design, so an in-place edit mid-run would key fresh scores to new contents
    while still folding against the alignment loaded earlier.
    """
    from esm.utils.msa import MSA

    key = (os.path.abspath(path), int(max_sequences))
    if key not in _MSA_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"target MSA not found: {path}")
        _MSA_CACHE[key] = MSA.from_a3m(path, max_sequences=int(max_sequences))
    return _MSA_CACHE[key]


def _target_msas(target_seqs: list[str], cfg: dict) -> list[object | None]:
    """One MSA (or None) per target chain, from cfg.

    ``target_msa`` accepts a single path for a single-chain target;
    ``target_msa_paths`` a list aligned with the target chains, with null for
    chains that have none. The binder never gets one: de novo miniproteins have
    no meaningful alignment, and handing the model a spurious one would change
    the prediction for the worse.
    """
    paths = cfg.get("target_msa_paths")
    if paths is None:
        single = cfg.get("target_msa")
        paths = [single] + [None] * (len(target_seqs) - 1) if single else None
    if not paths:
        return [None] * len(target_seqs)
    paths = list(paths)
    if len(paths) != len(target_seqs):
        raise ValueError(
            f"target_msa_paths has {len(paths)} entries for {len(target_seqs)} target chain(s); "
            "pass one entry per chain (null where a chain has no MSA)"
        )

    max_sequences = int(cfg.get("msa_max_sequences", 16384))
    if max_sequences < 1:
        raise ValueError("msa_max_sequences must be positive")

    msas: list[object | None] = []
    for chain_idx, (path, seq) in enumerate(zip(paths, target_seqs, strict=True)):
        if not path:
            msas.append(None)
            continue
        msa = _load_msa(path, max_sequences)
        query = msa.query.replace("-", "").upper()
        if query != seq.upper():
            raise ValueError(
                f"target MSA {path} does not match target chain {chain_idx}: "
                f"query is {len(query)} residues, chain is {len(seq)}"
            )
        if msa.depth < 2:
            raise ValueError(f"target MSA {path} has depth {msa.depth}; need at least 2 sequences")
        msas.append(msa)
    return msas


def _esmfold2_model(cfg: dict):
    """The complex-folding checkpoint, cached per process by the shared loader.

    Defaults to the full Experimental-Cutoff2025 checkpoint rather than the Fast
    one: this path folds target+binder and can take a target MSA, which is the
    setting the fork's own deploy scripts use their "critic" model for. Monomer
    refolding uses Fast instead -- see ``esmfold2_loader``.
    """
    from proteinfoundation.metrics.esmfold2_loader import complex_model_id, load_esmfold2

    model_id = str(cfg.get("model_id") or complex_model_id())
    return load_esmfold2(model_id, cuda=bool(cfg.get("cuda", True)))


def clear_consensus_model_cache() -> None:
    """Release cached advisory models (tests, or before switching checkpoints)."""
    from proteinfoundation.metrics.esmfold2_loader import clear_esmfold2_cache

    clear_esmfold2_cache()


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


# cfg keys whose values are file paths. Their *contents* belong in the cache key:
# editing an MSA in place while leaving its path alone would otherwise serve
# scores computed against the old alignment.
_PATH_VALUED_CFG_KEYS = ("target_msa", "target_msa_paths")


def _digest_file(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return "sha256:" + hashlib.sha256(handle.read()).hexdigest()[:32]
    except OSError:
        # Unreadable now; the scorer will fail and say so. Keep the path so the
        # key still changes if it is later pointed somewhere else.
        return f"unreadable:{path}"


def cfg_for_fingerprint(cfg: dict) -> dict:
    """cfg with file paths replaced by content digests."""
    resolved = {}
    for key in sorted(cfg):
        value = cfg[key]
        if key in _PATH_VALUED_CFG_KEYS and value:
            if isinstance(value, str):
                resolved[key] = _digest_file(value)
            elif isinstance(value, (list, tuple)):
                resolved[key] = [_digest_file(v) if v else None for v in value]
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def consensus_fingerprint(backend: str, cfg: dict, target_seqs: list[str]) -> str:
    """Identity of an advisory scorer: backend, its settings, and the target.

    The target is part of the key because these are complex metrics -- the same
    binder against a different target is a different number. Settings are taken
    through :func:`cfg_for_fingerprint` so an MSA is keyed on its contents rather
    than its filename.
    """
    canonical = json.dumps(
        {"backend": backend, "cfg": cfg_for_fingerprint(cfg), "target_seqs": list(target_seqs)},
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
