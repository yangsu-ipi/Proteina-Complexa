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

    (target_seqs: list[str], binder_seq: str, cfg: dict, out_pdb_path: str | None)
        -> dict[str, float]

writing the folded complex to ``out_pdb_path`` when one is given, and otherwise
not writing at all,

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
import math
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


def _score_esmfold2(
    target_seqs: list[str],
    binder_seq: str,
    cfg: dict,
    out_pdb_path: str | None = None,
    seed: int = 0,
) -> dict[str, float]:
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
    # Folded one binder at a time here, so the seed can be a pure function of the
    # fold's inputs: the same target and binder always give the same structure,
    # and a cached score therefore equals a recomputed one. cfg may pin a seed
    # instead, e.g. to draw a second independent sample of the same complex.
    logger.debug(f"Advisory fold of a {len(binder_seq)}-residue binder (seed {seed})")
    folded = builder.fold(
        model,
        request,
        num_loops=int(cfg.get("num_loops", 20)),
        num_sampling_steps=int(cfg.get("num_sampling_steps", 200)),
        num_diffusion_samples=int(cfg.get("num_diffusion_samples", 1)),
        seed=int(seed),
    )

    # fold() returns a bare MolecularComplexResult only when
    # num_diffusion_samples == 1; otherwise a list (processor.py, "if
    # num_diffusion_samples == 1 and len(results) == 1"). Since
    # num_diffusion_samples is a documented consensus_cfg knob, reading fields off
    # the return value directly would silently yield no metrics the moment anyone
    # raised it.
    results = folded if isinstance(folded, list) else [folded]
    target_len = sum(len(s) for s in target_seqs)
    # Keep results and metrics index-aligned: the chosen structure has to be the
    # one whose metrics are reported.
    paired = [(r, _esmfold2_metrics(r, target_len)) for r in results]
    paired = [(r, m) for r, m in paired if m]
    if not paired:
        return {}
    results = [r for r, _ in paired]
    scored = [m for _, m in paired]
    # Best-of-N by interface PAE, matching how the primary backend picks a
    # representative refold (select_best_sample_idx on i_pAE, lower is better).
    # Falls back to i_pTM, then to the first sample.
    if all("i_pAE" in m for m in scored):
        best = min(range(len(scored)), key=lambda i: scored[i]["i_pAE"])
    elif all("i_pTM" in m for m in scored):
        best = max(range(len(scored)), key=lambda i: scored[i]["i_pTM"])
    else:
        best = 0

    if out_pdb_path:
        # Write the sample the metrics describe, so a disagreement with the
        # primary backend can be looked at rather than only read as numbers.
        # Same call run_esmfold2 uses for monomers.
        try:
            os.makedirs(os.path.dirname(out_pdb_path), exist_ok=True)
            results[best].complex.to_protein_complex().to_pdb(out_pdb_path)
            scored[best]["pdb_path"] = out_pdb_path
        except Exception as exc:  # advisory: a failed write must not lose the metrics
            logger.warning(f"Could not write advisory complex structure to {out_pdb_path}: {exc}")
    return scored[best]


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


def _agreeing_indices(row: dict, columns: list[str]) -> set[int] | None:
    """Indices at which every scalar equals its own ``_all`` entry.

    None when there is nothing to check. NaN counts as agreeing with NaN: a
    backend that failed writes NaN to both, and that is consistent, not a
    mismatch.
    """
    candidates: set[int] | None = None
    for column in columns:
        values = row.get(f"{column}_all")
        if not isinstance(values, list):
            continue
        scalar = row.get(column)
        here = {
            i
            for i, value in enumerate(values)
            if value == scalar
            or (isinstance(value, float) and isinstance(scalar, float) and math.isnan(value) and math.isnan(scalar))
        }
        candidates = here if candidates is None else candidates & here
    return candidates


def assert_headline_indices_agree(row: dict, seq_type: str, backend: str) -> None:
    """Fail if the advisory headline describes a different sequence than the primary one.

    Every ``*_all`` column on a row is parallel: index *i* is one sequence, and the
    scalar beside each list is that list's entry at the index the row's headline
    refers to. The advisory scalars used ``advisory[0]`` while the primary ones
    used ``seq_best_idx``, so whenever the best sequence was not the first,
    ``{seq}_esmfold2_i_pAE`` and ``{seq}_complex_i_pAE`` described different
    redesigns -- with nothing in either number saying so.

    Checked on the row rather than at the point of assignment, because that is
    where the property has to hold: a future call site can reintroduce the bug
    with entirely different code and this still catches it.

    Raises:
        ValueError: If no single index explains both sets of headlines.
    """
    primary = _agreeing_indices(row, [f"{seq_type}_complex_{m}" for m in CONSENSUS_METRIC_SUFFIXES])
    advisory = _agreeing_indices(row, [advisory_column(seq_type, backend, m) for m in CONSENSUS_METRIC_SUFFIXES])
    if primary is None or advisory is None:
        return  # best-only mode, or a metric this backend does not report
    if primary and advisory and not (primary & advisory):
        raise ValueError(
            f"Advisory headline for '{seq_type}' / '{backend}' describes a different sequence than the "
            f"primary headline: primary headline is at index(es) {sorted(primary)}, advisory at "
            f"{sorted(advisory)}. Both scalars must be their own _all list's entry at one shared index."
        )


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
    from proteinfoundation.metrics.esmfold2_loader import SEED_DERIVATION_VERSION

    canonical = json.dumps(
        {
            "backend": backend,
            "cfg": cfg_for_fingerprint(cfg),
            "target_seqs": list(target_seqs),
            # Every input to the seed is already covered -- target_seqs here, the
            # binder sequence as the entry key, an explicit cfg.seed in cfg --
            # but the derivation that turns them into a seed is not.
            "seed_derivation": SEED_DERIVATION_VERSION,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


CONSENSUS_CACHE_SCHEMA = 2  # 1 held one fold per binder; 2 holds one per (binder, seed)


def read_consensus_cache(
    cache_dir: str, backend: str, fingerprint: str, seed_for=None
) -> dict[str, dict[int, dict[str, float | str]]]:
    """Cached advisory scores as ``{binder_seq: {seed: metrics}}``.

    Keyed by seed VALUE rather than position, for the reason the monomer cache is:
    a seed is what produced a result, while "the k-th seed" means something only
    relative to a derivation the key does not record.

    *seed_for* maps a binder sequence to the seed a schema-1 entry must have used,
    letting those entries be adopted instead of discarded -- the derivation is a
    pure function of the target and binder sequences, so it is recoverable. Without
    it, schema-1 entries are dropped.
    """
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
        raw = cached.get("scores") or {}
        if cached.get("schema") == CONSENSUS_CACHE_SCHEMA:
            return {
                seq: {int(k): v for k, v in by_seed.items() if isinstance(v, dict)}
                for seq, by_seed in raw.items()
                if isinstance(by_seed, dict)
            }
        out: dict[str, dict[int, dict]] = {}
        for seq, metrics in raw.items():
            if not isinstance(metrics, dict):
                continue
            if seed_for is None:
                continue
            out[seq] = {int(seed_for(seq)): metrics}
        if out:
            logger.info(f"Adopted {len(out)} schema-1 advisory entries at {path} under their derived seeds")
        return out
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable advisory fold cache {path}: {exc}")
        return {}


def write_consensus_cache(cache_dir: str, backend: str, fingerprint: str, scores: dict[str, dict[int, dict]]) -> None:
    """Persist ``{binder_seq: {seed: metrics}}``, merging with what is there.

    Merging rather than replacing is what lets a later run add seeds to an
    existing set instead of refolding all of them; the caller passes only what it
    holds, which after adoption may be fewer entries than the file has.
    """
    path = consensus_cache_path(cache_dir, backend)
    merged: dict[str, dict[str, dict]] = {}
    try:
        if os.path.exists(path):
            with open(path) as handle:
                existing = json.load(handle)
            if existing.get("fingerprint") == fingerprint and existing.get("schema") == CONSENSUS_CACHE_SCHEMA:
                merged = {k: dict(v) for k, v in (existing.get("scores") or {}).items() if isinstance(v, dict)}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        merged = {}
    for seq, by_seed in scores.items():
        merged.setdefault(seq, {}).update({str(seed): metrics for seed, metrics in by_seed.items()})
    try:
        blob = json.dumps({"fingerprint": fingerprint, "schema": CONSENSUS_CACHE_SCHEMA, "scores": merged})
        with open(path, "w") as handle:
            handle.write(blob)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"Could not write advisory fold cache for {cache_dir}: {exc}")


# =============================================================================
# Public API
# =============================================================================


def mean_over_seeds(by_seed: dict[int, dict[str, float | str]]) -> dict[str, float | str]:
    """Average a binder's metrics across its seeds.

    Seeds are exchangeable draws from a sampler -- seed k of one input has no
    correspondence to seed k of another -- so the only meaningful reduction is to
    pool them. Non-numeric entries (``pdb_path``) are taken from the first seed
    rather than averaged; the structures differ per seed, and one of them has to
    be the one a reader is pointed at.
    """
    if not by_seed:
        return {}
    ordered = [by_seed[s] for s in sorted(by_seed)]
    out: dict[str, float | str] = {}
    for key in ordered[0]:
        values = [m[key] for m in ordered if key in m]
        numeric = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
        out[key] = sum(numeric) / len(numeric) if numeric else values[0]
    out["n_seeds"] = float(len(ordered))
    return out


def advisory_structure_path(cache_dir: str, backend: str, binder_seq: str, seed: int | None = None) -> str:
    """Where a backend's folded complex for this binder and seed goes.

    Content-addressed on the binder sequence, so the path a cache entry records
    stays valid across runs and two sequences never collide. The seed is part of
    the name because each seed folds a different structure; without it, seeds
    overwrite one another and the last one silently answers for all.

    ``seed=None`` gives the pre-seed name, which is where a structure folded before
    seeds existed still lives -- see ``existing_advisory_structure``.
    """
    digest = hashlib.sha256(binder_seq.encode("utf-8")).hexdigest()[:12]
    name = f"{digest}.pdb" if seed is None else f"{digest}_seed{seed}.pdb"
    return os.path.join(cache_dir, f"{backend}_complex", name)


def existing_advisory_structure(cache_dir: str, backend: str, binder_seq: str, seed: int) -> str | None:
    """An already-folded structure for this binder and seed, wherever it lives.

    Checks the seeded name, then the pre-seed one: a structure folded before seeds
    existed was produced by the derivation's first seed, so it answers for that
    seed and should not be refolded just because the naming changed.
    """
    seeded = advisory_structure_path(cache_dir, backend, binder_seq, seed)
    if os.path.exists(seeded):
        return seeded
    legacy = advisory_structure_path(cache_dir, backend, binder_seq, None)
    return legacy if os.path.exists(legacy) else None


def score_binders(
    backend: str,
    target_seqs: list[str],
    binder_seqs: list[str],
    cfg: dict | None = None,
    cache_dir: str | None = None,
    reuse_cache: bool = True,
    keep_structures: bool = False,
) -> list[dict[str, float | str]]:
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

    from proteinfoundation.metrics.seeding import deterministic_seed, deterministic_seeds

    fingerprint = consensus_fingerprint(backend, cfg, target_seqs)

    # Seeds are derived here rather than inside the scorer, so one place decides
    # what a fold's identity is and the scorer stays a pure function of its
    # inputs. A pinned cfg.seed means exactly one fold, however many are asked
    # for: it names a specific sample, and repeating it would be the same fold
    # counted twice.
    pinned = cfg.get("seed")
    n_seeds = max(1, int(cfg.get("n_seeds", 1)))

    def seeds_for(seq: str) -> list[int]:
        if pinned is not None:
            return [int(pinned)]
        return deterministic_seeds(*target_seqs, seq, count=n_seeds)

    def first_seed_for(seq: str) -> int:
        return int(pinned) if pinned is not None else deterministic_seed(*target_seqs, seq)

    # per binder sequence: {seed: metrics}
    scores: dict[str, dict[int, dict[str, float | str]]] = {}
    if cache_dir and reuse_cache:
        scores = read_consensus_cache(cache_dir, backend, fingerprint, seed_for=first_seed_for)

    # A cached score does not imply the structure this run asked for. An earlier
    # run with keep_folding_outputs=false cached metrics and wrote no PDB, so
    # enabling retention later returned the scores and produced nothing -- the
    # request was for a file, and the cache answered about a number. Refold when
    # the structure is wanted and absent, which also repairs an entry whose PDB
    # was deleted since.
    def _needs_structure(seq: str, seed: int) -> bool:
        if not (cache_dir and keep_structures):
            return False
        return existing_advisory_structure(cache_dir, backend, seq, seed) is None

    # One unit of work is a (sequence, seed) pair, so adding a seed folds only
    # what is new rather than everything for that sequence.
    pending = [
        (seq, seed)
        for seq in dict.fromkeys(binder_seqs)
        if seq
        for seed in seeds_for(seq)
        if seed not in scores.get(seq, {}) or _needs_structure(seq, seed)
    ]
    if pending:
        scorer = CONSENSUS_BACKENDS[backend]
        fresh: dict[str, dict[int, dict[str, float | str]]] = {}
        for seq, seed in pending:
            out_pdb = (
                advisory_structure_path(cache_dir, backend, seq, seed) if (cache_dir and keep_structures) else None
            )
            try:
                metrics = scorer(target_seqs, seq, cfg, out_pdb, seed)
            except Exception as exc:
                logger.warning(f"Advisory backend '{backend}' failed on a {len(seq)}-residue binder: {exc}")
                continue
            usable = {k: float(v) for k, v in metrics.items() if k in CONSENSUS_METRIC_SUFFIXES and v == v}
            if usable:
                # pdb_path rides along in the same entry; it is not a metric, so
                # column emission filters on CONSENSUS_METRIC_SUFFIXES and picks
                # it up explicitly. Entries written before structures were kept
                # simply lack the key.
                if metrics.get("pdb_path"):
                    usable["pdb_path"] = metrics["pdb_path"]
                fresh.setdefault(seq, {})[seed] = usable
        for seq, by_seed in fresh.items():
            scores.setdefault(seq, {}).update(by_seed)
        if cache_dir and fresh:
            write_consensus_cache(cache_dir, backend, fingerprint, fresh)
        folded = sum(len(v) for v in fresh.values())
        logger.info(f"Advisory backend '{backend}' scored {folded}/{len(pending)} (sequence, seed) folds")

    return [mean_over_seeds(scores.get(seq, {})) for seq in binder_seqs]
