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

A note for anyone reading a CSV written before this: the PAE family here used to
be stored in Angstroms while every other backend stored it divided by the top PAE
bin, so ``{seq}_complex_esmfold2_i_pAE`` read 3.86 where ``..._af2_i_pAE`` read
0.173 for the same design. It is divided now, like the rest. Over the EFNB3
production run the medians land at 0.181 and 0.164 -- the same scale, with the
disagreement in the upper tail that the advisory track exists to show, rather
than a factor of 31 hiding it. The change rides along with a fold-fingerprint
change, so no cache can serve the old units into a new run; only CSVs already
written hold them.

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
import string
from collections.abc import Callable

import numpy as np
from loguru import logger

from proteinfoundation.metrics.column_names import rename
from proteinfoundation.metrics.ensembling import PAE_MAX_BIN, mean_chain_plddt
from proteinfoundation.result_analysis.binder_analysis_utils import COMPLEX_BACKEND_COLUMN

# Metrics a backend may report. Named to mirror the primary backend's metrics so
# a column-to-column comparison reads naturally, without reusing its prefix.
# target_pLDDT and binder_pLDDT split the complex mean the way the AF2 side does.
# Advisory like everything else here: ESMFold2 runs on a compressed scale (see
# above), so these are for looking at, not for filtering on.
CONSENSUS_METRIC_SUFFIXES = (
    # The PAE family, on the same 0-1 scale and by the same definitions the
    # primary backend's columns of these names carry -- see _esmfold2_metrics.
    "i_pAE",
    "pAE",
    "min_ipAE",
    "min_ipSAE",
    "max_ipSAE",
    "avg_ipSAE",
    "min_ipSAE_10",
    "max_ipSAE_10",
    "avg_ipSAE_10",
    "i_pTM",
    "pTM",
    "pLDDT",
    "target_pLDDT",
    "binder_pLDDT",
)

# One cache file per backend. A single shared file would thrash the moment two
# backends are enabled together: each writes its own fingerprint, and the other's
# entries are discarded on every design.
CONSENSUS_CACHE_TEMPLATE = "consensus_fold_cache_{backend}.json"


def consensus_cache_path(cache_dir: str, backend: str) -> str:
    return os.path.join(cache_dir, CONSENSUS_CACHE_TEMPLATE.format(backend=backend))




# =============================================================================
# Backends
# =============================================================================


class AdvisoryStructureWriteError(RuntimeError):
    """A requested advisory structure could not be written.

    Distinct from a fold that failed, which is per-design and survivable. This
    says the run cannot honour ``keep_folding_outputs``, which is systematic --
    the next design will fail the same way -- so it stops rather than repeating
    an expensive fold whose only requested output is a file it cannot produce.
    """


def advisory_chain_ids(n_target_chains: int) -> list[str]:
    """Chain IDs for the advisory complex: targets in order, binder last.

    Single characters because PDB gives the chain ID exactly one column. The
    earlier ``T0``/``T1`` labels were readable and unwriteable, so every advisory
    structure write failed -- and since a missing structure is what asks for a
    refold, the folds recurred on every run, indefinitely, producing nothing.

    Only the order carries meaning: the metrics index the complex by
    ``target_len`` rather than by name, and nothing reads these structures back.
    A single-chain target therefore gets ``A``/``B``, which is also what the
    generated complexes use.
    """
    if n_target_chains < 1:
        raise ValueError("an advisory complex needs at least one target chain")
    needed = n_target_chains + 1
    if needed > len(string.ascii_uppercase):
        raise ValueError(
            f"{needed} chains exceed the {len(string.ascii_uppercase)} single-character IDs a PDB "
            f"chain column can hold; this complex cannot be written as PDB at all"
        )
    return list(string.ascii_uppercase[:needed])


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
    ids = advisory_chain_ids(len(target_seqs))
    chains = [
        ProteinInput(id=ids[i], sequence=s, msa=m)
        for i, (s, m) in enumerate(zip(target_seqs, target_msas, strict=True))
    ]
    # msa=None for the binder, always. A de novo miniprotein has no meaningful
    # alignment, and this is not a knob for that reason.
    chains.append(ProteinInput(id=ids[-1], sequence=binder_seq, msa=None))
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
        except Exception as exc:
            # Not a warning. keep_folding_outputs asked for this file, and a
            # missing structure is what triggers the refold -- so swallowing the
            # failure means folding this complex again on every future run and
            # producing nothing, which is what happened here for months of runs.
            raise AdvisoryStructureWriteError(
                f"Could not write the advisory complex structure requested by keep_folding_outputs "
                f"to {out_pdb_path}: {exc}. The metrics were computed, but a structure that cannot "
                f"be written will be refolded on every run, so this stops instead."
            ) from exc
    return scored[best]


def _esmfold2_metrics(result, target_len: int) -> dict[str, float]:
    """Reduce one MolecularComplexResult to advisory metrics.

    Field names are as declared on the dataclass (plddt, ptm, iptm, pae); each is
    optional there, so every one is guarded.
    """
    metrics: dict[str, float] = {}
    if getattr(result, "iptm", None) is not None:
        metrics["i_pTM"] = float(result.iptm)
    if getattr(result, "ptm", None) is not None:
        metrics["pTM"] = float(result.ptm)
    plddt = getattr(result, "plddt", None)
    if plddt is not None:
        array = _np(plddt)
        metrics["pLDDT"] = float(array.mean())
        # The complex mean is mostly target on any real binder target, so it
        # barely moves when a binder folds badly. Splitting it costs nothing
        # here: the per-residue array is already in hand.
        metrics.update(mean_chain_plddt(array, target_len))
    pae = getattr(result, "pae", None)
    if pae is not None:
        # Imported here rather than at the top of the function: these are the only
        # metrics that need esm, and a result without a pae should not pay for
        # loading it.
        from esm.models.esmfold2.interface_metrics import ipsae, pae_interaction

        array = _np(pae)
        # Divided by the top bin, because that is what every other backend's
        # column of this name holds: ColabDesign divides inside its loss, the RF3
        # adapter divides on the way in, and a threshold carries the divisor as
        # `scale` so it can state itself in Angstroms. Left raw, i_pAE here read
        # 3.86 beside AF2's 0.173 for the same design -- one name, 31x apart, in
        # the column pair the advisory track exists to compare. Both heads bin to
        # the same 31, so this is a shared convention rather than one model's
        # scale imposed on another.
        metrics["i_pAE"] = float(pae_interaction(array, target_len)) / PAE_MAX_BIN
        # The binder's own rows against everything, symmetrised: ColabDesign's
        # `pae` is get_pae_loss(mask_1d=binder_id) over (p + p.T) / 2.
        symmetric = (array + array.T) / 2
        metrics["pAE"] = float(symmetric[target_len:].mean()) / PAE_MAX_BIN
        # And the single most confident target-binder pair, unsymmetrised --
        # get_min_ipae_loss leaves its symmetrisation commented out on purpose.
        metrics["min_ipAE"] = float(array[target_len:, :target_len].min()) / PAE_MAX_BIN
        # ipSAE is NOT divided: it is already a TM-like 0-1, and it is computed
        # from the PAE in Angstroms against a cutoff in Angstroms, exactly as the
        # vendored ColabDesign computes it -- 15 A for the plain columns, 10 for
        # the _10 ones. The fork ships its own implementation of the same
        # algorithm (same d0, same 1/(1 + (pae/d0)^2) term, same bidirectional
        # max-then-min/max), so this uses that rather than a third copy. They
        # differ in one place, the floor on the d0 length -- 27 there, 26 here --
        # so a very small interface can read slightly differently between them.
        for cutoff, suffix in ((15.0, ""), (10.0, "_10")):
            scored = ipsae(array, target_len, cutoff)
            forward = float(scored["ipsae_target_binder"])
            reverse = float(scored["ipsae_binder_target"])
            metrics[f"min_ipSAE{suffix}"] = min(forward, reverse)
            metrics[f"max_ipSAE{suffix}"] = max(forward, reverse)
            metrics[f"avg_ipSAE{suffix}"] = (forward + reverse) / 2
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
    """An advisory column, under the same slots every other metric uses.

    The kind slot says complex because that is what an advisory backend folds.
    These columns used to avoid ``_complex_`` on purpose, as the marker that they
    are never gated; that separation is now structural rather than lexical --
    a criterion's backend comes from binder_folding_method, and an advisory
    backend arrives through consensus_backends, so no criterion can name one.
    """
    return f"{seq_type}_complex_{backend}_{metric_suffix}"


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
    # The primary columns carry a backend slot -- `mpnn_complex_af2_i_pAE`, not
    # `mpnn_complex_i_pAE`. This read the pre-rename shape, so `_agreeing_indices`
    # found no `_all` lists, returned None, and the function returned before
    # checking anything. It had been dead on every real row since the slot scheme
    # landed, while its tests passed because they built rows the old way -- a name
    # written in one place and read in another, which is the failure this very
    # function exists to catch. The backend travels on the row for exactly this
    # reason, so read it there rather than taking another argument.
    complex_backend = row.get(COMPLEX_BACKEND_COLUMN) or "af2"
    primary_columns = [
        rename(f"{seq_type}_complex_{m}", complex_backend) for m in CONSENSUS_METRIC_SUFFIXES
    ]
    # The verdict is a headline column too, and the one that actually drifted: it
    # is re-derived downstream, so it is the one most able to end up describing a
    # different sequence than the metrics beside it.
    primary_columns.append(f"{seq_type}_pass")
    primary = _agreeing_indices(row, primary_columns)
    advisory = _agreeing_indices(row, [advisory_column(seq_type, backend, m) for m in CONSENSUS_METRIC_SUFFIXES])
    if primary is not None and not primary:
        raise ValueError(
            f"No single sequence explains the '{seq_type}' headline: its scalars disagree with each "
            f"other about which entry of their own _all lists they are. Most often {seq_type}_pass "
            f"against the metric columns -- see {seq_type}_best_idx."
        )
    if primary is None or advisory is None:
        return  # best-only mode, or a metric this backend does not report
    if primary and advisory and not (primary & advisory):
        raise ValueError(
            f"Advisory headline for '{seq_type}' / '{backend}' describes a different sequence than the "
            f"primary headline: primary headline is at index(es) {sorted(primary)}, advisory at "
            f"{sorted(advisory)}. Both scalars must be their own _all list's entry at one shared index."
        )


def assert_columns_are_advisory(
    columns: list[str], gated_columns: set[str], existing_columns: set[str] | None = None
) -> None:
    """Fail loudly if an advisory column could be read as a gated one.

    The whole contract of this module is that nothing it emits can change a
    pass/fail decision.

    Two lexical versions of this check have now been wrong, in opposite
    directions, and both because a column was classified by what its name
    contains rather than by what reads it:

    * the first refused any advisory column containing ``_complex_``, which under
      the slot scheme is exactly what an advisory complex refold is called
      (``{seq}_complex_esmfold2_i_pAE``);
    * the second refused any *gated* column containing ``_{backend}_`` for a
      configured advisory backend. But a model can serve two tracks at once --
      the CBLN1 campaign runs ``esmfold2`` as both ``consensus_backends`` and
      ``apo_folding_models`` -- and the apo criterion is gated on purpose. Once
      the rename moved the model into the backend slot, the gated, deliberate
      ``{seq}_apo_esmfold2_binder_scRMSD_ca`` became indistinguishable from an
      advisory column, and every evaluate run under that config died on its
      first design.

    So no name is inspected here at all. ``gated_columns`` is the set the pass
    criteria actually resolve to (:func:`binder_eval_utils.gated_columns`), and
    the invariant is that the advisory track's columns are disjoint from it. A
    gate that genuinely reads an advisory column still fails, because that column
    is in both sets; one that reads an apo fold from the same model does not,
    because it is in only one.

    ``existing_columns`` is a separate, weaker guard: an advisory column must not
    silently overwrite one already built for this row, whether gated or not.
    """
    read_by_a_gate = sorted(set(columns) & gated_columns)
    if read_by_a_gate:
        raise ValueError(
            f"A pass criterion reads advisory columns: {read_by_a_gate}. An advisory fold must not "
            f"decide a pass/fail; gate on the backend from binder_folding_method instead."
        )
    if existing_columns:
        collisions = sorted(set(columns) & existing_columns)
        if collisions:
            raise ValueError(f"Advisory columns collide with columns already built: {collisions}")


# =============================================================================
# Cache
# =============================================================================


# cfg keys whose values are file paths. Their *contents* belong in the cache key:
# editing an MSA in place while leaving its path alone would otherwise serve
# scores computed against the old alignment.
_PATH_VALUED_CFG_KEYS = ("target_msa", "target_msa_paths")


def _digest_file(path: str) -> str:
    """Kept as a local name; the implementation is shared with the refolding
    fingerprint so the two cannot digest the same file differently."""
    from proteinfoundation.evaluation.binder_eval_cache import digest_file

    return digest_file(path)


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
            # Which metrics the FOLDER reports, not only how it folded. These
            # come out of the prediction -- i_pAE, i_pTM, pTM and the pLDDTs are
            # not recoverable from a PDB -- so a cache written before one of them
            # existed holds no way to answer for it, and refolding is the only
            # repair. Metrics that ARE readable from the kept structure belong in
            # consensus_derivation_fingerprint instead, where a change re-reads
            # the file rather than spending minutes per complex reproducing a
            # structure the folder would return unchanged.
            "metrics": sorted(CONSENSUS_METRIC_SUFFIXES),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


CONSENSUS_CACHE_SCHEMA = 2  # 1 held one fold per binder; 2 holds one per (binder, seed)

# Geometry keys, mapped from what calculate_prot_prot_binder_rmsd returns to the
# suffix the primary backend's columns carry. Its ``complex_scRMSD_ca`` lands in
# the kind slot on the primary side -- {seq}_complex_{backend}_scRMSD_ca -- so the
# advisory suffix is ``scRMSD_ca``, not ``complex_scRMSD_ca``, which would emit
# the word twice. The legacy ``binder_scRMSD`` / ``complex_scRMSD`` aliases are
# deliberately absent: they equal the CA values and exist only for frames written
# before the modes were named.
CONSENSUS_RMSD_SUFFIXES: dict[str, str] = {
    "binder_scRMSD_ca": "binder_scRMSD_ca",
    "binder_scRMSD_bb3": "binder_scRMSD_bb3",
    "binder_scRMSD_bb3o": "binder_scRMSD_bb3o",
    "binder_scRMSD_allatom": "binder_scRMSD_allatom",
    "complex_scRMSD_ca": "scRMSD_ca",
    "binder_scRMSD_target_aligned_ca": "binder_scRMSD_target_aligned_ca",
}

# Metrics read off a kept advisory structure rather than reported by the folder.
# They are not part of a structure's identity: the same fold answers for any of
# them, so changing which are computed -- or how -- must re-read the PDBs already
# on disk. Empty until a caller registers one; the mechanism exists so that
# registering is cheap.
# Read off the kept structure, in the same names and by the same code the
# mandatory side uses, so an ESMFold2 interface can be compared with an AF2 one
# rather than being a second definition of buried area that happens to share a
# word. Shape complementarity was held out of this list as "the most expensive
# of the set", which stopped being true when it moved in process: over CBLN1's
# kept ESMFold2 complexes it is ~0.07s of a ~0.33s derivation, behind the SASA
# step. It answers what the rest cannot -- whether an advisory interface PACKS
# like the primary one, or only buries the same area.
CONSENSUS_DERIVED_SUFFIXES: tuple[str, ...] = (
    "sasa_engine",
    "sasa_radii",
    "interface_sc",
    "binder_dSASA",
    "target_dSASA",
    "interface_dSASA",
    "binder_buried_fraction",
    "binder_surface_hydrophobicity",
    "binder_interface_hydrophobicity",
    "binder_interface_nres",
    "target_interface_nres",
    "binder_ss_counts",
    "binder_ss_total",
    "binder_interface_ss_counts",
    "binder_interface_ss_total",
    "target_ss_counts",
    "target_ss_total",
    "target_interface_ss_counts",
    "target_interface_ss_total",
    # Geometry against the designed backbone. Derived rather than folder-reported
    # because the structure holds it: the advisory PDB and the design are both on
    # disk, so asking whether ESMFold2 places the binder where AF2 does costs a
    # re-read, never a refold.
    *CONSENSUS_RMSD_SUFFIXES.values(),
)
# Bumped when the derivation of any registered metric changes without its name
# changing, which the name alone cannot express.
CONSENSUS_DERIVATION_VERSION = 1


def consensus_derivation_fingerprint() -> str:
    """Identity of what is read OFF an advisory structure, not of the structure.

    Kept apart from :func:`consensus_fingerprint` so a metrics-only change
    re-derives from kept structures instead of refolding: on the CBLN1 campaign
    that is the difference between re-reading 22k PDBs and folding them again,
    three seeds deep, for numbers the folder does not influence.
    """
    canonical = json.dumps(
        {"derived": sorted(CONSENSUS_DERIVED_SUFFIXES), "version": CONSENSUS_DERIVATION_VERSION},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rmsd_against_design(pdb_path: str, reference_pdb_path: str) -> dict[str, float]:
    """The advisory fold measured against the designed backbone.

    By the primary backend's own function, not a second definition of scRMSD:
    :func:`calculate_prot_prot_binder_rmsd` is what produces
    ``{seq}_complex_af2_binder_scRMSD_ca``, so the ESMFold2 column can be read
    beside the AF2 one rather than being a similarly-named number. Both it and
    the loader are imported lazily -- they pull torch and atomworks, which a run
    with no geometry registered should not pay for.

    The binder is the last chain in both structures: :func:`advisory_chain_ids`
    writes it last on purpose, and the generated complexes use the same order.
    """
    from atomworks.io.utils.io_utils import load_any

    from proteinfoundation.metrics.binder_metrics import calculate_prot_prot_binder_rmsd

    advisory = load_any(pdb_path, file_type="pdb")[0]
    design = load_any(reference_pdb_path)[0]
    computed = calculate_prot_prot_binder_rmsd(
        refolded_complex=advisory, gen_complex=design, label="advisory"
    )
    return {
        suffix: float(computed[key]) for key, suffix in CONSENSUS_RMSD_SUFFIXES.items() if key in computed
    }


def derive_from_structure(
    pdb_path: str, n_target_chains: int, reference_pdb_path: str | None = None
) -> dict[str, float]:
    """Read the registered derived metrics off one advisory structure.

    *reference_pdb_path* is the designed complex, needed only by the geometry
    family -- everything else is a property of the advisory structure alone.
    Without it the geometry keys are simply absent, which a caller sees as "not
    derivable" and retries next run; the only caller in the pipeline always has
    the design in hand.

    Returns ``{}`` while nothing is registered, which is what makes the split
    inert until a caller opts in. Raises nothing of its own: a caller treats a
    failure as "not derivable for this structure" and leaves the columns absent,
    so one unreadable PDB does not cost a refold of everything.
    """
    if not CONSENSUS_DERIVED_SUFFIXES:
        return {}
    from proteinfoundation.utils.pr_alternative_utils import pr_alternative_score_interface

    # The chain ids this module wrote. Derived rather than sniffed so the mapping
    # stays with the writer: advisory_chain_ids puts the binder last.
    chains = advisory_chain_ids(n_target_chains)
    scores, _, _ = pr_alternative_score_interface(
        pdb_path,
        binder_chain=chains[-1],
        target_chain=",".join(chains[:-1]),
    )
    derived = {name: scores[name] for name in CONSENSUS_DERIVED_SUFFIXES if name in scores}
    wanted = set(CONSENSUS_RMSD_SUFFIXES.values()) & set(CONSENSUS_DERIVED_SUFFIXES)
    if reference_pdb_path and wanted:
        derived.update(
            {k: v for k, v in rmsd_against_design(pdb_path, reference_pdb_path).items() if k in wanted}
        )
    return derived


def read_consensus_cache(
    cache_dir: str, backend: str, fingerprint: str, seed_for=None, derivation: str | None = None
) -> tuple[dict[str, dict[int, dict[str, float | str]]], bool]:
    """Cached advisory scores as ``({binder_seq: {seed: metrics}}, derivation_stale)``.

    *derivation* is the current :func:`consensus_derivation_fingerprint`. A cache
    written under a different one still holds usable structures and folder
    metrics, so it is returned rather than discarded, with the flag set: the
    caller re-reads the kept PDBs for the derived metrics. Returning the flag
    instead of a bare dict is deliberate -- a caller cannot then forget to ask.

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
        return {}, False
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") != fingerprint:
            logger.info(
                f"Advisory fold cache at {path} was produced by a different scorer "
                f"({str(cached.get('fingerprint'))[:12]} != {fingerprint[:12]}); recomputing"
            )
            return {}, False
        stale = derivation is not None and cached.get("derivation") != derivation
        if stale:
            logger.info(
                f"Advisory fold cache at {path} was read off under a different derivation "
                f"({str(cached.get('derivation'))[:12]} != {derivation[:12]}); "
                f"re-deriving from the kept structures rather than refolding"
            )
        raw = cached.get("scores") or {}
        if cached.get("schema") == CONSENSUS_CACHE_SCHEMA:
            return {
                seq: {int(k): v for k, v in by_seed.items() if isinstance(v, dict)}
                for seq, by_seed in raw.items()
                if isinstance(by_seed, dict)
            }, stale
        out: dict[str, dict[int, dict]] = {}
        for seq, metrics in raw.items():
            if not isinstance(metrics, dict):
                continue
            if seed_for is None:
                continue
            out[seq] = {int(seed_for(seq)): metrics}
        if out:
            logger.info(f"Adopted {len(out)} schema-1 advisory entries at {path} under their derived seeds")
        return out, stale
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable advisory fold cache {path}: {exc}")
        return {}, False


def write_consensus_cache(
    cache_dir: str, backend: str, fingerprint: str, scores: dict[str, dict[int, dict]],
    derivation: str | None = None,
) -> None:
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
        # Merged on the fold fingerprint alone, then stamped with the current
        # derivation: entries that could not be re-derived (no kept structure)
        # keep their folder metrics and simply lack the derived ones, and the
        # missing-key check on the next read picks them up if the PDB reappears.
        blob = json.dumps(
            {
                "fingerprint": fingerprint,
                "schema": CONSENSUS_CACHE_SCHEMA,
                "derivation": derivation if derivation is not None else consensus_derivation_fingerprint(),
                "scores": merged,
            }
        )
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
    pool them. Non-numeric entries (``pdb_path``, the SASA engine and radii) are
    taken from the first seed rather than averaged; the structures differ per
    seed, and one of them has to be the one a reader is pointed at, while the
    engine and radii are identical across seeds by construction.
    """
    if not by_seed:
        return {}
    ordered = [by_seed[s] for s in sorted(by_seed)]
    out: dict[str, float | str] = {}
    for key in ordered[0]:
        values = [m[key] for m in ordered if key in m]
        # Packed eight-state counts average elementwise. Taking the first seed's
        # would report one draw's secondary structure beside pLDDTs that are
        # means of three, and nothing in the row would say so.
        lists = [v for v in values if isinstance(v, (list, tuple))]
        if lists and len({len(v) for v in lists}) == 1:
            out[key] = [sum(col) / len(col) for col in zip(*lists, strict=True)]
            continue
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
    reference_pdb_path: str | None = None,
) -> list[dict[str, float | str]]:
    """Advisory metrics for each binder against the target, in input order.

    Never raises and never blocks a campaign: an unavailable backend or a failed
    fold yields empty dicts, and the caller writes NaN columns. Failures are not
    cached, so a transient one does not become permanent across resumes.

    A diffusion-based folder costs minutes per complex. The caller passes every
    sequence of the type anyway: scoring only the primary's pick would condition
    the advisory sample on the ranking it exists to check, and would leave a
    single-entry ``_all`` list that no later stage can re-rank from.

    *reference_pdb_path* is the designed complex, which the geometry family is
    measured against. Only the derived side uses it, so a caller without one gets
    everything except scRMSD.
    """
    cfg = dict(cfg or {})
    if backend not in CONSENSUS_BACKENDS:
        logger.error(f"Unknown advisory folding backend '{backend}'. Known: {available_backends()}")
        return [{} for _ in binder_seqs]
    if not target_seqs or not binder_seqs:
        return [{} for _ in binder_seqs]

    from proteinfoundation.metrics.seeding import deterministic_seed, deterministic_seeds

    fingerprint = consensus_fingerprint(backend, cfg, target_seqs)
    # Only ask about the derivation when something is actually read off the
    # structures. With nothing registered there is no staleness that matters, and
    # asking would report every pre-split cache as stale on every run.
    derivation = consensus_derivation_fingerprint() if CONSENSUS_DERIVED_SUFFIXES else None

    # Seeds are derived here rather than inside the scorer, so one place decides
    # what a fold's identity is and the scorer stays a pure function of its
    # inputs. A pinned cfg.seed means exactly one fold, however many are asked
    # for: it names a specific sample, and repeating it would be the same fold
    # counted twice.
    pinned = cfg.get("seed")
    n_seeds = max(1, int(cfg.get("n_seeds", cfg.get("n_esmfold2_seeds", 1))))

    def seeds_for(seq: str) -> list[int]:
        if pinned is not None:
            return [int(pinned)]
        return deterministic_seeds(*target_seqs, seq, count=n_seeds)

    def first_seed_for(seq: str) -> int:
        return int(pinned) if pinned is not None else deterministic_seed(*target_seqs, seq)

    # per binder sequence: {seed: metrics}
    scores: dict[str, dict[int, dict[str, float | str]]] = {}
    derivation_stale = False
    if cache_dir and reuse_cache:
        scores, derivation_stale = read_consensus_cache(
            cache_dir, backend, fingerprint, seed_for=first_seed_for, derivation=derivation
        )

    # Re-read the kept structures for metrics that are read off them, rather than
    # refolding. Runs when the derivation changed, and also when an entry simply
    # lacks a derived key -- an entry cached before its structure existed heals
    # itself once the PDB is there, instead of staying blank forever behind a
    # derivation fingerprint that already matches.
    if scores and CONSENSUS_DERIVED_SUFFIXES and cache_dir:
        rederived: dict[str, dict[int, dict[str, float | str]]] = {}
        failed = 0
        for seq, by_seed in scores.items():
            for seed, metrics in by_seed.items():
                if not derivation_stale and all(k in metrics for k in CONSENSUS_DERIVED_SUFFIXES):
                    continue
                pdb = metrics.get("pdb_path") or existing_advisory_structure(cache_dir, backend, seq, seed)
                if not (isinstance(pdb, str) and os.path.exists(pdb)):
                    continue
                try:
                    derived = derive_from_structure(pdb, len(target_seqs), reference_pdb_path)
                except Exception as exc:
                    failed += 1
                    logger.warning(f"Could not re-derive advisory metrics from {pdb}: {exc}")
                    continue
                # Lists (the packed eight-state counts) and the engine/radii
                # strings pass through; float() on either would raise inside the
                # loop that exists to avoid refolding.
                usable = {
                    k: v
                    for k, v in derived.items()
                    if isinstance(v, (list, tuple, str)) or (isinstance(v, (int, float)) and v == v)
                }
                if usable:
                    metrics.update(usable)
                    rederived.setdefault(seq, {})[seed] = metrics
        if rederived:
            write_consensus_cache(cache_dir, backend, fingerprint, rederived, derivation=derivation)
            logger.info(
                f"Advisory backend '{backend}' re-derived metrics for "
                f"{sum(len(v) for v in rederived.values())} (sequence, seed) structures without refolding"
            )
        if failed:
            logger.warning(f"Advisory re-derivation failed for {failed} structures; their columns stay absent")

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
            except AdvisoryStructureWriteError:
                # Systematic, not per-design: the next binder writes to the same
                # kind of path and fails the same way. Tolerating it here is what
                # made an unwriteable structure look like a survivable hiccup.
                raise
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
            write_consensus_cache(cache_dir, backend, fingerprint, fresh, derivation=derivation)
        folded = sum(len(v) for v in fresh.values())
        logger.info(f"Advisory backend '{backend}' scored {folded}/{len(pending)} (sequence, seed) folds")

    return [mean_over_seeds(scores.get(seq, {})) for seq in binder_seqs]
