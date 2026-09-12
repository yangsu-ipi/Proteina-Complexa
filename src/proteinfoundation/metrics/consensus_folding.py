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
from proteinfoundation.metrics.pae_store import save_pae
from proteinfoundation.metrics.tmol_interface import TMOL_METRIC_COLS, tmol_interface_metrics
from proteinfoundation.result_analysis.binder_analysis_utils import COMPLEX_BACKEND_COLUMN, complex_backend_of

# Metrics a backend may report. Named to mirror the primary backend's metrics so
# a column-to-column comparison reads naturally, without reusing its prefix.
# target_pLDDT and binder_pLDDT split the complex mean the way the AF2 side does.
# Advisory like everything else here: ESMFold2 runs on a compressed scale (see
# above), so these are for looking at, not for filtering on.
# Confidence the folder reports and nothing on disk can reproduce. These, and
# only these, belong in the fold fingerprint.
CONSENSUS_CONFIDENCE_SUFFIXES = (
    "i_pTM",
    "pTM",
    "pLDDT",
    "target_pLDDT",
    "binder_pLDDT",
)

# The KINDS of number the PAE family produces, without the cutoffs that
# instantiate them. This is the family's identity for fingerprinting: whether
# ipSAE is reported at all is a property of the backend, while the distances it
# is reported at are a property of the request. Conflating them put "_10" in the
# fold fingerprint, where adding a third cutoff would have refolded a campaign to
# recompute arithmetic over a matrix already on disk.
PAE_BASE_METRICS = (
    "i_pAE",
    "pAE",
    "min_ipAE",
    "min_ipSAE",
    "max_ipSAE",
    "avg_ipSAE",
)

# The distance cutoffs the ipSAE columns are scored at, in Angstroms, each with
# the suffix its columns carry: the plain ones at 15, the "_10" ones at 10.
#
# Changing this list costs a re-read of the stored PAE matrices and nothing more.
# Cutoffs already computed for a structure are reused, so rounds of comparison
# with overlapping cutoffs pay only for what is new -- see
# :func:`missing_pae_cutoffs`.
IPSAE_CUTOFFS: tuple[tuple[float, str], ...] = ((15.0, ""), (10.0, "_10"))


def ipsae_suffixes(cutoffs: tuple[tuple[float, str], ...] | None = None) -> tuple[str, ...]:
    """The ipSAE column names a set of cutoffs produces."""
    return tuple(
        f"{kind}ipSAE{suffix}"
        for _, suffix in (IPSAE_CUTOFFS if cutoffs is None else cutoffs)
        for kind in ("min_", "max_", "avg_")
    )


def consensus_metric_suffixes(cutoffs: tuple[tuple[float, str], ...] | None = None) -> tuple[str, ...]:
    """Every column a backend reports, at the cutoffs this run asks for."""
    return ("i_pAE", "pAE", "min_ipAE", *ipsae_suffixes(cutoffs), *CONSENSUS_CONFIDENCE_SUFFIXES)


# The columns of the current request. A module-level name because most readers
# want exactly this; anything that must not move when a cutoff changes uses
# CONSENSUS_CONFIDENCE_SUFFIXES and PAE_BASE_METRICS instead.
CONSENSUS_METRIC_SUFFIXES = consensus_metric_suffixes()

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
    # Reduced by mean, never best-of. This used to take the best by i_pAE,
    # matching how the primary backend picks a representative refold -- but a
    # refold and a diffusion sample are not the same thing. num_diffusion_samples
    # draws N times from ONE distribution; seeds draw from N different ones,
    # which is why this repo takes several seeds and holds this at 1
    # (assert_single_diffusion_sample). A best-of over draws from one
    # distribution reports the luckiest draw as though it were the prediction,
    # and puts a best-of in the same row as the seed means beside it.
    #
    # Nominal in practice: with the invariant held this is a mean over one
    # sample, identical to the value it replaces. It earns its keep on any path
    # the validator does not cover.
    scored = [_mean_sample_metrics(scored)]
    # Sample 0's structure, not a ranked winner: with one sample it is the only
    # one, and with several no structure is the mean of the others, so an
    # arbitrary-but-stated choice beats a flattering one.
    best = 0

    if out_pdb_path:
        # Write the sample the metrics describe, so a disagreement with the
        # primary backend can be looked at rather than only read as numbers.
        # Same call run_esmfold2 uses for monomers.
        try:
            os.makedirs(os.path.dirname(out_pdb_path), exist_ok=True)
            results[best].complex.to_protein_complex().to_pdb(out_pdb_path)
            scored[best]["pdb_path"] = out_pdb_path
            # The matrix every PAE-family column here is computed from, beside the
            # structure it describes. Chain lengths in the order they were written
            # -- targets first, binder last -- so a reader can split the interface
            # block without opening the PDB.
            save_pae(
                out_pdb_path,
                getattr(results[best], "pae", None),
                chain_lengths=[len(x) for x in target_seqs] + [len(binder_seq)],
                backend="esmfold2",
                # Resolved the same way the fold resolved it, not read off a cfg
                # key the campaigns do not set: consensus_cfg carries model_id
                # only when someone overrides the checkpoint, so this recorded an
                # empty string on every real run -- a stored matrix that could not
                # say which model produced it.
                model=_esmfold2_model_id(cfg),
                seed=seed,
            )
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


def pae_family(
    pae,
    target_len: int,
    cutoffs: tuple[tuple[float, str], ...] | None = None,
    include_base: bool = True,
) -> dict[str, float]:
    """Every metric that is a pure function of the PAE matrix.

    One definition, two callers: the backend computes these while it still holds
    the folder's output, and the derivation recomputes them from the matrix
    stored beside the structure. Two copies of this arithmetic would be two
    answers to "what is min_ipSAE".

    *cutoffs* defaults to the whole request. Passing a subset is how a round of
    comparison pays only for the cutoffs it does not already have, and
    *include_base* drops i_pAE / pAE / min_ipAE, which no cutoff affects.
    """
    from esm.models.esmfold2.interface_metrics import ipsae, pae_interaction

    array = _np(pae)
    metrics: dict[str, float] = {}
    if not include_base:
        # Skip straight to the cutoff-dependent half.
        for cutoff, suffix in (IPSAE_CUTOFFS if cutoffs is None else cutoffs):
            scored = ipsae(array, target_len, cutoff)
            forward = float(scored["ipsae_target_binder"])
            reverse = float(scored["ipsae_binder_target"])
            metrics[f"min_ipSAE{suffix}"] = min(forward, reverse)
            metrics[f"max_ipSAE{suffix}"] = max(forward, reverse)
            metrics[f"avg_ipSAE{suffix}"] = (forward + reverse) / 2
        return metrics
    # Divided by the top bin, because that is what every other backend's column
    # of this name holds: ColabDesign divides inside its loss, the RF3 adapter
    # divides on the way in, and a threshold carries the divisor as `scale` so it
    # can state itself in Angstroms. Left raw, i_pAE here read 3.86 beside AF2's
    # 0.173 for the same design -- one name, 31x apart, in the column pair the
    # advisory track exists to compare. Both heads bin to the same 31, so this is
    # a shared convention rather than one model's scale imposed on another.
    metrics["i_pAE"] = float(pae_interaction(array, target_len)) / PAE_MAX_BIN
    # The binder's own rows against everything, symmetrised: ColabDesign's `pae`
    # is get_pae_loss(mask_1d=binder_id) over (p + p.T) / 2.
    symmetric = (array + array.T) / 2
    metrics["pAE"] = float(symmetric[target_len:].mean()) / PAE_MAX_BIN
    # And the single most confident target-binder pair, unsymmetrised --
    # get_min_ipae_loss leaves its symmetrisation commented out on purpose.
    metrics["min_ipAE"] = float(array[target_len:, :target_len].min()) / PAE_MAX_BIN
    # ipSAE is NOT divided: it is already a TM-like 0-1, and it is computed from
    # the PAE in Angstroms against a cutoff in Angstroms, exactly as the vendored
    # ColabDesign computes it. The fork ships its own implementation of the same
    # algorithm (same d0, same 1/(1 + (pae/d0)^2) term, same bidirectional
    # max-then-min/max), so this uses that rather than a third copy. They differ
    # in one place, the floor on the d0 length -- 27 there, 26 here -- so a very
    # small interface can read slightly differently between them.
    for cutoff, suffix in (IPSAE_CUTOFFS if cutoffs is None else cutoffs):
        scored = ipsae(array, target_len, cutoff)
        forward = float(scored["ipsae_target_binder"])
        reverse = float(scored["ipsae_binder_target"])
        metrics[f"min_ipSAE{suffix}"] = min(forward, reverse)
        metrics[f"max_ipSAE{suffix}"] = max(forward, reverse)
        metrics[f"avg_ipSAE{suffix}"] = (forward + reverse) / 2
    return metrics


# Where an entry records which distance produced each ipSAE suffix, so a suffix
# reused from an earlier round is known to have been computed at the cutoff this
# round is asking for -- and a suffix whose cutoff moved is recomputed rather
# than silently kept. Not a metric; carried alongside pdb_path.
PAE_CUTOFF_KEY = "pae_cutoffs"


def missing_pae_cutoffs(entry: dict, cutoffs: tuple[tuple[float, str], ...] | None = None) -> tuple:
    """The (cutoff, suffix) pairs this entry cannot already answer.

    Presence of the columns is not enough: a suffix is only reusable if it was
    produced at the distance now being asked for. ``_10`` written at 10 A stays
    valid when a round adds 12; the plain suffix written at 15 does not when a
    round redefines it to 12, even though the column name is unchanged.
    """
    produced = entry.get(PAE_CUTOFF_KEY) or {}
    wanted = IPSAE_CUTOFFS if cutoffs is None else cutoffs
    out = []
    for cutoff, suffix in wanted:
        names = (f"min_ipSAE{suffix}", f"max_ipSAE{suffix}", f"avg_ipSAE{suffix}")
        if produced.get(suffix) == float(cutoff) and all(n in entry for n in names):
            continue
        out.append((cutoff, suffix))
    return tuple(out)


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
        # By the same function the re-derivation uses, so a column computed while
        # the folder's output was in hand and one recomputed from the stored
        # matrix are the same number by construction rather than by review.
        metrics.update(pae_family(pae, target_len))
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
    from proteinfoundation.metrics.esmfold2_loader import load_esmfold2

    return load_esmfold2(_esmfold2_model_id(cfg), cuda=bool(cfg.get("cuda", True)))


def _mean_sample_metrics(scored: list[dict]) -> dict:
    """One fold's metrics, meaned over the diffusion samples that produced them.

    Non-numeric entries are taken from the first sample -- provenance, identical
    by construction -- the way :func:`mean_interface_metrics` treats the SASA
    engine. A sample that produced no usable number for a metric is dropped
    rather than counted, so one NaN cannot cost the others.
    """
    if len(scored) <= 1:
        return dict(scored[0]) if scored else {}
    out: dict = {}
    for key in scored[0]:
        values = [m.get(key) for m in scored]
        numbers = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
        out[key] = sum(numbers) / len(numbers) if numbers else values[0]
    return out


def assert_single_diffusion_sample(consensus_cfg) -> None:
    """ESMFold2 draws once per fold here, and the ensemble comes from seeds.

    Both knobs produce several structures, and they are not interchangeable.
    ``num_diffusion_samples`` draws N times from ONE distribution; N seeds build
    N different distributions and draw once from each. The second is the wider
    ensemble and the one this repo buys -- ``n_esmfold2_seeds`` -- so
    num_diffusion_samples stays at 1 and every ESMFold2 path can assume a fold
    returns one structure.

    Checked at the top of evaluate rather than where the folds happen, because
    the alternative is discovering it after hours of generation: the config
    errors this repo has actually been bitten by all surfaced deep in a stage
    that had already earned its input.
    """
    requested = int((consensus_cfg or {}).get("num_diffusion_samples", 1) or 1)
    if requested != 1:
        raise ValueError(
            f"consensus_cfg.num_diffusion_samples={requested}, but ESMFold2 in this repo folds one "
            f"sample per seed. Several samples come from one distribution; several seeds come from "
            f"several, which is the ensemble worth paying for -- so raise n_esmfold2_seeds to "
            f"{requested} instead and leave num_diffusion_samples at 1."
        )


def _esmfold2_model_id(cfg: dict) -> str:
    """Which checkpoint :func:`_esmfold2_model` will load for this cfg."""
    from proteinfoundation.metrics.esmfold2_loader import complex_model_id

    return str(cfg.get("model_id") or complex_model_id())


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


def advisory_backends_in(columns, seq_type: str, complex_backend: str) -> list[str]:
    """Advisory backends a frame carries columns for, read off the column names.

    The alternative is threading ``consensus_backends`` from the config into
    analyze, which would then describe the run's *request* rather than the
    frame's *contents* -- and a pooled frame can hold runs that asked for
    different backends. The names are this module's to parse.
    """
    prefix, suffix = f"{seq_type}_complex_", "_i_pAE"
    found = set()
    for column in columns:
        if not (column.startswith(prefix) and column.endswith(suffix)):
            continue
        backend = column[len(prefix) : -len(suffix)]
        # One slot: a backend name occupies exactly one, so anything with a
        # separator in it is a metric that happens to end in i_pAE.
        if backend and "_" not in backend and backend != complex_backend:
            found.add(backend)
    return sorted(found)


def assert_frame_headline_indices_agree(df, seq_types: list[str]) -> None:
    """Every row's headline describes one sequence, across both tracks.

    Run after the headline is chosen and the verdicts refreshed: that is when the
    invariant exists, and when the verdict -- the column most able to drift, since
    it is re-derived downstream -- has been written.
    """
    complex_backend = complex_backend_of(df) or "af2"
    for seq_type in seq_types:
        for backend in advisory_backends_in(df.columns, seq_type, complex_backend):
            for row in df.to_dict("records"):
                assert_headline_indices_agree(row, seq_type, backend)


def _agreeing_indices(row: dict, columns: list[str]) -> set[int] | None:
    """Indices at which every scalar equals its own ``_all`` entry.

    None when there is nothing to check -- no ``_all`` lists, or no scalars beside
    them. NaN counts as agreeing with NaN: a backend that failed writes NaN to
    both, and that is consistent, not a mismatch.
    """
    candidates: set[int] | None = None
    for column in columns:
        values = row.get(f"{column}_all")
        if not isinstance(values, list):
            continue
        if column not in row:
            # A column with no scalar makes no claim about which sequence the row
            # describes, so it cannot disagree with one. Evaluate emits per-sequence
            # lists and nothing else -- analyze builds the headline -- so at that
            # point EVERY scalar is absent, and reading them as None made the
            # function report that no index explained the headline. It was right
            # that none did: there was no headline yet.
            continue
        scalar = row[column]
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

    Call it where headlines exist. Evaluate emits per-sequence lists and no
    scalars at all, so there is nothing there for this to compare; the invariant
    comes into being in :func:`pick_headline_sequence`, and that is where this
    runs.

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
            # The KINDS of number the folder is asked for, not the cutoffs they
            # are instantiated at. With the PAE matrix stored beside every
            # structure, a cutoff is answerable by re-reading a file, so putting
            # one here would refold a campaign to recompute arithmetic. What
            # still belongs is whether a metric is reported at all: a cache
            # written before ipSAE existed holds no way to produce it.
            "metrics": sorted((*CONSENSUS_CONFIDENCE_SUFFIXES, *PAE_BASE_METRICS)),
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
# The force-field family, kept apart because it is the one derived set that is
# not always wanted. It needs a compiled extension that not every environment
# has, it costs a scorer construction and a pose build per structure, and the
# binder campaigns run with it off -- so registering it unconditionally would
# make every advisory entry look permanently under-derived on a box that cannot
# compute it, re-reading every kept PDB on every run to produce nothing. It is
# requested instead, by the same config flag that turns TMOL on for the generated
# and primary-backend structures, and the request is part of the fingerprint.
CONSENSUS_TMOL_SUFFIXES: tuple[str, ...] = tuple(TMOL_METRIC_COLS)

# Bumped when the derivation of any registered metric changes without its name
# changing, which the name alone cannot express.
CONSENSUS_DERIVATION_VERSION = 1


def consensus_derived_suffixes(include_tmol: bool = False) -> tuple[str, ...]:
    """What this run reads off an advisory structure."""
    return CONSENSUS_DERIVED_SUFFIXES + (CONSENSUS_TMOL_SUFFIXES if include_tmol else ())


def consensus_derivation_fingerprint(include_tmol: bool = False) -> str:
    """Identity of what is read OFF an advisory structure, not of the structure.

    Kept apart from :func:`consensus_fingerprint` so a metrics-only change
    re-derives from kept structures instead of refolding: on the CBLN1 campaign
    that is the difference between re-reading 22k PDBs and folding them again,
    three seeds deep, for numbers the folder does not influence.

    *include_tmol* is in the hash rather than assumed, so turning the force field
    on re-reads the structures and turning it off does not: a run that asked for
    less is not stale, it asked for less.
    """
    canonical = json.dumps(
        {
            "derived": sorted(consensus_derived_suffixes(include_tmol)),
            "version": CONSENSUS_DERIVATION_VERSION,
            # The ipSAE cutoffs are deliberately NOT here. Hashing them made any
            # change re-derive every structure-read metric -- SASA, shape
            # complementarity, secondary structure, the RMSDs -- to recompute a
            # number that takes under a millisecond, ~2 hours of re-reading on
            # CBLN1 for ~13 seconds of arithmetic. Worse, it recomputed cutoffs
            # the entry already held, so rounds of comparison with overlapping
            # cutoffs paid for the overlap every time. Each entry records which
            # distance produced each suffix instead, so a round pays only for
            # what is new: see missing_pae_cutoffs.
        },
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
    pdb_path: str,
    n_target_chains: int,
    reference_pdb_path: str | None = None,
    include_tmol: bool = False,
    have: dict | None = None,
) -> dict[str, float]:
    """Read the registered derived metrics off one advisory structure.

    *reference_pdb_path* is the designed complex, needed only by the geometry
    family -- everything else is a property of the advisory structure alone.
    Without it the geometry keys are simply absent, which a caller sees as "not
    derivable" and retries next run; the only caller in the pipeline always has
    the design in hand.

    *include_tmol* adds the force-field family. It is a request rather than a
    registration for the reasons on :data:`CONSENSUS_TMOL_SUFFIXES`, and it must
    match the flag the caller hashed into the derivation fingerprint.

    *have* is what the entry already holds. Anything already answered is skipped,
    which is what keeps a round that only adds an ipSAE cutoff from re-reading
    the structure: opening the PDB and scoring its interface costs ~0.3 s and the
    new cutoff costs under a millisecond. Pass ``{}`` -- or leave it out -- when
    the derivation fingerprint moved and everything must be computed again.

    Returns ``{}`` while nothing is registered, which is what makes the split
    inert until a caller opts in. Raises nothing of its own: a caller treats a
    failure as "not derivable for this structure" and leaves the columns absent,
    so one unreadable PDB does not cost a refold of everything.
    """
    wanted_suffixes = consensus_derived_suffixes(include_tmol)
    held = have or {}
    derived: dict[str, float] = {}

    # Everything below reads the structure; skip each part whose answers are
    # already in hand.
    from_structure = [n for n in wanted_suffixes if n not in held]
    geometry = set(CONSENSUS_RMSD_SUFFIXES.values()) & set(from_structure)
    scored_names = [n for n in from_structure if n not in geometry]
    if scored_names:
        from proteinfoundation.utils.pr_alternative_utils import pr_alternative_score_interface

        # The chain ids this module wrote. Derived rather than sniffed so the
        # mapping stays with the writer: advisory_chain_ids puts the binder last.
        chains = advisory_chain_ids(n_target_chains)
        scores, _, _ = pr_alternative_score_interface(
            pdb_path,
            binder_chain=chains[-1],
            target_chain=",".join(chains[:-1]),
        )
        derived.update({name: scores[name] for name in scored_names if name in scores})
    if reference_pdb_path and geometry:
        derived.update(
            {k: v for k, v in rmsd_against_design(pdb_path, reference_pdb_path).items() if k in geometry}
        )
    if include_tmol and any(n not in held for n in CONSENSUS_TMOL_SUFFIXES):
        # By the same function and the same scorer the generated complex is read
        # through, on a structure whose chains this module wrote. TMOL works the
        # chains out itself, so an advisory complex needs no special casing.
        derived.update(tmol_interface_metrics(pdb_path))
    derived.update(pae_family_from_store(pdb_path, n_target_chains, have=held))
    return derived


def pae_family_from_store(pdb_path: str, n_target_chains: int, have: dict | None = None) -> dict[str, float]:
    """The PAE family recomputed from the matrix stored beside a structure.

    This is what makes a cutoff change cost a re-read. Only the cutoffs *have*
    cannot already answer are computed, and the base metrics -- i_pAE, pAE,
    min_ipAE, which no cutoff affects -- only when they are absent. So a second
    round of comparison that keeps 15 A and adds 12 pays for 12 alone, and a
    third round that asks for 10, 12 and 15 together pays for nothing at all.

    The returned dict carries :data:`PAE_CUTOFF_KEY`, the record of which
    distance produced each suffix. Without it, reuse would be by column name,
    and a round that redefined the plain suffix from 15 A to 12 would keep the
    15 A numbers under a name that had come to mean something else.

    Empty when no matrix was stored, which is every fold from before the store
    existed. Those entries keep their folder-reported values -- nothing on disk
    can move them -- and :func:`score_binders` counts them in a warning.
    """
    from proteinfoundation.metrics.pae_store import load_pae

    held = have or {}
    needed = missing_pae_cutoffs(held)
    needs_base = any(name not in held for name in ("i_pAE", "pAE", "min_ipAE"))
    if not needed and not needs_base:
        return {}

    stored = load_pae(pdb_path)
    if not stored:
        return {}
    lengths = stored.get("chain_lengths")
    if lengths and len(lengths) >= 2:
        target_len = int(sum(lengths[:-1]))
    else:
        # Nothing said where the binder starts. Guessing would put the interface
        # block in the wrong place and produce plausible, wrong numbers.
        logger.warning(f"Stored PAE for {pdb_path} records no chain lengths; cannot place the interface")
        return {}
    if n_target_chains and len(lengths) - 1 != n_target_chains:
        logger.warning(
            f"Stored PAE for {pdb_path} describes {len(lengths) - 1} target chain(s), not {n_target_chains}; "
            f"leaving the PAE family to the folder-reported values"
        )
        return {}
    try:
        computed = pae_family(stored["pae"], target_len, cutoffs=needed, include_base=needs_base)
    except Exception as exc:
        logger.warning(f"Could not recompute the PAE family from the stored matrix at {pdb_path}: {exc}")
        return {}
    if needed:
        produced = dict(held.get(PAE_CUTOFF_KEY) or {})
        produced.update({suffix: float(cutoff) for cutoff, suffix in needed})
        computed[PAE_CUTOFF_KEY] = produced
    return computed


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
    derive_tmol: bool = False,
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

    *derive_tmol* asks for the force-field family as well, and should carry the
    same value as the run's ``compute_tmol``: the advisory structures then answer
    the same four questions the generated complex does.
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
    wanted_suffixes = consensus_derived_suffixes(derive_tmol)
    derivation = consensus_derivation_fingerprint(derive_tmol) if wanted_suffixes else None

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
    # refolding. Runs when the derivation changed, when an entry simply lacks a
    # derived key -- an entry cached before its structure existed heals itself
    # once the PDB is there, instead of staying blank forever behind a derivation
    # fingerprint that already matches -- and when a cutoff this run asks for is
    # one the entry has not been scored at.
    if scores and cache_dir:
        rederived: dict[str, dict[int, dict[str, float | str]]] = {}
        failed = 0
        # Entries whose PAE family the stored matrix could not refresh. Only
        # interesting when the derivation moved: their folder-reported values
        # then answer the question the cutoffs used to ask, and no file on disk
        # can produce the new answer without predicting the complex again.
        unrefreshable_pae = 0
        for seq, by_seed in scores.items():
            for seed, metrics in by_seed.items():
                complete = all(k in metrics for k in wanted_suffixes) and not missing_pae_cutoffs(metrics)
                if not derivation_stale and complete:
                    continue
                pdb = metrics.get("pdb_path") or existing_advisory_structure(cache_dir, backend, seq, seed)
                if not (isinstance(pdb, str) and os.path.exists(pdb)):
                    continue
                try:
                    derived = derive_from_structure(
                        pdb,
                        len(target_seqs),
                        reference_pdb_path,
                        include_tmol=derive_tmol,
                        # A moved derivation fingerprint means the numbers
                        # themselves changed meaning, so nothing may be reused;
                        # otherwise only what is genuinely absent is computed,
                        # which is what makes adding a cutoff cost a millisecond
                        # rather than a re-read of the structure.
                        have={} if derivation_stale else metrics,
                    )
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
                if missing_pae_cutoffs(metrics) and PAE_CUTOFF_KEY not in derived:
                    unrefreshable_pae += 1
                if PAE_CUTOFF_KEY in derived:
                    # Not a metric, so it does not survive the filter above; kept
                    # explicitly, the way pdb_path is.
                    usable[PAE_CUTOFF_KEY] = derived[PAE_CUTOFF_KEY]
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
        if unrefreshable_pae:
            logger.warning(
                f"{unrefreshable_pae} advisory structures have no stored PAE matrix, so their ipSAE "
                f"columns keep the values the folder reported when they were folded, at whatever "
                f"cutoffs were current then. Only a refold can score them at "
                f"{[c for c, _ in IPSAE_CUTOFFS]} A. Folds made from now on store the matrix."
            )

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
            if any(k.endswith("ipSAE") or "ipSAE_" in k for k in usable):
                # Which distances the folder just scored at, recorded the same
                # way a re-derivation records them -- so a later round asking for
                # one of these reuses it instead of reading the matrix back.
                usable[PAE_CUTOFF_KEY] = {suffix: float(cutoff) for cutoff, suffix in IPSAE_CUTOFFS}
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
