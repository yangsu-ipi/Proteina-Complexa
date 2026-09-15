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
refers to a backend prefix. ``report_gated_and_reported_columns`` enforces that rather
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
import shutil
import string
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger

from proteinfoundation.metrics.column_names import rename
from proteinfoundation.metrics.ensembling import (
    PAE_MAX_BIN,
    PER_MODEL_STATS_KEY,
    PLACEMENT_METRICS,
    mean_chain_plddt,
)
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
# Provenance, not measurement: how many predictions were reduced into each value.
# Emitted as a column like any other, but kept out of CONSENSUS_METRIC_SUFFIXES so
# it joins neither the fold fingerprint -- a count is not a structure request --
# nor assert_headline_indices_agree, which asks which sequence a scalar describes
# and would be answered by a constant for every index.
CONSENSUS_PROVENANCE_SUFFIXES: tuple[str, ...] = ("n_predictions",)

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


@dataclass
class ComplexFoldContext:
    """What a complex folder may need beyond the sequences.

    The old split between the "primary" complex refold and the "advisory" one was
    not a split in capability -- AF2 and ESMFold2 both fold a complex -- but in
    what each mechanism was handed. The sequence-only contract could not express
    a folder that templates on a structure, so the only folder that did one lived
    in the other mechanism, and which folders could cross-check which was decided
    by that accident.

    Widened here rather than special-cased: every backend receives the same
    context and takes what it needs. ESMFold2 ignores all of it and folds from
    sequence; AF2 templates on ``design_pdb`` and reads the target from
    ``target_pdb``.
    """

    design_pdb: str | None = None
    target_pdb: str | None = None
    target_chains: tuple[str, ...] = ()
    binder_chain: str | None = None
    design_name: str = "design"
    output_dir: str | None = None
    # A folder that is an object rather than a function of its inputs. RF3 is
    # handed a constructed runner holding its weights; the field is here rather
    # than in that backend's own signature because the whole point of this
    # dataclass is that every folder receives the same context and takes what it
    # needs -- a backend with a private argument is a backend that cannot be
    # reached the same way as the others, which is how the split this removes
    # came about in the first place.
    runner: object | None = None
    # Whether the target is a ligand rather than a protein. Changes how a folder
    # templates: RF3 selects a ground-truth conformer for ligands and a template
    # for proteins. The sequence-only folders ignore it.
    is_target_ligand: bool = False
    smiles: str | None = None


def _score_esmfold2(
    target_seqs: list[str],
    binder_seq: str,
    cfg: dict,
    out_pdb_path: str | None = None,
    draw: str | int = 0,
    context: ComplexFoldContext | None = None,
    out_path_for=None,
) -> dict[str, float]:
    """Fold target+binder with ESMFold2 and reduce to interface metrics.

    One draw per call: ESMFold2's draws are seeds, and a seed is an input to a
    single prediction. *out_path_for* is accepted and unused -- it exists for a
    folder that answers for several draws at once, which this is not.

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
    seed = seed_of_draw(draw)
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


def _score_af2(
    target_seqs: list[str],
    binder_seq: str,
    cfg: dict,
    out_pdb_path: str | None = None,
    draw: str | int = 0,
    context: ComplexFoldContext | None = None,
    out_path_for=None,
) -> dict[str, float]:
    """Fold target+binder with AlphaFold2, through the harness that already does it.

    This is the same ColabDesign call the gated complex refold makes; registering
    it here is what removes the primary/advisory distinction as a matter of
    MECHANISM. Which folder's columns decide a verdict is a threshold question,
    and is answered in analyze.

    AF2 here templates on the designed complex -- binder included -- because
    ``predict_initial_guess`` is on. That makes it a more permissive predictor
    than a template-free one, and it is why an AF2 complex number and an ESMFold2
    complex number are not two independent opinions about the same thing. They
    are still worth having side by side; they are not worth gating on one
    threshold. See docs and the per-folder thresholds.

    Answers for EVERY draw in one call, because that is what one call does:
    ColabDesign predicts with each of its parameter sets and the harness returns
    all of them. AF2's draws are those parameter sets -- its prediction is
    deterministic given one, which is why it takes a single seed however many a
    sampler beside it asks for.

    Those per-model numbers used to be averaged by ``average_af2_stats`` before
    anything outside the harness saw them, and only the first model's structure
    was named. So a metric read back off "the" structure was one model's while
    the numbers beside it were five models' mean -- measured at up to 0.55 on
    avg_ipSAE over 120 EFNB3 complexes. Each model is its own entry now, with its
    own structure, and the reduction happens once at the end over things that
    were all recorded.

    *draw* names the model the caller asked about; the return covers all of them.
    """
    if context is None or not context.design_pdb or not context.target_pdb:
        raise ValueError(
            "af2 folds a complex by templating on the designed structure, so it needs a "
            "ComplexFoldContext carrying design_pdb and target_pdb. A caller that has only "
            "sequences cannot use this backend -- use esmfold2, which folds from sequence."
        )

    from proteinfoundation.utils.colabdesign_utils import get_af2_advanced_settings, run_af_eval

    n_models = n_af2_models_in(cfg)
    settings = get_af2_advanced_settings(num_af2_models=n_models)
    stats, paths = run_af_eval(
        trajectory_pdb=context.design_pdb,
        binder_sequences=[{"seq": binder_seq}],
        design_name=context.design_name,
        output_path=context.output_dir or os.path.dirname(out_pdb_path or "") or ".",
        target_settings={"starting_pdb": context.target_pdb, "chains": ",".join(context.target_chains)},
        advanced_settings=settings,
        binder_length=len(binder_seq),
        binder_chain=context.binder_chain or "B",
        sequence_type_list=["self"],
    )
    if not stats:
        return {}
    # run_af_eval envelopes each sequence's stats as {"seq_N": stats}. Unwrapped
    # here because it was NOT: the metric filter below ran over the envelope, so
    # every key missed CONSENSUS_METRIC_SUFFIXES and this backend returned a
    # pdb_path and no numbers at all. It went unnoticed because af2 has only ever
    # been reached as folders.complex[0], which does not come through here -- the
    # first thing routing it through score_binders would have hit.
    payload = _unwrap_sequence_envelope(stats[0])
    # What each parameter set said, kept apart. The harness rides them along
    # beside its own mean for exactly this; a harness too old to do so leaves the
    # key absent and this degrades to the single reduced entry it used to return.
    per_model = payload.get(PER_MODEL_STATS_KEY)
    model_paths = payload.get("complex_pdb_paths") or paths or []
    if not isinstance(per_model, list) or not per_model:
        metrics = {k: float(v) for k, v in payload.items() if k in CONSENSUS_METRIC_SUFFIXES and v == v}
        produced = (model_paths or [None])[0]
        if produced:
            metrics["pdb_path"] = _place_structure(produced, out_pdb_path)
        return metrics

    draws: dict[str, dict[str, float]] = {}
    for index, model_stats in enumerate(per_model):
        draw_id = f"model{index + 1}"
        metrics = {k: float(v) for k, v in model_stats.items() if k in CONSENSUS_METRIC_SUFFIXES and v == v}
        if not metrics:
            continue
        produced = model_paths[index] if index < len(model_paths) else None
        if produced:
            # Each model's OWN structure at its own path. Pointing several draws
            # at one file is what made a per-draw derivation read the same model
            # five times and call it an ensemble.
            wanted = out_path_for(draw_id) if out_path_for else (out_pdb_path if draw_id == str(draw) else None)
            metrics["pdb_path"] = _place_structure(produced, wanted)
        draws[draw_id] = metrics
    return {"draws": draws}


def _unwrap_sequence_envelope(entry) -> dict:
    """The stats inside a harness's ``{"seq_N": stats}`` wrapper.

    Both complex harnesses envelope each sequence's statistics this way. Reading
    the envelope as though it were the statistics is not hypothetical: it is what
    _score_af2 did, so the metric filter matched nothing and the backend returned
    a structure path and no numbers. It went unnoticed because af2 was only ever
    reached as folders.complex[0], which does not come through here.
    """
    if not isinstance(entry, dict):
        return {}
    inner = [v for k, v in entry.items() if k.startswith("seq_") and isinstance(v, dict)]
    return inner[0] if len(inner) == 1 else entry


def _place_structure(produced: str, wanted: str | None) -> str:
    """Put a harness-produced structure where the advisory store wants it.

    The store owns where a kept structure lives, so the harness's own output is
    copied to the path the caller asked for rather than the caller being told to
    look somewhere else. Without a requested path the harness's own is reported,
    which is what a run with keep_folding_outputs off gets.
    """
    if not (wanted and os.path.exists(produced)):
        return produced
    os.makedirs(os.path.dirname(wanted), exist_ok=True)
    shutil.copy(produced, wanted)
    return wanted


def _score_rf3(
    target_seqs: list[str],
    binder_seq: str,
    cfg: dict,
    out_pdb_path: str | None = None,
    draw: str | int = 0,
    context: ComplexFoldContext | None = None,
    out_path_for=None,
) -> dict[str, float]:
    """Fold target+binder with RF3, through the harness that already does it.

    The third complex folder, and the last one that could only be reached through
    the gated path. Registering it here is what makes "every complex folder is
    reached the same way" true rather than true of two out of three -- while RF3
    was reachable only through run_binder_eval's dispatch, a campaign naming it
    could not have a second folder beside it, and one naming it second could not
    use it at all.

    RF3 reports no pTM or i_pTM and no per-chain pLDDT, so those columns are
    absent for it rather than zero. Its confidences arrive on a 0-1 scale already
    divided by PAE_MAX_BIN, which is the harness's own normalisation and is left
    alone: a number comparable to AF2's is the harness's business, not this
    function's.

    One binder per call, where the harness can batch several. That is a real cost
    and it is paid deliberately: batching is across SEQUENCES, while the contract
    here is one call per (sequence, draw), and widening it for the one folder no
    campaign currently runs would complicate the path both folders that are run
    take. Worth revisiting when RF3 is actually used.
    """
    if context is None or context.runner is None or not context.design_pdb:
        raise ValueError(
            "rf3 folds a complex through a constructed runner holding its weights, so it needs a "
            "ComplexFoldContext carrying runner and design_pdb. A caller with only sequences "
            "cannot use this backend -- use esmfold2, which folds from sequence."
        )
    from proteinfoundation.utils.rf3_model import run_rf3_eval

    stats, paths = run_rf3_eval(
        rf3_runner=context.runner,
        target_chain_ids=list(context.target_chains),
        is_target_ligand=context.is_target_ligand,
        binder_sequences=[{"seq": binder_seq}],
        sequence_type_list=["self"],
        design_name=context.design_name,
        output_path=context.output_dir or os.path.dirname(out_pdb_path or "") or ".",
        updated_pdb_path=context.design_pdb,
        binder_chain_id=context.binder_chain or "B",
        smiles=context.smiles,
    )
    if not stats:
        return {}
    payload = _unwrap_sequence_envelope(stats[0])
    metrics = {k: float(v) for k, v in payload.items() if k in CONSENSUS_METRIC_SUFFIXES and v == v}
    produced = (paths or [None])[0]
    if produced:
        metrics["pdb_path"] = _place_structure(produced, out_pdb_path)
    return metrics


CONSENSUS_BACKENDS: dict[str, Callable[[list[str], str, dict], dict[str, float]]] = {
    "esmfold2": _score_esmfold2,
    # Registered beside it, not above it. The distinction that used to live here
    # -- one folder gates, the rest advise -- was never a property of the models.
    "af2": _score_af2,
    # The last folder that could only be reached through the gated path.
    "rf3": _score_rf3,
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
        # X_best is where the headline scalar lives; bare X is where it lived
        # before it was named for what it is. Both are read, because a frame
        # written either way must still be checkable -- and because reading only
        # the new name would make this function vacuous on old frames rather than
        # wrong, which is the quieter of the two failures and the one this
        # function was already killed by once.
        column = f"{column}_best" if f"{column}_best" in row else column
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
    primary_columns.append(f"{seq_type}_pass")  # resolved to _pass_best by _agreeing_indices
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


def report_gated_and_reported_columns(
    columns: list[str], gated_columns: set[str], existing_columns: set[str] | None = None
) -> None:
    """Say which of this folder's columns decide a verdict, and which only report.

    This used to refuse any overlap: the module's contract was that nothing it
    emitted could change a pass/fail. That contract described the old
    primary/advisory split, which was never a property of a fold -- AF2 and
    ESMFold2 both fold a complex, and which one gates is a question the threshold
    config answers. With one registry, a second complex folder whose columns a
    criterion deliberately reads is an ordinary thing to configure.

    Still guarded: a column must not silently OVERWRITE one already built for
    this row, which is a mistake in any configuration.

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
        # No longer a refusal. "Advisory" was never a property of a fold -- it was
        # a property of whether any threshold named its columns, and that is a
        # question the threshold config answers. A second complex folder whose
        # columns a criterion deliberately reads is now an ordinary thing to
        # configure, so the check reports rather than refuses.
        #
        # What made the refusal worth having survives: a reader must be able to
        # see, from the run's own log, which columns decided a verdict and which
        # were only reported. That is the line below.
        logger.info(
            f"Complex folder columns read by a pass criterion: {read_by_a_gate}. "
            f"Reported-only from this folder: {sorted(set(columns) - gated_columns)}"
        )
    else:
        logger.info(f"Complex folder columns, all reported-only (no criterion reads them): {len(columns)}")
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


# Config keys that say how MANY predictions to make, not what any one of them is.
#
# A count does not belong in a fold fingerprint. The cache is keyed per draw and
# merges, so asking for more folds only what is new -- which is the whole reason
# deterministic_seeds is prefix-stable, and is only true if the count stays out
# of the identity.
#
# n_af2_models is here because putting it in cost real folds. It is AF2's draw
# count, and it was added to the SHARED consensus_cfg so AF2 would fold with the
# campaign's five parameter sets rather than one. Since the whole cfg was hashed,
# that changed ESMFOLD2's fingerprint too -- a knob it cannot see -- and every
# cached ESMFold2 complex on the campaign was discarded. The refold then failed
# on an SVD that does not always converge, so the columns came back NaN. One
# backend's knob must not be able to invalidate another's cache.
#
# n_seeds is the same key for a sampler, and it is here for the same reason --
# it was "nobody is currently hitting this" right up until EFNB3's retry, where
# the campaign now asks for one seed and its cache holds three. The fingerprint
# said different-scorer, every ESMFold2 complex was discarded, and the refold ran
# into an SVD that does not always converge. Three seeds of good folds thrown
# away to use one of them.
#
# Removing a key from a hash changes every fingerprint, which would discard the
# very caches this is meant to keep -- so the old value is reconstructed and
# accepted instead: a cache holding N draws was written when the count was N, so
# legacy_count_fingerprint(N) reproduces exactly what that run would have
# computed. Not a guess; the file says how many it holds.
_COUNT_ONLY_CFG_KEYS = ("n_af2_models", "n_seeds", "n_esmfold2_seeds")


def legacy_count_fingerprint(backend: str, cfg: dict, target_seqs: list[str], n_draws: int) -> str:
    """The fingerprint a run would have computed when the draw count was hashed.

    Reconstructed from the cache itself rather than guessed: an entry holding
    *n_draws* draws was written by a run asking for that many, so putting the
    count back gives the exact value that run stored. Accepting it is what lets
    the count leave the hash without discarding every cache keyed under it.
    """
    axis = CONSENSUS_DRAW_AXIS.get(backend, "seed")
    restored = dict(cfg)
    restored["n_af2_models" if axis == "model" else "n_seeds"] = n_draws
    return consensus_fingerprint(backend, restored, target_seqs, _hash_counts=True)


def consensus_fingerprint(
    backend: str, cfg: dict, target_seqs: list[str], _hash_counts: bool = False
) -> str:
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
            "cfg": cfg_for_fingerprint(
                cfg if _hash_counts else {k: v for k, v in cfg.items() if k not in _COUNT_ONLY_CFG_KEYS}
            ),
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


# 1 held one fold per binder; 2 one per (binder, seed); 3 one per (binder, DRAW),
# where a draw is one prediction however the backend repeats itself -- a seed for
# a sampler, a parameter set for AF2. See CONSENSUS_DRAW_AXIS.
CONSENSUS_CACHE_SCHEMA = 3

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
    cache_dir: str,
    backend: str,
    fingerprint: str,
    seed_for=None,
    derivation: str | None = None,
    legacy_fingerprint_for=None,
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
        stored = cached.get("fingerprint")
        if stored != fingerprint and legacy_fingerprint_for is not None:
            # A cache written while the draw count was part of the identity. How
            # many it holds is how many that run asked for, so the old value is
            # reconstructable rather than guessable -- see legacy_count_fingerprint.
            held = max(
                (len(v) for v in (cached.get("scores") or {}).values() if isinstance(v, dict)),
                default=0,
            )
            if held and legacy_fingerprint_for(held) == stored:
                logger.info(
                    f"Advisory fold cache at {path} was written when the draw count was part of the "
                    f"scorer's identity ({held} draws). Adopting its folds under {fingerprint[:12]} "
                    f"rather than discarding them for a count."
                )
                stored = fingerprint
        if stored != fingerprint:
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
        schema = cached.get("schema")
        if schema == CONSENSUS_CACHE_SCHEMA:
            return {
                seq: {str(k): v for k, v in by_draw.items() if isinstance(v, dict)}
                for seq, by_draw in raw.items()
                if isinstance(by_draw, dict)
            }, stale
        if schema == 2:
            # Schema 2 keyed by bare seed integer, which is the seed axis of the
            # draw key this schema uses. Renaming it is the whole migration: the
            # structures keep the names advisory_structure_path already gave
            # them, so nothing on disk moves and nothing refolds.
            out2 = {
                seq: {f"seed{int(k)}": v for k, v in by_seed.items() if isinstance(v, dict)}
                for seq, by_seed in raw.items()
                if isinstance(by_seed, dict)
            }
            if out2:
                logger.info(f"Adopted {len(out2)} schema-2 advisory entries at {path} onto the draw axis")
            return out2, stale
        out: dict[str, dict[str, dict]] = {}
        for seq, metrics in raw.items():
            if not isinstance(metrics, dict):
                continue
            if seed_for is None:
                continue
            out[seq] = {f"seed{int(seed_for(seq))}": metrics}
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
    for seq, by_draw in scores.items():
        merged.setdefault(seq, {}).update({str(draw): metrics for draw, metrics in by_draw.items()})
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


# The advisory names of the metrics that reduce by worst case rather than by
# mean, mapped from the primary side's PLACEMENT_METRICS through the same table
# that renames them. One definition of "this metric is about placement", not two
# lists that agree only by inspection -- the primary side calls it
# complex_scRMSD_ca and the advisory slot calls it scRMSD_ca.
CONSENSUS_PLACEMENT_SUFFIXES = frozenset(
    suffix for key, suffix in CONSENSUS_RMSD_SUFFIXES.items() if key in PLACEMENT_METRICS
)


def draws_by_metric(by_draw: dict[str, dict[str, float | str]]) -> dict[str, list]:
    """``{metric: [value per draw]}`` in draw order, reducing nothing.

    The un-reduced form of :func:`reduce_over_draws`, for a caller that wants the
    draws themselves in the artifact so the reduction can be re-asked later.
    Which reduction is right is a formulation over recorded values -- mean here,
    worst case there -- and a formulation belongs with the thresholds in analyze,
    where changing it costs a re-read rather than a refold. That is the same
    argument pick_headline_sequence makes for choosing among sequences.

    Draws a metric is missing from hold NaN at that position, so every list is
    the same length and position k is draw k in every one of them.
    """
    if not by_draw:
        return {}
    order = sorted(by_draw)
    keys: list[str] = []
    for draw in order:
        for key in by_draw[draw]:
            if key not in keys:
                keys.append(key)
    out: dict[str, list] = {
        key: [by_draw[draw].get(key, float("nan")) for draw in order] for key in keys
    }
    out["n_predictions"] = float(len(order))
    return out


def reduce_over_draws(by_draw: dict[str, dict[str, float | str]]) -> dict[str, float | str]:
    """Pool a binder's metrics across the draws that produced them.

    Draws are exchangeable repeat predictions -- draw k of one input has no
    correspondence to draw k of another -- so pooling is the only meaningful
    reduction. Non-numeric entries (``pdb_path``, the SASA engine and radii) are
    taken from the first draw rather than averaged; the structures differ per
    draw, and one of them has to be the one a reader is pointed at, while the
    engine and radii are identical across draws by construction.

    Placement reduces by worst case, by the rule and for the reason
    reduce_rmsd_over_models gives: every draw has to agree the binder is where it
    belongs, and a mean pulls a misplaced design toward the cutoff. That rule
    used to apply only to AF2's models, because only AF2 had more than one
    structure per prediction in reach. It now applies to every draw of every
    folder, which is a real change to ESMFold2's multi-seed placement columns --
    strictly harder to satisfy, and measurable against the baselines.

    Note what does NOT belong here: this is a reduction over repeat predictions
    of ONE sequence. Choosing among sequences is analyze's, and stays there.
    """
    if not by_draw:
        return {}
    ordered = [by_draw[s] for s in sorted(by_draw)]
    out: dict[str, float | str] = {}
    for key in ordered[0]:
        values = [m[key] for m in ordered if key in m]
        # Packed eight-state counts average elementwise. Taking the first seed's
        # would report one draw's secondary structure beside pLDDTs that are
        # means of three, and nothing in the row would say so.
        lists = [v for v in values if isinstance(v, (list, tuple))]
        numeric_lists = all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for v in lists for x in v
        )
        if lists and numeric_lists and len({len(v) for v in lists}) == 1:
            out[key] = [sum(col) / len(col) for col in zip(*lists, strict=True)]
            continue
        if lists:
            # A list that is not a vector of numbers is provenance, not a
            # measurement -- which keys an adopted entry carries as the folder's
            # own reduction, say. Identical on every draw by construction, so the
            # first one answers; averaging it tried to add strings together.
            out[key] = values[0]
            continue
        numeric = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
        if not numeric:
            out[key] = values[0]
        elif key in CONSENSUS_PLACEMENT_SUFFIXES:
            out[key] = max(numeric)
        else:
            out[key] = sum(numeric) / len(numeric)
    # Same name the primary side uses (ensembling.average_interface_rows): how
    # many predictions were reduced into this value. It was n_seeds here and
    # n_interface_models there, and only the latter ever became a column. With
    # AF2's models on the same axis it finally counts the same thing for both.
    out["n_predictions"] = float(len(ordered))
    return out


mean_over_seeds = reduce_over_draws  # pre-draw-axis name


def advisory_structure_path(cache_dir: str, backend: str, binder_seq: str, draw: str | None = None) -> str:
    """Where a backend's folded complex for this binder and draw goes.

    Content-addressed on the binder sequence, so the path a cache entry records
    stays valid across runs and two sequences never collide. The draw is part of
    the name because each draw IS a different structure; without it, draws
    overwrite one another and the last one silently answers for all.

    ``draw=None`` gives the pre-draw name, which is where a structure folded before
    any of this existed still lives -- see ``existing_advisory_structure``. Draw
    ids spell the seed axis ``seed{n}``, which is exactly the name schema 2 wrote,
    so migrating the cache key moves no file.
    """
    digest = hashlib.sha256(binder_seq.encode("utf-8")).hexdigest()[:12]
    name = f"{digest}.pdb" if draw is None else f"{digest}_{draw}.pdb"
    return os.path.join(cache_dir, f"{backend}_complex", name)


def existing_advisory_structure(cache_dir: str, backend: str, binder_seq: str, draw: str) -> str | None:
    """An already-folded structure for this binder and draw, wherever it lives.

    Checks the draw name, then the pre-draw one: a structure folded before draws
    existed was produced by the derivation's first one, so it answers for that
    draw and should not be refolded just because the naming changed.
    """
    drawn = advisory_structure_path(cache_dir, backend, binder_seq, draw)
    if os.path.exists(drawn):
        return drawn
    legacy = advisory_structure_path(cache_dir, backend, binder_seq, None)
    return legacy if os.path.exists(legacy) else None


# How a backend repeats itself when predicting one complex.
#
# A draw is one prediction. ESMFold2's draws differ by the seed it sampled at;
# AF2's differ by which of its parameter sets produced them, which is why
# fold_seeds_for gives it exactly one seed however many a sampler beside it asks
# for. Those are the same KIND of thing -- repeat measurements of one complex
# whose disagreement is information about how confident the prediction really is
# -- and the only reason they were treated differently is that one of them used
# to be averaged inside the folding harness before anything else could see it.
#
# Cached identically now: one entry per draw, each with its own structure, each
# with its own numbers read off that structure. That is what lets the reduction
# be a question anyone downstream can re-ask, and it is what stops a metric read
# off "the" structure from silently meaning the first model of five.
CONSENSUS_DRAW_AXIS: dict[str, str] = {"esmfold2": "seed", "af2": "model"}


def seed_of_draw(draw: str | int) -> int:
    """The sampler seed a draw id names, for a backend whose axis is the seed.

    Decoded rather than passed alongside, so the cache key stays the single
    statement of what produced an entry. A bare int is accepted because schema 2
    entries and direct callers still speak in seeds.
    """
    if isinstance(draw, int):
        return draw
    text = str(draw)
    return int(text[len("seed"):]) if text.startswith("seed") else int(text)


def n_af2_models_in(cfg: dict) -> int:
    """How many parameter sets AF2 predicts each complex with."""
    return max(1, int(cfg.get("n_af2_models", 1) or 1))


def draw_ids_for(backend: str, cfg: dict, target_seqs: list[str], binder_seq: str) -> list[str]:
    """The draws *backend* makes for this binder, as cache keys, in order.

    Strings rather than the bare seed integers schema 2 used, because the key has
    to say what KIND of repeat it identifies: seed 3 and model 3 are not the same
    prediction, and an int cannot tell them apart. ``seed{n}`` reproduces the
    structure filenames schema 2 already wrote, so no structure on disk moves.
    """
    if CONSENSUS_DRAW_AXIS.get(backend, "seed") == "model":
        return [f"model{k}" for k in range(1, n_af2_models_in(cfg) + 1)]
    return [f"seed{s}" for s in fold_seeds_for(backend, cfg, target_seqs, binder_seq)]


def fold_seeds_for(backend: str, cfg: dict, target_seqs: list[str], binder_seq: str) -> list[int]:
    """The seeds one binder is folded at by *backend*, in order.

    Module level rather than a closure inside :func:`score_binders` because a
    cache is keyed by seed VALUE, so anything that writes an entry -- the scorer,
    or a migration adopting folds made by another mechanism -- has to agree with
    it exactly. Two copies of this rule is two ways for the same fold to be
    filed under two keys, which reads as a miss and refolds.

    A pinned ``cfg.seed`` means exactly one fold, however many are asked for: it
    names a specific sample, and repeating it would be the same fold counted
    twice. Otherwise a sampler wants several draws and a deterministic folder
    wants one, however many a sampler beside it asks for -- the same rule
    ``_fold_seeds`` applies on the monomer side, for the same reason: its
    ensemble comes from its parameter sets instead.

    The first seed is the unindexed :func:`deterministic_seed`, so a single-seed
    entry written before seeds were a list is still found.
    """
    from proteinfoundation.metrics.seeding import deterministic_seeds

    pinned = cfg.get("seed")
    if pinned is not None:
        return [int(pinned)]
    n_seeds = max(1, int(cfg.get("n_seeds", cfg.get("n_esmfold2_seeds", 1)))) if backend == "esmfold2" else 1
    return deterministic_seeds(*target_seqs, binder_seq, count=n_seeds)


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
    context: "ComplexFoldContext | None" = None,
    reduce: bool = True,
) -> list[dict[str, float | str]]:
    """Advisory metrics for each binder against the target, in input order.

    *reduce* false returns ``{metric: [value per draw]}`` instead of a pooled
    scalar per metric, so the artifact carries the draws and the reduction can be
    re-asked in analyze. See :func:`draws_by_metric`.

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

    fingerprint = consensus_fingerprint(backend, cfg, target_seqs)
    # Only ask about the derivation when something is actually read off the
    # structures. With nothing registered there is no staleness that matters, and
    # asking would report every pre-split cache as stale on every run.
    wanted_suffixes = consensus_derived_suffixes(derive_tmol)
    derivation = consensus_derivation_fingerprint(derive_tmol) if wanted_suffixes else None

    def draws_for(seq: str) -> list[str]:
        return draw_ids_for(backend, cfg, target_seqs, seq)

    def first_seed_for(seq: str) -> int:
        # Schema 1 adoption only: it keyed by binder alone, and the seed that
        # must have produced such an entry is recoverable. Draw-aware callers
        # use draws_for.
        return fold_seeds_for(backend, cfg, target_seqs, seq)[0]

    # per binder sequence: {draw id: metrics}
    scores: dict[str, dict[str, dict[str, float | str]]] = {}
    derivation_stale = False
    if cache_dir and reuse_cache:
        scores, derivation_stale = read_consensus_cache(
            cache_dir,
            backend,
            fingerprint,
            seed_for=first_seed_for,
            derivation=derivation,
            legacy_fingerprint_for=lambda n: legacy_count_fingerprint(backend, cfg, target_seqs, n),
        )
        if scores:
            # Rewritten under the current fingerprint so the reconstruction is
            # paid for once rather than on every run.
            write_consensus_cache(cache_dir, backend, fingerprint, scores, derivation=derivation)

    # Re-read the kept structures for metrics that are read off them, rather than
    # refolding. Runs when the derivation changed, when an entry simply lacks a
    # derived key -- an entry cached before its structure existed heals itself
    # once the PDB is there, instead of staying blank forever behind a derivation
    # fingerprint that already matches -- and when a cutoff this run asks for is
    # one the entry has not been scored at.
    def _derive_into_scores(subject: dict, stale: bool) -> tuple[dict, int, int]:
        """Fill in what is read off the kept structures, re-reading not refolding.

        Returns the entries that changed, plus the counts worth reporting. Shared
        by the two callers that need it -- entries read from cache, and folds made
        moments ago -- because a fresh fold that skips this ships NaN in every
        derived column and only heals on the NEXT run. That is the failure
        refresh_monomer_derivation names on the monomer side: "the difference
        between a column being there and a campaign having to be evaluated twice
        to populate it".
        """
        changed: dict[str, dict[str, dict[str, float | str]]] = {}
        failed = 0
        # Entries whose PAE family the stored matrix could not refresh. Only
        # interesting when the derivation moved: their folder-reported values
        # then answer the question the cutoffs used to ask, and no file on disk
        # can produce the new answer without predicting the complex again.
        unrefreshable_pae = 0
        for seq, by_draw in subject.items():
            for draw, metrics in by_draw.items():
                complete = all(k in metrics for k in wanted_suffixes) and not missing_pae_cutoffs(metrics)
                if not stale and complete:
                    continue
                # This draw's own structure, never "the" structure for the
                # sequence. That distinction is the whole point of the axis: a
                # PAE family re-read off one model while the numbers beside it
                # were a mean over five moved avg_ipSAE by up to 0.55 on EFNB3.
                pdb = metrics.get("pdb_path") or existing_advisory_structure(cache_dir, backend, seq, draw)
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
                        have={} if stale else metrics,
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
                    changed.setdefault(seq, {})[draw] = metrics
        return changed, failed, unrefreshable_pae

    if scores and cache_dir:
        rederived, failed, unrefreshable_pae = _derive_into_scores(scores, derivation_stale)
        if rederived:
            write_consensus_cache(cache_dir, backend, fingerprint, rederived, derivation=derivation)
            logger.info(
                f"Advisory backend '{backend}' re-derived metrics for "
                f"{sum(len(v) for v in rederived.values())} (sequence, draw) structures without refolding"
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
    def _needs_structure(seq: str, draw: str) -> bool:
        if not (cache_dir and keep_structures):
            return False
        return existing_advisory_structure(cache_dir, backend, seq, draw) is None

    # One unit of work is a (sequence, draw) pair, so adding a draw folds only
    # what is new rather than everything for that sequence.
    pending = [
        (seq, draw)
        for seq in dict.fromkeys(binder_seqs)
        if seq
        for draw in draws_for(seq)
        if draw not in scores.get(seq, {}) or _needs_structure(seq, draw)
    ]
    if pending:
        scorer = CONSENSUS_BACKENDS[backend]
        fresh: dict[str, dict[str, dict[str, float | str]]] = {}

        def _keep(metrics: dict) -> dict:
            usable = {k: float(v) for k, v in metrics.items() if k in CONSENSUS_METRIC_SUFFIXES and v == v}
            if any(k.endswith("ipSAE") or "ipSAE_" in k for k in usable):
                # Which distances the folder just scored at, recorded the same
                # way a re-derivation records them -- so a later round asking for
                # one of these reuses it instead of reading the matrix back.
                usable[PAE_CUTOFF_KEY] = {suffix: float(cutoff) for cutoff, suffix in IPSAE_CUTOFFS}
            if usable and metrics.get("pdb_path"):
                # pdb_path rides along in the same entry; it is not a metric, so
                # column emission filters on CONSENSUS_METRIC_SUFFIXES and picks
                # it up explicitly. Entries written before structures were kept
                # simply lack the key.
                usable["pdb_path"] = metrics["pdb_path"]
            return usable

        # Grouped by sequence, because a folder may answer for several draws in
        # one call: AF2 predicts with every parameter set per invocation, so
        # asking it once per draw would do five times the work to keep four
        # answers it already had. A scorer says so by returning {"draws": {...}};
        # one that returns a flat dict answered for the draw it was asked about.
        by_sequence: dict[str, list[str]] = {}
        for seq, draw in pending:
            by_sequence.setdefault(seq, []).append(draw)
        for seq, wanted_draws in by_sequence.items():
            remaining = list(wanted_draws)
            while remaining:
                draw = remaining.pop(0)
                out_pdb = (
                    advisory_structure_path(cache_dir, backend, seq, draw)
                    if (cache_dir and keep_structures)
                    else None
                )

                def out_path_for(other: str, _seq: str = seq) -> str | None:
                    """Where a sibling draw of this same binder should be written.

                    A multi-draw folder needs one path per structure it produces,
                    or four of its five land nowhere and _needs_structure asks for
                    them again on every run -- the indefinite-refold failure
                    AdvisoryStructureWriteError exists to make loud.
                    """
                    if not (cache_dir and keep_structures):
                        return None
                    return advisory_structure_path(cache_dir, backend, _seq, other)

                try:
                    produced = scorer(target_seqs, seq, cfg, out_pdb, draw, context, out_path_for)
                except AdvisoryStructureWriteError:
                    # Systematic, not per-design: the next binder writes to the same
                    # kind of path and fails the same way. Tolerating it here is what
                    # made an unwriteable structure look like a survivable hiccup.
                    raise
                except Exception as exc:
                    logger.warning(f"Advisory backend '{backend}' failed on a {len(seq)}-residue binder: {exc}")
                    continue
                answered = produced.get("draws") if isinstance(produced.get("draws"), dict) else {draw: produced}
                for answered_draw, metrics in answered.items():
                    usable = _keep(metrics)
                    if usable:
                        fresh.setdefault(seq, {})[str(answered_draw)] = usable
                remaining = [d for d in remaining if d not in answered]
        # Read off the structures these folds just wrote, before they are cached
        # or returned. The scorer reports what the FOLDER knows -- pTM, PAE, the
        # ipSAE family -- and `usable` above keeps only those; everything read off
        # the structure (buried area, shape complementarity, secondary structure,
        # geometry against the design) arrives here absent. Deriving only in the
        # cached-entry pass above meant a fresh fold shipped NaN in all of those
        # and healed on the NEXT run, so a campaign had to be evaluated twice to
        # fill columns whose structures were already on disk the first time.
        if cache_dir:
            _derive_into_scores(fresh, stale=False)
        for seq, by_draw in fresh.items():
            scores.setdefault(seq, {}).update(by_draw)
        if cache_dir and fresh:
            write_consensus_cache(cache_dir, backend, fingerprint, fresh, derivation=derivation)
        folded = sum(len(v) for v in fresh.values())
        logger.info(f"Advisory backend '{backend}' scored {folded}/{len(pending)} (sequence, draw) folds")

    shape = reduce_over_draws if reduce else draws_by_metric
    return [shape(scores.get(seq, {})) for seq in binder_seqs]
