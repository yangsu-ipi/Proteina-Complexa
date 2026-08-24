"""One place to load and cache ESMFold2 checkpoints.

Two callers need ESMFold2 for different jobs, and they want different weights:

* monomer/binder refolding (``folding_models.run_esmfold2``) folds one chain,
  single-sequence, and uses the Fast checkpoint;
* advisory complex refolding (``consensus_folding``) folds target+binder and
  uses the full checkpoint, optionally with a target MSA.

They share this loader so a process that does both keeps one copy of each
checkpoint rather than one per call site. Note that enabling both does mean two
checkpoints resident at once -- deliberate, since they are different models, but
worth knowing before running them together on a small card.

Caching matters more than it looks: ``ESMFold2Model.from_pretrained`` rebuilds
the model and reloads weights every call, so a per-design call site without a
cache pays that per design. That is the mistake ``run_esmfold`` still makes with
``facebook/esmfold_v1``.
"""

import hashlib
import os

from loguru import logger

# Checkpoint defaults, overridable per call or by environment.
#
# Fast for single-chain single-sequence work, full for complexes where a target
# MSA is available; this split follows the fork's own deploy scripts, which use
# ESMFold2-Experimental-Fast-Cutoff2025 as the fast "inversion" model and
# ESMFold2-Experimental-Cutoff2025 as the "critic" that scores.
DEFAULT_ESMFOLD2_MONOMER_MODEL = "biohub/ESMFold2-Experimental-Fast-Cutoff2025"
DEFAULT_ESMFOLD2_COMPLEX_MODEL = "biohub/ESMFold2-Experimental-Cutoff2025"

MONOMER_MODEL_ENV = "ESMFOLD2_MONOMER_MODEL"
COMPLEX_MODEL_ENV = "ESMFOLD2_COMPLEX_MODEL"

_MODELS: dict[tuple[str, bool], object] = {}


def monomer_model_id() -> str:
    return os.environ.get(MONOMER_MODEL_ENV) or DEFAULT_ESMFOLD2_MONOMER_MODEL


def complex_model_id() -> str:
    return os.environ.get(COMPLEX_MODEL_ENV) or DEFAULT_ESMFOLD2_COMPLEX_MODEL


def load_esmfold2(model_id: str, cuda: bool = True):
    """Load an ESMFold2 checkpoint, once per (id, device) per process.

    ``ESMFold2Model.from_pretrained`` dispatches to ESMFold2ExperimentalModel by
    itself when the config says so, so the Experimental checkpoints load through
    this same call.
    """
    key = (model_id, bool(cuda))
    if key in _MODELS:
        return _MODELS[key]

    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    logger.info(f"Loading ESMFold2 checkpoint {model_id} (cuda={cuda})")
    model = ESMFold2Model.from_pretrained(model_id)
    if cuda:
        model = model.cuda()
    _MODELS[key] = model.eval()
    return _MODELS[key]


def clear_esmfold2_cache() -> None:
    """Release cached checkpoints (tests, or to free VRAM between stages)."""
    n = len(_MODELS)
    _MODELS.clear()
    if n:
        logger.info(f"Released {n} cached ESMFold2 checkpoint(s)")


# Bump whenever deterministic_seed changes -- adding a component, reordering the
# parts, changing the hash. Every seed changes when it does, and therefore every
# structure and every number, while nothing else in a cache fingerprint moves. It
# goes into both cache fingerprints so that a derivation change invalidates
# instead of silently serving structures drawn from the old seeds beside freshly
# drawn ones.
SEED_DERIVATION_VERSION = 1


def deterministic_seed(*parts: str) -> int:
    """A stable seed derived from the inputs a fold depends on.

    ESMFold2 is a diffusion sampler: unseeded, folding the same sequence twice
    gives different structures and therefore different scRMSD and confidence.
    That makes results depend on whether a run was resumed -- a cache hit returns
    the first run's numbers while a fresh run draws new samples -- and it makes
    the advisory per-sequence distributions incomparable between runs.

    Deriving the seed from the fold's own inputs makes the fold a pure function
    of them, so a cached value and a recomputed one agree. ``_seed_context`` in
    the fork saves and restores python/numpy/torch RNG state around the call, so
    seeding here perturbs nothing upstream.

    Returns a value in the numpy seed range, which is narrower than torch's.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**32 - 1)
