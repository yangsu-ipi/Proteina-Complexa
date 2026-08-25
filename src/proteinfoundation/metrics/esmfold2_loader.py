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

import os

from loguru import logger

# Re-exported: this module was their original home, and callers import them from
# here. They are shared with ProteinMPNN seeding now, so they live in a module
# that does not drag an ESMFold2 checkpoint loader in with them.
from proteinfoundation.metrics.seeding import (  # noqa: F401
    SEED_DERIVATION_VERSION,
    deterministic_seed,
)

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
