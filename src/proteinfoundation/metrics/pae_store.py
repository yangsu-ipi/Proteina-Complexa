"""Predicted aligned error, kept beside the structure it describes.

Why this exists. Everything in the PAE family -- i_pAE, pAE, min_ipAE and the
whole ipSAE set -- is computed from one L x L matrix the folder returns and
nothing on disk records. So the family is *folder-reported*: it rides in the
fold fingerprint, and adding a member, or changing the distance cutoff an
existing one uses, can only be answered by folding every complex again. That bill
came due once already -- expanding the advisory metric list from 6 entries to 14
invalidated ~20,000 cached ESMFold2 complex folds on CBLN1 and ~5,900 on EFNB3,
for numbers that were sitting in memory when the structure was written and were
then thrown away.

A stored matrix turns all of that into a re-read. ipSAE at 12 A instead of 10 is
then arithmetic over a file, not a diffusion sampler over a campaign.

Format
------
``<structure>.pae.npz``, holding the matrix quantised to ``uint8`` at
:data:`PAE_QUANT_STEP` A per level, plus a JSON metadata blob.

*uint8, not float32.* Every backend here bins PAE at 0.5 A -- AF2's head has 64
bins over 0-31.5, ESMFold2's ``_pae_bins`` the same -- and the reported value is
a weighted mean over those bin centres. A 1/8 A step is four times finer than the
resolution the number is produced at, so the largest possible round-trip error,
0.0625 A, is well under the model's own. It is a quarter the size of float32 and
half of float16.

*Angstroms, not the 0-1 scale the columns carry.* ipSAE compares PAE against a
cutoff in Angstroms and computes a d0 from chain lengths in Angstroms; storing
the divided form would make every reader undo it, and one reader forgetting is
the 31x mismatch this codebase has already had to correct once.

*The full matrix, not the interface block.* PAE is not symmetric -- ``pae[i][j]``
is the error at j when the prediction is aligned on i -- and ipSAE reads both
directions and takes their min and max. A symmetrised or cropped store could not
reproduce the columns it exists to reproduce.

*Chain lengths in the metadata.* The target/binder split is what makes an
interface metric an interface metric, and recovering it by reparsing the PDB is
both slower and a second definition of a fact the writer already knew.
"""

import json
import os

import numpy as np
from loguru import logger

PAE_STORE_SUFFIX = ".pae.npz"

# Angstroms per quantisation level, and the largest value representable in 255 of
# them. AF2 tops out at a 31.75 A bin centre and ESMFold2 at the same, so the cap
# is above anything either reports; a matrix that exceeds it is clipped and said
# so, rather than wrapping.
PAE_QUANT_STEP = 0.125
PAE_QUANT_MAX = 255 * PAE_QUANT_STEP  # 31.875

# Bumped if the on-disk layout changes in a way a reader must notice. The step is
# recorded per file, so changing it alone does not need this.
PAE_STORE_VERSION = 1


def pae_sidecar_path(structure_path: str) -> str:
    """Where one structure's PAE lives.

    Appended rather than substituted for the extension, because these files are
    named things like ``esm_1_seed7.pdb_esm_apo_mpnn`` and there is no extension
    to replace.
    """
    return f"{structure_path}{PAE_STORE_SUFFIX}"


def _as_array(pae) -> np.ndarray | None:
    if pae is None:
        return None
    array = np.asarray(pae.detach().cpu() if hasattr(pae, "detach") else pae, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not array.size:
        logger.warning(f"Ignoring a PAE of shape {array.shape}; expected a square L x L matrix")
        return None
    return array


def save_pae(
    structure_path: str,
    pae,
    chain_lengths: list[int] | None = None,
    backend: str | None = None,
    model: str | None = None,
    seed: int | None = None,
) -> str | None:
    """Write one structure's PAE beside it. Returns the path, or None.

    Never raises. A fold costs minutes and this costs milliseconds: a sidecar
    that cannot be written must not take the fold down with it, and an absent
    one already means "this fold predates the store", which every reader handles.

    *chain_lengths* are the chains in the order the structure was written, which
    for a binder complex is targets first and the binder last. With them a reader
    can split the matrix without opening the PDB.
    """
    array = _as_array(pae)
    if array is None:
        return None
    path = pae_sidecar_path(structure_path)
    try:
        over = float(array.max())
        if over > PAE_QUANT_MAX:
            logger.warning(
                f"PAE for {os.path.basename(structure_path)} reaches {over:.2f} A, above the "
                f"{PAE_QUANT_MAX} A this store represents; clipping"
            )
        quantised = np.clip(np.rint(array / PAE_QUANT_STEP), 0, 255).astype(np.uint8)
        meta = {
            "version": PAE_STORE_VERSION,
            "units": "angstrom",
            "step": PAE_QUANT_STEP,
            "length": int(array.shape[0]),
            "chain_lengths": [int(n) for n in chain_lengths] if chain_lengths else None,
            "backend": backend,
            "model": model,
            "seed": None if seed is None else int(seed),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as handle:
            np.savez_compressed(
                handle,
                pae=quantised,
                meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            )
        return path
    except Exception as exc:
        logger.warning(f"Could not store PAE for {structure_path}: {exc}")
        return None


def load_pae(structure_path: str) -> dict | None:
    """One structure's stored PAE as ``{"pae": (L, L) float32 in A, **metadata}``.

    None when there is no sidecar, which is what a fold from before this store
    looks like -- unmeasured, not zero. Never raises.
    """
    path = pae_sidecar_path(structure_path)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as loaded:
            meta = json.loads(bytes(loaded["meta"]).decode("utf-8"))
            step = float(meta.get("step") or PAE_QUANT_STEP)
            out = dict(meta)
            out["pae"] = loaded["pae"].astype(np.float32) * step
        return out
    except Exception as exc:
        logger.warning(f"Ignoring unusable PAE sidecar {path}: {exc}")
        return None


def stored_pae_bytes(structure_path: str) -> int:
    """Size of one sidecar, for anyone counting what a campaign will cost."""
    path = pae_sidecar_path(structure_path)
    return os.path.getsize(path) if os.path.exists(path) else 0
