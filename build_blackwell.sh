#!/usr/bin/env bash
# Build Proteina-Complexa for NVIDIA Blackwell (sm_120, e.g. RTX PRO 6000).
# proteinfoundation needs Python>=3.12; torch cu128 for sm_120; tmol builds from source.
# The dep set is over-constrained -> the reconcile in [6] goes LAST (order matters).
# No source changes required. Verified 2026-07 on IPI gnode2.
# Despite the name this is NOT sm_120-only: torch cu128 is built for 7.5;8.0;8.6;9.0;10.0;12.0+PTX,
#   so it also covers Turing/Ampere/Hopper, and Ada sm_89 runs the sm_86 cubin (CUDA binary compat).
#   CUDA 12.8 dropped sm_50-sm_70, so Volta and older cannot use this recipe at all.
# EVERY VERSION HERE IS PINNED ON PURPOSE. Unpinned installs of flax/chex/dm-haiku silently upgrade
#   jax past 0.10.2 and break the source patches this branch carries -- see [6b].
#   usage: bash build_blackwell.sh [ENV_DIR]     (default: ./.venv-blackwell)
set -eo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="${1:-$REPO/.venv-blackwell}"
command -v mamba >/dev/null 2>&1 && CONDA=mamba || CONDA=conda
[ -x "$ENV_DIR/bin/python" ] || "$CONDA" create -y -p "$ENV_DIR" python=3.12   # proteinfoundation needs >=3.12
PY="$ENV_DIR/bin/python"; PIP="$ENV_DIR/bin/pip"

"$PIP" install --upgrade pip
# [1] torch cu128 for Blackwell (upstream uses 2.7.0+cu126):
"$PIP" install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# [2] PyG extensions (prebuilt pt27cu128 wheels):
"$PIP" install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
# [3] tmol from source (builds on py3.12/cu128):
"$PIP" install "git+https://github.com/uw-ipd/tmol.git@d8a6f7f9"
# [4] graphein (--no-deps, like upstream) + atomworks:
"$PIP" install graphein==1.7.7 --no-deps
"$PIP" install "atomworks[ml,openbabel,dev]"
# [5] the package (editable) — pulls proteinfoundation + downgrades biotite/scipy/numpy (see [6]):
( cd "$REPO" && "$PIP" install -e . )
# [6] RECONCILE (order-sensitive, do LAST). Each line repairs something [5] broke:
#   biotite  atomworks needs >=1.4; 1.6.0 adds the ligand support the pipeline uses. atomworks pins
#            ==1.4.0 exactly, so pip warns here -- deliberate.
#   numpy    --no-deps because [5] dragged numpy below scipy 1.12's own numpy<1.29 ceiling.
#   scipy    [5] enforces pyproject's scipy==1.12.0. This line used to be missing, which left scipy
#            1.12.0 sitting next to numpy 2.4.6 -- a pairing scipy's metadata forbids and its C ABI
#            cannot survive, since numpy-2 support first landed in scipy 1.13.0. Only jax's
#            transitive scipy>=1.14 in [6b] repaired it, AFTER the verification below had already
#            run against a broken env. atomworks independently needs >=1.13.1.
#   numba    reached via tmol -> sparse -> numba, and its numpy ceiling must clear the pin above:
#            0.62/0.63 cap at <2.4 and refuse to import against 2.4.6. 0.67.0 allows <2.6.
"$PIP" install biotite==1.6.0
"$PIP" install --no-deps numpy==2.4.6
"$PIP" install "scipy>=1.14" "numba==0.67.0"

# `import proteinfoundation` on its own proves very little -- src/proteinfoundation/__init__.py is a
# few lines of commented-out imports. scipy and numba are the real canaries: a numpy-1-ABI scipy, or a
# numba whose ceiling excludes the pinned numpy, raises HERE instead of on first real use.
"$PY" - <<'PYEOF'
import importlib, sys
bad = []
for m in ["proteinfoundation","atomworks","tmol","graphein","biotite","torch","numpy","scipy","numba"]:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}: {e}")
import numpy, scipy, numba
if tuple(map(int, scipy.__version__.split(".")[:2])) < (1, 13):
    bad.append(f"scipy {scipy.__version__} predates numpy-2 support (needs >=1.13)")
print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  numba {numba.__version__}")
if bad:
    print("FAILED:", *bad, sep="\n  "); sys.exit(1)
print("Proteina-Complexa (Blackwell): all core imports OK")
PYEOF

# [6b] JAX + colabdesign AF2 reward stack — required even for GENERATION (search/__init__.py imports
#   colabdesign at module load) AND for the reward-guided search. torch cu128 + jax 0.10 coexist ONLY
#   with cudnn 9.24: jax needs it, torch cu128 runs fine on it (torch's ==9.7.1.26 pin is stricter than
#   reality). VALIDATED: reward-guided binder design produces real AF2 scores on Blackwell.
#   PINNED, and in ONE resolve so pip sees every constraint at once. Unpinned, `pip install flax`
#   alone drags jax off 0.10.2: current flax declares jax>=0.11.1 in its CORE deps. Newest versions
#   whose declared jax floor still admits 0.10.2 -- flax 0.12.0 (>=0.7.1), dm-haiku 0.0.16 (0.0.17
#   targets 0.11), chex 0.1.92 (>=0.7.0), optax 0.2.8 (>=0.5.3). optax is REQUIRED, not optional:
#   community_models/colabdesign/shared/model.py imports it at module load.
"$PIP" install "jax[cuda12]==0.10.2" "nvidia-cudnn-cu12==9.24.0.43" \
               "flax==0.12.0" "dm-haiku==0.0.16" "chex==0.1.92" "optax==0.2.8"
# colabdesign is vendored here; its bundled AlphaFold + this repo's AF2 reward carry jax-0.10 fixes
# committed on this branch: clip min=/max= (af/loss.py, af/alphafold/model/modules{,_multimer}.py),
# a jax.tree_*/jax.util compat shim (community_models/colabdesign/__init__.py), and
# jax.clear_backends -> jax.clear_caches (src/proteinfoundation/rewards/alphafold2_reward.py).
( cd "$REPO/community_models/colabdesign" && "$PIP" install -e . )
# The reward model's exceptions are triple-hidden at runtime (CompositeRewardModel try/except ->
# warnings.warn, then the CLI's -W ignore), so prove the AF2 entry points import HERE.
"$PY" - <<'PYEOF'
import sys, jax, haiku
bad = []
if jax.__version__ != "0.10.2":
    bad.append(f"jax drifted to {jax.__version__} -- the source patches on this branch target 0.10.x")
from colabdesign import mk_afdesign_model
import proteinfoundation.rewards.alphafold2_reward
print(f"  colabdesign + AF2 reward import OK (jax {jax.__version__}, haiku {haiku.__version__})")
if bad:
    print("FAILED:", *bad, sep="\n  "); sys.exit(1)
PYEOF

# [6c] AF2 params for the reward model — PUBLIC (Google storage, no key). Needs the MULTIMER set
#   (2022-12-06) for binder-complex folding; our older 2021 monomer store is NOT sufficient.
AF2="$REPO/community_models/ckpts/AF2"; mkdir -p "$AF2/params"
if [ ! -f "$AF2/params/params_model_1_multimer_v3.npz" ]; then
  wget -qO "$AF2/af2.tar" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
  tar -xf "$AF2/af2.tar" -C "$AF2/params" && rm -f "$AF2/af2.tar"
fi
echo "  AF2 params: $(ls "$AF2/params" | grep -c npz) npz (set AF2_DIR=$AF2 in .env)"

# [6d] External tools for the analyze stage's diversity metrics (foldseek + mmseqs, via bioconda).
#   Point FOLDSEEK_EXEC / MMSEQS_EXEC (or the UV_* vars in .env) at these. Without them the pipeline
#   still completes but logs "Foldseek/MMseqs diversity failed" and skips clustering.
command -v mamba >/dev/null 2>&1 && "$CONDA" install -y -p "$ENV_DIR" -c conda-forge -c bioconda foldseek mmseqs2 \
  || echo "  (install foldseek + mmseqs2 into the env for diversity metrics)"
echo "  set in .env: UV_FOLDSEEK_EXEC=$ENV_DIR/bin/foldseek  UV_MMSEQS_EXEC=$ENV_DIR/bin/mmseqs"

# [7] Model checkpoints — PUBLIC on NGC (no key). Protein-binder pair (~7 GB); validated loadable.
CK="$REPO/ckpts"; mkdir -p "$CK"
MOD="https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa/1.0/files?redirect=true&path="
[ -f "$CK/complexa.ckpt" ]    || wget -qO "$CK/complexa.ckpt"    "${MOD}complexa.ckpt"
[ -f "$CK/complexa_ae.ckpt" ] || wget -qO "$CK/complexa_ae.ckpt" "${MOD}complexa_ae.ckpt"
echo "  checkpoints in $CK: $(du -h "$CK"/*.ckpt 2>/dev/null | cut -f1 | tr '\n' ' ')"

cat <<EOF
=== Proteina-Complexa (Blackwell) env + checkpoints ready.
  Checkpoints (PUBLIC NGC, no key) are in $CK; the binder pipeline config points ckpt_path there.
  A FULL binder-design run (complexa design configs/search_binder_local_pipeline.yaml) additionally
  needs: a target spec, the community reward/refolding models (ESM2 via a free HF token, AF2/RF3/Boltz2),
  and external tools (foldseek/mmseqs/dssp). See docs/INFERENCE.md. Generation-only uses just the
  checkpoints above.
EOF
