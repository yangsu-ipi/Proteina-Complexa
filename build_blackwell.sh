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
#   optional: WITH_ESMFOLD2=1 [ESM_SRC=...] adds ESMC/ESMFold2 -- see [6e]. Off by default because
#   it replaces PyPI transformers with Biohub's fork for the whole env.
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

# [6e] OPTIONAL: ESMC + ESMFold2. OFF unless WITH_ESMFOLD2=1, because it swaps PyPI transformers
#   for Biohub's fork -- a non-PyPI dependency for the whole env, and not upstreamable. Nothing in
#   the default configs needs it: apo/monomer folding defaults to `esmfold`, complex folding to
#   colabdesign, consensus_backends is empty, and ESM perplexity defaults to facebook/esm2.
#   Enable it for advisory complex refolding (metric.consensus_backends=[esmfold2]),
#   apo_folding_models=[esmfold2], or an ESMC perplexity model.
#
#   Two packages, from two places. The fork carries ESMFold2Model and ESMCForMaskedLM (they exist
#   nowhere else); the `esm` source tree carries ESMFold2InputBuilder, pae_interaction and MSA,
#   which the fork does not. Both are required -- installing only one leaves import errors that
#   surface at first fold, not at build.
#
#   usage: WITH_ESMFOLD2=1 [ESM_SRC=/path/to/esmfold2] bash build_blackwell.sh
if [ -n "${WITH_ESMFOLD2:-}" ]; then
  ESM_SRC="${ESM_SRC:-$HOME/projects/esmfold2}"
  [ -f "$ESM_SRC/pyproject.toml" ] || { echo "  [6e] no pyproject.toml under ESM_SRC=$ESM_SRC"; exit 1; }
  # Freeze what [1]/[6]/[6b] fought for, and install everything below under it. These deps reach
  # numpy through rdkit and scipy through scikit-learn, and torch through esm's own torch>=2.2.0 --
  # which on PyPI is a cu126 wheel that would silently replace the cu128 build sm_120 needs.
  # A constraint file makes pip refuse rather than resolve. transformers is excluded on purpose:
  # it is the one thing meant to move.
  CONS="$ENV_DIR/esmfold2-constraints.txt"
  "$PY" - > "$CONS" <<'PYEOF'
import importlib.metadata as md
for pkg in ["torch", "numpy", "scipy", "numba", "einops", "biotite", "jax", "jaxlib"]:
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
PYEOF
  # An empty or truncated constraint file would constrain nothing while looking like it did,
  # which is worse than not having one -- pip reports no error for a file with no matching lines.
  grep -q "^torch==" "$CONS" && grep -q "^numpy==" "$CONS" || {
    echo "  [6e] constraint file $CONS is missing torch/numpy -- refusing to install unconstrained"; exit 1; }
  echo "  [6e] holding: $(tr '\n' ' ' < "$CONS")"
  # The fork, at the commit their pixi.lock resolves -- their pyproject says @main, which floats,
  # and pip does not read pixi.lock. Downgrades transformers 5.x -> 4.57.6; pyproject's
  # >=4.57,<6 admits it, and the fork keeps models/esm/ so ESM2 + ESMFold still work.
  "$PIP" install -c "$CONS" \
    "transformers @ git+https://github.com/Biohub/transformers.git@f9a5a374be135f63b3019c1cefb91ea9e2d27e10"
  # `esm` itself: a COPY install, not editable -- the standalone ESMFold2 checkout stays
  # independent of this env. --no-deps because its transformers pin is @main (would undo the line
  # above) and its torch>=2.2.0 would re-resolve torch off cu128.
  "$PIP" install --no-deps "$ESM_SRC"
  # ...which means every runtime dep of `esm` is now ours to install. This is the set its pyproject
  # declares minus what the env already has (einops/biotite/biopython/scikit-learn/pandas) and minus
  # cuequivariance: cuequivariance_ops_torch is imported inside esm/models/esmfold2/fast.py, reached
  # only via enable_fast_inference(), which this repo never calls. Add it later for the ~4.8x trunk
  # speedup at L~=768 (Linux-only wheels).
  "$PIP" install -c "$CONS" accelerate freesasa rdkit msgpack-numpy brotli attrs cloudpathlib \
    httpx tenacity zstd ipywidgets ipython py3dmol pydssp boto3 pygtrie dna_features_viewer
  # Prove the exact symbols this repo imports, not just that the packages exist. esm_eval,
  # folding_models and consensus_folding each reach a different one of these, and a missing
  # re-export would otherwise surface mid-campaign.
  CONS="$CONS" "$PY" - <<'PYEOF'
import os
import sys
import importlib.metadata as md
bad = []
try:
    import transformers
    if not transformers.__version__.startswith("4.57"):
        bad.append(f"transformers {transformers.__version__} is not the 4.57.6 fork")
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # noqa: F401
    # ESM2 + ESMFold must survive the downgrade -- the fork keeps models/esm/, but check it.
    from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmForProteinFolding  # noqa: F401
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37  # noqa: F401
    from transformers.models.esm.openfold_utils.protein import to_pdb  # noqa: F401
except Exception as e:
    bad.append(f"transformers fork: {type(e).__name__}: {e}")
try:
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput  # noqa: F401
    from esm.models.esmfold2.interface_metrics import pae_interaction  # noqa: F401
    from esm.utils.msa import MSA  # noqa: F401
except Exception as e:
    bad.append(f"esm package: {type(e).__name__}: {e}")
# The constraint file should have held. Checked against the file itself rather than against
# versions repeated here: those would drift the moment a pin above changes, and then this would
# fail for the wrong reason. "nothing moved since we froze it" is the actual claim.
for line in open(os.environ["CONS"]):
    line = line.strip()
    if not line or "==" not in line:
        continue
    pkg, want = line.split("==", 1)
    try:
        got = md.version(pkg)
    except md.PackageNotFoundError:
        bad.append(f"{pkg} disappeared (was {want})")
        continue
    if got != want:
        bad.append(f"{pkg} moved {want} -> {got} despite the constraint")
if bad:
    print("  [6e] FAILED:", *bad, sep="\n    "); sys.exit(1)
print(f"  [6e] ESMFold2 imports OK (transformers {transformers.__version__}, esm {md.version('esm')})")

# ESMC is reported, not required. The repo loads it through AutoModelForMaskedLM
# (esm_eval.py:542) rather than a fixed class path, esm_model defaults to
# facebook/esm2, and the esm package's own ESMC builders leave parameters on the
# meta device -- so esmc_pkg is not a fallback. Failing the build over a scorer
# nothing is configured to use would be the wrong trade.
try:
    import importlib
    importlib.import_module("transformers.models.esmc")
    print("  [6e] ESMC available via AutoModelForMaskedLM (set metric.esm_model to a transformers-format ESMC repo)")
except Exception as e:
    print(f"  [6e] note: ESMC not importable ({type(e).__name__}) -- ESM2 perplexity unaffected")
PYEOF
  cat <<'EOF'
  [6e] Weights are GATED HF repos -- set HF_TOKEN and accept the licences for
       biohub/ESMFold2-Experimental-Fast-Cutoff2025 (monomer/apo) and
       biohub/ESMFold2-Experimental-Cutoff2025 (complex, MSA-capable).
       Point HF_HOME at the hub cache root, NOT a snapshot directory.
       No ESMFold2 path in this repo has been run against real weights yet.
EOF
fi

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
  ESMC/ESMFold2 are NOT installed unless you pass WITH_ESMFOLD2=1 -- see [6e]. The default configs
  do not need them: apo refolding uses plain ESMFold, complex refolding uses colabdesign.
EOF
