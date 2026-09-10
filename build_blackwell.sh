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
#   ESMC/ESMFold2 are installed BY DEFAULT -- see [6e]. That puts the env on Biohub's transformers
#   fork rather than PyPI's, deliberately. Opt out with WITH_ESMFOLD2=0; override the source tree
#   with ESM_SRC=/path/to/esmfold2.
set -eo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="${1:-$REPO/.venv-blackwell}"
command -v mamba >/dev/null 2>&1 && CONDA=mamba || CONDA=conda
[ -x "$ENV_DIR/bin/python" ] || "$CONDA" create -y -p "$ENV_DIR" python=3.12   # proteinfoundation needs >=3.12
PY="$ENV_DIR/bin/python"; PIP="$ENV_DIR/bin/pip"

# [0] Preconditions. ESMC/ESMFold2 are on by default and are the one input this script cannot
#   fetch itself -- the packages are a source-only release, not on PyPI. Checked HERE rather than
#   at [6e] so a missing tree costs a second instead of arriving after torch, tmol and jax are in.
#   Not silently skipped: "on by default" that quietly turns itself off is worse than a clear stop.
WITH_ESMFOLD2="${WITH_ESMFOLD2:-1}"
ESM_SRC="${ESM_SRC:-/data/shared/esmfold2}"
if [ "$WITH_ESMFOLD2" != "0" ] && [ ! -f "$ESM_SRC/pyproject.toml" ]; then
  echo "ESMC/ESMFold2 source tree not found: $ESM_SRC/pyproject.toml"
  echo "  It is a source-only release (no PyPI), so this script cannot fetch it. Either point at it:"
  echo "    ESM_SRC=/path/to/esmfold2 ..."
  echo "  or build without it -- apo folding falls back to plain ESMFold, complex to colabdesign:"
  echo "    WITH_ESMFOLD2=0 ..."
  exit 1
fi

# [0b] conda-forge layer, placed before any pip so the two never fight over site-packages.
#   freesasa is a compiled Python extension and PyPI ships NO linux wheel for it, so pip would
#   compile the sdist on every build; conda-forge has a prebuilt linux-64 py3.12 binary. It has to
#   land before [5], because pyproject.toml declares freesasa and the editable install there
#   would otherwise build it from source before conda ever got a turn.
#   It belongs here and not in [6e]: utils/pr_alternative_utils.py uses it for the SASA
#   bioinformatics metrics, which have nothing to do with ESMFold2. It used to ride along in [6e]'s
#   pip line, so WITH_ESMFOLD2=0 silently dropped it -- silently because that import has a
#   Biopython fallback, which changes the numbers rather than raising.
#   Non-fatal: if conda cannot place it, [5] still installs it from the sdist.
"$CONDA" install -y -p "$ENV_DIR" -c conda-forge freesasa \
  || echo "  [0b] conda-forge freesasa failed -- [5] will build it from the sdist instead"

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
#   WITH the dev extra (ruff, pytest, ipdb). Not developer convenience: the guards that keep the
#   campaign templates honest ARE the test suite, so a box that runs campaigns needs to be able to
#   run them, at the versions pyproject declares. A box built before this carried ruff 0.8.3 --
#   below the >=0.15.0 floor -- and linted the templates with a years-older ruleset while reporting
#   success. The extra adds nothing heavy: measured on that box, `-e ".[dev]"` over `-e "."` is
#   ipdb alone, since ipython is already here via [6e].
( cd "$REPO" && "$PIP" install -e ".[dev]" )
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
for m in ["proteinfoundation","atomworks","tmol","graphein","biotite","torch","numpy","scipy","numba",
          # protein_interface is an abi3 Rust extension installed from a manylinux wheel by [5].
          # A wheel that resolves but will not load is otherwise only discovered when an
          # evaluation asks what an interface is, hours into a campaign.
          "protein_interface"]:
    try: importlib.import_module(m)
    except Exception as e: bad.append(f"{m}: {type(e).__name__}: {e}")
import numpy, scipy, numba
if tuple(map(int, scipy.__version__.split(".")[:2])) < (1, 13):
    bad.append(f"scipy {scipy.__version__} predates numpy-2 support (needs >=1.13)")
# The extension is pinned because its atomic radii are compiled in and unreadable from
# metadata; a silent version drift would move every shape complementarity value.
try:
    from importlib.metadata import version
    if version("protein-interface") != "0.1.3":
        bad.append(f"protein-interface {version('protein-interface')} is not the pinned 0.1.3")
except Exception as e:
    bad.append(f"protein-interface version unreadable: {e}")
print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  numba {numba.__version__}")
if bad:
    print("FAILED:", *bad, sep="\n  "); sys.exit(1)
print("Proteina-Complexa (Blackwell): all core imports OK")
PYEOF

# [6b] JAX + colabdesign AF2 reward stack — required even for GENERATION (search/__init__.py imports
#   colabdesign at module load) AND for the reward-guided search. torch cu128 + jax 0.10 coexist ONLY
#   with cudnn 9.24: jax needs it, torch cu128 runs fine on it (torch's ==9.7.1.26 pin is stricter than
#   reality). VALIDATED: reward-guided binder design produces real AF2 scores on Blackwell.
#   pip WARNS about the torch conflict on this line. Leave it. Both halves were measured on the
#   CBLN1 A100 box: torch reports cudnn 92400 and runs a conv, and jax compiles one too. Downgrading
#   to satisfy the warning is what broke that box -- jaxlib 0.10.2 is built against 9.8.0 and needs
#   the minor equal or higher, so torch's 9.7.1.26 misses by one and every jax GPU compile dies with
#   `RET_CHECK ... dnn_support != nullptr`, naming neither cudnn nor this file.
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

# [6c] Community-model weights — both PUBLIC, no key. AF2 params for the reward model, and
#   the MPNN weights the redesign step inverse-folds with.
#   AF2 needs the MULTIMER set (2022-12-06) for binder-complex folding; our older 2021
#   monomer store is NOT sufficient.
AF2="$REPO/community_models/ckpts/AF2"; mkdir -p "$AF2/params"
if [ ! -f "$AF2/params/params_model_1_multimer_v3.npz" ]; then
  wget -qO "$AF2/af2.tar" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
  tar -xf "$AF2/af2.tar" -C "$AF2/params" && rm -f "$AF2/af2.tar"
fi
echo "  AF2 params: $(ls "$AF2/params" | grep -c npz) npz (set AF2_DIR=$AF2 in .env)"

# MPNN weights (files.ipd.uw.edu). The binder pipeline's redesign step needs these and
#   nothing here fetched them: a build_blackwell-only install had no
#   community_models/LigandMPNN/model_params at all, so a campaign reached
#   `missing soluble ProteinMPNN checkpoint` -- after generation had finished.
#   `complexa download --ligandmpnn` does it, but that wants a working .env, which is a
#   later step than this one and not one this script performs.
#
#   Through the vendored get_model_params.sh rather than a list of URLs here: the URLs are
#   upstream's, and a second copy of them is a second thing to go stale. It fetches all 15
#   (118 MB) when only two are named in code -- ligandmpnn_v_32_010_25.pt at
#   inverse_folding_models.py:413 and solublempnn_v_48_020.pt at :526 -- so the guard is on
#   those two, which are what a run actually opens.
MPNN="$REPO/community_models/LigandMPNN/model_params"
if [ ! -f "$MPNN/solublempnn_v_48_020.pt" ] || [ ! -f "$MPNN/ligandmpnn_v_32_010_25.pt" ]; then
  ( cd "$REPO/community_models/LigandMPNN" && bash get_model_params.sh ./model_params )
fi
# get_model_params.sh is `wget -q -O` with no error check, so a network failure leaves a
#   truncated or empty file in place. That matters because the other downloader's skip test
#   is "exists and is non-empty" (complexa-setup/reference/downloads.md), which would read a
#   partial checkpoint as installed forever. Both files are torch zip archives -- measured,
#   they start `PK` -- so reading the central directory proves a complete one.
"$PY" - "$MPNN" <<'MPNNEOF'
import sys
import zipfile
from pathlib import Path

root = Path(sys.argv[1])
bad = []
for name in ("solublempnn_v_48_020.pt", "ligandmpnn_v_32_010_25.pt"):
    path = root / name
    if not path.is_file():
        bad.append(f"{name} missing")
        continue
    try:
        zipfile.ZipFile(path).namelist()
    except Exception as exc:
        bad.append(f"{name} is not a complete archive ({type(exc).__name__}); delete it and re-run")
if bad:
    print("FAILED:", *bad, sep="\n  ")
    sys.exit(1)
weights = sorted(root.glob("*.pt"))
size = sum(p.stat().st_size for p in weights) / 2**20
print(f"  MPNN weights: {len(weights)} .pt, {size:.0f} MiB in {root}")
MPNNEOF

# [6d] External tools for the analyze stage's diversity metrics (foldseek + mmseqs, via bioconda).
#   Point FOLDSEEK_EXEC / MMSEQS_EXEC (or the UV_* vars in .env) at these. Without them the pipeline
#   still completes but logs "Foldseek/MMseqs diversity failed" and skips clustering.
#   $CONDA is already mamba-or-conda from the top of this script. This was gated on
#   `command -v mamba`, which skipped the install outright on a conda-only box -- and said so in the
#   voice of advice, so a machine that never got the tools read the same as one where they failed.
"$CONDA" install -y -p "$ENV_DIR" -c conda-forge -c bioconda foldseek mmseqs2 \
  || echo "  WARNING: foldseek + mmseqs2 install failed -- analyze will skip diversity clustering"
echo "  set in .env: UV_FOLDSEEK_EXEC=$ENV_DIR/bin/foldseek  UV_MMSEQS_EXEC=$ENV_DIR/bin/mmseqs"

# [6e] ESMC + ESMFold2 -- ON by default (WITH_ESMFOLD2=0 to skip). This env therefore runs on
#   Biohub's transformers fork rather than PyPI's: a non-PyPI dependency for everything here, and
#   not upstreamable to proteininnovation/Proteina-Complexa. Accepted deliberately, because the
#   models are wanted for real designs -- advisory complex refolding
#   (metric.consensus_backends=[esmfold2]), apo_folding_models=[esmfold2], and ESMC perplexity.
#   The default configs still do not require them: apo folding falls back to `esmfold` and complex
#   folding to colabdesign, so WITH_ESMFOLD2=0 yields a working pipeline, just a narrower one.
#
#   Two packages, from two places. The fork carries ESMFold2Model and ESMCForMaskedLM (they exist
#   nowhere else); the `esm` source tree carries ESMFold2InputBuilder, pae_interaction and MSA,
#   which the fork does not. Both are required -- installing only one leaves import errors that
#   surface at first fold, not at build.
#
#   Plus colabfold, --no-deps and for one function, so a campaign can fetch the target MSA that
#   ESMFold2's complex folding optionally takes. See the note at its install line.
#
#   WITH_ESMFOLD2 and ESM_SRC are resolved and validated in [0].
if [ "$WITH_ESMFOLD2" != "0" ]; then
  # Freeze what [1]/[6]/[6b] fought for, and install everything below under it. These deps reach
  # numpy through rdkit and scipy through scikit-learn, and torch through esm's own torch>=2.2.0 --
  # which on PyPI is a cu126 wheel that would silently replace the cu128 build sm_120 needs.
  # A constraint file makes pip refuse rather than resolve. transformers is excluded on purpose:
  # it is the one thing meant to move.
  CONS="$ENV_DIR/esmfold2-constraints.txt"
  "$PY" - > "$CONS" <<'PYEOF'
import importlib.metadata as md
# nvidia-cudnn-cu12 is deliberately NOT frozen here, even though [6b] fought for 9.24.0.43 and
# jax needs exactly that. torch 2.7.1+cu128 pins `nvidia-cudnn-cu12==9.7.1.26` in its own
# metadata -- exactly, under a `platform_system == "Linux" and platform_machine == "x86_64"`
# marker. Constraining 9.24.0.43 sets an exact pin against an exact pin, so ANY resolve that
# contains torch becomes impossible rather than merely warned about, and the install below
# contains torch through accelerate and pydssp. pip does not say that: it backtracks through the
# entire release history of everything requested, building old sdists to read their metadata, for
# hours at 100% CPU, before finally reporting ResolutionImpossible. The marker means it cannot
# happen on macOS, only on the boxes this script is for.
# What actually holds cudnn at 9.24 is keeping torch's dependency set out of the resolve -- the
# --no-deps line below -- with EXPECT_CUDNN after it as the safety net.
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
  # accelerate and pydssp are the only two that require torch, and pulling torch into a resolve
  # pulls its `nvidia-cudnn-cu12==9.7.1.26` with it, which would downgrade the cudnn [6b]
  # installed and break every jax compile. --no-deps keeps that dependency set out of the resolve
  # entirely; the line after supplies what --no-deps then skipped. Their remaining requirements --
  # torch, numpy, einops, tqdm -- are already in this env, which is why they are not repeated.
  "$PIP" install -c "$CONS" --no-deps accelerate pydssp
  "$PIP" install -c "$CONS" huggingface_hub safetensors psutil pyyaml packaging
  "$PIP" install -c "$CONS" rdkit msgpack-numpy brotli attrs cloudpathlib \
    httpx tenacity zstd ipywidgets ipython py3dmol boto3 pygtrie dna_features_viewer
  # ColabFold, for ONE function: `colabfold.colabfold.run_mmseqs2`, which is how
  # scripts/prepare_target_msa.py fetches a target MSA from the public MMseqs2 server.
  # It belongs inside [6e] because consensus_folding.py is the only thing in this repo
  # that reads an a3m, so a WITH_ESMFOLD2=0 env has no use for it.
  #
  # --no-deps, and not for tidiness: colabfold caps `biopython <1.86` and this env is on
  # 1.88, so a plain install DOWNGRADES it -- measured, `Would install appdirs-1.4.4
  # biopython-1.85 colabfold-1.6.2 importlib_metadata-8.9.0 orjson-3.12.0`. Nothing here
  # requires >=1.86 (protein-interface's >=1.83 is the tightest floor), so pip check would
  # stay quiet while Bio.PDB moved three releases under the evaluation path that uses
  # PDBParser and Superimposer. --no-deps installs the wheel and nothing else.
  #
  # Safe because run_mmseqs2 lives in colabfold/colabfold.py, whose module-level imports are
  # requests, tqdm, numpy and matplotlib plus stdlib -- all present -- and whose `import jax`
  # sits in a try/except. colabfold/__init__.py is empty.
  #
  # NEVER add the [alphafold] or [alphafold-minus-jax] extra: it wants dm-tree>=0.1.9 against
  # pyproject's dm-tree==0.1.8, and alphafold-colabfold==2.3.18 beside the vendored
  # colabdesign that carries this branch's jax-0.10 patches.
  #
  # The four console scripts it installs (colabfold_batch, _relax, _search, _split_msas) will
  # all fail at import without that extra. Nothing calls them; they are collateral.
  "$PIP" install -c "$CONS" --no-deps colabfold==1.6.2
  # Then give back the core deps --no-deps skipped, same as the accelerate/pydssp pair above.
  # These three are absent from the env and depend on nothing that is pinned, so they are pure
  # additions; matplotlib, numpy, pandas, requests and tqdm already satisfy their ranges. Adding
  # them leaves `pip check` with exactly ONE new complaint -- the biopython cap -- which is the
  # deliberate one. Every other line in that output is accounted for somewhere in this script,
  # and three unexplained ones would erode that.
  #
  # importlib-metadata carries its bound explicitly: colabfold is installed but not part of
  # this resolve, so pip does not consult its requirements, and a bare name took 9.0.1 against
  # colabfold's <9.0.0 -- trading three pip-check lines for a different one. Observed, not
  # feared. appdirs and orjson have no upper bound worth restating.
  "$PIP" install -c "$CONS" appdirs "importlib-metadata>=8.6.1,<9" orjson
  # [6b]'s jax verification ran BEFORE this step, so nothing here has yet re-checked that jax
  # still works -- a check that runs before the thing that can break it. AF2 reward guidance is
  # used by generation, not just evaluation, so a jax broken here takes the whole pipeline down
  # at the first campaign rather than at build time.
  "$PY" - <<'PYEOF'
import importlib.metadata as md
import sys

EXPECT_CUDNN = "9.24.0.43"   # [6b]
bad = []
try:
    cudnn = md.version("nvidia-cudnn-cu12")
    if cudnn != EXPECT_CUDNN:
        # jaxlib 0.10.2 is built against cudnn 9.8.0 and requires a matching major with an
        # equal-or-higher minor, so torch's own 9.7.1.26 is below the floor by one minor
        # version. Observed on the CBLN1 box: "Loaded runtime CuDNN library: 9.7.1 but source
        # was compiled with: 9.8.0", then RET_CHECK dnn_support != nullptr.
        bad.append(f"nvidia-cudnn-cu12 is {cudnn}, not {EXPECT_CUDNN} -- jax will fail to compile")
except md.PackageNotFoundError:
    bad.append("nvidia-cudnn-cu12 is gone -- jax has no cudnn to compile against")
try:
    import jax
    if jax.__version__ != "0.10.2":
        bad.append(f"jax drifted to {jax.__version__}")
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        print("  [6e] no GPU visible; cudnn version checked, XLA compile not exercised")
    else:
        # Any jitted op is enough: gpu_compiler.cc checks dnn_support while compiling, so on
        # the CBLN1 box a bare jnp.ones() raised this. A convolution is kept because it is the
        # one op that also exercises the handle after compilation, and costs nothing extra.
        import jax.numpy as jnp
        x = jnp.ones((1, 4, 4, 1))
        k = jnp.ones((2, 2, 1, 1))
        jax.jit(lambda a, b: jax.lax.conv_general_dilated(
            a, b, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")))(x, k).block_until_ready()
        print(f"  [6e] jax {jax.__version__} still compiles a convolution on {gpus[0].device_kind}")
except Exception as e:
    bad.append(f"jax: {type(e).__name__}: {e}")
if bad:
    print("FAILED (the ESMC/ESMFold2 install disturbed the jax stack):", *bad, sep="\n  ")
    print("  Reinstall the [6b] line, or rebuild with WITH_ESMFOLD2=0.")
    sys.exit(1)
PYEOF

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
# The one symbol prepare_target_msa.py needs. Installed --no-deps, so this is exactly the
# check that matters: a missing runtime import would surface at the top of a campaign, in a
# prepare step, rather than here. Not fatal to the rest of the env -- a campaign that passes
# no target_msa never reaches it -- but it is a build that cannot do what it says it can.
try:
    from colabfold.colabfold import run_mmseqs2  # noqa: F401
except Exception as e:
    bad.append(f"colabfold (target MSA retrieval): {type(e).__name__}: {e}")
# The [6] canary ran while transformers was still 5.x. Every build now ends on 4.57.6, so re-check
# the repo imports downstream of it rather than assuming a downgrade is transparent.
try:
    import proteinfoundation  # noqa: F401
    import atomworks  # noqa: F401
except Exception as e:
    bad.append(f"post-downgrade import: {type(e).__name__}: {e}")
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
  needs: a target spec, ESM2 (free HF token), and RF3/Boltz2 if you refold with those. AF2 params and
  MPNN weights are installed at [6c], foldseek/mmseqs at [6d]. No dssp: DSSP_EXEC is referenced
  nowhere in src/ or configs/, and secondary structure comes from mdtraj (metrics/structure_ss.py).
  See docs/INFERENCE.md. Generation-only uses just the checkpoints above.
  ESMC/ESMFold2 are installed by default ([6e]), so this env is on Biohub's transformers fork.
  Their WEIGHTS are gated and are not fetched here -- set HF_TOKEN and accept the licences before
  using metric.consensus_backends=[esmfold2] or apo_folding_models=[esmfold2]. Build without them
  with WITH_ESMFOLD2=0: apo refolding falls back to plain ESMFold, complex to colabdesign.
EOF
