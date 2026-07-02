#!/usr/bin/env bash
# Build Proteina-Complexa for NVIDIA Blackwell (sm_120, e.g. RTX PRO 6000).
# proteinfoundation needs Python>=3.12; torch cu128 for sm_120; tmol builds from source.
# The dep set is over-constrained -> biotite==1.6.0 and numpy==2.4.6 pins go LAST (order matters).
# No source changes required. Verified 2026-07 on IPI gnode2.
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
# [5] the package (editable) — pulls proteinfoundation + downgrades biotite/numpy:
( cd "$REPO" && "$PIP" install -e . )
# [6] RECONCILE (order-sensitive, do LAST):
"$PIP" install biotite==1.6.0          # atomworks needs >=1.4
"$PIP" install --no-deps numpy==2.4.6  # numba/tmol need numpy < 2.5

"$PY" - <<'PYEOF'
import importlib
for m in ["proteinfoundation","atomworks","tmol","graphein","biotite","torch","numpy"]:
    importlib.import_module(m)
print("Proteina-Complexa (Blackwell): all core imports OK")
PYEOF

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
