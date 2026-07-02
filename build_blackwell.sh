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
echo "Ready in $ENV_DIR. Checkpoints: complexa init && complexa download --complexa-all (NGC)."
