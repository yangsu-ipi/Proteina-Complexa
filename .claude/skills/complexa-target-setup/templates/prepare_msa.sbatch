#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# CPU-only by design: no --gres and no explicit --mem reservation.
set -euo pipefail
CAMPAIGN_DIR="${CAMPAIGN_DIR:?submit_campaign.sh must export CAMPAIGN_DIR}"
# shellcheck source=/dev/null
source "$CAMPAIGN_DIR/campaign.env"
export CAMPAIGN_DIR COMPLEXA_REPO TARGET_MSA HF_HOME
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -a
# shellcheck source=/dev/null
source "$COMPLEXA_REPO/env.sh"
set +a
export CCD_MIRROR_PATH="" PDB_MIRROR_PATH=""
cd "$CAMPAIGN_DIR"
mkdir -p data/msa metadata logs/slurm
if [[ ${PREPARE_STEPS+set} != set ]] || ((${#PREPARE_STEPS[@]} == 0)); then
  echo "campaign.env declares no PREPARE_STEPS; nothing to prepare"
  exit 0
fi
for step in "${PREPARE_STEPS[@]}"; do
  echo "prepare: $step"
  # shellcheck disable=SC2086
  python $step
done
