#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# GPU count, GPU-stage host memory, and stage-specific time are supplied by
# submit_campaign.sh. CPU stages deliberately request neither GPUs nor memory.
set -euo pipefail
CAMPAIGN_DIR="${CAMPAIGN_DIR:?submit_campaign.sh must export CAMPAIGN_DIR}"
exec "$CAMPAIGN_DIR/scripts/run_campaign.sh" "$@"
