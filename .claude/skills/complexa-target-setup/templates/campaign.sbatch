#!/usr/bin/env bash
# CAMPAIGN TEMPLATE -- copy to <campaign>/slurm/<name>.sbatch, one per run kind.
# Only the SBATCH header and the two arguments on the last line differ between
# kinds; everything else is in campaign.env.
#
# --gres=gpu:2 must match SHARDS in campaign.env: run_campaign.sh starts one
# process per shard and pins each to its own card.
#SBATCH --job-name=CAMPAIGN-KIND
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=58G
# Five AF2 models and three ESMFold2 seeds per fold put evaluation several times
# above what a single-model, single-seed run cost, and the CBLN1 production run
# already needed 3 days at the old settings. A job killed at the wall clock
# loses only the shards it had not finished -- the caches keep the rest -- but
# it loses them after paying for them.
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
set -euo pipefail
CAMPAIGN_DIR="${CAMPAIGN_DIR:?set CAMPAIGN_DIR, or edit this line to the package path}"
export CAMPAIGN_DIR
exec "$CAMPAIGN_DIR/scripts/run_campaign.sh" smoke all
