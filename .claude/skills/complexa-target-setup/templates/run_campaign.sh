#!/usr/bin/env bash
# CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged and edit
# <campaign>/campaign.env instead. Validated by CBLN1/5KC5, first complete run
# 2026-08-28.
#
# Everything below the config load is campaign-independent, and most of it exists
# because a specific thing went wrong once: the conda + env.sh sourcing order, the
# community_models symlink guard, clearing the CCD/PDB mirrors, per-shard GPU
# pinning, and the JAX memory fraction. Re-deriving this per campaign is how those
# bugs come back.
set -euo pipefail
KIND="${1:?usage: run_campaign.sh smoke|production [all|generate|filter|evaluate|analyze]}"
STAGE="${2:-all}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$HERE/campaign.env"

case "$KIND" in
  smoke)      RUN_NAME="${RUN_PREFIX}_smoke";      SEEDS=$SMOKE_SEEDS;      RAW=$SMOKE_RAW;      KEEP=$SMOKE_KEEP;      EXPECT=$SMOKE_EXPECT ;;
  production) RUN_NAME="${RUN_PREFIX}_production"; SEEDS=$PRODUCTION_SEEDS; RAW=$PRODUCTION_RAW; KEEP=$PRODUCTION_KEEP; EXPECT=$PRODUCTION_EXPECT ;;
  *) echo "invalid run kind: $KIND" >&2; exit 2 ;;
esac

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "run inside a Slurm allocation" >&2; exit 2; }
export COMPLEXA_REPO CAMPAIGN_DIR
# Cleared before AND after sourcing env.sh: env.sh sets them, and an unreachable
# mirror surfaces as "Error locating target ...collate_fn", which names neither.
export CCD_MIRROR_PATH="" PDB_MIRROR_PATH=""
# shellcheck source=/dev/null
source "$CONDA_SH"; conda activate "$CONDA_ENV"
set -a; source "$COMPLEXA_REPO/env.sh"; set +a
export CCD_MIRROR_PATH="" PDB_MIRROR_PATH=""
cd "$CAMPAIGN_DIR"
mkdir -p metadata logs/slurm

COMMUNITY_MODELS_PATH="${COMMUNITY_MODELS_PATH:-$COMPLEXA_REPO/community_models}"
export COMMUNITY_MODELS_PATH
[[ -d "$COMMUNITY_MODELS_PATH" ]] || { echo "missing community models: $COMMUNITY_MODELS_PATH" >&2; exit 2; }
if [[ -e community_models && ! -L community_models ]]; then
  echo "refusing to replace non-symlink $CAMPAIGN_DIR/community_models" >&2; exit 2
fi
[[ -L community_models ]] || ln -s "$COMMUNITY_MODELS_PATH" community_models
[[ "$(readlink community_models)" == "$COMMUNITY_MODELS_PATH" ]] || { echo "community_models symlink points elsewhere" >&2; exit 2; }

CONFIG="$CAMPAIGN_DIR/${CONFIG_NAME}.yaml"
INF="$CAMPAIGN_DIR/inference/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
EVAL="$CAMPAIGN_DIR/evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
RESOLVED="$CAMPAIGN_DIR/metadata/resolved_config_${KIND}.yaml"
TRIM="$CAMPAIGN_DIR/metadata/shard_trim_${KIND}.json"
OVERRIDES=("++run_name=$RUN_NAME" "++generation.dataloader.dataset.nres.nsamples=$SEEDS" "++generation.filter.filter_samples_limit=$EXPECT")

# Campaign-specific preparation, before anything validates or resolves: MSA
# building, target extraction, whatever this package needs. Declared in
# campaign.env so the runner needs no knowledge of what they are. Dropping this
# hook is how the first template lost the original's prepare_target_msa.py step --
# a retrofit did not notice, because its MSA already existed, and a fresh campaign
# would have discovered it much later.
if [[ ${PREPARE_STEPS+set} == set ]] && ((${#PREPARE_STEPS[@]})); then
  for step in "${PREPARE_STEPS[@]}"; do
    echo "prepare: $step"
    # shellcheck disable=SC2086
    python $step
  done
fi

# pipeline.yaml resolves ${oc.env:VAR} when Hydra loads it, and an unset one
# surfaces as an omegaconf KeyError several frames deep, AFTER the checkpoint has
# loaded. Checking here turns that into one line before anything expensive starts.
# Only the no-default form is required; ${oc.env:VAR,fallback} is not.
required_env=()
while read -r var; do
  [[ -z "$var" ]] || [[ -n "${!var:-}" ]] || required_env+=("$var")
done < <(grep -oE '\$\{oc\.env:[A-Z_][A-Z0-9_]*\}' "$CONFIG" | sed 's/.*oc\.env://; s/}//' | sort -u)
if ((${#required_env[@]})); then
  echo "$CONFIG needs environment variables that are not set: ${required_env[*]}" >&2
  echo "The package root is exported as CAMPAIGN_DIR; a config carried over from an" >&2
  echo "older package may still name it something campaign-specific." >&2
  exit 2
fi

python scripts/validate_resolved_config.py --config "$CONFIG" --expected-seeds "$SEEDS" --expected-generated "$RAW" \
  --output "$RESOLVED" --override "${OVERRIDES[@]}"
python "$COMPLEXA_REPO/docs/binder-target-setup/scripts/check_target_pdb.py" \
  --pdb "$TARGET_PDB" --chain "$TARGET_CHAIN" --target-input "$TARGET_INPUT"
bash "$COMPLEXA_REPO/.claude/skills/_shared/scripts/preflight.sh" --quiet --out "metadata/preflight_${KIND}.json"
python scripts/check_preflight.py "metadata/preflight_${KIND}.json" --resolved-config "$RESOLVED" \
  --expected-designs "$EXPECT" --min-vram-gb "$MIN_VRAM_GB" \
  "${REQUIRE_HF_REPOS[@]/#/--require-hf-repo=}"

# One shard per GPU, set INSIDE srun so the step cannot override it. --gres=gpu:1
# does not isolate where GRES cgroups are unenforced: both steps then report
# CUDA_VISIBLE_DEVICES [0,1], pick device 0, and fight over one card.
run_gpu_stage() {
  local module=$1 shard=$2 fraction=$3
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
    env CUDA_VISIBLE_DEVICES="$shard" XLA_PYTHON_CLIENT_MEM_FRACTION="$fraction" \
    python -m "$module" --config-path "$CAMPAIGN_DIR" --config-name "$CONFIG_NAME" \
    "++job_id=$shard" "++base_config_name=$CONFIG_NAME" "${OVERRIDES[@]}"
}

all_shards() {
  local module=$1 fraction=$2 pids=()
  for ((s = 0; s < SHARDS; s++)); do run_gpu_stage "$module" "$s" "$fraction" & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

# No output-directory-existence guard here. Generation distinguishes skip,
# clear-and-regenerate and abort per shard; a guard that refuses whenever the
# directory exists disables resume for the case resume is for.
if [[ "$STAGE" == all || "$STAGE" == generate ]]; then
  all_shards proteinfoundation.generate "$XLA_MEM_FRACTION_GENERATE"
  python scripts/trim_shards.py --inference-dir "$INF" --per-shard "$KEEP" --shards "$SHARDS" --output "$TRIM"
fi
if [[ "$STAGE" == all || "$STAGE" == filter ]]; then
  python -m proteinfoundation.filter --config-path "$CAMPAIGN_DIR" --config-name "$CONFIG_NAME" \
    "++job_id=0" "++base_config_name=$CONFIG_NAME" "${OVERRIDES[@]}"
fi
if [[ "$STAGE" == all || "$STAGE" == evaluate ]]; then
  [[ -d "$INF" ]] || { echo "missing $INF" >&2; exit 2; }
  all_shards proteinfoundation.evaluate "$XLA_MEM_FRACTION_EVALUATE"
fi
if [[ "$STAGE" == all || "$STAGE" == analyze ]]; then
  python -m proteinfoundation.analyze --config-path "$CAMPAIGN_DIR" --config-name "$CONFIG_NAME" \
    "++job_id=0" "++base_config_name=$CONFIG_NAME" "${OVERRIDES[@]}"
  python scripts/verify_run_outputs.py --inference-dir "$INF" --evaluation-dir "$EVAL" \
    --expected-retained "$EXPECT" --resolved-config "$RESOLVED" --shards "$SHARDS" --trim-report "$TRIM" \
    --redesign-model "$REDESIGN_MODEL" --output "metadata/run_outputs_${KIND}.json" \
    "${REQUIRE_COLUMNS[@]/#/--require-column=}"
fi
echo "Completed kind=$KIND stage=$STAGE inference=$INF evaluation=$EVAL"
