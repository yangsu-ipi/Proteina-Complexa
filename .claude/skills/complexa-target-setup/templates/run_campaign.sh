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
KIND="${1:?usage: run_campaign.sh smoke|production [STAGE] | followup N_DESIGNS [STAGE] | pooled}"
# followup takes the one number that cannot be predicted before a production run:
# how many more designs are wanted. Everything else is derived from what
# production actually produced -- see scripts/plan_followup.py.
#
# FOLLOWUP_INDEX pins which follow-up this is. Every stage re-plans, and the
# index otherwise comes from the records already on disk -- so a chained
# generate and evaluate would take consecutive indices and become two different
# runs, the second reading an inference directory the first never wrote.
# submit_campaign.sh sets it once for the whole chain.
if [[ "$KIND" == followup ]]; then
  WANT_DESIGNS="${2:?followup needs a design count, e.g. run_campaign.sh followup 700}"
  STAGE="${3:-all}"
else
  STAGE="${2:-all}"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$HERE/campaign.env"

case "$KIND" in
  smoke)      RUN_NAME="${RUN_PREFIX}_smoke";      SEEDS=$SMOKE_SEEDS;      RAW=$SMOKE_RAW;      KEEP=$SMOKE_KEEP;      EXPECT=$SMOKE_EXPECT;      RNG_SEED=${SMOKE_RNG_SEED:?set SMOKE_RNG_SEED in campaign.env} ;;
  production) RUN_NAME="${RUN_PREFIX}_production"; SEEDS=$PRODUCTION_SEEDS; RAW=$PRODUCTION_RAW; KEEP=$PRODUCTION_KEEP; EXPECT=$PRODUCTION_EXPECT; RNG_SEED=${PRODUCTION_RNG_SEED:?set PRODUCTION_RNG_SEED in campaign.env} ;;
  followup)   : ;;  # derived below, once conda and the campaign dir are up
  pooled)     : ;;  # reports over finished runs; no sizing of its own
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

# A campaign reaches its number over several runs, so the campaign total is not
# any one run's analyze output. One threshold set over every pooled run, with
# verdicts re-derived rather than read -- changing a threshold changes no metric,
# so this costs a comparison and not a re-evaluation.
if [[ "$KIND" == pooled ]]; then
  python -m proteinfoundation.analyze_pooled \
    --evaluation-root "$CAMPAIGN_DIR/evaluation_results" \
    --config-name "$CONFIG_NAME" --task-name "$TASK_NAME" --run-prefix "$RUN_PREFIX" \
    --output "$CAMPAIGN_DIR/metadata/pooled_analysis.json" \
    --pooled-csv "$CAMPAIGN_DIR/evaluation_results/pooled_results.csv"
  echo "Completed kind=pooled report=$CAMPAIGN_DIR/metadata/pooled_analysis.json"
  exit 0
fi

COMMUNITY_MODELS_PATH="${COMMUNITY_MODELS_PATH:-$COMPLEXA_REPO/community_models}"
export COMMUNITY_MODELS_PATH
[[ -d "$COMMUNITY_MODELS_PATH" ]] || { echo "missing community models: $COMMUNITY_MODELS_PATH" >&2; exit 2; }
if [[ -e community_models && ! -L community_models ]]; then
  echo "refusing to replace non-symlink $CAMPAIGN_DIR/community_models" >&2; exit 2
fi
[[ -L community_models ]] || ln -s "$COMMUNITY_MODELS_PATH" community_models
[[ "$(readlink community_models)" == "$COMMUNITY_MODELS_PATH" ]] || { echo "community_models symlink points elsewhere" >&2; exit 2; }

# Follow-ups are numbered, so their metadata does not overwrite each other's --
# a campaign's audit trail is one file per run, not one file per kind.
if [[ "$KIND" == followup ]]; then
  PLAN="$(python scripts/plan_followup.py \
    --campaign-dir "$CAMPAIGN_DIR" --want-designs "$WANT_DESIGNS" --shards "$SHARDS" \
    --base-seed "${PRODUCTION_RNG_SEED:?set PRODUCTION_RNG_SEED in campaign.env}" \
    --reference-seeds "${PRODUCTION_SEEDS:?set PRODUCTION_SEEDS in campaign.env}" \
    --run-prefix "$RUN_PREFIX" --config-name "$CONFIG_NAME" --task-name "$TASK_NAME" \
    ${FOLLOWUP_INDEX:+--index "$FOLLOWUP_INDEX"})"
  eval "$PLAN"
  RUN_NAME="$FOLLOWUP_RUN_NAME"
  SEEDS=$FOLLOWUP_SEEDS
  RAW=$FOLLOWUP_RAW
  KEEP=$FOLLOWUP_KEEP
  EXPECT=$FOLLOWUP_EXPECT
  RNG_SEED=$FOLLOWUP_RNG_SEED
  KIND_TAG="followup${FOLLOWUP_INDEX}"
  echo "follow-up #${FOLLOWUP_INDEX}: ${WANT_DESIGNS} more designs -> ${SEEDS} seeds, seed ${RNG_SEED}"
  echo "  parameters recorded in ${FOLLOWUP_RECORD}"
  echo "  deduplicated against the runs in ${FOLLOWUP_POOL_MANIFEST}"
else
  KIND_TAG="$KIND"
fi
CONFIG="$CAMPAIGN_DIR/${CONFIG_NAME}.yaml"
INF="$CAMPAIGN_DIR/inference/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
EVAL="$CAMPAIGN_DIR/evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
RESOLVED="$CAMPAIGN_DIR/metadata/resolved_config_${KIND_TAG}.yaml"
TRIM="$CAMPAIGN_DIR/metadata/shard_trim_${KIND_TAG}.json"
# ++seed is passed explicitly rather than left to the pipeline yaml, because it
# decides what a run produces: nres draws its binder lengths under it and the
# sampler noise follows, so two kinds sharing a seed draw overlapping designs.
# It is part of the generation digest, so a kind that changes it cannot resume
# another kind's shards by accident.
OVERRIDES=("++run_name=$RUN_NAME" "++seed=$RNG_SEED" "++generation.dataloader.dataset.nres.nsamples=$SEEDS" "++generation.filter.filter_samples_limit=$EXPECT")
# A follow-up exists because production fell short, and it samples the same
# target from the same model -- so it regenerates designs production already
# has. Without this the pooled set is smaller than its row count, and which rows
# were duplicates cannot be recovered afterwards.
if [[ "$KIND" == followup ]]; then
  OVERRIDES+=("++generation.filter.dedup_against_manifest=$FOLLOWUP_POOL_MANIFEST")
fi

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
bash "$COMPLEXA_REPO/.claude/skills/_shared/scripts/preflight.sh" --quiet --out "metadata/preflight_${KIND_TAG}.json"
python scripts/check_preflight.py "metadata/preflight_${KIND_TAG}.json" --resolved-config "$RESOLVED" \
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
    --redesign-model "$REDESIGN_MODEL" --output "metadata/run_outputs_${KIND_TAG}.json" \
    "${REQUIRE_COLUMNS[@]/#/--require-column=}"
fi
echo "Completed kind=$KIND stage=$STAGE inference=$INF evaluation=$EVAL"
