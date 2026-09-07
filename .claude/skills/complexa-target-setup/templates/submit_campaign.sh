#!/usr/bin/env bash
# Submit a campaign run as a dependency chain, so one command produces a result
# rather than a sequence of jobs somebody has to babysit.
#
#   submit_campaign.sh production
#   submit_campaign.sh followup 900
#   DRY_RUN=1 submit_campaign.sh followup 900     # print, submit nothing
#
# A stage name re-runs from there to the end of the chain:
#
#   submit_campaign.sh production evaluate        # evaluate, analyze, pooled
#   submit_campaign.sh followup 900 evaluate      # the follow-up that wanted 900
#   submit_campaign.sh production pooled          # the campaign total alone
#
# From a stage rather than a list of them, because a re-run exists because
# something changed, and everything downstream of a changed stage reads what it
# produced. Re-running evaluate and not analyze leaves a results CSV that
# disagrees with the per-job CSVs it was built from, and nothing says so.
#
# Stages run as separate jobs joined by afterok rather than as one long job, for
# two reasons. A failure then costs the stage that failed and not the hours
# before it -- generation's output survives an evaluation that dies. And the
# stages want different machines: generate and evaluate need the GPUs for hours,
# while filter, analyze and the pooled report need neither and would otherwise
# hold them idle.
set -euo pipefail

KIND="${1:?usage: submit_campaign.sh smoke|production [FROM_STAGE] | followup N_DESIGNS [FROM_STAGE] [-- HYDRA_OVERRIDE...]}"
shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$HERE/campaign.env"
export CAMPAIGN_DIR

SBATCH_TEMPLATE="$CAMPAIGN_DIR/slurm/campaign.sbatch"
[[ -f "$SBATCH_TEMPLATE" ]] || { echo "missing $SBATCH_TEMPLATE" >&2; exit 2; }

# Defaults rather than required settings: a campaign that never thought about
# wall clock still submits, and one that did can say so in campaign.env.
GPU_TIME="${SLURM_TIME_GPU:-3-00:00:00}"
CPU_TIME="${SLURM_TIME_CPU:-04:00:00}"

RUN_ARGS=("$KIND")
TAG="$KIND"
if [[ "$KIND" == followup ]]; then
  WANT_DESIGNS="${1:?followup needs a design count, e.g. submit_campaign.sh followup 900}"
  shift
  RUN_ARGS=(followup "$WANT_DESIGNS")
fi

# Parsed before the follow-up is planned, because which stages run decides
# whether this is a new follow-up or a re-run of one that exists.
FROM_STAGE="all"
if [[ $# -gt 0 && "${1}" != --* ]]; then FROM_STAGE="$1"; shift; fi

# Everything left is passed to Hydra by every stage, so a run that differs from
# the campaign's config differs the same way at each step -- a redesign count set
# for generate and not for evaluate would refold a different number of sequences
# than were designed.
[[ "${1:-}" == "--" ]] && shift
EXTRA_OVERRIDES=("$@")

# generate and evaluate are the GPU stages; the rest are bookkeeping over files
# those two produced. The pooled report is the campaign total rather than this
# run's, and smoke designs are a throwaway check that is not part of the pool.
STAGES=(generate:gpu filter:cpu evaluate:gpu analyze:cpu pooled:cpu)
case "$KIND" in
  smoke) STAGES=(generate:gpu filter:cpu evaluate:gpu analyze:cpu) ;;
  production|followup) ;;
  *) echo "submit_campaign.sh does not submit '$KIND'" >&2; exit 2 ;;
esac

# The chain from FROM_STAGE to the end. Never a subset in the middle: a stage
# whose inputs were just rewritten and whose outputs are not is the state this
# refuses to create.
if [[ "$FROM_STAGE" == all ]]; then
  SELECTED=("${STAGES[@]}")
else
  SELECTED=()
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]%%:*}" == "$FROM_STAGE" ]]; then SELECTED=("${STAGES[@]:$i}"); break; fi
  done
  if ((${#SELECTED[@]} == 0)); then
    echo "unknown stage '$FROM_STAGE' for kind '$KIND'." >&2
    echo "This campaign runs, in order: ${STAGES[*]%%:*} -- or 'all' for the whole chain." >&2
    exit 2
  fi
fi
FIRST_STAGE="${SELECTED[0]%%:*}"

if [[ "$KIND" == followup ]]; then
  # A chain that includes generate is a new follow-up; one that starts later is a
  # re-run of a follow-up that already exists and must reuse its index. Allocating
  # a fresh one there would name an inference directory nothing ever wrote, and
  # would burn a seed no run will ever use.
  RESUME=()
  [[ "$FIRST_STAGE" == generate ]] || RESUME=(--resume)
  # Planned once, here, so every job in the chain is the same follow-up. Left to
  # each job, the index would come from the records on disk and advance between
  # them -- evaluate would then look for an inference directory generate never
  # wrote. Planning here also puts the parameters on disk before anything is
  # queued, which is what makes a submitted chain auditable.
  PLAN="$(python3 "$CAMPAIGN_DIR/scripts/plan_followup.py" \
    --campaign-dir "$CAMPAIGN_DIR" --want-designs "$WANT_DESIGNS" --shards "$SHARDS" \
    --base-seed "${PRODUCTION_RNG_SEED:?set PRODUCTION_RNG_SEED in campaign.env}" \
    --reference-seeds "${PRODUCTION_SEEDS:?set PRODUCTION_SEEDS in campaign.env}" \
    --run-prefix "$RUN_PREFIX" --config-name "$CONFIG_NAME" --task-name "$TASK_NAME" \
    ${RESUME[@]+"${RESUME[@]}"})"
  eval "$PLAN"
  export FOLLOWUP_INDEX="$FOLLOWUP_INDEX"
  TAG="followup${FOLLOWUP_INDEX}"
  echo "follow-up #${FOLLOWUP_INDEX}: ${WANT_DESIGNS} designs -> ${FOLLOWUP_SEEDS} seeds, seed ${FOLLOWUP_RNG_SEED}"
  echo "  planned in ${FOLLOWUP_RECORD}"
  echo "  deduplicated against ${FOLLOWUP_POOL_MANIFEST}"
fi

submit() {  # name kind_of_node dependency args...
  local name="$1" node="$2" dep="$3"; shift 3
  local flags=(--parsable --job-name="$name")
  if [[ "$node" == gpu ]]; then
    flags+=(--gres="gpu:${SHARDS}" --time="$GPU_TIME")
  else
    flags+=(--time="$CPU_TIME")
  fi
  [[ -n "$dep" ]] && flags+=(--dependency="afterok:${dep}")
  flags+=(--export="ALL,CAMPAIGN_DIR=${CAMPAIGN_DIR}${FOLLOWUP_INDEX:+,FOLLOWUP_INDEX=${FOLLOWUP_INDEX}}")
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "sbatch ${flags[*]} $SBATCH_TEMPLATE $*" >&2
    echo "DRY"
  else
    sbatch "${flags[@]}" "$SBATCH_TEMPLATE" "$@"
  fi
}

[[ "$FROM_STAGE" == all ]] || echo "re-running from ${FIRST_STAGE}: ${SELECTED[*]%%:*}"

dep=""
for entry in "${SELECTED[@]}"; do
  stage="${entry%%:*}"; node="${entry##*:}"
  if [[ "$stage" == pooled ]]; then
    # The campaign total, not this run's. Last in the chain because it reads every
    # run's results and would otherwise report a number that predates the run just
    # submitted. It takes no run kind and no metric overrides: it reads finished
    # CSVs and applies thresholds, so an override there would describe folding
    # that already happened.
    dep="$(submit "${TAG}-pooled" "$node" "$dep" pooled)"
  else
    dep="$(submit "${TAG}-${stage}" "$node" "$dep" "${RUN_ARGS[@]}" "$stage" ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"})"
  fi
  echo "  ${stage} -> job ${dep}"
done
echo "Submitted ${TAG}: each stage starts only if the one before it succeeded."
