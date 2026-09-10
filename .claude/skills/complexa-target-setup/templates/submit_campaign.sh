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
# `from..to` stops early, for the one case where stopping is right: several runs
# re-evaluated together share ONE pooled report, and chaining a pooled to each of
# them queues a campaign total per run, every one of which reads whatever has
# finished so far. The last to run happens to be correct; the others write a
# mid-flight number to pooled_analysis.json under the same name.
#
#   for r in production "followup 900" "followup 1110"; do
#     submit_campaign.sh $r evaluate..analyze
#   done
#   submit_campaign.sh production pooled          # once, after all of them
#
# `..analyze` starts from the beginning; `evaluate..` runs to the end.
#
# Stages run as separate jobs joined by afterok rather than as one long job, for
# two reasons. A failure then costs the stage that failed and not the hours
# before it -- generation's output survives an evaluation that dies. And the
# stages want different machines: generate and evaluate need the GPUs for hours,
# while filter, analyze and the pooled report need neither and would otherwise
# hold them idle.
set -euo pipefail

KIND="${1:?usage: submit_campaign.sh smoke|production [STAGE|FROM..TO] | followup N_DESIGNS [STAGE|FROM..TO] [-- HYDRA_OVERRIDE...]}"
shift
CAMPAIGN_DIR_FROM_ENV="${CAMPAIGN_DIR:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
# shellcheck source=/dev/null
source "$HERE/campaign.env"
# The package's own location is the truth about where it is. campaign.env carries
# an absolute CAMPAIGN_DIR default, so before this a moved or copied package cd'd
# to wherever it was FIRST created -- HERE was computed and then used only to
# source campaign.env. Resolved with `pwd -P` on both sides so reaching the same
# package through a symlink is not mistaken for a mismatch.
if [[ -n "${CAMPAIGN_DIR_FROM_ENV:-}" ]]; then
  want="$(cd "$CAMPAIGN_DIR_FROM_ENV" 2>/dev/null && pwd -P || echo "$CAMPAIGN_DIR_FROM_ENV")"
  if [[ "$want" != "$HERE" ]]; then
    echo "CAMPAIGN_DIR=$CAMPAIGN_DIR_FROM_ENV is not where these scripts live ($HERE)." >&2
    echo "Running one package's scripts against another's data mixes two campaigns." >&2
    exit 2
  fi
fi
CAMPAIGN_DIR="$HERE"
export CAMPAIGN_DIR

SBATCH_TEMPLATE="$CAMPAIGN_DIR/slurm/campaign.sbatch"
[[ -f "$SBATCH_TEMPLATE" ]] || { echo "missing $SBATCH_TEMPLATE" >&2; exit 2; }

# Defaults rather than required settings: a campaign that never thought about
# wall clock still submits, and one that did can say so in campaign.env.
GPU_TIME="${SLURM_TIME_GPU:-3-00:00:00}"
CPU_TIME="${SLURM_TIME_CPU:-04:00:00}"

RUN_ARGS=("$KIND")
TAG="$KIND"
SIZED=""
if [[ "$KIND" == followup ]]; then
  WANT_DESIGNS="${1:?followup needs a design count, e.g. submit_campaign.sh followup 900}"
  shift
  RUN_ARGS=(followup "$WANT_DESIGNS")
  SIZED="--want-designs $WANT_DESIGNS"
elif [[ "$KIND" == production && "${1:-}" =~ ^[0-9]+$ ]]; then
  # `production N` is N seeds, which is what a run actually takes. The design
  # target is a separate question -- ask scripts/plan_followup.py --want-designs
  # what a target converts to, then pass the number you decided on.
  RUN_SEEDS="$1"
  shift
  RUN_ARGS=(production "$RUN_SEEDS")
  SIZED="--seeds $RUN_SEEDS"
  WANT_DESIGNS=""
fi

# Parsed before the follow-up is planned, because which stages run decides
# whether this is a new follow-up or a re-run of one that exists.
FROM_STAGE="all"
TO_STAGE=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
  if [[ "$1" == *".."* ]]; then
    FROM_STAGE="${1%%..*}"; TO_STAGE="${1##*..}"
    # `..analyze` means from the beginning, `evaluate..` to the end. Both halves
    # optional so the range form degrades to the plain one rather than erroring
    # on a stage name that is merely absent.
    [[ -n "$FROM_STAGE" ]] || FROM_STAGE="all"
  else
    FROM_STAGE="$1"
  fi
  shift
fi

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

# Truncated after the from-stage selection, so an end before the start is caught
# as the contradiction it is rather than silently yielding nothing.
if [[ -n "$TO_STAGE" ]]; then
  END=-1
  for i in "${!SELECTED[@]}"; do
    if [[ "${SELECTED[$i]%%:*}" == "$TO_STAGE" ]]; then END=$i; break; fi
  done
  if ((END < 0)); then
    if [[ " ${STAGES[*]%%:*} " == *" $TO_STAGE "* ]]; then
      echo "stage range '${FROM_STAGE}..${TO_STAGE}' ends before it starts: ${TO_STAGE} runs before ${FROM_STAGE}." >&2
    else
      echo "unknown end stage '$TO_STAGE' for kind '$KIND'." >&2
    fi
    echo "This campaign runs, in order: ${STAGES[*]%%:*} -- or 'all' for the whole chain." >&2
    exit 2
  fi
  SELECTED=("${SELECTED[@]:0:$((END + 1))}")
fi
FIRST_STAGE="${SELECTED[0]%%:*}"
# Not ${SELECTED[-1]}: negative subscripts need bash 4.3 and macOS ships 3.2,
# so the template would parse everywhere and run only on the cluster.
LAST_STAGE="${SELECTED[${#SELECTED[@]}-1]%%:*}"

if [[ -n "$SIZED" ]]; then
  # A chain that includes generate is a new run; one that starts later is a
  # re-run of a follow-up that already exists and must reuse its index. Allocating
  # a fresh one there would name an inference directory nothing ever wrote, and
  # would burn a seed no run will ever use.
  #
  # A pre-set FOLLOWUP_INDEX names one outright. That is the way past an ambiguous
  # design count -- two follow-ups that asked for the same number -- and it is
  # honoured whatever the starting stage, because naming an index is a stronger
  # statement than the count is.
  RESUME=()
  if [[ -n "${FOLLOWUP_INDEX:-}" ]]; then
    RESUME=(--index "$FOLLOWUP_INDEX")
    echo "follow-up index pinned by the environment: ${FOLLOWUP_INDEX}"
  elif [[ "$FIRST_STAGE" != generate ]]; then
    RESUME=(--resume)
  fi
  # Planned once, here, so every job in the chain is the same follow-up. Left to
  # each job, the index would come from the records on disk and advance between
  # them -- evaluate would then look for an inference directory generate never
  # wrote. Planning here also puts the parameters on disk before anything is
  # queued, which is what makes a submitted chain auditable.
  PLAN="$(python3 "$CAMPAIGN_DIR/scripts/plan_followup.py" \
    --campaign-dir "$CAMPAIGN_DIR" $SIZED --shards "$SHARDS" \
    --base-seed "${PRODUCTION_RNG_SEED:?set PRODUCTION_RNG_SEED in campaign.env}" \
    --reference-seeds "${PRODUCTION_SEEDS:?set PRODUCTION_SEEDS in campaign.env}" \
    --run-prefix "$RUN_PREFIX" --config-name "$CONFIG_NAME" --task-name "$TASK_NAME" \
    ${DRY_RUN:+--dry-run} ${RESUME[@]+"${RESUME[@]}"})"
  eval "$PLAN"
  export FOLLOWUP_INDEX="$FOLLOWUP_INDEX"
  export RUN_NUMBER="$RUN_NUMBER"
  # The tag the planner chose, not one rebuilt here. A run already on disk keeps
  # the name it was written under; a new one gets the current spelling. Rebuilding
  # it from the follow-up index is how the job name, the metadata filenames and
  # the inference directory came to be able to disagree.
  TAG="$RUN_TAG"
  echo "run #${RUN_NUMBER} (${RUN_TAG}): ${FOLLOWUP_SEEDS} seeds, seed ${FOLLOWUP_RNG_SEED}${WANT_DESIGNS:+ -- sized for ${WANT_DESIGNS} designs}"
  echo "  planned in ${FOLLOWUP_RECORD}${DRY_RUN:+ (not written: this is a dry run)}"
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
  flags+=(--export="ALL,CAMPAIGN_DIR=${CAMPAIGN_DIR}${FOLLOWUP_INDEX:+,FOLLOWUP_INDEX=${FOLLOWUP_INDEX}}${RUN_NUMBER:+,RUN_NUMBER=${RUN_NUMBER}}")
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "sbatch ${flags[*]} $SBATCH_TEMPLATE $*" >&2
    echo "DRY"
  else
    sbatch "${flags[@]}" "$SBATCH_TEMPLATE" "$@"
  fi
}

[[ "$FROM_STAGE" == all && -z "$TO_STAGE" ]] || echo "re-running: ${SELECTED[*]%%:*}"
# Said once, here, rather than left to be noticed when the total looks stale. The
# pooled report is the campaign's headline number and this chain does not produce
# it -- which is the point of stopping early, but only if it is followed up.
if [[ "$LAST_STAGE" != pooled && "$KIND" != smoke ]]; then
  # Named as `production pooled` whatever this chain's kind is: the report is
  # campaign-wide and takes no run, so the kind is immaterial -- and `followup`
  # is not a runnable spelling here, since it would demand a design count for a
  # stage that has no run of its own.
  echo "  note: no pooled report is queued. The report is campaign-wide, so once every chain"
  echo "        you are re-running has finished, run: submit_campaign.sh production pooled"
fi

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
