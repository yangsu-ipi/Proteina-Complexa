#!/usr/bin/env bash
# CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged.
#
# How big should the next run be?
#
#   scripts/estimate_run.sh 900              # how many seeds for ~900 designs
#   scripts/estimate_run.sh --orderable 500  # ... for ~500 sequences past the gate
#   scripts/estimate_run.sh --seeds 200      # what 200 seeds should yield
#
# Answers the question and stops. It writes nothing, reserves no run number, and
# submits nothing -- the number it prints is one you pass to
# `submit_campaign.sh production <SEEDS>` when you have decided.
#
# Separate from submitting because a design target is a guess about yield and a
# seed count is what a run actually takes. Converting one to the other is advice;
# baking the conversion into the submit path made the estimate binding, and there
# was then no way to run a size the arithmetic had not chosen.
set -euo pipefail

usage() { echo "usage: estimate_run.sh N_DESIGNS | --orderable N | --seeds N" >&2; exit 2; }
[[ $# -ge 1 ]] || usage
case "$1" in
  --seeds)     [[ $# -eq 2 ]] || usage; SIZED=(--seeds "$2") ;;
  --orderable) [[ $# -eq 2 ]] || usage; SIZED=(--want-orderable "$2") ;;
  *)           [[ "$1" =~ ^[0-9]+$ ]] || usage; SIZED=(--want-designs "$1") ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$HERE/campaign.env"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-$HERE}"

PLAN="$(python3 "$HERE/scripts/plan_followup.py" \
  --campaign-dir "$CAMPAIGN_DIR" "${SIZED[@]}" --dry-run --shards "$SHARDS" \
  --base-seed "${PRODUCTION_RNG_SEED:?set PRODUCTION_RNG_SEED in campaign.env}" \
  --reference-seeds "${PRODUCTION_SEEDS:?set PRODUCTION_SEEDS in campaign.env}" \
  --run-prefix "$RUN_PREFIX" --config-name "$CONFIG_NAME" --task-name "$TASK_NAME")"
eval "$PLAN"

echo "run #${RUN_NUMBER} would be ${FOLLOWUP_SEEDS} seeds -> about ${RUN_PROJECTED_DESIGNS} designs${RUN_PROJECTED_ORDERABLE:+, of which ~${RUN_PROJECTED_ORDERABLE} orderable}"
echo "  ${FOLLOWUP_RAW} raw, trimmed to ${FOLLOWUP_EXPECT} before global dedup"
echo "  at ${RUN_DESIGNS_PER_SEED} designs per seed, measured over: ${RUN_CALIBRATED_ON}"
if [[ -n "${RUN_ORDERABLE_PER_SEED:-}" ]]; then
  # The orderable rate has a direction, so the series is shown rather than an
  # average: on CBLN1 it fell 2.56 -> 2.19 -> 1.78 across three runs, and planning
  # on the mean of a declining series plans for a run that already happened.
  echo "  at ${RUN_ORDERABLE_PER_SEED} orderable per seed, from the most recent run (${RUN_ORDERABLE_BASIS})"
  echo "    per run so far: ${RUN_ORDERABLE_SERIES}"
  echo "    a falling series means the pooled average would over-promise; read the trend"
fi
echo
echo "  submit with: scripts/submit_campaign.sh production ${FOLLOWUP_SEEDS}"
echo "  (an estimate, not a promise -- pass whatever seed count you decide on)"
