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

echo "run #${RUN_NUMBER} would be ${FOLLOWUP_SEEDS} seeds -> about ${RUN_PROJECTED_DESIGNS} designs${RUN_PROJECTED_ORDERABLE:+, of which at least ~${RUN_PROJECTED_ORDERABLE} orderable}"
echo "  ${FOLLOWUP_RAW} raw, trimmed to ${FOLLOWUP_EXPECT} before global dedup"
echo "  at ${RUN_DESIGNS_PER_SEED} designs per seed, measured over: ${RUN_CALIBRATED_ON}"
if [[ -n "${RUN_ORDERABLE_PER_DESIGN:-}" ]]; then
  # Sized on the low end, and the interval is clustered. Beam search expands one
  # nres draw into several candidates, so designs sharing a root are not
  # independent -- and binder length, which dominates whether a design passes, is
  # drawn per root. Treating designs as independent understated the variance
  # 4-7 fold on CBLN1, and the naive interval after the first run excluded what
  # the third run actually delivered.
  echo "  at ${RUN_ORDERABLE_PER_DESIGN} orderable per design (95% CI ${RUN_ORDERABLE_PER_DESIGN_LOWER}-${RUN_ORDERABLE_PER_DESIGN_UPPER}, ${RUN_ORDERABLE_CLUSTERS} clusters)"
  echo "    per run so far: ${RUN_ORDERABLE_SERIES}"
  echo "    sized on ${RUN_ORDERABLE_PER_DESIGN_LOWER}, the low end -- at the mean it would be ~${RUN_PROJECTED_ORDERABLE_MID}"
  echo "    interval is ${RUN_ORDERABLE_DESIGN_EFFECT}x wider than treating designs as independent draws"
fi
echo
echo "  submit with: scripts/submit_campaign.sh production ${FOLLOWUP_SEEDS}"
echo "  (an estimate, not a promise -- pass whatever seed count you decide on)"
