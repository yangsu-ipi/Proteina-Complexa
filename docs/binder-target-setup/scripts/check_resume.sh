#!/usr/bin/env bash
# Verify that in-stage resume actually reuses work, on the machine that can run it.
#
#   bash check_resume.sh --config /path/to/pipeline.yaml [--samples 2] [--nsteps 50]
#
# Drives four real `complexa` invocations against a throwaway run_name and
# asserts on filesystem state, not on log text. Exits non-zero on the first
# failed assertion. Safe to run beside real campaigns: everything lands under
# ./inference/<config>_<task>_<run_name> with a run_name of resume_check_<pid>,
# and --keep is required to leave it behind.
#
# What it proves, in order:
#   1. a completed shard writes a marker, and evaluate writes a fold cache
#   2. re-running generate with the same config SKIPS (no new sample dirs)
#   3. re-running evaluate with the same config REUSES every fold cache
#   4. changing a generation parameter INVALIDATES the marker (regenerates)
#   5. removing a sample directory INVALIDATES the marker (regenerates)
#   6. changing the folding backend INVALIDATES the fold caches
#
# Steps 4-6 are the ones worth having: a resume that never invalidates is
# indistinguishable from a resume that silently serves stale results.

set -euo pipefail

CONFIG=""; SAMPLES=2; NSTEPS=50; KEEP=0; SKIP_BACKEND=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)   CONFIG="$2"; shift 2 ;;
    --samples)  SAMPLES="$2"; shift 2 ;;
    --nsteps)   NSTEPS="$2"; shift 2 ;;
    --keep)     KEEP=1; shift ;;
    --skip-backend-check) SKIP_BACKEND=1; shift ;;
    -h|--help)  sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CONFIG" ]] || { echo "--config is required" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 2; }

RUN_NAME="resume_check_$$"
CONFIG_NAME="$(basename "${CONFIG%.*}")"
PASS=0; FAIL=0

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '   \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 1; }

# Resolve the run directory the way generate.py does: ./inference/<config>_<task>[_<run>]
# The task name comes from the resolved config, not from a guess.
TASK_NAME="$(python - "$CONFIG" <<'PY'
import sys, pathlib
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
p = pathlib.Path(sys.argv[1]).resolve()
with initialize_config_dir(version_base=None, config_dir=str(p.parent)):
    cfg = compose(config_name=p.stem)
print(OmegaConf.select(cfg, "generation.task_name") or "")
PY
)" || die "could not resolve generation.task_name from $CONFIG"
[[ -n "$TASK_NAME" ]] || die "generation.task_name is empty in $CONFIG"

RUN_DIR="./inference/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
EVAL_DIR="./evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
printf 'config      %s\ntask_name   %s\nrun_name    %s\ncwd         %s\nrun dir     %s\n' \
  "$CONFIG" "$TASK_NAME" "$RUN_NAME" "$PWD" "$RUN_DIR"
# ./inference and ./evaluation_results are cwd-relative with no chdir anywhere
# in the codebase, so run this from the campaign directory.
[[ -w . ]] || die "cwd is not writable; run this from the campaign directory"

cleanup() {
  if [[ $KEEP -eq 1 ]]; then
    printf '\nkept %s and %s\n' "$RUN_DIR" "$EVAL_DIR"
  else
    rm -rf "$RUN_DIR" "$EVAL_DIR"
  fi
}
trap cleanup EXIT

BASE_OVERRIDES=(
  "++run_name=$RUN_NAME"
  "++generation.dataloader.dataset.nres.nsamples=$SAMPLES"
  "++generation.args.nsteps=$NSTEPS"
  "++generation.filter.filter_samples_limit=$SAMPLES"
)

# Count per-design directories, matching count_shard_sample_dirs' definition.
n_sample_dirs() {
  local d="$1" n=0 sub
  [[ -d "$d" ]] || { echo 0; return; }
  for sub in "$d"/job_*; do
    [[ -d "$sub" ]] || continue
    compgen -G "$sub/*.pdb" >/dev/null && n=$((n+1))
  done
  echo "$n"
}
n_caches()  { find "${1:-/nonexistent}" -name binder_eval_cache.json 2>/dev/null | wc -l | tr -d ' '; }
# Refolding outputs land in a subdirectory of the design dir (AF2/ for
# colabdesign, its own for other backends), so depth>=3 is backend-agnostic
# where a name like '*_self_seq_*.pdb' would silently match nothing under rf3
# and make the reuse assertion pass vacuously.
n_refolds() { find "${1:-/nonexistent}" -mindepth 3 -name '*.pdb' 2>/dev/null | wc -l | tr -d ' '; }
newer_refolds() { find "${1:-/nonexistent}" -mindepth 3 -name '*.pdb' -newer "$2" 2>/dev/null | wc -l | tr -d ' '; }
n_markers() { find "${1:-/nonexistent}" -maxdepth 1 -name 'shard_*_complete.json' 2>/dev/null | wc -l | tr -d ' '; }

gen()  { complexa generate "$CONFIG" "${BASE_OVERRIDES[@]}" "$@" >/dev/null; }
eval_() { complexa evaluate "$CONFIG" "${BASE_OVERRIDES[@]}" "$@" >/dev/null; }

# -----------------------------------------------------------------------------
say "1. first run writes a marker and fold caches"
gen || die "initial generate failed"
DIRS_1=$(n_sample_dirs "$RUN_DIR")
[[ "$(n_markers "$RUN_DIR")" -ge 1 ]] && ok "shard marker written" || bad "no shard_*_complete.json in $RUN_DIR"
[[ "$DIRS_1" -gt 0 ]] && ok "$DIRS_1 sample directories produced" || die "generate produced no sample directories"

complexa filter "$CONFIG" "${BASE_OVERRIDES[@]}" >/dev/null || die "filter failed"
eval_ || die "initial evaluate failed"
CACHES_1=$(n_caches "$EVAL_DIR")
REFOLDS_1=$(n_refolds "$EVAL_DIR")
[[ "$CACHES_1" -gt 0 ]] && ok "$CACHES_1 fold caches written" || bad "no binder_eval_cache.json under $EVAL_DIR"
# Guards steps 3 and 6 against passing vacuously: if refolding never wrote
# anything, "nothing was rewritten" proves nothing.
[[ "$REFOLDS_1" -gt 0 ]] && ok "$REFOLDS_1 refolding outputs written (reuse checks are meaningful)" \
  || die "no refolding outputs under $EVAL_DIR -- cannot tell reuse from inactivity"

python - "$EVAL_DIR" <<'PY' && ok "caches carry a fingerprint" || bad "a cache is missing its fingerprint"
import json, pathlib, sys
missing = [str(p) for p in pathlib.Path(sys.argv[1]).rglob("binder_eval_cache.json")
           if not json.loads(p.read_text()).get("fingerprint")]
sys.exit(1 if missing else 0)
PY

# -----------------------------------------------------------------------------
say "2. same config -> generate skips (no new sample directories)"
gen || die "second generate failed"
DIRS_2=$(n_sample_dirs "$RUN_DIR")
if [[ "$DIRS_2" -eq "$DIRS_1" ]]; then
  ok "still $DIRS_2 sample directories (skipped, nothing duplicated)"
else
  bad "directories went $DIRS_1 -> $DIRS_2; the shard regenerated instead of skipping"
fi

# -----------------------------------------------------------------------------
say "3. same config -> evaluate reuses every fold cache"
STAMP=$(mktemp); sleep 1
eval_ || die "second evaluate failed"
REFOLDED=$(newer_refolds "$EVAL_DIR" "$STAMP")
rm -f "$STAMP"
if [[ "$REFOLDED" -eq 0 ]]; then
  ok "none of the $REFOLDS_1 refolding outputs rewritten -- all $CACHES_1 caches reused"
else
  bad "$REFOLDED of $REFOLDS_1 refolding outputs rewritten; the cache was not honoured"
fi

# -----------------------------------------------------------------------------
say "4. changed generation parameter -> marker invalidated"
# From here on every generate uses ALT so the digest matches the marker this
# step leaves behind. Step 5 must differ from the marker in exactly one way --
# the missing directory -- or it would regenerate because of a stale digest and
# pass without testing anything.
ALT=("++generation.args.nsteps=$((NSTEPS + 1))")
gen "${ALT[@]}" || die "generate with changed nsteps failed"
DIRS_4=$(n_sample_dirs "$RUN_DIR")
if [[ "$DIRS_4" -gt "$DIRS_2" ]]; then
  ok "regenerated on a config change ($DIRS_2 -> $DIRS_4 directories)"
else
  bad "config changed but the shard was skipped anyway -- STALE RESULTS ACCEPTED"
fi

# -----------------------------------------------------------------------------
say "5. removed sample directory -> marker invalidated"
# Control: the same config now skips, so anything that changes below is the
# deletion talking and not the digest.
gen "${ALT[@]}" || die "control generate failed"
DIRS_5CTL=$(n_sample_dirs "$RUN_DIR")
if [[ "$DIRS_5CTL" -eq "$DIRS_4" ]]; then
  ok "control: matching digest skips ($DIRS_5CTL directories unchanged)"
else
  bad "control failed -- digest should have matched; steps below prove nothing"
fi

VICTIM=$(find "$RUN_DIR" -maxdepth 1 -type d -name 'job_*' | head -1)
[[ -n "$VICTIM" ]] || die "no sample directory to remove"
rm -rf "$VICTIM"
DIRS_5a=$(n_sample_dirs "$RUN_DIR")
[[ "$DIRS_5a" -lt "$DIRS_5CTL" ]] || die "removing $VICTIM did not reduce the directory count"
gen "${ALT[@]}" || die "generate after removal failed"
DIRS_5b=$(n_sample_dirs "$RUN_DIR")
if [[ "$DIRS_5b" -gt "$DIRS_5a" ]]; then
  ok "regenerated when output went missing ($DIRS_5a -> $DIRS_5b)"
else
  bad "a deleted design was not regenerated -- the shard is unrecoverable"
fi

# -----------------------------------------------------------------------------
if [[ $SKIP_BACKEND -eq 1 ]]; then
  say "6. folding-backend change -- SKIPPED (--skip-backend-check)"
  printf '   \033[33mNOTE\033[0m the stale-numbers guard was not exercised\n'
else
  say "6. changed folding backend -> fold caches invalidated"
  STAMP=$(mktemp); sleep 1
  if eval_ "++metric.binder_folding_method=rf3_latest" >/dev/null 2>&1; then
    REFOLDED_6=$(newer_refolds "$EVAL_DIR" "$STAMP")
    if [[ "$REFOLDED_6" -gt 0 ]]; then
      ok "refolded under a different backend ($REFOLDED_6 outputs) -- cache correctly rejected"
    else
      bad "backend changed but nothing refolded -- AF2 NUMBERS SERVED FOR AN RF3 RUN"
    fi
  else
    printf '   \033[33mNOTE\033[0m rf3_latest unavailable here; re-run with --skip-backend-check\n'
    printf '          to silence, or set RF3_CKPT_PATH to exercise this guard.\n'
  fi
  rm -f "$STAMP"
fi

# -----------------------------------------------------------------------------
printf '\n%s\n' "------------------------------------------------------------"
printf 'resume check: %d passed, %d failed\n' "$PASS" "$FAIL"
[[ $FAIL -eq 0 ]] || exit 1
