#!/usr/bin/env bash
# Verify that in-stage resume actually reuses work, on the machine that can run it.
#
#   bash check_resume.sh --config /path/to/pipeline.yaml [--samples 2] [--nsteps N]
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

CONFIG=""; SAMPLES=2; NSTEPS=""; KEEP=0; SKIP_BACKEND=0
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

# The CLI refuses to run without COMPLEXA_INIT, and `.env` has no export lines,
# so a caller who merely sourced `.env` gets empty path variables. Check both
# here rather than letting `complexa generate` fail thirty lines into a log.
command -v complexa >/dev/null 2>&1 || {
  echo "complexa is not on PATH -- activate the environment first" >&2; exit 2; }
if [[ -z "${COMPLEXA_INIT:-}" ]]; then
  cat >&2 <<'MSG'
COMPLEXA_INIT is unset, so every `complexa` subcommand this script runs would
abort with "Environment not initialized". Source the generated env.sh, from
bash, with allexport on:

    set -a; source "$COMPLEXA_REPO/env.sh"; set +a

See docs/binder-target-setup/env-and-mirrors.md ("The env.sh export gap").
MSG
  exit 2
fi
for _k in LOCAL_CODE_PATH CKPT_PATH; do
  [[ -n "${!_k:-}" ]] || { echo "$_k is empty -- source env.sh with 'set -a' (the export gap)" >&2; exit 2; }
done

LOGDIR="$(mktemp -d)"       # before the first thing that can fail
RUN_NAME="resume_check_$$"
CONFIG_NAME="$(basename "${CONFIG%.*}")"
PASS=0; FAIL=0

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '   \033[31mFAIL\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
die()  { printf '\n\033[31mABORT\033[0m %s\n' "$*" >&2; exit 1; }

# Resolve the run directory the way generate.py does: ./inference/<config>_<task>[_<run>]
# The task name comes from the resolved config, not from a guess.
TASK_NAME="$(python - "$CONFIG" 2>"$LOGDIR/00_task_name.log" <<'PY'
import sys, pathlib
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
p = pathlib.Path(sys.argv[1]).resolve()
with initialize_config_dir(version_base=None, config_dir=str(p.parent)):
    cfg = compose(config_name=p.stem)
ckpts = OmegaConf.select(cfg, "generation.search.step_checkpoints") or []
print(OmegaConf.select(cfg, "generation.task_name") or "")
print(max([int(c) for c in ckpts], default=0))

# Report environment variables the config interpolates but the shell lacks. The
# run needs them (target_path is commonly built from a campaign directory
# variable), and failing here beats failing after the model has loaded.
missing = []
for key in ("generation.dataloader.dataset.conditional_features",
            "generation.target_dict_cfg", "generation"):
    node = OmegaConf.select(cfg, key)
    if node is None:
        continue
    try:
        OmegaConf.to_yaml(node, resolve=True)
    except Exception as exc:                       # noqa: BLE001 - message is the payload
        for tok in str(exc).replace("'", " ").split():
            if tok.isupper() and "_" in tok and tok not in missing:
                missing.append(tok)
        break
print(",".join(missing))
PY
)" || {
  printf '\n\033[31mABORT\033[0m could not compose %s to read generation.task_name.\n' "$CONFIG" >&2
  printf '        The run directory name is derived from it, so this has to work first.\n' >&2
  printf '        Check that hydra imports in the active environment and that the config\n' >&2
  printf '        composes at all: complexa validate design %s\n\n' "$CONFIG" >&2
  sed 's/^/    /' "$LOGDIR/00_task_name.log" >&2
  rm -rf "$LOGDIR"
  exit 1
}
MISSING_ENV="$(printf '%s\n' "$TASK_NAME" | sed -n '3p')"
MAX_STEP_CKPT="$(printf '%s\n' "$TASK_NAME" | sed -n '2p')"
TASK_NAME="$(printf '%s\n' "$TASK_NAME" | sed -n '1p')"

if [[ -n "$MISSING_ENV" ]]; then
  cat >&2 <<MSG

ABORT the config interpolates environment variables this shell does not define:
      ${MISSING_ENV//,/, }

      Campaign configs commonly build target_path from a campaign-directory
      variable, and the runner exports it while an interactive shell does not.
      Export it and re-run, e.g.:

          export ${MISSING_ENV%%,*}="\$PWD"

      (Generation would otherwise load the model, then fail resolving the
      target PDB path.)
MSG
  rm -rf "$LOGDIR"; exit 2
fi
[[ -n "$TASK_NAME" ]] || die "generation.task_name is empty in $CONFIG -- pin it under _self_"

# Search schedules are absolute step indices, not fractions. A config with
# step_checkpoints [0,100,200,300,400] and nsteps=50 puts every checkpoint but
# the first past the end of the trajectory, and generation fails -- which is
# exactly how the first version of this script broke, by defaulting nsteps=50.
# So nsteps is left alone unless asked for, and refused when it would not fit.
if [[ -n "$NSTEPS" ]]; then
  if [[ "${MAX_STEP_CKPT:-0}" -gt "$NSTEPS" ]]; then
    cat >&2 <<MSG

ABORT --nsteps $NSTEPS is smaller than this config's largest search checkpoint
      (${MAX_STEP_CKPT}). generation.search.step_checkpoints are absolute step
      indices, so the search schedule would point past the end of the
      trajectory. Drop --nsteps (the config's own value is used, which is what
      a resume check wants anyway) or pass at least ${MAX_STEP_CKPT}.
MSG
    rm -rf "$LOGDIR"; exit 2
  fi
  BASE_OVERRIDES_NSTEPS=("++generation.args.nsteps=$NSTEPS")
else
  BASE_OVERRIDES_NSTEPS=()
fi

RUN_DIR="./inference/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
EVAL_DIR="./evaluation_results/${CONFIG_NAME}_${TASK_NAME}_${RUN_NAME}"
printf 'config      %s\ntask_name   %s\nrun_name    %s\ncwd         %s\nrun dir     %s\n' \
  "$CONFIG" "$TASK_NAME" "$RUN_NAME" "$PWD" "$RUN_DIR"
# ./inference and ./evaluation_results are cwd-relative with no chdir anywhere
# in the codebase, so run this from the campaign directory.
[[ -w . ]] || die "cwd is not writable; run this from the campaign directory"

KEEP_LOGS=0
cleanup() {
  if [[ $KEEP -eq 1 ]]; then
    printf '\nkept %s and %s\n' "$RUN_DIR" "$EVAL_DIR"
  else
    rm -rf "$RUN_DIR" "$EVAL_DIR"
  fi
  if [[ ${KEEP_LOGS:-0} -eq 1 || $KEEP -eq 1 ]]; then
    printf 'step logs in %s\n' "$LOGDIR"
  else
    rm -rf "$LOGDIR"
  fi
}
trap cleanup EXIT

# Pin the shard counts to 1. The campaign config may set gen_njobs=2, which
# would (a) split --samples across shards so this script's job 0 produces half
# of them, and (b) take run_step's parallel path, where the real error goes to
# per-job stage logs instead of stdout. One shard keeps the counts predictable;
# the marker mechanism under test is per-shard either way.
BASE_OVERRIDES=(
  "++run_name=$RUN_NAME"
  "++generation.dataloader.dataset.nres.nsamples=$SAMPLES"
  "++generation.filter.filter_samples_limit=$SAMPLES"
  "++gen_njobs=1"
  "++eval_njobs=1"
  ${BASE_OVERRIDES_NSTEPS[@]+"${BASE_OVERRIDES_NSTEPS[@]}"}
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
# Total designs this run owns, wherever they now live. The filter stage moves
# non-surviving designs into filtered_out_samples/, so a live-only count drops
# across a filter even though nothing was lost -- comparing live counts either
# side of a filter reports a correct skip as a regeneration.
n_all_sample_dirs() {
  local d="$1"
  echo $(( $(n_sample_dirs "$d") + $(n_sample_dirs "$d/filtered_out_samples") ))
}
n_caches()  { find "${1:-/nonexistent}" -name binder_eval_cache.json 2>/dev/null | wc -l | tr -d ' '; }
# Refolding outputs land in a subdirectory of the design dir (AF2/ for
# colabdesign, its own for other backends), so depth>=3 is backend-agnostic
# where a name like '*_self_seq_*.pdb' would silently match nothing under rf3
# and make the reuse assertion pass vacuously.
n_refolds() { find "${1:-/nonexistent}" -mindepth 3 -name '*.pdb' 2>/dev/null | wc -l | tr -d ' '; }
newer_refolds() { find "${1:-/nonexistent}" -mindepth 3 -name '*.pdb' -newer "$2" 2>/dev/null | wc -l | tr -d ' '; }
n_markers() { find "${1:-/nonexistent}" -maxdepth 1 -name 'shard_*_complete.json' 2>/dev/null | wc -l | tr -d ' '; }

STEP=0

# Run a complexa step quietly, but print the tail of its log if it fails. The
# first version of this script sent stdout to /dev/null and reported only
# "initial generate failed", which is useless: the whole point of a checker is
# to say what went wrong.
run_cx() {
  local what="$1"; shift
  STEP=$((STEP+1))
  local log="$LOGDIR/$(printf '%02d' "$STEP")_${what}.log"
  # Capture the status before anything else runs. Inside `if ! cmd; then`, $? is
  # the status of the negation, which is always 0 -- the first version of this
  # reported every failure as "exit 0".
  local rc=0
  complexa "$@" >"$log" 2>&1 || rc=$?
  if (( rc != 0 )); then
    printf '\n\033[31mABORT\033[0m complexa %s failed (exit %d). Tail of %s:\n' \
      "$what" "$rc" "$log" >&2
    tail -40 "$log" | sed 's/^/    /' >&2
    # Without --verbose the wrapper logs the real error to a stage file and only
    # re-raises CalledProcessError, so the captured stdout holds a traceback
    # about the traceback. We pass --verbose, but a stage log may still exist.
    local stage
    stage="$(ls -t ./logs/${what}_*"$RUN_NAME"*.log 2>/dev/null | head -1)"
    if [[ -n "$stage" ]]; then
      printf '\n    --- tail of stage log %s ---\n' "$stage" >&2
      tail -30 "$stage" | sed 's/^/    /' >&2
    fi
    printf '\n    full logs kept in %s\n' "$LOGDIR" >&2
    KEEP_LOGS=1
    exit 1
  fi
}

gen()   { run_cx generate "generate" "$CONFIG" --verbose "${BASE_OVERRIDES[@]}" "$@"; }
eval_() { run_cx evaluate "evaluate" "$CONFIG" --verbose "${BASE_OVERRIDES[@]}" "$@"; }

# -----------------------------------------------------------------------------
say "1. first run writes a marker and fold caches"
gen
DIRS_1=$(n_all_sample_dirs "$RUN_DIR")
[[ "$(n_markers "$RUN_DIR")" -ge 1 ]] && ok "shard marker written" || bad "no shard_*_complete.json in $RUN_DIR"
[[ "$DIRS_1" -gt 0 ]] && ok "$DIRS_1 sample directories produced" || die "generate produced no sample directories"

run_cx filter "filter" "$CONFIG" --verbose "${BASE_OVERRIDES[@]}"
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
gen
DIRS_2=$(n_all_sample_dirs "$RUN_DIR")
SKIP_WORKED=0
if [[ "$DIRS_2" -eq "$DIRS_1" ]]; then
  ok "still $DIRS_2 designs owned by this run (skipped, nothing duplicated)"
  SKIP_WORKED=1
else
  bad "designs went $DIRS_1 -> $DIRS_2; the shard regenerated instead of skipping"
fi

# -----------------------------------------------------------------------------
say "3. same config -> evaluate reuses every fold cache"
if [[ $SKIP_WORKED -eq 0 ]]; then
  # Step 2 leaving extra designs behind guarantees refolds here, so a failure
  # would just be step 2's echo. Reporting it as a second finding sends you
  # looking for a cache bug that is not there.
  printf '   \033[33mSKIP\033[0m step 2 regenerated, so new designs would refold regardless\n'
else
STAMP=$(mktemp); sleep 1
eval_ || die "second evaluate failed"
REFOLDED=$(newer_refolds "$EVAL_DIR" "$STAMP")
rm -f "$STAMP"
if [[ "$REFOLDED" -eq 0 ]]; then
  ok "none of the $REFOLDS_1 refolding outputs rewritten -- all $CACHES_1 caches reused"
else
  bad "$REFOLDED of $REFOLDS_1 refolding outputs rewritten; the cache was not honoured"
fi
fi

# -----------------------------------------------------------------------------
say "4. changed generation parameter (batch_size) -> marker invalidated"
# From here on every generate uses ALT so the digest matches the marker this
# step leaves behind. Step 5 must differ from the marker in exactly one way --
# the missing directory -- or it would regenerate because of a stale digest and
# pass without testing anything.
# The perturbation has to sit inside `generation`, because that is the subtree
# the marker digest hashes -- top-level `seed` would leave the digest unchanged
# and this step would fail for the wrong reason. batch_size qualifies, changes
# no counts, and cannot invalidate an absolute search schedule the way nsteps
# does.
ALT=("++generation.dataloader.batch_size=1")
gen "${ALT[@]}"
DIRS_4=$(n_all_sample_dirs "$RUN_DIR")
if [[ "$DIRS_4" -gt "$DIRS_2" ]]; then
  ok "regenerated on a config change ($DIRS_2 -> $DIRS_4 directories)"
else
  bad "config changed but the shard was skipped anyway -- STALE RESULTS ACCEPTED"
fi

# -----------------------------------------------------------------------------
say "5. removed sample directory -> marker invalidated"
# Control: the same config now skips, so anything that changes below is the
# deletion talking and not the digest.
gen "${ALT[@]}"
DIRS_5CTL=$(n_all_sample_dirs "$RUN_DIR")
if [[ "$DIRS_5CTL" -eq "$DIRS_4" ]]; then
  ok "control: matching digest skips ($DIRS_5CTL directories unchanged)"
else
  bad "control failed -- digest should have matched; steps below prove nothing"
fi

# Delete a directory the *current* marker recorded. Picking any job_* directory
# can hit one from an earlier phase that this marker never listed, in which case
# its recorded set is still intact, the shard rightly skips, and the assertion
# fails while the code is correct.
MARKER="$RUN_DIR/shard_0_complete.json"
[[ -f "$MARKER" ]] || die "no shard marker at $MARKER"
VICTIM_NAME=$(python - "$MARKER" <<'PYV'
import json, sys
names = json.load(open(sys.argv[1])).get("sample_dirs") or []
print(names[0] if names else "")
PYV
)
[[ -n "$VICTIM_NAME" ]] || die "marker records no sample_dirs; cannot test the deletion path"
VICTIM="$RUN_DIR/$VICTIM_NAME"
[[ -d "$VICTIM" ]] || VICTIM="$RUN_DIR/filtered_out_samples/$VICTIM_NAME"
[[ -d "$VICTIM" ]] || die "recorded directory $VICTIM_NAME is already absent"
printf '   removing recorded design %s\n' "$VICTIM_NAME"
rm -rf "$VICTIM"
DIRS_5a=$(n_all_sample_dirs "$RUN_DIR")
gen "${ALT[@]}"
DIRS_5b=$(n_all_sample_dirs "$RUN_DIR")
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
  if complexa evaluate "$CONFIG" "${BASE_OVERRIDES[@]}" \
       "++metric.binder_folding_method=rf3_latest" >"$LOGDIR/99_rf3.log" 2>&1; then
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
