# Gating a Batch Campaign

How a campaign's own gate script decides whether a run may proceed — which model weights it
requires, how much output disk it needs, and why those decisions cannot live in
`preflight.sh`.

**Read this when generating, refreshing, or debugging a campaign gate** (typically a
`check_preflight.py` next to the campaign's `pipeline.yaml`). For the environment itself —
`.env` discovery, `env.sh`, the atomworks mirror variables — see
[`env-and-mirrors.md`](env-and-mirrors.md).

## Gate on the resolved config, not on a fixed list

`preflight.sh` reports **facts about the host** and is deliberately config-blind — it cannot
know which models a run needs, because that depends on a Hydra composition (defaults list,
`hydra.searchpath`, `_self_` ordering, `++` overrides) that bash has no way to resolve.
Reading `configs/pipeline/binder/binder_evaluate.yaml` directly is *wrong* for the same
reason: the effective config is whatever composes for *this* run.

So deciding what is **required** belongs to the gate, and the gate needs the resolved
config. Resolve it with Hydra itself:

```python
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent.resolve())):
    cfg = compose(config_name=cfg_path.stem, overrides=overrides)   # same overrides the run uses
OmegaConf.resolve(cfg)
OmegaConf.save(cfg, resolved_path, resolve=True)
```

then derive requirements from it rather than hardcoding a list:

```python
metric = resolved.get("metric", {})
required = set()
if metric.get("compute_binder_metrics"):
    backend = metric.get("binder_folding_method", "")
    if backend == "colabdesign":  required.add("AF2_DIR")
    elif "rf3" in backend:        required.add("RF3_CKPT_PATH")
if metric.get("compute_esm_metrics"):      required.add("ESM_DIR")
if metric.get("compute_monomer_metrics"):  required.add("ESMFOLD")

cm = preflight["community_models"]
for name in sorted(required):
    entry = cm.get(name, {})
    # HF caches report has_weights; plain paths only report exists
    ok = entry.get("has_weights", entry.get("exists", False))
    if not ok:
        failures.append(f"{name}: {entry.get('path') or '<unset>'} has no usable weights")
```

Two things this gets right that a fixed list does not:

- **`AF2_DIR` is only needed for `colabdesign`.** Switch `binder_folding_method` to
  `rf3_latest` and you need `RF3_CKPT_PATH` instead — a hardcoded list would demand AF2 and
  ignore RF3.
- **`ESMFOLD` is checked at all.** Nothing else in the stack probes it, so
  `compute_monomer_metrics: true` (the default) otherwise fails at evaluate time with the
  preflight showing green.

**Prefer `has_weights` over `exists` for HF caches.** `exists` is satisfied by an empty
`mkdir` and by no loader — creating the directory to silence a preflight failure converts a
cheap gate failure into a `RuntimeError` partway through evaluation, after generation has
already spent the GPU time. `preflight.sh` reports both: `exists` keeps its original
path-existence meaning, `has_weights` additionally looks for the repo's own
`models--org--name` snapshot under the cache root.

**Gate disk on `cwd_free_gb`, not `free_gb`.** Two different filesystems answer two
different questions, and `preflight.sh` now reports both:

| Field | Filesystem | Answers |
|---|---|---|
| `ckpt_free_gb` (alias `free_gb`), `ckpt_fs` | wherever `CKPT_PATH` lives — the install | room to **download** more weights |
| `cwd_free_gb`, `cwd_fs` | the working directory | room for **this run's outputs** |

`./inference/…` (`generate.py:62`) and `./logs` (`cli_runner.py:128`) are cwd-relative, so a
campaign run writes nowhere near `CKPT_PATH`. Gating a design run on `free_gb` therefore
measures the wrong volume — it can fail on a full install disk while the output volume is
empty, or pass while the output volume is full.

**Symlinks are handled; equal free space is not proof of a shared volume.** `df` and the
`-d` test both resolve symlinks in the kernel, so a `CKPT_PATH` or campaign directory that
is a symlink onto another filesystem is measured on its *target* volume — no `readlink`
needed. But shared-pool filesystems (APFS containers, Btrfs subvolumes, thin LVM) report
the *same* free figure for genuinely distinct mounts, so identical `free_gb` values tell you
nothing. Compare the mount points instead:

```python
shared = disk.get("ckpt_fs") and disk["ckpt_fs"] == disk["cwd_fs"]
```

When they match, weights and outputs compete for the same space and the two budgets must be
added. Two edge cases inherit the original fallback behaviour: a *dangling* symlink and a
symlink to a *file* both fall back to measuring the parent directory, so the reported
`*_fs` is the parent's mount, not the target's.

Size the requirement from the design count rather than a fixed number.
`_shared/reference/hardware.md` puts protein-binder output at **~10–20 GB per 100 designs**,
and `keep_folding_outputs: true` (the eval default) roughly doubles it:

```python
per_100 = 20                                  # GB, upper end of the empirical range
factor  = 2 if metric.get("keep_folding_outputs", True) else 1
need_gb = max(5, int(expected_designs / 100 * per_100 * factor))
if disk.get("cwd_free_gb") is not None and disk["cwd_free_gb"] < need_gb:
    failures.append(f"{disk['cwd_free_gb']} GB free at {disk['cwd']}; ~{need_gb} GB needed "
                    f"for {expected_designs} designs")
```

A 1000-design run lands near **200–400 GB**, which is why a fixed 50 GB floor is both too
strict for a smoke test and far too loose for production.

## The gate must compose the same overrides as the run

Gating on a resolved config is only meaningful if that config is the one the run will
actually use. If the resolver and the `complexa design` invocation are handed *different*
override lists, the gate is checking a config nobody runs — and it will reject valid runs
and pass broken ones.

This is easy to get wrong, because the natural shape is two hardcoded lists:

```python
# resolver — reconstructs overrides from named args
overrides = [f"++run_name={args.run_name}",
             f"++generation.dataloader.dataset.nres.nsamples={args.seed_samples}", ...]
```

```bash
# runner — passes its own copy
python -m proteinfoundation.cli.cli_runner design "$CONFIG" \
  "++run_name=$RUN_NAME" "++generation.dataloader.dataset.nres.nsamples=$SEED_SAMPLES" ...
```

Those agree only while someone keeps them in sync by hand. Add `++metric.compute_monomer_metrics=false`
to the runner and the resolved config still says `true`, so the gate demands ESMFold weights
for a run that never loads them.

**Define the override list once and pass it to both:**

```bash
COMPLEXA_OVERRIDES=(
  "++run_name=$RUN_NAME"
  "++generation.dataloader.dataset.nres.nsamples=$SEED_SAMPLES"
  "++generation.filter.filter_samples_limit=$FILTER_LIMIT"
)
# per-kind additions go here, so the resolver sees them too
[[ "$RUN_KIND" == "smoke" ]] && COMPLEXA_OVERRIDES+=( "++metric.compute_monomer_metrics=false" )

python scripts/validate_resolved_config.py --config "$CONFIG" ... \
    --override "${COMPLEXA_OVERRIDES[@]}" --output "$RESOLVED"
python scripts/check_preflight.py "$PREFLIGHT" --resolved-config "$RESOLVED" ...
python -m proteinfoundation.cli.cli_runner design "$CONFIG" "${COMPLEXA_OVERRIDES[@]}"
```

and in the resolver, compose from the passthrough list rather than rebuilding it:

```python
parser.add_argument("--override", nargs="*", default=[],
                    help="Hydra overrides — must be exactly those passed to `complexa design`")
...
cfg = compose(config_name=args.config.stem, overrides=list(args.override))
```

Keep the resolver's named args (`--seed-samples`, `--expected-generated`, …) for its
*invariant assertions*. They then serve a second purpose: they express intent independently
of the override list, so an assertion like
`cfg.generation.dataloader.dataset.nres.nsamples == args.seed_samples` fails loudly if the
two ever disagree. That turns the duplication into a consistency check instead of a
liability.

A setting that should apply to *every* run of a campaign belongs in `pipeline.yaml` instead
— both the resolver and the runner compose the same file, so it cannot drift. Reserve the
override list for what varies between run kinds.

## Acceptance checks: count what the stage actually consumed

The post-run acceptance check (`verify_run_outputs.py`-style) fails the same way the gate
does — by asserting on a number that looks like the right one. Three real defects, all found
by a smoke test that had otherwise produced correct science:

**1. `timing_*.csv` catches the analyze stage's own summary.** The evaluate stage writes one
`timing_{job_id}.csv` per worker with `job_id,evaluation_time_s,nsamples,evals_run`
(`evaluate.py:971-974`). The analyze stage then writes `timing_summary.csv` **into the same
directory** with a completely different schema — `eval_config,num_jobs,…,total_samples,…`,
no `nsamples` column (`result_analysis/analysis.py:1765`). A `timing_*.csv` glob picks up
both and `row["nsamples"]` raises `KeyError` on the summary row. Match the digits:

```python
_TIMING_CSV_RE = re.compile(r"^timing_\d+\.csv$")
timing_csvs = sorted(p for p in evaluation_dir.glob("timing_*.csv")
                     if _TIMING_CSV_RE.match(p.name))
```

The repo's own reader already does this, for this reason — `analysis.py:1671-1672` documents
it as avoiding "aggregated outputs like `timing_summary.csv`". Follow the convention that
exists rather than inventing a glob against it.

**2. Do not assert a worker count.** `if len(timing_csvs) != 2` looks like a check that both
GPUs did work. It is not: the number of evaluate workers is a scheduling detail, and one
worker plus one summary file happens to equal two — which is how defect 1 stayed hidden
until the count was fixed. Assert on *what ran* instead, which a file count can never
capture, using the same `compute_*_metrics` flags the gate reads (`evaluate.py:108-111`,
`960-968`):

```python
ran = {t for row in timing_rows for t in (row.get("evals_run") or "").split("+") if t}
wanted = {name for name, flag in _TRACK_FLAGS.items() if metric.get(flag)}
if ran != wanted:
    raise SystemExit(f"evaluation ran {sorted(ran)}; resolved config expects {sorted(wanted)}")
```

Pass the acceptance check the **same** `--resolved-config` the gate consumed. That closes the
loop opened in the previous section: one resolved config now feeds the pre-run gate *and* the
post-run acceptance check, so "what we required", "what we ran", and "what we verified" cannot
diverge.

**3. `top_samples_*.csv` is a report, not the evaluation input set.** This is the subtle one.
It is tempting to read the filter stage as a funnel — generate N, filter to M, evaluate M —
and assert `evaluated == M`. Two things break that:

- The filter **deduplicates by sequence** (`filter.py:148-150`), so `top_samples_*.csv` can
  be smaller than the sample set for reasons that have nothing to do with filtering.
- The entire pruning branch is guarded by `if len(combined_rewards) > filter_samples_limit`
  (`filter.py:173`). Below the limit it logs `No filtering needed` and **leaves every sample
  directory in place** — nothing is deleted, nothing is moved to `filtered_out_samples/`.

So a smoke test generating 8 with `filter_samples_limit: 8` yields 8 generated, 6 rows in
`top_samples_pipeline.csv` (2 duplicate sequences dropped), 8 directories on disk, and 8
evaluated. `evaluated == survivors` fails on a run where nothing whatsoever went wrong.
Nothing in `src/` reads `top_samples_*.csv` back, which is the tell — it bounds the run, it
does not define it.

Count the directories the stage could actually see:

```python
def sample_dirs(inference_dir):
    return sorted(p for p in inference_dir.iterdir()
                  if p.is_dir() and p.name not in {"filtered_out_samples", "timing"}
                  and any(p.glob("*.pdb")))
```

then assert `evaluated == len(sample_dirs(...))` for coverage, `0 < deduped <= generated` as a
report sanity bound, and — when `filtered_out_samples/` exists — that live plus filtered-out
equals generated. That last one is the only assertion that actually proves the filter did what
it was asked, and it is well defined whether or not pruning happened.

**Know which of your numbers are independent.** The timing CSV's `nsamples` is
`max(len(df))` over the result frames (`evaluate.py:944-946`) — the same frames the combined
CSV is written from. So `evaluated == combined` is a schema guard, not a cross-check; keep it,
but do not mistake it for evidence that evaluation covered the run. The genuinely independent
numbers are the generation reward rows, the on-disk directory count, and the result rows.

## The pattern behind all of these

`preflight.sh` reports facts; the gate applies thresholds. Every gate failure in this
document came from breaking that split in one of three ways:

- **Measuring the wrong subject** — a `.env` file instead of the variables, `$DATA_PATH/target_data`
  when the target came from `target_path`, the install filesystem instead of the output one.
- **Measuring the wrong depth** — path existence instead of contents.
- **Measuring the wrong config** — a resolved config composed with different overrides than
  the run will use, so the gate judges a config nobody executes.
- **Measuring a derived number** — a report artifact (`top_samples_*.csv`) or a value computed
  from the very frames under test (`nsamples`), mistaken for an independent observation.

The common shape is *two things that must agree, with nothing enforcing it*. When you find
one, either collapse them to a single source (one override list, one `.env`, one checkpoint
directory convention) or add an assertion that fails when they diverge.

When adding a probe: report the *narrowest true fact* about the *right subject*, name the
field so the subject is unambiguous (`cwd_free_gb`, not `free_gb`), and leave the threshold
to whoever knows the config.

**Order matters.** Resolve the config *before* the gate runs, so a config that will not
compose fails before you probe hardware and before the gate has nothing to read.

