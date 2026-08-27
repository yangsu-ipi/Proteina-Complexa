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

`./inference/…` (`generate.py:68`) and `./logs` (`cli_runner.py:128`) are cwd-relative, so a
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
(`evaluate.py:989-992`). The analyze stage then writes `timing_summary.csv` **into the same
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
capture, using the same `compute_*_metrics` flags the gate reads (`evaluate.py:109-112`,
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
report sanity bound, and — **unconditionally** — that live plus filtered-out directories equals
generated. Assert that last one even when nothing was pruned: it is the only check that proves
the filter did what it was asked, and it doubles as the detector for the stale-directory hazard
in the next section.

**Know which of your numbers are independent.** The timing CSV's `nsamples` is
`max(len(df))` over the result frames (`evaluate.py:962-964`) — the same frames the combined
CSV is written from. So `evaluated == combined` is a schema guard, not a cross-check; keep it,
but do not mistake it for evidence that evaluation covered the run. The genuinely independent
numbers are the generation reward rows, the on-disk directory count, and the result rows.

## Restarting a failed or interrupted run

Resume granularity is **the stage**, and only manually. A failed stage raises
`CalledProcessError` (`cli_runner.py:793-797`) which propagates out of `run_design_pipeline`, so
later stages never run and you restart with `--steps`:

```bash
complexa design ./pipeline.yaml --steps evaluate analyze "${COMPLEXA_OVERRIDES[@]}"
```

Pass the *same* override array — a resumed leg composed differently is the same divergence the
gate section warns about.

Within a stage there is no checkpointing, and two details make a naive retry of `generate`
actively dangerous:

- **Nothing is persisted until sampling finishes.** `trainer.predict` returns every batch
  prediction in memory (`generate.py:1566`); only afterwards does `save_predictions` write the
  PDBs and `save_rewards_to_csv` write the rewards CSV (`:1400`, called at `:1483`/`:1508`, plain
  `to_csv`, no append). An interruption during sampling — the long part — therefore loses the
  entire shard and leaves no partial state to resume from. The same structure means peak memory
  scales with the design count rather than the batch size.
- **A retry's directory names mostly miss the previous attempt's, and the ones that hit are
  overwritten silently.** The per-design directory name encodes the beam-search path
  (`job_0_n_195_id_3_beam_orig0_bm0-s0to100br3-…`) and the `id_N` counter restarts at zero each
  run, so most regenerated designs land on fresh names — evaluation then folds designs belonging
  to no run and the counts inflate. But the beam path is not perfectly reproducible, and it is
  not perfectly *un*reproducible either: a measured 16-design retry produced only 11 new
  directories, so five names collided and were replaced in place
  (`os.makedirs(..., exist_ok=True)` plus a `write_prot_to_pdb(..., overwrite=True)`). So a
  retry both accumulates orphans *and* destroys some of the previous attempt's structures, with
  no record of which. That was the stronger reason to clear the directory rather than retry
  over it: the counts you can detect afterwards, the overwritten designs you cannot. It is
  also why the marker logic below exists — and once it did, "clear the directory" stopped
  being the answer. See **Do not guard a runner on the output directory existing**.

This used to be unguarded. `generate.py` had an early-exit keyed on
`results_{config_name}_{job_id}.csv`, a filename nothing in the codebase writes — evaluate
writes the prefixed forms `binder_results_…`, `monomer_results_…` (`evaluate.py:871-948`) — so
the guard was dead and generate always restarted from scratch.

### Do not guard a runner on the output directory existing

Campaign runners are generated from this file, and an earlier version of it said, in bold, to
clear the run directory before re-running generate over it. That advice predates the
completion marker and is now actively wrong: a runner that does

```bash
[[ ! -e "$INF" ]] || { echo "refusing to generate over $INF" >&2; exit 2; }
```

**disables resume.** It cannot tell a completed shard from a damaged one or from a different
config, so it refuses all three — including the case resume exists for, where the digest
matches and there is nothing to do but skip. It also replaces per-shard reasoning with a
whole-campaign veto, so one bad shard blocks fifteen good ones.

Let generation decide. It already distinguishes every case, per shard, and every branch either
skips safely or aborts with a message naming the recovery (see the table above). A runner's job
is to pass `--job-id` and get out of the way.

Two things a runner *may* usefully do: fail fast when the config digest will obviously mismatch
(a deliberate config change), and make its own auxiliary steps **idempotent** rather than
fatal — a trim or move step that refuses because its destination already exists turns a
resumable run into a manual cleanup, which is the same mistake one layer up.

It now keys on a **completion marker** instead. A finished shard writes
`shard_{job_id}_complete.json` recording a SHA-256 of its `generation` config subtree, and the
next run compares digests:

| Marker state | Behaviour (default `skip_completed_shards: true`) |
|---|---|
| absent, output root empty for this job | generate, silently |
| **absent, but this job's directories exist** | **abort** — an interrupted run, not a new one |
| digest matches, every recorded file present and nonempty | **skip the shard** |
| digest matches, a recorded file gone, empty or unreadable | clear what survived, regenerate |
| digest matches, marker predates per-file records | each recorded directory must still hold a usable design |
| digest matches, marker claims samples but records no outputs | **abort** — it disagrees with itself |
| **digest differs** | **abort** — a different request already owns this directory |
| **older digest formula, v1 digest matches this config** | same request; behave as if it matched |
| **older digest formula, v1 digest does not match** | **abort** — cannot tell whether the config changed |
| **unreadable** | **abort** — a marker exists only after output does, and its contents cannot be checked |
| **a clear leaves anything behind** | **abort**, keeping the marker — see below |
| **forced rerun, marker names no directories** | **abort** — cannot identify what it would replace |

Aborting rather than continuing, because directory names are
`job_{job}_n_{length}_id_{counter}` with the counter restarting each run and PDBs written
`overwrite=True`. Only a metadata tag (a beam path) makes them differ, so for pipelines
without one, continuing overwrote structures and left the previous run's evaluation files
beside the new designs. The recovery is a new `generation.run_name` or a clean directory.

Markers written before the digest was versioned are recognised where possible: the older
formula is recomputed for the current config, including for each value the operational keys
it used to hash might have held. So an unchanged campaign resumes normally, and only a
genuinely unrecognisable marker is refused.

Clearing verifies rather than trusts. `clear_shard_output` reports which recorded directories
are still on disk *after* the attempt, not which deletions raised, because a delete that
reports success and leaves the directory is the case that matters. If anything remains the
run aborts and the **marker is kept** — it is the only record of which output belongs to
that shard, so deleting it after a failed clear would remove the means of recovery. Root-level
outputs are cleared too, not only sample directories: once reward CSVs became recorded output,
removing the directories stopped clearing the shard, and the residue showed up as a repeated
GPU run that then failed writing over the CSV it had not removed.

**A directory outlives its contents.** The marker records every file a shard produced,
relative to the output root, not just the directories holding them — a PDB can be deleted,
truncated to zero bytes or left unreadable while its parent stays put, and a directory-level
check then reports the shard complete, with evaluation the place it surfaces. Ligand
generation writes a protein-ligand complex PDB beside every binder, and those are recorded
too: `nsamples` counts designs, `outputs` counts files. Relocation into
`filtered_out_samples/` is not a loss for a file *inside* a sample directory — the filter
moves those. It is not a fallback for root-level files, which nothing moves and which the
filter looks for in the output root and nowhere else; accepting a reward CSV there reported
the shard complete while filtering could not see it.

A recorded output counts only if it is present, nonempty **and readable** — `isfile` and
`getsize` both read metadata without opening anything, so a mode-000 PDB passed a check that
claimed otherwise. One byte is read.

Markers written before `outputs` existed (`marker_schema_version` below 2) cannot name every
file they expect, but they can name one: all three save paths write `{dir}/{dir}.pdb`, so the
design is reconstructable from a recorded directory name. Requiring *that* file rather than
any `.pdb` matters because the other PDBs in a sample directory were vouching for it — a
ligand sample holds its complex as `{dir}.pdb` and a binder beside it, and binder evaluation
writes a `{dir}_binder.pdb` sidecar into ordinary sample directories, which outlives the
design it was extracted from while evaluation itself skips the design when the base PDB is
gone. Which files a sample needs comes from the *contract* rather than from the marker: a
ligand sample needs its binder as well as its complex, and a marker without `outputs` does
not record that it was a ligand run — so the config is asked instead, which the matching
digest is what licenses. A marker at or above the schema that introduced `outputs` and recording
none against a positive sample count is contradictory rather than merely old, and aborts.

Reward CSVs are shard output, not a side effect: the filter stage reads
`rewards_{config}_{job}.csv` and raises `No reward files found!` without them, and across
several shards it processes the ones it can see — so evaluation covers fewer designs than
generation claimed, silently. They are recorded and verified like any other file.

Schema 3 is what records them, and schema 2 markers exist on both sides of that change while
claiming the same version, so the marker cannot be asked whether a reward CSV is required.
The config is asked instead: generation passes the root-level files the *current* config
would produce for this shard, and they are verified whether or not the marker names them.
Motif generation writes no rewards, and a shard whose marker says it produced nothing wrote
none either.

**Which save path a config takes is one rule.** `generation_save_branch` decides it and both
the save dispatch and the resume check read it from there, because the conditions are not
exclusive: the shipped AME config sets *both* MotifFeatures and LigandFeatures, so an AME run
is a ligand save with a motif dataset. Saving asks `ligand?` first; a resume check that asked
`motif?` first classified every AME shard as reward-free while saving wrote a reward CSV per
shard — each site correct read alone, in incompatible orders.

The motif contig table is required output too, and hangs off the MotifFeatures mode rather
than the save branch. MotifFeatures writes `{task_name}_{job_id}_motif_info.csv` into the
output root whenever `motif_atom_spec` is unset, and indexed motif evaluation *requires* it
to map samples to contigs — it raises `FileNotFoundError` without it. So a shard that skipped
after the file was deleted left evaluation unable to run at all. A config can set both
MotifFeatures and LigandFeatures, save on the ligand branch, and still owe this file, which is
why it is a separate input to the contract rather than a property of the branch. Markers
written from here on record the file, so the config-derived requirement only has to carry
markers that predate it — and where `motif_atom_spec` interpolates over a variable that is
unset, the requirement is dropped rather than assumed: being wrong that way costs a loud
`FileNotFoundError` naming the file, while being wrong the other way clears a completed
shard's GPU work over a file that never existed.

Requiring a file and producing one are separate problems, and for a while only the first was
solved. `main` passed MotifFeatures its output path under `if hasattr(entry, "motif_csv_path")`
— and on a DictConfig `hasattr` is false for a key the config does not declare, which no
shipped motif config does, and `open_dict` does not change that. So the table was never
written, indexed evaluation raised `FileNotFoundError` every time, and adding the requirement
without fixing the plumbing would have turned that into a loop: finish, write a marker that
omits the file because the file is not there, then clear the designs and regenerate them to
the same end. The entry is now found by `_target_` and the key set unconditionally, and two
guards keep the loop closed — the run stops right after dataset construction if the table it
owes is absent, which costs a checkpoint load rather than a sampling run, and **no completion
marker is written over a shard that owes a file at all**. A marker means complete; when the
shard is not, the designs stay and the claim is withheld.

What a shard *owns* follows from what it owed, not from what happens to be on disk under the
right name. Every motif run is handed a `motif_csv_path`, atom-spec runs included — AME is one
— and those write no table; recording whatever sat at that path claimed a file the run neither
produced nor needed, and cleanup then deleted a stray table belonging to an earlier
contig-mode run. Marker inclusion is gated on the requirement instead. Where `motif_atom_spec`
cannot be resolved the requirement is dropped, so a table that does get written goes
unrecorded and outlives the shard — a stale file rather than a deleted one, which is the right
way round for a guess.

Cleanup owns what the contract requires as well as what the marker recorded. A reward-unaware
marker does not list its CSV, so clearing from the marker alone left behind the very file
whose absence triggered the clear — and reported success, because that file was equally
absent from the check that verifies removal.

An absent marker does **not** mean an empty directory. Sample directories are created inside
the save loop, one per design, while the marker is written only after every design, reward
and timing record has been handled — so a kill between those points leaves populated output
with no marker at all. Before treating a missing marker as a new run, generation looks for
`job_{job_id}_n_*` directories under the output root and under `filtered_out_samples/`, and
refuses if any exist — where `job_{job_id}_` is the whole ownership rule, defined once in
`shard_dir_prefix` and used by all three save paths. That matters because they do not agree
on the rest of the name: `save_predictions` and `save_protein_ligand_predictions` produce
`job_{id}_n_{length}_id_{counter}[_tag]`, while `save_motif_predictions` produces
`job_{id}_id_{index}_motif_{name}` with no `_n_` at all. A scan written against one format
silently misses the others. The trailing underscore keeps job ids apart, so job 1 does not
match job 10 and shards generated in parallel do not block each other.

Found by name rather than by an in-progress marker written before the save loop. The latter
would make the question answerable directly instead of inferred, but only for runs started
after it exists — which leaves exactly the interrupted campaigns this is meant to catch
uncovered.

`sample_dirs` was added (`c09a82c`) after markers themselves (`d40066a`), so a campaign
finished between those commits has a valid digest-matching marker that names nothing.
Skipping such a shard is harmless — nothing is written — but a *forced* rerun cannot
identify what it would replace, so it aborts rather than clearing nothing and proceeding. An
empty `sample_dirs` beside `nsamples: 0` is the one exception: the shard genuinely produced
nothing, so there is nothing to clear.

`skip_completed_shards: false` forces regeneration of a shard whose digest matches, and
**removes the directories that shard recorded first**. The digest matching means this is a
request to redo exactly that shard, and redoing it means replacing its output rather than
writing over whichever names happen to collide. It previously warned that regeneration
would produce new directory names; that was the same mistaken claim corrected above.

**Resolution is attempted, never required.** The generation subtree carries `oc.env`
interpolations for things a given run may not touch — `af_params_dir: ${oc.env:AF2_DIR}` sits
uncommented at `binder_generate.yaml:135`, and campaign configs build `target_path` from a
campaign-directory variable. Hashing with `resolve=True` unconditionally therefore aborts
generation over an unset variable the run never reads; that regression escaped into a pushed
commit and only surfaced when the checker ran on a real campaign. An unresolvable config now
falls back to the unresolved text, with the mode mixed into the hash so the two forms cannot
collide.

Hashing the whole `generation` subtree rather than comparing a sample count keeps this correct
for every code path — length-based, repeat-based, motif conditional features — without
duplicating `split_by_job`'s arithmetic, which is the "two things that must agree" trap this
document keeps returning to. It also means *any* changed generation parameter (`nsteps`,
guidance weight, reward config) invalidates the marker, not just the design count.

**A marker alone is not enough to skip: the output has to still be there — and a count cannot
tell you.** The marker records the directory *names* the shard produced, and the guard checks
each individually, accepting it in either its original location or under
`filtered_out_samples/`.

Counting was the first design, and a real campaign broke it in both directions:

- **Filter relocation reads as deletion.** `--samples 2` generated 16 designs, and
  `filter_samples_limit: 2` moved 14 into `filtered_out_samples/` (`filter.py:207-226`). Two
  live directories against a recorded 16 looked like data loss, so every shard regenerated —
  resume was inoperative in exactly the campaigns that filter, which is all of them.
- **Accumulation defeats the other direction.** Directories pile up across reruns, so the live
  count (34) exceeded the recorded count (16) and `found < recorded` could never fire. A
  deleted design went undetected — the very case the check existed for.

Names fix both, and are checkable without knowing which of the three save paths wrote them.
Markers predating `sample_dirs` skip verification with a debug note rather than being treated
as damaged.

**A damaged shard is cleared before it is regenerated.** Generation has no per-design resume,
so the directories that survived are a partial version of what the retry is about to produce.
Leaving them makes the shard's output a mix of two attempts with only the newer recorded, and
the retry silently overwrites whichever names collide. So when the digest matches and files are
missing, the recorded directories are deleted along with the marker, and the shard is redone as
a whole. Deletion is scoped to names the marker itself lists — designs from a run with a
*different* config are never touched, because that case warns instead of clearing.

The same trap catches anything *checking* resume from outside: a live-directory count drops
across a filter stage even though nothing was lost, so comparing live counts either side of a
filter reports a correct skip as a regeneration. Count live plus `filtered_out_samples/` —
that total is invariant across filtering, which is what makes it comparable at all.

That check is what makes defaulting to skip safe, and skipping is the point: a resume feature
that warns and then burns the GPU time anyway has saved nothing.

### Sizing shards so resume is worth having

**The shard is the unit of loss.** Generation has no per-design checkpointing, so an interrupted
shard loses everything it had done; only *completed* shards are skipped on the retry. A run
sharded to match the GPU count therefore resumes almost nothing — kill it near the end and every
in-flight shard restarts from zero.

`gen_njobs` is what divides the work (`split_by_job` splits
`generation.dataloader.dataset.nres.nsamples` across shards), **but it is also the fan-out width
of a single `complexa design` invocation, and the two cannot be separated there.** `run_step`
launches all `gen_njobs` shards at once and pins each to a GPU by index —
`job_env["CUDA_VISIBLE_DEVICES"] = str(job_id)` (`cli_runner.py:716-719`). Raising `gen_njobs`
past the number of GPUs hands later shards device indices that do not exist. So you cannot get
many small shards by turning that one knob up.

To shard finely, drive the shards yourself and let `gen_njobs` mean only "how many pieces":

```bash
# gen_njobs: 32 in the config; one shard per array task, one GPU each
#SBATCH --array=0-31%4
python -m proteinfoundation.generate \
    --config-path "$CAMPAIGN_DIR" --config-name pipeline \
    ++job_id="$SLURM_ARRAY_TASK_ID" "${COMPLEXA_OVERRIDES[@]}"
```

**Invoke the stage module directly rather than going through `complexa generate`.** That is what
`run_step` runs anyway (`cli_runner.py:636-651`), minus the two things an array task must not
inherit: the fan-out, and the `CUDA_VISIBLE_DEVICES = str(job_id)` pinning that would override
SLURM's allocation. Nothing is lost — `generate.py` applies the atomworks patches and calls
`load_dotenv()` at import, and `config_name` falls back to the `--config-name` stem
(`generate.py:1415`), so `++base_config_name` is optional. Output lands in the task's own SLURM
log, which is what you wanted. One GPU per task comes from `--gres=gpu:1`; how many run at once
is the array throttle (`%4`), not `gen_njobs`.

The wrapper form — `complexa generate ./pipeline.yaml --verbose --job-id N` — also works today,
but only by accident, and it is not worth depending on. `--verbose` means *"send output to my
terminal instead of capturing it into per-job log files"*, exactly as the name suggests. The
fan-out branch **requires** capture: N concurrent processes each need `stdout=PIPE` and a demux
thread, because N interleaved streams cannot go coherently to one terminal. Running
single-process is therefore a side effect of the output choice, not its purpose
(`cli_runner.py:611`). Output routing and process topology are welded together in one flag, so
anyone who later adds proper demuxing for verbose mode would silently reinstate the fan-out and
every array task would spawn 32 subprocesses. Prefer the explicit invocation, where the topology
is stated rather than inferred from a logging flag.

**Choose designs-per-shard by time, not by count.** A shard should be a tolerable loss and long
enough to amortise its startup — every shard is a fresh process that loads the checkpoint. Read
the per-seed cost from a smoke run's `timing_{job_id}.csv` (`generation_time / seeds_in_shard`)
rather than assuming: the CBLN1/5KC5 smoke test spent 337 s on 4 seeds at `nsteps: 400`, about
**85 s per seed per GPU**, so a 1000-seed campaign is roughly 24 GPU-hours and a 32-way split
puts ~45 min in each shard. Aim for a shard in the tens of minutes; minutes-long shards pay
model loading repeatedly, hour-plus shards give resume little to save.

**`eval_njobs` must equal `gen_njobs`, and the evaluate array must be the same size.** In
`input_mode: generated` — what campaigns use — evaluation does not chunk by `njobs` at all:
`split_by_job_generated(root, job_id)` selects directories whose names begin with
`job_{job_id}_` (`evaluation/utils.py:279-287`). So evaluate shard *N* processes exactly what
generate shard *N* produced. Shard generation 32 ways and evaluate 4 ways and the designs from
shards 4–31 are never evaluated; the only signal is one `No files assigned to job N/M` line per
idle worker before it exits 0 (`evaluate.py:807-808`). The repo says the same thing in one line
(`docs/INFERENCE.md:312`), and it becomes load-bearing the moment generation is sharded for
resume rather than for throughput.

**Under SLURM, GPU count is not what `njobs` expresses.** With one array task per shard, each
task takes one GPU via `--gres=gpu:1`, and how many run at once is the array throttle
(`--array=0-31%4`), not `gen_njobs`. Setting `gen_njobs: 1` to "let SLURM handle the GPUs" would
break sharding outright — it is the divisor in `split_by_job`, so every task would generate the
whole campaign. Keep `gen_njobs` at the shard count and use `--verbose` to suppress the
in-process fan-out; that is the only thing standing between the CLI and its own GPU assignment.

One caveat worth stating in the job file itself: skipping only helps for shards that *finished*
— a shard interrupted midway is cleared and redone whole.

A campaign sharded this way is no longer one `complexa design` call, because that runs all four
steps in-process. It becomes four SLURM submissions chained on `afterok`:

```bash
gen=$(sbatch --parsable --array=0-31%4 --gres=gpu:1 gen_shard.sbatch)      # gen_njobs: 32
flt=$(sbatch --parsable --dependency=afterok:$gen           filter.sbatch)  # single task
evl=$(sbatch --parsable --dependency=afterok:$flt --array=0-31%4 --gres=gpu:1 eval_shard.sbatch)
       sbatch          --dependency=afterok:$evl            analyze.sbatch  # single task
```

`filter` and `analyze` are single-GPU-free aggregation steps and must not be arrayed.

`scripts/check_resume.sh` exercises all of this against a throwaway `run_name`:

```bash
bash docs/binder-target-setup/scripts/check_resume.sh --config ./pipeline.yaml --samples 2
```

It deliberately does **not** shrink `nsteps`. `generation.search.step_checkpoints` are
absolute step indices (`[0, 100, 200, 300, 400]` in the CBLN1 campaign), so `--nsteps 50` puts
every checkpoint but the first past the end of the trajectory and generation dies — which is
how the script's own first default broke a real run. It now leaves `nsteps` at the config's
value unless asked, and refuses a `--nsteps` smaller than the largest checkpoint. For the same
reason its "changed parameter" case perturbs `generation.dataloader.batch_size`, which is
inside the subtree the digest hashes; top-level `seed` is not, and would leave the digest
unchanged.

It asserts on filesystem state rather than log text, and it checks the *invalidation* paths as
well as the reuse ones — a resume that never invalidates is indistinguishable from one that
silently serves stale results. Two design points worth copying if you write your own: it aborts
unless the first evaluate actually produced refolding outputs, because otherwise "nothing was
rewritten" proves nothing; and its deleted-directory case reuses the *same* overrides as the
marker on disk, so the regeneration it observes is the missing output talking rather than a
digest that no longer matches.

The remaining limitation is provenance. A skipped shard was produced by a different process
than the one writing the manifest, so "seed S produced these N designs" spans two runs; the
digest proves they were the same *request*, not the same RNG stream.

Retry at shard granularity rather than clearing. The
standalone step commands take `--job-id` (`add_job_args`, applied to `generate_parser` but not
`design_parser`), so when a multi-GPU generate loses one shard you can re-run just that shard:

```bash
complexa generate ./pipeline.yaml --job-id 3 "${COMPLEXA_OVERRIDES[@]}"
```

This is why the unconditional `live + filtered_out == generated` assertion matters: it is what
catches a retry that silently inherited the previous attempt's directories.

Note that shard skipping does not remove the need for this assertion — it narrows what the
assertion has to catch. A shard skipped on a matching digest contributes its old directories to
the count, which is correct; a shard regenerated after a digest mismatch contributes both sets,
which is not.

`evaluate`'s binder track reuses work by default. Each design's refolding results are cached in
`binder_eval_cache.json` next to its PDB, and `metric.reuse_cached_folding` (default `true`)
loads that instead of refolding.

**The cache is keyed on a fingerprint of everything that determines the result** — folding
backend, inverse-folding model, target path/chain/task, interface cutoff, redesign count,
sequence types, and the per-design fixed-residue override. Without that, switching
`binder_folding_method` from `colabdesign` to an `rf3` model would silently serve AF2 numbers
for an RF3 run, which is the failure that makes an unfingerprinted cache worse than no cache.
Widening `sequence_types` from `["self"]` to `["self", "mpnn"]` also changes the fingerprint, so
it recomputes rather than returning a partial answer.

It is written alongside the pre-existing `sequence_type_stats.json`, whose schema is unchanged,
and caches without a fingerprint field are ignored — so designs evaluated before this existed
recompute once and are cached thereafter. The monomer track is unaffected: it folds a batch of
sequences per call rather than looping per design, so it has no per-design artifact to key on.

`filter` and `analyze` are safe to re-run. `filter` recomputes `keep_dirs` from paths that still
exist (`filter.py:188-191`) and explicitly skips `filtered_out_samples/` when moving, so a second
pass is a no-op; `analyze` regenerates its aggregates and its timing reader is written for
re-runs (`analysis.py:1671-1672`). `evaluate` has no skip logic at all — no result-existence
check, no per-design fold cache — so it redoes every AF2 fold. That is the expensive leg to
avoid repeating.

One rough edge in parallel stages: shards are launched together, then waited on in order, and
the first non-zero return code raises immediately (`cli_runner.py:747-750`). The remaining
`Popen` children are not killed, so a shard-0 failure leaves later shards running. Under SLURM
the step teardown reaps them; outside SLURM, check for orphaned GPU processes before retrying.

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

