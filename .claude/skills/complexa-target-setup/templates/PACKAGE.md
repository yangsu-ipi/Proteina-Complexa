# Campaign package layout

A campaign package is a self-contained directory that runs Complexa against one target
without editing the repo. This layout is not a suggestion: the templates in this
directory assume it, and an agent that invents a different shape gets none of their
benefit.

```
<campaign>/
  campaign.env            <- EVERY campaign-specific value. From campaign.env.example.
  pipeline.yaml           <- the Hydra config. Authored per campaign.
  data/                   <- target PDB, MSA, provenance
  scripts/
    run_campaign.sh       TEMPLATE, verbatim
    submit_campaign.sh    TEMPLATE, verbatim
    plan_followup.py      TEMPLATE, verbatim
    estimate_run.sh       TEMPLATE, verbatim -- how big should the next run be
    trim_shards.py        TEMPLATE, verbatim
    check_preflight.py    TEMPLATE, verbatim
    verify_run_outputs.py TEMPLATE, verbatim
    refresh_checksums.py  TEMPLATE, verbatim
    prepare_target_msa.py TEMPLATE, verbatim -- target MSA from the ColabFold
                          public server; needs ColabFold only when it RUNS
    validate_resolved_config.py   authored: asserts the config resolves to the intended run
    capture_metadata.py           authored: run provenance
    prepare_<target>.py           authored: target-specific PDB prep
  slurm/
    campaign.sbatch       TEMPLATE (campaign.sbatch.generic), verbatim -- what
                          submit_campaign.sh chains; the stage comes as an argument
    <name>_smoke.sbatch   TEMPLATE (campaign.sbatch), header + last line edited;
                          only for running one stage by hand, without the chain
  community_models -> $COMPLEXA_REPO/community_models   (symlink, made by run_campaign.sh)
  inference/              <- generation output; created by the run
  evaluation_results/     <- evaluation output; created by the run
  metadata/               <- resolved config, preflight, trim + verification reports
  logs/slurm/             <- job logs
```

## What varies, and where it goes

Everything campaign-specific belongs in `campaign.env`: identity, paths, target, the
shape of the smoke and production runs, the GPU budget, and which result columns the
campaign depends on. **If you find yourself editing a template, that is a bug in the
template** — it means something campaign-specific was left baked in. Add a variable.

## Two rules that cost several GPU runs to learn

**Do not guard on the output directory existing.** Generation distinguishes skip,
clear-and-regenerate, and abort *per shard*, and every branch either skips safely or
aborts with a message naming the recovery. A runner that refuses whenever the directory
exists disables resume for the exact case resume is for.

**Designs move out of the root.** After `filter` they are under `filtered_out_samples/`,
and campaign post-processing may group them further (`pre_filter_shard_trim/`,
`global_sequence_duplicates/`). Anything counting or locating designs must search the root
*and* `filtered_out_samples/` recursively. A step that runs before `filter` still sees
*post*-filter state on a resumed run.

## Environment contract with `pipeline.yaml`

`run_campaign.sh` exports exactly these for the config to resolve against:

| variable | meaning |
|---|---|
| `CAMPAIGN_DIR` | the package root — **use this in `pipeline.yaml`**, not a campaign-specific name |
| `COMPLEXA_REPO` | the Complexa checkout |
| `COMMUNITY_MODELS_PATH` | community models, also symlinked into the package |
| `TARGET_MSA` | **only when campaign.env sets it** — the target alignment, package-relative |
| everything from `env.sh` | `CKPT_PATH`, `DATA_PATH`, … |

`TARGET_MSA` is package-relative and exported conditionally. `prepare_target_msa.py --out
$TARGET_MSA` writes that file and `pipeline.yaml` reads it back as
`${oc.env:CAMPAIGN_DIR}/${oc.env:TARGET_MSA}`, so the alignment retrieved and the
alignment folded against are the same string rather than two conventions that agree until
someone renames one; `check_preflight.py` then gates on it existing. Relative here and
absolute there is the same split `TARGET_PDB` already uses: **nothing inside the package
names its own location**, so the package copies to another machine unedited, while the
resolved config still carries absolute paths and does not depend on anyone's cwd. A campaign with no MSA leaves the
variable unset, and the runner exports nothing — an unconditional export would create it
EMPTY, and `${oc.env:TARGET_MSA}` would resolve to `""` instead of raising, which is a
config that folds with no alignment and says nothing.

So a config refers to its own package as `${oc.env:CAMPAIGN_DIR}`. A package carried
over from an older layout may name it something campaign-specific
(`${oc.env:CBLN1_CAMPAIGN_DIR}`), which resolves to nothing and surfaces as an
omegaconf `KeyError` several frames deep, *after* the checkpoint has loaded. The
runner now checks every no-default `${oc.env:VAR}` in the config before anything
expensive starts and names what is missing. `${oc.env:VAR,fallback}` is not required
and is not checked.

## Stage contract

`run_campaign.sh <smoke|production> [all|generate|filter|evaluate|analyze]`

`generate` → `trim_shards.py` → `filter` → `evaluate` → `analyze` → `verify_run_outputs.py`.
Generation and evaluation run one process per shard, each pinned to its own GPU. Filter and
analyze are single-process and operate on the whole campaign. `run_campaign.sh` runs the one
stage it is given; `submit_campaign.sh` is what chains them.

## Running a campaign

    scripts/submit_campaign.sh production
    scripts/submit_campaign.sh followup 900
    DRY_RUN=1 scripts/submit_campaign.sh followup 900   # print the chain, submit nothing

Each submits generate, filter, evaluate, analyze and the pooled report as
separate jobs joined by `afterok`. Separate rather than one long job because a
failure then costs the stage that failed and not the hours before it, and
because generate and evaluate want GPUs for hours while the rest want none.

### Re-running part of a chain

A stage name re-runs from there to the end:

    scripts/submit_campaign.sh production evaluate     # evaluate, analyze, pooled
    scripts/submit_campaign.sh followup 900 evaluate   # the follow-up that wanted 900
    scripts/submit_campaign.sh production pooled       # the campaign total alone

From a stage rather than a list of them. A re-run exists because something
changed, and everything downstream of a changed stage reads what it produced --
re-running evaluate and not analyze leaves a results CSV disagreeing with the
per-job CSVs it was built from, with nothing saying so.

`from..to` stops early, for the one case where stopping is right: several runs
re-evaluated together share **one** pooled report, and chaining a pooled to each
of them queues a campaign total per run, every one of which reads whatever has
finished so far. The last to run happens to be correct; the others write a
mid-flight number to `pooled_analysis.json` under the same name.

    for r in production "followup 900" "followup 1110"; do
        scripts/submit_campaign.sh $r evaluate..analyze
    done
    scripts/submit_campaign.sh production pooled     # once, after all of them

`..analyze` starts from the beginning and `evaluate..` runs to the end, so the
range form degrades to the plain one. A chain that omits the pooled report says
so on submission — the campaign total is the headline number, and stopping early
is only right if it is followed up.

**A follow-up re-run reuses its index; it does not become a new follow-up.** The
chain that includes `generate` plans a new one, as before. A chain starting later
resolves the existing follow-up by the design count it was planned for, and
refuses rather than guesses when no record matches or several do. This is what
the earlier hand-written `sbatch ... FOLLOWUP_INDEX=1` workaround was for: an
unpinned re-plan takes the next index, so the chain evaluates an inference
directory nothing ever wrote and burns a seed on a run that never happens.

The design count is the caller's handle on a follow-up, and it stops being one as
soon as two of them share it. Name the run instead:

    FOLLOWUP_INDEX=2 scripts/submit_campaign.sh followup 900 evaluate

An explicit index wins over the count for any starting stage — naming a run is a
stronger statement than describing it.

### One numbered sequence, three spellings

A campaign's pooled runs are one numbered sequence, and a run is named for its
position in it:

| spelling | meaning |
|---|---|
| `production` | the first run of a campaign written before the kinds merged — run 1 |
| `followup{K}` | the K-th run after that one — run K+1 |
| `production{N}` | the current spelling, N from 1 |

The merge is a rename, not a renumbering. Seeds are `base + (number - 1) × 1000`,
which is exactly what the two old kinds produced, so **nothing on disk moves**:
`production` keeps the base seed, `followup1` keeps `base + 1000`. Runs already
written keep the names they were written under; only new ones use the current
spelling.

    scripts/submit_campaign.sh production 200      # a new run, 200 seeds
    scripts/submit_campaign.sh production          # the first run, sized from campaign.env
    scripts/submit_campaign.sh followup 900        # still works: 900 more designs

`production N` is **N seeds**, which is what a run actually takes. A design
target is a separate question, and `estimate_run.sh` answers it both ways:

    scripts/estimate_run.sh 900              # how many seeds for ~900 designs
    scripts/estimate_run.sh --orderable 500  # ... for ~500 sequences past the gate
    scripts/estimate_run.sh --seeds 200      # what 200 seeds should yield

    run #4 would be 164 seeds -> about 904 designs
      1312 raw, trimmed to 1280 before global dedup
      at 5.51 designs per seed, measured over: production, followup1, followup2
      submit with: scripts/submit_campaign.sh production 164

It writes nothing, reserves no run number and submits nothing. The conversion is
advice: baking it into the submit path made the estimate binding, and left no way
to run a size the arithmetic had not chosen.

**Orderable is the noisier target, and it is sized on the low end of an
interval.** Designs per seed is close to stable (5.31 / 5.53 / 5.56 on CBLN1) and
pools safely as a point estimate. Orderable per design is not: 0.482 / 0.396 /
0.319, a spread of 3.1 standard errors end to end. Sizing on its mean is what
over-promised the campaign's first two follow-ups by about 40% — predicted ~434
and ~535 orderable, delivered 372 and 373.

**The interval is clustered, and this matters more than it sounds.** Beam search
expands one `nres` draw into several candidates, so designs sharing a root are not
independent draws — and binder length, which dominates whether a design passes
(0.14 / 0.37 / 0.58 orderable-per-design by length band), is drawn per root.
Treating the designs as independent understates the variance 4–7×. After the
first CBLN1 run the naive interval was [0.391, 0.574] and *excluded* the 0.319
the third run delivered; clustered by beam root it was [0.292, 0.673] and covered
both later runs. So `estimate_run.sh` clusters, sizes on the lower bound, and
prints how much wider that made the interval.

It needs the analyze stage to have run, since orderable counts come from the
per-sequence verdicts in `RAW_*_combined.csv`; without those, size by designs or
seeds.

Only the campaign's first run has to be sized by hand, because it is the only one
with nothing to calibrate on. Every later run derives raw, keep and expect from
what a seed has been worth **across every completed run**, not just the first —
calibrating on the first alone anchors a campaign on its smallest and least
representative run and never updates.

A run that already exists is replayed, not re-derived: its size decided what
generation drew and what `verify_run_outputs` expects, so it is history rather
than a derivation. Its parameters are recorded before anything is queued, and it
is deduplicated against every earlier pooled run.

`SLURM_TIME_GPU` and `SLURM_TIME_CPU` in campaign.env override the wall clocks;
both have defaults, so neither has to be set.
