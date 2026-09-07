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
    trim_shards.py        TEMPLATE, verbatim
    check_preflight.py    TEMPLATE, verbatim
    verify_run_outputs.py TEMPLATE, verbatim
    refresh_checksums.py  TEMPLATE, verbatim
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
| everything from `env.sh` | `CKPT_PATH`, `DATA_PATH`, … |

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

A follow-up takes only the number of additional designs wanted. Seeds, raw,
keep, expect and its own RNG seed are derived from what the production run
actually produced, recorded in `metadata/followup_<n>.json` before anything is
queued, and it is deduplicated against every earlier pooled run.

`SLURM_TIME_GPU` and `SLURM_TIME_CPU` in campaign.env override the wall clocks;
both have defaults, so neither has to be set.
