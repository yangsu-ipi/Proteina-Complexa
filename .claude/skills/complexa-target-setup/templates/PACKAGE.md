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
    trim_shards.py        TEMPLATE, verbatim
    check_preflight.py    TEMPLATE, verbatim
    verify_run_outputs.py TEMPLATE, verbatim
    refresh_checksums.py  TEMPLATE, verbatim
    validate_resolved_config.py   authored: asserts the config resolves to the intended run
    capture_metadata.py           authored: run provenance
    prepare_<target>.py           authored: target-specific PDB prep
  slurm/
    <name>_smoke.sbatch   TEMPLATE (campaign.sbatch), header + last line edited
    <name>_500.sbatch     TEMPLATE, likewise
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
analyze are single-process and operate on the whole campaign.
