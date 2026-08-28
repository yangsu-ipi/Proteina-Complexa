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

## Stage contract

`run_campaign.sh <smoke|production> [all|generate|filter|evaluate|analyze]`

`generate` → `trim_shards.py` → `filter` → `evaluate` → `analyze` → `verify_run_outputs.py`.
Generation and evaluation run one process per shard, each pinned to its own GPU. Filter and
analyze are single-process and operate on the whole campaign.
