---
name: complexa-design
description: >
  End-to-end Proteina-Complexa design pipeline driver. Reach for this skill whenever
  the user wants to "design a binder", "design binders for X", "run complexa
  design", "de novo binder", "PDL1 binder", "TrkA binder", "design proteins for
  target", "protein binder design", "ligand binder", "design a small-molecule
  binder", "ATP-binding protein", "AME motif scaffolding", "scaffold a motif
  near a ligand", "motif + ligand design", "enzyme scaffolding", "flow matching
  protein design", "beam-search binder", "FK steering", "MCTS protein design",
  "refold with AF2", "refold with RF3", or wants success rates, interface pAE,
  scRMSD, or
  FoldSeek diversity from a single command. This is the scientific anchor of
  the skill set: it drives `complexa design <pipeline>` from target picking to
  manifest emission and tells the user how many designs passed.
compatibility: "complexa CLI installed (pip install -e .); .env populated; 1x CUDA GPU >=40GB VRAM (A100/H100/L40S); 24 CPUs; ~50GB disk"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Complexa Design Skill

Drive the full four-stage `complexa design` pipeline: generate (flow matching +
search) -> filter (top-N by reward) -> evaluate (refold with AF2/RF3) ->
analyze (success rate, FoldSeek/MMseqs diversity). Pick the right pipeline
config for the design intent, validate the run upfront so the user does not
discover a missing ckpt mid-folding, run it, and emit a replayable manifest +
per-design success CSV.

## What this skill enables

**Evaluation gained a fourth success criterion.** A protein-binder design must now
fold as designed *without* its target as well as with it: `apo scRMSD_ca < 2.0`
alongside the three AlphaProteo criteria. On by default
(`metric.compute_apo_metrics`), folded by plain ESMFold. Pass rates from before
this change are not comparable — they were measured against three criteria.

ESMC perplexity, ESMFold2 advisory complex refolding (optionally with a target
MSA) and ESMFold2 apo folding are all reachable from the same `metric.*` keys;
`complexa-evaluate-pdbs/reference/esm_esmfold2.md` is the authoritative reference
for those, including three failure modes that are silent.

Protein binder, ligand binder, and AME motif-scaffolding design; search-based
optimization (single-pass, best-of-n, beam-search, fk-steering, mcts); refold
with ColabDesign (AF2) or RF3 — `metric.binder_folding_method` takes only
`colabdesign` or an `rf3*` name, else `ValueError` (`binder_eval.py:120-140`);
pass-rate and diversity analysis per `result_type`.

## Step 1: Pre-flight

Always run the shared preflight before launching a design — generation needs the
GPU and the right checkpoint, evaluation needs AF2/RF3 weights and tool
binaries. Bail early if the host cannot run the chosen pipeline.

```bash
bash .claude/skills/_shared/scripts/preflight.sh
```

Read `./preflight.json` (that is where `preflight.sh` writes unless you pass
`--out`) and bail if any of these are missing for the chosen pipeline:

- `gpu.available: false` -> all pipelines fail.
- `gpu.vram_gb < 40` -> generation OOMs at default `batch_size: 16`; lower to 8.
- `checkpoints["complexa.ckpt"].exists` -> required for protein binder.
- `checkpoints["complexa_ligand.ckpt"].exists` -> required for ligand binder.
- `checkpoints["complexa_ame.ckpt"].exists` -> required for AME.
- `community_models.AF2_DIR.exists` false -> protein binder default eval (`colabdesign`) fails.
- `community_models.RF3_CKPT_PATH.exists` or `tools.rf3.exists` false -> ligand binder / AME default eval (`rf3_latest`) fails.

`checkpoints` is keyed by full filename (`preflight.sh:135-150`); the AF2/RF3
paths are under `community_models`, **not** `env` (`:203-205`).

If a ckpt is missing, point at `complexa-setup` and have the user run
`complexa download --complexa-<variant>` first.

## Step 2: Pick the pipeline

**One default pipeline (protein binder) and two extensions** (ligand binder,
AME / enzyme), selected entirely by the `configs/search_*_pipeline.yaml` you
pass. Each YAML pins its own checkpoints, targets dict, reward, and refold
backend — switching is "swap the config path, take the target from that dict".

### Default — protein binder

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_v1 ++generation.task_name=02_PDL1
```

For "design a binder for X", "PDL1 binder", "de novo binder" — any protein
surface target. Rewards with AF2, inverse-folds with SolubleMPNN, evaluates with
ColabDesign. **If the user did not specify, this is what they want.**

### Extension A — ligand binder (small-molecule pocket)

```bash
complexa design configs/search_ligand_binder_local_pipeline.yaml \
    ++run_name=v11_v1 ++generation.task_name=39_7V11_LIGAND \
    ++metric.binder_folding_method=rf3_latest
```

For "ligand binder", "small-molecule pocket", "SMILES target", "ATP-binding
protein", or a target ending in `_LIGAND`. The config **activates LoRA**
(`r=32`, `lora_alpha=64`), required by the released ligand checkpoint — leave
the `lora:` block alone.

### Extension B — AME / motif + ligand (enzyme scaffolding)

```bash
complexa design configs/search_ame_local_pipeline.yaml \
    ++run_name=ame_chm ++generation.task_name=M0096_1chm
```

For "scaffold a motif near a ligand", "active-site design", "enzyme
scaffolding", or an `M####_<pdb>` target. The config sets
`env_vars.USE_V2_COMPLEXA_ARCH=True` and uses `MotifFeatures` +
`LigandFeatures`. Default search is `single-pass`; `best-of-n` needs the
`CompositeRewardModel` enabled (commented out in `ame_generate.yaml`).

### Pipeline cheat sheet — what changes when you switch

| Knob | Protein binder (default) | Ligand binder | AME (enzyme) |
|---|---|---|---|
| **Pipeline YAML** | `configs/search_binder_local_pipeline.yaml` | `configs/search_ligand_binder_local_pipeline.yaml` | `configs/search_ame_local_pipeline.yaml` |
| **Targets dict** | `configs/targets/targets_dict.yaml` | `configs/targets/ligand_targets_dict.yaml` | `configs/design_tasks/ame_dict_v2.yaml` |
| **Task-name pattern** | `<NN>_<NAME>` (e.g. `02_PDL1`, `22_DerF21`) | `<NN>_<PDB>_LIGAND` (e.g. `39_7V11_LIGAND`) | `M####_<pdb>` (e.g. `M0096_1chm`) |
| **Default refold backend** | `colabdesign` (AF2) | `rf3_latest` | `rf3_latest` |
| **Required ckpts (`complexa download`)** | `--complexa --all` (AF2 in community) | `--complexa-ligand --all` (RF3 in community) | `--complexa-ame --all` (RF3 in community) |

Everything else the switch changes — model and autoencoder ckpts, LoRA, default
search algorithm, reward model, inverse folder, `result_type`,
`USE_V2_COMPLEXA_ARCH` — is pinned by the config, not typed by you. Full
breakdown in [reference/pipelines.md](reference/pipelines.md).

## Step 3: Gather parameters

Use AskUserQuestion to fill in the four parameters that vary every run. Default
to sensible production settings if the user has no preference.

- **Target name** — must be a key in the relevant dict (`targets_dict.yaml` for
  protein binder, `ligand_targets_dict.yaml` for ligand, `ame_dict_v2.yaml` for
  AME). If the user names a target that is not in the dict, hand off to
  `complexa-target` to add it first.
- **Run name** — a short identifier appended to the output dir (e.g. `pdl1_v1`).
- **Search algorithm** — default to `beam-search` with `beam_width=8` and
  `n_branch=4` for production. Use `single-pass` for a quick smoke test.
- **Evaluation refold backend** — `colabdesign` (protein binder) or
  `rf3_latest` (ligand/AME); these two families are the only ones accepted. To
  iterate faster keep the default and lower `++metric.num_redesign_seqs` or
  `++generation.dataloader.batch_size`.

## Step 4: Validate

Validate before running. This is cheap (seconds) and catches missing ckpts and
missing env vars — either of which would otherwise abort the pipeline
mid-evaluation after hours of generation. It does **not** validate override
keys; there is no config-key checking in `validate.py`.

`complexa validate` takes no Hydra overrides — its subparser has only `type`,
`config`, and `--target` (`cli_runner.py:1326-1348`), so appending `++...`
aborts with `unrecognized arguments`. Validate the config as shipped:

```bash
complexa validate design configs/search_binder_local_pipeline.yaml
```

Returns non-zero on failure with a pass/fail report; fix the reported ckpt /
env problems and re-run until clean.

## Step 5: Run the pipeline

`complexa design` orchestrates `generate → filter → evaluate → analyze` as
sequential subprocesses sharing a run name, log dir, and multi-GPU split
(`run_design_pipeline`). Re-implementing it loses the per-stage log routing.

Use `++` (forced) Hydra overrides; they apply to all stages. The minimal
production protein-binder invocation:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_v1 \
    ++generation.task_name=02_PDL1 \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8 \
    ++metric.binder_folding_method=colabdesign
```

For ligand binder / AME, swap the pipeline YAML and target name per Step 2's
cheat sheet — every other override above is pipeline-agnostic and can be
reused as-is.

`--verbose` streams logs to the terminal instead of `./logs/` (and, as a side
effect, disables the multi-job fan-out). The skill does not poll progress —
point the user at `complexa status` and `./logs/design_pipeline_*/`.

To debug one stage, or to run a single shard under a scheduler, invoke the
Hydra module directly — see "Running one stage directly" in
[reference/troubleshooting.md](reference/troubleshooting.md).

**AME + RF3 refold needs the ligand renamed to `L:0`** or RF3 completes missing
ligand atoms and corrupts the RMSDs. The canonical AME pipeline already encodes
it that way; for hand-made PDBs see "AME ligand residue name must be `L:0`" in
[reference/troubleshooting.md](reference/troubleshooting.md).

## Step 6: Collect results

Outputs land in two directories. Surface both:

```bash
ls ./inference/${CONFIG_STEM}_${TASK}_*${RUN_NAME}/            # generated PDBs + filter
ls ./evaluation_results/${CONFIG_STEM}_${TASK}_${RUN_NAME}/    # per-design CSV + analysis
```

The results dir is `./evaluation_results/{config_name}_{task_name}`
(`evaluate.py:774`) with `_{run_name}` appended when `run_name` is set
(`:784-785`) — it is never just `./evaluation_results/{run_name}`.

Read the combined results CSV and summarize:
```bash
ls ./evaluation_results/*/RAW_protein_binder_results_*_combined.csv       # combined
ls ./evaluation_results/*/RAW_motif_ligand_binder_results_*_combined.csv  # AME combined
ls ./evaluation_results/*/binder_results_*_*.csv                         # per-job (evaluate)
ls ./evaluation_results/*/filter_results/res_filter_*_pass_*.csv          # success rate
ls ./evaluation_results/*/diversity/res_div_foldseek_*.csv                # FoldSeek diversity
```

The only combined CSV is `RAW_{result_type}_results_{config_name}_combined.csv`
(`analyze.py:3047`); `binder_results_*_{job_id}.csv` (`evaluate.py:899`) is
per-job and never `_combined`. By the time you look, `organize_results()`
(`analyze.py:2813-2856`) has moved `res_filter_*` into `filter_results/`,
`res_div_*` into `diversity/`, `clusters_*` into `clusters/` (AME's
`res_filter_motif_binder_*` into `motif_binder_metrics/`) — top-level globs
match nothing.

Pull the success rate from `res_filter_binder_pass_*.csv` (protein binder only
— ligand writes `res_filter_ligand_pass_*`, AME `res_filter_motif_binder_pass_*`;
`binder_analysis.py:637`), the per-design
metrics (interface pAE, pLDDT, scRMSD) from the combined CSV, and FoldSeek
TM-score diversity from `res_div_foldseek_*.csv`. Report top-N designs by
i_pAE (protein binder) or min_ipAE (ligand binder).

## Step 7: Emit manifest

Drop a JSON manifest beside the results so the run is replayable. The shared
helper captures the command, git SHA, and pointers to the result CSVs. Its
`config` and `checkpoints` fields come out `null`: it reads
`<output_dir>/.hydra/config.yaml` (`write_manifest.py:79-80`), and no pipeline
writes `.hydra/` under the output dir — `search_binder_local_pipeline.yaml:37-39`
points `hydra.run.dir` at `./logs/hydra_outputs/...`.

```bash
python3 .claude/skills/_shared/scripts/write_manifest.py \
    --output-dir ./evaluation_results/${CONFIG_STEM}_${TASK}_${RUN_NAME} \
    --command "complexa design configs/search_binder_local_pipeline.yaml ++run_name=${RUN_NAME} ++generation.task_name=${TASK}" \
    --skill complexa-design \
    --out ./run_manifest.json
```

Surface the manifest path and the result CSV to the user.

## Most-common overrides

The 10 overrides that cover ~90% of runs. Full reference (every key, type,
default) is in [reference/overrides.md](reference/overrides.md).

| Override | Default | What it controls |
|----------|---------|------------------|
| `++generation.task_name=<name>` | (per config) | Which target / AME task to design for |
| `++run_name=<str>` | the config's `run_name:` field (e.g. `search_binder_local`; **not** the config stem `search_binder_local_pipeline`) | Output dir suffix and CSV tag |
| `++generation.search.algorithm=beam-search` | `best-of-n` (binder/ligand), `single-pass` (AME) | Search strategy |
| `++generation.search.beam_search.beam_width=8` | `4` | Beam-search width (more = better designs, slower) |
| `++generation.args.nsteps=200` | `400` | Diffusion steps (fewer = faster, lower quality) |
| `++generation.dataloader.batch_size=8` | `16` (binder/ligand/AME) | Drop to 8 on a 40GB GPU |
| `++generation.filter.filter_samples_limit=500` | `1000` | Top-N samples to keep after filtering |
| `++metric.binder_folding_method=rf3_latest` | `colabdesign` (binder), `rf3_latest` (ligand/AME) | Evaluation refold backend — only `colabdesign` or an `rf3*` name is accepted |
| `++metric.num_redesign_seqs=8` | `8` (protein target) / `1` (ligand target), from `binder_eval_utils.py:50-51` | ProteinMPNN/LigandMPNN/SolubleMPNN sequences per design |
| `aggregation.success_thresholds` (full dict, see below) | `i_pAE*31<=7.0`, `pLDDT>=0.9`, `scRMSD_ca<1.5` (protein binder) | Loosen / tighten success criteria |

> **Never override `success_thresholds` partially.** A partial override replaces
> the whole dict, silently dropping the other criteria, and the reported success
> rate can become 100%. Supply the complete dict — see "success_thresholds" in
> [reference/overrides.md](reference/overrides.md).

## Hardware requirements

1x CUDA GPU (40 GB min, 80 GB recommended), 16 CPUs (24 is the `ncpus_`
default), 50 GB disk, 32 GB RAM. Wall-clock for 100 designs at `beam_width=8`,
`nsteps=400`, on 1x A100/H100: protein binder + colabdesign ~60–120 min; ligand
binder + RF3 ~90–180 min; AME + RF3 ~120–240 min (RF3 dominates).

Bumping `gen_njobs=2` and `eval_njobs=2` halves wall-clock on a 2-GPU host. **Keep the two
equal** — eval shard *N* takes the designs named `job_N_*`, so a lower `eval_njobs` silently
leaves the later shards unevaluated. Neither may exceed the GPU count here: each job is pinned
to GPU index `job_id`. For a campaign that must survive interruption, generation is sharded far
more finely and driven per shard — see "Sizing shards so resume is worth having" in
`docs/binder-target-setup/campaign-gating.md`. See
`.claude/skills/_shared/reference/hardware.md` for per-pipeline VRAM tables.

## Troubleshooting (common cases)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` in generate | `batch_size: 16` too big on 40GB GPU | `++generation.dataloader.batch_size=8` |
| `InterpolationKeyError: AF2_DIR` / `RF3_CKPT_PATH` | eval backend's weights absent | set it in `.env`, or `complexa download --all` |
| `KeyError: 'task_name' not in target_dict_cfg` | target not in the pipeline's dict | add it with the `complexa-target` skill |
| 0 designs pass success thresholds | defaults too strict for this target | supply the **complete** `success_thresholds` dict |

For the full list (chain-ID mismatches, hotspot residues, ligand residue
renaming for RF3, missing inverse-folding models, etc.) see
[reference/troubleshooting.md](reference/troubleshooting.md).

