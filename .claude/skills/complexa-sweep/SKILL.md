---
name: complexa-sweep
description: Use this skill whenever the user wants to run a parameter sweep over a Proteina-Complexa design pipeline — cartesian-product hyperparameter scans, Pareto search over generation/reward/evaluation knobs, or any "compare configurations" workflow. Trigger phrases include "sweep beam width", "sweep nsteps", "hyperparameter sweep", "parameter scan", "scan beam_width and temperature", "compare configurations", "find the best generation params", "what's the optimal nsteps", "Pareto search for binder quality vs wall-clock", "complexa sweep", "tune Complexa", "ablate the reward weights", "configs/sweeps", "--sweeper", "run beam_width.yaml". This is the only skill that owns sweeper YAML authoring, cartesian-product expansion, and per-config result ranking.
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# complexa-sweep

Run cartesian-product parameter sweeps over Proteina-Complexa design pipelines. Pick or author a sweeper YAML in `configs/sweeps/`, expand it to N **config pairs** with `script_utils/generate_inference_configs.py`, run `generate`/`filter` from each `inf_*.yaml` and `evaluate`/`analyze` from its matching `eval_*.yaml`, then aggregate per-config success metrics into a ranked summary CSV plus a manifest.

> **Important — two things, both easy to get wrong:**
>
> 1. The `complexa design` CLI does **NOT** accept `--sweeper`. Sweeps are driven by
>    `script_utils/generate_inference_configs.py`.
> 2. That script writes a **pair** per combination — `configs/inference_configs/inf_{idx}_{run_name}.yaml`
>    *and* `configs/eval_configs/eval_{idx}_{run_name}.yaml` (`:327-337`) — and only the eval half
>    carries the paths the evaluation stages need. Looping `complexa design` over the `inf_*.yaml`
>    alone produces structures and **no usable evaluation results at any path**. See Step 4.

## What this skill enables

- Pick an existing sweeper YAML from `configs/sweeps/` (`beam_width`, `bb_ca_temperature`, `search_replicas`, `example`).
- Author a new sweeper YAML with arbitrary dot-notation axes (cartesian product).
- Generate N inference + evaluation config pairs with `script_utils/generate_inference_configs.py`.
- Run the four stages per combination — `generate`/`filter` from `inf_N.yaml`, `evaluate`/`analyze` from `eval_N.yaml` (one combination at a time on a single GPU, or in parallel on a multi-GPU host by sharding the config list across `CUDA_VISIBLE_DEVICES`).
- Walk per-config output directories and parse the analyze-step CSV from each.
- Emit `sweep_summary.csv` (one row per config: axis values + success rate + mean iPAE + diversity) and `sweep_manifest.json`.
- Identify the best config by success rate and the Pareto frontier (wall-clock vs success).

## Step 1: Pre-flight

```bash
bash .claude/skills/_shared/scripts/preflight.sh
```

Read `./preflight.json` (`preflight.sh:19` sets `OUT="./preflight.json"`; only `--out` changes it, and nothing here passes `--out`). A sweep multiplies GPU time by the number of configs. **Before launching, confirm the cost with the user**:

> "This sweep produces N configs × ~M minutes per config ≈ TOTAL GPU-hours. OK to proceed? (y / reduce / cancel)"

If `gpu.available=false`, stop — sweeps are not feasible on CPU.

## Step 2: Pick the pipeline + target

Use the same dialogue as `complexa-design` — do **not** duplicate it here. See [`.claude/skills/complexa-design/SKILL.md`](../complexa-design/SKILL.md) Step 2 ("Pick the pipeline") and Step 3 ("Gather parameters"). Capture:

- `pipeline_config_name` — e.g. `search_binder_local_pipeline` (default), `search_ligand_binder_local_pipeline`, `search_ame_local_pipeline`.
- `task_name` — e.g. `02_PDL1`, `22_DerF21`, `39_7V11_LIGAND`. Passed as `--override generation.task_name=<task>`.
- `run_name` — short tag for output dir naming.

## Step 3: Pick or author the sweeper YAML

Sweeper YAMLs live in `configs/sweeps/`. Each key is a dot-notation Hydra path; each value is a list. The cartesian product becomes N configs.

### Canned sweepers

| File | Axis | Values | Configs |
|---|---|---|---|
| `configs/sweeps/beam_width.yaml` | `generation.search.beam_search.beam_width` | 1, 2, 4, 8 | 4 |
| `configs/sweeps/bb_ca_temperature.yaml` | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | 0.1, 0.4 | 2 |
| `configs/sweeps/search_replicas.yaml` | `generation.search.best_of_n.replicas` | 1, 4, 16, 64 | 4 |
| `configs/sweeps/example.yaml` | beam_width × nsteps | (2,4) × (200,400) | 4 |

If one matches the user's intent, use it as-is. Otherwise author a new file.

### Authoring a new sweeper

Minimal multi-axis example (saved to `configs/sweeps/my_sweep.yaml`):

```yaml
# 3 beam widths × 2 nsteps = 6 configs
generation.search.beam_search.beam_width:
  - 2
  - 4
  - 8

generation.args.nsteps:
  - 200
  - 400
```

Rules (`script_utils/generate_inference_configs.py`):

- Top-level mapping only. Keys are dot-notation Hydra paths. (`load_sweeper_file`, `:108-131`)
- Values must be **lists**. A scalar is auto-wrapped into a single-element list, which pins a value without adding a dimension. (`load_sweeper_file`, `:129-130`)
- Cartesian product: total configs = product of list lengths. Two 4-value axes = 16 configs; budget accordingly. (`apply_sweeper_and_save_configs`, `:289-296`)
- If a key appears in both the sweeper file and an `--override`, the override wins and that axis collapses. This is done in **`build_sweeper` (`:160-162`)**, not `load_sweeper_file` — the latter only loads and validates.

See [reference/sweep_axes.md](reference/sweep_axes.md) for the full catalogue of swept keys (typical ranges, cost multipliers, what improves/regresses).

### Dry-run preview before generating

Always confirm the config count first:

```bash
python script_utils/generate_inference_configs.py \
    --config_name search_binder_local_pipeline \
    --sweeper configs/sweeps/my_sweep.yaml \
    --override generation.task_name=22_DerF21 \
    --run_name my_sweep \
    --dryrun
```

The output lists every axis + value list and prints `DRY RUN — would generate N config pair(s)`.

## Step 4: Generate configs, then run the stages against the matching half of each pair

Once the dry-run looks right, drop `--dryrun` to materialize `inf_{idx}_{run_name}.yaml` under `configs/inference_configs/` and `eval_{idx}_{run_name}.yaml` under `configs/eval_configs/` (`--infer_dir_cfgs` / `--eval_dir_cfgs` defaults, `:362-373`).

> **Do not run `complexa design inf_N.yaml`.** Only the *inference* config gets
> `root_path: ./inference/inf_{idx}_{run_name}` (`:327`); only the *eval* config gets
> `sample_storage_path = root_path` plus `output_dir` / `results_dir = ./evaluation_results/eval_{idx}_{run_name}`
> (`create_eval_config`, `:225-253`). `complexa design` runs all four stages —
> `generate → filter → evaluate → analyze` (`cli_runner.py:122`) — from the one config you hand
> it. Generation honours the injected `root_path` (`generate.py:1417, :68-84`), but
> **`evaluate.py` never reads `root_path`**: with `sample_storage_path` absent it builds
> `./inference/{config_name}_{target_task_name}` and appends `_{run_name}` (`:750, :770-785`),
> e.g. `./inference/inf_0_my_sweep_22_DerF21_search_binder_local` — not where generate wrote.
> `analyze.py:2922, :2947-2953` does the same for `results_dir`. A sweep driven by
> `complexa design` therefore yields structures and **no usable evaluation results**.

Split the stages instead. All four subcommands take a config-path positional plus Hydra overrides (`add_common_args`, `cli_runner.py:904-921`, wired into `generate`, `filter`, `evaluate`, `analyze` at `:979-1017`), so the pair can be threaded explicitly:

```bash
# 1. Generate one inf_*.yaml + one eval_*.yaml per combination.
python script_utils/generate_inference_configs.py \
    --config_name search_binder_local_pipeline \
    --sweeper configs/sweeps/my_sweep.yaml \
    --override generation.task_name=22_DerF21 \
    --run_name my_sweep

# 2. One combination = generate + filter from inf_N, evaluate + analyze from eval_N.
run_one() {                                   # $1 = path to an inf_*.yaml
    local cfg="$1" base evalcfg
    base=$(basename "$cfg" .yaml)             # e.g. inf_0_my_sweep
    evalcfg="configs/eval_configs/eval_${base#inf_}.yaml"
    [ -f "$evalcfg" ] || { echo "missing eval config: $evalcfg"; return 1; }
    complexa generate "$cfg" \
      && complexa filter   "$cfg" \
      && complexa evaluate "$evalcfg" \
      && complexa analyze  "$evalcfg"
}

for cfg in configs/inference_configs/inf_*_my_sweep.yaml; do
    run_one "$cfg" || echo "FAILED: $cfg"
done
```

This serialises the sweep on a single GPU — be honest with the user that wall-clock = N × per-run.

Two constraints worth stating up front:

- **`evaluate` and `analyze` must get the same config.** `analyze` finds the per-job CSVs by config stem (`find_result_files(results_dir, config_name, …)`; the CLI injects `++base_config_name=<stem>` at `cli_runner.py:650`). Feeding `inf_N` to one stage and `eval_N` to the other ends in `No result files found`.
- **The output path picks up the pipeline's `run_name`, not `--run_name`.** `--run_name my_sweep` only names the generated *files*; the eval config inherits `run_name: search_binder_local` from `search_binder_local_pipeline.yaml:21`, and both stages append it when it is not already the suffix. Results therefore land in `./evaluation_results/eval_{idx}_my_sweep_search_binder_local/`. Pass `++run_name=<tag>` to both stages if you want a path you chose.

### Multi-GPU host (optional speed-up)

On a host with K GPUs, shard the config list K ways and launch one `run_one` loop per GPU in parallel. Each stage uses exactly one GPU at default `gen_njobs=1` / `eval_njobs=1` (all three shipped pipelines set both to `1` — `search_binder_local_pipeline.yaml:31-32`), so pinning via `CUDA_VISIBLE_DEVICES` keeps them from colliding:

```bash
CONFIGS=(configs/inference_configs/inf_*_my_sweep.yaml)
N=${#CONFIGS[@]}; K=4   # 4 GPUs
for gpu in $(seq 0 $((K-1))); do
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        for i in $(seq $gpu $K $((N-1))); do
            run_one "${CONFIGS[$i]}" || echo "FAILED on GPU $gpu: ${CONFIGS[$i]}"
        done
    ) &
done
wait
```

Drop `gen_njobs` / `eval_njobs` overrides into the loop only if you intentionally want a stage to consume multiple GPUs (and you have a way to keep them out of each other's way).

## Step 5: Collect results

Each combination writes structures to `./inference/inf_{idx}_{run_name}/` (the inference config's `root_path`) and evaluation output to `./evaluation_results/eval_{idx}_{run_name}_{pipeline_run_name}/` (the eval config's `output_dir`/`results_dir` with the pipeline `run_name` appended — see Step 4). After the sweep finishes:

```bash
ls -d ./inference/inf_*_my_sweep/
ls -d ./evaluation_results/eval_*_my_sweep*/
ls ./evaluation_results/eval_*_my_sweep*/RAW_*_combined.csv
```

**No emitted file starts with `results_`.** The real names:

| File | Written by | Notes |
|---|---|---|
| `binder_results_{config_name}_{job_id}.csv` | `evaluate` (`evaluate.py:899`) | per-job. `{config_name}` is the eval config stem, e.g. `eval_0_my_sweep`. Other flavours: `monomer_results_*` (`:871`), `motif_results_*` (`:922`), `motif_binder_results_*` (`:948`) |
| `RAW_{result_type}_results_{config_name}_combined.csv` | `analyze` (`analyze.py:3047`) | **the file to parse.** `result_type` is `protein_binder` for `search_binder_local_pipeline` (`binder_analyze.yaml:12`) |
| `filter_results/res_filter_binder_pass_*.csv` | `analyze` (`binder_analysis.py:648`, relocated by `organize_results`, `analyze.py:2803-2881`) | pre-computed pass rates — read these instead of rethresholding by hand. Ligand runs write `res_filter_ligand_pass_*`, motif runs `res_filter_motif_binder_pass_*` (`motif_binder_analysis.py:252`) |

The combined CSV has **one row per generated sample** (`id_gen`, an enumerate index — `binder_eval.py:576, :593`), with one column *prefix* per requested `metric.sequence_types` value:

| Column | Meaning |
|---|---|
| `{seq}_complex_i_pAE` | interface PAE of the best refold — stored **0–1 scaled** |
| `{seq}_complex_pLDDT` | complex pLDDT of the best refold, 0–1 |
| `{seq}_binder_scRMSD_ca` | binder CA scRMSD, Å |
| `{seq}_sequence` | the binder sequence (`binder_eval.py:704`) |
| `{seq}_{prefix}_{metric}_all` | the per-redesign list the threshold filter actually reads (`binder_analysis_utils.py:182-193`) |

`i_pae`, `i_plddt`, `sc_rmsd`, `binder_seq` and `passes_filter` **do not exist anywhere in this repo** — a repo-wide grep for `passes_filter` matches only this skill's own files. There is no interface-pLDDT column at all, and no boolean pass column in the raw CSV.

Do not invent thresholds either. The protein-binder defaults are `DEFAULT_PROTEIN_BINDER_THRESHOLDS` (`binder_analysis_utils.py:75-94`): `i_pAE` with `scale: 31.0`, `threshold: 7.0`, `op: "<="`, `column_prefix: complex`; `pLDDT >= 0.9` on `complex`; `scRMSD_ca < 1.5` on `binder`. Because the stored `i_pAE` column is 0–1, an `i_pae < 10` test passes every single sample and reports 100% success. And a *partial* `aggregation.success_thresholds` override replaces the whole default dict rather than merging (`binder_analysis.py:411-412`), so if you retune, supply all three entries complete with `scale` and `column_prefix`.

**Preferred path: read `success_rate` per config out of `filter_results/res_filter_binder_pass_*.csv`** rather than recomputing it from the raw CSV. Fall back to the raw columns above only if that file is absent (e.g. `aggregation.analysis_modes` excluded `binder`).

## Step 6: Rank configs

Emit `sweep_summary.csv` to the run directory. One row per config:

| Column | How to compute |
|---|---|
| `config_id` | The `{idx}` from `inf_{idx}_{run_name}` |
| `<axis_1>`, `<axis_2>`, ... | The swept value at this combination (read from the per-config `inf_*.yaml`) |
| `n_samples` | Row count in `RAW_{result_type}_results_{config_name}_combined.csv` |
| `success_rate` | Read from `filter_results/res_filter_binder_pass_*.csv` for this config. Only recompute from the raw CSV if that file is missing, and then use the real defaults: `{seq}_complex_i_pAE_all * 31 <= 7.0` AND `{seq}_complex_pLDDT_all >= 0.9` AND `{seq}_binder_scRMSD_ca_all < 1.5` |
| `mean_i_pae` | `{seq}_complex_i_pAE.mean()` (lower = better; the column is 0–1 scaled — multiply by 31 to report in the same units as the threshold) |
| `mean_plddt` | `{seq}_complex_pLDDT.mean()` (higher = better; complex pLDDT, **not** interface pLDDT — no interface-pLDDT column exists) |
| `mean_binder_scRMSD_ca` | `{seq}_binder_scRMSD_ca.mean()` (lower = better) |
| `diversity_score` | Unique `{seq}_sequence` count / `n_samples`, or the FoldSeek/MMseqs2 output under `diversity/` and `clusters/` if `aggregation.compute_diversity` was on |
| `wall_clock_min` | From the per-config log timestamps |

Then report:

- **Best config** = argmax of `success_rate`. Tie-break on `mean_i_pae` (lower).
- **Pareto frontier** over (`wall_clock_min`, `success_rate`): a config is on the frontier iff no other config is both faster AND has higher success rate.

Print the best config + the frontier to the terminal. Save the full table to `sweep_summary.csv`.

## Step 7: Emit manifest

Capture the invocation + outputs for replay. The shared helper takes a single `--output-dir` and walks it for CSV pointers. It also *tries* to read `<output-dir>/.hydra/config.yaml` (`write_manifest.py:79-80`) — but no `complexa` stage writes `.hydra/` under the output dir (the pipelines point Hydra at `./logs/hydra_outputs/…`, e.g. `search_binder_local_pipeline.yaml:37-39`, and `cli_runner.py` sets no override for `evaluate`/`analyze`), so `config`, `config_path` and `checkpoints` come out `null` and no checkpoint hashes are recorded. What you actually get is `timestamp`, `skill`, `command`, `output_dir`, `git_sha`, `repo_root`, `pointers.csv_files` and the invocation environment (`:179-200`) — so put everything replay-critical into `--command`. Call it against the best config's evaluation directory:

```bash
BEST_ID=4   # from Step 6 ranking
# Note the trailing _search_binder_local: evaluate/analyze append the pipeline run_name.
python3 .claude/skills/_shared/scripts/write_manifest.py \
    --output-dir ./evaluation_results/eval_${BEST_ID}_my_sweep_search_binder_local \
    --command "python script_utils/generate_inference_configs.py --config_name search_binder_local_pipeline --sweeper configs/sweeps/my_sweep.yaml --override generation.task_name=22_DerF21 --run_name my_sweep; complexa generate configs/inference_configs/inf_${BEST_ID}_my_sweep.yaml; complexa filter configs/inference_configs/inf_${BEST_ID}_my_sweep.yaml; complexa evaluate configs/eval_configs/eval_${BEST_ID}_my_sweep.yaml; complexa analyze configs/eval_configs/eval_${BEST_ID}_my_sweep.yaml" \
    --skill complexa-sweep \
    --out ./sweep_runs/my_sweep/sweep_manifest.json
```

Alongside the manifest, save the ranked `sweep_summary.csv` from Step 6 to `./sweep_runs/my_sweep/sweep_summary.csv` and surface both paths to the user.

## Recommended sweep recipes

| Symptom | Sweep this | Why |
|---|---|---|
| Quality not good enough | `beam_width × nsteps` (use `example.yaml` as a starting point) | Both raise compute → quality; find the cheapest combination that lands. |
| Too slow / want speed-up | `nsteps` downward (e.g. `[100, 200, 400]`) with fixed `beam_width=4` | Find the smallest nsteps that retains success rate. |
| Mode collapse / low diversity | `generation.model.bb_ca.simulation_step_params.sc_scale_noise` (use `bb_ca_temperature.yaml`) | Higher noise → more diverse backbones. |
| Reward over-fitting (high reward, bad metrics) | `generation.reward_model.reward_models.af2folding.reward_weights.{i_pae, plddt}` ratio | Re-balance composite reward. |
| Want statistical robustness on one config | `search_replicas.yaml` (`best_of_n.replicas`) | Same config, more samples → tighter success-rate estimate. |
| Algorithm shoot-out | `generation.search.algorithm` over `[single-pass, best-of-n, beam-search, fk-steering]` | Compare search regimes; pin everything else. |

## Hardware

Total GPU-time for a sweep = `N_configs × per_run_GPU_time`. A single `search_binder_local_pipeline` run on one A100 is roughly 30–90 min (binder length + nsteps dependent). A 4-axis × 4-value sweep = 256 configs × ~60 min = ~256 GPU-hours — at that scale, plan on a multi-GPU host with the Step-4 sharding pattern.

Refer to [`.claude/skills/_shared/reference/hardware.md`](../_shared/reference/hardware.md) for the per-run baseline + VRAM minima.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Sweeper file not found` from `generate_inference_configs.py` | Path resolved from wrong CWD | Use a path relative to the repo root; or pass an absolute path. |
| `Sweeper YAML must be a mapping` | Top-level YAML is a list or scalar | Rewrite as `key: [v1, v2]` mapping. |
| Generator exits 0 but writes nothing | One of the value lists is empty `[]` | **There is no error message.** `itertools.product` over an empty axis yields no combinations, so `generate_inference_configs.py:450` logs `Generated 0 config pair(s)` and exits 0 — a silent no-op. (The string "No configs were generated" does not exist in the repo.) Always check the `Generated N config pair(s)` line, or `ls configs/inference_configs/`, before launching. |
| `complexa design` rejects `--sweeper` | Confusion between entrypoints | The CLI does not accept `--sweeper`. Use `generate_inference_configs.py` first, then run the four stages per pair as in Step 4. |
| Override silently collapses a sweep axis | `--override key=v` shadowed a sweep key | Drop either the override OR the matching key from the sweeper file (`build_sweeper`, `:160-162`). |
| Structures exist under `./inference/inf_*` but no evaluation results anywhere | The sweep was run with `complexa design inf_N.yaml`, so `evaluate`/`analyze` never saw `sample_storage_path` | Re-run only the tail: `complexa evaluate configs/eval_configs/eval_N_<run>.yaml && complexa analyze configs/eval_configs/eval_N_<run>.yaml`. See the Step 4 warning. |
| One config in the loop fails, sweep keeps going | The `\|\| echo "FAILED…"` in Step 4 swallows the error | Re-run that combination's `run_one` standalone; check the per-stage log under `./logs/`. Skip failed `config_id` when ranking. |

---

For per-axis reference (typical ranges, cost, what gets better/worse), see [reference/sweep_axes.md](reference/sweep_axes.md).

For the user-facing sweep system overview (config generation, output layout), see [`docs/SWEEP.md`](../../../docs/SWEEP.md).
