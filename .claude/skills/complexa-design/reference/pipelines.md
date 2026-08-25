# Pipeline Reference

The three `complexa design` pipelines side-by-side. Every row is sourced from a
config file under `configs/pipeline/` or a top-level `configs/search_*.yaml`.
When this doc disagrees with the configs, the configs win — re-read them.

## Three pipelines

| Aspect | Protein Binder | Ligand Binder | AME (motif + ligand) |
|--------|----------------|---------------|----------------------|
| Pipeline YAML | `configs/search_binder_local_pipeline.yaml` | `configs/search_ligand_binder_local_pipeline.yaml` | `configs/search_ame_local_pipeline.yaml` |
| Generate config | `pipeline/binder/binder_generate.yaml` | `pipeline/ligand_binder/ligand_binder_generate.yaml` | `pipeline/ame/ame_generate.yaml` |
| Evaluate config | `pipeline/binder/binder_evaluate.yaml` | `pipeline/ligand_binder/ligand_binder_evaluate.yaml` | `pipeline/ame/ame_evaluate.yaml` |
| Analyze config | `pipeline/binder/binder_analyze.yaml` | `pipeline/ligand_binder/ligand_binder_analyze.yaml` | `pipeline/ame/ame_analyze.yaml` |
| Model ckpt | `complexa.ckpt` | `complexa_ligand.ckpt` | `complexa_ame.ckpt` |
| Autoencoder ckpt | `complexa_ae.ckpt` | `complexa_ligand_ae.ckpt` | `complexa_ame_ae.ckpt` |
| LoRA | (none) | `r: 32, lora_alpha: 64.0` | `r: 32, lora_alpha: 64.0` |
| `env_vars.USE_V2_COMPLEXA_ARCH` | (not set) | (not set) | `"True"` |
| Targets dict | `configs/targets/targets_dict.yaml` | `configs/targets/ligand_targets_dict.yaml` | `configs/design_tasks/ame_dict_v2.yaml` |
| Conditional features | `TargetFeatures` | `LigandFeatures` | `MotifFeatures` + `LigandFeatures` |
| Default search algorithm | `best-of-n` | `best-of-n` | `single-pass` |
| Generation reward model | AF2 (`af2folding`) | RF3 (`rf3folding`) | `null` (single-pass) |
| Inverse-folding model | `soluble_mpnn` | `ligand_mpnn` | `ligand_mpnn` |
| Evaluation `protein_type` | `binder` | `binder` | `motif_binder` |
| Evaluation folding | `colabdesign` (AF2) | `rf3_latest` | `rf3_latest` |
| Analysis `result_type` | `protein_binder` | `ligand_binder` | `motif_ligand_binder` |
| Analysis modes | `[binder, monomer]` | `[binder, monomer]` | `[motif_binder, monomer]` |
| Default `gen_njobs` / `eval_njobs` | 1 / 1 | 1 / 1 | 1 / 1 |
| Default `dataloader.batch_size` | 16 | 16 | 16 |

## Protein binder

Designs a binder for a protein target. AF2 (ColabDesign) drives both the
reward during search and the final evaluation refold. Uses `SolubleMPNN` for
inverse folding so the redesigned binder is soluble.

- Reward weights: `af2folding.reward_weights.i_pae = -1.0` (minimize interface
  PAE). Other AF2 metrics (`con`, `plddt`, `dgram_cce`, `min_ipae`, `min_ipsae`,
  `avg_ipsae`, `max_ipsae`, `min_ipsae_10`, `max_ipsae_10`, `avg_ipsae_10`)
  default to 0.0 and can be enabled by override.
- Default success thresholds: `i_pAE * 31 <= 7.0`, `pLDDT >= 0.9`, `scRMSD_ca <
  1.5` Å, and `apo scRMSD_ca < 2.0` Å — **four** criteria
  (`DEFAULT_PROTEIN_BINDER_THRESHOLDS`, `binder_analysis_utils.py:76-115`). The apo
  one asks whether the binder folds as designed *without* its target; it needs
  `metric.compute_apo_metrics` (on by default) and a criterion whose column is
  absent removes gating entirely rather than weakening it.
  The key is `scRMSD_ca`, not `scRMSD`, and these thresholds can only be
  overridden as a complete dict — see
  [overrides.md](overrides.md) "Threshold overrides must be complete dicts".
- Quick command:
  ```bash
  complexa design configs/search_binder_local_pipeline.yaml \
      ++run_name=pdl1_v1 ++generation.task_name=02_PDL1
  ```

## Ligand binder

Designs a binder for a small-molecule ligand. RF3 handles both reward and
evaluation because it supports protein-ligand complexes. Uses `LigandMPNN` so
the inverse-folded sequence is conditioned on the ligand.

- Reward weights: `rf3folding.reward_weights.min_ipAE = -1.0` (minimize the
  minimum interface PAE, normalized by 31). Other RF3 metrics (`plddt`, `ipAE`,
  `mean_min_ipAE`, `mean_ipAE`, `min_mean_ipAE`, `pAE`, `ipTM`, `pTM`,
  `ranking_score`, `has_clash`, `min_ipSAE`, `max_ipSAE`, `avg_ipSAE`) default
  to 0.0.
- Default success thresholds: `min_ipAE * 31 < 2.0`, `scRMSD_ca < 2.0` Å,
  `ligand_scRMSD_aligned_allatom < 5.0` Å.
- LoRA is required for the released ligand checkpoint (`r: 32, lora_alpha:
  64.0, lora_dropout: 0.0, train_bias: none`) and is baked into
  `search_ligand_binder_local_pipeline.yaml`.
- Quick command:
  ```bash
  complexa design configs/search_ligand_binder_local_pipeline.yaml \
      ++run_name=v11_v1 ++generation.task_name=39_7V11_LIGAND
  ```

## AME (motif + ligand binder)

Combines `MotifFeatures` (atom-spec mode) with `LigandFeatures`. Designed for
active-site / enzyme scaffolding: place a functional motif near a small-molecule
ligand and let the model build the rest of the protein around them.

- Generation reward defaults to `null` (single-pass). To enable reward-guided
  search, uncomment the `CompositeRewardModel` block in `ame_generate.yaml` (it
  pre-wires RF3) and switch `search.algorithm` to `best-of-n` or `beam-search`.
- LoRA is required: `r: 32, lora_alpha: 64.0, lora_dropout: 0.0, train_bias:
  none`.
- `env_vars.USE_V2_COMPLEXA_ARCH: "True"` is set in
  `search_ame_local_pipeline.yaml`. The CLI runner injects this into the
  subprocess environment.
- Default success thresholds (`motif_ligand_binder`), from
  `DEFAULT_MOTIF_LIGAND_BINDER_SUCCESS` (`motif_binder_analysis_utils.py:72-89`):
  the binder side is **only** `scRMSD_bb3 <= 2.0`; the motif side is
  `motif_rmsd_pred <= 1.5`, `correct_motif_sequence >= 1.0`, and
  `has_ligand_clashes < 0.5`. There is no `i_pAE` and no `pLDDT` criterion — the
  comment at `ame_analyze.yaml:27-29` that says otherwise is stale. The analysis
  is *joint*: a sample passes only when one redesign satisfies binder + motif
  criteria simultaneously.
- Pre- and post-refolding interface metrics are **disabled**:
  `compute_pre_refolding_metrics` and `compute_refolded_structure_metrics` are
  both `false` (`ame_evaluate.yaml:73, :81`), as are all four sub-toggles
  (`bioinformatics`, `tmol`). There is no HBPLUS toggle — `evaluate.py:405-407`
  reads only `bioinformatics` and `tmol`.
- Quick command:
  ```bash
  complexa design configs/search_ame_local_pipeline.yaml \
      ++run_name=ame_chm ++generation.task_name=M0096_1chm
  ```

## LoRA settings

Both non-protein pipelines (ligand binder, AME) require LoRA because their
checkpoints (`complexa_ligand.ckpt`, `complexa_ame.ckpt`) were trained with
LoRA adapters. The pipeline YAMLs declare:

```yaml
lora:
  r: 32
  lora_alpha: 64.0
  lora_dropout: 0.0
  train_bias: none
```

Removing the `lora:` block from the pipeline YAML will load the base weights
without the LoRA delta and produce garbage. Leave it alone unless the user is
fine-tuning their own LoRA — in which case override `r`, `lora_alpha`,
`lora_dropout`, or `train_bias` via `++lora.r=64`.

## `USE_V2_COMPLEXA_ARCH` toggle

This env var picks between two on-disk architecture layouts. The CLI runner
reads `env_vars:` from the pipeline YAML and injects them into the subprocess
environment (see `cli_runner.py::run_step` -> `env.update(config.env_vars)`).

| Pipeline | `USE_V2_COMPLEXA_ARCH` | Notes |
|----------|------------------------|-------|
| Protein binder | (not set; defaults to v1) | `complexa.ckpt` is v1 architecture. |
| Ligand binder | (not set; defaults to v1) | `complexa_ligand.ckpt` is v1 architecture. |
| AME | `"True"` | `complexa_ame.ckpt` is v2 architecture for the AME pipeline. |

If the user sees `KeyError` or shape mismatches at model-load time, the most
likely cause is the wrong `USE_V2_COMPLEXA_ARCH` value — the configs in the
repo are correct, but a custom override (`++env_vars.USE_V2_COMPLEXA_ARCH=...`)
can break it.

## Evaluation `protein_type` and `result_type`

`protein_type` (set in the evaluation config) controls *which metrics are
computed*. `result_type` (set in the analysis config) controls *which default
success thresholds apply*. They are paired:

| Pipeline | `protein_type` | `result_type` |
|----------|----------------|---------------|
| Protein binder | `binder` | `protein_binder` |
| Ligand binder | `binder` | `ligand_binder` |
| AME | `motif_binder` | `motif_ligand_binder` |
| Motif protein binder (standalone, not a `design` pipeline) | `motif_binder` | `motif_protein_binder` |

The motif protein binder evaluation is not a top-level `design` pipeline, and
**no motif-protein-binder evaluate config ships in `configs/`**. There is no
`configs/evaluate_motif_binder.yaml`; the only related files are the
fully-commented template `configs/example/evaluate_motif_binder.yaml` (which
defaults to the AME dict, i.e. `motif_ligand_binder`) and
`configs/evaluate_motif_from_pdb_dir.yaml`, which is `monomer_motif` — a
different task. Running `motif_protein_binder` means writing your own evaluate
config; the `complexa-evaluate-pdbs` skill covers the configs that do ship.

## Evaluation types (summary)

From `docs/EVALUATION_METRICS.md`, the types reachable from the three design
pipelines:

1. **Protein binder** (`protein_type: binder`, `result_type: protein_binder`) —
   AF2 (ColabDesign) or RF3 refold + SolubleMPNN redesign. Metrics: i_pAE, i_pTM,
   pLDDT, binder scRMSD, complex scRMSD.
2. **Ligand binder** (`protein_type: binder`, `result_type: ligand_binder`) —
   RF3 refold + LigandMPNN. Adds min_ipAE, ipSAE, has_clash, ligand_scRMSD,
   ligand_scRMSD_aligned_allatom.
3. **Monomer** (`protein_type: monomer`) — ESMFold designability +
   codesignability + secondary structure. Used as a component of binder
   evaluation.
4. **Motif protein binder** (`protein_type: motif_binder`, `result_type:
   motif_protein_binder`) — Standalone variant of #1 + motif RMSD / sequence
   recovery on the refolded structure. Reachable in code, but **no shipped
   config selects it** (see the note above) — you must author one. Not a
   top-level design pipeline.
5. **Motif ligand binder / AME** (`protein_type: motif_binder`, `result_type:
   motif_ligand_binder`) — #2 + motif overlay + ligand clash detection on the
   refolded structure. This is what `search_ame_local_pipeline.yaml` runs.

Pass-rate columns land in `filter_results/res_filter_*_pass_*.csv` (AME's
`res_filter_motif_binder_pass_*.csv` goes to `motif_binder_metrics/` instead);
diversity columns in `diversity/res_div_foldseek_*.csv` and
`diversity/res_div_mmseqs_*.csv`; per-design metrics in the combined CSV. The
subdirectories are created by `organize_results()` (`analyze.py:2813-2856`),
which moves these files out of the results-dir root.
