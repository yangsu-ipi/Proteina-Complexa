---
name: complexa-evaluate-pdbs
description: >
  Standalone evaluation of an existing PDB directory with Proteina-Complexa.
  Use this skill whenever the user wants to "evaluate PDB files", "re-fold these
  designs", "compute interface pAE", "compute i_pLDDT for a folder",
  "run AF2 / RF3 / ESMFold on my designs", "score binder candidates",
  "designability of this folder", "scRMSD for designs", "motif RMSD for these
  PDBs", "complexa analysis", "complexa evaluate from a PDB directory",
  "evaluate from pdb dir", or score third-party outputs (BindCraft, AlphaProteo,
  RFdiffusion, hand-curated decoys). It picks the correct `evaluate_*.yaml`
  config, wires `++sample_storage_path` and the folding backend, runs
  `complexa analysis` (the evaluate → analyze chain), parses the result CSV,
  reports pass-rates against the right `result_type` thresholds, and emits a
  replayable `eval_manifest.json`. Reach for this skill before hand-rolling
  refolding scripts.
compatibility: "complexa CLI installed (pip install -e .); CUDA GPU; AF2_DIR (colabdesign) or RF3_CKPT_PATH+RF3_EXEC_PATH (rf3_latest); ESMFold weights for monomer paths"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Complexa Evaluate-PDBs Skill

Score a directory of pre-existing PDB files against the same metrics Proteina-Complexa uses internally. Wraps `complexa analysis <evaluate_config> ++sample_storage_path=<dir>`: the CLI runs the `evaluate` step (refold + interface metrics + monomer metrics) and then the `analyze` step (success thresholds, diversity, pass-rate CSVs). Do **not** run `complexa generate` here — the inputs already exist.

## What this skill enables

- Re-fold a directory of designed PDBs with AF2 (`colabdesign`) or RF3 (any value containing `rf3`, e.g. `rf3_latest`). Those are the **only** two values `metric.binder_folding_method` accepts — `binder_eval.py:108-128` raises `ValueError: Folding model '<x>' not supported` for anything else, including `esmfold`, `boltz2_default` and `protenix_*` (the stale comments at `evaluate_from_pdb_dir.yaml:70` and `binder_evaluate.yaml:23` notwithstanding). `esmfold` is valid only for the *monomer* key `metric.monomer_folding_models` (`monomer_eval_utils.py:30`: `VALID_FOLDING_MODELS = ["esmfold", "colabfold"]`) — the two keys are easy to conflate.
- Compute binder interface metrics: `i_pAE`, `min_ipAE`, `i_pTM`, `pLDDT`, binder/complex scRMSD.
- Compute monomer **designability** (ProteinMPNN-redesigned scRMSD) and **codesignability** (original sequence refold scRMSD).
- For motif inputs: motif RMSD (CA + all-atom), motif-region designability/codesignability, sequence recovery.
- Aggregate into per-PDB CSVs plus pass-rate summaries using the default thresholds for the `result_type`.

## Step 1: Pre-flight

Always check GPU / disk / tool binaries before launching a refold job. RF3 and ColabDesign-AF2 are large.

```bash
bash .claude/skills/_shared/scripts/preflight.sh
```

Surface from `preflight.json`:

- `gpu.available` and `gpu.vram_gb` — colabdesign/RF3 need ≥40 GB; the optional monomer ESMFold stage tolerates ≥24 GB.
- `env.missing_required` — must include the keys for the chosen folding backend:
  - `colabdesign` → `AF2_DIR`
  - `rf3_latest` (or any `*rf3*` name) → `RF3_CKPT_PATH`, `RF3_EXEC_PATH`
  - ESMFold weights matter only if you additionally enable monomer metrics (`metric.compute_monomer_metrics=true` + `metric.monomer_folding_models=[esmfold]`). ESMFold is not a `binder_folding_method`; there is no third refolding backend to preflight for.
- `tools.{foldseek,mmseqs}` — required by `aggregation.compute_diversity` / `compute_mmseqs_diversity` (both default `true`).

If any required key is missing, route the user to the `complexa-setup` skill.

## Step 2: Identify the design type

Like `complexa design`, evaluation has one default flow (protein binder) and two extensions (ligand binder, AME). The evaluate config you pass to `complexa analysis` decides everything else (which metrics, which refolder defaults, which thresholds the analyze step applies).

### Default — protein binder

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=02_PDL1 \
    ++result_type=protein_binder \
    ++metric.binder_folding_method=colabdesign \
    ++metric.inverse_folding_model=soluble_mpnn \
    ++run_name=eval_pdl1_af2
```

Use this when the user's PDBs are protein-binder designs (multi-chain, binder is the last chain) or third-party outputs from BindCraft / AlphaProteo / RFdiffusion. Pulls thresholds for `protein_binder` (`i_pAE * 31 ≤ 7.0`, `pLDDT ≥ 0.9`, `scRMSD_ca < 1.5 Å`).

> **Defect in the shipped config — read before running the command above.**
> `configs/evaluate_from_pdb_dir.yaml:22` composes `defaults: - generation/targets_dict@dataset`,
> but there is no `configs/generation/targets_dict.yaml` (`configs/generation/` ships only
> `base_gen_data.yaml`, `validation.yaml`, `validation_local_latents.yaml`). Hydra fails to
> compose the config before any GPU work starts. The real dicts are
> `configs/targets/targets_dict.yaml` and `configs/targets/ligand_targets_dict.yaml`, and the
> group is fixed by the defaults list, so it **cannot** be redirected from the CLI. Two working
> routes:
>
> - **Protein-binder targets** — use `configs/evaluate.yaml`, which composes the right dict
>   (`:31`, `- /targets/targets_dict@dataset`), and add `++input_mode=pdb_dir`. It defines no
>   `result_type`, so set it explicitly:
>   `complexa analysis configs/evaluate.yaml ++input_mode=pdb_dir ++sample_storage_path=<dir> ++dataset.task_name=02_PDL1 ++result_type=protein_binder ++metric.binder_folding_method=colabdesign ++metric.inverse_folding_model=soluble_mpnn ++run_name=<run>`
>   (its shipped `dataset.task_name: COM03_NIPAH` at `:88` is not a key in `targets_dict.yaml`,
>   so the override is mandatory).
> - **Ligand targets** — no shipped evaluate config composes `ligand_targets_dict.yaml` at all.
>   Copy `configs/evaluate_from_pdb_dir.yaml` and change line 22 to
>   `- /targets/ligand_targets_dict@dataset`. Without the right dict, `get_target_info` raises
>   `target_task_name <name> not found in target_dict_cfg` (`binder_eval_utils.py:238-252`).

### Extensions — pick the matching config

One config drives both steps, so there is no separate "analyze config" to choose. `complexa
analysis` takes a single config and threads the same overrides through `evaluate` and `analyze`
(`cli_runner.py:1021-1042`), and the `analyze` step locates the per-job CSVs by that config's
stem (`find_result_files(results_dir, config_name, …)`) — pass the **evaluate** config to both.
The standalone `configs/analyze*.yaml` files are not runnable on their own: they define neither
`results_dir` nor `output_dir`, so `complexa analyze configs/analyze.yaml` auto-constructs
`./evaluation_results/analyze` and exits 1 with `results_dir does not exist`
(`analyze.py:2921, :2946-2958`; `validate_config` at `:390-409`).

| Design type | Use the protein-binder default? | Config (drives `evaluate` **and** `analyze`) | `result_type` | Backend as shipped |
|---|---|---|---|---|
| Protein binder | **Yes (default)** | `configs/evaluate_from_pdb_dir.yaml` (see the defect note above) | `protein_binder` — **must be overridden** | `rf3_latest` (`:72`) |
| Ligand binder (binder + small-molecule) | Same config, no overrides needed | `configs/evaluate_from_pdb_dir.yaml` | `ligand_binder` (shipped default, `:167`) | `rf3_latest` (`:72`) |
| AME / motif + ligand (enzyme outputs) | No — needs motif-aware config | `configs/evaluate_ame_from_pdb_dir.yaml` | `motif_ligand_binder` | `rf3_latest` |
| Motif protein binder (standalone) | No — no `_from_pdb_dir` variant | `configs/example/evaluate_motif_binder.yaml` + `++input_mode=pdb_dir` | `motif_protein_binder` — override; the config's own default is `motif_ligand_binder` | `rf3_latest` (`:75`) |

`configs/evaluate_from_pdb_dir.yaml` as shipped is a **ligand-binder** config:
`binder_folding_method: rf3_latest` (`:72`), `inverse_folding_model: ligand_mpnn` (`:84`),
`result_type: ligand_binder` (`:167`), `aggregation.analysis_modes: [binder]` (`:174`). Omit the
overrides in the protein-binder command above and you get a ligand-binder run, not an AF2
protein-binder one.

**Extending to ligand binder** — the three `metric.*`/`result_type` values below are already
the shipped defaults, so they document intent rather than change behaviour. What you *do* have to
change is the target dict: run against a copy of the config whose defaults line names
`- /targets/ligand_targets_dict@dataset` (see the note above), not
`configs/evaluate_from_pdb_dir.yaml` itself.

```bash
complexa analysis configs/evaluate_from_pdb_dir_ligand.yaml \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=39_7V11_LIGAND \
    ++result_type=ligand_binder \
    ++metric.binder_folding_method=rf3_latest \
    ++metric.inverse_folding_model=ligand_mpnn \
    ++run_name=eval_v11_rf3
```

Run against the pristine `configs/evaluate_from_pdb_dir.yaml`, this fails twice: the
`generation/targets_dict` group does not exist (above), and even if it did, `39_7V11_LIGAND`
lives in `configs/targets/ligand_targets_dict.yaml:2` — not `targets_dict.yaml` — so
`get_target_info` would raise `not found in target_dict_cfg`
(`binder_eval_utils.py:238-252`). Nothing in the code dispatches between the two dicts by task
name; the lookup uses whatever `dataset.target_dict_cfg` Hydra composed.

**Extending to AME** (different config; ligand auto-completion gotcha — see Step 4):

```bash
complexa analysis configs/evaluate_ame_from_pdb_dir.yaml \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=M0096_1chm \
    ++run_name=eval_ame_chm
```

See `reference/eval_configs.md` for the full matrix (every `result_type`, every threshold default, every supported folding backend, motif-protein-binder variant).

## Step 3: Gather inputs (AskUserQuestion)

Ask in one batched `AskUserQuestion`:

1. **`pdb_dir`** — absolute path to the directory of PDBs to evaluate.
2. **Design type** — protein binder / ligand binder / AME (motif + ligand).
3. **Folding backend** — `colabdesign` (AF2, protein binders) or `rf3_latest` (ligand / AME). Offer exactly these two; every other value raises `ValueError` in `binder_eval.py:108-128`.
4. **Target / task name** — must match a key in `configs/targets/targets_dict.yaml`, `configs/targets/ligand_targets_dict.yaml`, or `configs/design_tasks/ame_dict_v2.yaml`. Required for binder + AME evaluation (needed to identify the target reference and, for AME, the motif contigs).
5. **AME-only**: confirm ligand residue name is already renamed to `L:0` in every PDB (see Troubleshooting). If not, do that rename first.

## Step 4: Run evaluate → analyze

Prefer `complexa analysis` (the evaluate→analyze chain) — it reuses the same config for both steps and writes a single log dir.

```bash
# Protein binder PDB dir, AF2 refold
complexa analysis configs/evaluate_from_pdb_dir.yaml \
  ++sample_storage_path=/abs/path/to/pdbs \
  ++dataset.task_name=02_PDL1 \
  ++metric.binder_folding_method=colabdesign \
  ++metric.inverse_folding_model=soluble_mpnn \
  ++result_type=protein_binder \
  ++run_name=eval_pdl1_af2
```

For ligand binders flip `binder_folding_method=rf3_latest`, `inverse_folding_model=ligand_mpnn`, `result_type=ligand_binder`. For AME use `configs/evaluate_ame_from_pdb_dir.yaml` — see `reference/eval_configs.md` for full worked examples.

If you need to inspect output between stages, run them separately. The configs above are shared between `evaluate` and `analyze`:

```bash
complexa evaluate configs/evaluate_from_pdb_dir.yaml ++sample_storage_path=/abs/path/to/pdbs ++run_name=eval_pdl1_af2
complexa analyze  configs/evaluate_from_pdb_dir.yaml ++run_name=eval_pdl1_af2
```

Dry-run first if the user is unsure (no GPU work happens; the planned file walk + invocation prints):

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml ++sample_storage_path=/abs/path/to/pdbs ++dryrun=true
```

### Direct module invocation (debug fallback)

`complexa evaluate` / `analyze` are subprocess wrappers around the Hydra
modules with logging + parallel job splitting bolted on. To attach a debugger
or run under a profiler, invoke the module directly:

```bash
python -m proteinfoundation.evaluate \
    --config-path "$(realpath configs)" \
    --config-name evaluate_from_pdb_dir \
    ++sample_storage_path=/abs/path/to/pdbs \
    ++dataset.task_name=02_PDL1 \
    ++metric.binder_folding_method=colabdesign \
    ++run_name=eval_debug
```

For normal one-shot runs prefer `complexa analysis` — you get the shared log
dir and a single replayable invocation, instead of having to thread the same
overrides through two `python -m` calls.

## Step 5: Parse results

Output lands in the config's `output_dir`, with `run_name` appended unless it is already the
suffix (`evaluate.py:766-768`; `analyze.py:2956-2958` does the same for `results_dir`). For
`evaluate_from_pdb_dir.yaml` that resolves to `./evaluation_results/${run_name}` because
`output_dir: ./evaluation_results/${run_name}` (`:39`).

- **Per-job CSV** from `evaluate` — `binder_results_{config_name}_{job_id}.csv`
  (`evaluate.py:881`). The other flavours are `monomer_results_*` (`:853`), `motif_results_*`
  (`:904`) and `motif_binder_results_*` (`:930`). `{config_name}` is the config file stem
  (`cli_runner.py:650` passes `++base_config_name=<stem>`).
- **Primary CSV** from `analyze` — `RAW_{result_type}_results_{config_name}_combined.csv`
  (`analyze.py:3036`), plus a transposed twin. There is **one row per input PDB**: `id_gen` is an
  enumerate index over the file walk (`binder_eval.py:377, :383`). `sequence_types` are column
  **prefixes** on that single row, not extra rows — `self_complex_i_pAE`,
  `mpnn_fixed_binder_scRMSD_ca`, `self_sequence`, and `_all` variants holding the per-redesign
  lists (`binder_eval.py:476-512`; `binder_analysis_utils.py:160-171` builds
  `{seq}_{prefix}_{metric}_all`).
- **Pass-rate summaries and everything else are moved into subdirectories** by
  `organize_results` (`analyze.py:2802-2880`), so do not glob the top level:
  `res_filter_*` → `filter_results/`, `res_div_*` → `diversity/`, `res_monomer_*` →
  `monomer_metrics/`, `res_ss_*` → `secondary_structure/`, `res_aa_*` →
  `amino_acid_distribution/`, `res_motif_*` → `motif_metrics/`,
  `res_motif_binder_*` / `res_filter_motif_binder_*` → `motif_binder_metrics/`, and the
  `clusters_*` directories → `clusters/`. The binder pass-rate file is
  `filter_results/res_filter_binder_pass_*.csv` for `protein_binder` and
  `res_filter_ligand_pass_*.csv` for `ligand_binder` (`binder_analysis.py:548`);
  motif runs write `res_filter_motif_binder_pass_*.csv` (`motif_binder_analysis.py:252`).
- Diversity output — FoldSeek/MMseqs2 cluster files under `diversity/` and `clusters/` when
  `aggregation.compute_diversity=true` (default).

Summarize to the user:

- Per-PDB row count and number of successful designs vs total.
- Default-threshold pass rate by `result_type` (e.g. for `protein_binder`: `i_pAE*31 <= 7.0 AND pLDDT >= 0.9 AND scRMSD_ca < 1.5`).
- Top 5 designs by primary metric (`i_pAE` for protein, `min_ipAE` for ligand, `motif_rmsd_pred_all` for motif binders).

## Step 6: Emit manifest

```bash
python3 .claude/skills/_shared/scripts/write_manifest.py \
  --output-dir ./evaluation_results/${run_name} \
  --command "complexa analysis configs/evaluate_from_pdb_dir.yaml ++sample_storage_path=<dir> ++dataset.task_name=<task> ++metric.binder_folding_method=<backend> ++run_name=<run>" \
  --skill complexa-evaluate-pdbs \
  --out ./eval_manifest.json
```

It pins `timestamp`, `skill`, `command`, `output_dir`, `git_sha`, `repo_root`, the CSV pointers
found by walking `--output-dir`, and the invocation environment (`write_manifest.py:179-200`).
Two caveats:

- There is **no** `result_type` field — put it in the `--command` string to keep it replayable
  (the `++result_type=…` override above already does).
- `config`, `config_path` and `checkpoints` come out `null`, with no checkpoint hashes: they are
  read from `<output-dir>/.hydra/config.yaml` (`write_manifest.py:79-80`) and no `complexa` stage
  writes `.hydra/` there — the pipeline configs point Hydra at `./logs/hydra_outputs/…`.

## Most common overrides

| Override                                       | Effect                                                                |
|------------------------------------------------|-----------------------------------------------------------------------|
| `++sample_storage_path=<dir>`                  | The directory of PDBs to evaluate (required).                         |
| `++dataset.task_name=<name>`                   | Target / AME task name (binders, AME). Resolves target PDB + contigs. |
| `++metric.binder_folding_method=<backend>`     | `colabdesign`, or any name containing `rf3` (e.g. `rf3_latest`). Nothing else is accepted (`binder_eval.py:108-128`). |
| `++metric.inverse_folding_model=<model>`       | `protein_mpnn` / `soluble_mpnn` / `ligand_mpnn`.                      |
| `++metric.sequence_types=[self,mpnn,mpnn_fixed]` | Which sequence flavors to refold.                                   |
| `++metric.num_redesign_seqs=N`                 | ProteinMPNN/LigandMPNN redesign count.                                |
| `++metric.compute_pre_refolding_metrics=true`  | Add bioinformatics/TMOL metrics on the input structures. The only sub-toggles are `metric.pre_refolding.{bioinformatics,tmol}` — `evaluate.py:401-403` reads no others, and there is no `hbplus` reward or key anywhere in `src/`. |
| `++metric.keep_folding_outputs=true`           | Save the refolded PDBs (large, but useful for inspection).            |
| `++result_type=<type>`                         | Override default thresholds: `protein_binder` / `ligand_binder` / `motif_protein_binder` / `motif_ligand_binder`. |
| `++aggregation.success_thresholds.<…>`         | Tighten or loosen specific thresholds (see `reference/eval_configs.md`). |
| `++eval_njobs=N`                               | Parallel GPUs for the evaluate step.                                  |
| `++dryrun=true`                                | Plan without running any folding.                                     |
| `++file_limit=N`                               | Cap input PDBs (handy for first-pass smoke tests).                    |

## Hardware

- **GPU**: ≥1 CUDA GPU. Both refolding backends — AF2 (`colabdesign`) and RF3 (`rf3_latest`) — need ≥40 GB VRAM (A100/H100/L40S). The optional monomer ESMFold stage runs on ≥24 GB. Multi-GPU via `++eval_njobs=N`.
- **CPU/disk**: 24 CPUs default (`ncpus_: 24`). Each refolded PDB + intermediate output is ~1–5 MB; `keep_folding_outputs=true` can balloon to tens of GB for thousands of inputs.
- See `_shared/reference/hardware.md` for per-backend wall-clock and VRAM tables.

## Troubleshooting

- **`Error: Config file not found`** — paths are relative to the repo root; `cd` to the repo before invoking `complexa analysis`.
- **`compute_motif_binder_metrics=True` but `result_type=protein_binder`** — `result_type` and the underlying `compute_*_metrics` must agree. Use `configs/evaluate_ame_from_pdb_dir.yaml` for AME inputs rather than mutating `evaluate_from_pdb_dir.yaml`.
- **RF3 shape errors on AME PDBs** — RF3 tries to auto-complete the ligand atoms from CCD. Rename the ligand residue to `L:0` in every input PDB before evaluation; see the snippet in `README.md` (`atom_array.res_name[ligand_mask] = "L:0"`).
- **Diversity step fails (`foldseek not found`)** — `FOLDSEEK_EXEC` not on `PATH`. Either fix `.env` (preferred) or disable: `++aggregation.compute_diversity=false ++aggregation.compute_mmseqs_diversity=false`.
- **All pass-rates are 0%** — check the `binder_folding_method` matches the target type (RF3 for ligand, AF2 for protein) and that `++dataset.task_name` resolves to the correct reference PDB (`complexa target show <name>` to verify).

## Reference

Full evaluate/analyze config matrix, every supported `result_type`, per-threshold defaults, and three worked examples (protein binder / ligand binder / AME): see `reference/eval_configs.md`.
