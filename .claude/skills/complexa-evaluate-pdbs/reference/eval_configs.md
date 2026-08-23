# Evaluate-from-PDB-Dir Config Reference

Companion to `SKILL.md`. Every `evaluate_*_from_pdb_dir.yaml` and its paired `analyze_*.yaml`, with the `result_type` they emit, the `dataset.*` keys they require, and the `metric.*` / `aggregation.*` knobs they accept. Three worked examples at the bottom.

## 1. Design type → config matrix

| Design type            | Evaluate config                              | Analyze config              | `result_type`            | Folding backends         | Inverse-folding default |
|------------------------|----------------------------------------------|-----------------------------|--------------------------|--------------------------|-------------------------|
| Protein binder         | `configs/evaluate_from_pdb_dir.yaml`         | `configs/analyze.yaml`      | `protein_binder`         | `colabdesign`, `rf3_latest` | `soluble_mpnn` |
| Ligand binder          | `configs/evaluate_from_pdb_dir.yaml`         | `configs/analyze.yaml`      | `ligand_binder`          | `rf3_latest` (default)   | `ligand_mpnn` |
| AME / motif ligand binder | `configs/evaluate_ame_from_pdb_dir.yaml`  | `configs/analyze_motif_binder.yaml` | `motif_ligand_binder` | `rf3_latest`            | `ligand_mpnn`           |
| Motif protein binder   | `configs/example/evaluate_motif_binder.yaml` (no `_from_pdb_dir` variant; set `input_mode=pdb_dir`) | `configs/analyze_motif_binder.yaml` | `motif_protein_binder` | `colabdesign`, `rf3_latest` | `protein_mpnn` / `soluble_mpnn` |

Notes:
- **The folding-backend column is exhaustive.** `metric.binder_folding_method` accepts `colabdesign` and any name containing `rf3`, and nothing else: `binder_eval.py:97-117` ends in `raise ValueError(f"Folding model '{folding_model}' not supported")`. `esmfold`, `boltz2_default` and `protenix_base_default_v0.5.0` all crash the evaluate step, despite the comments at `evaluate_from_pdb_dir.yaml:70`, `binder_evaluate.yaml:23` and `example/evaluate_motif_binder.yaml:73-74`. `esmfold` belongs to the separate monomer key `metric.monomer_folding_models` (`monomer_eval_utils.py:30`).
- The "Analyze config" column above is informational only. `complexa analysis` takes **one** config for both steps (`cli_runner.py:1021-1042`), and `analyze` finds the per-job CSVs by that config's stem — so pass the *evaluate* config to both. `configs/analyze.yaml` and `configs/analyze_motif_binder.yaml` define neither `results_dir` nor `output_dir`, so running them directly (`complexa analyze configs/analyze.yaml`) exits 1 with `results_dir does not exist: ./evaluation_results/analyze` (`analyze.py:2921`, `validate_config` at `:390-409`).
- The protein-binder and ligand-binder cases share `evaluate_from_pdb_dir.yaml`; switch behavior by setting `result_type`, `metric.binder_folding_method`, and `metric.inverse_folding_model` on the CLI. Note the shipped defaults are the *ligand* ones — `rf3_latest` (`:72`), `ligand_mpnn` (`:84`), `result_type: ligand_binder` (`:139`), `analysis_modes: [binder]` (`:146`).
- There is no shipped `evaluate_motif_protein_binder_from_pdb_dir.yaml`; reuse `configs/example/evaluate_motif_binder.yaml` with `++input_mode=pdb_dir ++sample_storage_path=<dir> ++result_type=motif_protein_binder`. That config composes `- /design_tasks/ame_dict_v2@dataset` (`:24`), so `dataset.task_name` must be a key in `configs/design_tasks/ame_dict_v2.yaml` unless you point it at your own dict — see the `motif_target_dict_cfg` note in §2.

## 2. Evaluate config schema

All `evaluate_*` configs share a top-level shape (run identification, `input_mode`, `protein_type`, `sample_storage_path`, `output_dir`, `eval_njobs`, `seed`, `ncpus_`, `dataset.*`, `metric.*`). Below: the differences that matter when running from a PDB directory.

`motif_target_dict_cfg` is a **key**, not a Hydra config group: `configs/design_tasks/ame_dict_v2.yaml:11` defines `motif_target_dict_cfg:` and the motif configs pull the whole file in with `- /design_tasks/ame_dict_v2@dataset`. There is no `configs/dataset/motif_target_dict_cfg` option to select — `configs/dataset/` contains only `unified/`. To use your own motif dict, swap the defaults entry for another file under `configs/design_tasks/`, or override the leaf entries with `++dataset.motif_target_dict_cfg.<task>.…`.

### `evaluate_from_pdb_dir.yaml`

- `protein_type: binder` (one config handles both protein and ligand binders).
- `input_mode: pdb_dir` (already set; never override back to `generated`).
- `defaults: - generation/targets_dict@dataset` (`:22`) — **broken as shipped.** `configs/generation/targets_dict.yaml` does not exist (`configs/generation/` holds `base_gen_data.yaml`, `validation.yaml`, `validation_local_latents.yaml`), so Hydra cannot compose this config at all. The real dicts live at `configs/targets/targets_dict.yaml` and `configs/targets/ligand_targets_dict.yaml`; `configs/evaluate.yaml:31` composes the first correctly with `- /targets/targets_dict@dataset`.
  - There is also **no dispatch between the two dicts by task name.** `get_target_info` (`binder_eval_utils.py:238-252`) looks `dataset.task_name` up in whatever `dataset.target_dict_cfg` Hydra composed and raises `target_task_name <name> not found in target_dict_cfg` otherwise. A ligand task such as `39_7V11_LIGAND` (in `ligand_targets_dict.yaml:2`) is not reachable from a config that composed `targets_dict.yaml`.
  - Workarounds: for protein targets use `configs/evaluate.yaml` with `++input_mode=pdb_dir ++result_type=protein_binder`; for ligand targets copy this config and set the defaults entry to `- /targets/ligand_targets_dict@dataset`. The group name is fixed by the defaults list, so it cannot be redirected from the CLI.
- Required `dataset.*`:
  - `dataset.task_name` — target key (e.g. `02_PDL1`, `39_7V11_LIGAND`).
- Key `metric.*` fields:
  - `compute_binder_metrics: true`.
  - `binder_folding_method` — picks the refolding backend (table above).
  - `sequence_types: [self|mpnn|mpnn_fixed]` — which sequence(s) to refold.
  - `num_redesign_seqs` — MPNN sequence count (default 8 here).
  - `interface_cutoff` — Å cutoff for interface residue detection (default 8.0 protein, 6.0 motif).
  - `inverse_folding_model` — `soluble_mpnn` / `protein_mpnn` (protein) or `ligand_mpnn` (ligand).
  - `compute_pre_refolding_metrics`, `pre_refolding.{bioinformatics,tmol}` — optional pre-refold interface metrics. Those are the only two sub-toggles: `evaluate.py:401-403` reads `bioinformatics` and `tmol` and nothing else, and `src/proteinfoundation/rewards/` ships only `alphafold2_reward.py`, `base_reward.py`, `bioinformatics_reward.py`, `rf3_reward.py`, `tmol_reward.py`. There is no `hbplus` reward model or config key anywhere in the repo.
  - `compute_refolded_structure_metrics`, `refolded.{bioinformatics,tmol}` — optional post-refold, same two sub-toggles.
  - `compute_monomer_metrics`, `compute_designability`, `compute_codesignability`, `designability_modes`, `codesignability_modes` — monomer-on-binder eval.
  - `compute_co_sequence_recovery`, `compute_ss` — sequence/secondary-structure metrics.
  - `compute_novelty_{pdb,afdb,afdb_rep_v4}` — FoldSeek novelty against known DBs.
  - `keep_folding_outputs` — keep refolded PDBs.
- File walk control:
  - `ignore_generated_pdb_suffix: "_binder.pdb"` (default) — drop intermediate binder-only PDBs from the walk.
  - `file_limit` — **not a field of this config.** It ships in the AME/motif variants only (`evaluate_ame_from_pdb_dir.yaml:41`, `evaluate_motif_from_pdb_dir.yaml:51`). It is still usable here because `evaluate.py:793` reads it with `cfg.get("file_limit", None)`, so `++file_limit=N` works (`++` creates the key).
- `result_type` is set inline (`ligand_binder` or `protein_binder`) and propagates to the paired analyze step.

### `evaluate_ame_from_pdb_dir.yaml`

- `protein_type: motif_binder`.
- `defaults: - /design_tasks/ame_dict_v2@dataset` — resolves AME tasks (`M0024_1nzy`, `M0096_1chm`, etc.).
- Required `dataset.task_name` — must match a key in `ame_dict_v2`.
- Key `metric.*`:
  - `compute_motif_binder_metrics: True`.
  - `binder_folding_method: rf3_latest` (only RF3 makes sense for ligand motif binders).
  - `inverse_folding_model: ligand_mpnn`.
  - `sequence_types: [mpnn_fixed, self]` (default — `mpnn_fixed` keeps the motif residues constant).
  - `interface_cutoff: 6.0`.
  - `compute_binder_metrics`, `compute_monomer_metrics` — optional add-ons; `compute_motif_metrics` is incompatible with `motif_binder`.
- `result_type: motif_ligand_binder`.

### Shared reference configs (no `_from_pdb_dir` suffix)

These ship for completeness; the `from_pdb_dir` variants are derived from them with `input_mode=pdb_dir` baked in.

- `configs/evaluate.yaml` — unified binder evaluation, defaults `input_mode: generated`. Pass `++input_mode=pdb_dir ++sample_storage_path=<dir>` to evaluate an external directory.
- `configs/example/evaluate_motif_binder.yaml` — motif binder (protein + ligand variants). Note the path: it ships under `configs/example/`, not `configs/` (`ls configs/*.yaml` has no `evaluate_motif_binder.yaml`, and the citations in `docs/INFERENCE.md:104` and `configs/analyze_motif_binder.yaml:3` are stale). Set `++result_type=motif_protein_binder` (with AF2/ColabDesign + ProteinMPNN/SolubleMPNN) or rely on the default `motif_ligand_binder` (RF3 + LigandMPNN). Its own shipped defaults are `binder_folding_method: rf3_latest` (`:75`), `inverse_folding_model: ligand_mpnn` (`:79`), `sequence_types: [mpnn_fixed, self]` (`:82`), `num_redesign_seqs: 1` (`:85`), `interface_cutoff: 6.0` (`:88`), and it composes `- /design_tasks/ame_dict_v2@dataset` (`:24`) with `dataset.task_name: M0024_1nzy`. Use this when you have motif-protein-binder PDBs — there is no dedicated `_from_pdb_dir` variant, so add `++input_mode=pdb_dir ++sample_storage_path=<dir>`.

## 3. Analyze config schema

### `analyze.yaml`

- `result_type: protein_binder` (default) or `ligand_binder` (override).
- `aggregation.analysis_modes` — default `[binder, monomer]` for binder result types.
- `aggregation.success_thresholds` — `null` (defaults) or a dict of `{metric: {threshold, op, scale, column_prefix}}` entries.
- Built-in defaults:
  - `protein_binder`: `i_pAE * 31 <= 7.0`, `pLDDT >= 0.9`, `scRMSD_ca < 1.5`.
  - `ligand_binder`: `min_ipAE * 31 < 2.0`, `scRMSD_ca < 2.0`, `ligand_scRMSD_aligned_allatom < 5.0`.
- Monomer-mode thresholds (when `[monomer]` is in `analysis_modes`):
  - `aggregation.designability_thresholds` — `{mode: {model: {threshold, op}}}`; default 2.0 Å.
  - `aggregation.ca_codesignability_thresholds` — default 2.0 Å.
  - `aggregation.allatom_codesignability_thresholds` — default 2.0 Å.
  - `aggregation.require_all_thresholds: false` — `false` = OR across modes/models, `true` = AND.

### `analyze_motif_binder.yaml`

- `result_type: motif_protein_binder` (default in the YAML) or `motif_ligand_binder` (override on CLI).
- `aggregation.analysis_modes` — for `motif_protein_binder` / `motif_ligand_binder` the code default is **`["motif_binder"]` only** (`analyze.py:3065-3075`); `[binder, monomer]` is the default for `protein_binder` / `ligand_binder`. The `[motif_binder, binder, monomer]` claim in this file's own header comment and in `configs/analyze_motif_binder.yaml:12` is stale. Add `binder` / `monomer` explicitly if you want them.
- `aggregation.motif_binder_success_thresholds`:
  - **`motif_protein_binder` defaults** — binder: `i_pAE*31 <= 7.0`, `pLDDT >= 0.8`, `scRMSD_ca < 2.0`; motif: `motif_rmsd_pred_all < 2.0`, `correct_motif_sequence_all >= 1.0`.
  - **`motif_ligand_binder` defaults** — binder: `scRMSD_bb3 <= 2.0`; motif: `motif_rmsd_pred_all <= 1.5`, `correct_motif_sequence_all >= 1.0`, `has_ligand_clashes_all < 0.5`.
- A sample is successful when **at least one redesign index** passes **every** binder criterion AND **every** motif criterion jointly. Per-redesign joint evaluation, not pooled.
- `aggregation.success_thresholds` (binder-only mode) and the three monomer threshold dicts work identically to `analyze.yaml`.

## 4. `motif_protein_binder` vs `motif_ligand_binder`

Both come from the `configs/example/evaluate_motif_binder.yaml` family + `analyze_motif_binder.yaml`. The split is target-driven:

| Property                    | `motif_protein_binder`           | `motif_ligand_binder` (AME)        |
|-----------------------------|----------------------------------|------------------------------------|
| Target                      | Protein receptor                 | Small molecule ligand              |
| Folding (`binder_folding_method`) | `colabdesign` (AF2) or `rf3_latest` | `rf3_latest`                  |
| Inverse folding             | `protein_mpnn` / `soluble_mpnn`  | `ligand_mpnn`                      |
| Motif criteria              | RMSD + seq recovery              | RMSD + seq recovery + ligand clashes |
| Default binder threshold    | `i_pAE*31 <= 7.0`, `pLDDT >= 0.8`, `scRMSD_ca < 2.0` | `scRMSD_bb3 <= 2.0` |
| Default motif threshold     | `motif_rmsd_pred_all < 2.0`, `correct_motif_sequence_all >= 1.0` | `motif_rmsd_pred_all <= 1.5`, seq recovery, `has_ligand_clashes_all < 0.5` |
| `evaluate_*_from_pdb_dir`   | none — use `configs/example/evaluate_motif_binder.yaml` + `++input_mode=pdb_dir` | `evaluate_ame_from_pdb_dir.yaml` |

When in doubt: if every PDB has a small-molecule ligand chain, it is `motif_ligand_binder`. If the second chain is another protein receptor, it is `motif_protein_binder`.

## 5. Ligand `L:0` rename for AME RF3 evaluation

Lifted from `README.md` "Evaluating AME Designs with Ligand Targets (RF3)":

> When running RF3 evaluation on AME-generated PDB files that contain a ligand (small molecule on chain A), RF3 will attempt to add missing atoms based on the ligand's CCD code. This can cause shape errors in downstream RMSD calculations and provide the incorrect structure.
>
> **Solution:** rename the ligand residue to `L:0` before passing to RF3. This tells RF3 to treat it as a generic ligand and skip atom completion.

```python
from atomworks.io import load_any, to_pdb_file

# The [0] is required: load_any returns an AtomArrayStack. Without it the mask
# assignment does not touch a single model's res_name array and to_pdb_file gets
# the wrong type. Matches README.md:326-335.
atom_array = load_any("my_design.pdb")[0]

ligand_mask = atom_array.chain_id == "A"
atom_array.res_name[ligand_mask] = "L:0"
to_pdb_file(atom_array, "my_design_rf3_ready.pdb")
```

Apply this transform once over the whole input directory before invoking
`complexa analysis configs/evaluate_ame_from_pdb_dir.yaml ...`.

**Do not assume the rename has already happened.** `protein_type` is defined only in *evaluate*
configs (`binder_evaluate.yaml:9`, `ame_evaluate.yaml:8`, `evaluate_from_pdb_dir.yaml:32`,
`evaluate_ame_from_pdb_dir.yaml:33`, `example/evaluate_motif_binder.yaml:38`, …); no generate
config defines it, so there is no such thing as running `complexa generate` "with
`protein_type=motif_binder`". `README.md:319-320` says the reverse of the old guarantee: the
bundled AME targets under `assets/target_data/ame_input_structures/` are *not* all prepared this
way (`M0024_1nzy_v3` is the prepared example). Check `res_name` on chain A yourself before
running RF3.

## 6. Worked examples

> Examples A and C invoke `configs/evaluate_from_pdb_dir.yaml`, whose `defaults` entry
> `generation/targets_dict@dataset` (`:22`) does not resolve — see §2. Against the pristine
> checkout both die in Hydra composition before any GPU work. Run them against a fixed copy of
> the config (defaults entry `- /targets/targets_dict@dataset` for protein targets,
> `- /targets/ligand_targets_dict@dataset` for ligand targets), or use `configs/evaluate.yaml`
> with `++input_mode=pdb_dir` for the protein-binder case.

### Example A — protein binder PDB directory, AF2 refold

User: "Re-fold these 200 PDL1 binders with AlphaFold2 and tell me what fraction passes the default AlphaProteo thresholds."

```bash
complexa analysis configs/evaluate_from_pdb_dir.yaml \
  ++sample_storage_path=/data/pdl1_designs \
  ++dataset.task_name=02_PDL1 \
  ++metric.binder_folding_method=colabdesign \
  ++metric.inverse_folding_model=soluble_mpnn \
  ++metric.sequence_types=[self,mpnn_fixed] \
  ++metric.num_redesign_seqs=8 \
  ++result_type=protein_binder \
  ++eval_njobs=2 \
  ++run_name=pdl1_pdb_dir_af2
```

Pass-rate filter (applied by analyze), evaluated separately for each `sequence_type` present in
the results. The command above requests `sequence_types=[self,mpnn_fixed]`, so the columns are
`self_*` and `mpnn_fixed_*` — there are no `mpnn_*` columns to filter on. Thresholding reads the
`_all` list columns that `build_column_name` constructs
(`binder_analysis_utils.py:160-171`: `f"{seq_type}_{column_prefix}_{metric_suffix}_all"`), so for
`mpnn_fixed` the criteria are:

`mpnn_fixed_complex_i_pAE_all * 31 <= 7.0 AND mpnn_fixed_complex_pLDDT_all >= 0.9 AND mpnn_fixed_binder_scRMSD_ca_all < 1.5`

### Example B — AME PDB directory, RF3 refold (motif ligand binder)

User: "Score this folder of AME 1nzy designs — joint binder + motif success please."

There is no `scripts/` directory in this repo (root: `assets`, `community_models`, `configs`,
`docs`, `env`, `licenses`, `script_utils`, `src`) and no rename helper anywhere. Do the one-time
rename inline with the snippet from §5:

```bash
python3 - <<'PY'
import glob, os
from atomworks.io import load_any, to_pdb_file

src_dir = "/data/ame_1nzy_designs"
out_dir = "/data/ame_1nzy_designs_rf3_ready"
os.makedirs(out_dir, exist_ok=True)

for path in sorted(glob.glob(os.path.join(src_dir, "*.pdb"))):
    atom_array = load_any(path)[0]          # [0] is required — load_any returns a stack
    atom_array.res_name[atom_array.chain_id == "A"] = "L:0"
    to_pdb_file(atom_array, os.path.join(out_dir, os.path.basename(path)))
PY
```

```bash
complexa analysis configs/evaluate_ame_from_pdb_dir.yaml \
  ++sample_storage_path=/data/ame_1nzy_designs_rf3_ready \
  ++dataset.task_name=M0024_1nzy \
  ++metric.binder_folding_method=rf3_latest \
  ++metric.inverse_folding_model=ligand_mpnn \
  ++metric.sequence_types=[mpnn_fixed,self] \
  ++metric.num_redesign_seqs=2 \
  ++result_type=motif_ligand_binder \
  ++run_name=ame_m0024_pdb_dir
```

Joint success criterion (from `analyze_motif_binder.yaml`, `motif_ligand_binder` defaults): at least one redesign index satisfies binder `scRMSD_bb3 <= 2.0` AND motif `motif_rmsd_pred_all <= 1.5` AND `correct_motif_sequence_all >= 1.0` AND `has_ligand_clashes_all < 0.5`.

### Example C — ligand binder PDB directory, RF3 refold

User: "Score this folder of 7V11 ligand-binder designs with RF3 and LigandMPNN."

This is the example that exposes both target-dict defects at once. `39_7V11_LIGAND` is a key in
`configs/targets/ligand_targets_dict.yaml:2`, **not** in `targets_dict.yaml`, and no shipped
evaluate config composes the ligand dict — `evaluate_from_pdb_dir.yaml:22` names a group that
does not exist and `evaluate.yaml:31` composes the protein dict. Nothing routes between the two
dicts by task name; `get_target_info` (`binder_eval_utils.py:238-252`) just looks the name up in
whatever was composed and raises otherwise. So run the command below against a copy of
`evaluate_from_pdb_dir.yaml` whose defaults entry reads `- /targets/ligand_targets_dict@dataset`.
The three `metric.*` / `result_type` overrides shown are already this config's shipped defaults
(`:72`, `:84`, `:139`) and are listed for explicitness.

```bash
complexa analysis configs/evaluate_from_pdb_dir_ligand.yaml \
  ++sample_storage_path=/data/7v11_designs \
  ++dataset.task_name=39_7V11_LIGAND \
  ++metric.binder_folding_method=rf3_latest \
  ++metric.inverse_folding_model=ligand_mpnn \
  ++metric.sequence_types=[self,mpnn_fixed] \
  ++metric.num_redesign_seqs=8 \
  ++result_type=ligand_binder \
  ++eval_njobs=2 \
  ++run_name=7v11_pdb_dir_rf3
```

Pass-rate filter (applied by analyze): `min_ipAE * 31 < 2.0 AND scRMSD_ca < 2.0 AND ligand_scRMSD_aligned_allatom < 5.0`. Override via `++aggregation.success_thresholds.min_ipAE.threshold=…` etc.

## 7. Pointers

- Per-metric output column names: `docs/EVALUATION_METRICS.md` "Result CSV Reference".
- Default thresholds and Python filter examples: `docs/EVALUATION_METRICS.md` "Success Criteria" and "Reading Results in Python".
- Hardware tables (VRAM per backend, wall-clock per N PDBs): `_shared/reference/hardware.md`.
