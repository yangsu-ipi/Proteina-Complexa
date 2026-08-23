# Sweep Axes Reference

Catalogue of swept keys for Proteina-Complexa design pipelines, grouped by pipeline stage. Every key is a Hydra dot-path you can put in a `configs/sweeps/*.yaml` file as a sweep axis (list value) or as an `--override KEY=VAL` pin.

Defaults are read from `configs/pipeline/binder/binder_generate.yaml`, `configs/pipeline/binder/binder_evaluate.yaml`, and `configs/pipeline/model_sampling.yaml`. Verify against your actual base pipeline config before launching — different pipelines (e.g. `search_ligand_binder_local_pipeline`, `search_ame_local_pipeline`) inherit different defaults.

## Generation axes

These live under `generation.*` in the pipeline config (because the base pipeline pulls `binder_generate@generation`).

### Search algorithm + width

| Key | Default | Typical sweep values | Cost multiplier | Effect |
|---|---|---|---|---|
| `generation.search.algorithm` | `best-of-n` | `single-pass`, `best-of-n`, `beam-search`, `fk-steering`, `mcts` | 1× → 8× | Search regime. `single-pass` = baseline. `best-of-n` linear in replicas. `beam-search`/`fk-steering` ~ `n_branch × beam_width` extra forward passes. |
| `generation.search.beam_search.beam_width` | 4 | 1, 2, 4, 8, 16 | linear | More beams = wider search → higher success rate. Diminishing returns past 8 in practice. |
| `generation.search.beam_search.n_branch` | 4 | 2, 4, 8 | linear | Branching factor per beam step. Together with `beam_width` controls total samples per checkpoint. |
| `generation.search.beam_search.keep_lookahead_samples` | `true` | `true`/`false` | minor | Whether to keep intermediate lookahead candidates as outputs. Off = fewer final PDBs. |
| `generation.search.best_of_n.replicas` | 2 (`binder_generate.yaml:56-57`) | 1, 4, 16, 64 | linear | Best-of-N draws. Increasing tightens success-rate estimate; not search depth. |
| `generation.search.fk_steering.beam_width` | 4 | 1, 2, 4, 8 | linear | FK-steering equivalent of beam_width. |
| `generation.search.fk_steering.n_branch` | 4 | 2, 4, 8 | linear | FK-steering branching factor. |
| `generation.search.fk_steering.temperature` | 0.1 | 0.05, 0.1, 0.2, 0.5 | none | Softmax temperature for FK resampling. Higher = closer to uniform (more exploration). |
| `generation.search.mcts.n_simulations` | 20 | 10, 20, 50 | linear | MCTS rollouts per decision. |
| `generation.search.mcts.exploration_constant` | 1.0 | 0.5, 1.0, 2.0 | none | UCB exploration weight. |
| `generation.search.reward_threshold` | `null` | `null`, -0.2, -0.1 | filters | Drop search candidates below this reward at each step. `null` = keep all. |

### Diffusion / flow sampling

| Key | Default | Typical sweep values | Cost multiplier | Effect |
|---|---|---|---|---|
| `generation.args.nsteps` | 400 | 100, 200, 400, 800 | linear | ODE integration steps. Lower = faster, lossy past ~100. 400 is a strong default. |
| `generation.args.self_cond` | `true` | `true`, `false` | ~1.05× when true | Self-conditioning across steps; usually helps. |
| `generation.args.guidance_w` | 1.0 | 0.0, 0.5, 1.0, 2.0 | none | Classifier-free guidance scale; 0 = unconditional, 1 = nominal, >1 = sharper conditioning. |
| `generation.args.fold_cond` | `false` | `true`, `false` | minor | Fold-conditioning toggle. Off for most binder runs. |
| `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | 0.1 | 0.05, 0.1, 0.2, 0.4, 0.8 | none | Backbone CA noise scale ("temperature"). Higher = more diverse, less ordered structures. The `bb_ca_temperature.yaml` canned sweep tests 0.1 vs 0.4. |
| `generation.model.bb_ca.simulation_step_params.sc_scale_score` | 1.0 | 0.5, 1.0, 1.5 | none | Score multiplier. Rarely swept. |
| `generation.model.bb_ca.simulation_step_params.t_lim_ode` | 0.98 | 0.9, 0.95, 0.98 | none | ODE cutoff time near t=1. |
| `generation.model.local_latents.simulation_step_params.sc_scale_noise` | 0.1 | 0.05, 0.1, 0.2 | none | Latent-feature noise. Affects sequence/local-structure diversity. |
| `generation.n_recycle` | 0 | 0, 1, 2 | linear in N+1 | Recycling iterations through the model. Costly. |

### Reward weights (`af2folding` composite)

Path prefix: `generation.reward_model.reward_models.af2folding.reward_weights.*`. These compose into a single scalar reward at each search step; tuning the ratio rebalances what search optimises for.

| Key (under the prefix above) | Default | Typical sweep | Effect |
|---|---|---|---|
| `i_pae` | -1.0 | -2.0, -1.0, -0.5, 0.0 | Negative weight on interface PAE. More negative = search pushes harder for low iPAE. |
| `plddt` | 0.0 | 0.0, 0.5, 1.0 | Positive weight on overall pLDDT. Turn on to bias toward confident monomers. |
| `con` | 0.0 | 0.0, 0.1, 0.5 | Binder-internal contact loss. Higher = more compact binders. |
| `dgram_cce` | 0.0 | 0.0, 0.1, 1.0 | AF2 distogram cross-entropy. Rarely swept above 0. |
| `min_ipae` | 0.0 | 0.0, -0.5, -1.0 | Min iPAE across chains; aggressive interface-quality push. |

> **The bioinformatics reward axes do nothing until you edit the pipeline config.** The entire
> `bioinformatics:` block under `reward_model.reward_models` is **commented out** in
> `configs/pipeline/binder/binder_generate.yaml:165-188` (as are `tmol:` at `:154-163`, `rf3folding:` at `:194-215` and `boltz2folding:` at `:217-235`), so
> the key is absent from the composed config. `generate_inference_configs.py` merges sweep values
> with OmegaConf struct mode off (`:309`, re-enabled only at `:328`), so
> `generation.reward_model.reward_models.bioinformatics.reward_weights.interface_sc: [0.5, 1.0]`
> is silently *created* as a bare node with no `_target_` instead of tuning a real reward model —
> the axis does not do what it looks like it does. **Uncomment the `bioinformatics:` block in
> `binder_generate.yaml` first**, then sweep its `reward_weights.*` /
> `reward_thresholds.*` leaves.

Once enabled, `generation.reward_model.reward_models.bioinformatics.reward_weights.{interface_sc, interface_hydrophobicity, surface_hydrophobicity, interface_dSASA, interface_fraction, interface_nres}` are the shape-complementarity / SASA / hydrophobicity weights, with the commented-out defaults at `binder_generate.yaml:167-173`. Same pattern: list of floats per axis. The `af2folding` block above it **is** active (`:131-152`), so those axes work as documented.

### Refinement + filtering

| Key | Default | Typical sweep | Effect |
|---|---|---|---|
| `generation.refinement.algorithm` | `null` | `null`, `sequence_hallucination` | Enable post-generation refinement loop. ~2× wall-clock when on. |
| `generation.filter.reward_threshold` | `null` | `null`, -0.2, -0.1 | Hard floor on reward before top-N filtering. |
| `generation.filter.filter_samples_limit` | 1000 | 100, 500, 1000 | Max samples to keep. |
| `generation.filter.dedup_sequence` | `true` | `true`, `false` | Drop sequence-duplicate samples before ranking. |

## Evaluation axes

These live at top level (the binder_evaluate config is loaded with `@_global_`), not under `generation.*`.

| Key | Default | Typical sweep | Cost multiplier | Effect |
|---|---|---|---|---|
| `metric.binder_folding_method` | `colabdesign` | `colabdesign`, `rf3_latest` — **and nothing else** | varies | Which refolder validates the binder. AF2 (`colabdesign`) is the standard; RF3 (any name containing `rf3`) is required for ligand targets. `binder_eval.py:97-117` raises `ValueError: Folding model '<x>' not supported` for every other value, so an axis containing `esmfold`, `boltz2_default` or `protenix_*` crashes every config generated from it. `esmfold` belongs to the separate monomer key `metric.monomer_folding_models` (`monomer_eval_utils.py:30`). |
| `metric.num_redesign_seqs` | 2 | 1, 2, 4, 8, 16 | linear | Number of MPNN redesigns to refold per binder. Higher = more reliable designability signal. |
| `metric.sequence_types` | `[self]` | `[self]`, `[self, mpnn]`, `[self, mpnn_fixed]` | linear per type | Which sequences to evaluate: generated, MPNN-redesigned, or MPNN with fixed interface. |
| `metric.interface_cutoff` | 8.0 | 6.0, 8.0, 10.0 | none | Angstrom cutoff defining interface residues for MPNN_fixed and interface metrics. |
| `metric.inverse_folding_model` | `soluble_mpnn` | `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn` | none | MPNN variant used for redesign. |
| `metric.compute_pre_refolding_metrics` | `false` | `true`, `false` | minor | Compute interface metrics on the generated structure (no fold) — fast. |
| `metric.compute_refolded_structure_metrics` | `false` | `true`, `false` | minor | Compute interface metrics on the refolded structure — slower. |
| `metric.pre_refolding.{bioinformatics,tmol}` | both `false` (`binder_evaluate.yaml:43-45`) | `true`/`false` | minor | Toggle individual pre-refold interface metric modules. Only these two exist — `evaluate.py:401-403` reads `bioinformatics` and `tmol` and nothing else, and there is no `hbplus` module anywhere in `src/proteinfoundation/rewards/`. `metric.refolded.{bioinformatics,tmol}` are the post-refold twins and are also both `false` (`:51-53`). |

## Reading the sweeper YAML format

Annotated example based on `configs/sweeps/example.yaml`:

```yaml
# --- Sweep axes (lists, cartesian-producted) ---
# Each key is a Hydra dot-path. Each list value becomes one dimension.
# Total configs = product of list lengths. Here 2 × 2 = 4.
generation.search.beam_search.beam_width:
  - 2
  - 4
generation.args.nsteps:
  - 200
  - 400

# --- Pinned scalar (no extra dimension) ---
# Scalars are auto-wrapped into a single-element list, so they pin a value
# across the whole sweep without growing the cartesian product.
# generation.args.self_cond: true

# --- Long block-style list (e.g. checkpoint paths) ---
# ckpt_path:
#   - /lustre/checkpoints/model_v1/epoch_100.ckpt
#   - /lustre/checkpoints/model_v2/epoch_200.ckpt
```

Validation rules (`script_utils/generate_inference_configs.py:load_sweeper_file`, `:108-131`):

- Top-level YAML must be a mapping. List or scalar at top = `ValueError: Sweeper YAML must be a mapping`.
- All keys must be strings. Numeric keys = `ValueError: Sweeper keys must be strings`.
- List values are kept verbatim. Scalar values are wrapped to `[value]`.
- Empty list `[]` for an axis = **silent no-op, exit 0.** `itertools.product` over an empty axis yields no combinations, so `apply_sweeper_and_save_configs` writes nothing and `:450` logs `Generated 0 config pair(s)`. There is no "No configs were generated" error — that string does not exist in the repo. Check the `Generated N config pair(s)` line before launching.

## Generating sweeper YAMLs programmatically

For large parameter grids it is cleaner to render the YAML than to edit by hand:

```python
import itertools, yaml

axes = {
    "generation.search.beam_search.beam_width": [1, 2, 4, 8],
    "generation.args.nsteps": [200, 400, 800],
    "generation.model.bb_ca.simulation_step_params.sc_scale_noise": [0.1, 0.2, 0.4],
}
# 4 * 3 * 3 = 36 configs
total = 1
for v in axes.values():
    total *= len(v)
print(f"Will generate {total} configs")

with open("configs/sweeps/big_grid.yaml", "w") as f:
    yaml.safe_dump(axes, f, sort_keys=False, default_flow_style=False)
```

For an irregular set of `(key1, key2)` pairs (not a full cartesian product), there is no native support — emit one sweeper file per pair and concatenate the summary CSVs.

## Result-aggregation logic

`sweep_summary.csv` is a skill-side artifact — nothing in the repo emits it. It is assembled in
Step 6 of the skill from two real per-config outputs:

- `./evaluation_results/eval_{idx}_{run_name}_{pipeline_run_name}/RAW_{result_type}_results_{config_name}_combined.csv`
  (`analyze.py:3036`) — one row per generated sample.
- `.../filter_results/res_filter_binder_pass_*.csv` (`binder_analysis.py:548`, moved there by
  `organize_results`, `analyze.py:2802-2880`) — the pre-computed pass rates.

No emitted file is named `results_*.csv`; the per-job files are
`{monomer,binder,motif,motif_binder}_results_{config_name}_{job_id}.csv`
(`evaluate.py:853, :881, :904, :930`).

| Column | Source | Notes |
|---|---|---|
| `config_id` | The index in `inf_{idx}_{run_name}.yaml` | 0-based, sequential, set by `apply_sweeper_and_save_configs`. |
| `<axis_name>` (one per axis) | Read from the per-config `inf_*.yaml` at the swept Hydra path | Strip the dot-path to a short column header (e.g. `beam_width`). |
| `n_samples` | `len(RAW_..._combined.csv)` | One row per sample (`id_gen`), not per sequence type — sequence types are column prefixes. |
| `success_rate` | **Read it from `filter_results/res_filter_binder_pass_*.csv`.** | Recompute only if that file is absent, and then use the real defaults from `DEFAULT_PROTEIN_BINDER_THRESHOLDS` (`binder_analysis_utils.py:75-94`): `{seq}_complex_i_pAE_all * 31 <= 7.0` AND `{seq}_complex_pLDDT_all >= 0.9` AND `{seq}_binder_scRMSD_ca_all < 1.5`. There is **no** `passes_filter` column — repo-wide grep matches only this skill's files. |
| `mean_i_pae` | `{seq}_complex_i_pAE.mean()` | Lower = better. AF2 interface PAE, stored **0–1 scaled** — multiply by 31 to report in threshold units. A naive `i_pae < 10` test passes every sample. |
| `mean_plddt` | `{seq}_complex_pLDDT.mean()` | Higher = better. **Complex** pLDDT, 0–1. There is no interface-pLDDT column in the raw CSV — `i_plddt` does not exist. |
| `mean_binder_scRMSD_ca` | `{seq}_binder_scRMSD_ca.mean()` | Lower = better. Binder CA scRMSD in Å. `sc_rmsd` does not exist. |
| `diversity_score` | Unique `{seq}_sequence` count / `n_samples`, or the FoldSeek/MMseqs2 output under `diversity/` and `clusters/` | Higher = more diverse pool. `binder_seq` does not exist; the column is `{seq}_sequence` (`binder_eval.py:425`). |
| `wall_clock_min` | Timestamp delta from the per-stage logs under `./logs/` | Approximate (process wall-clock, not GPU time). With the split-stage loop from Step 4 there is one log per stage, not one per pipeline. |

`{seq}` is a `metric.sequence_types` value (`self`, `mpnn`, `mpnn_fixed`) used as a column
**prefix**; pick the one the sweep actually requested. The `_all` suffix marks the per-redesign
list columns that the threshold filter reads (`binder_analysis_utils.py:160-171`).

Ranking:

- **Best by success** = argmax `success_rate`; tie-break on `mean_i_pae` ascending.
- **Pareto frontier** on (`wall_clock_min`, `success_rate`): a config is on the frontier iff no other config has both lower wall-clock AND higher success rate. Implement with a sort + linear sweep.
- **Sanity check**: if every config has `success_rate == 0`, the threshold is too strict OR the sweep regime is broken — surface this to the user before reporting "best". Conversely, a 100% success rate across the board usually means a partial `aggregation.success_thresholds` override replaced the whole default dict (`binder_analysis.py:317-318`) and left an unscaled `i_pAE` comparison that everything passes.
