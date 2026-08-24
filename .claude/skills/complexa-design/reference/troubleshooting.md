# Troubleshooting Reference

Symptoms, causes, and fixes for `complexa design` failures. Sourced from
`docs/INFERENCE.md` "Troubleshooting", `docs/EVALUATION_METRICS.md`, and the
pipeline configs themselves.

## GPU OOM during generation

**Symptom:** `torch.cuda.OutOfMemoryError` during the `generate` step;
process dies before the first PDB is written.

**Cause:** Default `generation.dataloader.batch_size: 16` is tuned for 80 GB
A100/H100. On a 40 GB GPU the latent flow-matching forward pass overflows.
Beam-search amplifies this because `max_batch_size` inherits from
`batch_size` and multiple beams run concurrently.

**Fix:**

```bash
++generation.dataloader.batch_size=8
++gen_njobs=1
```

If still OOMing, drop to `batch_size=4` and `++generation.args.nsteps=200` to
shrink activation history. Reference: `docs/INFERENCE.md` "Memory Issues".

## GPU OOM during folding

**Symptom:** OOM during the `evaluate` step, typically after the AF2 or RF3
model is loaded.

**Cause:** Multiple eval jobs share the GPU; ColabDesign with
`num_recycles: 3` and `num_redesign_seqs > 2` keeps a large state tree.

**Fix:**

```bash
++eval_njobs=1
++metric.num_redesign_seqs=2
++metric.sequence_types=[self]      # skip mpnn / mpnn_fixed redesigns
```

For RF3 specifically, also lower `++metric.num_redesign_seqs=1` — RF3 carries a
heavier per-sample state than AF2.

## Missing `AF2_DIR` (colabdesign fails)

**Symptom:** Hydra `InterpolationKeyError: AF2_DIR` raised when the reward
model or the evaluator tries to resolve `${oc.env:AF2_DIR}`.

**Cause:** The protein binder pipeline's `af2folding` reward model and the
default `binder_folding_method: colabdesign` both read `AF2_DIR` from the
environment. If `.env` does not define it, Hydra interpolation fails.

**Fix:** Add `AF2_DIR=/path/to/af2_params` to `.env`. There is no cheaper
backend to fall back to: `metric.binder_folding_method` accepts only
`colabdesign` or a name containing `rf3` (`binder_eval.py:108-128`), so
`colabdesign` is the only AF2 path. (`esmfold` is valid only for the different
key `metric.monomer_folding_models`.) If the GPU is the problem rather than the
weights, lower `++generation.dataloader.batch_size`, `++eval_njobs`, or
`++metric.num_redesign_seqs`.

Run `complexa download --all` to fetch AF2 weights into the canonical
location. Reference: `docs/INFERENCE.md` "Missing Model Weights".

## Missing `RF3_CKPT_PATH` / `RF3_EXEC_PATH` (rf3_latest fails)

**Symptom:** `InterpolationKeyError: RF3_CKPT_PATH` or `RF3_EXEC_PATH` during
the **generate** step of the ligand binder pipeline; or RF3 failing to load at
refold time on ligand binder / AME.

**Cause:** Only the ligand binder pipeline interpolates these vars —
`ligand_binder_generate.yaml:82-83` bakes `${oc.env:RF3_CKPT_PATH}` and
`${oc.env:RF3_EXEC_PATH}` into its RF3 reward model, so an unset var is a hard
Hydra error there. AME does **not**: `ame_generate.yaml:85` is
`reward_model: null` with the RF3 block commented out (`:88-110`), and
`ame_evaluate.yaml` contains no `oc.env` at all. At refold time RF3 resolves via
`os.environ.get(...)` with a fallback under `DATA_PATH`
(`binder_eval.py:117-122`), so the evaluate stage cannot raise an
`InterpolationKeyError` — it fails later with a missing-file error instead. RF3
is not downloaded by `complexa download --complexa-*`; it ships with the
community model bundle.

**Fix:** Export both env vars (or set them in `.env`):

```bash
export RF3_CKPT_PATH=/path/to/rf3_latest.pt
export RF3_EXEC_PATH=/path/to/rf3
```

Run `complexa download --all` to install RF3 into the canonical location.
Reference: `docs/INFERENCE.md` "RF3 Environment Variables".

## Chain-ID mismatch between target PDB and `target_input`

**Symptom:** Target loads but `n_target_residues == 0` after dataloader runs;
or `KeyError: 'A'` raised by the target featurizer.

**Cause:** The `target_input` field in the targets dict (e.g. `A1-115`)
specifies chain A, but the PDB uses chain B (or unlabelled chains).

**Fix:** Inspect the target PDB:

```bash
grep '^ATOM' /path/to/target.pdb | head -1   # check chain ID at col 22
```

Then update the target entry in `configs/targets/targets_dict.yaml` so
`target_input` matches the actual chain. Or use the `complexa-target` skill to
re-add the target with the correct chain.

## Hotspot residue not in target PDB

**Symptom:** Warning / error from `TargetFeatures` that hotspot residue
`A45` does not exist; or hotspots silently dropped, leading to non-specific
binders.

**Cause:** `hotspot_residues: [A45, A67, A89]` references residues that the
PDB does not contain — usually because the PDB has been re-numbered or the
chain has been truncated.

**Fix:** Pull the actual residue numbers:

```bash
grep '^ATOM' /path/to/target.pdb | awk '{print $5, $6}' | sort -u
```

Update `hotspot_residues` to a subset of residues that exist in `target_input`.

## AME requires `USE_V2_COMPLEXA_ARCH: "True"`

**Symptom:** Shape mismatch or `KeyError` at model-load time when running the
AME pipeline.

**Cause:** AME uses the v2 architecture. The `search_ame_local_pipeline.yaml`
sets `env_vars.USE_V2_COMPLEXA_ARCH: "True"`, but a stray
`++env_vars.USE_V2_COMPLEXA_ARCH=False` override (or a stale shell export)
flips it back.

**Fix:** Do not override `USE_V2_COMPLEXA_ARCH`. If a shell export is set, unset
it:

```bash
unset USE_V2_COMPLEXA_ARCH
```

Reference: `configs/search_ame_local_pipeline.yaml` line 22.

## AME ligand residue name must be `L:0` for RF3

**Symptom:** RF3 reward model raises a residue-name parse error during AME
generation; or the ligand silently disappears from the refolded structure.

**Cause:** RF3 expects the ligand HETATM residue named `L` with sequence-number
`0` (canonical AME convention). PDBs downloaded from RCSB usually have
arbitrary ligand residue names (e.g. `OQO`, `FAD`, `ATP`).

**Fix:** There is no `atomworks` CLI for this. Rewrite the residue name with
`atomworks.io` in Python — note it is a single residue *name* string `"L:0"`,
not a resname plus a resnum, and `load_any` returns a list so the `[0]` index is
required (`README.md:326-335`):

```python
from atomworks.io import load_any, to_pdb_file

atom_array = load_any("my_design.pdb")[0]

# Select chain A (ligand) and rename residues
ligand_mask = atom_array.chain_id == "A"
atom_array.res_name[ligand_mask] = "L:0"

to_pdb_file(atom_array, "my_design_rf3_ready.pdb")
```

Update the AME task in `configs/design_tasks/ame_dict_v2.yaml` to point at the
renamed PDB. Reference: `README.md` "Evaluating AME Designs with Ligand Targets
(RF3)" and `assets/target_data/README.md`.

## Running one stage directly (debug, or SLURM array shards)

**Symptom:** you need to attach `ipdb`, run under `nsys`, skip the pipeline log
dir, or run exactly one generation shard without the CLI fanning out.

**Cause:** `complexa generate CONFIG` is a logged subprocess wrapper. It also
launches `gen_njobs` shards at once and pins each to GPU index `job_id`, which
is wrong under a scheduler that already allocated the GPU.

**Fix:** invoke the Hydra module directly — this is what the wrapper runs:

```bash
python -m proteinfoundation.generate \
    --config-path "$(realpath configs)" \
    --config-name search_binder_local_pipeline \
    ++run_name=debug_pdl1 ++generation.task_name=02_PDL1
```

Same pattern for `proteinfoundation.{filter,evaluate,analyze}`. Add
`++job_id=N` to run one shard of a `gen_njobs=M` split. Prefer
`complexa generate/filter/evaluate/analyze` for normal one-shot runs — you get
logging and job splitting for free. For the campaign form of this, see "Sizing
shards so resume is worth having" in
`docs/binder-target-setup/campaign-gating.md`.

## Override key not recognized

**Symptom:** Hydra raises `InterpolationKeyError`, `MissingMandatoryValue`,
or "Could not override 'X'" when launching the pipeline.

**Cause:** Prefix semantics. Bare `key=value` requires the key to already exist
in the merged config; `+key=value` **adds** the key and errors if it already
exists; `++key=value` adds-or-overrides and never errors.

**Fix:** Always use `++` for design pipeline overrides (the pipeline composes
multiple configs and key existence varies by stage). Re-check the key against
`reference/overrides.md`.

> **A typo'd `++` key is a silent no-op, not an error.** Because `++` creates
> missing keys, `++metric.num_redesign_seq=8` (missing `s`) runs happily with
> the default and warns about nothing. If an override appears to have no effect,
> check the resolved config in the stage log.

Validating first catches missing ckpts and env vars, but **not** override keys —
`validate.py` has no config-key validation, and `complexa validate` accepts no
Hydra overrides at all (its subparser has only `type`, `config`, `--target`;
`cli_runner.py:1326-1348`), so appending `++...` aborts with
`unrecognized arguments`:

```bash
complexa validate design <pipeline_config>
```

## Missing checkpoint reported by `complexa download --status`

**Symptom:** `complexa download --status` lists a Complexa or community model
as `Missing`, and the pipeline fails immediately at the load step.

**Cause:** The relevant `complexa download --complexa-<variant>` or
`complexa download --all` has not been run, or the download was interrupted.

**Fix:**

```bash
complexa download --status                     # see what is missing
complexa download --complexa                   # protein binder weights
complexa download --complexa-ligand            # ligand binder weights
complexa download --complexa-ame               # AME / motif weights
complexa download --all                        # community models (AF2, RF3, MPNN, ESM2)
```

Hand off to the `complexa-setup` skill if the `.env` is also incomplete.

## ColabDesign env var missing

**Symptom:** ColabDesign fails on import or at AF2-params load with
`FileNotFoundError`.

**Cause:** `AF2_DIR` is set to a directory that does not contain
`params_model_*.npz` files, or the path is correct but does not exist.

**Fix:**

```bash
ls $AF2_DIR | grep '^params_model'
```

If empty, re-download:

```bash
complexa download --all
```

Or point `AF2_DIR` at the correct directory in `.env`.

## ProteinMPNN / LigandMPNN model directory missing

**Symptom:** Evaluation fails at the inverse-folding step with
`FileNotFoundError` on the MPNN weights directory.

**Cause:** The inverse-folding model is part of the community bundle, not the
Complexa bundle. `complexa download --complexa*` does not fetch it.

**Fix:**

```bash
complexa download --all
```

Or fetch the specific MPNN you need (LigandMPNN for ligand / AME pipelines,
SolubleMPNN for protein binder, ProteinMPNN for the monomer designability
calculation). All three live under the community-model directory.

## Inverse folder returns 0 sequences

**Symptom:** Per-design CSV has empty `{seq}_sequence` columns; success rate
is 0% even though designs visually look reasonable.

**Cause:** The target is too short, the interface cutoff is too tight, or the
binder length is constrained so heavily that the MPNN sampler cannot find a
sequence consistent with the fixed positions.

**Fix:** Loosen the interface cutoff and re-run:

```bash
++metric.interface_cutoff=10.0
++metric.num_redesign_seqs=8
```

If the binder is shorter than ~30 residues, also expand
`generation.dataloader.dataset.nres.low/high` — the MPNN models are unreliable
at very short lengths.

## 0 designs pass success thresholds

**Symptom:** Pipeline completes and the `_res_{seq_type}_pass_rate_{filter_name}_{suffix}`
columns in `filter_results/res_filter_*_pass_*.csv` are all `0.0` (there is no
column literally named `pass_rate`). Per-design metrics look plausible but
nothing passes.

**Cause:** Default thresholds are tuned for the published targets (PDL1,
TrkA, etc.) and may be too strict for harder targets or smaller search
budgets.

**Fix:** Loosen the thresholds — but you must supply the **complete**
`success_thresholds` dict. `binder_analysis.py:317-318` uses
`DEFAULT_PROTEIN_BINDER_THRESHOLDS` only when the key is entirely absent, so a
per-metric override such as
`++aggregation.success_thresholds.i_pAE.threshold=10.0` replaces the whole dict:
it drops the `pLDDT` and `scRMSD_ca` criteria, and `parse_threshold_spec`
(`analysis_utils.py:124-129`) fills the missing `scale` with `1.0` so a
0-1-scaled column is compared against `10.0`. Everything passes and the reported
success rate becomes 100% — the opposite failure, and much harder to notice.

Put the full dict in the analyze config (or a YAML overlay you compose in). Note
the key is `scRMSD_ca`, not `scRMSD`:

```yaml
aggregation:
  success_thresholds:
    i_pAE:     {threshold: 10.0, op: "<=", scale: 31.0, column_prefix: complex}
    pLDDT:     {threshold: 0.8,  op: ">=", scale: 1.0,  column_prefix: complex}
    scRMSD_ca: {threshold: 2.0,  op: "<",  scale: 1.0,  column_prefix: binder}
```

Then re-run analyze alone — no need to re-generate:

```bash
complexa analyze configs/search_binder_local_pipeline.yaml ++run_name=<same>
```

Reference: `docs/EVALUATION_METRICS.md` "Customizing Binder Thresholds".
