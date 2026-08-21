# Proteina-Complexa: Protein Binder Design — Onboarding Guide

A practical walkthrough for a new user going from a fresh clone to a first protein-binder
design campaign on a multi-GPU node. Everything here is grounded in this repo's actual
configs, CLI, and source — file paths and config keys are quoted verbatim so you can
`grep` for them.

Scope: **protein–protein binder design** (the `search_binder_local_pipeline.yaml` path).
Ligand-binder, AME/motif, and training paths are mentioned only where they clarify a
contrast.

> **Start here: the repo ships five skills in `.claude/skills/`.** If you drive this repo with
> Claude Code, `complexa-setup`, `complexa-target`, `complexa-design`, `complexa-evaluate-pdbs`,
> and `complexa-sweep` automate most of what follows, and `complexa-design/SKILL.md` is the
> single best document in the repo for binder design. This guide is the conceptual companion:
> read it to understand *why*, use the skills to do the work.
>
> They contained 129 verified defects when I audited them (see `SKILLS_AUDIT.md`); I patched
> `.claude/skills/` in place, so a fresh `git diff` shows exactly what changed. Two of those
> defects were serious enough that they're called out inline below, in **Part 5** (a folding
> backend that doesn't exist) and **Part 6** (a threshold override that silently reports 100%
> success).

---

## Part 1 — The mental model

### What the method actually does

The paper's argument is that de novo binder design has been split into two camps that
shouldn't be:

| Camp | How it works | Weakness |
|---|---|---|
| **Conditional generative** (RFdiffusion-style) | Sample a binder backbone conditioned on the target | Prior is good, but no way to push a *specific* sample toward success |
| **Hallucination** (BindCraft-style) | Optimize a sequence against a structure predictor's confidence | Strong optimization signal, but starts from no prior and drifts into adversarial designs |

Proteina-Complexa does both. It trains a **fully atomistic flow-matching generative prior**
over protein complexes (backbone geometry + side chains + sequence, jointly), then spends
**test-time compute** searching that prior's sample space against reward models — AF2
interface confidence, RF3, force-field terms. So the generative prior keeps you in the
manifold of realistic proteins, and the search gives you the hallucination-style
optimization pressure, without the drift.

Three things follow from this that shape how you use the code:

1. **Structure prediction is used twice, for different purposes.** During generation, AF2 is
   a *reward model* steering the search. During evaluation, AF2 (or RF3) is an *oracle*
   scoring the finished designs. Same tool, different role — don't confuse the numbers.
2. **Compute is a dial, not a constant.** The `search.algorithm` setting is where you spend
   GPU-hours to buy success rate. `single-pass` is the no-search baseline; `beam-search`
   with a wide beam is the expensive, highest-quality end.
3. **The model generates sequence too.** Unlike backbone-only generators, Complexa emits
   side chains and sequence. That's why `sequence_types: [self]` (evaluate the model's own
   sequence, no redesign) is a meaningful option, not just a debug mode.

### The architecture in one diagram

```
                  ┌──────────────────────────────────────────┐
   target PDB ───►│  Flow-matching generative prior          │
   + hotspots     │  (complexa.ckpt + complexa_ae.ckpt)      │
   + length range │  backbone + side chains + sequence       │
                  └───────────────┬──────────────────────────┘
                                  │  denoising trajectory, nsteps=400
                                  │
              step_checkpoints ───┼─── [0, 100, 200, 300, 400]
                                  │      at each: branch, score, prune
                                  ▼
                  ┌──────────────────────────────────────────┐
                  │  Search (test-time compute)              │
                  │  single-pass │ best-of-n │ beam-search   │
                  │  fk-steering │ mcts                      │
                  └───────────────┬──────────────────────────┘
                                  │  scored by ↓
                  ┌───────────────┴──────────────────────────┐
                  │  CompositeRewardModel                    │
                  │  af2folding (i_pae) [default]            │
                  │  + tmol (H-bonds)      [optional]        │
                  │  + bioinformatics (SC) [optional]        │
                  └───────────────┬──────────────────────────┘
                                  ▼
                            designs → filter → evaluate → analyze
```

### The four stages

Every pipeline in this repo — binder, ligand binder, AME, monomer motif — is the same four
stages. Learn them once.

| Stage | CLI | Config section | What it does |
|---|---|---|---|
| 1. Generate | `complexa generate` | `generation.*` | Flow-matching sampling, search, reward scoring |
| 2. Filter | `complexa filter` | `generation.filter.*` | Keep top-N by reward, dedup sequences |
| 3. Evaluate | `complexa evaluate` | `metric.*` | Redesign sequence (MPNN), refold (AF2/RF3), compute metrics |
| 4. Analyze | `complexa analyze` | `aggregation.*` | Apply success thresholds, pass rates, diversity clustering |

`complexa design <config>` runs all four in order. Run stages individually when you want to
re-evaluate existing samples with different metrics without regenerating.

---

## Part 1b — The bundled skills

Five project-local Claude Code skills live in `.claude/skills/`. They are the intended
interface to this repo, and each one owns a stage of the workflow:

| Skill | Owns | Reach for it when |
|---|---|---|
| `complexa-setup` | `.env`, weight downloads, preflight | Fresh clone; "what do I have installed?" |
| `complexa-target` | `targets_dict.yaml` entries | Registering a new protein or ligand target |
| `complexa-design` | The 4-stage `complexa design` run | Any actual design campaign |
| `complexa-evaluate-pdbs` | `complexa analysis` on an existing PDB dir | Scoring third-party designs (BindCraft, RFdiffusion, decoys) |
| `complexa-sweep` | `script_utils/generate_inference_configs.py` + a design loop | Tuning beam_width, nsteps, reward weights |

Shared infrastructure: `_shared/scripts/preflight.sh` (GPU/disk/ckpt/tool probe → writes
`./preflight.json`), `_shared/scripts/write_manifest.py` (replayable run manifest), and
`_shared/reference/hardware.md`.

Two practical notes. First, these are **Claude Code** skills scoped to this repo — they
auto-load when you run `claude` from the repo root, but not in other Claude surfaces, where
you can still read them as plain files. Second, `complexa-design/SKILL.md` carries a
pipeline-switching cheat sheet that's worth internalizing even if you only ever do protein
binders, because it makes clear how little changes between the three pipelines:

| Knob | Protein binder (default) | Ligand binder | AME (enzyme) |
|---|---|---|---|
| Pipeline YAML | `search_binder_local_pipeline.yaml` | `search_ligand_binder_local_pipeline.yaml` | `search_ame_local_pipeline.yaml` |
| Model / AE ckpt | `complexa.ckpt` / `complexa_ae.ckpt` | `complexa_ligand.ckpt` / `..._ae.ckpt` | `complexa_ame.ckpt` / `..._ae.ckpt` |
| Targets dict | `configs/targets/targets_dict.yaml` | `configs/targets/ligand_targets_dict.yaml` | `configs/design_tasks/ame_dict_v2.yaml` |
| Task-name pattern | `02_PDL1`, `33_TrkA` | `39_7V11_LIGAND` | `M0024_1nzy_v3` |
| LoRA | none | required (`r=32, alpha=64`) | required |
| `USE_V2_COMPLEXA_ARCH` | unset (v1) | unset (v1) | `"True"` |
| Default search | `best-of-n` | `best-of-n` | `single-pass` |
| Reward model | AF2 (`af2folding`) | RF3 (`rf3folding`) | `null` — none by default |
| Inverse folder | `soluble_mpnn` | `ligand_mpnn` | `ligand_mpnn` |
| Refold backend | `colabdesign` | `rf3_latest` | `rf3_latest` |
| `result_type` | `protein_binder` | `ligand_binder` | `motif_ligand_binder` |
| Download flags | `--complexa --all` | `--complexa-ligand --all` | `--complexa-ame --all` |

Switching pipelines is genuinely just "swap the config path and use a task name from a
different dict."

Rough wall-clock for ~100 designs at `nsteps=400`, `beam_width=8`, on one A100/H100
(empirical, from `complexa-design/SKILL.md` — not sourced from any measurement in the repo):
protein binder + ColabDesign ≈ 60–120 min; ligand binder + RF3 ≈ 90–180 min; AME + RF3 ≈
120–240 min. Divide by your GPU count via `gen_njobs`/`eval_njobs`.

---

## Part 2 — Setup on a multi-GPU node

Your hardware answer (multi-GPU node) matters mainly at the end: each generation job and
each evaluation job takes **one whole GPU**. A single design is never sharded across GPUs.
So `gen_njobs`/`eval_njobs` = your GPU count is the whole story for parallelism.

### Requirements check

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | 1× CUDA, ≥24 GB VRAM | A100-80 / H100 / L40S, 40–80 GB |
| CUDA | 12.0 | 12.4+ |
| Disk | 50 GB for checkpoints | 150 GB (covers `--everything`) |
| RAM | 16 GB | 64 GB+ |
| OS | **Ubuntu 22.04+** | 22.04+ or Docker |

> **Ubuntu 20.04 will fail** with GLIBC errors on the UV path. Use the Docker runtime there.

### Step 1 — Build the Python environment

```bash
cd /path/to/Proteina-Complexa

./env/build_uv_env.sh
source .venv/bin/activate
which complexa            # should resolve inside .venv/
```

The `complexa` CLI lives inside the venv, not on your system path. On a fresh clone
`.venv/` does not exist yet, and running `complexa init` before building it gives a
confusing `command not found` rather than a useful error.

> **Known snag:** `tmol` can fail to install on Python 3.12 (its `llvmlite`/`numba` pins).
> If the build dies at the tmol step, pre-install compatible versions before the tmol line
> in `build_uv_env.sh`:
> ```bash
> uv pip install "llvmlite>=0.41" "numba>=0.59"
> ```
> tmol is only needed for force-field rewards and H-bond interface metrics — both are
> **disabled by default** in `binder_generate.yaml` and `binder_evaluate.yaml`, so a failed
> tmol install does not block your first run.

Docker alternative, if you're on an older host:

```bash
docker build -t proteina-complexa -f env/docker/Dockerfile .
docker run --gpus all -it -v /path/to/PFM_data:/workspace/data proteina-complexa
```

### Step 2 — Create and edit `.env`

```bash
complexa init            # create .env from .env_example
complexa init uv         # ...and target the UV runtime
complexa init docker     # ...and target the Docker runtime
```

> `runtime` is a **positional** argument (`nargs="?"`, choices `uv`/`docker`), not a
> `--runtime` flag — the README and the bundled `complexa-setup` skill both show
> `--runtime docker`, which errors. Bare `complexa init` creates `.env` without selecting a
> runtime.

Then edit two lines by hand — there is no CLI for this, and everything else derives from
them:

```bash
LOCAL_CODE_PATH=/absolute/path/to/Proteina-Complexa
LOCAL_DATA_PATH=/absolute/path/to/PFM_data
```

`DATA_PATH` is where target PDBs live (`$DATA_PATH/target_data/<source>/<file>.pdb`). Note
that the bundled targets in `configs/targets/targets_dict.yaml` set an explicit
`target_path: ./assets/target_data/...`, which overrides the `$DATA_PATH` lookup — so the
shipped targets work even before you populate `PFM_data`.

Optional keys, only if you need them:

| Key | Needed for |
|---|---|
| `AF2_DIR` | **Required.** AF2 is the default generation reward and evaluation oracle. Already pre-populated in `.env_example` as `${COMMUNITY_MODELS_PATH}/ckpts/AF2` — just make sure the weights actually landed there |
| `RF3_CKPT_PATH`, `RF3_EXEC_PATH` | Only if you switch to `binder_folding_method=rf3_latest` |
| `ESM_DIR` / `CACHE_DIR` | ESM metrics (on by default in `binder_evaluate.yaml`) |
| `HF_TOKEN` | ESMFold / gated HuggingFace models |
| `SC_EXEC`, `DSSP_EXEC` | Shape complementarity / secondary structure — **not distributed here**, get prebuilt binaries from [FreeBindCraft](https://github.com/cytokineking/FreeBindCraft/tree/master/functions) |
| `FOLDSEEK_EXEC`, `MMSEQS_EXEC` | Diversity clustering in the analyze stage |

Missing tool binaries degrade evaluation but do not block generation.

A missing `AF2_DIR` fails **loudly at generation** — `binder_generate.yaml` interpolates
`${oc.env:AF2_DIR}` and Hydra raises `InterpolationKeyError` at config resolution, which is
deliberate. But **evaluation fails quietly**: `colabdesign_utils.py` reads it with
`os.getenv("AF2_DIR", f"{data_path}/tools/AF2")`, so an unset variable silently falls back
to a path that probably doesn't exist. Confirm AF2 weights are where you think they are.

### Step 3 — Download weights

Use the CLI. It dispatches to `env/download_startup.sh` (~1000 lines of NGC URLs, retries,
skip-if-present logic) — do not hand-roll a `wget` loop.

For protein binder design specifically:

```bash
complexa download --complexa      # complexa.ckpt + complexa_ae.ckpt   (~3 GB) → ./ckpts/
complexa download --all           # AF2, ESM2, ESMFold, ProteinMPNN,
                                  # LigandMPNN, RF3                     (~50 GB) → ./community_models/
complexa download --status        # verify what landed
```

The full menu, for reference:

| Flag | Contents | Size |
|---|---|---|
| `--complexa` | Protein binder model + AE | ~3 GB |
| `--complexa-ligand` | Ligand binder model + AE | ~3 GB |
| `--complexa-ame` | AME motif model + AE | ~3 GB |
| `--complexa-all` | All three variants | ~9 GB |
| `--all` | All community models | ~50 GB |
| `--everything` | Everything + Boltz2/Protenix | ~100+ GB |

### Step 4 — Point the config at your checkpoints

`configs/search_binder_local_pipeline.yaml` ships with relative defaults:

```yaml
ckpt_path: ./ckpts
ckpt_name: complexa.ckpt
autoencoder_ckpt_path: ./ckpts/complexa_ae.ckpt
```

If you downloaded to `./ckpts` from the repo root, you're done. Otherwise edit those three
lines, or override on the command line:

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++ckpt_path=/my/checkpoints \
    ++ckpt_name=complexa.ckpt \
    ++autoencoder_ckpt_path=/my/checkpoints/complexa_ae.ckpt
```

### Step 5 — Validate before you burn GPU hours

```bash
complexa validate env
complexa validate design configs/search_binder_local_pipeline.yaml
```

`validate env` checks `.env` exists and `DATA_PATH` resolves. `validate design` walks the
full Hydra defaults tree, checks checkpoint files, and resolves every `${oc.env:...}`
interpolation. This is the cheapest way to find a missing `AF2_DIR` — five seconds instead
of twenty minutes into a run.

---

## Part 3 — Understanding the config system

This is the part that trips people up. Complexa uses **Hydra**, and the pipeline config is
almost entirely composition.

```yaml
# configs/search_binder_local_pipeline.yaml
defaults:
  - pipeline/binder/binder_generate@generation   # → keys land under generation.*
  - pipeline/binder/binder_evaluate@_global_     # → keys land at root: metric.*, protein_type
  - pipeline/binder/binder_analyze@_global_      # → keys land at root: aggregation.*, result_type
  - _self_                                       # this file's own keys win
```

Read `@generation` as *"mount this whole file under the key `generation`"*. That's why every
generation override is `++generation.something` while evaluation overrides are plain
`++metric.something` — the evaluate fragment is mounted at `@_global_`, i.e. the root.

The full tree for binder design:

```
configs/search_binder_local_pipeline.yaml
├── configs/pipeline/binder/binder_generate.yaml   → generation.*
│   ├── configs/targets/targets_dict.yaml          → target_dict_cfg.*
│   ├── configs/pipeline/base_gen_data.yaml        → dataloader defaults
│   └── configs/pipeline/model_sampling.yaml       → generation.args.*, generation.model.*
├── configs/pipeline/binder/binder_evaluate.yaml   → metric.*, protein_type
└── configs/pipeline/binder/binder_analyze.yaml    → aggregation.*, result_type
```

Override anything with `++key=value`. Lists need quoting: `++metric.sequence_types="[self,mpnn]"`.

---

## Part 4 — Your first run

### Smoke test first

Before a real campaign, run something tiny to confirm the whole chain works end to end.

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=smoke_test \
    ++generation.task_name=02_PDL1 \
    ++generation.args.nsteps=100 \
    ++generation.dataloader.dataset.nres.nsamples=2 \
    --verbose
```

`--verbose` streams output to your terminal instead of a log file — you want that the first
time.

This uses PD-L1, the canonical benchmark target. Its entry in `configs/targets/targets_dict.yaml`:

```yaml
  02_PDL1:
    source: bindcraft_targets
    target_filename: PD-L1
    target_path: ./assets/target_data/bindcraft_targets/PD-L1.pdb
    target_input: A1-115                          # chain A, residues 1-115
    hotspot_residues: ["A37", "A39", "A49", "A98"] # where the binder should contact
    binder_length: [64, 155]                       # sampled uniformly in this range
    pdb_id: null
```

### A real campaign on a multi-GPU node

```bash
complexa design configs/search_binder_local_pipeline.yaml \
    ++run_name=pdl1_beam_v1 \
    ++generation.task_name=02_PDL1 \
    ++gen_njobs=8 ++eval_njobs=8 \
    ++generation.search.algorithm=beam-search \
    ++generation.search.beam_search.beam_width=8 \
    ++generation.search.beam_search.n_branch=4 \
    ++generation.dataloader.dataset.nres.nsamples=16 \
    ++metric.sequence_types="[self,mpnn]" \
    ++metric.num_redesign_seqs=8
```

Substitute your actual GPU count for `8`. Keep `eval_njobs == gen_njobs` so each evaluation
job consumes exactly one generation job's output shard.

### Checking on it

```bash
complexa status           # summarizes ./inference and ./evaluation_results
complexa status --logs    # plus recent log files
```

> The README shows `complexa status <config>`. The argument parser
> (`cli_runner.py`) defines no positional for `status` — only `--logs`. Call it bare.

Outputs land in:

```
./inference/{config_name}_{task_name}[_{run_name}]/
    e.g. ./inference/search_binder_local_02_PDL1_pdl1_beam_v1/
    ├── job_0_n_97_id_0_<tag>/          # one directory per sample, directly at the root
    │   └── job_0_n_97_id_0_<tag>_binder.pdb
    ├── job_0_n_112_id_1_<tag>/
    ├── filtered_out_samples/           # samples the filter stage rejected
    └── *.csv                           # per-sample reward scores
./evaluation_results/{config_name}_{task_name}[_{run_name}]/   # evaluate + analyze output
./logs/hydra_outputs/{date}/{time}/     # resolved configs — "what did I actually run?"
```

> **Two README/demo-text errors worth knowing.** There is no timestamp in the run directory
> name — the timestamp line in `generate.py` is commented out. And there are no `samples/`
> or `filtered/` subdirectories: sample directories sit directly under the run root, and the
> filter stage only ever *removes* things into `filtered_out_samples/`. `complexa demo`
> advertises `./inference/.../filtered/`; it doesn't exist.
>
> Note also that the **run directory name has no `_binder` suffix but the PDB inside it
> does** — directory stem ≠ filename stem. `docs/SEARCH_METADATA.md` documents the filename
> as `job_{id}_n_{len}_id_{idx}_{tag}.pdb`, omitting the `_binder`.

The 44 bundled protein targets are listed with `complexa target list`, or read
`configs/targets/targets_dict.yaml` directly. Useful ones beyond PD-L1: `33_TrkA` (the
config's own default), `38_TNFalpha`, `36_VEGFA`, `37_IL17A`, `30_SC2RBD`, `24_SpCas9`.

---

## Part 5 — The knobs that matter

### Search algorithm — where you spend compute

Set with `++generation.search.algorithm=...`. **Names are hyphenated**, per
`src/proteinfoundation/search/search_factory.py`:

| Algorithm | Parameters | Cost vs single-pass | When |
|---|---|---|---|
| `single-pass` | — | 1× | Baseline, debugging, "does this target work at all" |
| `best-of-n` *(default)* | `best_of_n.replicas: 2` | N× | Simple, embarrassingly parallel, good default |
| `beam-search` | `beam_width: 4`, `n_branch: 4` | ~W·B× | Highest quality; the paper's production setting |
| `fk-steering` | `n_branch`, `beam_width`, `temperature: 0.1` | ~2N× | Softer selection, more exploration |
| `mcts` | `n_simulations: 20`, `exploration_prob: 0.5` | ≥W× | Exploration-heavy campaigns |

> ⚠️ `.claude/skills/_shared/reference/hardware.md` lists these with **underscores**
> (`best_of_n`, `beam_search`). That's wrong — the factory only accepts hyphens and raises
> `ValueError: Unknown search algorithm` otherwise. The *sub-config blocks* use underscores
> (`search.beam_search.beam_width`); only the `algorithm:` string is hyphenated.

Search happens at `step_checkpoints: [0, 100, 200, 300, 400]` — points in the 400-step
denoising trajectory where the partially-denoised samples are decoded, scored by the reward
model, and pruned.

**Output volume scales fast.** With `keep_lookahead_samples: true` (the default), for
beam-search:

```
Total PDBs = nsamples × beam_width × (n_branch × (len(step_checkpoints) - 1) + 1)
```

nsamples=4, W=4, B=4, 5 checkpoints → 4×4×(4×4+1) = **272 PDBs** from 4 nominal designs.
Set `++generation.search.beam_search.keep_lookahead_samples=false` to keep only the
`nsamples × beam_width` finals if disk is a concern.

### Reward models — what the search optimizes

Default for protein binders is AF2-Multimer interface PAE only:

```yaml
reward_model:
  _target_: proteinfoundation.rewards.base_reward.CompositeRewardModel
  reward_models:
    af2folding:
      _target_: proteinfoundation.rewards.alphafold2_reward.AF2RewardModel
      protocol: binder
      use_multimer: True
      num_recycles: 3
      use_initial_guess: True
      reward_weights:
        i_pae: -1.0      # ← the only non-zero weight by default
        con: 0.0
        plddt: 0.0
        # ... all other terms 0.0
```

Sign convention: **negative weight where lower is better**. `i_pae: -1.0` means "minimize
interface PAE". Total reward is `sum(model_weight × sub_model_total_reward)`.

Change weights from the CLI:

```bash
++generation.reward_model.reward_models.af2folding.reward_weights.i_pae=-2.0
++generation.reward_model.reward_models.af2folding.reward_weights.plddt=0.5
```

Two more reward models are **commented out** in `binder_generate.yaml` and worth knowing:

- **`tmol`** — force-field H-bond counting at the interface. This is what the paper's
  "interface hydrogen bond optimization" experiments used. Needs a working tmol install.
- **`bioinformatics`** — shape complementarity (`interface_sc`), buried surface area
  (`interface_dSASA`), interface hydrophobicity. Needs the `sc`/`dssp` binaries. Set
  `structure_source: af2folding` to score the *refolded* structure rather than the
  generated one.

Folding rewards always run before interface rewards, so `structure_source` can consume the
refolded output.

### Sampling

From `configs/pipeline/model_sampling.yaml`, mounted at `generation.args.*` / `generation.model.*`:

| Key | Default | Meaning |
|---|---|---|
| `generation.args.nsteps` | `400` | Denoising steps. 100–200 for smoke tests; keep 400 for real runs |
| `generation.args.guidance_w` | `1.0` | Classifier-free guidance strength |
| `generation.args.self_cond` | `true` | Self-conditioning |
| `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | `0.1` | Backbone stochasticity — the closest thing to a "temperature" |
| `generation.model.local_latents.simulation_step_params.sc_scale_noise` | `0.1` | Same, for side chains + sequence |
| `generation.dataloader.batch_size` | `16` | Samples per forward pass — **first lever to pull on OOM** |
| `generation.dataloader.dataset.nres.nsamples` | `4` | Distinct binder lengths sampled from `binder_length` range |

Binder length is **not** a generation config field — it comes from the target entry's
`binder_length: [min, max]` in `targets_dict.yaml`, sampled uniformly.

### Refinement (optional, off by default)

`generation.refinement.algorithm: null`. Set it to `sequence_hallucination` to run a
BindCraft-style ColabDesign optimization loop on the *final* selected samples, after search
and before reward scoring. Knobs: `n_greedy_iters: 15`, `n_temp_iters: 45`, and a
`loss_weights` block (`pae: 0.4`, `i_pae: 0.1`, `rg: 0.3`, `helix_binder: -0.3`, …). If
refinement fails on a sample, the unrefined structure passes through — silently.

### Filtering

```yaml
filter:
  filter_samples_limit: 1000        # keep top-N by total_reward
  delete_non_top_n_samples: false   # false → move to filtered_out_samples/
  dedup_sequence: true              # drop identical sequences before ranking
  reward_threshold: null            # hard reward floor
```

### Evaluation

```yaml
metric:
  binder_folding_method: colabdesign   # AF2-Multimer. Only other option: rf3_latest
  sequence_types: [self]               # ← repo default. Add mpnn for redesign
  inverse_folding_model: soluble_mpnn  # solubility-aware — the right default for binders
  num_redesign_seqs: 2                 # ← repo default is low; 8 for real campaigns
  interface_cutoff: 8.0
  compute_monomer_metrics: true
```

> **`binder_folding_method` accepts exactly two things.** `binder_eval.py:96-116` handles
> `colabdesign` and any name containing `rf3`; everything else raises
> `ValueError: Folding model '<x>' not supported`. Its own docstring (`:78`) advertises
> `protenix_*` and `boltz2_*`, three config comments list four backends, and `esmfold` is
> offered as the cheap option in `docs/`, `hardware.md`, and four skill files. None of them
> work. `esmfold` **is** valid for the different key `metric.monomer_folding_models`
> (`monomer_eval_utils.py:30`) — the two keys got conflated. Git history shows
> `binder_eval.py` has a single commit ("release 1.0.0"), so this was wrong from the public
> release rather than a regression.

- `self` = evaluate the sequence Complexa generated. `mpnn` = redesign the whole binder with
  SolubleMPNN. `mpnn_fixed` = redesign with interface residues held fixed.
- The default `sequence_types: [self]` and `num_redesign_seqs: 2` are tuned for speed. For a
  real campaign use `++metric.sequence_types="[self,mpnn]" ++metric.num_redesign_seqs=8`.
- Every metric column is prefixed by sequence type, so enabling `mpnn` gives you
  `mpnn_complex_i_pAE` alongside `self_complex_i_pAE`.

---

## Part 6 — Reading your results

### What counts as success

Defaults for `result_type: protein_binder`, following AlphaProteo's criteria. A sample passes
if **all three** hold for **at least one** redesigned sequence:

| Metric | Column | Threshold |
|---|---|---|
| Interface PAE | `{seq}_complex_i_pAE` | `× 31 ≤ 7.0` |
| Confidence | `{seq}_complex_pLDDT` | `≥ 0.9` |
| Self-consistency | `{seq}_binder_scRMSD_ca` | `< 1.5 Å` |

> **The ×31 is not a typo.** The stored i_pAE column is normalized to 0–1; 31 Å is AF2's PAE
> ceiling. So a raw column value of `0.19` is ~5.9 Å actual PAE and passes. Do not compare
> raw column values against 7.0.
>
> **pLDDT is on 0–1 here**, not 0–100, despite the docs' interpretation table using the
> 0–100 scale in places.

### ⚠️ Never override a threshold partially

Every doc in this repo — `docs/INFERENCE.md:220`, the `docs/CONFIGURATION_GUIDE.md` cheat
sheet, and (before I patched them) the bundled skills — tells you to relax thresholds like
this:

```bash
# DO NOT DO THIS
++aggregation.success_thresholds.i_pAE.threshold=10.0
```

That is actively harmful. `binder_analysis.py:317-318`:

```python
if success_thresholds is None:
    success_thresholds = DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()
```

The defaults apply **only when the key is entirely absent**. Supplying one metric replaces
the whole dict. And `parse_threshold_spec` (`analysis_utils.py:124-129`) defaults `scale` to
`1.0`, discarding the `scale: 31.0`. So that one override does three things at once:

1. drops the `pLDDT >= 0.9` criterion,
2. drops the `scRMSD_ca < 1.5` criterion,
3. compares a 0–1-scaled column against `10.0` — which every sample passes.

**Your reported success rate becomes 100%.** Silently.

Override with the complete dict instead:

```yaml
aggregation:
  success_thresholds:
    i_pAE:     {threshold: 10.0, op: "<=", scale: 31.0, column_prefix: complex}
    pLDDT:     {threshold: 0.9,  op: ">=", scale: 1.0,  column_prefix: complex}
    scRMSD_ca: {threshold: 1.5,  op: "<",  scale: 1.0,  column_prefix: binder}
```

Each entry takes `threshold`, `op`, `scale`, and `column_prefix` (`complex` vs `binder` —
which column family the metric name attaches to).

Note the key is **`scRMSD_ca`**, per `DEFAULT_PROTEIN_BINDER_THRESHOLDS` in
`src/proteinfoundation/result_analysis/binder_analysis_utils.py:88`. `normalize_metric_name`
maps `"scrmsd" → "scRMSD"` (`analysis_utils.py:45`), so a `scRMSD` key resolves to a column
suffix that doesn't exist. `docs/CONFIGURATION_GUIDE.md` uses the wrong name in its example.

The safest habit: leave the defaults alone, let `analyze` write `RAW_*_combined.csv`, and do
your own filtering in pandas. Then you can see exactly what you filtered on.

### The metrics, decoded

| Column | Meaning | Good |
|---|---|---|
| `{seq}_complex_i_pAE` | Interface predicted aligned error — the headline binding-confidence number | lower; `< 5 Å` excellent |
| `{seq}_complex_i_pTM` | Interface pTM | `> 0.5` |
| `{seq}_complex_pLDDT` | Overall confidence | `> 0.9` (0–1 scale) |
| `{seq}_binder_scRMSD_ca` | Did the binder refold to what you designed? | `< 1.5–2.0 Å` |
| `{seq}_binder_scRMSD_allatom` | Same, all atoms | `< 2 Å` |
| `{seq}_sequence` | The sequence of the best-ranked redesign | — |
| `{seq}_complex_pdb_path` | Path to the best refolded structure | — |
| `{seq}_complex_i_pAE_all` | *All* per-redesign values (list) | — |
| `_res_scRMSD_ca_esmfold` | Monomer designability — does the binder fold on its own? | `< 2.0 Å` |
| `_res_co_scRMSD_ca_esmfold` | Codesignability — does Complexa's *own* sequence fold to its own structure? | `< 2.0 Å` |
| `generated_binder_interface_sc` | Shape complementarity (needs `sc` binary) | higher |
| `generated_n_interface_hbonds_tmol` | Interface H-bond count (needs tmol) | higher |

`scRMSD` is self-consistency RMSD: design → predict → compare. It's the standard "is this
design real or an adversarial artifact of the predictor" check. Codesignability is the
stricter, more interesting version for Complexa specifically, because the model produced the
sequence itself.

### Which redesign gets reported

When `num_redesign_seqs > 1`, the non-`_all` columns report the redesign selected by a
composite ranking score — **default: minimize `i_pAE`**. Change it with `metric.ranking_criteria`:

```yaml
metric:
  ranking_criteria:
    i_pAE: {scale: 1.0, direction: minimize}
    pLDDT: {scale: 0.5, direction: maximize}
```

Changing this changes which numbers appear in your headline columns. Worth knowing before
you compare two runs.

### Output layout after `analyze`

```
evaluation_results/{config_name}_{task_name}[_{run_name}]/
├── monomer_metrics/              # designability, codesignability
├── filter_results/               # the subsets that passed
├── diversity/                    # res_div_*.csv — the diversity numbers
├── clusters/                     # the cluster membership directories themselves
├── secondary_structure/
├── amino_acid_distribution/
├── RAW_*.csv                     # everything, all columns
└── success_criteria_*.json       # the exact thresholds that were applied
```

Foldseek gives structural diversity (TM-score clustering); MMseqs2 gives sequence diversity
(`min_seq_id 0.1`, `coverage 0.7`). The scratch dirs during the run are named
`tmp_foldseek_diversity_joint/` and `tmp_mmseqs_diversity_seq/` and get deleted.

Key summary files: `res_filter_binder_pass_*.csv` (success rates),
`res_designability.csv`, `res_div_foldseek_{mode}_*.csv`, `res_div_mmseqs_*.csv`.

Diversity is computed twice — over all samples and over successful ones only. The successful
subset is the number that matters: 100 designs that all pass but cluster into 2 folds is a
worse campaign than 30 passes across 15 clusters.

### Search provenance — a genuinely useful feature

Every sample's filename encodes its full search lineage:

```
job_{id}_n_{len}_id_{idx}_{tag}_binder.pdb

beam_orig0_bm2-s0to100br3-s100to200br0-s200to300br1-s300to400br2
│    │     │   └── one segment per search step: branch 3 chosen at step 0→100
│    │     └────── initial beam index 2
│    └──────────── original sample 0
└───────────────── algorithm
```

The tag is also a `metadata_tag` CSV column, and **prefix = ancestry**: if tag A is a prefix
of tag B, A is B's ancestor. So you can reconstruct exactly which branching decisions
produced a winner:

```bash
grep "beam_orig0_bm2" results.csv                 # everything descended from that beam
grep "beam_orig0.*s300to400" results.csv          # the whole final-step candidate pool
```

The `sample_type` column distinguishes `lookahead` (intermediate candidates) from `final`.
When you're deciding whether a wider beam is worth the compute, this is the data to look at.

---

## Part 7 — Your own target

Two routes. The CLI:

```bash
complexa target add my_target \
    --target-path /abs/path/to/my_protein.pdb \
    --target-input A1-150 \
    --hotspot-residues A45 A67 A89 \
    --binder-length 60 120

complexa target list
complexa target show my_target
```

Or edit `configs/targets/targets_dict.yaml` directly:

```yaml
target_dict_cfg:
  my_target:
    source: my_targets              # subfolder under $DATA_PATH/target_data/
    target_filename: my_protein     # without .pdb
    target_path: /abs/path.pdb      # optional; overrides source/filename lookup
    target_input: A1-150            # "A1-150" | "A1-100,B1-50" | "A" (whole chain)
    hotspot_residues: ["A45", "A67", "A89"]
    binder_length: [60, 120]
    pdb_id: null
```

Then `++generation.task_name=my_target`.

**Getting this right matters more than any hyperparameter.** Three things to think about:

1. **Hotspots** are how you tell the model where to bind. Pick them from known epitope /
   functional-site knowledge, not from what looks convenient in PyMOL. Too few and the
   binder wanders; too many over-constrains it.
2. **`target_input`** should be the folded domain you actually want targeted. Trim
   disordered tails and irrelevant domains — they cost compute and can distract the reward
   model.
3. **`binder_length`** range: the bundled targets use spans like `[64, 155]` and `[80, 150]`.
   Wider ranges explore more; `nsamples` controls how many distinct lengths get drawn.

---

## Part 8 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `complexa: command not found` | venv not built or not activated | `./env/build_uv_env.sh` then `source .venv/bin/activate` |
| `InterpolationKeyError: AF2_DIR` | Reward config wants AF2, `.env` doesn't define it | `complexa download --all`, set `AF2_DIR` in `.env` |
| `ValueError: Unknown search algorithm` | Used underscores | Hyphens: `beam-search`, not `beam_search` |
| CUDA OOM during generation | Batch too large | In order: `++generation.dataloader.batch_size=8`, then `++gen_njobs=1`, then reduce `beam_width`/`n_branch` |
| CUDA OOM during evaluation | AF2 batching | `++eval_njobs=1`, `++metric.num_redesign_seqs=2`, `++metric.keep_folding_outputs=false`. **Not** `esmfold` — see below |
| `ValueError: Folding model 'esmfold' not supported` | Wrong key | `esmfold` belongs to `metric.monomer_folding_models`, not `metric.binder_folding_method` |
| Disk filling up fast | Lookahead samples | `++generation.search.beam_search.keep_lookahead_samples=false` |
| `.env_example not found` | Wrong working directory | `cd` to repo root |
| GLIBC errors on import | Ubuntu 20.04 + UV | `complexa init docker` (positional, not `--runtime`) |
| Import errors running modules directly | PYTHONPATH | `export PYTHONPATH=$PWD/src:$PWD/community_models:$PYTHONPATH` |
| Missing weights | — | `complexa download --status` |

### Gotchas the docs don't flag loudly

Several of these are places where the repo's own docs disagree with its source. Source wins.

The two that can actually corrupt a result, repeated here because they matter most:

- **A partial `success_thresholds` override silently reports 100% success** (Part 6). Every
  doc in the repo recommends the broken form.
- **`metric.binder_folding_method=esmfold` raises `ValueError`** (Part 5). It's offered as the
  cheap option in six places. Only `colabdesign` and `rf3_*` work.

The rest are cosmetic-to-annoying:

- **`.claude/skills/_shared/reference/hardware.md` had wrong algorithm names** (underscored) —
  patched. Trust `search_factory.py`.
- **`complexa init --runtime docker`** (README + `complexa-setup` skill) is wrong — `runtime`
  is positional: `complexa init docker`.
- **`complexa status <config>`** (README) is wrong — `status` takes no positional.
- **`./inference/.../filtered/` and `samples/`** (`complexa demo` output) don't exist.
- **`docs/CONFIGURATION_GUIDE.md`'s threshold example keys on `scRMSD`**, but the default
  constant uses `scRMSD_ca` — the example adds a criterion instead of relaxing one.
- **`docs/SEARCH_METADATA.md`'s filename format omits the `_binder` suffix** that binder-run
  PDBs actually carry.
- **i_pAE is stored normalized** (÷31). Comparing raw column values to the 7.0 threshold
  will make everything look like it passed.
- **pLDDT columns are 0–1**, though parts of the docs discuss the 0–100 convention.
- **`_all` columns are strings after `pd.read_csv`** — parse with `ast.literal_eval`.
- **The default `sequence_types: [self]`** means no MPNN columns at all unless you add them.
  If your analysis is looking for `mpnn_complex_i_pAE` and finding nothing, this is why.
- **`keep_folding_outputs: true` is the eval default** and roughly doubles disk footprint.
- **Monomer column collisions**: after merging monomer results into binder results, the
  monomer copy of a shared column gets a `_monomer` suffix.
- **ColabDesign cannot handle ligand targets** — irrelevant for protein binders, relevant the
  moment you try the ligand pipeline.

---

## Part 9 — Where to go next

| Want to | Read |
|---|---|
| Every config field | `docs/CONFIGURATION_GUIDE.md` |
| Every metric, in detail | `docs/EVALUATION_METRICS.md` |
| SLURM, stage-by-stage runs | `docs/INFERENCE.md` |
| Parameter sweeps | `docs/SWEEP.md` |
| Search lineage tags | `docs/SEARCH_METADATA.md` |
| Training / fine-tuning | `docs/TRAINING.md` |
| Run anything without hand-writing commands | `.claude/skills/complexa-*/SKILL.md` — start with `complexa-design` |
| What's wrong with those skills | `SKILLS_AUDIT.md` (repo root) |
| The skills' design rationale | `docs/AGENT_SKILLS.md`, `.claude/skills/README.md` |

Plus `SKILLS_AUDIT.md` at the repo root — the full defect list for `.claude/skills/`, with the
nine root causes that account for most of it. Worth skimming before you trust any number in
those files that I didn't patch.

### Known repo defects to be aware of

Found while auditing; **not** patched, because they live outside `.claude/skills/`:

| Where | Defect | Fix |
|---|---|---|
| `configs/evaluate_from_pdb_dir.yaml:22` | `defaults: - generation/targets_dict@dataset`, but `configs/generation/targets_dict.yaml` doesn't exist. Hydra cannot compose it, and because it's a `defaults:` entry you can't redirect it from the CLI. **The "score an existing PDB directory" entry point is broken as shipped.** | One line: `- /targets/targets_dict@dataset` (or `/targets/ligand_targets_dict`) |
| `binder_eval.py:78` | Docstring promises `protenix_*` / `boltz2_*` support the code doesn't have | Fix the docstring, or restore the backends |
| `docs/INFERENCE.md:104`, `docs/EVALUATION_METRICS.md:82, :262` | Cite `configs/evaluate_motif_binder.yaml`; the real path is `configs/example/evaluate_motif_binder.yaml` | Repoint |
| `docs/INFERENCE.md:220`, `docs/CONFIGURATION_GUIDE.md` cheat sheet | Recommend the partial `success_thresholds` override that silently reports 100% success (Part 6) | Show the full-dict form |
| `configs/pipeline/binder/binder_evaluate.yaml:23`, `configs/evaluate_from_pdb_dir.yaml:70`, `configs/pipeline/ame/ame_analyze.yaml:27-29` | Stale comments listing four folding backends / wrong AME thresholds | Update comments |
| `README.md:286` | `complexa status <config>` — takes no positional | Drop the argument |

### A sensible learning path

0. If you're using Claude Code in this repo, invoke `complexa-setup` and let it drive the
   install; then `complexa-design` for the runs below. The steps are the same either way —
   the skill just fills in the flags.
1. Smoke test on `02_PDL1` with `nsteps=100`, `nsamples=2`, `single-pass`. Confirm the chain
   runs and read every output file it produces.
2. Same target, `best-of-n` with `replicas=4`, full 400 steps. Compare success rates. Now you
   know what search buys you.
3. Switch to `beam-search`, `beam_width=8`. Use the metadata tags to see which branches won.
4. Add `++metric.sequence_types="[self,mpnn]"` and compare the model's own sequences against
   SolubleMPNN redesigns — this is the most informative diagnostic Complexa offers, and it's
   specific to the fact that it generates sequence.
5. Your own target. Expect the hotspot choice to matter more than anything you tuned in 1–4.
