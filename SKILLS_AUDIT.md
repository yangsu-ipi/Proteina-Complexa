# `.claude/skills/` Audit — Proteina-Complexa

Systematic verification of all five bundled skills (17 files, ~3,500 lines) against the
source they document. Every defect below cites the source file and line that contradicts it.

**Result: 129 defects.** 42 BREAKING (command errors out, or the documented setting silently
does nothing), 54 MISLEADING (works but wrong information), 33 STALE (refers to something
absent).

Audit date: 2026-08-20. Method: read each file in full, then verify every CLI invocation
against the argparse definitions in `src/proteinfoundation/cli/`, every Hydra override key
against the composed config tree, every default against the YAML, every path against the
filesystem, and every helper-script invocation against that script's own argument parser.

---

## Part 1 — Nine root causes

Most of the 128 defects are ten or so mistakes repeated across files. Fixing these root
causes fixes ~70% of the list.

### RC-1. Phantom folding backends: `esmfold`, `boltz2_default`, `protenix_*` — BREAKING

`metric.binder_folding_method` accepts exactly two things. From
`src/proteinfoundation/evaluation/binder_eval.py:96-116`:

```python
if folding_model == "colabdesign":
    ...
elif "rf3" in folding_model:
    ...
else:
    raise ValueError(f"Folding model '{folding_model}' not supported")
```

Everything else raises. But these values are advertised as valid in **at least 11 places**:

| Location | Claim |
|---|---|
| `binder_eval.py:78` (the function's own docstring) | "Supported models: colabdesign, protenix_*, rf3_*, boltz2_*" |
| `configs/pipeline/binder/binder_evaluate.yaml:23` | comment lists 4 backends |
| `configs/evaluate_from_pdb_dir.yaml:70` | same stale comment |
| `complexa-design/SKILL.md:35, :289, :318` | "ESMFold (fast iteration)"; offered as the OOM fix |
| `complexa-design/reference/overrides.md:161, :234` | enum + "Cheap iteration with ESMFold eval" |
| `complexa-design/reference/troubleshooting.md:56-60` | "switch the eval backend to ESMFold" |
| `complexa-evaluate-pdbs/SKILL.md:27, :44-47, :108, :196, :210` | offered as an AskUserQuestion choice |
| `complexa-evaluate-pdbs/reference/eval_configs.md:9, :10` | enum incl. `protenix_base_default_v0.5.0` |
| `complexa-sweep/reference/sweep_axes.md:70` | offered as a **sweep axis** — every config in that axis crashes |
| `_shared/reference/hardware.md:33` | `++metric.binder_folding_method=colabdesign\|rf3_latest\|esmfold` |
| `docs/EVALUATION_METRICS.md`, `docs/INFERENCE.md` | listed among alternatives |

`esmfold` **is** valid for the different key `metric.monomer_folding_models`
(`monomer_eval_utils.py:30`: `VALID_FOLDING_MODELS = ["esmfold", "colabfold"]`). The two keys
appear to have been conflated.

> **This one is probably a code regression, not a docs bug.** When a function's own
> docstring, three config comments, two `docs/` files, and four skill files all promise
> backends the implementation doesn't have, the likelihood is that support was removed (or
> never landed) and nothing else was updated. Worth checking git history before patching the
> docs to match — you may want to restore the backends instead.

### RC-2. Partial `success_thresholds` overrides silently pass everything — BREAKING

`src/proteinfoundation/result_analysis/binder_analysis.py:317-318`:

```python
if success_thresholds is None:
    success_thresholds = DEFAULT_PROTEIN_BINDER_THRESHOLDS.copy()
```

The defaults are used **only when the key is entirely absent**. Supplying one metric
*replaces the whole dict*. And `analysis_utils.py:124-129` (`parse_threshold_spec`) defaults
`scale` to `1.0` when not given, discarding the `scale: 31.0` that
`binder_analysis_utils.py:79` relies on.

So the single most widely-documented override in this repo:

```bash
++aggregation.success_thresholds.i_pAE.threshold=10.0
```

does three harmful things at once: drops the `pLDDT >= 0.9` criterion, drops the
`scRMSD_ca < 1.5` criterion, and compares a 0–1-scaled column against `10.0` — which every
sample passes. **Reported success rate becomes 100%.**

Documented in this broken form at: `complexa-design/SKILL.md:291`,
`overrides.md:197-205, :239-240`, `troubleshooting.md:273-276, :281`, `docs/INFERENCE.md:220`,
`docs/CONFIGURATION_GUIDE.md` cheat sheet — and in the onboarding guide I wrote you before
running this audit.

**Correct usage** is to supply the complete dict (as `docs/EVALUATION_METRICS.md:1099` does):

```yaml
aggregation:
  success_thresholds:
    i_pAE:     {threshold: 10.0, op: "<=", scale: 31.0, column_prefix: complex}
    pLDDT:     {threshold: 0.9,  op: ">=", scale: 1.0,  column_prefix: complex}
    scRMSD_ca: {threshold: 1.5,  op: "<",  scale: 1.0,  column_prefix: binder}
```

Note the key is **`scRMSD_ca`** (`binder_analysis_utils.py:88`), not `scRMSD`.
`normalize_metric_name` maps `"scrmsd" → "scRMSD"` (`analysis_utils.py:45`), so `scRMSD`
resolves to a column suffix that doesn't exist.

### RC-3. `complexa init` is misdocumented, and the real gate is undocumented — BREAKING

Three separate problems in `complexa-setup`:

1. **`--runtime` is not a flag.** `cli_runner.py:1130-1136` declares `runtime` as a
   positional (`nargs="?"`, choices `uv`/`docker`). `complexa init --runtime docker` exits
   with `unrecognized arguments`. Correct: `complexa init docker`.
2. **`COMPLEXA_RUNTIME` does not exist.** Zero occurrences in `.env_example` (154 lines),
   `src/`, `configs/`, `env/`, `script_utils/`. The skill greps for it three times
   (`SKILL.md:102, :125, :238`); all three return empty, and the Step 6 command substitution
   expands to an empty string.
3. **`env.sh` and `COMPLEXA_INIT` are never mentioned at all** — and they are the actual
   gate. `cli_runner.py:1619-1673` writes a *new* `env.sh` exporting
   `COMPLEXA_INIT="<runtime>"`. `cli_runner.py:1962-1977` (`_check_complexa_init`) hard-exits
   `design`/`generate`/`filter`/`evaluate`/`analyze`/`analysis`/`target` with "Environment
   not initialized" unless `COMPLEXA_INIT` is in the environment.

Consequence: the skill's **recommended path** ("Path A: file edit, preferred for agents") tells
you to `cp .env_example .env` and skip the CLI. Follow it and every subsequent `complexa`
command exits 1. The skill also claims `complexa init` rewrites lines inside `.env`
(`SKILL.md:31, :92-108, :118-119`; `env_keys.md:21, :130-132`) — it never modifies `.env`
after the initial copy, and re-running it without a runtime positional prints usage and
`sys.exit(1)` rather than swapping anything.

The non-existent helper `_swap_runtime_in_env` is cited as the implementation
(`SKILL.md:31`); repo-wide grep finds it only in that skill file.

### RC-4. `_shared/scripts/` interface drift — BREAKING

Both shared scripts are called with arguments they don't accept, or their output is read from
the wrong path.

**`preflight.sh` writes `./preflight.json`**, not `./complexa_setup/preflight.json`
(`preflight.sh:13`: `OUT="./preflight.json"`; only `--out` changes it, and no skill passes
`--out`). Wrong in `complexa-setup/SKILL.md:55`, `complexa-design/SKILL.md:48`,
`complexa-sweep/SKILL.md:29`. (`complexa-evaluate-pdbs/SKILL.md:41` gets it right.)

**`preflight.sh`'s JSON keys are different from what's documented.** `complexa-design/SKILL.md:53-57`
tells you to read `ckpts.complexa[.ckpt]`, `env.AF2_DIR`, `env.RF3_CKPT_PATH`,
`env.RF3_EXEC_PATH`. The script emits top-level `"checkpoints"` keyed by full filename
(`:98-113`); `env` contains only `.env_loaded`, `.env_path`, `missing_required`,
`LOCAL_CODE_PATH`, `LOCAL_DATA_PATH`, `CKPT_PATH` (`:136-138`); AF2/RF3 ckpt paths live under
`community_models` (`:127-128`) and `RF3_EXEC_PATH` under `tools.rf3` (`:117-124`). All six
documented lookups return nothing.

**`write_manifest.py` accepts only `--output-dir`, `--command`, `--skill`, `--out`**
(`write_manifest.py:45-51`, first three required). `complexa-setup/SKILL.md:236-240` passes
`--kind`, `--runtime`, `--preflight` and omits all three required args → argparse exit 2.

**`write_manifest.py` reads `<output-dir>/.hydra/config.yaml`** (`:79-80`), but no pipeline
writes `.hydra/` under the output dir — `search_binder_local_pipeline.yaml:37-39` sets
`hydra.run.dir: ./logs/hydra_outputs/...`, and `complexa evaluate`/`analyze` set no override
(`cli_runner.py:632-651`). So `config` and `checkpoints` come out `null` in every manifest,
contradicting `complexa-design/SKILL.md:262-263`, `complexa-sweep/SKILL.md:173`, and
`complexa-evaluate-pdbs/SKILL.md:188`.

### RC-5. Phantom `hbplus` — BREAKING / STALE

`hbplus` is documented as a reward model and as a `pre_refolding`/`refolded` sub-toggle. It
exists nowhere: `ls src/proteinfoundation/rewards/` gives only `alphafold2_reward.py`,
`base_reward.py`, `bioinformatics_reward.py`, `rf3_reward.py`, `tmol_reward.py`; the config
blocks contain exactly `bioinformatics` and `tmol` (`binder_evaluate.yaml:43-53`); and
`evaluate.py:401-403` reads only those two keys.

Occurrences: `overrides.md:116` (`hbplus`, `hbplus_af2`, `hbplus_boltz2` "pre-wired but
commented out"), `overrides.md:172, :174`, `pipelines.md:89`, `eval_configs.md:36-37`,
`sweep_axes.md:77`, `complexa-evaluate-pdbs/SKILL.md:200`, `complexa-setup/SKILL.md:103`
(`HBPLUS_EXEC`), `env_keys.md:142, :145`. The only real trace is a preflight env probe
(`preflight.sh:118`).

### RC-6. `configs/evaluate_motif_binder.yaml` is cited at the wrong path — BREAKING

> **Corrected after the first pass.** The file *does* exist — at
> `configs/example/evaluate_motif_binder.yaml`, not `configs/`. My initial audit checked
> `ls configs/*.yaml` and wrongly concluded it was absent. It is a working config
> (`defaults: - /design_tasks/ame_dict_v2@dataset`, `protein_type: motif_binder`) that
> supports both `motif_protein_binder` and `motif_ligand_binder`. So this is a **wrong-path**
> defect, not a phantom-file defect, and the fix is to repoint the citations rather than
> delete them.

Cited as `configs/evaluate_motif_binder.yaml` in `complexa-design/reference/pipelines.md:147, :167`,
`complexa-evaluate-pdbs/SKILL.md:77`, `eval_configs.md:12, :16, :66, :106` — and in
`docs/INFERENCE.md:104`, `docs/EVALUATION_METRICS.md:82, :262`, and
`configs/analyze_motif_binder.yaml:3`, which is probably where the skills picked it up. All of
those paths 404; the real one is `configs/example/evaluate_motif_binder.yaml`. Note this means
`docs/` carries the same defect, so it is worth fixing there too.

### RC-7. `complexa validate` takes no Hydra overrides, and `validate target` can never pass — BREAKING

`cli_runner.py:1296-1318` gives the `validate` subparser only `type`, `config` (`nargs="?"`),
and `--target`. It does **not** get the `overrides` positional that `design`/`generate`/`analyze`
receive via `add_common_args`. So `complexa-design/SKILL.md:159-162` and
`troubleshooting.md:176` — which pass `++generation.task_name=...` to it — abort with
`unrecognized arguments`.

Worse, `complexa validate target CONFIG --target NAME` can never succeed on any config in the
repo. `validate.py:183-187` uses a plain `yaml.safe_load` and does not compose Hydra
defaults. `validate.py:306-308` only follows a default entry when it is a `dict` containing
`"generation"`, but every entry in `search_binder_local_pipeline.yaml:12-16` is a plain
*string* (`'pipeline/binder/binder_generate@generation'`). The fallback at
`validate.py:350-352` looks for `configs/generation/targets_dict.yaml`, which doesn't exist.
Result: `target_dict_cfg is None` → `add_fail("Target config", "Could not find target_dict_cfg
in config")`. `complexa-target/SKILL.md:185-197` presents four checks that never execute, and
instructs the reader to "stop and fix on the first failure."

Also `complexa-target/SKILL.md:153-155` and `troubleshooting.md:179` claim the validator
catches "unknown override keys" — there is no config-key validation anywhere in
`validate.py` (859 lines).

### RC-8. `complexa target` writes ligand targets into the protein dict — BREAKING

`target_manager.py:24` — `DEFAULT_TARGETS_DICT_PATH = Path("configs/targets/targets_dict.yaml")`,
and `get_default_dict_path()` (`:130-148`) returns only that or the legacy
`configs/generation/targets_dict.yaml`. There is no ligand-aware routing. The ligand examples
in `complexa-target/SKILL.md:163-172` and `target_schema.md:176-215` pass no `--dict`, so they
append ligand entries to `targets_dict.yaml`, where the **protein** pipeline reads them
(`binder_generate.yaml:8`) and the ligand pipeline never sees them.

`--ligand`/`--protein` filter *within* one dict (`target_manager.py:495-496`), and
`targets_dict.yaml` has zero entries with a `ligand:` key — so `complexa target list --ligand`
always prints "No targets found", making it useless for the collision check the skill
prescribes.

### RC-9. The sweep pipeline does not produce evaluable output — BREAKING

`generate_inference_configs.py:236-242` emits a matched pair per combination:
`inf_{idx}_{run_name}.yaml` **and** `eval_{idx}_{run_name}.yaml`, where only the *eval* config
sets `sample_storage_path=root_path` and `output_dir`/`results_dir=./evaluation_results/eval_{idx}_{run_name}`.

But `complexa-sweep/SKILL.md:100-114` loops `complexa design` over the `inf_*.yaml` files
only, and never invokes the eval configs. `complexa design` runs all four stages from that one
config. Generation honours the injected `root_path` (`generate.py:544, :573-574` →
`./inference/inf_0_my_sweep`), but `evaluate.py` never reads `root_path` — with
`sample_storage_path` absent it constructs `./inference/{config_name}_{target_task_name}` then
appends `_{run_name}` (`:732, :752-762`), i.e. `./inference/inf_0_my_sweep_22_DerF21_search_binder_local`.
That is not where generate wrote. Same for analyze (`analyze.py:2921, :2946-2957`).

**A sweep run as documented produces structures and no usable evaluation results at any
documented path.**

Compounding it, the ranking columns the skill parses are invented: `i_pae`, `i_plddt`,
`sc_rmsd`, `binder_seq`, `passes_filter` (`SKILL.md:147, :158-160`; `sweep_axes.md:145-148`).
Repo-wide grep for `passes_filter` matches only those two skill files. Real columns are
`{seq}_complex_i_pAE`, `{seq}_complex_pLDDT`, `{seq}_binder_scRMSD_ca`; there is no
interface-pLDDT column at all and no boolean pass column in the raw CSV. The stated defaults
(`i_pae < 10`, `i_plddt > 0.7`, `sc_rmsd < 2.0`) are also wrong in both value and scale — see
RC-2.

### RC-10. `configs/evaluate_from_pdb_dir.yaml` cannot compose — BREAKING (repo bug, not a skill bug)

Found while patching, outside the original scope. `configs/evaluate_from_pdb_dir.yaml:22`:

```yaml
defaults:
  - generation/targets_dict@dataset
```

`configs/generation/` contains exactly `base_gen_data.yaml`, `validation.yaml`,
`validation_local_latents.yaml`. There is no `targets_dict.yaml` there, and this is the only
reference to that path anywhere in `configs/`. Every other config uses the real group,
`- /targets/targets_dict@dataset` (e.g. `configs/evaluate.yaml:31`).

Because the broken entry is in the `defaults:` list rather than an override, it cannot be
redirected from the command line. **The entire "score an existing PDB directory" entry point —
the thing `complexa-evaluate-pdbs` is built around — fails at Hydra composition as shipped.**

Fix is one line:

```yaml
defaults:
  - /targets/targets_dict@dataset        # or /targets/ligand_targets_dict for ligand targets
```

This is a defect in `configs/`, not in the skills. The skills faithfully documented a config
that doesn't load. I have not patched it — it's outside the `.claude/skills/` scope you
approved, and you may prefer a different fix (e.g. shipping a
`configs/generation/targets_dict.yaml` symlink for backwards compatibility).

---

## Part 2 — Additional defects, by file

Defects already covered by a root cause above are marked `→ RC-n` and not restated.

### `_shared/reference/hardware.md`

| Line | Defect | Severity |
|---|---|---|
| 41-44 | Search algorithms written with underscores (`single_pass`, `best_of_n`, `beam_search`, `fk_steering`). `search_factory.py:31-40` accepts hyphens only and raises `ValueError: Unknown search algorithm` otherwise. | BREAKING |
| 44 | `fk_steering` + `num_particles=N`. No such key — `binder_generate.yaml:66-70` has `n_branch`, `beam_width`, `temperature`. | BREAKING |
| 33 | `esmfold` offered as a `binder_folding_method` → RC-1 | BREAKING |
| 56-58 | `gen_njobs` / `eval_njobs` "defaults pulled from configs" given as 2/2. All three pipelines ship `1`/`1`. | MISLEADING |

### `complexa-setup/SKILL.md`

| Line | Defect | Severity |
|---|---|---|
| 114, 268, 280 | `complexa init --runtime docker` → RC-3 | BREAKING |
| 92-108 | "Path A: file edit" omits `env.sh`; following it blocks every later command → RC-3 | BREAKING |
| 102, 125, 238 | `COMPLEXA_RUNTIME` → RC-3 | BREAKING |
| 55 | `./complexa_setup/preflight.json` → RC-4 | BREAKING |
| 236-240 | `write_manifest.py --kind/--runtime/--preflight` → RC-4 | BREAKING |
| 31 | `_swap_runtime_in_env` does not exist → RC-3 | STALE |
| 103 | `HBPLUS_EXEC` → RC-5 | STALE |
| 104 | `CACHE_DIR=` has no line in `.env_example` to swap (only `LOCAL_CACHE_DIR`, `DOCKER_CACHE_DIR`) — yet `configs/dataset/unified/plinder.yaml:35` requires `${oc.env:CACHE_DIR}` | STALE |
| 118-119 | "only runtime-dependent lines are swapped, user edits preserved" — `init` never modifies `.env`; re-running without a runtime exits 1 (`cli_runner.py:1706-1723`) | MISLEADING |
| 172 | `--all` includes ESMFold. `download_startup.sh:848-855` runs exactly pmpnn, ligmpnn, af2, esm2, rf3. ESMFold has no download function; it comes from `script_utils/download/download_esmfold_model.py`. | MISLEADING |
| 173 | `--everything` includes Boltz2/Protenix. `:856-866` = 3 Complexa + the same 5. `boltz`/`protenix` appear nowhere in the script. | MISLEADING |
| 59, 160, 172, 173 | Sizes: `--all` "~50 GB" and `--everything` "~100+ GB". Script's own annotations (`:680-684`, `:199`): pmpnn ~50 MB + ligmpnn ~500 MB + af2 ~5 GB + esm2 ~2.6 GB + rf3 ~2.5 GB ≈ **10.7 GB**; `--everything` ≈ **20 GB**. Off by ~4-5×. | MISLEADING |
| 157 | "~6 community-model families" — 5 | MISLEADING |
| 33, 156 | "~1000 lines of bash" — `wc -l` = 902 | MISLEADING |
| 58 vs 261 | Contradicts itself on VRAM minimum: 40 GB at :58, 24 GB at :261 (and `hardware.md:12` says 24). A 24 GB host is wrongly warned off. | MISLEADING |
| 135, 277 | Repo called `protein-foundation-models`; it is `Proteina-Complexa` (`.env_example:25`) | MISLEADING |
| 204 | `--status` "groups by Complexa / community / optional" — `download_startup.sh:650-672` has only `Complexa Models (Required):` and `Core Models:` | MISLEADING |
| 254 | manifest "records init + download invocations + runtime" — `write_manifest.py` records one `--command` string and has no runtime field | MISLEADING |

### `complexa-setup/reference/env_keys.md`

| Line | Defect | Severity |
|---|---|---|
| 60 | "`--esm2` or `--esmfold` downloads fail" — `--esmfold` hits the `*)` branch → `Unknown option`, exit 1 (`download_startup.sh:831-893`) | BREAKING |
| 130-132, 134-140 | `--runtime` flag / `COMPLEXA_RUNTIME` section / `CACHE_DIR` as a shipped alias → RC-3 | BREAKING / STALE |
| 142, 145 | `HBPLUS_EXEC` → RC-5 | STALE |
| 24, 148 | `ESMFOLD_DIR` does not exist (grep: these two lines only) | STALE |
| 3 | "Complete reference for every variable in `.env_example`" — omits `CLUSTER_USER` (:37) and the entire 20-key SLURM block (:110-154). "Sections mirror `.env_example`" is also false. | MISLEADING |
| 50-51 | W&B placeholder claim wrong twice: the guard is in `env/docker-ops.sh:352, :355` (not training code) and tests against `"YOUR WANDB KEY"` (spaces) while `.env_example:20-21` ships `YOUR_WANDB_KEY` (underscores) — so the shipped placeholders **are** injected. `train.py:384` reads the entity from `cfg_exp.log.wandb_entity`, not `WANDB_ENTITY`. | MISLEADING |
| 150, 163-165 | `AF2_DIR`/`ESM_DIR`/`RF3_DIR`/`RF3_CKPT_PATH` and the `UV_*` families "auto-managed by `complexa init`" — `_generate_env_sh` touches only `_TOOL_VARS` plus five docker-branch paths | MISLEADING |
| 177 | "`.env` … or any parent up to the repo root" — `validate.py:254` is `Path(".env")`, CWD only. The parent-walk (`cli_runner.py:1610-1616`) applies to `.env_example` during `init`. | MISLEADING |
| 25, 26 | placeholder given as `/path/to/protein-foundation-models` — `.env_example:25` is `/path/to/Proteina-Complexa` | MISLEADING |
| 66-71 | `CACHE_DIR` "active alias resolves to this for UV runtime" — no such line; `cli_runner.py:1650-1661` exports `LOCAL_CACHE_DIR` | MISLEADING |
| 7 | `proteinfoundation/cli/validate.py` — path is `src/proteinfoundation/...` (:175 gets it right) | STALE |

### `complexa-setup/reference/downloads.md`

| Line | Defect | Severity |
|---|---|---|
| 28, 39 | `--esmfold` "passes through unchanged" + a table row for it → not accepted, exit 1 | BREAKING |
| 28, 41 | `--boltz2` → not accepted; `boltz` appears nowhere in the script | BREAKING |
| 72-76 | "`LOCAL_CHECKPOINT_PATH` defaults to `${LOCAL_CODE_PATH}/ckpts`, which matches where `complexa download` writes." `.env_example:28` is `${LOCAL_CODE_PATH}/checkpoints`; `download_startup.sh:238` writes `$PROJECT_ROOT/ckpts`. They do **not** match, and the doc's conclusion is inverted. `env_keys.md:75-79` states this correctly, contradicting this file. | BREAKING |
| 30 | "listed in `show_help`" — `show_help:776-805` lists 5 flags, not these | STALE |
| 37, 65 | AF2 destination `AF2/params/` — `download_startup.sh:187, :217, :222` extracts directly into `community_models/ckpts/AF2/` | MISLEADING |
| 37 | `--af2` "~3 GB" — script says ~5 GB (`:199`, `:682`) | MISLEADING |
| 40 | `--rf3` "~10 GB" — script says ~2.5 GB (`:684`). Off 4×. | MISLEADING |
| 69 | `Boltz2/` in the layout tree — never created | STALE |
| 116-133 | The `--status` sample output is fabricated: header is `Current Installation Status`; groups are `Complexa Models (Required):` / `Core Models:`; the AF2 row reads `AlphaFold2:`; there are no `ESMFold:` or `Boltz2:` rows; missing models print a `○ Missing (dir)` header plus one indented `✗ <filename>` per file (`:607-614`), not a flat line | MISLEADING |
| 144 | "Failed downloads leave a zero-byte file then `rm -f` it — safe to retry." The only `rm` (`:228`) removes the AF2 tar **after success**. On failure the partial file stays, and the skip check is `[ -f ] && [ -s ]` (`:245`) — so a partial non-empty ckpt is silently treated as installed on retry. The opposite of the claim. | MISLEADING |
| 47-48 | "relative to the current working directory" — `main()` does `cd "$PROJECT_ROOT"` (`:817`); CWD is irrelevant | MISLEADING |

### `complexa-design/SKILL.md`

| Line | Defect | Severity |
|---|---|---|
| 35, 289, 318 | `esmfold` / Boltz2 backends → RC-1 | BREAKING |
| 159-162 | `complexa validate design` with `++` overrides → RC-7 | BREAKING |
| 48, 53-57 | preflight path + JSON keys → RC-4 | BREAKING |
| 243, 267 | `./evaluation_results/${RUN_NAME}/`. `evaluate.py:756` → `./evaluation_results/{config_name}_{task_name}`, then `:766-767` appends `_{run_name}` | BREAKING |
| 249-250 | `binder_results_*_combined.csv`. `analyze.py:3036` writes only `RAW_{result_type}_results_{config_name}_combined.csv`; `binder_results_{config_name}_{job_id}.csv` (`evaluate.py:881`) is the per-job file and is never `_combined` | BREAKING |
| 251-252 | `ls ./evaluation_results/*/res_filter_*` and `res_div_*` — `organize_results()` (`analyze.py:2812-2855`) has already moved them into `filter_results/` and `diversity/`. Globs match zero files. | BREAKING |
| 291 | partial `success_thresholds` override → RC-2 | BREAKING |
| 153-155 | "catches … unknown override keys" → RC-7 | MISLEADING |
| 255 | "Pull the success rate from `res_filter_binder_pass_*.csv`" — that prefix is protein-binder-only (`binder_analysis.py:548`); ligand writes `res_filter_ligand_pass_*`, AME `res_filter_motif_binder_pass_*` | MISLEADING |
| 283 | `++run_name` default "(config stem)" — the default is the `run_name:` field (`search_binder_local`), while the config stem is `search_binder_local_pipeline`. Both appear in the output path, so conflating them breaks path reconstruction. | MISLEADING |

### `complexa-design/reference/overrides.md`

| Line | Defect | Severity |
|---|---|---|
| 116 | `hbplus`, `hbplus_af2`, `hbplus_boltz2` reward models → RC-5 | BREAKING |
| 172, 174 | `metric.pre_refolding.hbplus` / `refolded.hbplus` → RC-5 | BREAKING |
| 161, 234 | `esmfold` backend → RC-1 | BREAKING |
| 197-205, 239-240 | partial `success_thresholds` → RC-2 | BREAKING |
| 206-207 | `aggregation.motif_binder_success_thresholds.motif_rmsd_pred.threshold` — wrong schema. `motif_binder_analysis_utils.py:40-89` uses `{"binder": {...}, "motif": [ {column, threshold, op}, ... ]}`; `parse_motif_binder_success` reads `.get("binder", {})` / `.get("motif", [])` (`:282, :287`). A flat dict yields an empty binder map and `motif_binder_analysis.py:217-218` then `continue`s — **no pass rates computed at all**. `motif_seq_recovery` is not a criterion; the column is `{seq}_correct_motif_sequence_all`, threshold 1.0. | BREAKING |
| 8-9 | Hydra `+` semantics inverted: "`+` would error on keys not already in the config". Actual: bare `key=` requires existence; `+key=` **adds and errors if it already exists**; `++key=` adds-or-overrides. | MISLEADING |
| 19, 21 | `ckpt_path` default `${oc.env:CKPT_PATH}` — all three local pipelines ship literal `./ckpts` (`search_binder_local_pipeline.yaml:22-24`) | MISLEADING |
| 24-25 | `gen_njobs`/`eval_njobs` default 2 — all three ship 1 | MISLEADING |
| 169, 173 | `compute_pre_refolding_metrics` / `compute_refolded_structure_metrics` "`true` (AME)" — `ame_evaluate.yaml:28, :36` both `false` | MISLEADING |
| 170-171, 174 | sub-toggles default `true` — all six are `false` | MISLEADING |
| 163, 165 | `num_redesign_seqs: 2` / `interface_cutoff: 8.0` presented as AME defaults — `ame_evaluate.yaml` defines neither, so for a ligand target the code defaults apply: `DEFAULT_NUM_REDESIGN_SEQS_LIGAND = 1`, `DEFAULT_INTERFACE_CUTOFF_LIGAND = 6.0` (`binder_eval_utils.py:49, :51`) | MISLEADING |
| 74, 92-93, 101 | `beam_search.save_intermediate_states`, `refinement.refine_targets`, `refinement.save_pre_refinement`, `refinement.loss_weights.*` presented as pipeline-agnostic — they exist only in `binder_generate.yaml`; on ligand/AME the `++` creates a dead key | MISLEADING |
| 194 | `result_type` enum omits `monomer_motif` (`analyze.py:137-144`) | STALE |

### `complexa-design/reference/pipelines.md`

| Line | Defect | Severity |
|---|---|---|
| 147, 167 | `configs/evaluate_motif_binder.yaml` → RC-6 | BREAKING |
| 85-88 | AME default thresholds given as `i_pAE*31<=10.0`, `pLDDT>=0.8`, `scRMSD<2.0`, `motif_rmsd_pred<1.5`, `motif_seq_recovery>=0.5`. `motif_binder_analysis_utils.py:72-89` (`DEFAULT_MOTIF_LIGAND_BINDER_SUCCESS`): binder side is **only** `scRMSD_bb3 <= 2.0`; motif side is `motif_rmsd_pred <= 1.5`, `correct_motif_sequence >= 1.0`, `has_ligand_clashes < 0.5`. No `i_pAE`, no `pLDDT`, and the ligand-clash criterion is undocumented. The doc matches the **stale comment** at `ame_analyze.yaml:27-29`, not the code that runs. | BREAKING |
| 89 | AME "pre/post-refolding interface metrics are enabled (bioinformatics, TMOL, HBPLUS)" — all four toggles `false`; HBPLUS isn't a key → RC-5 | MISLEADING |
| 28 | `gen_njobs`/`eval_njobs` 2/2 — all ship 1 | MISLEADING |
| 41 | "scRMSD < 1.5 Å" — the key is `scRMSD_ca`; writing `scRMSD` is what produces the broken override in RC-2 | STALE |

### `complexa-design/reference/troubleshooting.md`

| Line | Defect | Severity |
|---|---|---|
| 149-157 | `python -m atomworks rename_ligand --in … --target-resname L --target-resnum 0` — no such CLI (grep for `rename_ligand`: this file only). The real procedure is the Python snippet in `README.md:326-335`, which sets a single residue *name* string `"L:0"`, not a resname `L` + resnum `0`. `SKILL.md:222-228` gets it right; this file contradicts it. | BREAKING |
| 56-60 | "switch the eval backend to ESMFold" → RC-1 | BREAKING |
| 176 | `complexa validate` with overrides → RC-7 | BREAKING |
| 273-276, 281 | partial `success_thresholds` → RC-2 | BREAKING |
| 167-168 | Hydra `+`/`++` semantics inverted (same as `overrides.md:8-9`) | MISLEADING |
| 179 | "the validator surfaces every unknown key" → RC-7 | MISLEADING |
| 70-71 | "Ligand binder **and AME** bake `${oc.env:RF3_*}` into reward and refold paths" — `ame_generate.yaml:85` is `reward_model: null` with the RF3 block commented out (`:88-110`), and `ame_evaluate.yaml` has no `oc.env` at all. Only the ligand pipeline interpolates them (`ligand_binder_generate.yaml:82-83`). At refold time RF3 resolves via `os.environ.get(...)` with a fallback (`binder_eval.py:105-110`), so an `InterpolationKeyError` cannot arise from the evaluate stage of either pipeline. | MISLEADING |
| 263 | "`res_filter_*_pass_*.csv` shows `pass_rate: 0.0`" — no such column; emitted columns are `_res_{seq_type}_pass_rate_{filter_name}_{suffix}` | STALE |

### `complexa-target/SKILL.md` and `reference/target_schema.md`

| Line | Defect | Severity |
|---|---|---|
| SKILL:187 | `configs/search_protein_local_pipeline.yaml` does not exist (the skill's own table at :46 names the real file) | BREAKING |
| SKILL:28, 37, 185-197; schema:225 | `complexa validate target` can never resolve the dict → RC-7 | BREAKING |
| SKILL:163-172; schema:176-215 | ligand `target add` writes to the protein dict → RC-8 | BREAKING |
| SKILL:67; schema:16, 76, 80 | Single-element `binder_length: [100]` offered for **protein** targets. `binder_generate.yaml:24-25` indexes `binder_length[1]` with no `oc.select` guard, so it fails Hydra interpolation. (The ligand and AME configs *do* guard it — `ligand_binder_generate.yaml:23`, `ame_generate.yaml:24`.) | BREAKING |
| schema:78 | "`[]` or unset falls back to `[60, 120]`" — that default exists only in the CLI writer (`target_manager.py:1008-1011`). Hand-edited entries (the skill's *preferred* path) hit an interpolation error. | BREAKING |
| schema:206-219 | "Resulting entry … `SMILES: null`" — `target_manager.py:1024-1025` writes `SMILES` only `if smiles`. Absent key + unguarded interpolation at `ligand_binder_generate.yaml:33` = broken pipeline. | BREAKING |
| SKILL:76; schema:30 | `ligand_only` documented as "generate pocket around ligand only (no protein-protein interface)". `gen_dataset.py:620-621, :722-731`: it means *use the entire file as the ligand* (True) vs *extract residues by name* (False). A reader will set `true` on a whole-protein PDB and silently treat the complex as ligand. | MISLEADING |
| schema:29-32; SKILL:76-77 | Ligand keys listed as optional with defaults — `ligand_binder_generate.yaml:31-34` and `binder_generate.yaml:34-37` interpolate them unguarded, so omitting any is a hard error | MISLEADING |
| SKILL:15, 30, 35, 80 | "`complexa target` manages/sees both dicts"; `list -v --ligand` → RC-8 | MISLEADING |
| schema:217 | "`--ligand` requires a value" — `nargs="?"`, `const="YOUR_LIGAND"` (`cli_runner.py:1147-1153`); the bare flag is legal and marks the target ligand-typed | MISLEADING |
| SKILL:212 | "automatic `.yaml.bak` backup written by `save_targets_dict`" — new targets go through `append_target_to_dict` (`:372-404`), a plain append with no backup. `save_targets_dict` (`:245-265`) runs only on overwrite. | MISLEADING |
| SKILL:151-157; schema:150-169 | Flagship example is `complexa target add 02_PDL1` without `-f`. That name already exists (`targets_dict.yaml:11-18`), so `target_manager.py:1035-1044` blocks on `input()` and exits 1 non-interactively. The shown "resulting entry" also disagrees with the live one (`pdb_id: null`, plus a `target_path:` line). | MISLEADING |
| SKILL:52; schema:121-131 | AME schema wrong three ways: lists `pdb_id` (0/44 entries have it); omits `hotspot_residues` (44/44) and `target_path` (42/44); types `ligand` as a 3-letter `str` when `ame_dict_v2.yaml:46` uses a list (`["ADP","MG","3PG"]`) and `:36` uses `"L:0"` | MISLEADING |
| schema:130; SKILL:52 | AME `use_bonds_from_file` presented as the bond-sourcing knob — `ame_generate.yaml:33-36` builds `LigandFeatures` without it, so editing it has no effect | MISLEADING |
| schema:108-117 | AME name grammar `M{NNNN}_{pdb_id}` doesn't admit the real `M0024_1nzy_og`, `M0024_1nzy_v3`, `M0096_1chm_og` | MISLEADING |
| schema:129 | AME `binder_length` "`[100, 160]` range" — all 44 entries are `[180]`, a single value | STALE |
| schema:126 | AME `target_filename` example `1nzy_v2` — real value `M0024_1nzy_v2` (`:25`); as written it resolves to a missing PDB | STALE |
| SKILL:125 | "`-i / --editor` opens an editor and blocks" — two distinct flags; `--editor` alone with a name is discarded, `-i` alone blocks | MISLEADING |
| SKILL:127, 235 | "`target_cli.py` — argparse source of truth". `complexa target` is defined in `cli_runner.py:1147-1292`; `target_cli.py` backs the separate `complexa-target` console script | STALE |
| SKILL:98-101 | "Quote chain/residue ranges … that's what the on-disk dump produces" — on disk single-segment values are unquoted (`targets_dict.yaml:6, 15, 24, 33`); the template also omits the `target_path:` line every live entry has | STALE |
| schema:100 | "Common existing `source` directories" lists `custom_targets` (0 entries) and omits `ame_targets` (44, the most common) | STALE |

### `complexa-evaluate-pdbs/SKILL.md` and `reference/eval_configs.md`

| Line | Defect | Severity |
|---|---|---|
| SKILL:12 | Frontmatter says it wires `++dataset.pdb_dir`. `evaluate.py:732` reads `cfg.get("sample_storage_path")`; `pdb_dir` is not a config key. Because `++` creates it silently, the job runs against the config default `/path/to/samples/processed`. (The skill body correctly uses `++sample_storage_path` — the description contradicts it.) | BREAKING |
| SKILL:27, 44-47, 108, 196, 210; eval_configs:9, 10 | `esmfold` / `boltz2_default` / `protenix_*` → RC-1 | BREAKING |
| SKILL:77; eval_configs:12, 16, 66, 106 | `configs/evaluate_motif_binder.yaml` → RC-6 | BREAKING |
| SKILL:82-88; eval_configs (Example C) | `complexa analysis configs/evaluate_from_pdb_dir.yaml ++dataset.task_name=39_7V11_LIGAND` fails twice over: that config's only target source is `defaults: - generation/targets_dict@dataset` (`:22`) and `configs/generation/targets_dict.yaml` doesn't exist; and `39_7V11_LIGAND` lives in `ligand_targets_dict.yaml:2`, not `targets_dict.yaml`, so `get_target_info` raises (`binder_eval_utils.py:249-252`) | BREAKING |
| eval_configs:26 | "`generation/targets_dict@dataset` resolves against `configs/targets/targets_dict.yaml` (and ligand_targets_dict via task naming)" — the literal path doesn't exist, and nothing dispatches between dicts by task name (`binder_eval_utils.py:238-252` just looks up whatever was composed) | BREAKING |
| eval_configs:159 | `python scripts/rename_ligand_to_L0.py` — there is no `scripts/` directory (root has assets, community_models, configs, docs, env, licenses, script_utils, src) and no such file anywhere | BREAKING |
| eval_configs:118-124 | The `L:0` snippet drops the `[0]` from `README.md:331`. Without it the object is an AtomArrayStack, so the mask assignment doesn't do what's claimed and `to_pdb_file` gets the wrong type. | BREAKING |
| SKILL:200; eval_configs:36-37 | `hbplus` → RC-5 | STALE |
| SKILL:188 | manifest pins "the user-stated `result_type`" and the resolved config → RC-4 (no `result_type` field; `config`/`checkpoints` are `null`) | MISLEADING |
| SKILL:166 | "one row per input PDB × `sequence_types`" — sequence types are column *prefixes*; one row per `id_gen` (`binder_analysis_utils.py:157-166`) | MISLEADING |
| SKILL:164-168 | Output layout stale: `organize_results` (`analyze.py:2802-2866`) moves `res_filter_*`→`filter_results/`, `res_div_*`→`diversity/`, `res_monomer_*`→`monomer_metrics/`, `res_ss_*`→`secondary_structure/`, `res_aa_*`→`amino_acid_distribution/`, `clusters_*`→`clusters/`. The primary CSV (`RAW_{result_type}_results_{config_name}_combined.csv`) is never named. | MISLEADING |
| SKILL:74 | Protein-binder row claims `configs/evaluate_from_pdb_dir.yaml` defaults to `colabdesign`. As shipped it is `rf3_latest` (`:72`), `ligand_mpnn` (`:84`), `result_type: ligand_binder` (`:139`) — omit the overrides and you get a ligand-binder run. | MISLEADING |
| SKILL:72-77 | The "Analyze config" column lists configs no documented command uses. `complexa analysis` takes one config for both steps (`cli_runner.py:1021-1042`); passing `configs/analyze.yaml` to `complexa analyze` fails because it defines no `results_dir` (`analyze.py:2921`). | MISLEADING |
| eval_configs:87 | `analysis_modes` default `[motif_binder, binder, monomer]` for motif result types — `analyze.py:3065-3075` sets `["motif_binder"]` only | MISLEADING |
| eval_configs:151 | Example filter uses `mpnn_*` columns but the example above (`:143`) passes `sequence_types=[self,mpnn_fixed]` — no `mpnn_` columns exist. Also drops the `_all` suffix the thresholded columns carry. | MISLEADING |
| eval_configs:20 | "`dataset/motif_target_dict_cfg` override" — not a config group (`configs/dataset/` has only `unified/`); it's a *key* from `design_tasks/ame_dict_v2.yaml:11` | MISLEADING |
| eval_configs:128-129 | "If the PDBs came out of `complexa generate` with `protein_type=motif_binder`, the rename is already done" — `protein_type` appears only in *evaluate* configs; no generate config defines it. `README.md:319-320` also contradicts the guarantee. | MISLEADING |
| eval_configs:44 | `file_limit` listed as a field of `evaluate_from_pdb_dir.yaml` — it's in the AME/motif configs instead. (`++file_limit=N` still works via `cfg.get`, so drift only.) | STALE |

### `complexa-sweep/SKILL.md` and `reference/sweep_axes.md`

| Line | Defect | Severity |
|---|---|---|
| SKILL:100-114, 140 | Generated `eval_*.yaml` never invoked → RC-9. **The sweep produces no usable evaluation results.** | BREAKING |
| SKILL:140, 144 | `results_*.csv` glob matches nothing — real names are `{monomer,binder,motif,motif_binder}_results_{config_name}_{job_id}.csv` (`evaluate.py:853, 881, 904, 930`) and `RAW_..._combined.csv` (`analyze.py:3036`) | BREAKING |
| SKILL:147, 158-160; sweep_axes:145-148 | Fabricated columns `i_pae`/`i_plddt`/`sc_rmsd`/`binder_seq`/`passes_filter` and wrong thresholds → RC-9 | BREAKING |
| SKILL:29 | preflight path → RC-4 | BREAKING |
| sweep_axes:70 | `esmfold`/`boltz2_default` as a sweep axis → RC-1; every config in that axis crashes | BREAKING |
| sweep_axes:53 | Bioinformatics reward-weight axes — the whole `bioinformatics:` block is commented out (`binder_generate.yaml:165-188`), so the key is absent from the composed config. `generate_inference_configs.py:309-314` writes it anyway (struct mode re-enabled only at `:328`), so the axis silently has zero effect. | BREAKING |
| SKILL:173 | manifest pulls the Hydra config from the run dir → RC-4 | MISLEADING |
| SKILL:209; sweep_axes:110 | Error string "No configs were generated" doesn't exist. With an empty axis, `generate_inference_configs.py:450` logs `Generated 0 config pair(s)` and exits 0 — a silent no-op. | MISLEADING |
| sweep_axes:19 | `best_of_n.replicas` default 10 — `binder_generate.yaml:56-57` is 2 | STALE |
| sweep_axes:77 | `metric.pre_refolding.{...,hbplus}` "mixed" defaults — both real toggles are `false`; `hbplus` doesn't exist | STALE |
| SKILL:74 | Rule attributed to `load_sweeper_file` (`:108-131`, load/validate only); the override-wins collapse is in `build_sweeper` (`:160-162`) | STALE |

---

## Part 3 — What could not be verified

Not counted as defects. Mostly empirical numbers with no in-repo source.

- **All wall-clock, VRAM, and cost-multiplier figures.** `complexa-design/SKILL.md:234-235, :297, :302-307`; `_shared/reference/hardware.md` (every "(empirical)" row, plus the cost-multiplier table); `complexa-sweep/SKILL.md:199`; `sweep_axes.md:15-39, :70`; `complexa-evaluate-pdbs/SKILL.md:43, :211`. Nothing in the repo states these. Note the ESMFold rows are moot given RC-1.
- **Complexa checkpoint sizes** (~3 GB per variant, ~9 GB for `--complexa-all`). `download_startup.sh` prints no size for the NGC Complexa downloads and `./ckpts/` is empty in this checkout. Would need the NGC API.
- **NGC URL reachability.** The model slugs (`proteina_complexa`, `_ligand`, `_ame`) match `download_startup.sh:252, :296, :340` and `README.md:176-178`, but the URLs were not fetched.
- **Disk minimum "~50 GB".** No source; and it is ~2.5× the real download footprint (see the size defects above).
- **`atomworks` API surface.** The package isn't vendored. `README.md` and `complexa-design/SKILL.md:222-228` use `atomworks.io.load_any`, while the repo itself imports `atomworks.io.utils.io_utils.load_any` — I couldn't confirm which is the public re-export. The `target_input` contig grammar (`target_schema.md:44-46`) is parsed by `AtomSelectionStack.from_contig`, also external.
- **`USE_V2_COMPLEXA_ARCH` → checkpoint mapping** (`pipelines.md:122-126`). The injection mechanism is confirmed (`cli_runner.py:679`; `search_ame_local_pipeline.yaml:22`) and `proteina.py:26-57` confirms the flag selects the architecture and defaults to V1, but the specific v1/v2 ckpt table wasn't traced.
- **Qualitative tuning guidance** — `sweep_axes.md:16` ("diminishing returns past 8"), `:31`, `:35`, and the "Recommended sweep recipes" table.
- **Sweep aggregation artifacts** (`sweep_summary.csv`, `diversity_score`, the Pareto procedure, `./sweep_runs/<run>/`) are skill-invented with no repo counterpart, so only the column names inside them were checkable.

---

## Part 4 — Suggested triage order

1. **RC-2** (partial `success_thresholds`) — silently reports 100% success. Highest risk of a
   wrong scientific conclusion, and it's documented in `docs/` too, not just the skills.
2. **RC-1** (phantom folding backends) — check git history first; this looks like a code
   regression, and restoring the backends may be the right fix rather than editing 11 docs.
3. **RC-9** (sweep produces no evaluable output) — the sweep skill cannot work as written.
4. **RC-3** (`complexa init` / undocumented `COMPLEXA_INIT` gate) — blocks new users at step one.
5. **RC-4** (shared-script interfaces) — cheap, mechanical, affects every skill.
6. **RC-7, RC-8** (validate/target) — wrong-but-quiet; corrupts `targets_dict.yaml`.
7. **RC-10** (`evaluate_from_pdb_dir.yaml` cannot compose) — one-line fix in `configs/`,
   and it unblocks a whole entry point.
8. **RC-5, RC-6** (phantom `hbplus`; wrong path for `evaluate_motif_binder.yaml`) — deletions
   and repoints. RC-6 also needs fixing in `docs/INFERENCE.md:104` and
   `docs/EVALUATION_METRICS.md:82, :262`.
9. The per-file MISLEADING/STALE defaults — mechanical, do in one pass.

### A note on the eval suite

`.claude/skills/README.md` says these skills were built with `skill-creator` and validated by
"parallel eval — for each prompt, run a with-skill agent vs a baseline; grade against
objective assertions (uses correct flag? cites real override key? runs preflight?)."

Four of the five skills fail exactly those three assertions somewhere: wrong flag
(`--runtime`, `--esmfold`, `--boltz2`), unreal override keys (`hbplus`, `dataset.pdb_dir`,
`motif_rmsd_pred`), and a preflight whose output path is wrong in three of four skills. The
assertions were most likely checking that a key was *cited*, not that it *resolves*. A grader
that composes the Hydra config and asserts every `++key` exists — and that shells out
`--help` for every documented flag — would have caught roughly 40 of these 128 mechanically.
