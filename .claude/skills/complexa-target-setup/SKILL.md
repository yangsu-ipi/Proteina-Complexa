---
name: complexa-target-setup
description: >
  Getting a new design target from a raw PDB entry to a running Proteina-Complexa
  pipeline, and diagnosing the setup failures that do not announce themselves. Reach
  for this skill whenever the user reports "Error locating target", "collate_fn error",
  "Error locating target proteinfoundation.datasets.gen_dataset.collate_fn", or a Hydra
  error that names a symbol which clearly exists; mentions "CCD_MIRROR_PATH",
  "PDB_MIRROR_PATH", "atomworks env vars", "download the CCD", "CCD mirror",
  "PDB mirror", "components.cif"; asks for a "self-contained target directory",
  "one yaml per target", "one config file per target", "keep target data together",
  "absolute path for the target pdb", "override the target pdb path",
  "hydra searchpath", or "custom targets dict"; or asks to "clean my target PDB",
  "strip heteroatoms", "remove waters from my target", "why are my hotspots ignored",
  "hotspots not working", "my target came out empty", "renumber residues", or
  "cif vs pdb numbering". Also covers running Complexa from outside the repo —
  "missing required environment keys", "LOCAL_CODE_PATH not set", "CKPT_PATH empty",
  "missing checkpoint but the file is there", "slurm job can't find .env",
  "where does complexa look for .env", "sourced env.sh but it didn't work",
  "environment not initialized", "COMPLEXA_INIT", "conda env instead of uv",
  "complexa init has no conda option", batch/SLURM campaign directories —
  and why `complexa validate target` passes on a broken target. This is the only skill
  that owns `.env` discovery and the atomworks mirror environment variables,
  self-contained per-target directories, and target-PDB preparation for protein-binder
  work; for `.env` *key meanings* defer to `complexa-setup`, for the target *schema*
  defer to `complexa-target`, and for pipeline knobs defer to `complexa-design`.
compatibility: "complexa CLI installed (pip install -e .); atomworks + biotite importable; bash 4+; no GPU needed for setup/preflight"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Complexa Target Setup Skill

Target setup fails **silently** more often than it errors. Nearly every mistake here
produces a run that completes and writes PDBs — just not the ones the user wanted. The
job of this skill is to front-load the checks so a bad target is caught before a GPU-hour
is spent, and to decode the one error that is genuinely misleading.

Full reference lives in [`docs/binder-target-setup/`](../../../docs/binder-target-setup/) —
shared with the Codex entry point at `.codex/prompts/complexa-target-setup.md`, so keep
substantive content there rather than duplicating it here.

## What this skill enables

- Decoding `Error locating target '…collate_fn'` — a masked lazy-import failure.
- Getting the environment right from a batch job or campaign dir outside the repo.
- Setting the atomworks mirror vars, and building either mirror if genuinely wanted.
- Defining a target in **one self-contained YAML file**, plus three alternatives.
- Preflighting a target PDB: heteroatoms, numbering, gaps, hotspot resolution.
- Recognising the silent-failure modes, none of which raise.

## Step 1: Preflight

```bash
bash .claude/skills/_shared/scripts/preflight.sh
```

Read `./preflight.json` for GPU/VRAM, disk, checkpoints, and `.env` state. Then check the
two variables that `preflight.sh` does **not** cover and that `.env_example` never mentions:

```bash
env | grep -E 'CCD_MIRROR|PDB_MIRROR|LOCAL_MSA' || echo "unset (this is the good case)"
```

If any is set to a path that does not exist, go to Step 2 before anything else — it will
break generation in a way that looks unrelated.

## Step 1b: If running from outside the repo (SLURM, campaign dir)

`.env` discovery is inconsistent — three mechanisms that disagree. Read
[`env-and-mirrors.md`](../../../docs/binder-target-setup/env-and-mirrors.md#how-the-environment-is-discovered)
before debugging any batch failure. The short version:

```bash
set -a; source "$COMPLEXA_REPO/env.sh"; set +a     # bash only, not zsh/sh
```

`env.sh` resolves `.env` next to itself, so it is cwd-independent — but older generated
copies source `.env` without `set -a`, and `.env` has no `export` lines, so only
`_TOOL_VARS` and `COMPLEXA_INIT` reach child processes. The `set -a` wrapper fixes that
for any version; `complexa init <runtime> --force` regenerates a fixed one.

Assert before spending GPU time — the tell-tale failure is tool binaries resolving while
every path is empty:

```bash
for k in LOCAL_CODE_PATH LOCAL_DATA_PATH CKPT_PATH DATA_PATH AF2_DIR ESM_DIR COMPLEXA_INIT; do
    printf '%-18s %s\n' "$k" "${!k:-<UNSET>}"
done
```

If the user reports `missing required environment keys: ['LOCAL_CODE_PATH',
'LOCAL_DATA_PATH', 'CKPT_PATH']` alongside missing checkpoints and `AF2_DIR`/`ESM_DIR`,
that is this one bug, not five — and the pipeline itself would have run fine, because the
stage modules find `.env` by walking up from their own module file.

**conda installs use the `uv` label.** `complexa init` accepts only `uv` or `docker`
(`cli_runner.py:1160-1166`); nothing branches on the value of `COMPLEXA_INIT`
(`cli_runner.py:2011` tests presence only), so point `UV_VENV` and the `UV_*` tool vars at
the conda prefix and use `uv`. Do not tell the user conda is unsupported.

## Step 2: Fix the atomworks env vars

Set both empty unless real mirrors exist on disk. Empty is the intended default and costs
almost nothing: `get_available_ccd_codes` unions the mirror with biotite's complete
built-in CCD (`atomworks/io/utils/ccd.py:184-186`), and `PDB_MIRROR_PATH` is read only by
training-time dataset parsers (`datasets/atomworks_default_metadata_row_parsers.py:106`).

```bash
export CCD_MIRROR_PATH= PDB_MIRROR_PATH=
```

A non-empty path to a missing directory crashes at **import** time, because
`datasets/atomworks_ligand_transforms.py:28` resolves the CCD set as a module-level
constant. The repo's guards (`cli/startup.py:121-124`,
`patches/atomworks_patches.py:15-20`) only default these when **unset** — a wrong value
sails past both. All-or-nothing: empty works, a valid directory works, nothing in between.

Find where it is set and fix it at source —
`grep -rn CCD_MIRROR_PATH .env env.sh ~/.bashrc "$CONDA_PREFIX"/etc/conda/activate.d/`.

To build a mirror instead, see
[`env-and-mirrors.md`](../../../docs/binder-target-setup/env-and-mirrors.md) —
`scripts/build_ccd_mirror.py` for CCD, rsync for PDB.

## Step 3: Author the target YAML

**Default to one file per target**, in the target's own directory, outside the repo
(`.gitignore` ignores `*.pdb` but not nested `inference/`, so a repo-internal target
directory silently untracks your input and tracks your outputs).

```yaml
# /data/targets/MINE/pipeline.yaml
defaults:
  - pipeline/binder/binder_generate@generation
  - pipeline/binder/binder_evaluate@_global_
  - pipeline/binder/binder_analyze@_global_
  - _self_

run_name: mytarget
ckpt_path: /data/shared/tools/Proteina-Complexa/ckpts
ckpt_name: complexa.ckpt
autoencoder_ckpt_path: /data/shared/tools/Proteina-Complexa/ckpts/complexa_ae.ckpt
ncpus_: 24
seed: 5
gen_njobs: 1
eval_njobs: 1

hydra:
  searchpath:
    - file:///data/shared/tools/Proteina-Complexa/configs
  run:
    dir: ./logs/hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

generation:
  task_name: 99_MYTARGET
  target_dict_cfg:
    99_MYTARGET:
      source: local              # required even though unused
      target_filename: mine      # required even though unused
      target_path: /data/targets/MINE/mine.pdb
      target_input: A1-115       # derive from the file — Step 4
      hotspot_residues: ["A37", "A39", "A49", "A98"]
      binder_length: [64, 155]
      pdb_id: null
```

Two traps to state to the user up front:

1. **Pin `generation.task_name` — omitting it fails silently.**
   `configs/pipeline/binder/binder_generate.yaml:16` defaults to `33_TrkA`, and because
   inline is a *merge*, `33_TrkA` survives from the shared dict. Verified result: a clean
   `RESULT: PASS` run against `1www_cropped.pdb`, chain X, hotspots `X294 X296 X333` — the
   user's target is loaded and never used. (Under the shadow approach the same omission
   raises `InterpolationKeyError: target_dict_cfg.33_TrkA.source` instead.)
2. **All seven keys, including `source` and `target_filename`.** OmegaConf evaluates the
   `oc.select` *default argument* regardless of whether the primary key resolves
   (`binder_generate.yaml:33`), so omitting `source` raises even with `target_path` set.
   Values are arbitrary.

`hydra.searchpath` is what lets this thin external file reach `pipeline/binder/*` in the
install. The config's own directory becomes Hydra's primary config dir
(`cli/cli_runner.py:646`).

Alternatives — full comparison in
[`target-config.md`](../../../docs/binder-target-setup/target-config.md):

| Approach | Semantics | Failure mode | Use when |
|---|---|---|---|
| **Inline in `_self_`** | merge, 45 targets | loud for a missing key on a **new** name; silent if `task_name` resolves to a shipped target | **default** |
| `targets/targets_dict.yaml` shadow | replace, 1 target | **silent** fallback to 44 when overriding a shipped name | pipeline must not see other targets |
| Own defaults entry (`- /my_specs/x@generation`) | merge, free naming | same as inline | custom layout |
| `++target_dict_cfg.<task>.target_path=…` | merge, per-invocation | silent if the key is typo'd | one-off |

Whichever you use, verify with the entry count in Step 5 rather than trusting a clean exit.

For the target schema fields themselves, defer to "Protein target schema" in
[`complexa-target/reference/target_schema.md`](../complexa-target/reference/target_schema.md) —
do not restate it.

## Step 4: Prepare and check the PDB

Never copy a `target_input` range or hotspot list from a bundled example — those are
correct for the *cropped* files that ship with the repo. Derive both from the file you will
actually feed in:

```bash
python docs/binder-target-setup/scripts/check_target_pdb.py \
    --pdb /data/targets/MINE/mine.pdb --chain A \
    --hotspots A37 A39 A49 A98 \
    --write-clean /data/targets/MINE/mine_clean.pdb
```

Exits non-zero on an unmatched hotspot. Require `hotspots MISS` empty and `IN-RANGE HET`
absent before running. Paste the printed `target_input` into the config.

Three things worth telling the user:

- The contig **crops for you** (`utils/pdb_utils.py:550-556`), so a raw download is usable
  — but `from_contig` filters on `(chain_id, res_id)` only, with no hetero filter
  (`atomworks/io/utils/selection.py:482-493`), so in-range waters and ions become protein
  residues.
- Hotspots are matched as `f"{chain_id}{res_id}"` strings and **misses are silent**
  (`utils/pdb_utils.py:571-575`).
- `.cif` and `.pdb` numbering differ — `load_any` sets `use_author_fields=False`
  (`atomworks/io/utils/io_utils.py:290`), so CIF gives `label_seq_id` and PDB gives author
  numbering. Choosing a format is a renumbering decision.

`complexa validate target` will not catch any of this: it checks the file exists and echoes
your config back, never opening the PDB (`cli/validate.py:379-503`). Do not present it as a
check that passed.

AME targets have a genuine mandatory checklist — defer to "Preparing AME Target PDBs" in
[`assets/target_data/README.md`](../../../assets/target_data/README.md) and the `L:0`
section of
[`complexa-design/reference/troubleshooting.md`](../complexa-design/reference/troubleshooting.md).
Details and the ligand-binder case: [`pdb-prep.md`](../../../docs/binder-target-setup/pdb-prep.md).

## Step 5: Run

```bash
cd /data/targets/MINE && complexa design ./pipeline.yaml --verbose
```

The `cd` matters: output roots are hardcoded relative paths — `./inference/...`
(`generate.py:66`) and `./logs` (`cli_runner.py:128`) — with no `os.chdir` or `cwd=`
anywhere, so they land in the shell's cwd.

Confirm the right target dict is live before letting a long run proceed:

```bash
grep -o "'target_dict_cfg': '<filtered: [0-9]* entries>'" logs/generate.log
```

`1` = shadow took, `45` = inline/merge worked, **`44` = nothing took**.

Shards and binder refolds are reused by default (`generation.skip_completed_shards`,
`metric.reuse_cached_folding`). Before a long campaign leans on that, confirm resume
*invalidates and refuses* as well as reuses — `bash
docs/binder-target-setup/scripts/check_resume.sh --config ./pipeline.yaml --samples 2`,
six checks, needs a GPU. Note what a **changed generation config** does: generation
aborts rather than regenerating, because the directory names are deterministic and the
counters restart, so continuing would overwrite the earlier run's structures. Pick a new
`generation.run_name` or clear the directory — the message says which.

## Step 6: Emit manifest

```bash
python3 .claude/skills/_shared/scripts/write_manifest.py \
    --output-dir ./inference/<run_dir> \
    --command "complexa design ./pipeline.yaml" \
    --skill complexa-target-setup \
    --out ./run_manifest.json
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `missing required environment keys: ['LOCAL_CODE_PATH', …]` after `env.sh` reported success | `.env` sourced without `set -a`; only `_TOOL_VARS` exported | `set -a; source env.sh; set +a`, or `complexa init <runtime> --force` |
| `Environment not initialized. Run: complexa init` | `COMPLEXA_INIT` unset — `env.sh` not sourced (`cli_runner.py:2004-2016`) | source `env.sh` from **bash** |
| `missing checkpoint` but the file exists; path ends `checkpoints/` | `LOCAL_CHECKPOINT_PATH` in an existing `.env` still says `checkpoints/`; downloaders write `ckpts/` (`download_startup.sh:239`) — usually gate-only if the pipeline YAML sets `ckpt_path` absolutely | `LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts`, then re-init |
| `missing community model path: ESM_DIR` | ESM2 genuinely absent; not gate-only, `compute_esm_metrics: true` (`binder_evaluate.yaml:47`) | `complexa download --esm2` on a login node, or `++metric.compute_esm_metrics=false` |
| `Error locating target '…collate_fn'` | masked lazy-import failure, usually invalid `CCD_MIRROR_PATH` | `python -c "import proteinfoundation.datasets.gen_dataset"`, or re-run with `HYDRA_FULL_ERROR=1` |
| **Clean run, wrong target** (`1www_cropped.pdb`, chain X) | `generation.task_name` unpinned → inherited `33_TrkA`, which exists in the shared 44 so nothing errors | pin it in `_self_`; check `task_name` + `pdb_path` in the log |
| `InterpolationKeyError: …33_TrkA.source` | same omission, but with a *replaced* dict | pin it in `_self_` |
| `InterpolationKeyError: …<NAME>.source` | entry omits `source`/`target_filename` | add both; values arbitrary |
| `Could not override 'targets@generation'` | `@_here_` makes the entry unaddressable | use the exact shadow path, your own defaults entry, or inline |
| Designs ignore the epitope | hotspots silently dropped | `check_target_pdb.py`; fix numbering |
| Target empty or truncated | `target_input` doesn't match author numbering | derive it from the file |
| 44 entries when 1 or 45 expected | shadow filename wrong | exact `targets/targets_dict.yaml` |
| Target PDB not in git | `*.pdb` globally ignored | keep target dirs outside the repo |
| `import atomworks` fails, build "passed" | `env/build_uv_env.sh:174` uses `\|\| echo` | reinstall and read the real error |

## Reference

Setup and preflight need no GPU; for per-pipeline VRAM and wall-clock see
[`_shared/reference/hardware.md`](../_shared/reference/hardware.md).

Full detail lives in [`docs/binder-target-setup/`](../../../docs/binder-target-setup/README.md),
whose README indexes all five files. Read
[`campaign-gating.md`](../../../docs/binder-target-setup/campaign-gating.md) **before writing a
campaign gate, runner, or job file** — it owns shard sizing and resume — and
[`target-config.md`](../../../docs/binder-target-setup/target-config.md) before authoring a
target YAML.
