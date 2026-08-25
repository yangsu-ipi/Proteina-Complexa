# Complexa target setup

Get a new Proteina-Complexa design target from a raw PDB entry to a running pipeline, and
diagnose the setup failures that do not announce themselves.

Use this when the request involves: `Error locating target` / `collate_fn` Hydra errors;
`CCD_MIRROR_PATH` or `PDB_MIRROR_PATH`; downloading the CCD or a PDB mirror; a
self-contained or one-YAML-per-target layout; overriding the target PDB path;
`hydra.searchpath`; cleaning a target PDB or stripping heteroatoms; hotspots being ignored;
an empty or truncated target; `.cif` vs `.pdb` residue numbering; or running Complexa from
outside the repo — `missing required environment keys`, `LOCAL_CODE_PATH` / `CKPT_PATH`
empty, "missing checkpoint" when the file exists, `COMPLEXA_INIT`, "sourced env.sh but it
didn't work", and SLURM/batch campaign directories.

**Full reference: `docs/binder-target-setup/`** — the same files the Claude skill at
`.claude/skills/complexa-target-setup/SKILL.md` uses. Read the relevant one rather than
working from this summary alone.

| File | Contents |
|---|---|
| `docs/binder-target-setup/README.md` | index + the silent-failure table |
| `docs/binder-target-setup/env-and-mirrors.md` | `.env` discovery, mirror env vars, layouts, build recipes |
| `docs/binder-target-setup/target-config.md` | one-file target YAML, alternatives, Hydra mechanics |
| `docs/binder-target-setup/pdb-prep.md` | cleaning by pipeline, numbering, `.cif` vs `.pdb` |
| `docs/binder-target-setup/campaign-gating.md` | **read before writing a campaign gate, runner, or job file** — owns shard sizing and resume |
| `docs/binder-target-setup/troubleshooting.md` | masked imports, full failure catalogue |

**If this task involves generating or refreshing campaign scaffolding, read
`campaign-gating.md` — not just the env sections.** A gate that hardcodes its required
models or a fixed disk floor will pass while the run is broken and fail while it is fine, and
`gen_njobs` chosen for throughput alone makes generation effectively non-resumable.

## The governing fact

This part of the pipeline **fails silently more often than it errors**. Almost every
mistake produces a run that completes and writes PDBs — just not the ones the user wanted.
Do not report success on the basis of a clean exit. And do not present
`complexa validate target` as a check that passed: it confirms the PDB *exists*, then
echoes the config back without ever opening the file
(`src/proteinfoundation/cli/validate.py:379-503`).

## Step 1 — Preflight

```bash
bash .claude/skills/_shared/scripts/preflight.sh     # writes ./preflight.json
env | grep -E 'CCD_MIRROR|PDB_MIRROR|LOCAL_MSA' || echo "unset (good)"
```

## Step 1b — Running from outside the repo (SLURM, campaign dir)

`.env` discovery uses three inconsistent mechanisms; read
`docs/binder-target-setup/env-and-mirrors.md` ("How the environment is discovered") before
debugging any batch failure. Short version:

```bash
set -a; source "$COMPLEXA_REPO/env.sh"; set +a     # bash only, not zsh/sh
```

`env.sh` resolves `.env` next to itself so it is cwd-independent, but older generated copies
source `.env` without `set -a`, and `.env` has no `export` lines — so only `_TOOL_VARS` and
`COMPLEXA_INIT` reach child processes. The `set -a` wrapper fixes any version;
`complexa init <runtime> --force` regenerates a fixed one.

Assert before spending GPU time:

```bash
for k in LOCAL_CODE_PATH LOCAL_DATA_PATH CKPT_PATH DATA_PATH AF2_DIR ESM_DIR COMPLEXA_INIT; do
    printf '%-18s %s\n' "$k" "${!k:-<UNSET>}"
done
```

`missing required environment keys: ['LOCAL_CODE_PATH', 'LOCAL_DATA_PATH', 'CKPT_PATH']`
together with missing checkpoints and `AF2_DIR`/`ESM_DIR` — while `foldseek`/`mmseqs`
resolve — is this one bug, not five. The pipeline itself would have run: the stage modules
find `.env` by walking up from their own module file.

**conda installs use the `uv` label.** `complexa init` accepts only `uv` or `docker`
(`cli_runner.py:1160-1166`), and nothing branches on the value of `COMPLEXA_INIT`
(`cli_runner.py:2011` tests presence only). Point `UV_VENV` and the `UV_*` tool vars at the
conda prefix and use `uv`. Conda is not unsupported — just unlabelled.

## Step 2 — Fix the atomworks env vars

Set both empty unless real mirrors exist on disk:

```bash
export CCD_MIRROR_PATH= PDB_MIRROR_PATH=
```

Empty is the intended default. `get_available_ccd_codes` unions the mirror with biotite's
complete built-in CCD (`atomworks/io/utils/ccd.py:184-186`), and `PDB_MIRROR_PATH` is read
only by training-time parsers (`datasets/atomworks_default_metadata_row_parsers.py:106`).

A non-empty path to a **missing** directory crashes at import time
(`datasets/atomworks_ligand_transforms.py:28` resolves the CCD set as a module-level
constant). The repo's two guards (`cli/startup.py:121-124`,
`patches/atomworks_patches.py:15-20`) only default these when *unset*, so a wrong value
passes straight through. To build a mirror instead, see `env-and-mirrors.md`.

## Step 3 — Author the target YAML

**Default to one file per target**, in the target's own directory, **outside the repo** —
`.gitignore` ignores `*.pdb` but not a nested `inference/`, so a repo-internal target
directory silently untracks the input and tracks the outputs.

Template and the three alternatives (shadow file, own defaults entry, `++` override) are in
`docs/binder-target-setup/target-config.md`. Two traps to surface explicitly:

1. **Pin `generation.task_name` — omitting it fails silently.**
   `configs/pipeline/binder/binder_generate.yaml:16` defaults to `33_TrkA`, and because
   the inline approach *merges*, `33_TrkA` survives from the shared dict. Verified result:
   a clean `RESULT: PASS` run against `1www_cropped.pdb`, chain X, hotspots
   `X294 X296 X333` — the user's target is loaded into the config and never used. (Under
   the shadow approach the same omission raises
   `InterpolationKeyError: target_dict_cfg.33_TrkA.source` instead.)
2. **Include all seven keys, `source` and `target_filename` included**, even though
   `target_path` makes them redundant. OmegaConf evaluates the `oc.select` *default
   argument* whether or not the primary key resolves (`binder_generate.yaml:33`). Values
   are arbitrary.

For the target schema fields themselves, read
`.claude/skills/complexa-target/reference/target_schema.md` ("Protein target schema") —
do not restate it.

## Step 4 — Prepare and check the PDB

Never copy `target_input` or a hotspot list from a bundled example; those match the
*cropped* files shipped in `assets/target_data/`. Derive both from the actual file:

```bash
python docs/binder-target-setup/scripts/check_target_pdb.py \
    --pdb /data/targets/MINE/mine.pdb --chain A \
    --hotspots A37 A39 A49 A98 \
    --write-clean /data/targets/MINE/mine_clean.pdb
```

Exits non-zero on an unmatched hotspot. Require `hotspots MISS` empty and `IN-RANGE HET`
absent. Paste the printed `target_input` into the config.

- The contig crops for you (`utils/pdb_utils.py:550-556`), so a raw download is usable —
  but `from_contig` filters on `(chain_id, res_id)` only, no hetero filter
  (`atomworks/io/utils/selection.py:482-493`), so in-range waters and ions get encoded as
  protein.
- Hotspot misses are **silent** (`utils/pdb_utils.py:571-575`).
- `.cif` yields `label_seq_id`, `.pdb` yields author numbering, because `load_any` sets
  `use_author_fields=False` (`atomworks/io/utils/io_utils.py:290`).

AME targets have a real mandatory checklist — see "Preparing AME Target PDBs" in
`assets/target_data/README.md`. Do not re-derive it.

## Step 5 — Run and confirm

```bash
cd /data/targets/MINE && complexa design ./pipeline.yaml --verbose
```

The `cd` matters: `./inference/...` (`generate.py:66`) and `./logs` (`cli_runner.py:128`)
are hardcoded relative paths and there is no `os.chdir` or `cwd=` anywhere.

Before letting a long run proceed, confirm the intended target dict is live:

```bash
grep -o "'target_dict_cfg': '<filtered: [0-9]* entries>'" logs/generate.log
```

`1` = shadow file took, `45` = inline/merge worked, **`44` = nothing took** — you are
running the shared dict and probably the wrong PDB.

## Step 5b — Verify resume before relying on it

Shard skipping and binder fold reuse are on by default. Confirm they invalidate correctly:

```bash
bash docs/binder-target-setup/scripts/check_resume.sh --config ./pipeline.yaml --samples 2
```

## Step 6 — Emit a manifest

```bash
python3 .claude/skills/_shared/scripts/write_manifest.py \
    --output-dir ./inference/<run_dir> \
    --command "complexa design ./pipeline.yaml" \
    --skill complexa-target-setup --out ./run_manifest.json
```

## Fast diagnosis

| Symptom | Cause | Fix |
|---|---|---|
| `missing required environment keys: ['LOCAL_CODE_PATH', …]` after `env.sh` succeeded | `.env` sourced without `set -a`; only `_TOOL_VARS` exported | `set -a; source env.sh; set +a` |
| `Environment not initialized. Run: complexa init` | `COMPLEXA_INIT` unset (`cli_runner.py:2004-2016`) | source `env.sh` from **bash** |
| `missing checkpoint` but the file exists; path ends `checkpoints/` | existing `.env` still says `checkpoints/`, downloaders write `ckpts/` (`download_startup.sh:239`); gate-only if the pipeline YAML sets `ckpt_path` absolutely | `LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts`, re-init |
| `missing community model path: ESM_DIR` | ESM2 absent; not gate-only (`binder_evaluate.yaml:47`) | `complexa download --esm2`, or `++metric.compute_esm_metrics=false` |
| `Error locating target '…collate_fn'` | masked lazy-import failure, usually a bad `CCD_MIRROR_PATH` | `python -c "import proteinfoundation.datasets.gen_dataset"` or `HYDRA_FULL_ERROR=1` |
| **Clean run, wrong target** (`1www_cropped.pdb`, chain X) | `task_name` unpinned → inherited `33_TrkA`, which exists in the shared 44 so nothing errors | pin it under `_self_`; check `task_name` + `pdb_path` in the log |
| `InterpolationKeyError: …33_TrkA.source` | same omission, but with a *replaced* dict | pin it under `_self_` |
| `InterpolationKeyError: …<NAME>.source` | missing `source`/`target_filename` | add both |
| `Could not override 'targets@generation'` | `@_here_` makes the entry unaddressable | exact shadow path, own defaults entry, or inline |
| Designs ignore the epitope | hotspots silently dropped | `check_target_pdb.py` |
| Target empty or truncated | `target_input` vs author numbering | derive from the file |
| 44 entries when 1 or 45 expected | shadow filename wrong | exact `targets/targets_dict.yaml` |
| `import atomworks` fails, build "passed" | `env/build_uv_env.sh:174` uses `\|\| echo` | reinstall, read the real error |

Out of scope: search algorithms, reward weights, success thresholds, OOM — those belong to
`.claude/skills/complexa-design/` and `docs/INFERENCE.md`.
