---
name: complexa-setup
description: >
  First-time setup, environment configuration, and model-weight installation for
  Proteina-Complexa. Reach for this skill whenever the user says "set up complexa",
  "install complexa", "configure my .env", "first-time setup", "what models do I
  have installed", "what's in my .env", "download model weights", "download
  Complexa / AF2 / RF3 / ProteinMPNN / LigandMPNN / ESM2 / ESMFold checkpoints",
  "preflight my GPU", "verify environment", "complexa init", "complexa download",
  "complexa download --status", "complexa validate env", or any time a fresh
  checkout needs to be made runnable. This is the first skill to run on a new
  clone — it drives `complexa init`, `complexa download`, and `complexa validate
  env` end-to-end, edits the required `.env` keys, picks the right runtime (UV
  vs Docker), and emits a replayable setup artifact.
compatibility: "complexa CLI installed (pip install -e .); bash 4+; nvidia-smi optional"
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# Complexa Setup Skill

Drive the three steps a fresh Proteina-Complexa checkout needs before any
design run: create `.env`, fetch model weights, and sanity-check the env.
Probe the host for GPU / disk / tool binaries first so the user does not
discover a missing dependency mid-pipeline. End with a JSON setup artifact the
user (or a future agent) can re-read instead of re-deriving state.

## CLI vs direct file-edit — pick the cheapest path per step

| Step | Preferred path | Why |
|---|---|---|
| `.env` creation (Step 2) | **CLI** (`complexa init`, then `complexa init <uv\|docker>`) | Two phases in `handle_init` (`cli_runner.py`): the first copies `.env_example` → `.env`, the second writes an `env.sh` that exports `COMPLEXA_INIT`. Every later `design`/`generate`/`evaluate`/`analyze` command hard-exits without it, so a hand-rolled `cp` is *not* equivalent. |
| `.env` value edits (Step 3) | **File edit** (StrReplace `LOCAL_CODE_PATH=…` etc.) | No CLI for this — the values are user-specific paths. |
| Download model weights (Step 4) | **CLI** (`complexa download --…`) | Dispatches to `env/download_startup.sh` (~900 lines of bash with NGC URLs, retries, checksum-style skip-if-present). Don't try to replicate. |
| Validate env (Step 5) | **CLI** (`complexa validate env`) or `test -f .env && test -d $DATA_PATH` | CLI prints a nicer report; the manual check is one-liner-safe. |
| Validate full design config (after picking a pipeline) | **CLI** (`complexa validate design CONFIG`) | Non-trivial Hydra defaults traversal + ckpt + env-var checks; not worth replicating. |

## What this skill enables

- A correctly-shaped `.env` for either UV or Docker runtime.
- Model checkpoints (Complexa protein/ligand/AME plus community models) downloaded to known paths.
- A `preflight.json` snapshot of the host (GPU, disk, .env, ckpts, tool binaries).
- A `run_manifest.json` capturing the `complexa init` + `complexa download` invocation as one command string, plus skill name, timestamp, and git SHA (replay-friendly).
- A pass/fail report from `complexa validate env` with clear next-step hints.

## Step 1: Pre-flight check

Always run the shared preflight before touching the environment. It does not
require `.env` to exist — it falls back to defaults — and it tells you whether
the host can run Complexa at all.

```bash
bash .claude/skills/_shared/scripts/preflight.sh
```

The script writes `./preflight.json` (pass `--out PATH` to put it elsewhere).
Read it and surface:

- `gpu.available` — if `false`, design / evaluate steps will fail; warn the user.
- `gpu.vram_gb` — Complexa needs ≥24 GB; 40–80 GB recommended (A100/H100/L40S).
- `disk.free_gb` at `CKPT_PATH` — the full Complexa + community model set is ≈20 GB of
  weights; budget ~50 GB to leave room for tar extraction and run outputs.
- `env.missing_required` — anything listed here must be edited in `.env` before validation passes.
- `tools.{foldseek,mmseqs,dssp,hbplus,sc}.exists` — missing tools degrade evaluation but do not block generation.

## Step 1b: Build the Python environment (only if `.venv/` is missing)

The `complexa` CLI is installed inside the project's Python environment, not on
the system path by default. On a **fresh clone**, the `.venv/` directory does
not yet exist and `complexa init` will fail with `command not found`. Build the
UV venv before anything else:

```bash
test -d .venv || ./env/build_uv_env.sh   # first-time UV build
source .venv/bin/activate
which complexa                            # sanity check: should point inside .venv
```

Skip this step if `which complexa` already resolves — that means a previous
build is still good. The Docker runtime skips it entirely; the venv lives
inside the container image instead. If the user said "I just cloned" or you
see no `.venv/` next to `pyproject.toml`, run the build script — `complexa init`
without a venv produces a confusing `command not found` rather than an obvious
"build the venv first" error.

## Step 2: Create `.env`

Pick the runtime. UV is the default and faster to start; Docker is required on
Ubuntu 20.04 or systems with GLIBC mismatches.

Use AskUserQuestion if it is not obvious from context:

> "Which runtime do you want to configure? `uv` (recommended, faster) or `docker` (use if you do not have a UV venv built locally)?"

### Path A: CLI (recommended — this is the only path that produces `env.sh`)

`complexa init` runs in two phases (`handle_init` in `cli_runner.py`), and the
runtime is a **positional** argument, not a flag:

```bash
complexa init                    # Phase 1: copy .env_example → .env, then stop
#   ... now do the Step 3 edits to .env ...
complexa init uv                 # Phase 2: write env.sh for the UV runtime
complexa init docker             # Phase 2: write env.sh for the Docker runtime
source env.sh                    # REQUIRED — exports COMPLEXA_INIT
complexa init docker --force     # Overwrite an existing env.sh
```

`complexa init --runtime docker` exits with `unrecognized arguments` — there is
no `--runtime` flag.

Phase 2 never touches `.env`. It writes a separate `env.sh` that sources `.env`,
repoints the six tool vars (`FOLDSEEK_EXEC`, `RF3_EXEC_PATH`, `SC_EXEC`,
`MMSEQS_EXEC`, `DSSP_EXEC`, `TMOL_PATH`) at the chosen runtime's `UV_*` /
`DOCKER_*` values, and exports `COMPLEXA_INIT="<runtime>"`. For `docker` it also
overrides `LOCAL_CODE_PATH`, `COMMUNITY_MODELS_PATH`, `LOCAL_CACHE_DIR`,
`CKPT_PATH`, and `DATA_PATH` with the container paths. Your Step 3 edits survive
runtime flips because `.env` is only ever read, never rewritten.

**`source env.sh` is not optional.** `_check_complexa_init` (`cli_runner.py`)
hard-exits `design`, `generate`, `filter`, `evaluate`, `analyze`, `analysis`, and
`target` with "Environment not initialized" unless `COMPLEXA_INIT` is in the
environment. Only `init`, `demo`, `download`, `validate`, and `status` are exempt.

### Path B: file edit (covers Phase 1 only)

`cp .env_example .env` is a fine substitute for Phase 1 — but you must still run
`complexa init <uv|docker>` and `source env.sh` afterwards, because nothing else
generates `env.sh`. There is no all-file-edit path.

Once `.env` exists, re-running `complexa init` with no runtime positional just
prints the usage block and `sys.exit(1)`; it swaps nothing. `--force` on its own
re-copies `.env_example` over `.env`, dropping your edits.

### Verify either way

```bash
test -f .env   && echo "OK: .env present"   || echo "MISSING: run complexa init"
test -f env.sh && echo "OK: env.sh present" || echo "MISSING: run complexa init <uv|docker>"
source env.sh && echo "COMPLEXA_INIT=$COMPLEXA_INIT"
```

## Step 3: Edit .env

No CLI for this — Step 2 Phase 1 only copied the template; you still need to
write your machine-specific paths into `.env` by hand (StrReplace or your
editor), before running Phase 2. The two absolutely-required edits are:

```bash
LOCAL_CODE_PATH=/absolute/path/to/Proteina-Complexa
LOCAL_DATA_PATH=/absolute/path/to/PFM_data
```

Everything else (cache, ckpts, community-model dirs, tool binaries) is derived
from `LOCAL_CODE_PATH` by default and only needs editing if you have a
non-standard layout. For the full table — every key, what it controls, what
fails if it is missing — see [reference/env_keys.md](reference/env_keys.md).

Quick decision table for the four edits most users make:

| Key | Default | Set this if |
|-----|---------|-------------|
| `LOCAL_CODE_PATH` | placeholder | Always — required |
| `LOCAL_DATA_PATH` | `/path/to/PFM_data` | Always — required, points at target PDBs |
| `HF_TOKEN` | placeholder | You need ESMFold or gated HF models |
| `WANDB_API_KEY` | placeholder | You want training runs logged to W&B |

## Step 4: Download checkpoints

**Always use the CLI here.** `complexa download` dispatches to
`env/download_startup.sh` — NGC URLs, retries, and skip-if-present logic across
5 community-model families. A hand-rolled wget loop gets paths wrong.

Ask which models are actually needed — `--everything` is ≈20 GB. Each Complexa
variant unlocks exactly one pipeline; AF2 / RF3 in the community set are what
`evaluate` and reward-guided search need at run time.

| Flag | What it downloads | Unlocks pipeline | Destination | Approx size |
|------|-------------------|------------------|-------------|-------------|
| `--complexa` | Complexa protein-binder model + AE (`complexa.ckpt`, `complexa_ae.ckpt`) | **Protein binder** (default) — `configs/search_binder_local_pipeline.yaml` | `./ckpts/` | ~3 GB |
| `--complexa-ligand` | Ligand-binder model + AE (`complexa_ligand.ckpt`, `complexa_ligand_ae.ckpt`) | Ligand binder — `configs/search_ligand_binder_local_pipeline.yaml` | `./ckpts/` | ~3 GB |
| `--complexa-ame` | AME motif-scaffolding model + AE (`complexa_ame.ckpt`, `complexa_ame_ae.ckpt`) | AME (enzyme) — `configs/search_ame_local_pipeline.yaml` | `./ckpts/` | ~3 GB |
| `--complexa-all` | All three Complexa variants | All three pipelines | `./ckpts/` | ~9 GB |
| `--all` | The 5 community models: ProteinMPNN + LigandMPNN + AF2 + ESM2 + RF3 (no ESMFold — it has no download function here) | Needed by **evaluate / reward**: AF2 (protein binder), RF3 (ligand binder + AME), MPNNs (inverse folding for every pipeline). | `./community_models/` | ≈10.7 GB |
| `--everything` | All 3 Complexa variants + the same 5 community models. Nothing else — the script has no Boltz2 or Protenix downloader. | All three pipelines, plus evaluate / reward | both | ≈20 GB |
| `--status` | Show install state — does not download | (none) | (none) | n/a |

**Minimum download per pipeline:**

- Protein binder (default): `complexa download --complexa --all`
- Ligand binder: `complexa download --complexa-ligand --all`
- AME / enzyme: `complexa download --complexa-ame --all`
- All three: `complexa download --everything`

ESMFold is in no `complexa download` flag — fetch it with
`python script_utils/download/download_esmfold_model.py`. Per-model destinations
and per-flag NGC sources: [reference/downloads.md](reference/downloads.md).

Run the smallest invocation that covers the goal. Without arguments
`complexa download` launches an interactive wizard — prefer explicit flags.

Verify what landed:

```bash
complexa download --status
```

Two groups — `Complexa Models (Required):` and `Core Models:` — each printing
`✓ Installed (<dir>)` or `○ Missing (<dir>):` plus one `✗ <filename>` per absent
file. Re-run the specific flag for anything missing.

## Step 5: Validate

Final check that `.env` is loadable and the required paths resolve:

```bash
complexa validate env
```

`validate env` checks: (1) `.env` exists, (2) `DATA_PATH` is set and points at
an existing directory. It does not check ckpt files — those are checked by
`complexa validate design <config>` once you have a pipeline config picked.

Common failures and fixes:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `.env file: No .env file found` | `complexa init` not run | Run `complexa init` first |
| `Environment not initialized` on `design`/`generate`/`evaluate` | `env.sh` never generated or never sourced | `complexa init <uv\|docker>` then `source env.sh` |
| `DATA_PATH: Not set in .env` | Placeholder not edited | Edit `LOCAL_DATA_PATH` in `.env` |
| `DATA_PATH: Directory not found` | Path edited but does not exist on disk | `mkdir -p $LOCAL_DATA_PATH` or copy target data there |
| Hydra error `InterpolationKeyError: AF2_DIR` | Reward/eval config wants AF2 but `.env` does not define it | Download AF2 weights or remove AF2 from the config |
| Hydra error `InterpolationKeyError: CACHE_DIR` | `.env_example` ships **no** `CACHE_DIR=` line (only `LOCAL_CACHE_DIR` / `DOCKER_CACHE_DIR`), yet `configs/dataset/unified/plinder.yaml:35` requires `${oc.env:CACHE_DIR}` | Add `CACHE_DIR=${LOCAL_CACHE_DIR}` to `.env` by hand — nothing generates it for you |

## Step 6: Emit setup artifact

Drop a JSON manifest in `./complexa_setup/` so the user has a single file
describing the resulting state. The shared helper writes it for you:

```bash
mkdir -p ./complexa_setup
python .claude/skills/_shared/scripts/write_manifest.py \
    --output-dir ./ckpts \
    --command "complexa init docker && complexa download --complexa --all" \
    --skill complexa-setup \
    --out ./complexa_setup/run_manifest.json
```

`--output-dir`, `--command`, and `--skill` are all **required**; there is no
`--kind`, `--runtime`, or `--preflight` flag. `--output-dir` is only where the
script looks for `.hydra/config.yaml` — setup produces no Hydra run dir, so the
manifest's `config` and `checkpoints` fields come back `null`. That is expected
here; keep `preflight.json` as the record of on-disk state.

Surface the resulting files to the user:

```bash
ls -la ./complexa_setup/
```

Expected contents:

```
./preflight.json            # GPU / disk / .env / ckpt / tool snapshot (repo root)
complexa_setup/
└── run_manifest.json       # one --command string + skill + timestamp + git SHA
```

## Hardware requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | 1× CUDA GPU, ≥24 GB VRAM | A100 / H100 / L40S, 40–80 GB VRAM |
| CUDA | 12.0 | 12.4+ |
| Disk (CKPT_PATH) | 50 GB | 150 GB (`--everything` is ≈20 GB; the rest is samples / eval output) |
| RAM | 16 GB | 64 GB+ |
| OS | Ubuntu 22.04+ (UV) | Ubuntu 22.04+ or Docker on any host |

Ubuntu 20.04 throws GLIBC errors with the UV runtime — use `complexa init
docker` on those hosts. See `.claude/skills/_shared/reference/hardware.md`
for per-pipeline (binder vs ligand vs AME) requirements.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `complexa: command not found` | Package not installed in active env | `source .venv/bin/activate` then `pip install -e .` |
| `complexa init` says `.env_example not found` | Running outside repo root | `cd` to the project root (where `.env_example` lives) |
| `.env_example not found. Cannot initialize .env.` | Not in project root | `cd` into `Proteina-Complexa/` and retry |
| `complexa download` fails on NGC URL | Behind firewall / no internet | Configure a proxy for the download script, or download the model `.ckpt`s manually from the NGC pages linked in the main `README.md` and drop them into `./ckpts/` |
| `complexa download --status` shows ckpts present but `validate` fails | `.env` `CKPT_PATH` points elsewhere | Either move ckpts or edit `LOCAL_CHECKPOINT_PATH` in `.env` |
| GLIBC error on import | Ubuntu 20.04 with UV runtime | Re-run `complexa init docker --force`, `source env.sh`, and use `./env/docker-ops.sh run` |

---

For the full `.env` reference (every key, defaults, failure modes), see
[reference/env_keys.md](reference/env_keys.md).

For the full download flag matrix, NGC URLs, and destination layout, see
[reference/downloads.md](reference/downloads.md).
