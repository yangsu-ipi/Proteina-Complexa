# `.env` Key Reference

Reference for the `.env_example` variables you are likely to edit. Each entry
says: required vs optional, default, what reads it, and the failure mode if it is
missing or wrong.

> Not exhaustive: the `CLUSTER_*` SLURM block (`.env_example:110-154`, ~19 keys)
> and `CLUSTER_USER` (`:37`) are not covered here — they only matter for remote
> SLURM submission. Section order here does not track `.env_example` exactly.

`.env` is loaded by `python-dotenv` via `src/proteinfoundation/cli/validate.py:load_env_config`
and resolved into Hydra configs via `${oc.env:VARIABLE_NAME}` interpolation.
Missing required variables surface as Hydra `InterpolationKeyError` at config
resolution time — intentional, so you see exactly which key is missing.

---

## Section 1 — Required

You must set these before running any pipeline command. `complexa init` does
not fill these in — Phase 1 copies `.env_example` to `.env` verbatim and Phase 2
(`complexa init <uv|docker>`) writes a separate `env.sh` without touching `.env`.

### `LOCAL_CODE_PATH`

- **Required.** No default — `.env_example` ships with a placeholder.
- Absolute path to this repo checkout on the host.
- Read by: `COMMUNITY_MODELS_PATH`, `AF2_DIR`, `ESM_DIR`, `RF3_DIR`, `RF3_CKPT_PATH`, `UV_VENV` (all derived via `${LOCAL_CODE_PATH}/...`).
- **Failure mode**: every community-model and tool path resolves to `/path/to/Proteina-Complexa/...` (the `.env_example:25` placeholder) which does not exist → `complexa validate evaluate` / Hydra `FileNotFoundError`.
- Fix: edit to an absolute path, e.g. `LOCAL_CODE_PATH=/home/me/code/Proteina-Complexa`.

### `LOCAL_DATA_PATH`

- **Required.** Default placeholder `/path/to/PFM_data`.
- Absolute path to the PFM data directory (target PDBs under `target_data/`, datasets, etc.).
- Read by: `DATA_PATH` (`.env_example:105` ships `DATA_PATH=${LOCAL_DATA_PATH}`; the generated `env.sh` re-exports it from `DOCKER_DATA_PATH` on the docker runtime only); `complexa validate env` requires this to point at an existing directory.
- **Failure mode**: `complexa validate env` reports `DATA_PATH: Directory not found`; `complexa validate target` fails to locate `target_data/`.
- Fix: edit, then `mkdir -p $LOCAL_DATA_PATH` and populate it with the target PDBs you plan to design against (the bundled examples ship under `assets/target_data/`, or build your own with the `complexa-target` skill).

---

## Section 2 — Credentials (all optional)

### `GITLAB_TOKEN`

- **Optional.** Default placeholder `TOKEN_HERE`.
- Used by `env/docker-ops.sh` to authenticate against a private Docker registry (only relevant if you build the image yourself and push it somewhere that needs a token).
- **Failure mode if missing**: `docker login` is skipped → cannot pull private images. The default Dockerfile build (`docker build -f env/docker/Dockerfile .`) and all NGC downloads still work without it.
- Fix: set if you have your own private registry configured via `REGISTRY` / `DOCKER_IMAGE`.

### `WANDB_API_KEY` / `WANDB_ENTITY`

- **Optional.** Default placeholders `YOUR_WANDB_KEY` / `YOUR_WANDB_ENTITY`.
- `WANDB_API_KEY` is forwarded into the container by `env/docker-ops.sh:352`; `WANDB_ENTITY` by `:355`.
- **The placeholder guard does not work as shipped.** Those two lines skip the value only when it equals `"YOUR WANDB KEY"` / `"YOUR WANDB ENTITY"` (spaces), while `.env_example:20-21` ships `YOUR_WANDB_KEY` / `YOUR_WANDB_ENTITY` (underscores) — so the untouched placeholders **are** injected. Blank them out if you do not want that.
- `WANDB_ENTITY` is not what training reads for the entity: `train.py:384` takes it from `cfg_exp.log.wandb_entity`. The env var only reaches the W&B SDK's own default resolution.
- **Failure mode if missing**: no W&B logging; training still runs.
- Fix: set both if you want training runs tracked, or clear them entirely.

### `HF_TOKEN`

- **Optional.** Default placeholder `HF_TOKEN_HERE`.
- Read by `env/download_startup.sh` (the script behind `complexa download`) when pulling ESM2 from Hugging Face Hub, and by `script_utils/download/download_esmfold_model.py` for ESMFold.
- **Failure mode if missing**: ESM2 / ESMFold downloads may hit anonymous rate limits or fail for gated repos. Other downloads (NGC, GitHub) work without it.
- Fix: set if `complexa download --esm2` fails with 401/429. Note there is no `--esmfold` flag — the bash script rejects it with `Unknown option` and exits 1.

---

## Section 3 — Local options (all optional)

### `LOCAL_CACHE_DIR`

- **Optional.** Default `${LOCAL_CODE_PATH}/.cache` (`.env_example:27`).
- Used for Hydra cache, foldseek temp, HuggingFace hub cache.
- **Gap worth knowing:** there is **no** `CACHE_DIR=` line anywhere in `.env_example` — only `LOCAL_CACHE_DIR` and `DOCKER_CACHE_DIR`. Nothing in `complexa init` creates one either: the generated `env.sh` re-exports `LOCAL_CACHE_DIR` (docker branch) and never defines `CACHE_DIR`. But `configs/dataset/unified/plinder.yaml:35` interpolates `${oc.env:CACHE_DIR}`, so any run composing that dataset raises `InterpolationKeyError: CACHE_DIR` until you add `CACHE_DIR=${LOCAL_CACHE_DIR}` to `.env` by hand.
- **Failure mode if missing**: defaults work for almost everyone; set only if `.cache` should live on a faster / larger disk.

### `LOCAL_CHECKPOINT_PATH`

- **Optional.** Default in `.env_example`: `${LOCAL_CODE_PATH}/checkpoints`.
- Active alias `CKPT_PATH` resolves to this for UV runtime.
- Note: `complexa download` always writes Complexa model + AE checkpoints to `$PROJECT_ROOT/ckpts/` (a sibling of `checkpoints/`) regardless of this setting. If you want `CKPT_PATH` to point at the download location, set `LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts` after running `complexa download`.
- **Failure mode if missing**: pipeline configs resolve `${oc.env:CKPT_PATH}` to the default — if the directory doesn't exist or is empty, loading the model fails.
- Fix: either run `complexa download --complexa-all` and set `LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts`, or move/symlink the downloaded ckpts into the `checkpoints/` default.

### `DOCKER_MOUNTS`

- **Optional.** Default empty.
- Comma-separated `host:container` pairs added to `env/docker-ops.sh run`.
- **Failure mode if missing**: only standard mounts (`LOCAL_CODE_PATH`, `LOCAL_DATA_PATH`) are exposed to the container.
- Fix: set if you need extra paths visible inside the container — e.g. `DOCKER_MOUNTS=/scratch:/scratch,/lustre:/lustre`.

### `LOGURU_LEVEL`

- **Optional.** Default `INFO`.
- Read by `loguru` for Python log verbosity.
- Set to `DEBUG` for verbose pipeline logs, `WARNING` for quieter runs.

### `USE_V2_COMPLEXA_ARCH`

- **Optional.** Default `False`.
- Set to `True` only when using V2 Complexa model weights. The default-shipped checkpoints are V1.
- **Failure mode if wrong**: loading a V2 ckpt with this `False` (or a V1 ckpt with this `True`) throws a state-dict mismatch at model load time.

---

## Section 4 — Docker image (rarely edited)

These are read by `env/docker-ops.sh build/pull/run`.

### `REGISTRY` / `REGISTRY_USER`

- **Required only for `docker-ops.sh push/pull` against a private registry.** Defaults in `.env_example`: `registry.example.com` / `'$oauthuser'` (placeholders — you must edit if pushing/pulling).
- Used in `docker login` and tagging. Local `docker build` does not need these.

### `DOCKER_IMAGE`

- **Required for `docker-ops.sh run` (Docker runtime).** Default placeholder `registry.example.com/org/repo:tag`.
- Tag of the image `docker-ops.sh run` will start. If you built the image yourself with `docker build -t proteina-complexa -f env/docker/Dockerfile .`, set this to `proteina-complexa:latest`.

### `CONTAINER_NAME`

- **Required for Docker runtime.** Default `proteina-dev`.
- Name applied to `docker run --name`; reused for `exec` / `stop`.

### `DOCKERFILE_PATH`

- **Required for `docker-ops.sh build`.** Default `env/docker/Dockerfile`.
- Path (relative to `LOCAL_CODE_PATH`) of the Dockerfile used by `docker-ops.sh build`.

---

## Active aliases and runtime-selected keys

`complexa init` **never rewrites `.env`.** `complexa init <uv|docker>` writes a
separate `env.sh` which sources `.env` and then re-exports a short list of vars
for the chosen runtime; everything else below is a plain `${...}` alias inside
`.env_example` that you change by editing the `LOCAL_*` / `DOCKER_*` / `UV_*`
member instead.

### `COMPLEXA_INIT` (in `env.sh`, not `.env`)

- Exported as `COMPLEXA_INIT="<runtime>"` by the generated `env.sh` (`cli_runner.py:1704`). It appears in no `.env` file.
- **This is the gate on the whole CLI.** `_check_complexa_init` (`cli_runner.py:2001-2016`) exits 1 with "Environment not initialized" for `design`, `generate`, `filter`, `evaluate`, `analyze`, `analysis`, and `target` when it is unset. `init`, `demo`, `download`, `validate`, and `status` are exempt.
- Fix: `complexa init <uv|docker>` then `source env.sh` in every shell (it is not persisted).
- `env.sh` refuses to overwrite itself; pass `--force` to regenerate after switching runtime.

### Re-exported by `env.sh`

- **Every runtime:** `FOLDSEEK_EXEC`, `RF3_EXEC_PATH`, `SC_EXEC`, `MMSEQS_EXEC`, `DSSP_EXEC`, `TMOL_PATH` — each set to `${<RUNTIME>_<VAR>:-$<VAR>}`, i.e. the `UV_*` or `DOCKER_*` member.
- **Docker only, additionally:** `LOCAL_CODE_PATH` (from `DOCKER_REPO_PATH`), `COMMUNITY_MODELS_PATH`, `LOCAL_CACHE_DIR`, `CKPT_PATH` (from `DOCKER_CHECKPOINT_PATH`), `DATA_PATH` (from `DOCKER_DATA_PATH`).
- Nothing else is touched. In particular `CACHE_DIR` is never defined (see `LOCAL_CACHE_DIR` above).

### `DATA_PATH` / `CKPT_PATH`

- Plain aliases in `.env_example` (`:105`, `:108`): `DATA_PATH=${LOCAL_DATA_PATH}`, `CKPT_PATH=${LOCAL_CHECKPOINT_PATH}`. On the docker runtime `env.sh` re-points both at the `DOCKER_*` values. Edit `LOCAL_*` / `DOCKER_*` instead of these.

### `FOLDSEEK_EXEC` / `RF3_EXEC_PATH` / `SC_EXEC` / `MMSEQS_EXEC` / `DSSP_EXEC` / `TMOL_PATH`

- Active tool binaries; `env.sh` resolves them to `${UV_*}` or `${DOCKER_*}` per runtime. Edit the prefix vars if you have a non-standard install (e.g. system-wide `foldseek` at `/usr/local/bin/foldseek` instead of `.venv/bin/foldseek`).
- Used by: `complexa evaluate` (foldseek for diversity; mmseqs for sequence clustering; sc for interface metrics; dssp for secondary structure; tmol for force-field metrics).
- **Failure mode if path is wrong**: the tool is silently skipped (treated as a warning in `complexa validate evaluate`), and the corresponding metric column is missing from the result CSV.

### `AF2_DIR` / `ESM_DIR` / `RF3_DIR` / `RF3_CKPT_PATH`

- **Not auto-managed** — `complexa init` never writes these; they are static `${...}` derivations in `.env_example:69-72` and are absent from the `env.sh` re-export list. Edit them here if your weights live elsewhere. (There is no `ESMFOLD_DIR` key anywhere in the repo.)
- Derived from `${COMMUNITY_MODELS_PATH}/ckpts/...`. After `complexa download --all` or `complexa download --af2`, the directories under `community_models/ckpts/` are populated.
- Read by: reward models (`AF2RewardModel`, `RF3RewardRunner`) and evaluation folding (colabdesign / rf3 backends).
- **Failure mode if wrong**: `complexa validate evaluate` reports `AF2 weights: Directory not found` or `RF3 checkpoint: File not found`. Generation can still run without these; only reward and refolding break.

### `RF3_CKPT_PATH`

- Default `${RF3_DIR}/rf3_foundry_01_24_latest_remapped.ckpt` — exact filename produced by `complexa download --rf3`. If you have a different RF3 checkpoint, edit to its full path.

### `COMMUNITY_MODELS_PATH`

- Default `${LOCAL_CODE_PATH}/community_models`. Edit only if you mirror community models on a separate disk.

### `UV_VENV` / `UV_*_EXEC` / `DOCKER_*_EXEC`

- Per-runtime tool-path families, hand-edited only. `complexa init <uv|docker>` reads them to decide what the active `FOLDSEEK_EXEC` etc. resolve to in `env.sh`, but it never rewrites the family members themselves. Edit the member (e.g. `UV_FOLDSEEK_EXEC`) if your local install lives somewhere unusual.

### `DOCKER_REPO_PATH` / `DOCKER_DATA_PATH` / `DOCKER_PYTHONPATH` / `DOCKER_CHECKPOINT_PATH` / `DOCKER_CACHE_DIR` / `DOCKER_HF_HOME` / `DOCKER_HF_HUB_CACHE`

- Container-internal paths inside the Complexa Docker image. Hard-coded to `/workspace/...`; only edit if you ship a custom image with different layout.

---

## What `complexa validate env` actually checks

From `src/proteinfoundation/cli/validate.py:validate_env`:

1. `.env` file exists **in the current working directory** — `validate.py:254` is a bare `Path(".env")`, with no parent walk. (`complexa init` *does* walk parents, but only to find `.env_example`.)
2. `DATA_PATH` env var is set and resolves to an existing directory.

That is the full check. It does not validate ckpts, tool binaries, or HF
tokens. Those are checked by `complexa validate {generate,evaluate,design}
<config>` which loads a pipeline YAML and verifies the paths each stage will
actually read. Run `complexa validate design configs/search_binder_local_pipeline.yaml`
once after editing `.env` to catch missing AF2/RF3/foldseek before the first
real pipeline run.
