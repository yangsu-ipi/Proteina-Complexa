# Target Setup Troubleshooting

Failures specific to target and environment setup. For OOM, missing weights, folding
backends, and success thresholds see
`.claude/skills/complexa-design/reference/troubleshooting.md`.

## Error locating target

**Symptom:** generation dies a few seconds in, after the checkpoint loads and the model
builds, with a Hydra error naming a symbol that plainly exists:

```
hydra.errors.InstantiationException: Error locating target
'proteinfoundation.datasets.gen_dataset.collate_fn', set env var HYDRA_FULL_ERROR=1
to see chained exception.
full_key: generation.dataloader.collate_fn
```

**Cause:** not a missing symbol. This is Hydra's message for *"I tried to import the module
holding this target and the import raised."* The real exception is suppressed unless
`HYDRA_FULL_ERROR=1`.

The most common underlying cause is an invalid `CCD_MIRROR_PATH` — see
[`env-and-mirrors.md`](env-and-mirrors.md#why-an-invalid-value-is-worse-than-no-value) —
which raises `FileNotFoundError` from a module-level statement in
`datasets/atomworks_ligand_transforms.py:28`.

**Why it lands on `collate_fn` specifically.** `generate.py` imports `atomworks.ml`,
`biotite`, `openfold`, and `torch` eagerly at module load, so those are all fine by the
time you see this. `gen_dataset` is the **first lazily-imported module** in the run — it is
pulled in only at `hydra.utils.instantiate(cfg_gen.dataloader)`
(`src/proteinfoundation/generate.py:835`) — and it drags in four modules nothing in
`generate.py`'s eager graph touches:

```
gen_dataset
├── datasets/atomworks_ligand_transforms.py  → rdkit, atomworks.io.tools.rdkit,
│                                               atomworks.ml.transforms.openbabel_utils,
│                                               scipy.linalg / scipy.sparse.linalg
└── utils/motif_utils.py → utils/constants.py → graphein.protein.resi_atoms
```

Any import failure in that subtree presents as this error.

**Fix.** Unmask it. Fastest is to bypass Hydra entirely:

```bash
python -c "import proteinfoundation.datasets.gen_dataset"
```

Or re-run with the mask off — `HYDRA_FULL_ERROR` propagates, because `run_step` passes
`os.environ.copy()` to the subprocess (`src/proteinfoundation/cli/cli_runner.py:674`):

```bash
HYDRA_FULL_ERROR=1 complexa design ./pipeline.yaml --verbose
```

Then check the four suspects in one shot:

```bash
python -c "import rdkit; import graphein.protein.resi_atoms; import scipy.sparse.linalg; \
from atomworks.io.tools.rdkit import atom_array_from_rdkit; \
from atomworks.ml.transforms.openbabel_utils import atom_array_to_openbabel; \
print('all four import paths OK')"
```

Remedies by failing line:

| Failing import | Fix |
|---|---|
| `graphein.protein.resi_atoms` | `pip install networkx` — graphein is installed `--no-deps` (`env/build_uv_env.sh:171`) |
| `rdkit` / `atomworks.io.tools.rdkit` | `pip install rdkit` — not declared in `pyproject.toml`, arrives via the `atomworks[ml]` extra |
| `atomworks.ml.transforms.openbabel_utils` | `pip install "atomworks[ml,openbabel]"` |
| `scipy` ABI error under numpy 2.x | `pip install -U "scipy>=1.14"` |
| a `FileNotFoundError` on a mirror path | [`env-and-mirrors.md`](env-and-mirrors.md) |

Also confirm you are in the environment you think you are — shared installs make this easy
to get wrong:

```bash
python -c "import proteinfoundation, sys; print(sys.executable); print(proteinfoundation.__file__)"
```

## `missing required environment keys: ['LOCAL_CODE_PATH', 'LOCAL_DATA_PATH', 'CKPT_PATH']`

**Symptom:** a batch job sources `env.sh` successfully (`Complexa environment initialized
for <runtime> runtime.` appears in the log) yet the preflight reports those three keys
missing, plus missing checkpoints and missing `AF2_DIR` / `ESM_DIR` — while
`foldseek` and `mmseqs` resolve fine.

**Cause:** `env.sh` sourced `.env` without `set -a`. `.env` has no `export` lines, so only
`_TOOL_VARS` and `COMPLEXA_INIT` reached child processes; the path variables stayed local
to the sourcing shell. The tool binaries resolving while every path is empty is the
fingerprint. The five failures are one root cause: `CKPT_PATH` empty makes checkpoint paths
resolve to `/complexa.ckpt`, and `AF2_DIR`/`ESM_DIR` derive from `LOCAL_CODE_PATH`
(`.env_example:68-70`).

`preflight.sh` is not at fault — it reads `$PWD/.env` then falls back to the live
environment (`preflight.sh:56-61`), and the fallback found nothing exported.

**Fix:** regenerate `env.sh` (`complexa init <uv|docker> --force`), or force allexport at
the call site:

```bash
set -a; source "$COMPLEXA_REPO/env.sh"; set +a
```

Full explanation, plus a SLURM template and a pre-run assertion, in
[`env-and-mirrors.md`](env-and-mirrors.md#the-envsh-export-gap).

**Note the pipeline would have run anyway.** The stage modules call bare `load_dotenv()`,
which finds the repo's `.env` by walking up from the module file regardless of cwd or
exports — so this failure is the *checks* disagreeing with the *pipeline*, not a broken
environment. `complexa validate design` fails for the related-but-distinct reason that
`load_env_config` reads `Path(".env")` from cwd only (`validate.py:137`).

## `missing checkpoint: complexa.ckpt` when the file is right there

**Symptom:** the preflight reports the Complexa checkpoints (and often `ESM_DIR`) missing,
while `LOCAL_CODE_PATH` / `LOCAL_DATA_PATH` / `CKPT_PATH` are all populated and
`missing_required` is empty. The reported path ends in `checkpoints/`.

**Cause:** `CKPT_PATH` derives from `LOCAL_CHECKPOINT_PATH` (`.env_example:108`), which
used to default to `${LOCAL_CODE_PATH}/checkpoints` — while every downloader writes to
**`ckpts/`** (`download_startup.sh:239`, and the same `./ckpts` in the ligand and AME
functions). The destination is computed from the script's own location
(`download_startup.sh:23-24` → `PROJECT_ROOT` → `cd`, then a relative `./ckpts`) and never
from `.env`, so the two could not reconcile. The shipped configs side with the downloaders
(`ckpt_path: ./ckpts`), as does `preflight.sh`'s own last-resort fallback
(`preflight.sh:66`, `$LOCAL_CODE_PATH/ckpts`) — `.env_example` was the lone dissenter.

The default is now `${LOCAL_CODE_PATH}/ckpts`, so fresh installs are correct. **An existing
`.env` is not updated by that change** — fix it by hand, then regenerate `env.sh`:

```bash
LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts
complexa init <uv|docker> --force
```

A symlink (`ln -s ckpts checkpoints`) also works, but the `.env` edit is what the rest of
the tooling expects. Note this variable is also the docker bind-mount source
(`docker-ops.sh:407-409`, `LOCAL_CHECKPOINT_PATH -> DOCKER_CHECKPOINT_PATH`), so the same
correction fixes container runs, which were mounting an empty directory.

**Check whether it actually blocks your run.** This is usually a *gate-only* failure: a
pipeline YAML that sets `ckpt_path` / `autoencoder_ckpt_path` to absolute `.../ckpts` paths
loads the checkpoints fine, because those config keys are read directly and never go
through `CKPT_PATH`. Confirm with:

```bash
grep -E '^(ckpt_path|autoencoder_ckpt_path):' pipeline.yaml
```

If they point at the real directory, only the preflight is wrong — but fix `.env` anyway,
so the next preflight tells the truth.

## `complexa validate design` fails on `.env` and `target_data` from a campaign directory

**Symptom:** preflight passes, the target PDB checks out, the config resolves — and then
`complexa validate design` reports exactly two errors and exits 1, killing the job under
`set -euo pipefail`:

```
✗ .env file
    No .env file found in current directory
✗ target_data directory
    Directory not found: <DATA_PATH>/target_data
```

…alongside passes for the checkpoints, `DATA_PATH`, the target PDB, and Foldseek.

**Cause (fixed — update the repo).** Two checks asked the wrong question.
`complexa validate design` fans out to all three sub-validators (`validate.py:794-823`:
`validate_env()`, `validate_generate()`, `validate_evaluate()`), and two of them assumed
the repo was the working directory:

1. `validate_env` tested for the `.env` **file** in cwd and returned early, never
   consulting the environment — so `env.sh`'s exports could not satisfy it. It now keys on
   the **variables**: `.env` in cwd is loaded when present (never overriding exports), and
   only a genuinely unset `DATA_PATH` fails. Same fall-through `preflight.sh:56-61` uses.
2. `validate_target` checked `$DATA_PATH/target_data` unconditionally, *before* resolving
   any target. It is now checked only in the fallback branch that actually uses it, so an
   entry with an explicit `target_path` no longer requires the shared tree — matching the
   `oc.select` in `configs/pipeline/binder/binder_generate.yaml:33`.

A real misconfiguration is still caught: nothing exported and no `.env` in cwd still fails
on `DATA_PATH`, and a `source`/`target_filename` target with no `target_data` tree still
fails on both the directory and the missing PDB.

**Fix:** pull a Complexa containing those changes. Nothing needs creating in the campaign
directory or the models tree.

**On older installs**, work around it with two one-liners:

```bash
printf '# empty — env comes from env.sh\n' > "$CAMPAIGN_DIR/.env"   # existence-only check
mkdir -p "$DATA_PATH/target_data"                                   # existence-only; empty is fine
```

Prefer a stub over `ln -s "$COMPLEXA_REPO/.env"`: `.env` carries `HF_TOKEN`,
`WANDB_API_KEY` and `GITLAB_TOKEN` (`.env_example:19-22`), and `cp -R`/`rsync` without `-l`
dereferences symlinks — so archiving or syncing the campaign bundle would carry the
credentials with it. A stub works because `load_env_config` only tests existence and then
reads `os.environ` (`validate.py:137-154`), and `load_dotenv` defaults to
`override=False`, so file content is irrelevant once `env.sh` is sourced.

**Both checks were gate-only** either way. Nothing in generate/filter/evaluate/analyze
reads them, and `validate_target` never opens the PDB (see
[`pdb-prep.md`](pdb-prep.md#complexa-validate-target-will-not-catch-any-of-this)).

**Expect one warning that is not a problem:** `Shape complementarity (sc)` pointing at
`$LOCAL_CODE_PATH/env/docker/internal/sc`. `UV_SC_EXEC` and `UV_DSSP_EXEC` default to
docker-internal paths (`.env_example:79-87`) that do not exist in a native install. They
only affect the `bioinformatics` interface metrics, which are off by default.

## `missing community model path: ESM_DIR`

**Symptom:** `ESM_DIR` reported missing while `AF2_DIR` exists.

**Cause:** genuinely not downloaded. `ESM_DIR` resolves to
`${LOCAL_CODE_PATH}/community_models/ckpts/ESM2` (`.env_example:69`), which is exactly where
`complexa download --esm2` writes (`download_startup.sh:371`) — so this is a missing asset,
not a path mismatch. Unlike the checkpoints above, it is **not** gate-only: the binder
evaluate config sets `compute_esm_metrics: true`
(`configs/pipeline/binder/binder_evaluate.yaml:35`).

**Do not just `mkdir` the directory.** `_resolve_esm_dir` tests only `os.path.isdir`
(`evaluation/esm_eval.py:372-377`), so an empty directory resolves, becomes the *first*
load location ahead of the HF cache (`:425-426`), and then `from_pretrained(...,
local_files_only=True)` fails. With `force_offline=True` — the default on
`compute_esm_ppl_for_sequences` (`:645`) — you get `RuntimeError: ESM model not found in
local paths` **partway through evaluation**, after generation has already spent the GPU
time. Silencing the gate this way makes the failure later and more expensive. Gate on
`community_models.ESM_DIR.has_weights` instead of `.exists`.

**Fix**, either:

```bash
cd "$COMPLEXA_REPO" && complexa download --esm2      # HF_TOKEN if rate-limited
```

**On older installs `--esm2` is rejected.** The wrapper's argparse used to declare only
`--complexa*`, `--all`, `--everything` and `--status`, so the per-model flags were refused
before they could be forwarded — even though the script has always accepted them (the
handler passes `sys.argv[2:]` verbatim). All five are now declared. If you hit
`unrecognized arguments: --esm2`, either update the repo or bypass the wrapper:
`complexa-download --esm2` (`pyproject.toml:72`, forwards `sys.argv[1:]`) or
`bash env/download_startup.sh --esm2`.

or skip the metric for this run:

```bash
++metric.compute_esm_metrics=false
```

Do the download on a login node, not inside the job. Without a local copy,
`_resolve_esm_dir()` returns `None` (`evaluation/esm_eval.py:362-378`) and the code falls
back to a HuggingFace fetch *during evaluation* — which needs network and possibly a token
on the compute node, mid-run.

`RF3_CKPT_PATH` missing is harmless when `metric.binder_folding_method: colabdesign`; the
default preflight gate does not require it.

## `only N GB free near checkpoints; require at least 50 GB`

**Symptom:** a gate rejects the run on disk space, while the volume the run will actually
write to has plenty.

**Cause:** the figure is free space at **`CKPT_PATH`** — the install — not at the working
directory where `./inference/` and `./logs` land. Those are frequently different mounts
(`/data/shared/tools/...` vs `/data/<user>/campaign/...`). Downloading weights eats the
install volume and trips a gate that was never measuring the right thing.

**Fix:** gate on `disk.cwd_free_gb`, which `preflight.sh` now reports alongside
`ckpt_free_gb` (`free_gb` is retained as an alias for the latter). Size the threshold from
the design count — see
[`campaign-gating.md`](campaign-gating.md#gate-on-the-resolved-config-not-on-a-fixed-list).

`preflight.sh` also reports `ckpt_fs` and `cwd_fs`, so the preflight JSON answers "same
volume?" on its own — no manual `df` needed. Do **not** infer it from equal `free_gb`:
APFS/Btrfs/thin-LVM report the same figure for distinct mounts. Compare the mount points:

```bash
python3 -c "import json;d=json.load(open('metadata/preflight_smoke.json'))['disk'];\
print(d['ckpt_fs'], d['cwd_fs'], 'SHARED' if d['ckpt_fs']==d['cwd_fs'] else 'separate')"
```

If shared, the download budget and the output budget come out of the same pool and must be
added. Symlinks need no special handling — `df` resolves them onto the target volume.

**Rough sizing** from `_shared/reference/hardware.md`: ~10–20 GB per 100 protein-binder
designs, roughly doubled by `keep_folding_outputs: true` (the eval default). So an 8-design
smoke test needs single-digit GB, while 1000 designs needs **200–400 GB** on the output
volume. A fixed 50 GB floor is simultaneously too strict for the former and far too loose
for the latter. `++metric.keep_folding_outputs=false` halves it.

## The silent-failure catalogue

None of these raise. Each produces a run that completes and writes PDBs.

| Symptom | Real cause | Check |
|---|---|---|
| **Run designs against a target you never asked for** | `generation.task_name` not pinned, so it inherits `33_TrkA` (`binder_generate.yaml:16`) — which *exists* in the shared 44, so nothing errors | generate log: `task_name` and `pdb_path`. Verified silent: yields `1www_cropped.pdb`, chain X, hotspots `X294 X296 X333` |
| Designs ignore your epitope | hotspot IDs don't match the file's numbering; mask is all-False (`pdb_utils.py:571-575`) | `check_target_pdb.py`; require the missing list to be empty |
| Target smaller than expected, or zero residues | `target_input` range doesn't match author numbering; `from_contig` selects literal `res_id`s | derive `target_input` from the file, not from an example |
| Hotspots match the wrong residue | `.cif` gives `label_seq_id`, `.pdb` gives author numbering (`io_utils.py:290`) | re-derive hotspots from the exact file you feed in |
| Waters / ions encoded as protein | `from_contig` filters on `(chain_id, res_id)` only (`selection.py:482-493`) | `check_target_pdb.py` reports in-range hetero residues |
| Shared 44 targets used despite a shadow file | shadow directory or filename is off; both are hardcoded (`binder_generate.yaml:8`). Silent when your `task_name` is one of the shipped 44; raises for a new name | generate log: `'target_dict_cfg': '<filtered: N entries>'` — `1` = took, `44` = did not |
| Relative `target_path` resolves nowhere | no `chdir` anywhere; paths resolve against your shell cwd | `cd` to the target dir, or use absolute paths |
| Target PDB missing from git | `*.pdb` is globally git-ignored | `git check-ignore -v <file>`; `git add -f` |
| Run outputs committed by accident | `/inference` is root-anchored, so nested `inference/` is not ignored | keep target dirs outside the repo |
| `import atomworks` fails but the build passed | `env/build_uv_env.sh:174` swallows the failure with `\|\| echo` | re-run the install without `\|\|` and read the error |
| Preflight says paths missing, but `env.sh` reported success | `.env` sourced without `set -a`; only `_TOOL_VARS` exported | `set -a; source env.sh; set +a`, or regenerate with `complexa init <runtime> --force` |
| `missing checkpoint` with a path ending `checkpoints/` | existing `.env` still says `checkpoints/`; downloaders write `ckpts/` | `LOCAL_CHECKPOINT_PATH=${LOCAL_CODE_PATH}/ckpts` |
| `missing community model path: ESM_DIR` | ESM2 not downloaded — real asset gap, and `compute_esm_metrics` defaults true | `complexa download --esm2`, or `++metric.compute_esm_metrics=false` |
| `validate design`: no `.env` in cwd + missing `$DATA_PATH/target_data` | two checks that assumed the repo was cwd — **fixed**; `validate_env` now keys on variables and `target_data` is only required by the fallback branch | update the repo; on older installs use a stub `.env` + `mkdir -p` (not a symlink — it carries secrets) |
| `env.sh` sourced with no error but nothing is set | sourced from zsh/dash — `${BASH_SOURCE[0]}` is empty, so `.env` was looked for in cwd | source it from bash |
| `preflight.sh: declare: -A: invalid option` | needs bash 4+; macOS ships 3.2 | run it on the cluster, or `brew install bash` |
| Disk gate fails but the output volume is empty | `free_gb`/`ckpt_free_gb` measure the **install**, not where `./inference` lands | gate on `disk.cwd_free_gb`; size from the design count |
| A `++` key has no effect | `++` adds-or-overrides and never errors, so a typo is a no-op | see "Override key not recognized" in `complexa-design/reference/troubleshooting.md` |

## Interpolation errors when defining a target inline

**Symptom:**

```
InterpolationKeyError: Interpolation key 'target_dict_cfg.<NAME>.source' not found
  full_key: generation.dataloader.dataset.conditional_features[0].pdb_path
```

**Cause:** your entry omits `source` or `target_filename`. OmegaConf evaluates the
`oc.select` **default argument** whether or not the primary key resolves, so both are
required even when `target_path` makes them redundant
(`configs/pipeline/binder/binder_generate.yaml:33`).

If `<NAME>` is `33_TrkA` rather than your target, the real problem is an unpinned
`generation.task_name` *and* a dict that replaced rather than merged. Note the asymmetry:

| Approach | `task_name` unpinned |
|---|---|
| Shadow / replace | raises this error — `33_TrkA` is absent from the replaced dict |
| Inline / merge | **silent** — `33_TrkA` survives from the shared 44 and gets designed |

**Fix:** pin `generation.task_name` and supply all seven keys. Full explanation in
[`target-config.md`](target-config.md#two-requirements-that-are-not-obvious).

## `Could not override 'targets@generation'`

**Symptom:** an attempt to point the `targets` config group at a different file is
rejected:

```
Could not override 'targets@generation'. No match in the defaults list.
```

**Cause:** the defaults entry uses the `@_here_` package keyword
(`binder_generate.yaml:8`), which appears to make it unaddressable by any override key.
`--info defaults` lists the entry, yet nothing matches it. All four plausible forms are
rejected; see
[`target-config.md`](target-config.md#shadow-file--the-only-way-to-fully-replace-the-dict).

**Fix:** do not override it. Either use the exact hardcoded shadow path
`targets/targets_dict.yaml`, or add your own defaults entry with a name you choose
(`- /my_specs/pdl1@generation`), or define the target inline.

## Confirming which target dict is live

The generate stage logs its resolved config. `grep` the entry count:

```bash
grep -o "'target_dict_cfg': '<filtered: [0-9]* entries>'" logs/generate.log
```

| Count | Meaning |
|---|---|
| `1` | shadow file took — only your target is visible |
| `45` | inline or merge worked — yours plus the shared 44 |
| `44` | **nothing took** — you are running the shared dict |
