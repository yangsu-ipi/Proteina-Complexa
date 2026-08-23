# Environment and Mirrors

Two related topics: how Complexa discovers its environment (and how that breaks when you
run from outside the repo, e.g. under SLURM), and the two atomworks mirror variables.

- [How the environment is discovered](#how-the-environment-is-discovered) — start here for
  batch jobs and `missing required environment keys` failures
- [atomworks mirror variables](#atomworks-mirror-variables) — `CCD_MIRROR_PATH`,
  `PDB_MIRROR_PATH`

---

# How the environment is discovered

There is no single mechanism. **Three** different ones coexist, and they disagree about
where `.env` lives — which is why a job can pass one check and fail another.

## A. cwd-only: `./.env`, no search

| Site | Code | Falls back to the live environment? |
|---|---|---|
| `.claude/skills/_shared/scripts/preflight.sh:33` | `ENV_FILE="$PWD/.env"` | yes (`:49-53`) |
| `cli/validate.py:137` (`load_env_config`) | `load_dotenv(Path(".env"))`, then reads `os.environ` | yes |
| `cli/validate.py:250` (`validate_env`) | reports whether `./.env` exists; **keys on the variables** | yes |
| `cli_runner.py:1721` (`complexa init`) | `Path(".env")` — writes here | no |

None of these walks up the tree or falls back to the repo, so **a `.env` sitting in the
install is invisible to all of them** — exported variables are what rescue them. Note
`validate.py` passes an **explicit** path to `load_dotenv`, which suppresses
python-dotenv's own search (contrast mechanism B below).

`validate_env` used to *fail* on a missing `./.env` and return early, which no exported
environment could satisfy. It now keys on the variables instead — see
[`troubleshooting.md`](troubleshooting.md#complexa-validate-design-fails-on-env-and-target_data-from-a-campaign-directory)
for the before/after and the workaround for older installs.

## B. upward search from the module file

Every stage module uses the **no-argument** form, which does search:

```python
# generate.py:674, filter.py:87, evaluate.py:677, analysis.py:1983, train.py:362, …
load_dotenv()
```

Verified against `python-dotenv==1.0.1`: run as `python -m proteinfoundation.generate` from
an unrelated cwd, `find_dotenv()` walks up from the *module's own* directory —
`src/proteinfoundation/` → `src/` → repo root — and finds the repo's `.env`. This works
because the install is editable src-layout.

**This is why the pipeline itself survives a misconfigured shell**, and why the bug below
went unnoticed: only the checks that read the live environment or `./.env` expose it.

`load_dotenv()` defaults to `override=False`, so anything already exported wins over `.env`.

## C. the `COMPLEXA_INIT` gate

`_check_complexa_init` (`cli_runner.py:2004-2016`) exits 1 for every non-exempt `complexa`
subcommand unless `COMPLEXA_INIT` is set, and only `env.sh` exports it. So `env.sh` is not
optional, regardless of how `.env` gets found.

## The `env.sh` export gap

`complexa init <runtime>` generates `env.sh`, which resolves `.env` next to itself and is
therefore cwd-independent by design (`cli_runner.py:1668-1679`). But `.env` holds plain
`KEY=value` lines with **no `export`**, so a bare `source .env` creates shell variables that
child processes never see.

The **docker** branch then explicitly exports the important ones
(`cli_runner.py:1686-1696`). The **uv** branch did not — it exported only `_TOOL_VARS`
(`FOLDSEEK_EXEC`, `RF3_EXEC_PATH`, `SC_EXEC`, `MMSEQS_EXEC`, `DSSP_EXEC`, `TMOL_PATH`,
`cli_runner.py:1630-1637`) plus `COMPLEXA_INIT`. `LOCAL_CODE_PATH`, `LOCAL_DATA_PATH`,
`CKPT_PATH`, `DATA_PATH`, `AF2_DIR`, `ESM_DIR` reached nothing.

The generator now wraps the source in `set -a` / `set +a`. Measured difference in a child
process, same `.env`:

| Variable | before | after |
|---|---|---|
| `FOLDSEEK_EXEC` | `/opt/bin/foldseek` | `/opt/bin/foldseek` |
| `COMPLEXA_INIT` | `uv` | `uv` |
| `LOCAL_CODE_PATH` | *unset* | `/data/shared/tools/Proteina-Complexa` |
| `CKPT_PATH` | *unset* | `…/Proteina-Complexa/ckpts` |
| `DATA_PATH` | *unset* | `/data/shared/PFM_data` |
| `AF2_DIR` / `ESM_DIR` | *unset* | `…/community_models/ckpts/{AF2,ESM2}` |

**Existing `env.sh` files still carry the old body** — the fix only affects newly generated
ones. Regenerate:

```bash
cd /path/to/Proteina-Complexa && complexa init uv --force
```

Or, without regenerating, force allexport at the call site — this works with either version:

```bash
set -a; source /path/to/Proteina-Complexa/env.sh; set +a
```

## conda environments use the `uv` runtime label

`complexa init` accepts only `uv` or `docker` (`cli_runner.py:1160-1166`) — there is no
`conda` choice. For a conda (or plain-venv) install, use **`uv`**, which is safe because:

- The runtime argument only selects which prefix the tool vars read:
  `export FOLDSEEK_EXEC="${UV_FOLDSEEK_EXEC:-$FOLDSEEK_EXEC}"`. Point the `UV_*` vars at
  your conda prefix and they resolve correctly.
- **Nothing branches on the value of `COMPLEXA_INIT`** — `cli_runner.py:2011` only tests
  presence (`if not os.environ.get("COMPLEXA_INIT")`). A conda env labelled `uv` is fine.

In `.env`, override `UV_VENV` and the tool vars follow (`.env_example:79-87`):

```bash
UV_VENV=/path/to/miniforge3/envs/proteina-complexa
UV_FOLDSEEK_EXEC=${UV_VENV}/bin/foldseek
UV_MMSEQS_EXEC=${UV_VENV}/bin/mmseqs
UV_TMOL_PATH=${UV_VENV}/lib/python3.12/site-packages/tmol
```

`UV_SC_EXEC` and `UV_DSSP_EXEC` default to `${LOCAL_CODE_PATH}/env/docker/internal/{sc,dssp}`
— docker-internal paths that do not exist in a conda install. They only matter for metrics
that need `sc`/`dssp`; `preflight.sh` reports them as missing either way, and the default
gate (`foldseek` + `mmseqs`) does not require them.

Verified end-to-end with the patched generator against a conda-style prefix: all path
variables exported, tool vars resolving into the conda `bin/`, and `CONDA_PREFIX` /
`CONDA_DEFAULT_ENV` untouched. The `set -a` mechanism is plain bash allexport and is
independent of the Python environment manager.

Order matters slightly: `conda activate` **before** sourcing `env.sh`, and keep the
activation *outside* the `set -a` block — otherwise conda's own internals get exported too
(harmless, but noisy in `env`).

`env.sh` exports both `COMPLEXA_INIT` (which gates the CLI) and `COMPLEXA_RUNTIME` (which
`preflight.sh` reports as `complexa_runtime`, `:48`/`:57`/`:216`), so preflight JSON and run
manifests record the runtime label. Older generated `env.sh` files set only
`COMPLEXA_INIT`, leaving `complexa_runtime` as `""` — cosmetic, but another reason to
regenerate.

## Running from outside the repo (SLURM, campaign directories)

```bash
#!/usr/bin/env bash            # bash, not sh/zsh -- see below
set -euo pipefail

COMPLEXA_REPO=/data/shared/tools/Proteina-Complexa

source /path/to/conda/etc/profile.d/conda.sh
conda activate proteina-complexa

set -a; source "$COMPLEXA_REPO/env.sh"; set +a    # exports + COMPLEXA_INIT
export CCD_MIRROR_PATH= PDB_MIRROR_PATH=          # see the mirror section below

cd "$MY_CAMPAIGN_DIR"                             # outputs land here
complexa design ./pipeline.yaml --verbose
```

Sanity check before the expensive part — if any of these is empty, `env.sh` did not take:

```bash
for k in LOCAL_CODE_PATH LOCAL_DATA_PATH CKPT_PATH DATA_PATH AF2_DIR ESM_DIR COMPLEXA_INIT; do
    printf '%-18s %s\n' "$k" "${!k:-<UNSET>}"
done
```

### Gating the run itself

Deciding which weights a run requires, and how much output disk it needs, is the *gate's*
job — not `preflight.sh`'s, which is deliberately config-blind. That belongs to a separate
topic: see [`campaign-gating.md`](campaign-gating.md).

Three footguns:

- **Source `env.sh` from bash.** It uses `${BASH_SOURCE[0]}`; under zsh or dash that is
  empty, so `_ENVSH_DIR` silently becomes your cwd and `.env` is not found. No error.
- **`env.sh` may not exist at the repo root.** `complexa init` writes it to **cwd**
  (`Path("env.sh")`, `cli_runner.py:1722`), so it is only there if someone ran init there.
  `ls "$COMPLEXA_REPO/env.sh"` before relying on it.
- **`preflight.sh` needs bash 4+** (`declare -A`). Fine on Linux; macOS ships bash 3.2.

---

# atomworks mirror variables

`CCD_MIRROR_PATH` and `PDB_MIRROR_PATH` are read by `atomworks`, not by this repo's
`.env` machinery. Neither appears in `.env_example`, so they are usually inherited from a
shell profile, a conda `activate.d` script, or a shared-install `env.sh` — which is
exactly how they end up pointing somewhere that does not exist.

## Short version

**Set both to empty unless you have real mirrors on disk.** Empty is the intended
default, and you lose almost nothing.

```bash
export CCD_MIRROR_PATH= PDB_MIRROR_PATH=
```

## Why an invalid value is worse than no value

`atomworks_ligand_transforms.py` resolves the CCD code set **at module import time**, as a
module-level constant:

```python
# src/proteinfoundation/datasets/atomworks_ligand_transforms.py:28
KNOWN_CCD_CODES = get_available_ccd_codes(CCD_MIRROR_PATH) - {UNKNOWN_LIGAND}
```

atomworks treats the mirror path as truthy-or-skip — an **empty** value is fine, but a
non-empty path that does not exist goes straight to `iterdir()` and raises
`FileNotFoundError`. Because this happens during an import that Hydra performs lazily, the
error surfaces as an unrelated-looking `Error locating target` on the dataloader's
`collate_fn`. See ["Error locating target"](troubleshooting.md#error-locating-target) for
the full trace and diagnosis.

This repo has two guards, and **both only handle the unset case**:

```python
# src/proteinfoundation/cli/startup.py:121-124
# src/proteinfoundation/patches/atomworks_patches.py:15-20
if "CCD_MIRROR_PATH" not in os.environ:
    os.environ["CCD_MIRROR_PATH"] = ""
```

A variable that *is* set, to a path that is *wrong*, sails past both. `PDB_MIRROR_PATH`
and `LOCAL_MSA_DIRS` are guarded the same narrow way in
`patches/atomworks_patches.py:15-20`.

Find where yours is set:

```bash
env | grep -E 'CCD_MIRROR|PDB_MIRROR|LOCAL_MSA'
grep -rn "CCD_MIRROR_PATH" .env env.sh ~/.bashrc ~/.profile "$CONDA_PREFIX"/etc/conda/activate.d/ 2>/dev/null
```

## Are the mirrors needed at all?

**CCD: no.** `get_available_ccd_codes` unions the mirror with biotite's built-in CCD, and
per-code lookups fall back to biotite:

```python
# atomworks/io/utils/ccd.py:184-186
mirror_codes = get_available_ccd_codes_in_mirror(ccd_mirror_path) if ccd_mirror_path else frozenset()
biotite_codes = get_available_ccd_codes_in_biotite()
return mirror_codes | biotite_codes
```

`atom_array_from_ccd_code` (`ccd.py:501-502`) prefers the mirror and falls back to biotite
when a code is absent. Biotite ships the complete dictionary, so an empty
`CCD_MIRROR_PATH` costs you only components too new for your biotite release. The mirror
is a freshness optimisation.

**PDB: no, unless you are training.** In this repo `PDB_MIRROR_PATH` is read only by
dataset metadata-row parsers — `datasets/atomworks_default_metadata_row_parsers.py:106`
and `:162` — and referenced by `configs/dataset/unified/pdb_interfaces.yaml:6`. Nothing on
the design → generate → filter → evaluate → analyze path touches it.

## Neither is downloaded by `complexa download`

`complexa download` handles **model weights only**. The complete option set is
`--complexa`, `--complexa-ligand`, `--complexa-ame`, `--complexa-all`, `--pmpnn`,
`--ligmpnn`, `--af2`, `--esm2`, `--rf3`, `--all`, `--everything`, `--status`, `--help`
(`env/download_startup.sh:776-806`) — every one maps to a `download_*_weights()`
function. There is no CCD option, no PDB option, and the string "mirror" does not appear
in the script. See the flag matrix in
`.claude/skills/complexa-setup/reference/downloads.md`.

All of those are accepted by `complexa download` too (`cli_runner.py:1060-1133`). The
handler forwards `sys.argv[2:]` to the script verbatim (`cli_runner.py:2044`) rather than
reading the parsed values, so the argparse declarations exist only to gate and document —
which is why a flag missing there is rejected even though the script would have handled it.
Older installs lack the five per-model flags; see
[`troubleshooting.md`](troubleshooting.md#missing-community-model-path-esm_dir).

The **example target PDBs need no download either** — 81 `.pdb` files are tracked under
`assets/target_data/`, pre-cleaned and cropped. See [`pdb-prep.md`](pdb-prep.md).

---

## Building a CCD mirror

Only worth it for ligand-binder or AME work on components newer than your biotite release.

atomworks expects a **first-character shard**, uppercase codes:

```
$CCD_MIRROR_PATH/H/HEM/HEM.cif
                 0/000/000.cif
```

From `atomworks/io/utils/ccd.py:233`:

```python
return os.path.join(ccd_mirror_path, ccd_code[0], ccd_code, ccd_code + ".cif")
```

The scanner (`ccd.py:141-157`) requires the level-1 directory name to be exactly one
character, the level-2 name to be the code and to start with that character, and the file
to be `<CODE>.cif`. An optional `.ccd_codes_cache` at the root (one code per line) skips
the 48k-directory walk, but only while its mtime is newer than the root directory's
(`ccd.py:121-133`).

wwPDB's own per-component tree at `/pub/pdb/refdata/chem_comp/` shards by the **last**
character (`H/CH/CH.cif`, `H/00H/00H.cif`), so it is not drop-in compatible. Take the
monolithic dictionary and split it instead:

```bash
CCD=/data/mirrors/ccd
mkdir -p "$CCD"
curl -o /tmp/components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
python docs/binder-target-setup/scripts/build_ccd_mirror.py /tmp/components.cif.gz "$CCD"
rm /tmp/components.cif.gz
```

118 MB gzipped, ~1.3 GB expanded, ~48k components. `components.cif` is a concatenation of
`data_<CODE>` blocks, one per component, so a boundary split yields valid single-block
CIFs that `biotite.structure.io.pdbx.CIFFile.read` accepts.

Verify:

```bash
CCD_MIRROR_PATH=/data/mirrors/ccd python -c "
import os
from atomworks.io.utils.ccd import get_available_ccd_codes_in_mirror, get_ccd_component_from_mirror
p = os.environ['CCD_MIRROR_PATH']
print(len(get_available_ccd_codes_in_mirror(p)), 'codes')
print(get_ccd_component_from_mirror('HEM').array_length(), 'atoms in HEM')"
```

## Building a PDB mirror

Only needed for training. atomworks expects the middle-two-character shard, lowercase:

```
$PDB_MIRROR_PATH/ab/1abc.cif.gz
```

From `atomworks/ml/utils/testing.py:20` (and `io/utils/testing.py:37`):

```python
filename = f"{base_dir}/{pdbid[1:3]}/{pdbid}.cif.gz"
```

That is exactly RCSB's *divided* mmCIF layout, so rsync lands it in place unmodified. The
divided tree is **not** served over HTTPS — both `files.rcsb.org` and `files.wwpdb.org`
return 403 for it; only the flat `/download/<id>.cif.gz` endpoint is public over HTTP.

```bash
PDBM=/data/mirrors/pdb
mkdir -p "$PDBM"
rsync -rlpt -v -z --delete --port=33444 \
  rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ "$PDBM"
```

Two cautions: this is the **full archive** — order of 100 GB, hours to days on first sync,
so check free space — and it carries `--delete`, so point it at a dedicated directory and
nothing else. Re-run the same command to update incrementally. wwPDB also ships a wrapper
with logging and lockfiles at `https://files.wwpdb.org/pub/pdb/software/rsyncPDB.sh`.

For a handful of entries, skip rsync — the layout is trivial:

```bash
for id in 1abc 4hhb; do
  d="$PDBM/${id:1:2}"; mkdir -p "$d"
  curl -o "$d/$id.cif.gz" "https://files.rcsb.org/download/$id.cif.gz"
done
```

> Note that a `.cif` mirror means `label_seq_id` numbering downstream, not author
> numbering. See ["`.cif` and `.pdb` do not agree"](pdb-prep.md#cif-and-pdb-do-not-agree).

## Wiring it up

Point both at existing directories, or leave them empty:

```bash
CCD_MIRROR_PATH=/data/mirrors/ccd
PDB_MIRROR_PATH=/data/mirrors/pdb
```

Both variables are all-or-nothing: empty works, a valid directory works, and a non-empty
path to a missing directory crashes at import. There is no partial-credit mode.

## A related silent failure

`env/build_uv_env.sh:174` installs atomworks non-fatally:

```bash
uv pip install "atomworks[ml,openbabel,dev]" || echo "Warning: atomworks install failed"
```

So a build can report success with the package missing or partially installed. If
`import atomworks` fails but the build "passed", this is why. `rdkit` and the OpenBabel
bindings arrive only through those extras and are not declared in `pyproject.toml`.
