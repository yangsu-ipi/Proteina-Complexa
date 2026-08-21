# atomworks Mirror Environment Variables

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
