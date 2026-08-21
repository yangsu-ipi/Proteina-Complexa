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
(`src/proteinfoundation/generate.py:643`) — and it drags in four modules nothing in
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
