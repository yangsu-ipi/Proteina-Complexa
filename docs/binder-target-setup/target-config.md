# Target Configuration

How to define a design target. The shipped path is a shared 44-entry dict at
`configs/targets/targets_dict.yaml`; this document covers that plus three ways to keep a
target's definition next to the target's own data.

**Preferred: one YAML file per target.** Everything — pipeline settings and the target
definition — in a single file, in the target's own directory.

## Why not just edit the shared dict

It works, and for a target you will reuse across the team it is the right answer — see
"Protein target schema" in `.claude/skills/complexa-target/reference/target_schema.md` for
the field reference. Two properties make it awkward for one-off or per-user targets:

**All 44 entries use relative paths.** `configs/targets/targets_dict.yaml:14`:

```yaml
02_PDL1:
  target_path: ./assets/target_data/bindcraft_targets/PD-L1.pdb
```

**Relative to what?** To the shell's working directory. There is no `os.chdir` and no
`cwd=` anywhere in `cli_runner.py` or `generate.py` — `run_step` calls
`subprocess.run(cmd, check=True, env=env)` with no `cwd`, so the stage subprocess inherits
your shell's cwd. Hydra does not save you either: `hydra.run.dir` does not change the
process cwd, and `hydra.job.chdir` defaults to `False` in Hydra 1.3. Run from anywhere but
the repo root and every one of those 44 paths breaks.

## The one-file approach

```yaml
# /data/targets/MINE/pipeline.yaml  — the only file you need
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
      source: local              # required even though unused — see below
      target_filename: mine      # required even though unused — see below
      target_path: /data/targets/MINE/mine.pdb
      target_input: A1-115
      hotspot_residues: ["A37", "A39", "A49", "A98"]
      binder_length: [64, 155]
      pdb_id: null
```

Directory:

```
/data/targets/MINE/
├── pipeline.yaml     # everything
├── mine.pdb
├── inference/        # created by the run
└── logs/             # created by the run
```

```bash
cd /data/targets/MINE && complexa design ./pipeline.yaml --verbose
```

### Two requirements that are not obvious

**1. Pin `generation.task_name` — this one fails silently.**
`configs/pipeline/binder/binder_generate.yaml:16` defaults `task_name: 33_TrkA`. Because
the inline approach *merges*, `33_TrkA` is still present from the shared dict, so nothing
errors. You get a clean run against **a completely different target**:

```
targets       : 45
task_name     : 33_TrkA
pdb_path      : ./assets/target_data/alpha_proteo_targets/1www_cropped.pdb
input_spec    : X282-382
hotspots      : ['X294', 'X296', 'X333']
binder_length : 50 - 120
RESULT: PASS
```

Your target is loaded into the config and never used. Setting `task_name` under
`generation:` in the `_self_` block fixes it — `_self_` is last in the defaults list, so it
wins.

(In the *shadow* approach the same omission raises
`InterpolationKeyError: target_dict_cfg.33_TrkA.source`, because a replaced dict has no
`33_TrkA`. Replace fails loudly here; merge does not.)

**2. Include all seven keys — including `source` and `target_filename`.** They look
redundant next to `target_path`, and they are never used when `target_path` is set. Omit
them anyway and you get:

```
InterpolationKeyError: Interpolation key 'target_dict_cfg.99_MYTARGET.source' not found
  full_key: generation.dataloader.dataset.conditional_features[0].pdb_path
```

The reason is `configs/pipeline/binder/binder_generate.yaml:33`:

```yaml
pdb_path: ${oc.select:.....target_dict_cfg.${.task_name}.target_path,
           ${oc.env:DATA_PATH}/target_data/${...source}/${...target_filename}.pdb}
```

OmegaConf evaluates the **default argument** of `oc.select` regardless of whether the
primary key resolves. So a missing `source` raises even when `target_path` is present and
would have won. Values are arbitrary — `source: local` is fine.

This does not bite when you patch an *existing* entry like `02_PDL1`, because `source` and
`target_filename` merge in from the shared dict. It bites on every brand-new key.

`target_input`, `hotspot_residues`, `binder_length[0]`, `binder_length[1]`, and `pdb_id`
are plain interpolations with no `oc.select` guard at all
(`binder_generate.yaml:24-25`, `:34-37`), so those were always required.

### What this gives you

`generation.target_dict_cfg` ends up with 45 entries — yours **merged on top of** the
shared 44, not replacing them. Harmless, since `task_name` selects yours. If you need the
pipeline to be unable to see other targets, use the shadow file below.

Merge cuts both ways. A **new** target name fails loudly if a required key is missing —
`InterpolationKeyError` names the exact key. But a name that already exists in the shared
44 resolves to the shared entry, so both of these are silent: an unpinned `task_name`
(above), and a shadow file that failed to load (below). Pin `task_name`, and check the
entry count in the log.

---

## Alternatives

| Approach | Semantics | Files | Failure mode | Use when |
|---|---|---|---|---|
| **Inline in `_self_`** | merge, 45 targets | 1 | loud (`InterpolationKeyError`) | **default** |
| Shadow `targets/targets_dict.yaml` | replace, 1 target | 2 | **silent** fallback to 44 | strict isolation required |
| Own defaults entry | merge, free naming | 2 | loud | per-target deltas, custom layout |
| CLI `++` override | merge, per-invocation | 0 | silent if key typo'd | one-off, no file edit |

### Shadow file — the only way to fully replace the dict

Put a `targets/targets_dict.yaml` next to your `pipeline.yaml`:

```
/data/targets/MINE/
├── pipeline.yaml
└── targets/
    └── targets_dict.yaml     # exact name required
```

This works because the config's own directory becomes Hydra's **primary config dir**, and
the primary dir shadows anything reachable via `hydra.searchpath`.
`src/proteinfoundation/cli/cli_runner.py:646` builds every stage subprocess as
`--config-path <config_file.parent.absolute()> --config-name <config_file.stem>`.

Result: `target_dict_cfg` contains **only** your target.

**Both names are hardcoded and not overridable.** They come from the defaults entry at
`configs/pipeline/binder/binder_generate.yaml:8`:

```yaml
- /targets/targets_dict@_here_
#  ^^^^^^^ group = directory   ^^^^^^^^^^^^ option = filename
```

Every plausible override key is rejected — the `@_here_` package keyword appears to make
the entry unaddressable:

```
targets@generation=pdl1              → Could not override 'targets@generation'. No match in the defaults list.
targets=pdl1                         → Could not override 'targets'. No match…
targets@_here_=pdl1                  → Could not override 'targets@_here_'. No match…
- override /targets@generation: pdl1 → In 'pipeline': Could not override 'targets@generation'. No match…
```

**The silent fallback is the hazard.** A wrong directory name (`mytargets/`) or wrong
filename (`pdl1.yaml`) does not error. Verified, shadowing `02_PDL1` from a `mytargets/`
directory:

```
targets       : 44
pdb_path      : ./assets/target_data/bindcraft_targets/PD-L1.pdb
RESULT: PASS
```

You get the shared dict and its **relative** path — which then resolves against your shell
cwd, so the run either uses the repo's stock PDB or dies on a missing file far downstream.

The severity depends on your target name: shadowing an **existing** name (`02_PDL1`) is
silent, as above; a **new** name (`99_MYTARGET`) raises `InterpolationKeyError` because the
shared 44 do not contain it. So the silent case is precisely the one where you meant to
*replace* a shipped target's settings.

Confirm it took: the generate log prints

```
cfg_gen: {'target_dict_cfg': '<filtered: 44 entries>', ...}
```

`1 entries` means the shadow took. `44 entries` means it did not.

### Own defaults entry — free naming, merge semantics

Write the entry yourself and any directory and filename work:

```yaml
defaults:
  - pipeline/binder/binder_generate@generation
  - /my_target_specs/pdl1@generation     # ← reads my_target_specs/pdl1.yaml
  - _self_
```

Place it **after** `binder_generate@generation` (later entries win) and **before**
`_self_` (so your inline block still has the last word). This merges — all 44 shared
entries remain, your fields win for the keys you specify.

### CLI override — no file edit

```bash
++target_dict_cfg.02_PDL1.target_path=/abs/path/PD-L1.pdb
```

Overriding the dict rather than the resolved leaf is the right level: `complexa design`
applies overrides to **all four stages**, and this hits the `oc.select` first branch so it
propagates into `pdb_path` automatically. The narrower
`++generation.dataloader.dataset.conditional_features.0.pdb_path=...` patches only the
generation dataloader and silently targets the wrong element if the list order changes.

There is no dedicated flag — `complexa design` accepts only `config`, `--verbose`,
`--steps`, and free-form Hydra overrides. For a permanent fix,
`complexa target add <name> --target-path /abs/... --force` has the flag you want
("Full path to target PDB file"). On `++` prefix semantics see "Override key not
recognized" in `.claude/skills/complexa-design/reference/troubleshooting.md`.

---

## Mechanics worth knowing

### `@package` is not a stage gate

`- /my_specs/pdl1@generation` names a *package* — where in the config tree the file lands
— not which stage uses it. All four stages compose the same file
(`STEP_MODULES`, `cli_runner.py:101-105`), so **one defaults line covers the pipeline**.

Evaluate and analyze never read `target_dict_cfg` anyway. The only Python references in
the package are a log filter (`utils/config_utils.py:9`), an analysis exclude list
(`result_analysis/analysis_utils.py:55`), and `cli/validate.py:286`. Target and binder
chain information reaches the later stages through the PDBs and manifests that `generate`
writes.

### Outputs follow cwd, not the config

Both output roots are hardcoded relative paths with no config key:

```python
# src/proteinfoundation/generate.py:65
root_path = f"./inference/{config_name}_{task_name}"   # + f"_{run_name}" if set
# src/proteinfoundation/cli/cli_runner.py:128
LOG_DIR = Path("./logs")
```

So `cd` into the target directory is what makes it self-contained. Passing an absolute
config path is enough for config resolution; the `cd` is what places `inference/` and
`logs/`.

### Keep target directories outside the repo

`/data/targets/<name>/` rather than a subdirectory of this checkout. Inside the repo,
`.gitignore` does the opposite of what you want:

| Path | Status | Consequence |
|---|---|---|
| `mytargets/X/PD-L1.pdb` | **ignored** (`*.pdb`) | your input is silently untracked |
| `mytargets/X/inference/results.csv` | **not** ignored (`/inference` is root-anchored) | run outputs get committed |
| `mytargets/X/logs/` | ignored (`logs/`) | correct |
| `mytargets/X/pipeline.yaml` | not ignored | correct |

The 81 PDBs under `assets/target_data/` survive only because they are already tracked.
If a target directory must live inside the checkout, `git add -f` the PDB and add a nested
`inference/` rule. Commit `0fb1458` fixed a sibling instance of this class for
`.claude/*` + `!.claude/skills/`.

### Per-pipeline names

| Pipeline | Defaults entry | Config key | Shadow file |
|---|---|---|---|
| Protein binder | `/targets/targets_dict@_here_` (`binder_generate.yaml:8`) | `target_dict_cfg` | `targets/targets_dict.yaml` |
| Ligand binder | `/targets/ligand_targets_dict@_here_` (`ligand_binder_generate.yaml:6`) | `target_dict_cfg` | `targets/ligand_targets_dict.yaml` |
| AME / enzyme | `/design_tasks/ame_dict_v2@_here_` (`ame_generate.yaml:7`) | `motif_target_dict_cfg` | `design_tasks/ame_dict_v2.yaml` |

One outlier: `configs/evaluate_from_pdb_dir.yaml:22` uses group `generation`, not
`targets`, so a self-contained directory used with `complexa analysis` needs
`generation/targets_dict.yaml`. The four-stage `design` path is consistent.

### `hydra.searchpath`

What lets a thin external YAML reach `pipeline/binder/*` inside the shared install. Legal
only in the primary config or on the command line. Without it, the `defaults:` entries do
not resolve.
