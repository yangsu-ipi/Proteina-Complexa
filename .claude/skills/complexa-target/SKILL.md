---
name: complexa-target
description: Use this skill whenever the user wants to add, register, edit, list, show, or validate a Proteina-Complexa design target for any pipeline — protein binder (default), ligand binder, or AME / enzyme scaffolding. Triggers include "add a target", "define a new target for binder design", "register a hotspot", "set up a PDL1 binder target", "ligand binder pocket", "SMILES target", "AME task", "enzyme motif", "M0024_1nzy", "complexa target add", "complexa target show", "configure target X", "what targets are available", "where do hotspots live", "what does target_input mean", "chain-spec syntax", "binder length range", or any question about `configs/targets/{,ligand_}targets_dict.yaml` and `configs/design_tasks/ame_dict_v2.yaml`. Also covers `complexa validate target`. This is the only skill that touches the three targets dict files.
allowed-tools: Bash, Read, Write, AskUserQuestion
---

# complexa-target

Add or edit a design target in Proteina-Complexa. Targets live in **three YAML files**, one per `complexa design` pipeline:

- `configs/targets/targets_dict.yaml` — protein binder (**default pipeline**)
- `configs/targets/ligand_targets_dict.yaml` — ligand binder
- `configs/design_tasks/ame_dict_v2.yaml` — AME / enzyme scaffolding

The `complexa target` CLI can manage the first two, but it **always writes `configs/targets/targets_dict.yaml`** unless you pass `--dict PATH` — there is no ligand-aware routing (`target_manager.py:24`, `get_default_dict_path()` at `:130-148`). For a ligand target you must pass `--dict configs/targets/ligand_targets_dict.yaml`, or the entry lands in the protein dict where the protein pipeline reads it (`binder_generate.yaml:8`) and the ligand pipeline never sees it. AME tasks use an extended schema (the same core fields as ligand targets plus `contig_atoms` for hand-curated per-residue motif atom selections) and are file-edit-only.

## Preferred path: edit the YAML directly

`complexa target add` is a thin wrapper around "load YAML → build dict → append a block" (see `src/proteinfoundation/cli/target_manager.py:add_target_cli` and `append_target_to_dict`). For agentic use, **just edit the targets dict directly** — the schema is short, the existing entries are great copy templates, and you skip 14 CLI flags / SMILES shell-escaping. Use the CLI only if you explicitly want its YAML auto-quoter or its overwrite-prompt safety.

The skill therefore presents both paths:

| Step | Direct file edit (preferred) | CLI |
|---|---|---|
| Look at existing targets | Read `configs/targets/{,ligand_}targets_dict.yaml` (or `configs/design_tasks/ame_dict_v2.yaml` for AME) | `complexa target list [--dict PATH]` — one dict per call, default `targets_dict.yaml` |
| Look at one target | Read the dict and grep for the key | `complexa target show NAME [--dict PATH]` |
| Add a new target | Append a YAML block (Step 3a) | `complexa target add ...` (Step 3b) |
| Verify the PDB resolves on disk | `ls` the path built from `target_path`, or from `source` + `target_filename` (Step 4) | — *(`complexa validate target` cannot resolve the dict on any config in this repo — see Step 4)* |

`complexa target` reads and writes exactly one dict per invocation: `configs/targets/targets_dict.yaml` by default, or whatever `--dict PATH` names. It does not discover `ligand_targets_dict.yaml` on its own. For AME tasks (which use a different schema), file-edit is the only path — see Step 1 below.

## What this skill enables

- Register a **protein target** (chain + residue range + hotspots) in `configs/targets/targets_dict.yaml`.
- Register a **ligand target** (PDB pocket + 3-letter code + SMILES) in `configs/targets/ligand_targets_dict.yaml` — via the CLI this requires an explicit `--dict configs/targets/ligand_targets_dict.yaml`.
- Resolve the right **AME task name** (e.g. `M0024_1nzy`, `M0096_1chm`) for the AME pipeline — these are not added via `complexa target add` (see Step 1).
- Confirm a target resolves to a real PDB on disk by reading the entry back and checking the path (Step 4 — `complexa validate target` is currently non-functional).
- Emit a replayable artifact (`target_definition.yaml`) for downstream design runs.

## Step 1: Decide the target type

Targets live in **three different files**, one per `complexa design` pipeline. Pick the file by working out which pipeline the user will run, then add the entry to that file. Each file is consumed by exactly one pipeline.

| User intent | Pipeline | Targets dict file | This skill? |
|---|---|---|---|
| Bind a protein surface (PD-L1, IFNAR2, TNF-α, …) — **default** | `configs/search_binder_local_pipeline.yaml` | `configs/targets/targets_dict.yaml` | Yes — Step 2 (protein) |
| Bind a small-molecule pocket (FAD, SAM, OQO, …) | `configs/search_ligand_binder_local_pipeline.yaml` | `configs/targets/ligand_targets_dict.yaml` | Yes — Step 2 (ligand) |
| AME / enzyme scaffolding (motif + ligand, `M####_<pdb>` names) | `configs/search_ame_local_pipeline.yaml` | `configs/design_tasks/ame_dict_v2.yaml` | Partial — see "AME tasks" below |

### AME tasks (enzyme pipeline)

Similar story — AME tasks live in `configs/design_tasks/ame_dict_v2.yaml` under `motif_target_dict_cfg:` with their own schema (`source`, `target_filename`, `ligand`, `contig_atoms`, `binder_length`, `hotspot_residues`, and usually `target_path`). No AME entry carries `pdb_id`, `ligand` may be a **list** (`["ADP", "MG", "3PG"]`) or a `"<resname>:<resnum>"` string (`"L:0"`), and the `use_bonds_from_file` key every entry carries has **no effect** — `ame_generate.yaml:33-36` builds `LigandFeatures` from `task_name`, `pdb_path`, `ligand` only. The `contig_atoms` string encodes per-residue motif atom selections like `"A64: [O, C]; A86: [CB, CA, N, C]; ..."` — these are hand-curated, so adding a new AME task is a file edit, not a CLI invocation. Browse the file, copy a similar entry as a template, and pass `++generation.task_name=<NAME>` to `complexa design configs/search_ame_local_pipeline.yaml`.

## Step 2: Gather required info

Use AskUserQuestion to collect — do not guess these. Required fields differ for protein vs ligand.

### Protein target

| Field | Question | Example | Required |
|---|---|---|---|
| name | "Target name (used as the dict key and `task_name`)?" | `50_PDL1_custom`, `MyTarget_v1` — must not already be a key in the dict | yes |
| source | "Source directory under `$DATA_PATH/target_data/`?" | `bindcraft_targets`, `custom_targets` | yes (or `target_path`) |
| target_filename | "PDB filename (no `.pdb` extension)?" | `PD-L1`, `IFNAR2` | yes (or `target_path`) |
| target_input | "Chain + residue range — see reference for grammar." | `A1-115`, `A1-50,B1-50` | yes |
| hotspot_residues | "Hotspot residues (interface contact residues)?" | `["A33", "A95", "A102"]` | **yes as a key** (`[]` is allowed; recommended to fill in) |
| binder_length | "Binder length range `[min, max]`?" | `[80, 150]` | **yes, two elements** — see warning below |
| pdb_id | "Reference PDB ID (optional, metadata only)?" | `"2lag"` | **yes as a key** (value may be `null`) |

> **A protein `binder_length` must have exactly two elements.** `binder_generate.yaml:24-25`
> indexes both `binder_length[0]` and `binder_length[1]` with no `oc.select` guard, so
> `[100]` dies in Hydra interpolation. (The ligand and AME configs *do* guard the second
> element — `ligand_binder_generate.yaml:23`, `ame_generate.yaml:24` — so a single-element
> list is fine there.) The "optional / has a default" story applies only to
> `complexa target add`, which fills gaps in the writer (`target_manager.py:1004-1015`). A
> **hand-written** entry must carry `target_input`, `hotspot_residues`, `binder_length` and
> `pdb_id` explicitly: `binder_generate.yaml:34-37` interpolates all of them unguarded, and a
> missing key is a hard `InterpolationKeyError`, not a default.

### Ligand target — protein fields above (minus `target_input`), plus:

| Field | Question | Example | Required |
|---|---|---|---|
| ligand | "3-letter PDB ligand residue code?" | `FAD`, `OQO`, `SAM` | yes (presence marks target as ligand) |
| smiles | "SMILES string for the ligand?" | `"O=C2C3=Nc1cc(c(...)..."` | **yes — always pass it** (see warning) |
| ligand_only | "Use the *entire input file* as the ligand (`true`), or extract the named residues from it (`false`)?" | `true` / `false` | **yes** (CLI writes `true`) |
| use_bonds_from_file | "Use bond info from the input PDB/CIF?" | `true` / `false` | **yes** (CLI writes `true`) |
| target_input | not used by the ligand pipeline | — | no |

> **`ligand_only` is not "pocket-only mode".** `gen_dataset.py:620-621` and `:722-731`:
> `true` = *use the whole file as the ligand* (the `ligand` residue names are ignored);
> `false` = *extract the residues named in `ligand`*. Setting `true` on a full
> protein-complex PDB silently treats the entire complex as the ligand. `false` requires at
> least one residue name, or `gen_dataset.py:662-667` raises. The live entries pair
> `ligand_only: True` with pre-trimmed `*_ligand_centered.pdb` files.
>
> **`SMILES` must be present.** `target_manager.py:1024-1025` writes the key only `if smiles`,
> so omitting `--smiles` leaves it **absent** — not `null` — and
> `ligand_binder_generate.yaml:33` interpolates it unguarded, breaking the ligand pipeline.
> Same for `ligand`, `ligand_only` and `use_bonds_from_file` (`:31-34`): all four are
> required keys at runtime, whatever the CLI's own defaults suggest.

Check for name collisions by reading the dict you are about to write to — `rg '^  NEW_NAME:' configs/targets/targets_dict.yaml` (or `ligand_targets_dict.yaml`). Do **not** use `complexa target list --ligand` for this: `--ligand`/`--protein` filter *within* the single dict that was loaded (`target_manager.py:495-496`), and `targets_dict.yaml` has zero entries with a `ligand:` key, so it always prints "No targets found".

## Step 3a: Append the YAML block directly (preferred)

Open `configs/targets/targets_dict.yaml` (or `ligand_targets_dict.yaml` for ligand targets), find a similar existing entry as a style template, and append the new block under `target_dict_cfg:`. Two-space indent, single blank line between entries.

### Protein template

```yaml
  50_PDL1_custom:
    source: bindcraft_targets
    target_filename: PD-L1
    target_path: ./assets/target_data/bindcraft_targets/PD-L1.pdb
    target_input: A1-115
    hotspot_residues: ["A37", "A39", "A49", "A98"]
    binder_length: [64, 155]
    pdb_id: null
```

Pick a name that is **not** already a key in the file — every live entry is listed in
`configs/targets/targets_dict.yaml` (`01_PD1` … `38_TNFalpha_REPACK`).

Rules to match the existing file style:

- Single-segment `target_input` values are **unquoted** on disk (`targets_dict.yaml:6, 15, 24, 33`: `target_input: A1-115`). Only multi-segment values are quoted, because the comma needs it (`:213`: `target_input: "A96-174,A306-446"`). Both parse identically — quote if you like, but the live style is unquoted for one segment. (`complexa target add` quotes every chain/residue string, so CLI-written and hand-written entries look slightly different; `target_manager.py:330-365`.)
- Always quote SMILES — brackets, `:` and `%` will otherwise break the parse.
- Use flow-style lists (`["A33", "A95"]`, `[64, 155]`) — that matches both the live file and the CLI dumper.
- Every live entry carries `target_path:` alongside `source` + `target_filename` (all 44 protein entries), and it wins: `binder_generate.yaml:33` reads `${oc.select:...target_path, <$DATA_PATH-derived path>}`. Include it when the PDB is in the repo (`./assets/target_data/...`) or outside `$DATA_PATH/target_data/`.
- `target_input`, `hotspot_residues`, `binder_length` and `pdb_id` are all required *keys* for a hand-written protein entry (see the warning in Step 2), plus either `target_path` or `source` + `target_filename`.

### Ligand template

```yaml
  43_7BKC_LIGAND:
    source: ligand_targets
    target_filename: 7BKC_ligand_centered
    target_path: ./assets/target_data/ligand_targets/7BKC_ligand_centered.pdb
    hotspot_residues: [null]
    binder_length: [100]
    pdb_id: 7BKC
    ligand: 'FAD'
    ligand_only: True
    SMILES: "O=C2C3=Nc1cc(c(cc1N(C3=NC(=O)N2)CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC6OC(n5cnc4c(ncnc45)N)C(O)C6O)C)C"
    use_bonds_from_file: True
```

This block goes in **`configs/targets/ligand_targets_dict.yaml`**, not `targets_dict.yaml` — that is the file `ligand_binder_generate.yaml:6` composes. Again, pick a name that is not already a key there (`39_7V11_LIGAND` … `42_7C7M_LIGAND`).

The presence of the `ligand:` key flips the target into ligand mode — there is no separate `is_ligand` flag. `target_input` is not required for ligands. A single-element `binder_length` *is* fine here (`ligand_binder_generate.yaml:23` guards the second element with `oc.select`).

After saving, skip to Step 4 to verify.

## Step 3b: CLI alternative (`complexa target add`)

Use the CLI when you want its automatic chain/residue quoting, the overwrite-confirm prompt, or to wire target creation into a non-Python script. For agentic use always pass `name` and never pass `-i`: `-i, --interactive` and `-e, --editor NAME` are two **separate** flags (`cli_runner.py:1194-1205`), interactive mode triggers on `args.interactive or not args.name` (`:1819`), so `-i` (or a missing name) blocks on an editor, while `-e vim` on its own with a name is silently discarded.

### Confirmed flags (from `src/proteinfoundation/cli/cli_runner.py:1177-1321`)

| Flag | Type | Applies to | Notes |
|---|---|---|---|
| `name` (positional) | str | both | dict key |
| `--dict PATH` | path | both | target dict to read/write. **Defaults to `configs/targets/targets_dict.yaml`** — pass `configs/targets/ligand_targets_dict.yaml` for ligand targets |
| `-i, --interactive` | flag | both | open `$EDITOR` |
| `-e, --editor NAME` | str | both | `code`, `nano`, `vim`, `cursor`, ... |
| `--source NAME` | str | both | directory under `$DATA_PATH/target_data/` |
| `--target-filename NAME` | str | both | PDB stem (no `.pdb`) |
| `--target-path PATH` | str | both | full path; overrides `source`+`filename` |
| `--target-input SPEC` | str | protein | chain/residue range, e.g. `A1-115` |
| `--hotspot-residues R [R ...]` | list | both | e.g. `A33 A95` |
| `--binder-length N [N ...]` | int list | both | two ints (`min max`) for protein; a single int is only valid for ligand/AME targets |
| `--pdb-id ID` | str | both | metadata only |
| `--ligand [CODE]` | str, optional value | ligand | presence marks ligand target; `nargs="?"` with `const="YOUR_LIGAND"`, so the bare flag is legal and writes that placeholder |
| `--ligand-only` | flag | ligand | use the entire input file as the ligand (see Step 2) |
| `--smiles STR` | str | ligand | SMILES for the ligand. **Always pass it** — omitted means the key is absent, not `null` |
| `--use-bonds-from-file` | flag | ligand | use PDB bonds (no effect for AME — that pipeline ignores the key) |
| `-f, --force` | flag | both | overwrite existing without prompt |

### Protein example

```bash
complexa target add 50_PDL1_custom \
  --source bindcraft_targets \
  --target-filename PD-L1 \
  --target-input A1-115 \
  --hotspot-residues A37 A39 A49 A98 \
  --binder-length 64 155 \
  --pdb-id 4z18
```

The name must be new. `02_PDL1` and friends already exist in `configs/targets/targets_dict.yaml`, and without `-f` a collision makes `target_manager.py:1035-1044` prompt on `input()` — which raises `EOFError` non-interactively, cancels the add, and exits 1 (`cli_runner.py:1909-1910`).

### Ligand example

```bash
complexa target add 43_7BKC_LIGAND \
  --dict configs/targets/ligand_targets_dict.yaml \
  --source ligand_targets \
  --target-filename 7BKC_ligand_centered \
  --pdb-id 7BKC \
  --ligand FAD \
  --ligand-only \
  --use-bonds-from-file \
  --smiles "O=C2C3=Nc1cc(c(cc1N(C3=NC(=O)N2)CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC6OC(n5cnc4c(ncnc45)N)C(O)C6O)C)C" \
  --binder-length 100
```

`--dict configs/targets/ligand_targets_dict.yaml` is **not optional**. Without it the entry goes to the default `configs/targets/targets_dict.yaml` (`target_manager.py:24`, `:130-148`), i.e. into the *protein* dict that `binder_generate.yaml:8` composes — the ligand pipeline reads `ligand_targets_dict.yaml` (`ligand_binder_generate.yaml:6`) and would never see it.

Why these defaults: `--source` defaults to `custom_targets` (protein) or `ligand_targets` (ligand) and `--target-filename` defaults to `name`, so be explicit when the PDB stem differs from the target name. The dict gets `ligand_only: true` and `use_bonds_from_file: true` by default for ligand targets, but `SMILES` is written **only when `--smiles` is given** (`target_manager.py:1024-1025`) — always pass it.

## Step 4: Verify

Run both checks. Neither uses `complexa validate target` — see the note below.

```bash
# 1. Confirm the entry landed and looks right.
#    Direct read works just as well: `rg -A 8 '^  50_PDL1_custom:' configs/targets/targets_dict.yaml`
complexa target show 50_PDL1_custom

# 2. Resolve the PDB path the same way the pipeline does, and confirm it exists.
#    Mirrors the oc.select in binder_generate.yaml:33 (target_path wins, else
#    $DATA_PATH/target_data/<source>/<target_filename>.pdb).
python3 - <<'EOF'
import os, pathlib, yaml
NAME = "50_PDL1_custom"
DICT = "configs/targets/targets_dict.yaml"   # ligand_targets_dict.yaml for ligand targets
cfg = yaml.safe_load(open(DICT))["target_dict_cfg"][NAME]
path = cfg.get("target_path") or (
    f"{os.environ['DATA_PATH']}/target_data/{cfg['source']}/{cfg['target_filename']}.pdb"
)
print(path, "->", "EXISTS" if pathlib.Path(path).exists() else "MISSING")
for k in ("target_input", "hotspot_residues", "binder_length", "pdb_id"):
    print(f"  {k}: {cfg.get(k, '<<ABSENT — will break Hydra interpolation>>')}")
EOF
```

> **`complexa validate target CONFIG --target NAME` cannot succeed on any config in this
> repo — do not use it as a gate.** `validate.py:183-187` loads the config with a plain
> `yaml.safe_load` and never composes Hydra defaults; `validate.py:315-317` only follows a
> `defaults:` entry that is a **dict** containing `"generation"`, but every entry in
> `search_binder_local_pipeline.yaml:12-16` is a plain string
> (`pipeline/binder/binder_generate@generation`); and the fallback at `validate.py:359-361`
> looks for `configs/generation/targets_dict.yaml`, which does not exist. So
> `target_dict_cfg` is always `None` and the command reports
> `✗ Target config: Could not find target_dict_cfg in config` (`validate.py:486-490`)
> regardless of whether your entry is correct. The `validate` subparser also takes no Hydra
> overrides (`cli_runner.py:1326-1348`: only `type`, `config`, `--target`).

For ligand targets, run the same script against `configs/targets/ligand_targets_dict.yaml` and additionally confirm the entry has all four of `ligand`, `ligand_only`, `SMILES`, `use_bonds_from_file` — `ligand_binder_generate.yaml:31-34` interpolates each unguarded.

## Step 5: Emit artifact

Save a replayable record under `./target_<name>/`:

```bash
mkdir -p target_50_PDL1_custom
complexa target show 50_PDL1_custom > target_50_PDL1_custom/target_show.txt
```

Then write the appended YAML snippet (the lines `complexa target add` wrote under `target_dict_cfg:`) to `target_50_PDL1_custom/target_definition.yaml` using the Write tool. The artifact lets the user diff target definitions across runs and re-create the entry on another checkout via `complexa target add ... --force`.

## Hardware requirements

None for target definition — this is a YAML edit, not a training/inference step. Disk impact is a few KB appended to the targets dict. **There is no backup on a normal add**: a new target goes through `append_target_to_dict` (`target_manager.py:372-404`), a plain append. The `.yaml.bak` copy is written by `save_targets_dict` (`:245-265`), which only runs on the overwrite path (existing name + `-f`/confirmed prompt) — so back the file up yourself if you are overwriting something you care about, or just use git.

For the downstream design / evaluate runs that consume the target, defer to `complexa-design` and `_shared/reference/hardware.md`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MISSING` from the Step 4 path check (or a `FileNotFoundError` at generate time) | neither `target_path` nor `$DATA_PATH/target_data/<source>/<target_filename>.pdb` exists | Confirm the PDB stem and source dir. Add a `target_path:` line (every live entry has one) or use `--target-path /full/path.pdb` if the file lives outside `target_data/`. |
| `⚠️  Target 'X' already exists!` then `Overwrite? (y/N):` | Name collision in the dict being written (`target_manager.py:1035-1044`) | Pick a new name, or pass `-f / --force`. Non-interactively the `input()` raises `EOFError`, the add is cancelled and `cli_runner.py:1909-1910` exits 1 — so always check the name first. |
| Hotspot residue not in PDB | Wrong chain or residue number | Open the PDB, re-check chain letters (case-sensitive) and residue indices. Hotspots use the format `<CHAIN><RESNUM>` — see reference. |
| Chain not found | `target_input` references a chain that does not exist in the PDB | Inspect the PDB with `grep "^ATOM" target.pdb \| awk '{print $5}' \| sort -u`. |
| Ligand code missing from PDB | The 3-letter `ligand` code does not appear as a `HETATM` residue name in the file | Open the PDB and check `HETATM` lines; you may need a `_ligand_centered` variant of the PDB. |
| SMILES parse failure downstream | Bad SMILES string | Validate with `rdkit.Chem.MolFromSmiles(smiles)` before adding. Quote SMILES in shell to escape brackets and parens. |
| Target name with leading digit breaks Hydra interpolation | YAML treats `02_PDL1` as a string fine; some override syntaxes need quoting | Use `++generation.task_name=02_PDL1`; quote if the shell strips characters. |
| `target_input` appears ignored for a ligand target | By design — ligand targets do not use `target_input`; pocket is defined by the ligand | Leave it unset for ligand targets. |

## Reference

- `reference/target_schema.md` — every field, chain-spec grammar, AME task-name grammar, three worked examples.
- `configs/targets/targets_dict.yaml` — live protein entries (copy a known-good one as a template).
- `configs/targets/ligand_targets_dict.yaml` — live ligand entries.
- `configs/design_tasks/ame_dict_v2.yaml` — AME task definitions (file-edit only, not exposed via `complexa target` CLI).
- `src/proteinfoundation/cli/cli_runner.py:1177-1321` — argparse source of truth for `complexa target`. (`src/proteinfoundation/cli/target_cli.py` backs the *separate* `complexa-target` console script declared at `pyproject.toml:73`; its flags are not the ones `complexa target` parses.)
- `src/proteinfoundation/cli/target_manager.py` — `add_target_cli`, `list_targets`, `show_target`, schema in `TARGET_FIELDS`, and the CLI-only defaults at `:1004-1030`.
- `src/proteinfoundation/cli/validate.py` — `validate_target` implementation (currently cannot resolve `target_dict_cfg`; see Step 4).
