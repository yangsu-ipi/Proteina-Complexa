# Target schema reference

Authoritative source: `src/proteinfoundation/cli/target_manager.py::TARGET_FIELDS` and the live YAML dicts under `configs/targets/`. This document mirrors them.

## Protein target schema

Stored in `configs/targets/targets_dict.yaml` under `target_dict_cfg.<name>`. Either (`source` + `target_filename`) OR `target_path` is required, plus `target_input`.

| Field | Type | Required | Default | Example | Controls |
|---|---|---|---|---|---|
| `source` | str | yes* | `custom_targets` | `bindcraft_targets` | Subdirectory of `$DATA_PATH/target_data/` where the PDB lives. |
| `target_filename` | str | yes* | (name of target) | `PD-L1` | PDB stem (no `.pdb` extension). Combined with `source` to build the path. |
| `target_path` | str | yes* | `null` | `/abs/path/target.pdb` | Full path to a PDB. If set, `source`+`target_filename` are optional. |
| `target_input` | str | **yes** | CLI writes `A1-100` | `A1-115`, `A1-50,B1-50` | Chain + residue range (the "target_input" chain-spec). |
| `hotspot_residues` | list[str] | **yes** (key) | CLI writes `[]` | `["A33", "A95", "A102"]` | Interface residues focused on during binder design. |
| `binder_length` | list[int] | **yes**, exactly 2 elements | CLI writes `[60, 120]` | `[80, 150]` | Binder length, sampled uniformly in `[min, max]`. |
| `pdb_id` | str \| null | **yes** (key) | CLI writes `null` | `"2lag"` | Reference PDB ID, metadata only. |

(* "yes" with the OR rule: either `target_path` OR (`source` AND `target_filename`). In practice all 44 live entries carry `target_path` *and* `source`/`target_filename`; `binder_generate.yaml:33` prefers `target_path` via `oc.select`.)

> **The "Default" column is the CLI writer, not a runtime default.**
> `target_manager.py:1004-1015` fills these in when `complexa target add` builds the entry.
> `binder_generate.yaml:34-37` then interpolates `target_input`, `hotspot_residues` and
> `pdb_id` — and `:24-25` interpolates `binder_length[0]` *and* `binder_length[1]` — with **no
> `oc.select` guard**. A hand-written entry that omits any of them raises
> `InterpolationKeyError` at compose time, and a single-element `binder_length` such as
> `[100]` fails on `binder_length[1]`. Protein `binder_length` must therefore have exactly two
> elements. (The ligand and AME configs guard the second element —
> `ligand_binder_generate.yaml:23`, `ame_generate.yaml:24` — so `[180]`-style single values
> are legal there and are in fact what all 44 AME entries use.)

## Ligand target schema

Stored in `configs/targets/ligand_targets_dict.yaml`. The presence of the `ligand` key is what marks an entry as a ligand target (no separate `is_ligand` flag). `target_input` is not used for ligand targets.

All protein fields above (with `target_input` optional / omitted), plus:

| Field | Type | Required | Default | Example | Controls |
|---|---|---|---|---|---|
| `ligand` | str OR list[str] | **yes** | — | `"FAD"`, `["DHZ", "ZN"]`, `"L:0"` | PDB residue name(s) for the ligand. Presence marks target as ligand. **Ignored when `ligand_only: true`.** |
| `ligand_only` | bool | **yes** (key) | CLI writes `true` | `true` | `true` = use the **entire input file** as the ligand; `false` = extract only the residues named in `ligand`. |
| `SMILES` | str | **yes** (key) | *no default — see below* | `"O=C2C3=Nc1cc(...)C"` | SMILES string for the ligand (single-ligand targets only). Used to regenerate bonds when `use_bonds_from_file: false`. |
| `use_bonds_from_file` | bool | **yes** (key) | CLI writes `true` | `true` | If `true`, use bond information from the input PDB/CIF file rather than inferring from coordinates. |

> **`ligand_only` does not mean "pocket-only mode".** `gen_dataset.py:620-621` documents it as
> "use the entire file as the ligand (True) or extract specific residues by name (False)", and
> `:722-731` implements exactly that: `True` returns the whole loaded complex as one ligand
> group and never looks at `ligand`. Set `true` on a full protein-complex PDB and the entire
> complex is treated as the ligand. `false` requires at least one residue name — with none,
> `gen_dataset.py:662-667` raises. The live entries pair `ligand_only: True` with pre-trimmed
> `*_ligand_centered.pdb` files.
>
> **All four keys are required at runtime.** `ligand_binder_generate.yaml:31-34` interpolates
> `ligand`, `ligand_only`, `SMILES` and `use_bonds_from_file` unguarded (and `binder_generate.yaml:34-37`
> does the same for the protein keys), so a missing key is an `InterpolationKeyError`, not a
> default. `SMILES` is the easy one to get wrong: `target_manager.py:1024-1025` writes it only
> `if smiles`, so `complexa target add` **without `--smiles` omits the key entirely** and the
> resulting entry breaks the ligand pipeline. Always pass `--smiles`, or add the line by hand.

Note: `hotspot_residues` is present on all four live ligand entries as `[null]`, but nothing reads it on the ligand path — `hotspot_residues` is interpolated only by `binder_generate.yaml:35` (protein). Keep it for consistency; it is not load-bearing here. `target_input` is likewise unused by the ligand pipeline.

## `target_input` chain-spec grammar

`target_input` selects which residues of the target PDB are exposed to the binder design model.

| Form | Meaning | Example |
|---|---|---|
| `<CHAIN><START>-<END>` | One contiguous chain segment | `A1-115` = chain A, residues 1 through 115 |
| `<CHAIN><START>-<END>,<CHAIN><START>-<END>` | Multiple chain segments (comma-separated) | `A1-50,B1-50` = chain A residues 1-50 plus chain B residues 1-50 |
| `<CHAIN><START>-<END>/0 <NRES>-<NRES>` | Chain break + contigs syntax (RFdiffusion-style) | `A1-115/0 50-100` = target A1-115 plus a 50-100 residue binder placeholder |

`<CHAIN>` is a single uppercase letter (case-sensitive — `A` and `a` are different). `<START>` and `<END>` are integer residue numbers as they appear in the PDB (not 0-indexed; respects insertion codes only if the PDB does).

When is `target_input` required vs optional?

- **Protein targets**: required, with no runtime default — `binder_generate.yaml:34` interpolates it unguarded. (`complexa target add` substitutes `A1-100` when `--target-input` is omitted, `target_manager.py:986`; a hand-written entry gets no such fallback.)
- **Ligand targets**: not required (and not used at runtime). The pocket is defined by the ligand position, not a chain range.

## Hotspot residue format

A hotspot is a single residue, identified by chain + residue number:

```
"A33"     # chain A, residue 33
"B17"     # chain B, residue 17
```

Rules:
- Must be a string (always quote in YAML — the YAML dumper auto-quotes any `<chain><digits>` pattern).
- Chain letter is case-sensitive and must match a chain in the PDB.
- Residue number must exist in the PDB (use `grep "^ATOM" target.pdb | awk '{print $5, $6}' | sort -u` to enumerate).
- The list may be empty (`[]`) — no hotspots = no special interface focus.
- The list may contain only `[null]` for ligand targets where the pocket is ligand-defined.

CLI form: `--hotspot-residues A33 A95 A102` (space-separated, no quotes needed on the command line).

## `binder_length` semantics

| Value | Protein target | Ligand / AME target |
|---|---|---|
| `[60, 120]` | Sample uniformly in [60, 120] inclusive. | Same. |
| `[80, 150]` | Sample uniformly in [80, 150]. | Same. |
| `[100]` | **Invalid** — `binder_generate.yaml:25` reads `binder_length[1]` unguarded, so Hydra fails to interpolate. | Valid: fixed length 100 (`ligand_binder_generate.yaml:23` / `ame_generate.yaml:24` guard the second element with `oc.select:...,null`). |
| `[]` or key absent | **Invalid** — `binder_length[0]` fails too. There is *no* runtime fallback; the `[60, 120]` default lives only in the CLI writer (`target_manager.py:1008-1011`), so it never applies to a hand-edited entry. | Also invalid. |

CLI form: `--binder-length 60 120` (two ints) or `--binder-length 100` (one int — ligand/AME only).

## Source directory convention

`source` is **not** a full path. It is a subdirectory name under `$DATA_PATH/target_data/`. The full path resolution is:

```
${DATA_PATH}/target_data/<source>/<target_filename>.pdb
```

Examples:

| `source` | `target_filename` | Resolved path |
|---|---|---|
| `bindcraft_targets` | `PD-L1` | `${DATA_PATH}/target_data/bindcraft_targets/PD-L1.pdb` |
| `ligand_targets` | `7BKC_ligand_centered` | `${DATA_PATH}/target_data/ligand_targets/7BKC_ligand_centered.pdb` |
| `custom_targets` | `MyTarget_v1` | `${DATA_PATH}/target_data/custom_targets/MyTarget_v1.pdb` |

Override the convention with `--target-path /absolute/path/to/file.pdb` (overrides both `source` and `target_filename`).

Existing `source` values, with entry counts across the three dicts: `ame_targets` (44, the most common — AME only), `bindcraft_targets` (28), `alpha_proteo_targets` (16), `ligand_targets` (4). `custom_targets` has **zero** entries — it is only the default `complexa target add` writes for a protein target when `--source` is omitted, so the directory may not exist under `$DATA_PATH/target_data/` at all.

## AME task names (NOT `complexa target`)

The AME pipeline uses **task names**, not targets-dict entries. Task names are defined in `configs/design_tasks/ame_dict_v2.yaml` under `motif_target_dict_cfg:` and are file-edit-only — they have a richer schema than protein/ligand targets and the `complexa target` CLI does not touch them.

### Grammar

```
M{NNNN}_{pdb}[_{variant}]
```

| Component | Values | Meaning |
|---|---|---|
| `M{NNNN}` | `M0001` … | Zero-padded sequential ID assigned when the task is added. |
| `pdb` | 4-char PDB code (lowercase) | Source PDB for the motif + ligand context. |
| `_{variant}` | optional; seen: `_og`, `_v3` | Free-form suffix distinguishing curated variants of the same motif. Not parsed by anything — the key is just a dict lookup. |

The name is a plain dict key with no validation anywhere, so treat the grammar as a convention. Real examples from `configs/design_tasks/ame_dict_v2.yaml`: `M0024_1nzy`, `M0024_1nzy_og`, `M0024_1nzy_v3`, `M0096_1chm`, `M0096_1chm_og`, `M0040_13pk`. Note that the suffixed variants are *separate tasks*, and that `target_filename` does not track the key (`M0024_1nzy` points at `M0024_1nzy_v2`).

### AME task schema

Each entry under `motif_target_dict_cfg.<name>` has:

| Field | Type | Example | Present in | Notes |
|---|---|---|---|---|
| `source` | str | `ame_targets` | 44/44 | Subdirectory under `$DATA_PATH/target_data/`. Always `ame_targets`. |
| `target_filename` | str | `M0024_1nzy_v2` | 44/44 | PDB stem (no `.pdb`). Usually, but not always, the task key. |
| `ligand` | str **or list[str]** | `"BCA"`, `["ADP", "MG", "3PG"]`, `"L:0"` | 44/44 | Residue name(s) of the ligand(s) in the motif PDB — a list for multi-ligand tasks (`:46`), or a `"<resname>:<resnum>"` selector (`:36`). |
| `contig_atoms` | str | `"A64: [O, C]; A86: [CB, CA, N, C]; ..."` | 44/44 | Hand-curated per-residue atom selection for the motif. |
| `binder_length` | list[int] | `[180]` | 44/44 | Scaffold length. Every entry is the single value `[180]`; `ame_generate.yaml:24` guards the second element, so single-element lists are legal here. |
| `hotspot_residues` | list | `[null]` | 44/44 | Present on every entry (always `[null]`) — carried from the shared target convention. |
| `target_path` | str | `./assets/target_data/ame_input_structures/M0024_1nzy.pdb` | 42/44 | Explicit PDB path; wins over `source` + `target_filename` via `oc.select` (`ame_generate.yaml:31, :35`). |
| `use_bonds_from_file` | bool | `true` | 44/44 | **No effect.** `ame_generate.yaml:33-36` builds `LigandFeatures` from `task_name`, `pdb_path` and `ligand` only, so this key is never read on the AME path. Present for consistency with the ligand schema. |

There is **no `pdb_id`** in the AME schema — 0 of 44 entries have it, and `ame_generate.yaml` never interpolates it.

To run an AME task:

```bash
complexa design configs/search_ame_local_pipeline.yaml \
  ++run_name=ame_1nzy \
  ++generation.task_name=M0024_1nzy
```

**Do not use `complexa target add` for AME tasks** — they live in a different dict and add the `contig_atoms` field (and use `motif_target_dict_cfg` instead of `target_dict_cfg`) that the CLI does not know how to construct.

## Worked examples

### 1. Protein target from PDB ID + chain (new BindCraft target)

User wants to design binders against chain A, residues 1-115, of an existing PDB at `${DATA_PATH}/target_data/bindcraft_targets/PD-L1.pdb`, hotspot residues A37, A39, A49, A98, binder length 64-155.

The name must not already be a key in `configs/targets/targets_dict.yaml` — `02_PDL1` is taken (`:11-18`), and re-using it without `-f` makes `target_manager.py:1035-1044` prompt on `input()`, which raises `EOFError` and exits 1 non-interactively.

```bash
complexa target add 50_PDL1_custom \
  --source bindcraft_targets \
  --target-filename PD-L1 \
  --target-input A1-115 \
  --hotspot-residues A37 A39 A49 A98 \
  --binder-length 64 155 \
  --pdb-id 4z18
```

Resulting YAML entry (appended to `configs/targets/targets_dict.yaml`). Key order and quoting are exactly what `_format_target_entry` (`target_manager.py:297-369`) emits — `target_input` first because `config["target_input"]` is set before `source`/`target_filename` (`:984-1002`), and `PD-L1` / `A1-115` quoted because the quoting predicate fires on `-`:

```yaml
  50_PDL1_custom:
    target_input: "A1-115"
    source: bindcraft_targets
    target_filename: "PD-L1"
    hotspot_residues: ["A37", "A39", "A49", "A98"]
    binder_length: [64, 155]
    pdb_id: 4z18
```

Note the CLI writes no `target_path`, while all 44 hand-maintained entries have one. Add it by hand if the PDB is not under `$DATA_PATH/target_data/`.

### 2. Ligand target with SMILES (FAD pocket)

User wants binders for the FAD-binding pocket of 7BKC, fixed length 100.

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

`--dict configs/targets/ligand_targets_dict.yaml` is **mandatory for every ligand target.** `complexa target` has no ligand-aware routing: `DEFAULT_TARGETS_DICT_PATH` is `configs/targets/targets_dict.yaml` and `get_default_dict_path()` returns only that or the legacy `configs/generation/targets_dict.yaml` (`target_manager.py:24`, `:130-148`). Omit `--dict` and the ligand entry lands in the **protein** dict, which `binder_generate.yaml:8` composes and the ligand pipeline never reads (`ligand_binder_generate.yaml:6` composes `ligand_targets_dict`). `43_7BKC_LIGAND` is used here because `41_7BKC_LIGAND` already exists in that file (`:26-36`).

Resulting YAML entry (appended to `configs/targets/ligand_targets_dict.yaml`):

```yaml
  43_7BKC_LIGAND:
    source: ligand_targets
    target_filename: 7BKC_ligand_centered
    hotspot_residues: []
    binder_length: [100]
    pdb_id: 7BKC
    ligand: FAD
    ligand_only: True
    SMILES: O=C2C3=Nc1cc(c(cc1N(C3=NC(=O)N2)CC(O)C(O)C(O)COP(=O)(O)OP(=O)(O)OCC6OC(n5cnc4c(ncnc45)N)C(O)C6O)C)C
    use_bonds_from_file: True
```

Two style notes on that output: the live entries use `hotspot_residues: [null]` and add a `target_path:` line, while the CLI writes `[]` and no `target_path` — both work (`hotspot_residues` is not read on the ligand path, and `target_path` only overrides the `$DATA_PATH`-derived default). And the CLI leaves this SMILES **unquoted**: its quoting predicate (`target_manager.py:335-361`) triggers on `, : [ ] { } # & * ! | > ' % @ \` +`, the `<chain><digits>` pattern, or a `-`, none of which appear here. It parses fine as a plain scalar, but when *hand-editing* always quote SMILES — a `[O-]` or `[n+]` charge group (as in `39_7V11_LIGAND`) genuinely needs it.

### 3. Ligand target with `--use-bonds-from-file`, and why you still need `--smiles`

User has a curated `_centered` PDB whose bonds are authoritative and would rather not supply a SMILES (e.g. uncommon ligand). They want a length range 40-88.

```bash
complexa target add Cambridge_bloodsugar_A_4D71_pdb \
  --dict configs/targets/ligand_targets_dict.yaml \
  --source ligand_targets \
  --target-filename Cambridge_bloodsugar_A_4D71_pdb \
  --pdb-id 4D71 \
  --ligand UNK \
  --ligand-only \
  --use-bonds-from-file \
  --smiles "OCC1OC(O)C(O)C(O)C1O" \
  --binder-length 40 88
```

Resulting entry:

```yaml
  Cambridge_bloodsugar_A_4D71_pdb:
    source: ligand_targets
    target_filename: Cambridge_bloodsugar_A_4D71_pdb
    hotspot_residues: []
    binder_length: [40, 88]
    pdb_id: 4D71
    ligand: UNK
    ligand_only: True
    SMILES: OCC1OC(O)C(O)C(O)C1O
    use_bonds_from_file: True
```

Two things to know:

- **Drop `--smiles` and you do not get `SMILES: null` — you get no `SMILES` key at all.** `target_manager.py:1024-1025` writes it only `if smiles`, and `ligand_binder_generate.yaml:33` interpolates `...SMILES` unguarded, so the entry raises `InterpolationKeyError` the first time the ligand pipeline composes it. Always pass `--smiles` (any valid SMILES for the ligand), or add `SMILES: null` to the entry by hand afterwards. `use_bonds_from_file: True` means the value is not used for bond generation, but the key still has to resolve.
- **`--ligand` does not require a value.** `cli_runner.py:1276-1282` declares it `nargs="?"` with `const="YOUR_LIGAND"`, so the bare `--ligand` is legal and writes that placeholder; passing anything at all (value or not) is what marks the target ligand-typed (`cli_runner.py:1855-1856`: `is_ligand = ligand_val is not None`). Prefer the real residue name when you have it.

(There is no `Cambridge_bloodsugar_A_4D71_pdb` entry in the shipped `ligand_targets_dict.yaml` — it holds only `39_7V11_LIGAND` … `42_7C7M_LIGAND` — so this example creates a new key rather than mirroring an existing one.)

## Cross-references

- CLI: `src/proteinfoundation/cli/cli_runner.py:1177-1321` (the parser behind `complexa target`). `src/proteinfoundation/cli/target_cli.py` is the separate `complexa-target` console script (`pyproject.toml:65`).
- Manager + schema: `src/proteinfoundation/cli/target_manager.py` (see `TARGET_FIELDS`; the CLI-only defaults are at `:984-1030`)
- Validator: `src/proteinfoundation/cli/validate.py::validate_target` — **currently cannot resolve `target_dict_cfg` for any config in this repo**, so `complexa validate target CONFIG --target NAME` always fails with `Could not find target_dict_cfg in config` (`validate.py:183-187` skips Hydra composition, `:315-317` needs a dict-valued `defaults:` entry where the pipelines have strings, `:359-361` falls back to the non-existent `configs/generation/targets_dict.yaml`). Verify targets by reading the YAML and checking the PDB path — see `SKILL.md` Step 4.
- Protein dict: `configs/targets/targets_dict.yaml`
- Ligand dict: `configs/targets/ligand_targets_dict.yaml`
- AME dict: `configs/design_tasks/ame_dict_v2.yaml`
