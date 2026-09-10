# Target PDB Preparation

The 81 PDBs under `assets/target_data/` are hand-prepared — cropped, repacked, re-chained
(`HER2_cropped.pdb`, `1tnf_repacked.pdb`, `1bj1_cropped.cif`). This document says which of
that preparation is **required**, which is convenience, and which risks a raw RCSB
download carries that cleaning does not address.

## Is cleaning mandatory?

| Pipeline | Verdict |
|---|---|
| **AME / enzyme** | **Yes** — documented checklist, malformed input causes silent evaluation errors |
| **Protein binder** | **Situationally** — the contig crops for you, but in-range heteroatoms corrupt the target and hotspot numbering fails silently |
| **Ligand binder** | **Structurally** — the PDB must contain only the ligand; this is a task property, not cleanliness |

## AME — the documented checklist

The only pipeline with required, written-down cleaning steps. See "Preparing AME Target
PDBs" in [`assets/target_data/README.md`](../../assets/target_data/README.md): ligand on
chain A and motif protein on chain B, ligand residue name set to `L:0`, metal ions
stripped, `contig_atoms` chain letters updated to match. The warning there is explicit —
most bundled `ame_input_structures/` still need preparation, and `M0024_1nzy_v3.pdb` is
the one ready-to-use reference.

The `L:0` rename recipe is already documented in "AME ligand residue name must be `L:0`
for RF3" in `.claude/skills/complexa-design/reference/troubleshooting.md` — use that, not
a fresh transcription.

## Protein binder — the contig crops for you

`load_target_from_pdb` crops logically at load time
(`src/proteinfoundation/utils/pdb_utils.py:550-556`):

```python
struct = load_any(pdb_path, model=1)                    # extra NMR models dropped
select = AtomSelectionStack.from_contig(target_spec)    # e.g. "A1-115"
mask = select.get_mask(struct)
struct = struct[mask]                                   # everything else dropped
```

So `target_input: A1-115` already discards other chains and out-of-range residues. **A raw
download works.** Physical cropping is about speed and clarity, not admissibility.

Missing atoms are likewise tolerated: `atom_array_to_encoding(..., default_coord=0.0)`
returns a `mask`, so gaps and absent sidechains degrade rather than crash. That is why
`alpha_proteo_targets/` ships `_repacked` and `_fixed` variants — quality improvements,
not entry requirements.

Two things the contig does **not** protect you from.

### Heteroatoms inside the selected range

`from_contig` builds its selection on `(chain_id, res_id)` pairs and nothing else
(`atomworks/io/utils/selection.py:482-493`):

```python
for i in range(int(start), int(stop) + 1):
    selections.append(AtomSelection(chain_id=chain_id, res_id=i))
```

No polymer filter, no hetero filter, no element filter. Any water, ion, glycan, or
modified residue carrying the selected chain ID with an in-range `res_id` is selected and
handed to the AF2 atom37 **protein** encoding. Raw RCSB entries routinely have this, and
PDB-format files often put solvent on the same chain ID as the protein.

**For raw downloads, stripping heteroatoms is normally required.** This is the one place
where cleaning a binder target is genuinely load-bearing.

### Residue numbering — and cleaning does not fix it

`target_input` selects **literal** `res_id` values, and **every residue in the range must
be present**. `AtomSelectionStack.from_contig(...).get_mask(...)`
(`src/proteinfoundation/utils/pdb_utils.py:554-555`) raises on the first one that is not:

```
ValueError: No atoms found for selection: A/*/116
```

Measured on atomworks 2.2.1 against the bundled targets:

| File | Chain holds | `target_input` | Result |
|---|---|---|---|
| `PD-L1.pdb` | A 1–115 | `A1-115` | 115 residues — the shipped `02_PDL1` value |
| `PD-L1.pdb` | A 1–115 | `A1-116` | raises at `A/*/116` |
| `PD-L1.pdb` | A 1–115 | `B1-115` | raises at `B/*/1` |
| `3di3_cropped.pdb` | B 17–209 | `B1-209` | raises at `B/*/1` |
| `sCas9_cropped.pdb` | A 96–446, 131 unresolved | `A96-446` | raises at `A/*/175` |
| `sCas9_cropped.pdb` | ″ | `A96-174,A306-446` | 220 residues — the shipped `24_SpCas9` value |

So an offset start (`3di3_cropped` begins at 17, not 1), a wrong chain letter, or a range
crossing an unresolved gap is a **hard failure**, not a silent truncation. This is the
good case: it cannot be mistaken for success.

It is also why several bundled entries carry multi-segment contigs. `24_SpCas9` is
`A96-174,A306-446` because residues 175–305 are absent from the file, and no single span
can cross them.

The catch is *where* it fails: inside generation's target reader
(`src/proteinfoundation/datasets/gen_dataset.py:513`), not at config time —
and `complexa validate target` will not reach it, because it never opens the PDB.
`check_preflight.py` applies the same selector before the job starts, so a campaign run
through it fails in seconds instead.

**Hotspots are the failure that stays silent.** They are matched as raw strings against the
CAs of the *already-contig-masked* structure, and misses are quiet
(`src/proteinfoundation/utils/pdb_utils.py:571-575`):

```python
target_hotspots_mask = torch.zeros(len(ca_struct), dtype=torch.bool)
if target_hotspots is not None:
    for idx, atom in enumerate(ca_struct):
        if f"{atom.chain_id}{atom.res_id}" in target_hotspots:
            target_hotspots_mask[idx] = True
```

No warning, no error → the run completes and designs something, with no epitope guidance.
Three cases survive the contig check above and land here: a hotspot **outside**
`target_input` (the mask is applied first, so `A115` is unaddressable under `A5-100` even
though the file has it), a hotspot naming a chain the contig does not cover, and hotspots
read in a different numbering from the file — off the RCSB page, a paper, or the `.cif` of
the same entry. Insertion codes land here too. See also
"Hotspot residue not in target PDB" and "Chain-ID mismatch between target PDB and
target_input" in `.claude/skills/complexa-design/reference/troubleshooting.md`.

Because cropping usually renumbers, a cleaning step can **introduce** this. The rule:

> **One numbering source of truth — the file exactly as you feed it in.** Derive
> `target_input` from that file; do not copy a range from a bundled example. Those ranges
> are correct for the cropped files they ship with.

Note also that `f"{chain_id}{res_id}"` ignores insertion codes, so `A52` matches both 52
and 52A — antibody-numbered targets cannot address them separately.

### `.cif` and `.pdb` do not agree

`load_any` reads mmCIF with author fields **off**
(`atomworks/io/utils/io_utils.py:290`):

```python
atom_array_stack = pdbx.get_structure(
    file_obj, model=model, extra_fields=extra_fields,
    use_author_fields=False, ...)
```

So `.cif` yields **`label_seq_id`** — 1-based sequential over the SEQRES entity — while
`.pdb` yields **author numbering**. The same entry in the two formats produces different
`res_id` values, both landing in the field your hotspots and contig match against.

**Choosing a format is itself a renumbering decision.** RCSB now defaults to `.cif`.
Hotspots read off the RCSB page or a paper are author numbering — they match a `.pdb`
download and silently miss on a `.cif` one.

There is a real tension here: `atomworks/io/parser.py:783` warns that the PDB reader
assumes all residues are resolved and recommends CIF otherwise — i.e. it pushes you toward
label numbering. If you take that advice, re-derive your hotspots from the CIF.

## Ligand binder

The target PDB contains **only the ligand**; the protein is generated de novo
(`ligand_only: True`, `use_bonds_from_file: True`, plus a `ligand` CCD code and `SMILES`).
A protein-containing PDB is not "dirty" here, it is the wrong input. The live entries pair
`ligand_only: True` with pre-trimmed `*_ligand_centered.pdb` files.

## Where the CCD / `L:0` concern applies

Orthogonal to pipeline type — it is about whether your target has a ligand *and* you
refold with RF3. RF3 ≥0.1.12 reconstructs ligands from their CCD code, adding heavy atoms
absent from your input and breaking RMSD shape matching. So it affects AME and
ligand-binder targets, and never protein-protein binders. Details in
[`env-and-mirrors.md`](env-and-mirrors.md) and the `L:0` section referenced above.

## `complexa validate target` will not catch any of this

It checks that the file **exists**, then echoes `target_input`, `hotspot_residues`, and
`binder_length` back as pass lines (`src/proteinfoundation/cli/validate.py:377-501`). It
never opens the PDB, so it cannot compare your config against the structure. Use the
preflight below instead.

## Preflight

Strip heteroatoms, keep numbering, read your config values off the result:

```bash
python docs/binder-target-setup/scripts/check_target_pdb.py \
    --pdb /data/targets/MINE/mine.pdb \
    --chain A --hotspots A37 A39 A49 A98 \
    --write-clean /data/targets/MINE/mine_clean.pdb
```

It prints the derived `target_input`, the residue gaps, and matched/missing hotspots, and
exits non-zero if any hotspot is unmatched. Paste the printed `target_input` straight into
your config and require the missing list to be empty.

Gaps matter more than that output suggests. A gap is only harmless if `target_input`
**avoids** it — which is what a multi-segment contig is for; a range crossing one raises
(table above). Absent side-chain atoms *within* a residue that is present are a different
thing, and those the encoding mask does handle. A hotspot inside a gap has no CA, so it
appears in the missing list like any other unmatched hotspot.

`--write-clean` writes the heteroatom-stripped structure **without renumbering**, so the
printed values stay valid for it. If you point the config at the cleaned file, use the
cleaned file's numbering — one source of truth.
