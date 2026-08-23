# Binder Target Setup

Getting a **new design target** from a raw PDB entry to a running `complexa design`
pipeline. Covers the atomworks environment variables, the target YAML, target-PDB
preparation, and the failure modes that do not announce themselves.

Read by two agent entry points, which share these files so they cannot drift:

| Entry point | How it loads |
|---|---|
| `.claude/skills/complexa-target-setup/SKILL.md` | auto-loads in `claude` from the repo root |
| `.codex/prompts/complexa-target-setup.md` | Codex slash command, `/complexa-target-setup` |

## The thing to internalise

**This part of the pipeline fails silently more often than it errors.** Nearly every
mistake below produces a run that completes and writes PDBs — just not the ones you
wanted. In rough order of how much time they cost:

| What you did | What happens | What you'd expect |
|---|---|---|
| Defined your target but didn't pin `generation.task_name` | a clean run against **TrkA** — the inherited default target — with yours loaded but unused | an error, or at least a warning |
| `CCD_MIRROR_PATH` points at a directory that isn't there | `Error locating target '…gen_dataset.collate_fn'` six seconds into generation | a message naming the missing directory |
| Hotspot IDs don't match the PDB's numbering | all hotspots dropped, design proceeds with no epitope guidance | an error naming the unmatched residue |
| `target_input` range doesn't match the file's numbering | target silently truncated, or empty | an error, or at least a count |
| Shadow `targets_dict.yaml` filename is off by a character | falls back to the shared 44-target dict and its relative paths | file-not-found |
| Heteroatoms sit inside your contig range | waters and ions encoded as protein residues | a warning |
| Target dir lives inside the repo | `*.pdb` is git-ignored, so the structure is silently untracked | the file being added |
| Sourced `env.sh` from a batch job outside the repo | tool binaries resolve, every *path* variable is empty, preflight blames five things | one error naming the export gap |

`complexa validate target` catches none of these — it confirms the PDB **exists**, then
echoes your config values back as pass lines without ever opening the file
(`src/proteinfoundation/cli/validate.py:379-503`).

So the workflow this directory documents front-loads the checks: verify the environment,
verify the structure against the config, *then* spend a GPU-hour.

## Files

| File | Contents |
|---|---|
| [`env-and-mirrors.md`](env-and-mirrors.md) | How `.env` is discovered (three inconsistent mechanisms) and the SLURM/batch recipe; then `CCD_MIRROR_PATH` / `PDB_MIRROR_PATH` semantics, why an invalid value crashes at import, and how to build either mirror |
| [`target-config.md`](target-config.md) | The one-file target YAML (**preferred**), three alternatives, and the Hydra composition mechanics behind them |
| [`pdb-prep.md`](pdb-prep.md) | Which pipelines require PDB cleaning, what the contig does and does not filter, and why `.cif` and `.pdb` numbering differ |
| [`campaign-gating.md`](campaign-gating.md) | **Writing or refreshing a campaign gate script** — deriving the required weights and the output-disk budget from the resolved config, rather than a fixed list |
| [`troubleshooting.md`](troubleshooting.md) | Masked Hydra import errors and the full silent-failure catalogue |

| Script | Purpose | Status |
|---|---|---|
| [`scripts/build_ccd_mirror.py`](scripts/build_ccd_mirror.py) | Split wwPDB `components.cif` into the CCD mirror layout atomworks expects | verified against real `components.cif` data; stdlib only |
| [`scripts/check_target_pdb.py`](scripts/check_target_pdb.py) | Preflight a target PDB: heteroatoms, numbering, gaps, hotspot resolution | verified on a real target inside a SLURM job — 136/136 residues selected by `A58-193`, no gaps, no in-range heteroatoms, all 11 hotspots matched, `RESULT: PASS` |
| [`scripts/check_resume.sh`](scripts/check_resume.sh) | Prove in-stage resume reuses work *and* invalidates correctly: shard skip, fold-cache reuse, and the three invalidation paths (config change, deleted output, changed folding backend) | helpers verified against real job outputs; the six live assertions need a GPU box — **unrun** |

## Quick path

```bash
# 1. Neutralise the mirror vars unless you have real mirrors (see env-and-mirrors.md)
export CCD_MIRROR_PATH= PDB_MIRROR_PATH=

# 2. Preflight the structure; paste the printed target_input into your config
python docs/binder-target-setup/scripts/check_target_pdb.py \
    --pdb /data/targets/PDL1/PD-L1.pdb --chain A --hotspots A37 A39 A49 A98

# 3. One self-contained YAML per target (see target-config.md), then:
cd /data/targets/PDL1 && complexa design ./pipeline.yaml --verbose
```

## Scope

This directory owns target *setup*. It does not cover choosing search algorithms, reward
weights, or success thresholds — see `.claude/skills/complexa-design/` and
[`docs/INFERENCE.md`](../INFERENCE.md). It does not cover `.env` keys generally; see the
"`.env` Key Reference" in `.claude/skills/complexa-setup/reference/env_keys.md`. The
atomworks mirror variables documented here are deliberately absent from `.env_example`,
which is part of why they bite.
