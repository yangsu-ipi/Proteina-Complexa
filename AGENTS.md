# Proteina-Complexa — agent orientation

Protein/ligand binder design and enzyme scaffolding with a partially-latent flow-matching
model. This file is auto-loaded every session, so it stays short and points elsewhere.

## Layout

- `src/proteinfoundation/` — the package (src layout; `"src" = ""` in `pyproject.toml`).
  Entry points: `generate.py`, `filter.py`, `evaluate.py`, `analyze.py`, `train.py`.
- `configs/` — Hydra configs. Pipelines are `configs/search_*_pipeline.yaml`; stage configs
  live under `configs/pipeline/{binder,ligand_binder,ame}/`.
- `community_models/` — vendored openfold, colabdesign, ProteinMPNN, LigandMPNN
  (re-exported as top-level packages by the wheel `sources` mapping).
- `assets/target_data/` — 81 tracked, pre-cleaned target PDBs.
- `docs/` — human-facing guides. `.claude/skills/` — agent skills.

## CLI

`complexa` (`pip install -e .`). `complexa design <pipeline.yaml> [++overrides] [--verbose]`
runs generate → filter → evaluate → analyze as four subprocesses over the **same** config
file. Also `complexa init`, `complexa download`, `complexa target`, `complexa validate`,
`complexa analysis`.

Overrides use Hydra `++`. A typo'd `++` key is a silent no-op, not an error.

## Read these before acting

| Task | Read |
|---|---|
| New design target: env vars, target YAML, PDB prep, setup failures | [`docs/binder-target-setup/`](docs/binder-target-setup/) — or run `/complexa-target-setup` |
| Running a pipeline, overrides, thresholds, OOM | [`.claude/skills/complexa-design/`](.claude/skills/complexa-design/) |
| First-time setup, `.env`, weights | [`.claude/skills/complexa-setup/`](.claude/skills/complexa-setup/) |
| Target schema fields | [`.claude/skills/complexa-target/`](.claude/skills/complexa-target/) |
| Scoring an existing PDB directory | [`.claude/skills/complexa-evaluate-pdbs/`](.claude/skills/complexa-evaluate-pdbs/) |
| Known defects in the bundled skills | [`SKILLS_AUDIT.md`](SKILLS_AUDIT.md) |
| Conceptual walkthrough, fresh clone → first campaign | [`BINDER_DESIGN_ONBOARDING.md`](BINDER_DESIGN_ONBOARDING.md) |

Codex slash commands live in `.codex/prompts/`; the equivalent Claude Code skills in
`.claude/skills/`. Both share the reference files under `docs/`, so edit the shared file
rather than one entry point.

## House rules

- Anchor factual claims to a source path, with a line number for code and configs
  (`cli_runner.py:646`) and a section heading for the frequently-edited `.claude/skills/*.md`.
- Target setup fails silently far more often than it errors. A clean exit is not evidence
  the run used the target you intended — verify, then report.
- `*.pdb`, `logs/`, and a root-level `/inference` are git-ignored. Keep per-target working
  directories outside this checkout.
