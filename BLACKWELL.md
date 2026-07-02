# Proteina-Complexa on NVIDIA Blackwell (sm_120) — IPI internal fork

This branch (`blackwell`) documents how to build **Proteina-Complexa** for NVIDIA Blackwell /
RTX PRO 6000 (`sm_120`). It is the hardest of the NVIDIA design tools to port; used on IPI **gnode2**.

## What changes (and what doesn't)

**No source changes are required** — only the environment, but the environment is fiddly:

1. **Python ≥ 3.12** — `proteinfoundation` requires it (3.11 fails at install).
2. **torch cu128** — upstream `env/build_uv_env.sh` installs `torch 2.7.0+cu126` (sm ≤ 90);
   Blackwell needs `torch 2.7.1` / **cu128** or CUDA kernels fail with *"no kernel image."*
3. **`tmol` from source** — builds cleanly on py3.12 + cu128 (the flagged risk did not materialise).
4. **Over-constrained deps** — `atomworks`, `proteinfoundation`, and `tmol`(numba) disagree on
   `biotite`/`numpy`. The fix (mirroring upstream's own end-of-script reconciliation) is to apply
   two pins **last, after `pip install -e .`**, in this order:
   - `biotite==1.6.0` (atomworks needs ≥1.4; the editable install pins it down to 0.41)
   - `numpy==2.4.6` (numba, used by tmol, needs `numpy < 2.5`; biotite 1.6.0 pulls 2.5)

The proven recipe is in [`build_blackwell.sh`](build_blackwell.sh):

```bash
bash build_blackwell.sh            # creates ./.venv-blackwell (py3.12) and installs everything
```

## Verified (2026-07-02, gnode2)

- `torch 2.7.1+cu128` on sm_120, real GPU matmul OK.
- All core modules import: `proteinfoundation`, `atomworks`, `tmol`, `graphein`, `biotite 1.6.0`.

- Model checkpoints downloaded + validated loadable: `complexa.ckpt` (415M params) +
  `complexa_ae.ckpt` (256M), valid PyTorch-Lightning checkpoints.

## Checkpoints & running

Checkpoints are **PUBLIC on NGC — no account or API key needed** (`complexa download` is just a
`wget ...?redirect=true` with no auth header; `build_blackwell.sh` fetches the protein-binder pair,
~7 GB, into `ckpts/`). The three variants (protein / ligand / AME) live at
`catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/proteina_complexa[_ligand|_ame]`.

A **full binder-design run** (`complexa design configs/search_binder_local_pipeline.yaml`) is a
4-stage pipeline (generate → filter → evaluate → analyze). Beyond the checkpoints it needs a target
spec, the community reward/refolding models (ESM2 via a **free** HF token; AF2/RF3/Boltz2), external
tools (foldseek/mmseqs/dssp), and `CCD_MIRROR_PATH`/`PDB_MIRROR_PATH` for atomworks I/O — a larger
setup than the generation checkpoints alone. Not run here.

## Scope / provenance

- Internal fork for IPI hardware. **Not** submitted upstream (by choice).
- This branch adds only `BLACKWELL.md` + `build_blackwell.sh` on top of stock upstream.
