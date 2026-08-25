# Proteina-Complexa on NVIDIA Blackwell (sm_120) — IPI internal fork

This branch (`blackwell`) makes **Proteina-Complexa** run on NVIDIA Blackwell / RTX PRO 6000
(`sm_120`). It is the hardest of the NVIDIA design tools to port (a torch **and** JAX stack in one
env). Used on IPI **gnode2**. **Validated end-to-end 2026-07-02**: reward-guided binder generation
for target `02_PDL1` (PD-L1) produces real AF2 reward scores on the Blackwell GPU.

## Environment (the hard part)

1. **Python ≥ 3.12** — `proteinfoundation` requires it (3.11 fails at install).
2. **torch cu128** — upstream pins `torch 2.7.0+cu126` (sm ≤ 90); Blackwell needs `torch 2.7.1`/**cu128**.
3. **`tmol` from source** — builds cleanly on py3.12 + cu128.
4. **Over-constrained torch deps** — apply `biotite==1.6.0` then `numpy==2.4.6` **last**, after
   `pip install -e .` (atomworks wants biotite ≥1.4; numba/tmol want numpy < 2.5).
5. **JAX + colabdesign AF2 reward stack** (the crux): the search code imports `colabdesign` (AF2/JAX)
   at module load, and reward-guided search folds candidates with AF2. So JAX must run **alongside
   torch in the same env**. That works **only with cudnn 9.24**: jax 0.10.2 needs it, and torch cu128
   runs fine on it at runtime (torch's `==9.7.1.26` pin is stricter than reality). Then `optax`,
   `flax`, `chex`, `dm-haiku`, and the vendored `colabdesign` (editable).

The whole recipe is in [`build_blackwell.sh`](build_blackwell.sh): `bash build_blackwell.sh`.

## Source patches (vs upstream)

Contrary to a pure recipe, Blackwell (jax ≥ 0.10) needs these code fixes, all on this branch:

- **colabdesign's bundled AlphaFold — `jnp.clip` kwargs** `a_min=`/`a_max=` → `min=`/`max=`
  (`community_models/colabdesign/af/loss.py`, `.../af/alphafold/model/modules.py`, `modules_multimer.py`).
- **colabdesign `__init__.py` compat shim** — jax ≥ 0.10 removed `jax.tree_map`/`jax.tree_multimap`
  (→ `jax.tree_util`) and `jax.util` (→ `jax._src.util`); the shim restores them before submodules load.
- **AF2 reward — `jax.clear_backends()` → `jax.clear_caches()`** (removed in jax 0.10)
  in `src/proteinfoundation/rewards/alphafold2_reward.py`.

Debugging note: the reward's exceptions were **triple-hidden** — swallowed by `CompositeRewardModel`'s
`try/except → warnings.warn`, then silenced by the CLI's `-W ignore`. Run `python -m
proteinfoundation.generate ...` directly (no `-W ignore`) to surface reward-model errors.

## Checkpoints & AF2 params — all PUBLIC on NGC/Google (no key)

- **Complexa checkpoints**: `complexa download` is an unauthenticated `wget ...?redirect=true`;
  `build_blackwell.sh` fetches the protein-binder pair (~7 GB) into `ckpts/`.
- **AF2 reward params**: needs the **MULTIMER** set `alphafold_params_2022-12-06` (public, Google
  storage, ~5.3 GB) — a 2021 monomer store is **not** enough (binder-complex folding needs
  `*_multimer_v3`). `build_blackwell.sh` fetches it into `community_models/ckpts/AF2`; set `AF2_DIR` there.

## Verified (2026-07-02, gnode2)

- Core imports (`proteinfoundation`/`atomworks`/`tmol`/`graphein`/`biotite`) + torch matmul on sm_120.
- Checkpoints validated loadable (`complexa.ckpt` 415M + `complexa_ae.ckpt` 256M).
- **Binder generation** (single-pass) → binder-target complexes for PD-L1.
- **Reward-guided generation** (best-of-n) → AF2 folds candidates and returns real scores
  (mean reward ≈ −0.71; a single fold scored `total_reward = −0.7191`).
- **Full 4-stage pipeline** (`complexa design`) → generate → filter → evaluate → analyze runs
  **clean end-to-end (zero errors)** for the PD-L1 quick test. Output CSVs carry real AF2 metrics
  (`af2folding_i_pae`, `_i_ptm`, `_plddt`, `_ptm`, `_rmsd`), plus ESM/monomer metrics and
  foldseek/mmseqs diversity clustering.

**Notes.** evaluate defaults to `binder_folding_method: colabdesign` (the AF2 stack above) — **no RF3
needed** for the protein-binder pipeline (rf3_latest / boltz2_default / esmfold are selectable but
untried here). The analyze diversity metrics need `foldseek` + `mmseqs` (bioconda; wire via
`FOLDSEEK_EXEC`/`MMSEQS_EXEC` or the `UV_*` vars in `.env`). Pre-refolding bioinformatics (sc/dssp)
are off by default.

## Scope / provenance

- Internal fork for IPI hardware. **Not** submitted upstream (by choice).
- Adds the source patches above + `BLACKWELL.md` + `build_blackwell.sh` on top of stock upstream.
