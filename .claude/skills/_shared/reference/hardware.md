# Proteina-Complexa Hardware Reference

Shared hardware reference for every `complexa-*` skill. Tables only — see each
skill's `SKILL.md` "Hardware" section for the user-facing prose. Numbers marked
`(empirical)` are not pulled from any doc and represent conservative defaults
based on the configs in this repo.

## Per-pipeline GPU requirements

| Pipeline           | Config                                            | Min VRAM (GB) | Recommended VRAM (GB) | Supported SKUs            | Single-GPU only |
|--------------------|---------------------------------------------------|--------------:|----------------------:|---------------------------|-----------------|
| Protein Binder     | `search_binder_local_pipeline.yaml`               |  24 (empirical) |  40 (empirical)       | H100, A100-80, L40S       | Yes             |
| Ligand Binder      | `search_ligand_binder_local_pipeline.yaml`        |  32 (empirical) |  48 (empirical)       | H100, A100-80, L40S       | Yes             |
| AME (motif+ligand) | `search_ame_local_pipeline.yaml`                  |  32 (empirical) |  48 (empirical)       | H100, A100-80, L40S       | Yes             |

Notes:
- All three inference pipelines are single-GPU (the `complexa generate` stage
  is one process per `gen_njobs` slot; jobs do not shard a single design
  across GPUs).
- AME requires `USE_V2_COMPLEXA_ARCH=True`, set via `env_vars:` in
  `configs/search_ame_local_pipeline.yaml` (no runtime VRAM impact).
- Multi-GPU hosts run multiple pipelines / stages in parallel by bumping
  `gen_njobs` / `eval_njobs` — each job takes one GPU, pinned by index, so neither may
  exceed the GPU count in a single `complexa design` run, and the two must be **equal**
  (eval shard *N* takes the designs named `job_N_*`). To shard generation more finely for
  resume, drive shards individually — see "Sizing shards so resume is worth having" in
  `docs/binder-target-setup/campaign-gating.md`.

## Per-evaluation-backend requirements

| Backend             | Min VRAM (GB) | Extra packages / env             | Wall-clock per sample      |
|---------------------|--------------:|----------------------------------|----------------------------|
| ColabDesign / AF2   |            16 | JAX + ColabFold; `AF2_DIR`       | ~30–60 s (empirical)       |
| RoseTTAFold3 (rf3)  |            24 | `RF3_EXEC_PATH`, `RF3_CKPT_PATH` | ~60–180 s (empirical)      |
| ESMFold (monomer only) |         16 | `fair-esm`, internet/cache OK    | ~5–15 s (empirical)        |

Binder / complex folding is selected via
`++metric.binder_folding_method=colabdesign|rf3_latest` — those are the only two accepted
values (`binder_eval.py:108-128` raises `ValueError: Folding model '<x>' not supported` for
anything else). ESMFold is **not** a valid binder backend; it is accepted only for the
separate monomer key `++metric.monomer_folding_models=[esmfold]` — which also accepts
`esmfold2` (single-chain, single-sequence, Fast-Cutoff2025 checkpoint)
(`monomer_eval_utils.py:37`, `VALID_FOLDING_MODELS = ["esmfold", "esmfold2", "colabfold"]`).

## Search-algorithm cost multipliers

Relative to `single-pass` (= 1.0×) at fixed `nsteps` and `dataloader.batch_size`.

The algorithm name is **hyphenated** — `src/proteinfoundation/search/search_factory.py:30-40` matches `single-pass`,
`best-of-n`, `beam-search`, `fk-steering`, `mcts` and raises
`ValueError: Unknown search algorithm` on anything else. The *sub-config block* names stay
underscored (`search.beam_search.beam_width`, `search.fk_steering.n_branch`, …).

| Algorithm    | Override key                                | Wall-clock | Peak VRAM |
|--------------|---------------------------------------------|-----------:|----------:|
| single-pass  | `++generation.search.algorithm=single-pass` |        1.0× |     1.0× |
| best-of-n    | `…=best-of-n` + `best_of_n.replicas=N`      |        N×  |     1.0× |
| beam-search  | `…=beam-search` + `beam_search.beam_width=W,beam_search.n_branch=B` |     W·B× ≈ W× |  ~1.1× |
| FK-steering  | `…=fk-steering` + `fk_steering.beam_width=W,fk_steering.n_branch=B` |     W·B× ≈ W× |     1.2× |
| MCTS         | `…=mcts` + `mcts.n_simulations=S`           |       ≥S×  |     1.2× |

Memory is roughly constant — search algorithms reuse the same model forward;
only beam/FK/MCTS retain extra candidate tensors per branch.

## CPU / RAM / disk

Defaults pulled from `configs/search_*_local_pipeline.yaml`:

| Pipeline           | `ncpus_` | `gen_njobs` | `eval_njobs` | RAM (rec.) | Output disk / 100 designs |
|--------------------|---------:|------------:|-------------:|-----------:|--------------------------:|
| Protein Binder     |       24 |           1 |            1 |  32 GB (empirical) | ~10–20 GB (empirical) |
| Ligand Binder      |       24 |           1 |            1 |  32 GB (empirical) | ~15–30 GB (empirical) |
| AME                |       24 |           1 |            1 |  32 GB (empirical) | ~20–50 GB (empirical) |

`keep_folding_outputs=true` (eval default) roughly doubles the output disk
footprint — set to `false` if disk is tight.

## When you hit OOM

Try these in order — cheapest mitigations first:

- Reduce `++generation.dataloader.batch_size` (often the biggest VRAM lever).
- Reduce `++gen_njobs` (frees one inference process worth of memory).
- Reduce `++generation.args.nsteps` (less VRAM tied up in trajectory buffers
  when using search algorithms that retain steps).
- Reduce `++generation.search.beam_search.beam_width` /
  `++generation.search.beam_search.n_branch`.
- Set `++metric.keep_folding_outputs=false` to free fold-stage RAM/disk
  pressure (helps when an OOM lands during evaluation).
- Switch fold backend to the cheaper of the two valid ones:
  `++metric.binder_folding_method=colabdesign` (~16 GB) instead of
  `rf3_latest` (~24 GB). There is no lighter third option — ESMFold is not a
  valid `binder_folding_method` (see above).
- For AME: confirm `USE_V2_COMPLEXA_ARCH=True` matches the AME checkpoint —
  loading the wrong arch wastes ~10–20% VRAM (empirical).
- Multi-GPU host: set `CUDA_VISIBLE_DEVICES=<idx>` to pin the run to a single
  card and avoid PyTorch placing tensors on a busy peer GPU.
