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
values (`binder_eval.py:121-141` raises `ValueError: Folding model '<x>' not supported` for
anything else). ESMFold is **not** a valid binder backend; it is accepted only for the
separate monomer key `++metric.monomer_folding_models=[esmfold]` — which also accepts
`esmfold2` (single-chain, single-sequence, Fast-Cutoff2025 checkpoint)
(`monomer_eval_utils.py:38`, `VALID_FOLDING_MODELS = ["esmfold", "esmfold2", "colabfold"]`).

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

## JAX preallocates most of the card, and it does it per visible GPU

The AF2 reward is JAX, and JAX takes **75% of every visible GPU on first use** —
not what the reward needs, 75%. On an 80 GB card that is ~60 GB reserved before
PyTorch asks for anything, and the diffusion model then fails to find tens of
megabytes. The error names PyTorch, because PyTorch is what asked last:

```
CUDA out of memory. Tried to allocate 72.00 MiB. GPU 0 has a total capacity of
79.25 GiB of which 54.38 MiB is free. Process N has 61.59 GiB memory in use.
```

`61.59 GiB` in a process whose PyTorch allocation is `1.87 GiB` is the signature.
Two knobs, either of which is enough on a dedicated card:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false      # allocate on demand
XLA_PYTHON_CLIENT_MEM_FRACTION=0.35      # or just take less
```

**Per visible GPU** is the part that surprises. `AF2RewardModel` selects one
device (`device_id`, defaulting to `torch.cuda.current_device()`), but the
preallocation happens when the JAX backend initialises, across everything
`CUDA_VISIBLE_DEVICES` exposes — so a process that only ever uses card 1 has
still reserved 60 GB on card 0.

## Shards driven individually are pinned by nothing

The `gen_njobs` path pins each job to a GPU by index. Driving shards yourself —
separate processes with `++job_id=N`, which is what campaign runners do for
resume granularity — does not, and neither does `srun --gres=gpu:1` on every
cluster: where GRES cgroups are not enforced, each step still sees every GPU and
every step picks device 0. Two shards then land on one card while its peer sits
idle. Pin them explicitly rather than trusting the scheduler:

```bash
CUDA_VISIBLE_DEVICES="$shard" srun ... python -m proteinfoundation.generate "++job_id=$shard"
```

Check it in the log rather than assuming: Lightning prints
`LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]`, and two shards both reporting
`[0,1]` means neither was pinned.
