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
Note *whose*: JAX and PyTorch live in the **same** process here — `AF2RewardModel`
is constructed inside `generate.py`'s `predict_step` — so this is not two
processes to separate, it is one card to divide.

### Which knob

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6       # bound JAX, leave the rest to torch
XLA_PYTHON_CLIENT_PREALLOCATE=false      # or allocate on demand
```

**Prefer bounding to on-demand**, which is the opposite of the intuition that you
cannot predict the split so both sides should float. Three reasons:

- **AF2 is the predictable half.** Its graph is fixed and its footprint is a
  function of target length, binder length and `num_recycles`, all pinned by
  config. The torch side is what varies — `batch_size`, and binder lengths drawn
  from `nres`. So bound the stable one and let the variable one take the
  remainder, rather than leaving both to negotiate at runtime.
- **PyTorch's caching allocator does not give memory back.** Freed tensors stay in
  its reserved pool. Under on-demand JAX, whichever side asks first during a long
  run keeps what it took, and JAX can be starved at hour three of a campaign
  rather than in the first minute.
- **Beam search interleaves them tightly.** Diffusion steps and AF2 scoring
  alternate at every `step_checkpoints` entry, which is the worst case for two
  on-demand allocators sharing one device.

A wrong fraction fails fast and deterministically, which is the failure you want.

Use `PREALLOCATE=false` when the two genuinely peak at different times and no
fraction fits both.

### Choosing MEM_FRACTION

Coarse beats precise. The knob exists to stop JAX crowding torch out of a shared
card, not to pack the card tightly, and a tight number is fragile: it goes stale
the moment `batch_size`, `nres` or the target changes, and it goes stale silently.

| Situation | Do |
|---|---|
| 80 GB card, complex ≲ 400 residues | Pin one shard per GPU, leave the default. `0.6` if you want a margin. |
| Small card, or a long target | Lower `++generation.dataloader.batch_size` first — the biggest lever |
| Still tight | Then measure, and set a fraction from it |

**This knob only affects generation.** JAX exists in that stage and nowhere else:
`AF2RewardModel` is built in `predict_step`. `complexa evaluate` runs as a separate
process with no JAX in it, so its budget is torch alone and `MEM_FRACTION` does
nothing there.

Do not read generation's numbers as the pipeline's. The `1.87`/`3.12 GiB` above is
the **diffusion model**, measured on a run that died early. Evaluation is a
different and much larger budget: ESMC 6B is ~12 GB of weights at bfloat16 and
about twice that if it loads in fp32, plus ESMFold2, plus whatever the folding
backend needs. Those figures are unverified — no ESMFold2 or ESMC path has run
against real weights yet, and `esm_eval.py:542` calls `from_pretrained` without a
`torch_dtype`, so which of the two ESMC sizes applies is an open question worth
answering from the first real run.

### What retries on OOM, and what does not

| Path | On OOM |
|---|---|
| ESM/ESMC scoring (`esm_eval.py:389`) | halves the batch, `empty_cache()`, retries; raises at 1 |
| ESMFold2 `fold_batch` | length-buckets to a token budget and retries — upstream `esm`, not this repo, and unverified |
| Generation | nothing catches it; the shard dies |

Scoring is batch-invariant — a sequence's log-probs do not depend on what it was
batched with — so halving and retrying returns identical numbers, and the retry is
free of consequences.

Generation is different, but **not because different designs would be wrong**.
Sampling is stochastic; if batching is independent, a smaller batch draws from the
same distribution and which particular designs come out does not matter
scientifically. The objection is bookkeeping, and it is specific:

- **The digest would describe a run that did not happen.**
  `generation_config_digest` hashes the whole `generation` subtree, `batch_size`
  included, and the marker records that digest. An in-process retry changes the
  effective batch without changing the config, so the shard would carry a marker
  claiming a batch size it did not use — and nothing anywhere records the
  effective one. Resume, skip and the config-mismatch abort all rest on that
  digest meaning what produced the output.
- **A mid-shard retry makes it worse.** Some designs at 8, the rest at 4, one
  number in the record. There is no single value that would be true.
- **Distribution-invariance is an assumption, not a guarantee.** It holds when
  batch members are independent and masking is exact. Binder lengths here vary
  (`nres: 40-70`), so batches are padded to their longest member — which is
  precisely where a masking flaw would make batch composition systematic rather
  than incidental. Worth verifying before relying on it.

So a generation retry is defensible engineering, not a scientific error, and the
thing that would make it safe is recording the effective batch size per shard
rather than inheriting the configured one. Until that exists, the lever is
`batch_size` in the config, set before the run.

### Can they hand memory back between turns?

Generation alternates — a diffusion step, then AF2 scoring — so it is fair to ask
whether each side can return what it is not using. Partly, and less than it
sounds, because the two costs are different:

- **Weights stay resident either way.** `self.reward_model` is cached on the
  LightningModule for the whole predict run, and `mk_afdesign_model` holds the AF2
  parameters inside it; the diffusion model is resident throughout by definition.
  Only *activations* are recoverable, so the floor is both sets of weights plus two
  CUDA contexts no matter what these settings say.
- **JAX**: `XLA_PYTHON_CLIENT_ALLOCATOR=platform` is the only setting that truly
  returns buffers — it swaps the BFC pool for direct `cudaMalloc`/`cudaFree`. BFC
  grows and never shrinks, so `PREALLOCATE=false` alone caps the *start*, not the
  high-water mark. `platform` is documented as a debugging aid and allocation is
  markedly slower; treat it as a last resort, not a default.
- **`jax.clear_caches()` does not do this.** It clears *compilation* caches. The
  AF2 reward already calls it twice in `_cleanup_jax_state`
  (`alphafold2_reward.py:378-382`, under a TODO), and it frees no device memory.
- **PyTorch**: `torch.cuda.empty_cache()` is the mechanism that returns cached
  blocks to the driver — `tmol_reward.py:570` already does this, the AF2 path does
  not. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` addresses
  **fragmentation**, not release: it lets a segment grow and shrink inside one
  virtual reservation. For proactive reclaim use
  `garbage_collection_threshold:0.8` alongside it.

So the honest ceiling on alternating-release is one side's activation peak, and
the price is slower allocation on the JAX side. Measure before paying it: run once
with `PREALLOCATE=false` on a pinned card and watch
`nvidia-smi --query-gpu=memory.used --format=csv -l 5`. That gives both real peaks
in a single run, which is what a fraction should be set from.

Pick the fraction from a real run rather than guessing: the OOM message reports
torch's own allocation (`1.87` and `3.12 GiB` on the CBLN1 smoke test), so
`1 - (torch_peak + headroom) / total` is a measurement, not an estimate.

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
