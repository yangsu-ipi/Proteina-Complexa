# ESMC scoring, ESMFold2 advisory complex folding, ESMFold2 apo folding

Three capabilities that share one dependency and one trap each. All are `metric.*`
keys on any `evaluate_*.yaml`; the design pipeline reads the same keys from
`configs/pipeline/binder/binder_evaluate.yaml`.

## Prerequisite for all three

ESMC and ESMFold2 ship in **one** source-only release, and the model classes live
in **Biohub's transformers fork**, not PyPI's. `build_blackwell.sh` installs both
by default (`WITH_ESMFOLD2=0` opts out); `ESM_SRC` points at the esmfold2 tree.

Weights are **gated HF repos**: set `HF_TOKEN` and accept the licences. Point
`HF_HOME` at the hub cache root, never at a snapshot directory — an
explicit `cache_dir=` overrides `HF_HOME`, and an empty-but-existing directory
resolves as the first load location and then fails there.

> **No ESMFold2 path in this repo has been run against real weights.** Treat the
> first run of either ESMFold2 feature as a debugging run, and prefer to change
> one thing at a time.

---

## 1. ESMC for pseudo-perplexity and log-likelihood

```yaml
metric:
  compute_esm_metrics: true
  esm_model: biohub/ESMC-6B     # any transformers-format ESMC repo
  esm_backend: esmc             # or omit: "auto" routes any name containing "esmc" here
  esm_batch_tokens: 16384       # rows x padded length per forward
  reuse_cached_esm: true        # cached per design in esm_eval_cache.json
```

Emits, per sequence type, **write-only** (nothing gates on them):
`{seq}_esm_pseudo_perplexity`, `{seq}_esm_log_likelihood`, and both `_all` lists.

**Trap: three backends, one of which cannot load.**

| `esm_backend` | route | use |
|---|---|---|
| `esm2` | HF `AutoModelForMaskedLM` | ESM2, e.g. `facebook/esm2_t33_650M_UR50D` |
| `esmc` | the *same* HF loader | **ESMC — this is the working route** |
| `esmc_pkg` | the `esm` package's own `ESMC` class | **never**; builders leave parameters on the meta device, so moving the model raises |

`auto` resolves ESMC to `esmc`, the working route. Do not "fix" a load failure by
switching to `esmc_pkg`.

**Cost.** ~15.5 s per 140-residue sequence on ESMC-6B, so ~30 s per design at two
sequences. The cache is keyed on (model, backend, sequence), so it survives a
resume and reuses per sequence when `sequence_types` grows.

**Values are not comparable to ESM2's**: ESMC `from_pretrained` casts to bfloat16
on GPU while the ESM2 path is fp32.

---

## 2. ESMFold2 advisory complex folding, with a target MSA

```yaml
metric:
  consensus_backends: [esmfold2]     # [] disables; "esmfold2" is the only registered backend
  consensus_best_only: false         # false = score every sequence (see below)
  reuse_cached_consensus: true       # per design, one cache file per backend
  consensus_cfg:
    target_msa: /path/to/target.a3m  # single-chain target
    # target_msa_paths: [/p/a.a3m, null]   # OR one entry per target chain, null where none
    msa_max_sequences: 16384
    num_loops: 20
    num_sampling_steps: 200
    num_diffusion_samples: 1
    # model_id: biohub/ESMFold2-Experimental-Cutoff2025   # default; the MSA-capable checkpoint
    # cuda: true
```

Emits `{seq}_esmfold2_{i_pAE,i_pTM,pTM,pLDDT}` (+ `_all`) and
`{seq}_esmfold2_pdb_path`. **Advisory: gates nothing**, enforced at runtime by
`assert_columns_are_advisory`, which raises if an emitted name collides with a
gated one. Deliberately not named `*_complex_*` for that reason.

**The binder never gets an MSA**, only the target. De novo miniproteins have no
meaningful alignment and a spurious one makes the prediction worse. There is no
key to give the binder one.

**`target_msa_paths` must have exactly one entry per target chain** — `null` for
chains without an alignment — or it raises with the counts. `target_msa` is the
single-chain shorthand and fills the remaining chains with `null`.

**Leave `consensus_best_only: false`.** Folding only the primary backend's
ranked-best sequence conditions the advisory sample on the ranking it is meant to
be checked against: rank disagreement becomes unmeasurable, any fit is estimated
on the primary's upper tail, and the sequences the primary rejected — the
interesting failures — are never folded. `true` is for cheap monitoring of a
backend already characterised.

**Skipped entirely for ligand targets**: these backends fold protein complexes, so
there is no target sequence to fold against.

MSA *contents* are digested into the cache fingerprint, so editing an alignment in
place invalidates rather than serving scores computed against the old one.

---

## 3. ESMFold2 (fast checkpoint) for apo folding, single sequence

Apo folding folds each sequence **alone** and measures it against the designed
binder backbone — the same construction as `{seq}_binder_scRMSD_ca`, minus the
target. On by default with plain ESMFold; to use ESMFold2:

```yaml
metric:
  compute_apo_metrics: true
  apo_folding_models: [esmfold2]   # -> biohub/ESMFold2-Experimental-Fast-Cutoff2025
  apo_rmsd_modes: [ca]
  reuse_cached_apo_folds: true
```

**No threshold override is needed.** The apo criterion is `scRMSD_ca_{model}`,
expanded against the apo columns the run produced, so `[esmfold]`, `[esmfold2]`
and `[esmfold, esmfold2]` all gate correctly. With two models listed a sequence
must pass under **both** — the conservative reading, and the cheapest way to see
whether two predictors agree before committing to one.

Single-sequence by construction: the monomer path passes no MSA, and the Fast
checkpoint is the fork's own choice for single-chain single-sequence work. Model
id override: `ESMFOLD2_MONOMER_MODEL` (complex side: `ESMFOLD2_COMPLEX_MODEL`).

**`apo_rmsd_modes` is not templated, only the model is.** The gate is defined for
`ca` at 2.0 Å; asking for `bb3o` or `all_atom` emits those columns ungated,
because all-atom RMSD is systematically larger and the same threshold would not
transfer. Evaluation warns at startup if the configured modes cannot satisfy the
criterion.

**If no apo column matches, no verdict is produced** — `{seq}_pass*` is absent
and pass rates are skipped, with an error naming the expected column shape. It
does not fall back to the three holo criteria: a design passing a three-criterion
gate must not be indistinguishable from one passing the four it was meant to
face.

`self`'s apo fold is delegated to the codesignability computation, so
`self_apo_scRMSD_{mode}_{model}` and `_res_co_scRMSD_{mode}_{model}` are the same
number sharing one fold. If they ever disagree, something upstream of both is
wrong.

**Caveat to report with any apo number:** apo and holo scRMSD are the same
quantity from *different predictors* (apo: `apo_folding_models`; holo:
`binder_folding_method`), so their difference mixes target dependence with
predictor disagreement, and the 1.5 Å and 2.0 Å thresholds are not calibrated
against each other.

---

## Columns added by this work, for reading results

| Column | Meaning |
|---|---|
| `{seq}_pass`, `{seq}_pass_all` | Per-sequence 1/0 against every success criterion. Absent (not zero) when a criterion's column is missing |
| `{seq}_redesign_score`, `_all` | The inverse folder's own per-sequence score |
| `redesign_score_kind` | `nll_lower_better` (protein_mpnn) or `confidence_higher_better` (soluble/ligand). **Read before sorting** |
| `redesign_model` | Which inverse folder produced the redesigns |
| `redesign_conditioning` | `complex` or `binder_only` for the designability redesigns |
| `{seq}_binder_scRMSD_target_aligned_ca` | Binder RMSD superimposed on the **target** — fold *and* placement. Emitted, not gated |

`metric.inverse_folding_model` now governs **both** the complex track and
designability. Designability values are not comparable across runs that set it
differently. Ligand targets force `ligand_mpnn` with a warning if configured
otherwise.
