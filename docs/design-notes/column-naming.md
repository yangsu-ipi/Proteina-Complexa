# Result column names: which structure, which part, which model

**Status:** decided, not yet implemented. Records the convention every result
column will follow, and the migration from the four schemes it replaces.

## Why

Four naming schemes coexist today, and none of them says which folding model
produced a number:

- `{seq}_complex_*` — AF2, but only by circumstance; nothing in the name says so
- `{seq}_esmfold2_*` — advisory, backend as an infix
- `{seq}_apo_*_{model}` — backend as a *suffix*
- `generated_*` / `refolded_{seq}_*` — a blanket rename applied in
  `evaluate.py` and `binder_eval.py`, with no slot for the model at all

`refolded_` is the sharpest case: the structures come from `best_paths_dict`,
whatever refolder filled those columns. Today that is AF2, so
`refolded_self_binder_interface_dSASA` is an AF2 number — but neither the name
nor the code says it, and pointing the same block at ESMFold2 structures would
silently produce a column that means something else.

## The convention

`{seq_type}_{kind}_{backend}_{scope}_{metric}`

| slot | values | notes |
|---|---|---|
| seq_type | `self`, `mpnn`, `mpnn_fixed` | omitted for `generated` (one co-designed sequence). Match LONGEST-FIRST: `mpnn_fixed` before `mpnn`, or it parses as `mpnn` + `fixed_...` |
| kind | `complex`, `apo` | the two structures produced. `binder`/`target` were never kinds |
| backend | `af2`, `esmfold2`, `generated` | `generated` is a backend, not a prefix |
| scope | `binder`, `target`, `interface`, `binder_interface`, `target_interface` | omitted = whole structure |
| metric | the quantity | alignment frame stays in the metric (`scRMSD_target_aligned_ca`) |

Grandfathered, no scope slot: `i_pAE`, `i_pTM`, `min_ipAE`, `*ipSAE*` — standard field terms, inherently interfacial.

**Construction only, never parsed.** `binder` + `interface_nres` and `binder_interface` + `nres` render
identically, so a parser would be ambiguous. Callers build names from slots and consult a registry;
the registry asserts uniqueness at import so a future collision is a startup error.

**Target SS exists only for `esmfold2`.** Measured over 25 designs, the target's ordered-SS fraction
(quantised at 1/136 = 0.0074) varies by at most 1 residue for `generated` (it *is* the input target:
0.000 A superposed, res_name identity 1.000) and for `af2` (templated on the target); `esmfold2`,
untemplated, varies by up to 4. `target_dSASA` is kept for all backends -- burial depends on where
the binder lands even when the target's coordinates do not move.

Totals for this campaign's seq types (self, mpnn): 217 today -> 355 — 134 renamed, 80 unchanged, 3 retired, 141 new. Every `{seq}_` row generalises to `mpnn_fixed`;
the alias map is generated over the full vocabulary so a campaign using it migrates too.

## Renamed (134)

| old | new | why |
|---|---|---|
| `mpnn_aa_interface_counts` | `mpnn_complex_generated_binder_interface_aa_counts` | generated defines the interface (verified) |
| `mpnn_apo_pLDDT_esmfold2` | `mpnn_apo_esmfold2_binder_pLDDT` | backend moves from suffix to slot |
| `mpnn_apo_pLDDT_esmfold2_all` | `mpnn_apo_esmfold2_binder_pLDDT_all` | backend moves from suffix to slot |
| `mpnn_apo_scRMSD_ca_esmfold2` | `mpnn_apo_esmfold2_binder_scRMSD_ca` | backend moves from suffix to slot |
| `mpnn_apo_scRMSD_ca_esmfold2_all` | `mpnn_apo_esmfold2_binder_scRMSD_ca_all` | backend moves from suffix to slot |
| `mpnn_binder_scRMSD` | `mpnn_complex_af2_binder_scRMSD` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_all` | `mpnn_complex_af2_binder_scRMSD_all` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_allatom` | `mpnn_complex_af2_binder_scRMSD_allatom` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_allatom_all` | `mpnn_complex_af2_binder_scRMSD_allatom_all` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_bb3` | `mpnn_complex_af2_binder_scRMSD_bb3` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_bb3_all` | `mpnn_complex_af2_binder_scRMSD_bb3_all` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_bb3o` | `mpnn_complex_af2_binder_scRMSD_bb3o` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_bb3o_all` | `mpnn_complex_af2_binder_scRMSD_bb3o_all` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_ca` | `mpnn_complex_af2_binder_scRMSD_ca` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_ca_all` | `mpnn_complex_af2_binder_scRMSD_ca_all` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_target_aligned_ca` | `mpnn_complex_af2_binder_scRMSD_target_aligned_ca` | binder was a scope, not a kind |
| `mpnn_binder_scRMSD_target_aligned_ca_all` | `mpnn_complex_af2_binder_scRMSD_target_aligned_ca_all` | binder was a scope, not a kind |
| `mpnn_complex_avg_ipSAE` | `mpnn_complex_af2_avg_ipSAE` | AF2 named explicitly |
| `mpnn_complex_avg_ipSAE_10` | `mpnn_complex_af2_avg_ipSAE_10` | AF2 named explicitly |
| `mpnn_complex_avg_ipSAE_10_all` | `mpnn_complex_af2_avg_ipSAE_10_all` | AF2 named explicitly |
| `mpnn_complex_avg_ipSAE_all` | `mpnn_complex_af2_avg_ipSAE_all` | AF2 named explicitly |
| `mpnn_complex_binder_pLDDT` | `mpnn_complex_af2_binder_pLDDT` | AF2 named explicitly |
| `mpnn_complex_binder_pLDDT_all` | `mpnn_complex_af2_binder_pLDDT_all` | AF2 named explicitly |
| `mpnn_complex_i_pAE` | `mpnn_complex_af2_i_pAE` | AF2 named explicitly |
| `mpnn_complex_i_pAE_all` | `mpnn_complex_af2_i_pAE_all` | AF2 named explicitly |
| `mpnn_complex_i_pTM` | `mpnn_complex_af2_i_pTM` | AF2 named explicitly |
| `mpnn_complex_i_pTM_all` | `mpnn_complex_af2_i_pTM_all` | AF2 named explicitly |
| `mpnn_complex_max_ipSAE` | `mpnn_complex_af2_max_ipSAE` | AF2 named explicitly |
| `mpnn_complex_max_ipSAE_10` | `mpnn_complex_af2_max_ipSAE_10` | AF2 named explicitly |
| `mpnn_complex_max_ipSAE_10_all` | `mpnn_complex_af2_max_ipSAE_10_all` | AF2 named explicitly |
| `mpnn_complex_max_ipSAE_all` | `mpnn_complex_af2_max_ipSAE_all` | AF2 named explicitly |
| `mpnn_complex_min_ipAE` | `mpnn_complex_af2_min_ipAE` | AF2 named explicitly |
| `mpnn_complex_min_ipAE_all` | `mpnn_complex_af2_min_ipAE_all` | AF2 named explicitly |
| `mpnn_complex_min_ipSAE` | `mpnn_complex_af2_min_ipSAE` | AF2 named explicitly |
| `mpnn_complex_min_ipSAE_10` | `mpnn_complex_af2_min_ipSAE_10` | AF2 named explicitly |
| `mpnn_complex_min_ipSAE_10_all` | `mpnn_complex_af2_min_ipSAE_10_all` | AF2 named explicitly |
| `mpnn_complex_min_ipSAE_all` | `mpnn_complex_af2_min_ipSAE_all` | AF2 named explicitly |
| `mpnn_complex_pAE` | `mpnn_complex_af2_pAE` | AF2 named explicitly |
| `mpnn_complex_pAE_all` | `mpnn_complex_af2_pAE_all` | AF2 named explicitly |
| `mpnn_complex_pTM` | `mpnn_complex_af2_pTM` | AF2 named explicitly |
| `mpnn_complex_pTM_all` | `mpnn_complex_af2_pTM_all` | AF2 named explicitly |
| `mpnn_complex_pdb_path` | `mpnn_complex_af2_pdb_path` | AF2 named explicitly |
| `mpnn_complex_pdb_path_all` | `mpnn_complex_af2_pdb_path_all` | AF2 named explicitly |
| `mpnn_complex_scRMSD` | `mpnn_complex_af2_scRMSD` | AF2 named explicitly |
| `mpnn_complex_scRMSD_all` | `mpnn_complex_af2_scRMSD_all` | AF2 named explicitly |
| `mpnn_complex_scRMSD_ca` | `mpnn_complex_af2_scRMSD_ca` | AF2 named explicitly |
| `mpnn_complex_scRMSD_ca_all` | `mpnn_complex_af2_scRMSD_ca_all` | AF2 named explicitly |
| `mpnn_complex_target_pLDDT` | `mpnn_complex_af2_target_pLDDT` | AF2 named explicitly |
| `mpnn_complex_target_pLDDT_all` | `mpnn_complex_af2_target_pLDDT_all` | AF2 named explicitly |
| `mpnn_esmfold2_binder_pLDDT` | `mpnn_complex_esmfold2_binder_pLDDT` | gains kind slot |
| `mpnn_esmfold2_binder_pLDDT_all` | `mpnn_complex_esmfold2_binder_pLDDT_all` | gains kind slot |
| `mpnn_esmfold2_binder_pLDDT_low_outlier` | `mpnn_complex_esmfold2_binder_pLDDT_low_outlier` | gains kind slot |
| `mpnn_esmfold2_binder_pLDDT_robust_z` | `mpnn_complex_esmfold2_binder_pLDDT_robust_z` | gains kind slot |
| `mpnn_esmfold2_i_pAE` | `mpnn_complex_esmfold2_i_pAE` | gains kind slot |
| `mpnn_esmfold2_i_pAE_all` | `mpnn_complex_esmfold2_i_pAE_all` | gains kind slot |
| `mpnn_esmfold2_i_pTM` | `mpnn_complex_esmfold2_i_pTM` | gains kind slot |
| `mpnn_esmfold2_i_pTM_all` | `mpnn_complex_esmfold2_i_pTM_all` | gains kind slot |
| `mpnn_esmfold2_pLDDT` | `mpnn_complex_esmfold2_pLDDT` | gains kind slot |
| `mpnn_esmfold2_pLDDT_all` | `mpnn_complex_esmfold2_pLDDT_all` | gains kind slot |
| `mpnn_esmfold2_pTM` | `mpnn_complex_esmfold2_pTM` | gains kind slot |
| `mpnn_esmfold2_pTM_all` | `mpnn_complex_esmfold2_pTM_all` | gains kind slot |
| `mpnn_esmfold2_pdb_path` | `mpnn_complex_esmfold2_pdb_path` | gains kind slot |
| `mpnn_esmfold2_pdb_path_all` | `mpnn_complex_esmfold2_pdb_path_all` | gains kind slot |
| `mpnn_esmfold2_target_pLDDT` | `mpnn_complex_esmfold2_target_pLDDT` | gains kind slot |
| `mpnn_esmfold2_target_pLDDT_all` | `mpnn_complex_esmfold2_target_pLDDT_all` | gains kind slot |
| `mpnn_esmfold2_target_pLDDT_low_outlier` | `mpnn_complex_esmfold2_target_pLDDT_low_outlier` | gains kind slot |
| `mpnn_esmfold2_target_pLDDT_robust_z` | `mpnn_complex_esmfold2_target_pLDDT_robust_z` | gains kind slot |
| `self_aa_interface_counts` | `self_complex_generated_binder_interface_aa_counts` | generated defines the interface (verified) |
| `self_apo_pLDDT_esmfold2` | `self_apo_esmfold2_binder_pLDDT` | backend moves from suffix to slot |
| `self_apo_pLDDT_esmfold2_all` | `self_apo_esmfold2_binder_pLDDT_all` | backend moves from suffix to slot |
| `self_apo_scRMSD_ca_esmfold2` | `self_apo_esmfold2_binder_scRMSD_ca` | backend moves from suffix to slot |
| `self_apo_scRMSD_ca_esmfold2_all` | `self_apo_esmfold2_binder_scRMSD_ca_all` | backend moves from suffix to slot |
| `self_binder_scRMSD` | `self_complex_af2_binder_scRMSD` | binder was a scope, not a kind |
| `self_binder_scRMSD_all` | `self_complex_af2_binder_scRMSD_all` | binder was a scope, not a kind |
| `self_binder_scRMSD_allatom` | `self_complex_af2_binder_scRMSD_allatom` | binder was a scope, not a kind |
| `self_binder_scRMSD_allatom_all` | `self_complex_af2_binder_scRMSD_allatom_all` | binder was a scope, not a kind |
| `self_binder_scRMSD_bb3` | `self_complex_af2_binder_scRMSD_bb3` | binder was a scope, not a kind |
| `self_binder_scRMSD_bb3_all` | `self_complex_af2_binder_scRMSD_bb3_all` | binder was a scope, not a kind |
| `self_binder_scRMSD_bb3o` | `self_complex_af2_binder_scRMSD_bb3o` | binder was a scope, not a kind |
| `self_binder_scRMSD_bb3o_all` | `self_complex_af2_binder_scRMSD_bb3o_all` | binder was a scope, not a kind |
| `self_binder_scRMSD_ca` | `self_complex_af2_binder_scRMSD_ca` | binder was a scope, not a kind |
| `self_binder_scRMSD_ca_all` | `self_complex_af2_binder_scRMSD_ca_all` | binder was a scope, not a kind |
| `self_binder_scRMSD_target_aligned_ca` | `self_complex_af2_binder_scRMSD_target_aligned_ca` | binder was a scope, not a kind |
| `self_binder_scRMSD_target_aligned_ca_all` | `self_complex_af2_binder_scRMSD_target_aligned_ca_all` | binder was a scope, not a kind |
| `self_complex_avg_ipSAE` | `self_complex_af2_avg_ipSAE` | AF2 named explicitly |
| `self_complex_avg_ipSAE_10` | `self_complex_af2_avg_ipSAE_10` | AF2 named explicitly |
| `self_complex_avg_ipSAE_10_all` | `self_complex_af2_avg_ipSAE_10_all` | AF2 named explicitly |
| `self_complex_avg_ipSAE_all` | `self_complex_af2_avg_ipSAE_all` | AF2 named explicitly |
| `self_complex_binder_pLDDT` | `self_complex_af2_binder_pLDDT` | AF2 named explicitly |
| `self_complex_binder_pLDDT_all` | `self_complex_af2_binder_pLDDT_all` | AF2 named explicitly |
| `self_complex_i_pAE` | `self_complex_af2_i_pAE` | AF2 named explicitly |
| `self_complex_i_pAE_all` | `self_complex_af2_i_pAE_all` | AF2 named explicitly |
| `self_complex_i_pTM` | `self_complex_af2_i_pTM` | AF2 named explicitly |
| `self_complex_i_pTM_all` | `self_complex_af2_i_pTM_all` | AF2 named explicitly |
| `self_complex_max_ipSAE` | `self_complex_af2_max_ipSAE` | AF2 named explicitly |
| `self_complex_max_ipSAE_10` | `self_complex_af2_max_ipSAE_10` | AF2 named explicitly |
| `self_complex_max_ipSAE_10_all` | `self_complex_af2_max_ipSAE_10_all` | AF2 named explicitly |
| `self_complex_max_ipSAE_all` | `self_complex_af2_max_ipSAE_all` | AF2 named explicitly |
| `self_complex_min_ipAE` | `self_complex_af2_min_ipAE` | AF2 named explicitly |
| `self_complex_min_ipAE_all` | `self_complex_af2_min_ipAE_all` | AF2 named explicitly |
| `self_complex_min_ipSAE` | `self_complex_af2_min_ipSAE` | AF2 named explicitly |
| `self_complex_min_ipSAE_10` | `self_complex_af2_min_ipSAE_10` | AF2 named explicitly |
| `self_complex_min_ipSAE_10_all` | `self_complex_af2_min_ipSAE_10_all` | AF2 named explicitly |
| `self_complex_min_ipSAE_all` | `self_complex_af2_min_ipSAE_all` | AF2 named explicitly |
| `self_complex_pAE` | `self_complex_af2_pAE` | AF2 named explicitly |
| `self_complex_pAE_all` | `self_complex_af2_pAE_all` | AF2 named explicitly |
| `self_complex_pTM` | `self_complex_af2_pTM` | AF2 named explicitly |
| `self_complex_pTM_all` | `self_complex_af2_pTM_all` | AF2 named explicitly |
| `self_complex_pdb_path` | `self_complex_af2_pdb_path` | AF2 named explicitly |
| `self_complex_pdb_path_all` | `self_complex_af2_pdb_path_all` | AF2 named explicitly |
| `self_complex_scRMSD` | `self_complex_af2_scRMSD` | AF2 named explicitly |
| `self_complex_scRMSD_all` | `self_complex_af2_scRMSD_all` | AF2 named explicitly |
| `self_complex_scRMSD_ca` | `self_complex_af2_scRMSD_ca` | AF2 named explicitly |
| `self_complex_scRMSD_ca_all` | `self_complex_af2_scRMSD_ca_all` | AF2 named explicitly |
| `self_complex_target_pLDDT` | `self_complex_af2_target_pLDDT` | AF2 named explicitly |
| `self_complex_target_pLDDT_all` | `self_complex_af2_target_pLDDT_all` | AF2 named explicitly |
| `self_esmfold2_binder_pLDDT` | `self_complex_esmfold2_binder_pLDDT` | gains kind slot |
| `self_esmfold2_binder_pLDDT_all` | `self_complex_esmfold2_binder_pLDDT_all` | gains kind slot |
| `self_esmfold2_binder_pLDDT_low_outlier` | `self_complex_esmfold2_binder_pLDDT_low_outlier` | gains kind slot |
| `self_esmfold2_binder_pLDDT_robust_z` | `self_complex_esmfold2_binder_pLDDT_robust_z` | gains kind slot |
| `self_esmfold2_i_pAE` | `self_complex_esmfold2_i_pAE` | gains kind slot |
| `self_esmfold2_i_pAE_all` | `self_complex_esmfold2_i_pAE_all` | gains kind slot |
| `self_esmfold2_i_pTM` | `self_complex_esmfold2_i_pTM` | gains kind slot |
| `self_esmfold2_i_pTM_all` | `self_complex_esmfold2_i_pTM_all` | gains kind slot |
| `self_esmfold2_pLDDT` | `self_complex_esmfold2_pLDDT` | gains kind slot |
| `self_esmfold2_pLDDT_all` | `self_complex_esmfold2_pLDDT_all` | gains kind slot |
| `self_esmfold2_pTM` | `self_complex_esmfold2_pTM` | gains kind slot |
| `self_esmfold2_pTM_all` | `self_complex_esmfold2_pTM_all` | gains kind slot |
| `self_esmfold2_pdb_path` | `self_complex_esmfold2_pdb_path` | gains kind slot |
| `self_esmfold2_pdb_path_all` | `self_complex_esmfold2_pdb_path_all` | gains kind slot |
| `self_esmfold2_target_pLDDT` | `self_complex_esmfold2_target_pLDDT` | gains kind slot |
| `self_esmfold2_target_pLDDT_all` | `self_complex_esmfold2_target_pLDDT_all` | gains kind slot |
| `self_esmfold2_target_pLDDT_low_outlier` | `self_complex_esmfold2_target_pLDDT_low_outlier` | gains kind slot |
| `self_esmfold2_target_pLDDT_robust_z` | `self_complex_esmfold2_target_pLDDT_robust_z` | gains kind slot |

## Retired (3)

- `_res_ss_alpha` — retired (biotite P-SEA: 4x over-calls beta)
- `_res_ss_beta` — retired (biotite P-SEA: 4x over-calls beta)
- `_res_ss_coil` — retired (biotite P-SEA: 4x over-calls beta)

## New (141)

- `complex_generated_binder_buried_fraction` — SASA
- `complex_generated_binder_dSASA` — SASA
- `complex_generated_binder_interface_hydrophobicity` — SASA
- `complex_generated_binder_interface_nres` — SASA
- `complex_generated_binder_interface_ss_counts` — SS
- `complex_generated_binder_interface_ss_total` — SS
- `complex_generated_binder_ss_counts` — SS
- `complex_generated_binder_ss_total` — SS
- `complex_generated_binder_surface_hydrophobicity` — SASA
- `complex_generated_interface_dSASA` — SASA
- `complex_generated_interface_sc` — SASA
- `complex_generated_target_dSASA` — SASA
- `complex_generated_target_interface_nres` — SASA
- `mpnn_apo_esmfold2_binder_ss_counts` — SS (apo)
- `mpnn_apo_esmfold2_binder_ss_counts_all` — SS (apo)
- `mpnn_apo_esmfold2_binder_ss_total` — SS (apo)
- `mpnn_apo_esmfold2_binder_ss_total_all` — SS (apo)
- `mpnn_complex_af2_binder_buried_fraction` — SASA
- `mpnn_complex_af2_binder_buried_fraction_all` — SASA
- `mpnn_complex_af2_binder_dSASA` — SASA
- `mpnn_complex_af2_binder_dSASA_all` — SASA
- `mpnn_complex_af2_binder_interface_hydrophobicity` — SASA
- `mpnn_complex_af2_binder_interface_hydrophobicity_all` — SASA
- `mpnn_complex_af2_binder_interface_nres` — SASA
- `mpnn_complex_af2_binder_interface_nres_all` — SASA
- `mpnn_complex_af2_binder_interface_ss_counts` — SS
- `mpnn_complex_af2_binder_interface_ss_counts_all` — SS
- `mpnn_complex_af2_binder_interface_ss_total` — SS
- `mpnn_complex_af2_binder_interface_ss_total_all` — SS
- `mpnn_complex_af2_binder_ss_counts` — SS
- `mpnn_complex_af2_binder_ss_counts_all` — SS
- `mpnn_complex_af2_binder_ss_total` — SS
- `mpnn_complex_af2_binder_ss_total_all` — SS
- `mpnn_complex_af2_binder_surface_hydrophobicity` — SASA
- `mpnn_complex_af2_binder_surface_hydrophobicity_all` — SASA
- `mpnn_complex_af2_interface_dSASA` — SASA
- `mpnn_complex_af2_interface_dSASA_all` — SASA
- `mpnn_complex_af2_interface_sc` — SASA
- `mpnn_complex_af2_interface_sc_all` — SASA
- `mpnn_complex_af2_target_dSASA` — SASA
- `mpnn_complex_af2_target_dSASA_all` — SASA
- `mpnn_complex_af2_target_interface_nres` — SASA
- `mpnn_complex_af2_target_interface_nres_all` — SASA
- `mpnn_complex_esmfold2_binder_buried_fraction` — SASA
- `mpnn_complex_esmfold2_binder_buried_fraction_all` — SASA
- `mpnn_complex_esmfold2_binder_dSASA` — SASA
- `mpnn_complex_esmfold2_binder_dSASA_all` — SASA
- `mpnn_complex_esmfold2_binder_interface_hydrophobicity` — SASA
- `mpnn_complex_esmfold2_binder_interface_hydrophobicity_all` — SASA
- `mpnn_complex_esmfold2_binder_interface_nres` — SASA
- `mpnn_complex_esmfold2_binder_interface_nres_all` — SASA
- `mpnn_complex_esmfold2_binder_interface_ss_counts` — SS
- `mpnn_complex_esmfold2_binder_interface_ss_counts_all` — SS
- `mpnn_complex_esmfold2_binder_interface_ss_total` — SS
- `mpnn_complex_esmfold2_binder_interface_ss_total_all` — SS
- `mpnn_complex_esmfold2_binder_ss_counts` — SS
- `mpnn_complex_esmfold2_binder_ss_counts_all` — SS
- `mpnn_complex_esmfold2_binder_ss_total` — SS
- `mpnn_complex_esmfold2_binder_ss_total_all` — SS
- `mpnn_complex_esmfold2_binder_surface_hydrophobicity` — SASA
- `mpnn_complex_esmfold2_binder_surface_hydrophobicity_all` — SASA
- `mpnn_complex_esmfold2_interface_dSASA` — SASA
- `mpnn_complex_esmfold2_interface_dSASA_all` — SASA
- `mpnn_complex_esmfold2_interface_sc` — SASA
- `mpnn_complex_esmfold2_interface_sc_all` — SASA
- `mpnn_complex_esmfold2_target_dSASA` — SASA
- `mpnn_complex_esmfold2_target_dSASA_all` — SASA
- `mpnn_complex_esmfold2_target_interface_nres` — SASA
- `mpnn_complex_esmfold2_target_interface_nres_all` — SASA
- `mpnn_complex_esmfold2_target_interface_ss_counts` — SS
- `mpnn_complex_esmfold2_target_interface_ss_counts_all` — SS
- `mpnn_complex_esmfold2_target_interface_ss_total` — SS
- `mpnn_complex_esmfold2_target_interface_ss_total_all` — SS
- `mpnn_complex_esmfold2_target_ss_counts` — SS
- `mpnn_complex_esmfold2_target_ss_counts_all` — SS
- `mpnn_complex_esmfold2_target_ss_total` — SS
- `mpnn_complex_esmfold2_target_ss_total_all` — SS
- `self_apo_esmfold2_binder_ss_counts` — SS (apo)
- `self_apo_esmfold2_binder_ss_counts_all` — SS (apo)
- `self_apo_esmfold2_binder_ss_total` — SS (apo)
- `self_apo_esmfold2_binder_ss_total_all` — SS (apo)
- `self_complex_af2_binder_buried_fraction` — SASA
- `self_complex_af2_binder_buried_fraction_all` — SASA
- `self_complex_af2_binder_dSASA` — SASA
- `self_complex_af2_binder_dSASA_all` — SASA
- `self_complex_af2_binder_interface_hydrophobicity` — SASA
- `self_complex_af2_binder_interface_hydrophobicity_all` — SASA
- `self_complex_af2_binder_interface_nres` — SASA
- `self_complex_af2_binder_interface_nres_all` — SASA
- `self_complex_af2_binder_interface_ss_counts` — SS
- `self_complex_af2_binder_interface_ss_counts_all` — SS
- `self_complex_af2_binder_interface_ss_total` — SS
- `self_complex_af2_binder_interface_ss_total_all` — SS
- `self_complex_af2_binder_ss_counts` — SS
- `self_complex_af2_binder_ss_counts_all` — SS
- `self_complex_af2_binder_ss_total` — SS
- `self_complex_af2_binder_ss_total_all` — SS
- `self_complex_af2_binder_surface_hydrophobicity` — SASA
- `self_complex_af2_binder_surface_hydrophobicity_all` — SASA
- `self_complex_af2_interface_dSASA` — SASA
- `self_complex_af2_interface_dSASA_all` — SASA
- `self_complex_af2_interface_sc` — SASA
- `self_complex_af2_interface_sc_all` — SASA
- `self_complex_af2_target_dSASA` — SASA
- `self_complex_af2_target_dSASA_all` — SASA
- `self_complex_af2_target_interface_nres` — SASA
- `self_complex_af2_target_interface_nres_all` — SASA
- `self_complex_esmfold2_binder_buried_fraction` — SASA
- `self_complex_esmfold2_binder_buried_fraction_all` — SASA
- `self_complex_esmfold2_binder_dSASA` — SASA
- `self_complex_esmfold2_binder_dSASA_all` — SASA
- `self_complex_esmfold2_binder_interface_hydrophobicity` — SASA
- `self_complex_esmfold2_binder_interface_hydrophobicity_all` — SASA
- `self_complex_esmfold2_binder_interface_nres` — SASA
- `self_complex_esmfold2_binder_interface_nres_all` — SASA
- `self_complex_esmfold2_binder_interface_ss_counts` — SS
- `self_complex_esmfold2_binder_interface_ss_counts_all` — SS
- `self_complex_esmfold2_binder_interface_ss_total` — SS
- `self_complex_esmfold2_binder_interface_ss_total_all` — SS
- `self_complex_esmfold2_binder_ss_counts` — SS
- `self_complex_esmfold2_binder_ss_counts_all` — SS
- `self_complex_esmfold2_binder_ss_total` — SS
- `self_complex_esmfold2_binder_ss_total_all` — SS
- `self_complex_esmfold2_binder_surface_hydrophobicity` — SASA
- `self_complex_esmfold2_binder_surface_hydrophobicity_all` — SASA
- `self_complex_esmfold2_interface_dSASA` — SASA
- `self_complex_esmfold2_interface_dSASA_all` — SASA
- `self_complex_esmfold2_interface_sc` — SASA
- `self_complex_esmfold2_interface_sc_all` — SASA
- `self_complex_esmfold2_target_dSASA` — SASA
- `self_complex_esmfold2_target_dSASA_all` — SASA
- `self_complex_esmfold2_target_interface_nres` — SASA
- `self_complex_esmfold2_target_interface_nres_all` — SASA
- `self_complex_esmfold2_target_interface_ss_counts` — SS
- `self_complex_esmfold2_target_interface_ss_counts_all` — SS
- `self_complex_esmfold2_target_interface_ss_total` — SS
- `self_complex_esmfold2_target_interface_ss_total_all` — SS
- `self_complex_esmfold2_target_ss_counts` — SS
- `self_complex_esmfold2_target_ss_counts_all` — SS
- `self_complex_esmfold2_target_ss_total` — SS
- `self_complex_esmfold2_target_ss_total_all` — SS

## Unchanged (80)

- `L` — unchanged (config / identity)
- `_res_co_scRMSD_all_atom_esmfold2` — unchanged (run-level aggregate, documented exception)
- `_res_co_scRMSD_all_atom_esmfold2_all` — unchanged (run-level aggregate, documented exception)
- `_res_co_scRMSD_ca_esmfold2` — unchanged (run-level aggregate, documented exception)
- `_res_co_scRMSD_ca_esmfold2_all` — unchanged (run-level aggregate, documented exception)
- `_res_mpnn_best_sequence` — unchanged (run-level aggregate, documented exception)
- `_res_mpnn_sequences` — unchanged (run-level aggregate, documented exception)
- `_res_scRMSD_ca_esmfold2` — unchanged (run-level aggregate, documented exception)
- `_res_scRMSD_ca_esmfold2_all` — unchanged (run-level aggregate, documented exception)
- `_res_scRMSD_single_ca_esmfold2` — unchanged (run-level aggregate, documented exception)
- `autoencoder_ckpt_path` — unchanged (config / identity)
- `base_config_name` — unchanged (config / identity)
- `binder_sequence` — unchanged (config / identity)
- `ckpt_name` — unchanged (config / identity)
- `ckpt_path` — unchanged (config / identity)
- `complex_pdb_path` — unchanged (config / identity)
- `generation_args_ag_ckpt_path` — unchanged (config / identity)
- `generation_args_ag_ratio` — unchanged (config / identity)
- `generation_args_fold_cond` — unchanged (config / identity)
- `generation_args_guidance_w` — unchanged (config / identity)
- `generation_args_nsteps` — unchanged (config / identity)
- `generation_args_save_trajectory_every` — unchanged (config / identity)
- `generation_args_self_cond` — unchanged (config / identity)
- `generation_model_bb_ca_gt_clamp_val` — unchanged (config / identity)
- `generation_model_bb_ca_gt_mode` — unchanged (config / identity)
- `generation_model_bb_ca_gt_p` — unchanged (config / identity)
- `generation_model_bb_ca_schedule_mode` — unchanged (config / identity)
- `generation_model_bb_ca_schedule_p` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_center_every_step` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_sampling_mode` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_sc_scale_noise` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_sc_scale_score` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_t_lim_ode` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_t_lim_ode_below` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_tsr_k` — unchanged (config / identity)
- `generation_model_bb_ca_simulation_step_params_tsr_sigma` — unchanged (config / identity)
- `generation_model_local_latents_gt_clamp_val` — unchanged (config / identity)
- `generation_model_local_latents_gt_mode` — unchanged (config / identity)
- `generation_model_local_latents_gt_p` — unchanged (config / identity)
- `generation_model_local_latents_schedule_mode` — unchanged (config / identity)
- `generation_model_local_latents_schedule_p` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_center_every_step` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_sampling_mode` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_sc_scale_noise` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_sc_scale_score` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_t_lim_ode` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_t_lim_ode_below` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_tsr_k` — unchanged (config / identity)
- `generation_model_local_latents_simulation_step_params_tsr_sigma` — unchanged (config / identity)
- `generation_n_recycle` — unchanged (config / identity)
- `id_gen` — unchanged (config / identity)
- `job_id` — unchanged (config / identity)
- `mpnn_aa_counts` — unchanged (sequence-level, no structure)
- `mpnn_esm_log_likelihood` — unchanged (sequence-level, no structure)
- `mpnn_esm_log_likelihood_all` — unchanged (sequence-level, no structure)
- `mpnn_esm_pseudo_perplexity` — unchanged (sequence-level, no structure)
- `mpnn_esm_pseudo_perplexity_all` — unchanged (sequence-level, no structure)
- `mpnn_pass` — unchanged (sequence-level, no structure)
- `mpnn_pass_all` — unchanged (sequence-level, no structure)
- `mpnn_redesign_score` — unchanged (sequence-level, no structure)
- `mpnn_redesign_score_all` — unchanged (sequence-level, no structure)
- `mpnn_sequence` — unchanged (sequence-level, no structure)
- `mpnn_sequence_all` — unchanged (sequence-level, no structure)
- `pdb_path` — unchanged (config / identity)
- `pooled_run` — unchanged (config / identity)
- `redesign_conditioning` — unchanged (config / identity)
- `redesign_model` — unchanged (config / identity)
- `redesign_score_kind` — unchanged (config / identity)
- `run_name` — unchanged (config / identity)
- `self_aa_counts` — unchanged (sequence-level, no structure)
- `self_esm_log_likelihood` — unchanged (sequence-level, no structure)
- `self_esm_log_likelihood_all` — unchanged (sequence-level, no structure)
- `self_esm_pseudo_perplexity` — unchanged (sequence-level, no structure)
- `self_esm_pseudo_perplexity_all` — unchanged (sequence-level, no structure)
- `self_pass` — unchanged (sequence-level, no structure)
- `self_pass_all` — unchanged (sequence-level, no structure)
- `self_sequence` — unchanged (sequence-level, no structure)
- `self_sequence_all` — unchanged (sequence-level, no structure)
- `target_sequence` — unchanged (config / identity)
- `task_name` — unchanged (config / identity)
