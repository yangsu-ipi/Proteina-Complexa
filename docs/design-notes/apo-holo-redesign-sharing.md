# Where binder redesign sequences come from, and what folds them

**Status:** decided, not yet implemented. Records why the implementation looks the
way it will, and what it displaces.

## The problem

A binder design is judged in two conditions:

- **holo** — the binder folded with its target. Gated today by
  `{seq_type}_complex_i_pAE`, `{seq_type}_complex_pLDDT` and
  `{seq_type}_binder_scRMSD_ca` (`binder_analysis_utils.py`,
  `DEFAULT_PROTEIN_BINDER_THRESHOLDS`).
- **apo** — the binder folded alone. Computed today as *codesignability*, on the
  design's own sequence only, and reported but never gated.

BoltzGen reports that requiring a binder to fold as designed *both* with and
without its target improves experimental success. Expressing that criterion needs
one sequence to carry both an apo and a holo verdict. Today no sequence does:

- the complex track redesigns with ProteinMPNN conditioned on the **complex**
  (`binder_metrics.py`, `all_chains=gen_target_chain + [binder_chain]`,
  `omit_AAs=["C"]`, `num_redesign_seqs` of them) and folds those holo;
- the monomer track redesigns conditioned on the **binder alone**
  (`monomer_eval.py`, `all_chains=[chain_to_design]`, `designability_num_seq` of
  them) and folds those apo, for designability;
- both run ProteinMPNN unseeded, so the two sets differ every run and cannot be
  joined per sequence.

The two sets are not the same generation with different seeds. They condition on
different structures, so seeding alone would never make them coincide.

## Options considered

**A — generate from the complex, share into designability.** The shared
`num_redesign_seqs` are interface-aware, which is what gets shipped and what the
holo gate already judges. Designability becomes partly a statement about
interface-conditioned sequences, for `num_redesign_seqs` of its
`designability_num_seq`, with the remainder generated some other way -- leaving
one metric averaging over two conditionings, and, since the complex track sets
`omit_AAs=["C"]` and the monomer track does not, over two alphabets.

**B — generate from the binder alone, share into the complex track.**
Designability keeps its exact current meaning. But the sequences refolded in
complex, gated, and shipped would have been designed without ever seeing the
target. For binder design that is the wrong trade: the interface is the point.

**C — share nothing; extend codesignability to the complex track's redesigns.**
Gives the apo/holo criterion on exactly the sequences that get shipped, leaves
designability untouched as a pure backbone property, and costs a second
ProteinMPNN run per design. Conservative; declines the question rather than
answering it.

**D — generate every redesign from the complex (chosen).** One ProteinMPNN run
per design, conditioned on the complex, producing `designability_num_seq`
sequences. All of them feed designability. A `num_redesign_seqs` subset is also
refolded holo by the complex track and apo for codesignability.

## Why D

B is wrong for the reason above, and the same reasoning applied to designability
is what makes D better than A or C rather than merely cheaper.

The question worth asking of a binder backbone is not "is this backbone
redesignable in general" but "is this backbone redesignable *for binding this
target*". Conditioning every redesign on the complex asks that question, and then
folding the redesigns both with and without the target asks whether the answer
survives in both conditions. That is a stronger claim than A, which mixes
conditionings within one metric, and stronger than C, which keeps a
target-agnostic designability number that is not the property being selected for.

The cost is a real change of meaning: designability stops being a target-agnostic
backbone property. Numbers are not comparable with any produced before this
change. That is accepted deliberately -- the target-agnostic version was
measuring something adjacent to what binder design cares about.

## Consequences

- **One ProteinMPNN run per design**, seeded, conditioned on the complex,
  `designability_num_seq` sequences, `omit_AAs=["C"]` throughout. One alphabet,
  one conditioning.
- **The apo folds are computed once and used twice.** Designability folds all
  `designability_num_seq` apo; extended codesignability needs the
  `num_redesign_seqs` subset apo. Same sequence, same condition, same seed, so
  they are the same fold -- shared through the refold cache, not recomputed. This
  is why D costs less than it appears.
- **`self` still needs an apo fold of its own**, since it is holo-gated like any
  other sequence type and is not a member of the redesign set. That fold is
  today's codesignability computation, unchanged.
- **Designability keeps its `min()` over all `designability_num_seq`** and stays
  a design-level number. Extended codesignability keeps per-sequence values, so
  they line up with the per-sequence holo values and the joint criterion is
  expressible per sequence.
- **Seeding ProteinMPNN is a prerequisite**, not a nicety: without it the shared
  subset is not reproducible and the apo and holo tables cannot be joined across
  a resume.

## Naming, against the grid

Two independent axes have been conflated, because until now only one combination
per axis existed.

**Axis 1 — whose sequence.** `co` in `_res_co_*` means *co-designed*, the field's
term for a model that emits sequence and structure jointly. The repo says so:
"Co-sequence-recovery measures how well the model recovers the original sequence
when co-designing structures" (`analysis.py`), and the logs read "Computing
co-designability". Proteina-Complexa is such a model, which is why `self` exists
at all -- it is read straight off the generated PDB. So `co` names the *source* of
a sequence, and ProteinMPNN redesigns are precisely what it excludes: an external
inverse-folder is the opposite of co-design. Extending a `co_` name to cover them
would destroy the one distinction the prefix exists to draw.

**Axis 2 — which condition the fold happens in.** holo, with the target present;
apo, the binder alone.

The grid, with what fills each cell today:

|                        | holo (with target)                    | apo (alone)                       |
|------------------------|---------------------------------------|-----------------------------------|
| co-designed (`self`)   | `self_complex_*`, `self_binder_scRMSD_*` | `_res_co_scRMSD_{mode}_{model}` |
| redesign (`mpnn`)      | `mpnn_complex_*`, `mpnn_binder_scRMSD_*` | **empty -- the gap**            |
| redesign (`mpnn_fixed`)| `mpnn_fixed_complex_*`, ...              | **empty -- the gap**            |

Designability sits outside the grid on purpose: it is a `min()` over
`designability_num_seq` sequences, a design-level summary rather than a
per-sequence value. It stays that way.

**Proposed name for the empty cells:** `{seq_type}_apo_scRMSD_{mode}_{model}`,
naming the *condition* rather than extending a name that means something else. It
reads uniformly across `self`, `mpnn` and `mpnn_fixed`, and sits beside the
existing holo per-type `{seq_type}_binder_scRMSD_{mode}`.

Two asymmetries are deliberate. The holo column carries no `{model}` because
`binder_folding_method` is a single value, while the apo column does because
`monomer_folding_models` is a list. And the holo column says `binder` where the
apo one says `apo`; renaming the holo column to match would touch a gated,
upstream-owned name for cosmetic gain, so it is left alone and the asymmetry is
recorded here instead.

## The three open questions, answered

**1. What are the new columns called?** `{seq_type}_apo_scRMSD_{mode}_{model}`,
per the grid above. Not a `co_` name.

**2. Does the apo gate start at 2.0 A?** Yes, as a starting point, but not in the
same change that introduces the columns. Emit ungated first, read the smoke-test
distribution, then add the threshold. scRMSD is geometric so the 2.0 A monomer
convention transfers in principle, but no run has produced the distribution, and
adopting a number before seeing it makes "the gate works" indistinguishable from
"the gate is mis-calibrated".

**3. Does `_res_co_scRMSD_*` survive?** Yes -- and not merely for continuity.
It is not a binder metric: `motif_eval_utils.py` emits it for motif scaffolding,
where there is no target and therefore no apo/holo distinction at all. Retiring
it would break a pipeline that has nothing to do with this work. For binders it
becomes numerically the `self` row of the grid -- same sequence, same condition,
same seed, therefore the same fold, shared through the refold cache rather than
computed twice. It keeps its name, and the equivalence is documented rather than
enforced by aliasing.

## A fourth question, not previously listed

Designability keeps the name `_res_scRMSD_{mode}_{model}` while its meaning
changes, since its sequences move from binder-only to complex conditioning. Old
and new numbers would then share a column name -- the same trap as a citation
that still points somewhere valid after the claim it supported stopped being
true.

Recommendation: emit a provenance column, `_res_mpnn_conditioning` with value
`complex` or `binder_only`, so any CSV states which meaning its designability
numbers carry. One column, written once per design, and it makes the two eras
distinguishable without renaming a gated metric.

## Still open

- Whether `_res_mpnn_conditioning` is worth its column, or the conditioning
  change is better recorded only here.
- Whether the apo gate, once calibrated, applies to every sequence type or
  only to the types a campaign intends to ship.

## Not reused: `binder_bound_unbound_RMSD`

`binder_metrics.py` documents this key in the rmsd_stats format and nothing
computes or reads it. It is left alone: upstream (NVIDIA) may implement it and a
divergent meaning would conflict, and it is a single `float`, so it cannot carry
an apo and a holo value in any case.
