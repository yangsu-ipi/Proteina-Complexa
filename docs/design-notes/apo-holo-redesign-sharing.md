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

Recommendation, now implemented: emit a provenance column so any CSV states
which meaning its designability numbers carry. One column, written once per
design, and it makes the two eras distinguishable without renaming a gated
metric.

**It is called `redesign_conditioning`, not `_res_mpnn_conditioning`.** The
proposed name was the one name that could not do the job it was invented for.
`get_groupby_columns` excludes any column containing `_res_` -- and any column
starting with `mpnn_` -- from grouping, so `_res_mpnn_conditioning` would have
been carried along as a passive string. Concatenating a pre-change and a
post-change run would then have averaged two different metrics into one row,
which is precisely the failure the column exists to prevent. The name that
survives the exclusion list is groupby-eligible, so the two eras split into
separate rows instead of merging.

The value is derived rather than asserted: `designability_mpnn_chains()` is the
single definition of what ProteinMPNN sees, the redesigns are generated from it,
and the column is computed from it. Widening it to the target's context changes
both in the same edit. The complex track has the matching
`complex_mpnn_chains()`, and reports its conditioning into
`success_criteria_binder_eval_{job_id}.json` rather than a column, since there it
is constant across every row.

## Also implemented ahead of the apo work

Neither is required by Option D, but both are prerequisites for expressing a
joint apo/holo criterion per sequence, and both stand on their own.

- **`{seq_type}_pass` / `{seq_type}_pass_all`.** Analysis reports a design as
  passing when *any* of its sequences does, so the verdict on an individual
  sequence existed nowhere on disk -- every consumer re-derived it by
  `ast.literal_eval`-ing the `_all` columns and re-applying thresholds by hand,
  and any two could disagree. The vector is now emitted during evaluation from
  `redesign_pass_vector()`, the same primitive `check_sample_has_passing_redesign`
  and `count_passing_redesigns` are now reductions of. When the apo gate lands it
  becomes a third condition in one place rather than in every consumer.
- **`success_criteria_binder_eval_{job_id}.json`.** Because those verdicts are
  baked into the rows at evaluation time, a CSV re-analysed under different
  thresholds would carry pass columns contradicting the pass rates beside them.
  Analysis now compares its thresholds against this file and says so.

## Implementation, step 1: seeded redesigns from the complex

Landed. What changed:

- **Both ProteinMPNN call sites are seeded**, from `mpnn_seed()` in
  `metrics/seeding.py`. The seed is derived from the design's own name, the
  chains conditioning it, the chains being redesigned, the alphabet and the
  temperature -- deliberately *not* from how many sequences are requested, since
  the two tracks want different counts of what is meant to be one draw. The
  design name is the design's stem, not the stem of whichever intermediate file
  a caller reads: `_updated.pdb` and `_binder.pdb` are views of one design and
  must not seed differently. `mpnn_fixed` passes `variant="fixed"` so holding
  positions fixed is a different draw rather than a constrained version of the
  same one.
- **Designability redesigns now condition on the complex.** ProteinMPNN reads
  the complex and redesigns the binder chain only; everything downstream --
  folding, RMSD -- stays on the binder alone. That asymmetry is the point: the
  redesign is judged apo, but it was made knowing what it has to bind.
- **One alphabet.** The monomer track omitted `X`, a letter ProteinMPNN does not
  emit, which is to say it omitted nothing. Both tracks now omit `C`, matching
  what the complex track always did.
- **Sequence recovery follows.** `_res_co_seq_rec` reuses the designability
  redesigns when they exist and generated its own when they did not; the
  fallback now uses the same conditioning, so the number does not depend on
  whether designability happened to be enabled.

The cache work this forced is the part worth recording. `monomer_fold_cache`
stores its sequences rather than keying on them -- they are an output, so keying
on them would mean running ProteinMPNN to find out whether the ProteinMPNN run
could be skipped. Its fingerprint therefore has to cover everything that
*produces* them, and did not: a cache written when designability redesigned the
binder alone would have been served for a request that redesigns it in the
target's context. Same design, same folding model, entirely different sequences,
and the served numbers would have been the old metric under the new name. The
conditioning and the seed are now in the key.

The conditioning fields are added only when ProteinMPNN is involved, which keeps
the codesignability key byte-identical to what it was. Codesignability reads its
sequence off the PDB and is untouched by any of this; invalidating its folds --
potentially a diffusion sampler over every design -- to record a fact that does
not apply to it would cost real compute for no information. `binder_eval_cache`
does invalidate, on purpose: entries written before seeding hold unseeded draws,
which are valid sequences but not the ones a fresh run produces, and reusing
them would quietly reintroduce the un-joinable state this work exists to remove.

**Numbers move.** Designability and sequence recovery both change -- new
conditioning, new alphabet, new seeds. `redesign_conditioning` now reports
`complex` for binder runs, which is what makes the two eras distinguishable
rather than merely different.

## Still open

- **Does ProteinMPNN with a fixed seed return the same first N sequences when
  asked for more than N?** Step 1 does not depend on it -- each track still runs
  its own sampling -- but the sharing step does, and it must be measured rather
  than assumed. If the prefix property does not hold, the shared subset has to be
  taken from one run of `designability_num_seq` rather than reproduced by a
  shorter run.
- Whether the apo gate, once calibrated, applies to every sequence type or
  only to the types a campaign intends to ship.

## Not reused: `binder_bound_unbound_RMSD`

`binder_metrics.py` documents this key in the rmsd_stats format and nothing
computes or reads it. It is left alone: upstream (NVIDIA) may implement it and a
divergent meaning would conflict, and it is a single `float`, so it cannot carry
an apo and a holo value in any case.
