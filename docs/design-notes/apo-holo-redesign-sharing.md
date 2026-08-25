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

## What the target actually is, and what `_updated.pdb` is for

Recorded because reading the code alone gave the wrong answer, twice.

The paper (ICLR 2026, `openreview.net/forum?id=qmCpJtFZra`) is explicit that the
target is **clean conditioning, never generated**: the denoiser processes "noisy
binder embeddings and clean target embeddings", and "the output of our flow
matching model is solely the binder" -- its generative output is "the velocity
field for the binder", by the same construction La-Proteina uses for motif
scaffolding, where the output is "restricted to the generated scaffold, not the
motif itself". Training also "center[s] the complex so that the target lies at
the origin".

So the target chain of a generated complex is the input target under a rigid
transform. Measured on a CBLN1/5KC5 design against its reference target: 136 CAs
either side, **res_name identity 1.000**, CA RMSD 44.8 A as-is and **0.000 A
superposed**. Bit-identical geometry in a moved frame, and identities already
correct.

Which means `replace_seq_in_generated_pdb` -- and therefore `_updated.pdb` --
does two things, one of which is currently a no-op:

- restores target residue identities: **a no-op today**, they already match;
- reduces to CA only: the real difference from the design PDB.

And `run_proteinmpnn` passes `--ca_only`, under which ProteinMPNN's `parse_PDB`
reads CA atoms and CA coordinates only. So the reduction the file performs is one
ProteinMPNN performs anyway.

The tempting inference from the code alone -- that the model emits target residue
identities and the replacement repairs them -- is wrong, and it matters: it would
make the design PDB an unsafe ProteinMPNN input and the file choice
load-bearing. It is neither. Conditioning designability on the design PDB
conditions it on the true target, with the true target sequence.

## The tracks now use the same inverse folder

*Resolved.* `metric.inverse_folding_model` governs both tracks: designability
routes through the same `inverse_fold` dispatcher the complex track uses, so the
two produce opinions about the same sequences rather than two models' views of
one backbone. `redesign_model` records which, on both CSVs, because designability
numbers are not comparable across runs that set it differently.

One residual asymmetry: the complex track forces `ligand_mpnn` for ligand targets
regardless of config, and `compute_monomer_metrics` cannot tell whether a target
is a ligand. Both shipped ligand configs already set `ligand_mpnn`, so they agree
in practice, and `redesign_model` on the two CSVs makes a disagreement visible in
the data rather than only in a reader's assumptions.

The ligand case was also quietly wrong before this. Designability reads the
complex, and for a ligand target that complex contains the ligand -- which
CA-ProteinMPNN parses by finding no CA atoms in the ligand chain and dropping it.
Routing through the dispatcher means those runs get LigandMPNN, which is the
model that can actually see what the binder has to bind.

What follows is what the situation was before that change.

`metric.inverse_folding_model` selected the model for the **complex track only**.
`monomer_eval` imported `run_proteinmpnn` directly, so designability was hardwired
to CA-ProteinMPNN whatever that key said. And
`configs/pipeline/binder/binder_evaluate.yaml` sets `soluble_mpnn`, so in the
binder pipeline as it actually runs:

| | model | weights | input |
|---|---|---|---|
| complex track | SolubleMPNN via LigandMPNN's `run.py` | `solublempnn_v_48_020.pt` | the design PDB, all atoms |
| designability | CA-ProteinMPNN, vanilla | `ca_model_weights/v_48_020.pt` | the design PDB, CA read out |

Different models, different sequences. The measurement below -- that the two
tracks produce identical redesigns -- was made by driving `run_proteinmpnn`
directly, so it establishes that for `inverse_folding_model: protein_mpnn` and
says nothing about the configuration the binder pipeline uses.

`_updated.pdb` is only ever ProteinMPNN's input. SolubleMPNN and LigandMPNN read
the plain design PDB, which is consistent with their being all-atom models.

Two consequences for the work ahead:

- **Sharing had a precondition nobody stated:** the two tracks must use the same
  inverse folder, or sharing hands one track sequences the other drew from a
  different distribution. Met now, by the change above.
- **`mpnn_seed` does not include the model**, which is correct as long as each
  model seeds its own draw, and wrong the moment a shared draw is keyed on it.

There is also an opportunity here, given the ranking criterion is chosen for
expressibility: SolubleMPNN is the model trained on soluble proteins, so its
likelihood is a more direct expressibility signal than vanilla ProteinMPNN's.
Which model *generates* the redesigns and which model's score *ranks* them are
separate knobs, and scoring existing redesigns under a second model is far
cheaper than changing generation.

## What sharing turned out to be

Measured, on the CBLN1/5KC5 design, with `verify_mpnn_prefix.py --target-pdb`:
the two tracks' ProteinMPNN inputs -- `_updated.pdb` for the complex track, the
design PDB for designability -- **produce identical sequences under one seed**,
6/6. Under `--ca_only` ProteinMPNN reads CA atoms and CA coordinates from either,
and the section above says why the two files carry the same ones.

Put together with the seeds already agreeing by construction and the prefix
property holding, the two tracks are *already* generating the same redesign set.
Sharing is therefore not plumbing to build but duplicated compute to delete: one
ProteinMPNN run per design, sliced to `num_redesign_seqs` for the complex track
and used whole for designability.

That also means nothing about correctness rides on it. The apo/holo criterion
needs one sequence carrying both verdicts, and it already can -- the sets
coincide. Sharing only removes a second run.

## Which `num_redesign_seqs` of the redesigns, and why it is not arbitrary

Designability uses every redesign. The complex and apo tracks use
`num_redesign_seqs` of them, and *which* ones has not been decided. Today it
falls out of the prefix property -- the first N in generation order, which is to
say whichever ones the RNG happened to emit first. That is a default, not a
choice.

It should become a choice, because these sequences are not metric inputs. They
are the candidate binders: what gets folded with the target, gated, clustered
and shipped. Designability asks a question about the backbone and can average
over anything; the complex track is selecting things to make. Picking the first N
spends the entire holo budget on an arbitrary slice of what was available.

**The constraint that narrows the search:** the criterion must be computable
*without folding*. The subset exists to bound how much folding happens, so
anything that needs a fold to evaluate -- i_pAE, complex pLDDT, scRMSD, every
metric that actually defines success here -- is circular. The ranking has to be
done on sequence and structure alone.

What survives that, roughly in order of how cheaply it is already available:

- **The inverse folder's own score.** Already parsed into `sequences_dict`,
  already persisted in `binder_eval_cache.json`, and until now read by nothing.
  Free. **The first criterion, explicitly provisional** -- see below.

  It is not one quantity. ProteinMPNN reports `score=`, an NLL averaged over
  designed positions (`protein_mpnn_utils._scores`), lower better.
  LigandMPNN and SolubleMPNN report `overall_confidence=`, `np.exp(-loss)` in
  (0, 1], higher better. Both land under the key `"score"`. They are
  monotonically related, so they rank identically up to direction -- which is
  precisely why a ranking must read `REDESIGN_SCORE_KIND` rather than assume:
  hardcoding one convention ranks correctly for one model and selects the *worst*
  sequences for the other, and the values themselves cannot tell you which
  happened. `redesign_score_kind` is emitted per run for that reason.

  It also means the binder pipeline, which configures `soluble_mpnn`, is already
  scoring with the soluble model -- so the expressibility-aware score is in hand
  without adding anything.
- **`seqid`**, recovery against the input sequence, parsed and unused likewise.
  For a co-designed binder this measures agreement with the model's own sequence,
  which is a different thing from quality.
- **ESM pseudo-perplexity**, already implemented (`esm_eval.py`, gated behind
  `compute_esm_metrics`) and already write-only. Costs a forward pass per
  residue, so not free, but far below a fold.
- Composition heuristics -- hydrophobic surface fraction and the like, some of
  which the interface metrics already compute for the generated structure.

**What the MPNN score is for, and what it is not.** It is not a binding proxy
and should not be expected to behave as one. Empirically it tracks
*expressibility* -- whether the protein can actually be made -- which is a
separate attribute of a good binder from whether it binds, and one that nothing
else in this pipeline measures at all. The holo gate selects for binding; ranking
by MPNN score adds a second axis rather than sharpening the first.

That changes what there is to measure, and an earlier version of this note got it
backwards. The question is **not** "does the MPNN score predict the holo gate" --
if the score is orthogonal to binding then a null result there is the expected
outcome, not a disqualification, and treating it as one would throw away a
criterion that was working as intended. The question is the harm check: **does
ranking by MPNN score cost gate pass rate** relative to the arbitrary first-N
subset? Neutral is a pass. Only a real reduction argues against it.

The benefit itself is not measurable here. Expressibility is a wet-lab outcome;
no arrangement of this pipeline will confirm it. It is carried on domain
knowledge, deliberately, and that is recorded here so a later in-silico null
result is read as consistent rather than as evidence against.

`{seq_type}_redesign_score_all` is emitted beside `{seq_type}_pass_all`, with
`redesign_score_kind` saying which way it points, so the harm check can be run on
an existing smoke test rather than argued.

**And it is a starting point, not a decision.** Binding is one attribute of a
good binder; expressibility is another; there are more, and ranking on the
non-binding ones is the point rather than a compromise for lacking a binding
proxy. The criterion is expected to change as the campaigns say more about which
axes matter. That expectation is a design constraint, not a caveat, and three
things follow from it:

- **The ranking is a configured, replaceable component, not a sort.** One named
  selection function, one config key, `none` available so the arbitrary
  first-N subset stays reachable as the control the harm check compares against.
  A hardcoded sort would have to be found and surgically replaced.
- **The criterion in force is provenance, and must be recorded per run.** This
  is exactly the trap `redesign_conditioning` exists for: two campaigns ranked by
  different criteria produce different candidate binders under identical column
  names, and concatenating them would average across a distinction nobody can
  see. The column has to be groupby-eligible for the same reason -- see the
  naming constraints in that section, which are not obvious and cost a rename to
  discover.
- **Candidate scores are recorded whether or not they rank.** Emitting only the
  active criterion makes every change of criterion a re-run. Emitting the others
  alongside makes "what would ranking by X have chosen?" answerable from data
  already on disk, which is how a criterion gets chosen on evidence instead of
  on argument. `esm_pseudo_perplexity` already lands this way and is already
  write-only; the MPNN score now joins it.

**One coupling to keep in view.** Selection and the prefix-property shortcut are
mutually exclusive. Picking the best 2 of 8 requires generating 8, so the complex
track can no longer run a shorter ProteinMPNN and rely on getting a prefix. The
moment a ranking lands, deleting the duplicate ProteinMPNN run stops being an
optimisation and becomes the implementation: one run of `designability_num_seq`,
ranked, sliced. Which is why, when the subset is first materialised in code, it
must pass through one named function rather than a slice written at each use.

## Still open

- **Does ProteinMPNN with a fixed seed return the same first N sequences when
  asked for more than N?** Step 1 does not depend on it -- each track still runs
  its own sampling -- but the sharing step does. Measured by
  `script_utils/bioinformatic/verify_mpnn_prefix.py`, which checks determinism,
  non-degeneracy and that ProteinMPNN actually used the seed passed to it before
  asking the prefix question, under both conditionings. Exit 0 means the shared
  subset can be reproduced by a shorter seeded run; exit 2 means it has to be
  taken from one run of `designability_num_seq`; exit 1 means the question was
  not answered.

  **Answered: it holds.** Measured on a CBLN1/5KC5 smoke-test design under both
  conditionings, at 2-of-8 (the `num_redesign_seqs` that
  `configs/pipeline/binder/binder_evaluate.yaml` actually uses) and again at
  6-of-8. Determinism, non-degeneracy and seed plumbing all passed --
  ProteinMPNN reported back the seed it was handed, so `--seed` is reaching it
  rather than being ignored, and the long run was 8/8 distinct, so the agreement
  is not an artifact of a draw that repeats itself. Six sequences matching
  exactly out of eight distinct ones is not something a coincidence produces.

  So the sharing step can reproduce the shared subset with a shorter seeded run.
  It does not have to thread one longer run's output between the two tracks, and
  designability does not have to wait on the complex track: each asks for the
  count it wants and they agree by construction.

  The remaining limit is not fixable by re-running: this is one design, one
  target, one binder length. Nothing here would catch a failure that depends on
  chain count or length. Established for this shape; re-check if a multi-chain
  target ever behaves oddly.
- **Ranking the redesigns.** MPNN score is the provisional first criterion; the
  implementation is not written, and it pulls in the duplicate-run deletion,
  since picking the best N requires generating all of them in the track that
  does the picking. The harm check -- that ranking does not cost gate pass rate
  against the unranked control -- has not been run. Which criterion it settles
  on is open indefinitely, by design.
- Whether the apo gate, once calibrated, applies to every sequence type or
  only to the types a campaign intends to ship.

## Not reused: `binder_bound_unbound_RMSD`

`binder_metrics.py` documents this key in the rmsd_stats format and nothing
computes or reads it. It is left alone: upstream (NVIDIA) may implement it and a
divergent meaning would conflict, and it is a single `float`, so it cannot carry
an apo and a holo value in any case.
