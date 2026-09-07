# Which code decides what an interface is, and what buries how much

**Status:** decided, not yet implemented.

## Where this started

Three interface definitions ship today, and two columns that both say
`interface` count different residues:

| | atoms | cutoff | multi-chain target | returns |
|---|---|---|---|---|
| `get_interface_residues` (`metric_utils.py`) | CA only | 8.0 Å | yes | 0-based sequence index |
| `get_interface_residues_atomistic` | all | 6.0 Å | yes | 0-based sequence index |
| `hotspot_residues` (`biopython_utils.py`) | all | 4.0 Å | **no** — `structure[0][target_chain]` | PDB residue number |

`{seq}_aa_interface_counts` uses the first; `interface_nres` and
`interface_hydrophobicity` use the third. Measured over 12 CBLN1 complexes they
select 15.9 and 17.9 binder residues respectively, mean Jaccard **0.664**, and
the 4 Å set was never a subset of the 8 Å set (0/12). They are not a coarse and
a fine version of one definition; roughly a third of the union is disputed.

The 0-based index form is also a latent bug: `residue_idx = res_id - offset`
assumes contiguous numbering from `res_id.min()` and then indexes a *sequence
string*, so any gap in numbering silently shifts every count.

## Decision

Use [`protein-interface`](https://github.com/aarteixeira/protein-interface)
(MIT, PyPI `protein-interface==0.1.3`) for **interface residues** and **shape
complementarity**, and keep **freesasa/ProtOr for dSASA**.

Its `strict` mode is burial-aware rather than purely geometric:

    MODES = {'strict':  {'dsasa_threshold': 3.0, 'contact_cutoff': 5.0, 'combine': 'or'},
             'lenient': {'dsasa_threshold': 0.0, 'contact_cutoff': 7.0, 'combine': 'or'}}

A residue is interface if it buries ≥3 Å² **or** has an atom within 5 Å.
`interface_residues(a, b)` returns *both* sides' sets, which retires the
swap-the-arguments trick, and the target is `chains_b`, so multi-chain targets
work by construction. Over 8 complexes our 4 Å set was a strict **subset** of
strict mode every time — it adds 2–7 residues that bury area without a 4 Å
contact. `interface_cutoff` feeds `contact_cutoff`.

Its `sc` is the same sc-rs the repo already ships: the binary in
`result_analysis/sc` contains `src/sc/surface_generator.rs`,
`src/sc/sc_calculator.rs` and `struct RadiusRecord`. Adopting the batched
version is the same algorithm on the same radii, without a subprocess, an
`SC_EXEC` path to resolve, or a placeholder to refuse.

Install is clean: abi3 `manylinux_2_17_x86_64` wheels, and it declares only
`numpy>=1.24` (raises our floor from 1.23.5) and `biopython>=1.83`. `scipy` and
`pandas` appear only in the `[residues]` extra, so the `scipy==1.12.0` pin is
untouched; `[openmm]` is skipped.

## Why dSASA stays on freesasa

`protein-interface`'s SASA is 5.7× faster (10.5 ms vs 52.3 ms for the three
calls dSASA needs) and agrees to r=0.99939 — but reads **+3.64%** high, because
`sasa.rs` reuses "sc-rs's embedded atomic radii (MS-style, from CCP4 sc Fortran
source)" rather than ProtOr. Those radii are right for SC, whose published
thresholds are calibrated on them, and they are not what we pinned dSASA to.

The radii cannot be overridden from Python. `sasa.rs` does expose
`compute_with_radii(..., table: &[AtomRadius], ...)`, but the PyO3 binding
passes the embedded table and offers no parameter.

The `ATOMIC_RADII` / `ATOMIC_RADII_PATH` environment variables that
`vendor/sc-rs/README.md` documents are real but apply **only to SC**, verified
by behaviour:

    ATOMIC_RADII=<one-entry table>  ->  compute_sc raises "No radius for GLY:N"
    ATOMIC_RADII=<one-entry table>  ->  delta_sasa = 1973.627, unchanged

Had SASA honoured it, nearly every atom would have fallen to the
`unwrap_or(0.0)` default and dSASA would have collapsed toward zero. It did not
move. The loader lives in `src/sc/atomic_radii.rs`, inside the sc module.

So the override offers custom radii for the metric that must not change them,
and none for the metric we want on ProtOr. Keeping freesasa for dSASA costs
~43 min per campaign (~42 ms × ~61k calls) against a 21-hour evaluate stage.

## Deferred: an upstream PR

Expose a radii table on the SASA bindings so `delta_sasa`, `compute_sasa`,
`per_residue_dsasa` and `unknown_sasa_radius_atoms` can reach the
`compute_with_radii` that already exists. That would let dSASA move to
`protein-interface` on ProtOr radii and pick up the 5.7×. Two defects to report
alongside it:

- `ATOMIC_RADII` is documented for the package but has no effect on SASA.
- A non-existent path in `ATOMIC_RADII` **silently falls back** to the embedded
  table — `ATOMIC_RADII=/nonexistent.json` left `sc` at 0.441559, unchanged and
  unwarned.

## Guards this imposes on us

- `unknown_sasa_radius_atoms()` must be called and must raise. `sasa.rs` gives
  an unrecognised atom radius `0.0` and returns early, so it contributes
  nothing and the total is quietly too small — the shape of fabrication removed
  in `74c31e3`, `b95b135` and `2867e12`.
- Pin `protein-interface==0.1.3` and put the version in the metric fingerprint:
  the radii are compiled in and cannot be audited from metadata.
- `analyze()` defaults `n_points=92` while `delta_sasa` defaults to `960`.
  Always pass it explicitly.
- `gemmi` is **not** worth adding. The README calls it a runtime dependency for
  fast parsing; the shipped wheel contains zero references to it, imports
  without it, and `load_atoms` takes 9.2 ms/complex with it and 9.2 ms without.
