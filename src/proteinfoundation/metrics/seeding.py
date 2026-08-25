"""Deterministic seeds for the sampling steps of evaluation.

Two stages of evaluation sample rather than compute: ESMFold2 is a diffusion
folder, and ProteinMPNN draws sequences at a temperature. Left unseeded, both
make a result depend on whether a run was resumed -- a cache hit returns the
first run's draws while a fresh run makes new ones -- and neither can be joined
to anything computed in a different pass.

Deriving each seed from the inputs of the thing being sampled makes it a pure
function of them, so a cached value and a recomputed one agree, and two call
sites asking for the same thing get the same answer. That last property is what
lets the apo and holo tracks share one set of redesigns; see
``docs/design-notes/apo-holo-redesign-sharing.md``.
"""

import hashlib

# Bump whenever deterministic_seed changes -- adding a component, reordering the
# parts, changing the hash. Every seed changes when it does, and therefore every
# structure, every redesign and every number, while nothing else in a cache
# fingerprint moves. It goes into the cache fingerprints so that a derivation
# change invalidates instead of silently serving samples drawn from the old seeds
# beside freshly drawn ones.
#
# Still 1: moving this function between modules did not change what it
# computes, and bumping for a move would invalidate caches for no reason.
SEED_DERIVATION_VERSION = 1


def deterministic_seed(*parts: str) -> int:
    """A stable seed derived from the inputs a sample depends on.

    ``_seed_context`` in the ESMFold2 fork saves and restores python/numpy/torch
    RNG state around the call, and ProteinMPNN runs in its own subprocess, so
    seeding here perturbs nothing upstream.

    Returns a value in the numpy seed range, which is narrower than torch's.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**32 - 1)


# Sampling settings shared by both ProteinMPNN call sites. The apo and holo
# tracks must draw from the same distribution for their redesigns to be the same
# sequences, so these are constants rather than per-track defaults.
#
# omit_AAs="C" follows the complex track: free cysteines in a de novo binder are
# a liability, and the monomer track's previous "X" omitted a letter ProteinMPNN
# never emits, which is to say it omitted nothing.
MPNN_OMIT_AAS = ["C"]
MPNN_SAMPLING_TEMP = 0.1


def mpnn_seed(
    design_name: str,
    context_chains: list[str],
    chains_to_design: list[str],
    variant: str = "",
) -> int:
    """The seed for one design's redesign set.

    Deliberately independent of how many sequences are requested. The complex
    track wants ``num_redesign_seqs`` and designability wants
    ``designability_num_seq``; deriving the seed from the count would make those
    two different draws, and the point of Option D is that they are one draw
    used twice.

    That leaves a property this does *not* establish: whether ProteinMPNN with a
    fixed seed returns the same first N sequences when asked for more than N.
    Nothing here depends on it yet -- each track still runs its own sampling --
    but the sharing step does, and must verify it rather than assume it.

    Args:
        design_name: Stable identifier for the design -- the stem of the design's
            own PDB, not of whatever intermediate file a tool happens to read.
            ``_updated.pdb`` and ``_binder.pdb`` are views of one design and must
            not seed differently.
        context_chains: Chains ProteinMPNN sees.
        chains_to_design: Chains it redesigns.
        variant: Distinguishes otherwise identical requests -- ``mpnn_fixed``
            holds positions fixed, so it is a different draw from ``mpnn`` and
            should not be seeded as the same one. Empty is the plain redesign,
            which is the draw designability shares.

    Returns:
        A seed in the numpy range.
    """
    seed = deterministic_seed(
        "proteinmpnn",
        str(SEED_DERIVATION_VERSION),
        design_name,
        ",".join(sorted(context_chains)),
        ",".join(sorted(chains_to_design)),
        "".join(sorted(MPNN_OMIT_AAS)),
        f"{MPNN_SAMPLING_TEMP:g}",
        variant,
    )
    # protein_mpnn_run.py reads its seed as `if args.seed:` and draws a random one
    # when that is falsy, so a derived 0 would silently make exactly one design in
    # 2**32 unreproducible -- and unreproducible in the way that is hardest to
    # notice, since every other design in the run would be fine. Not a derivation
    # change worth a version bump: every seed that was not 0 is unchanged, and the
    # one that was had a random seed anyway.
    return seed or 1
