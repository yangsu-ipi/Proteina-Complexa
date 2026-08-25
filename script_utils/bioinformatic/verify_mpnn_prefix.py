#!/usr/bin/env python3
"""Does ProteinMPNN with a fixed seed return the same first N sequences when asked for more?

The apo/holo work wants one redesign set used twice: designability folds all
``designability_num_seq`` sequences apo, and the complex track refolds
``num_redesign_seqs`` of them holo. If a seeded run of M sequences begins with
the same M sequences a seeded run of N>M produces, the shared subset can be
reproduced by the shorter run and the two tracks need no coordination. If it does
not, the subset has to be *taken* from a single run of the larger count.

Which it is decides how the sharing step is built, so it is measured rather than
assumed. See ``docs/design-notes/apo-holo-redesign-sharing.md``.

The prefix question is only meaningful if three things hold first, so they are
checked first and a failure in any of them aborts:

  1. determinism   -- same seed, same count, twice: identical.  Without this
                      nothing below means anything.
  2. non-degeneracy -- the sequences within a run are not all the same.  If they
                      were, a prefix would hold trivially and prove nothing.
  3. seed plumbing  -- ProteinMPNN records the seed it used in its FASTA header;
                      it must be the seed we passed.  A silently ignored --seed
                      would make check 1 pass on a machine that happens to be
                      deterministic and fail elsewhere.

Then the actual question, run under both conditionings the pipeline uses (the
binder alone, and the binder in its target's context), because the sharing step
needs it under complex conditioning specifically.

Exit codes are three-valued, because "the property does not hold" is an answer
rather than a malfunction:

  0  prefix property holds
  2  prefix property does not hold -- a real finding; take the subset from one run
  1  could not be determined (a prerequisite failed, ProteinMPNN errored, ...)

Run from the repo root; ProteinMPNN is invoked at ./community_models/ProteinMPNN.

  python script_utils/bioinformatic/verify_mpnn_prefix.py --pdb path/to/design.pdb
"""

import argparse
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, "src")

# Only the seeding module is imported eagerly -- it is pure stdlib, and this
# script must be readable and its logic exercisable without torch, biotite or
# the structure stack being importable. ProteinMPNN itself is imported at the
# point it is actually run.
from proteinfoundation.metrics.seeding import MPNN_OMIT_AAS, MPNN_SAMPLING_TEMP, mpnn_seed

EXIT_HOLDS = 0
EXIT_UNDETERMINED = 1
EXIT_DOES_NOT_HOLD = 2


def chain_ids(pdb_path: str) -> list[str]:
    """Chain IDs in a PDB, read without pulling in the structure stack."""
    seen = []
    with open(pdb_path) as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                chain = line[21]
                if chain not in seen:
                    seen.append(chain)
    return seen


def seed_from_fasta(fasta_path: str) -> int | None:
    """The seed ProteinMPNN reports having used, from its own output header."""
    try:
        with open(fasta_path) as handle:
            head = handle.readline()
    except OSError:
        return None
    match = re.search(r"seed=(\d+)", head)
    return int(match.group(1)) if match else None


def design_name(pdb_path: str) -> str:
    """The design identity the seed is derived from.

    Deferred to the real implementation rather than reimplemented: the seed is a
    function of this string, so a lookalike here would test a different seed than
    production uses.
    """
    from proteinfoundation.utils.pdb_utils import pdb_name_from_path

    return pdb_name_from_path(pdb_path)


def sample(pdb_path, workdir, tag, context_chains, design_chain, n, seed):
    """One ProteinMPNN run, in its own directory so nothing overwrites anything.

    Returns (sequences, reported_seed). Each run gets a fresh output directory
    because ProteinMPNN writes seqs/<pdb_stem>.fa -- two runs over one design in
    a shared directory would silently compare a file against itself.
    """
    from proteinfoundation.metrics.inverse_folding_models import run_proteinmpnn

    out_dir = os.path.join(workdir, tag)
    os.makedirs(out_dir, exist_ok=True)
    results = run_proteinmpnn(
        pdb_path,
        out_dir,
        all_chains=context_chains,
        pdb_path_chains=[design_chain],
        num_seq_per_target=n,
        omit_AAs="".join(MPNN_OMIT_AAS),
        sampling_temp=MPNN_SAMPLING_TEMP,
        seed=seed,
        verbose=False,
    )
    fasta_path = os.path.join(out_dir, "seqs", design_name(pdb_path) + ".fa")
    return [r["seq"] for r in results], seed_from_fasta(fasta_path)


def report(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return ok


def run_case(pdb_path, workdir, label, context_chains, design_chain, small, large):
    """Prerequisites then the prefix question, for one conditioning.

    Returns True/False for the prefix property, or None if it could not be asked.
    """
    print(f"\n{label}")
    print(f"  context chains: {context_chains}, designing: {design_chain}")
    seed = mpnn_seed(design_name(pdb_path), context_chains, [design_chain])
    print(f"  derived seed:   {seed}")

    prefix = f"{label.split()[0].lower()}"
    try:
        a, seed_a = sample(pdb_path, workdir, f"{prefix}_a", context_chains, design_chain, small, seed)
        b, seed_b = sample(pdb_path, workdir, f"{prefix}_b", context_chains, design_chain, small, seed)
        big, _ = sample(pdb_path, workdir, f"{prefix}_big", context_chains, design_chain, large, seed)
    except Exception as exc:
        print(f"  [ABORT] ProteinMPNN failed: {exc}")
        return None

    ok = True
    ok &= report(
        f"returned {small}/{small} and {large}/{large} sequences",
        len(a) == small and len(big) == large,
        f"got {len(a)} and {len(big)}",
    )
    ok &= report(
        "determinism: same seed and count twice",
        a == b,
        "identical" if a == b else f"{sum(x != y for x, y in zip(a, b, strict=False))}/{len(a)} differ",
    )
    # Measured on the long run too, because that is what the prefix is checked
    # against. A short run of 2 can be fully distinct while the 8 it is compared
    # to are near-identical, and then matching the first 2 says nothing -- the
    # sequences would agree whatever the RNG did.
    ok &= report(
        "non-degeneracy: sequences within a run differ",
        len(set(a)) > 1 and len(set(big)) > 1,
        f"{len(set(a))}/{len(a)} distinct (short), {len(set(big))}/{len(big)} distinct (long)",
    )
    ok &= report(
        "seed plumbing: ProteinMPNN used the seed we passed",
        seed_a == seed and seed_b == seed,
        f"reported {seed_a}, {seed_b}; passed {seed}",
    )

    if not ok:
        print("  [ABORT] prerequisites failed; the prefix question is not meaningful here")
        return None

    holds = big[: len(a)] == a
    n_match = sum(x == y for x, y in zip(a, big, strict=False))
    report(
        f"PREFIX: first {small} of a {large}-sequence run match the {small}-sequence run",
        holds,
        f"{n_match}/{len(a)} positions match",
    )
    if not holds:
        print(f"      short run [0]: {a[0]}")
        print(f"      long  run [0]: {big[0]}")
    elif len(a) < 4 or len(set(big)) < len(big):
        # Agreement over few sequences, or over a draw that repeats itself, is
        # weak evidence for something the sharing step will rely on everywhere.
        print(
            f"      NOTE: only {len(a)} sequence(s) compared and {len(set(big))}/{len(big)} of the "
            f"long run distinct; raise --small for a stronger result."
        )
    return holds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdb", required=True, help="A design PDB. For a binder, the complex.")
    ap.add_argument("--binder-chain", default=None, help="Chain to redesign (default: last chain)")
    ap.add_argument("--small", type=int, default=2, help="num_redesign_seqs, the shared subset size")
    ap.add_argument("--large", type=int, default=8, help="designability_num_seq, the full draw")
    ap.add_argument("--keep", action="store_true", help="Keep the working directory for inspection")
    args = ap.parse_args()

    if not os.path.exists(args.pdb):
        print(f"No such PDB: {args.pdb}")
        return EXIT_UNDETERMINED
    if not os.path.isdir("community_models/ProteinMPNN"):
        print("Run from the repo root: community_models/ProteinMPNN not found")
        return EXIT_UNDETERMINED
    if args.small >= args.large:
        print(f"--small ({args.small}) must be less than --large ({args.large})")
        return EXIT_UNDETERMINED

    chains = chain_ids(args.pdb)
    if not chains:
        print(f"No chains found in {args.pdb}")
        return EXIT_UNDETERMINED
    binder = args.binder_chain or chains[-1]
    if binder not in chains:
        print(f"Chain {binder} not in {args.pdb} (has {chains})")
        return EXIT_UNDETERMINED
    targets = [c for c in chains if c != binder]

    print(f"PDB:     {args.pdb}")
    print(f"Chains:  {chains} (binder {binder}, target {targets or 'none'})")
    print(f"Counts:  {args.small} shared out of {args.large}")

    workdir = tempfile.mkdtemp(prefix="mpnn_prefix_")
    verdicts = {}
    try:
        # Binder alone -- the conditioning designability used before this work.
        verdicts["binder_only"] = run_case(
            args.pdb, workdir, "binder_only (binder alone)", [binder], binder, args.small, args.large
        )
        # The complex -- what both tracks condition on now, and the case the
        # sharing step actually depends on.
        if targets:
            verdicts["complex"] = run_case(
                args.pdb, workdir, "complex (binder in target context)", chains, binder, args.small, args.large
            )
        else:
            print("\ncomplex: skipped -- single-chain PDB has no target context")
    finally:
        if args.keep:
            print(f"\nWorking directory kept: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    answered = {k: v for k, v in verdicts.items() if v is not None}
    print("\n" + "=" * 70)
    if not answered:
        print("RESULT: UNDETERMINED -- no case produced a usable answer")
        return EXIT_UNDETERMINED
    for name, holds in answered.items():
        print(f"  {name:12s} prefix property {'HOLDS' if holds else 'DOES NOT HOLD'}")
    if len(answered) < len(verdicts):
        print("  (some cases were undetermined; see above)")

    # The sharing step depends on the complex case. A binder_only result that
    # disagrees is worth seeing but does not decide anything.
    deciding = answered.get("complex", answered.get("binder_only"))
    if deciding:
        print("\nRESULT: PASS -- the shared subset can be reproduced by a shorter seeded run.")
        return EXIT_HOLDS
    print("\nRESULT: PREFIX PROPERTY DOES NOT HOLD.")
    print("The sharing step must take its subset from one run of designability_num_seq")
    print("rather than reproducing it with a run of num_redesign_seqs.")
    return EXIT_DOES_NOT_HOLD


if __name__ == "__main__":
    sys.exit(main())
