#!/usr/bin/env python3
"""Preflight a target PDB/CIF against the target config values you plan to use.

Checks the three things `complexa validate target` does not -- it confirms the file exists,
then echoes your config back without ever opening it
(`src/proteinfoundation/cli/validate.py:379-503`):

  1. Heteroatoms inside the contig range. `AtomSelectionStack.from_contig` filters on
     (chain_id, res_id) only, with no polymer/hetero/element filter
     (`atomworks/io/utils/selection.py:482-493`), so in-range waters, ions and glycans are
     fed to the AF2 atom37 protein encoding.
  2. The residue range actually present, so `target_input` matches the file's numbering
     rather than a range copied from a bundled example.
  3. Hotspot resolution. Hotspots are matched as f"{chain_id}{res_id}" strings and misses
     are silent (`src/proteinfoundation/utils/pdb_utils.py:571-575`).

Reads the file the same way the pipeline does -- `load_any(path, model=1)` via the import
path used at `src/proteinfoundation/utils/pdb_utils.py:50`. Note that .cif input yields
`label_seq_id` and .pdb yields author numbering (`atomworks/io/utils/io_utils.py:290` sets
`use_author_fields=False`), so run this on the exact file you will feed to the pipeline.

Usage:
    python check_target_pdb.py --pdb target.pdb --chain A --hotspots A37 A39 A49 A98
    python check_target_pdb.py --pdb target.cif --chain A --target-input A1-115
    python check_target_pdb.py --pdb raw.pdb --chain A --write-clean clean.pdb

Exits 1 if any requested hotspot is unmatched, or if the chain is absent.
Requires atomworks and biotite (i.e. the complexa environment).
"""

from __future__ import annotations

import argparse
import sys


def _load_any():
    """Import load_any lazily so --help works without the complexa environment."""
    try:
        # Same import path the pipeline uses (pdb_utils.py:50). Note that
        # `atomworks.io` itself exports only `parse` -- there is no top-level
        # `load_any` re-export in atomworks 2.2.1.
        from atomworks.io.utils.io_utils import load_any
    except ImportError as exc:
        sys.exit(
            f"error: cannot import atomworks ({exc}).\n"
            "Run this inside the complexa environment. If atomworks is genuinely missing,\n"
            "note that env/build_uv_env.sh:174 installs it non-fatally, so a 'successful'\n"
            "build can still lack it."
        )
    return load_any


def parse_contig(spec: str) -> dict[str, tuple[int, int]]:
    """Parse 'A1-115' or 'A1-50,B1-50' into {chain: (start, stop)} inclusive."""
    import re

    out: dict[str, tuple[int, int]] = {}
    for part in spec.replace(" ", "").split(","):
        m = re.match(r"^([A-Za-z]+)(\d+)-(\d+)$", part)
        if not m:
            sys.exit(f"error: cannot parse contig segment {part!r} (expected e.g. A1-115)")
        chain, start, stop = m.group(1), int(m.group(2)), int(m.group(3))
        out[chain] = (start, stop)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdb", required=True, help="Target PDB or CIF path")
    p.add_argument("--chain", required=True, help="Chain ID to inspect, e.g. A")
    p.add_argument("--hotspots", nargs="*", default=[], help="e.g. A37 A39 A49 A98")
    p.add_argument("--target-input", help="Existing contig to validate, e.g. A1-115")
    p.add_argument("--write-clean", help="Write a heteroatom-stripped copy here (no renumbering)")
    args = p.parse_args()

    import numpy as np

    load_any = _load_any()
    struct = load_any(args.pdb, model=1)
    # load_any may hand back a stack depending on input format; take the first model.
    if getattr(struct, "ndim", 1) > 1 or struct.__class__.__name__.endswith("Stack"):
        struct = struct[0]

    in_chain = struct[struct.chain_id == args.chain]
    if in_chain.array_length() == 0:
        present = sorted(set(struct.chain_id.tolist()))
        print(f"FAIL  chain {args.chain!r} not found. Chains present: {present}")
        return 1

    polymer = in_chain[~in_chain.hetero]
    hetero = in_chain[in_chain.hetero]
    ca = polymer[polymer.atom_name == "CA"]
    if ca.array_length() == 0:
        print(f"FAIL  chain {args.chain} has no CA atoms among non-hetero residues")
        return 1

    res_ids = np.unique(ca.res_id)
    lo, hi = int(res_ids.min()), int(res_ids.max())

    print(f"file          : {args.pdb}")
    print(f"chain         : {args.chain}  ({len(res_ids)} residues with CA)")
    print(f"target_input  : {args.chain}{lo}-{hi}")

    gaps = sorted(set(range(lo, hi + 1)) - set(int(r) for r in res_ids))
    if gaps:
        print(f"gaps          : {len(gaps)} unresolved res_id(s): {gaps[:20]}"
              f"{' ...' if len(gaps) > 20 else ''}")
    else:
        print("gaps          : none")

    # --- heteroatoms that from_contig would pull in ---
    lo_c, hi_c = lo, hi
    if args.target_input:
        seg = parse_contig(args.target_input).get(args.chain)
        if seg is None:
            print(f"WARN  --target-input {args.target_input!r} does not mention chain {args.chain}")
        else:
            lo_c, hi_c = seg
            n_sel = int(((res_ids >= lo_c) & (res_ids <= hi_c)).sum())
            print(f"contig check  : {args.chain}{lo_c}-{hi_c} selects {n_sel} of {len(res_ids)} residues")
            if n_sel == 0:
                print("FAIL  contig selects ZERO residues -- numbering mismatch")
                return 1
            if n_sel < len(res_ids):
                print(f"WARN  contig silently drops {len(res_ids) - n_sel} residue(s)")

    if hetero.array_length():
        mask = (hetero.res_id >= lo_c) & (hetero.res_id <= hi_c)
        in_range = hetero[mask]
        names = sorted(set(in_range.res_name.tolist()))
        if names:
            print(f"IN-RANGE HET  : {len(in_range)} atom(s), residues {names}")
            print("                these reach the AF2 atom37 protein encoding -- strip them")
        else:
            print(f"hetero        : {hetero.array_length()} atom(s), all outside the range")
    else:
        print("hetero        : none in this chain")

    # --- hotspots ---
    rc = 0
    if args.hotspots:
        ids = {f"{a.chain_id}{a.res_id}" for a in ca}
        ok = [h for h in args.hotspots if h in ids]
        miss = [h for h in args.hotspots if h not in ids]
        print(f"hotspots ok   : {ok}")
        if miss:
            print(f"hotspots MISS : {miss}")
            print("                these would be silently dropped -- the design gets no")
            print("                epitope guidance. Check numbering, chain, and .cif/.pdb choice.")
            rc = 1
        in_gap = [h for h in ok if int(h[len(args.chain):]) in gaps]
        if in_gap:
            print(f"WARN  hotspot(s) in an unresolved gap: {in_gap}")

    if args.write_clean:
        import biotite.structure.io as strucio  # same writer generate.py:1352 uses

        strucio.save_structure(args.write_clean, polymer)
        print(f"wrote clean   : {args.write_clean} (heteroatoms stripped, numbering preserved)")

    print("\nRESULT: " + ("PASS" if rc == 0 else "FAIL"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
