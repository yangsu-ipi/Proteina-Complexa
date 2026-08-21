#!/usr/bin/env python3
"""Split the wwPDB monolithic components.cif into the CCD mirror layout atomworks expects.

atomworks resolves a component to `<root>/<CODE[0]>/<CODE>/<CODE>.cif`
(`atomworks/io/utils/ccd.py:233`), and its scanner (`ccd.py:141-157`) requires the
level-1 directory name to be exactly one character, the level-2 name to be the code and to
start with that character, and the file to be named `<CODE>.cif`.

wwPDB's own per-component tree at /pub/pdb/refdata/chem_comp/ shards by the LAST character
(H/CH/CH.cif), so it is not drop-in compatible. components.cif is a concatenation of
`data_<CODE>` blocks, so a boundary split yields valid single-block CIFs instead.

Also writes `.ccd_codes_cache` (one code per line), which atomworks reads in place of
walking ~48k directories -- but only while the cache is newer than the root directory
(`ccd.py:121-133`). Written last so its mtime wins.

Usage:
    curl -o components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
    python build_ccd_mirror.py components.cif.gz /data/mirrors/ccd

Roughly 118 MB gzipped, ~1.3 GB expanded, ~48k components. Standard library only.
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys


def split_components(src: str, root: str) -> int:
    """Write one CIF per data_ block. Returns the number of components written."""
    opener = gzip.open if src.endswith(".gz") else open
    written = 0
    code: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal written
        if code is None:
            return
        # Codes are used as path components; refuse anything that could escape the root.
        if os.sep in code or code in (".", "..") or (os.altsep and os.altsep in code):
            print(f"  skipping unsafe component code: {code!r}", file=sys.stderr)
            return
        d = os.path.join(root, code[0], code)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, code + ".cif"), "w") as fh:
            fh.writelines(buf)
        written += 1

    with opener(src, "rt") as fh:
        for line in fh:
            if line.startswith("data_"):
                flush()
                code, buf = line[5:].strip(), [line]
            elif code is not None:
                buf.append(line)
    flush()
    return written


def write_cache(root: str) -> int:
    """Write .ccd_codes_cache so atomworks can skip the filesystem walk."""
    codes: list[str] = []
    for shard in sorted(os.listdir(root)):
        shard_path = os.path.join(root, shard)
        if len(shard) != 1 or not os.path.isdir(shard_path):
            continue
        for code in sorted(os.listdir(shard_path)):
            if os.path.isfile(os.path.join(shard_path, code, code + ".cif")):
                codes.append(code)
    with open(os.path.join(root, ".ccd_codes_cache"), "w") as fh:
        fh.write("".join(c + "\n" for c in codes))
    return len(codes)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("components", help="Path to components.cif or components.cif.gz")
    p.add_argument("root", help="CCD mirror root (becomes $CCD_MIRROR_PATH)")
    args = p.parse_args()

    if not os.path.isfile(args.components):
        print(f"error: no such file: {args.components}", file=sys.stderr)
        return 1

    os.makedirs(args.root, exist_ok=True)
    written = split_components(args.components, args.root)
    if written == 0:
        print(
            "error: no 'data_' blocks found -- is this really components.cif?",
            file=sys.stderr,
        )
        return 1

    cached = write_cache(args.root)
    print(f"wrote {written} components to {args.root}")
    print(f"cached {cached} codes in {os.path.join(args.root, '.ccd_codes_cache')}")
    print(f"\nset: CCD_MIRROR_PATH={os.path.abspath(args.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
