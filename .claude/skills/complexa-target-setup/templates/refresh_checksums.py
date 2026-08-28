#!/usr/bin/env python3
# CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged.
# Hashes the campaign package, excluding everything a run produces.
# Validated by the CBLN1/5KC5 campaign, first complete run 2026-08-28.
# Campaign-independent: every input is an argument or derived from the package
# layout. If you find yourself editing this file per campaign, that is a bug
# in the template -- add an argument instead, so the next campaign inherits it.
from pathlib import Path
import hashlib
root=Path(__file__).resolve().parents[1]
paths=[p for p in root.rglob("*") if p.is_file() and p.name!="CHECKSUMS.sha256" and not any(x in p.parts for x in ("metadata","inference","evaluation_results","logs","__pycache__"))]
(root/"CHECKSUMS.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in sorted(paths)))
