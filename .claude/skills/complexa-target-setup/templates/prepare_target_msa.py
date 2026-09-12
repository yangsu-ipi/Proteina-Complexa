#!/usr/bin/env python3
"""Fetch a target MSA from the ColabFold public MMseqs2 server, for ESMFold2.

CAMPAIGN TEMPLATE -- copy into <campaign>/scripts/ unchanged and drive it from
`PREPARE_STEPS` in campaign.env:

    PREPARE_STEPS=(
      "scripts/prepare_target_msa.py --pdb $TARGET_PDB --chain $TARGET_CHAIN --out $TARGET_MSA"
    )

`--out $TARGET_MSA` is the point: campaign.env defines TARGET_MSA once, run_campaign.sh
exports it, and pipeline.yaml reads it back as
`${oc.env:CAMPAIGN_DIR}/${oc.env:TARGET_MSA}`. The file this writes and the file folding
opens are then the same string by construction, and check_preflight.py gates on it.
Written by convention into two places instead, they drift, and the run dies at evaluate
time with FileNotFoundError.

TARGET_MSA is package-relative, so an --out arrives relative to the campaign package and
resolves against the cwd run_campaign.sh already cds to. That keeps the package portable;
the work directory below lands beside the alignment, inside the package, for the same
reason.

Only the *target* gets an MSA. The binder never does -- `consensus_folding.py:163`
passes `msa=None` for it unconditionally, because a de novo miniprotein has no
meaningful alignment and a spurious one makes the prediction worse.

WHY EVERYTHING HEAVY IS IMPORTED INSIDE FUNCTIONS
-------------------------------------------------
A campaign package is often assembled on a laptop and run on a GPU box. ColabFold
is not a dependency of this repo and will usually be absent at package-creation
time; `esm` and `proteinfoundation` may be too. So retrieval and validation are
deferred to campaign execution: this module imports nothing but the standard
library at import time, `--help` works anywhere, and a missing dependency is
reported by name at the moment it is actually needed.

Retrieval is now a SUBPROCESS -- `colabfold_batch --msa-only`, named by
COLABFOLD_EXEC_PATH -- rather than an in-process import, so nothing here needs
ColabFold installed beside `proteinfoundation` at all. Validation still does need
`proteinfoundation`, which is the environment this runs in anyway.

THE PUBLIC SERVER IS A SHARED FREE RESOURCE
-------------------------------------------
`https://api.colabfold.com` is run for the community. This script queries it once
per target chain per campaign -- `PREPARE_STEPS` runs before EVERY stage
(`run_campaign.sh:156`), so a re-run of `evaluate` would otherwise re-query -- by
skipping any output that already exists and still validates. Point `--host-url` at
your own MMseqs2 server if you are doing this at volume, and set a contact in
`--user-agent`: ColabFold asks for one and warns that it will become mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HOST = "https://api.colabfold.com"
DEFAULT_USER_AGENT = "Proteina-Complexa-campaign/1.0"


def missing(dep: str, what: str, how: str) -> SystemExit:
    """One shape for every deferred-import failure, naming the fix."""
    return SystemExit(
        f"error: {what} needs `{dep}`, which is not importable here.\n"
        f"  {how}\n"
        "  This script imports it lazily on purpose, so a campaign package can be\n"
        "  built on a machine that has neither ColabFold nor the complexa env."
    )


def target_sequence(pdb: str, chain: str) -> str:
    """The chain sequence, read the way evaluation reads it.

    Not the contig. `binder_eval.py:600` folds `_target_chain_sequences`, which is
    `extract_seq_from_pdb(path, chain_id=...)` over the WHOLE chain -- `target_input`
    never enters it. An MSA whose query is the contig subset would be rejected at
    evaluate time by the query-length check in `consensus_folding.py:322`, hours in.
    Calling the pipeline's own function is what keeps the two identical.
    """
    try:
        from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb
    except ImportError as exc:
        raise missing("proteinfoundation", "reading the target sequence",
                      f"run inside the complexa environment ({exc})") from exc
    seq = extract_seq_from_pdb(pdb, chain_id=chain)
    if not seq:
        raise SystemExit(f"error: chain {chain!r} of {pdb} yielded no sequence")
    return seq


def colabfold_batch() -> str:
    """The `colabfold_batch` to run, named the way the pipeline names it.

    Same variable the apo folding backend reads, because it is the same question
    -- which ColabFold installation -- and answering it twice is how two answers
    start to disagree. ColabFold lives in an environment of its own (its
    `[alphafold]` extra downgrades absl-py, biopython and chex and pins a jax a
    Blackwell card cannot use), so this is a subprocess and an absolute path is
    what reaches it.
    """
    import os
    import shutil

    named = os.environ.get("COLABFOLD_EXEC_PATH")
    if named:
        return named
    found = shutil.which("colabfold_batch")
    if found:
        return found
    raise SystemExit(
        "error: fetching an MSA needs colabfold_batch, which is not on PATH and\n"
        "  COLABFOLD_EXEC_PATH is unset. ColabFold lives in an environment of its own;\n"
        "  set COLABFOLD_EXEC_PATH to its colabfold_batch (the same variable apo folding\n"
        "  reads). Do not pip install colabfold into the complexa environment for this."
    )


def _supports(binary: str, flag: str) -> bool:
    """Whether `binary --help` mentions *flag*. False if it cannot be asked."""
    import subprocess

    try:
        helped = subprocess.run([binary, "--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return flag in (helped.stdout or "") + (helped.stderr or "")


def fetch_a3m(seq: str, work_prefix: Path, host_url: str, user_agent: str, use_env: bool) -> str:
    """One alignment for one sequence, from the MMseqs2 API.

    Shelled out to `colabfold_batch --msa-only`, which queries the same public
    server and writes `{jobname}.a3m` beside its other outputs without folding
    anything -- no weights are touched. This used to import
    `colabfold.colabfold.run_mmseqs2` in-process, which is why the complexa
    environment carried a `--no-deps colabfold` for one function, at the cost of
    a biopython cap it did not want and four broken console scripts, one of
    which then shadowed the real folder on PATH. A shared tool answers for
    itself; this script asks it a question instead of importing its internals.

    The query is written as a FASTA whose record id is the job name, because
    that is what names the output file.
    """
    import subprocess

    job = work_prefix.name
    work = work_prefix.parent / f"{job}_msa"
    work.mkdir(parents=True, exist_ok=True)
    fasta = work / f"{job}.fasta"
    fasta.write_text(f">{job}\n{seq}\n")

    binary = colabfold_batch()
    command = [
        binary, "--msa-only",
        # unpaired, with the environmental databases unless asked otherwise.
        # Pairing is for paired chains of one complex, not a single chain;
        # templates are a separate feature and ESMFold2 takes none.
        "--msa-mode", "mmseqs2_uniref_env" if use_env else "mmseqs2_uniref",
        "--host-url", host_url,
    ]
    # The library function took a user_agent; the CLI did not gain one until
    # after 1.6.1, where passing it is an argparse error rather than a warning.
    # Asked rather than assumed, so this works either side of that change and a
    # contact still reaches the server wherever it can.
    if _supports(binary, "--user-agent"):
        command += ["--user-agent", user_agent]
    else:
        print(f"  note: {Path(binary).name} takes no --user-agent; the contact "
              f"{user_agent!r} will not reach {host_url}")
    command += [str(fasta), str(work / "out")]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"error: cannot run {command[0]}: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"error: {command[0]} --msa-only failed (exit {exc.returncode}) against {host_url}"
        ) from exc

    produced = sorted((work / "out").glob("*.a3m"))
    if not produced:
        raise SystemExit(f"error: {host_url} returned no alignment for a {len(seq)}-residue query")
    # Named for the job when there is one; otherwise whatever single file appeared.
    exact = work / "out" / f"{job}.a3m"
    return (exact if exact.exists() else produced[0]).read_text()


def validate(path: Path, seq: str, max_sequences: int):
    """Apply evaluation's own acceptance test, and return the parsed alignment.

    Deliberately the pipeline's `_target_msas` rather than a second copy of its
    rules: it is what raises at evaluate time, so agreeing with it here is the
    whole point. It checks that the alignment's query IS the chain being folded
    and that depth >= 2, and its message names which. If that private name is ever
    renamed this breaks loudly, which is the failure mode to prefer -- a template
    carrying its own drifting copy of someone else's checks is the other one.
    """
    try:
        from proteinfoundation.metrics.consensus_folding import _target_msas
    except ImportError as exc:
        raise missing("proteinfoundation/esm", "validating an MSA",
                      f"run inside the complexa environment ({exc})") from exc
    msas = _target_msas([seq], {"target_msa": str(path), "msa_max_sequences": max_sequences})
    return msas[0]


def write_provenance(a3m: Path, msa, seq: str, pdb: str, chain: str, host_url: str) -> Path:
    """What produced this alignment, beside it.

    An a3m is an opaque blob months later, and the campaign's own metadata does not
    cover files fetched by a prepare step. The digest is what makes "the MSA changed"
    answerable -- and evaluation digests the same contents into its cache fingerprint
    (`consensus_folding.py:500-509`), so a changed alignment invalidates rather than
    serving scores computed against the old one.
    """
    side = a3m.with_suffix(a3m.suffix + ".provenance.json")
    side.write_text(json.dumps({
        "a3m": a3m.name,
        "sha256": hashlib.sha256(a3m.read_bytes()).hexdigest(),
        "query_residues": len(seq),
        "depth": int(msa.depth),
        "source_pdb": str(pdb),
        "source_chain": chain,
        "host_url": host_url,
        # timezone.utc, not datetime.UTC (which ruff UP017 asks for under this repo's
        # py312 target): UTC is 3.11+, and a package is often assembled with whatever
        # python the machine has -- macOS still ships 3.9, which is also why the
        # __future__ import above is here.
        "retrieved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
    }, indent=2) + "\n")
    return side


def prepare_one(pdb: str, chain: str, out: Path, args) -> Path:
    """Retrieve, write and validate one chain's alignment. Idempotent."""
    seq = target_sequence(pdb, chain)
    if out.is_file() and not args.force:
        try:
            msa = validate(out, seq, args.max_sequences)
        except Exception as exc:
            # Includes the case that matters most: the target PDB changed under an
            # alignment fetched for the old sequence. Refetching is the repair.
            print(f"  {out.name} exists but does not validate -- refetching. ({exc})")
        else:
            print(f"  {out.name} already valid: {len(seq)} residues, depth {msa.depth}")
            return out
    print(f"  fetching chain {chain} ({len(seq)} residues) from {args.host_url}")
    text = fetch_a3m(seq, out.parent / ".msa_work" / out.stem, args.host_url, args.user_agent, not args.no_env)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    # Validate what was WRITTEN, not what was returned. A short write, a full disk
    # or a server that answered with something else all land here rather than at
    # evaluate time.
    msa = validate(out, seq, args.max_sequences)
    write_provenance(out, msa, seq, pdb, chain, args.host_url)
    print(f"  wrote {out} -- depth {msa.depth}")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pdb", required=True, help="Target PDB, the same file the config points at")
    p.add_argument("--chain", action="append", required=True, metavar="ID",
                   help="Target chain; repeat, IN THE ORDER the target's chains are folded")
    p.add_argument("--out", action="append", type=Path, metavar="PATH",
                   help="Exact output path, one per --chain and in the same order. Prefer this "
                        "over --out-dir: it is what lets campaign.env name the file once, so "
                        "pipeline.yaml's target_msa and this step cannot drift apart")
    p.add_argument("--out-dir", type=Path, default=Path("data"),
                   help="Used only when --out is absent; writes <out-dir>/<pdb stem>_<chain>.a3m")
    p.add_argument("--host-url", default=DEFAULT_HOST)
    p.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                   help="ColabFold asks for 'tool/version contact@email' and warns without one")
    p.add_argument("--max-sequences", type=int, default=16384,
                   help="Match consensus_cfg.msa_max_sequences; validation must see what folding sees")
    p.add_argument("--no-env", action="store_true", help="Skip the environmental databases")
    p.add_argument("--force", action="store_true", help="Refetch even if a valid alignment exists")
    args = p.parse_args()

    if args.out and len(args.out) != len(args.chain):
        # Silently zipping these would write chain B's alignment to chain A's path,
        # and the result validates -- both are real alignments of real chains.
        p.error(f"--out given {len(args.out)} time(s) for {len(args.chain)} chain(s); "
                "pass one --out per --chain, in the same order")
    stem = Path(args.pdb).stem
    outs = args.out or [args.out_dir / f"{stem}_{c}.a3m" for c in args.chain]
    written = [prepare_one(args.pdb, c, o, args) for c, o in zip(args.chain, outs, strict=True)]

    # Printed to be pasted, like check_target_pdb.py's target_input. The plural form
    # is not optional for a multi-chain target: target_msa_paths takes one entry per
    # chain, null where a chain has none, or it raises with the counts
    # (`consensus_folding.py:308`).
    print("\nadd to pipeline.yaml under metric.consensus_cfg:")
    if len(written) == 1:
        print(f"    target_msa: {written[0]}")
    else:
        print(f"    target_msa_paths: [{', '.join(str(w) for w in written)}]")
    print(f"    msa_max_sequences: {args.max_sequences}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
