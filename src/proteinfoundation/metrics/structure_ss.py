"""Secondary structure read off a folded structure, in DSSP's own eight states.

Why mdtraj rather than biotite or the dssp binary, measured on 400 campaign
structures: mdtraj's Kabsch-Sander implementation agrees with mkdssp 4.5.5 on
38,287 of 38,288 residues once DSSP 4's polyproline-II state is read as coil,
which classic DSSP had no state for either. biotite's P-SEA -- the engine behind
the retired ``_res_ss_*`` columns -- agrees on helix (97% precision) but calls
four times as many beta residues as DSSP confirms, 877 claimed against 178 real,
so its sheet fraction was not safe to report let alone gate on. And mdtraj is
already a pinned dependency, where mkdssp needs an uncompressed 480 MB
components.cif before it will start.

Eight states are recorded and collapsed to three late. A change to the collapse
rule then costs a re-read of a CSV rather than a refold, which is the same reason
verdicts are re-derived in analyze rather than frozen in evaluate.
"""

from __future__ import annotations

import ast
from collections.abc import Collection

# DSSP's eight states, in the order a packed ``ss_counts`` column stores them.
# Order is part of the data format: a reader unpacks by position.
SS_STATE_ORDER = ("H", "G", "I", "E", "B", "T", "S", "-")

# The collapse this repo has always used, from calc_ss_percentage: 3-10 and pi
# helices count as helix, and only extended strand counts as sheet -- an isolated
# beta bridge (B) is loop. Kept rather than improved, so that one rule exists
# instead of two that disagree by a percent.
SS_COARSE: dict[str, str] = {"H": "helix", "G": "helix", "I": "helix", "E": "sheet"}
SS_COARSE_DEFAULT = "loop"
SS_COARSE_STATES = ("helix", "sheet", "loop")

# Bumped when what is read off a structure changes without a column being
# renamed: the engine, the state set, or the collapse rule.
SS_DERIVATION_VERSION = 1

def ss_provenance() -> dict:
    """What determines an SS number, for the metric fingerprint."""
    return {
        "engine": "mdtraj",
        "states": list(SS_STATE_ORDER),
        "collapse": {k: SS_COARSE.get(k, SS_COARSE_DEFAULT) for k in SS_STATE_ORDER},
        "version": SS_DERIVATION_VERSION,
    }


def counts_from_states(states: Collection[str]) -> list[float]:
    """Pack per-residue DSSP states into counts ordered by SS_STATE_ORDER.

    mdtraj writes coil as a space and biotite-era code wrote it as an empty
    string; both mean the same thing and are normalised to "-". A state outside
    the eight is counted as coil rather than dropped, so the counts always sum to
    the number of residues and a reader can trust the total.
    """
    index = {state: i for i, state in enumerate(SS_STATE_ORDER)}
    counts = [0.0] * len(SS_STATE_ORDER)
    for raw in states:
        state = (str(raw).strip() or "-").upper()
        counts[index.get(state, index["-"])] += 1.0
    return counts


def unpack_counts(value) -> list[float] | None:
    """Read a packed ss_counts back, from a list or from its CSV text.

    Returns None for anything that is not eight numbers, so a malformed cell
    reads as "not measured" rather than as zeros -- which would look like a
    structure with no secondary structure at all.
    """
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != len(SS_STATE_ORDER):
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def collapse_counts(value) -> dict[str, float]:
    """Three-state fractions from a packed eight-state count.

    Pooled, not paired: the counts handed here are already means over models or
    seeds, so this is a ratio of means. That is the intended reduction for the
    interface, where the denominator -- how many residues are within the cutoff --
    itself varies per model (13 to 16 across five AF2 models of one design was
    typical). Averaging per-model fractions instead would give a model that
    happens to present 13 interface residues the same weight as one presenting
    25, on a denominator small enough for that to be noise.
    """
    counts = unpack_counts(value)
    if counts is None:
        return {}
    total = sum(counts)
    if total <= 0:
        return {}
    out = dict.fromkeys(SS_COARSE_STATES, 0.0)
    for state, count in zip(SS_STATE_ORDER, counts, strict=True):
        out[SS_COARSE.get(state, SS_COARSE_DEFAULT)] += count
    return {state: out[state] / total for state in SS_COARSE_STATES}


def chain_states(pdb_path: str, chain_id: str | None = None) -> list[tuple[int, str]]:
    """(residue number, DSSP state) for one chain, or for the whole file.

    The complex is folded as a whole and read per chain, not folded per chain:
    DSSP's hydrogen bonds cross the interface, and 39 of 150 campaign complexes
    contain binder strands that exist only in the target's presence. Reading a
    chain out of the complex keeps those; folding the chain alone would not.
    """
    import mdtraj as md

    traj = md.load(pdb_path)
    states = md.compute_dssp(traj, simplified=False)[0]
    residues = list(traj.topology.residues)
    if len(states) != len(residues):
        raise ValueError(f"dssp returned {len(states)} states for {len(residues)} residues in {pdb_path}")
    out = []
    for residue, state in zip(residues, states, strict=True):
        if chain_id is not None and getattr(residue.chain, "chain_id", None) != chain_id:
            continue
        out.append((int(residue.resSeq), str(state)))
    return out


def structure_ss(
    pdb_path: str,
    binder_chain: str | None = None,
    interface_resseqs: Collection[int] | None = None,
) -> dict[str, list[float] | float]:
    """Eight-state counts for a binder chain, and for its interface subset.

    *interface_resseqs* is passed in rather than computed here: the interface is
    decided once, by metrics/interface.py, so the SS of interface residues
    describes exactly the residues the composition metrics beside it describe.
    Pass None for a monomer, which has no interface.
    """
    states = chain_states(pdb_path, binder_chain)
    out: dict[str, list[float] | float] = {
        "ss_counts": counts_from_states([s for _, s in states]),
        "ss_total": float(len(states)),
    }
    if interface_resseqs is not None:
        wanted = {int(r) for r in interface_resseqs}
        subset = [s for resseq, s in states if resseq in wanted]
        out["interface_ss_counts"] = counts_from_states(subset)
        out["interface_ss_total"] = float(len(subset))
    return out


def structure_ss_by_selection(
    pdb_path: str,
    selections: dict[str, tuple[Collection[str], Collection[int] | None]],
) -> dict[str, dict[str, list[float] | float]]:
    """Eight-state counts for several chain groups, from one read of the file.

    *selections* maps a label to ``(chains, interface_resseqs)``. Both sides of an
    interface are wanted for every complex, and folding the file twice to get
    them would double the cost of the cheapest metric in the set.

    Residue numbers are matched within their own chain group, so a target
    numbered from 1 and a binder numbered from 1 do not collide.
    """
    import mdtraj as md

    traj = md.load(pdb_path)
    states = md.compute_dssp(traj, simplified=False)[0]
    residues = list(traj.topology.residues)
    if len(states) != len(residues):
        raise ValueError(f"dssp returned {len(states)} states for {len(residues)} residues in {pdb_path}")

    out: dict[str, dict[str, list[float] | float]] = {}
    for label, (chains, interface_resseqs) in selections.items():
        wanted_chains = set(chains)
        rows = [
            (int(r.resSeq), str(state))
            for r, state in zip(residues, states, strict=True)
            if getattr(r.chain, "chain_id", None) in wanted_chains
        ]
        entry: dict[str, list[float] | float] = {
            "ss_counts": counts_from_states([st for _, st in rows]),
            "ss_total": float(len(rows)),
        }
        if interface_resseqs is not None:
            wanted = {int(r) for r in interface_resseqs}
            subset = [st for resseq, st in rows if resseq in wanted]
            entry["interface_ss_counts"] = counts_from_states(subset)
            entry["interface_ss_total"] = float(len(subset))
        out[label] = entry
    return out
