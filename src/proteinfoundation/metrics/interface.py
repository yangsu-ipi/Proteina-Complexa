"""One definition of what counts as a protein interface.

Three definitions shipped before this module, and two columns that both said
"interface" counted different residues: `{seq}_aa_interface_counts` used CA
atoms at 8 A, while `interface_nres` and `interface_hydrophobicity` used all
atoms at 4 A. Over 12 CBLN1 complexes they selected 15.9 and 17.9 binder
residues with a mean Jaccard of 0.664, and the 4 A set was never a subset of
the 8 A one -- roughly a third of the union was disputed.

This delegates to protein-interface's `strict` mode, which is burial-aware
rather than purely geometric: a residue is interface if it buries at least
`INTERFACE_DSASA_THRESHOLD` A^2 **or** has an atom within `contact_cutoff`.
The old 4 A set is a strict subset of it in every complex measured (8/8).

Two things it fixes beyond agreement. The classifier reports both sides, so the
target's interface no longer has to be obtained by swapping the arguments of a
binder-only function. And each record carries `seq_index`, a real position
within the chain -- the old sequence-index form computed `res_id - min(res_id)`
and used it to index a sequence string, which silently shifts every count if
residue numbering has a gap.

Radii here are protein-interface's embedded CCP4-sc table and are not
overridable from Python (the documented ATOMIC_RADII override reaches shape
complementarity only). They decide the dSASA half of the strict criterion, so
the package version is part of the provenance. dSASA as a *reported metric*
stays on freesasa/ProtOr -- see docs/design-notes/interface-and-sasa-engines.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

# protein-interface's own strict defaults, restated rather than imported so a
# change upstream is a visible diff here instead of a silent shift in what a
# campaign called an interface.
INTERFACE_MODE = "strict"
INTERFACE_DSASA_THRESHOLD = 3.0
DEFAULT_CONTACT_CUTOFF = 5.0
# Shrake-Rupley discretisation for the burial half. 960 is converged: measured
# 0.008% from n=5000, max 0.076%.
INTERFACE_N_POINTS = 960
INTERFACE_PROBE_RADIUS = 1.4
# Bumped when what counts as an interface changes without a caller changing.
INTERFACE_DERIVATION_VERSION = 1
# Above this, a contact cutoff is almost certainly a CA-CA number applied to all
# atoms. On CBLN1 generated complexes 8.0 A selects 79% of the binder against 49%
# at 5.0. Warned rather than refused: it is a config value, and a wide interface
# is a choice someone may mean -- but not one to make by inheriting a default.
SUSPICIOUS_CONTACT_CUTOFF = 6.0


class InterfaceError(RuntimeError):
    """The interface of a structure could not be determined.

    Raised rather than returning an empty set: no interface residues is a
    meaningful measurement about a binder that missed its target, and it must
    not be indistinguishable from a structure that could not be read.
    """


@dataclass(frozen=True)
class InterfaceResidue:
    """One residue at the interface, on whichever side it belongs to."""

    chain: str
    resseq: int
    icode: str
    seq_index: int  # 1-based position within its own chain
    resname: str
    dsasa: float
    min_interchain_dist: float


def interface_provenance(contact_cutoff: float = DEFAULT_CONTACT_CUTOFF) -> dict:
    """Everything that decides which residues are interface, for the fingerprint."""
    from importlib.metadata import version

    return {
        "engine": "protein-interface",
        "engine_version": version("protein-interface"),
        "mode": INTERFACE_MODE,
        "dsasa_threshold": INTERFACE_DSASA_THRESHOLD,
        "contact_cutoff": contact_cutoff,
        "n_points": INTERFACE_N_POINTS,
        "probe_radius": INTERFACE_PROBE_RADIUS,
        "version": INTERFACE_DERIVATION_VERSION,
    }


def _require_known_radii(pdb_path: str, chains: list[str], include_hetatm: bool) -> None:
    """Refuse a structure holding atoms the radius table does not know.

    protein-interface gives an unrecognised atom radius 0.0 and returns early,
    so it contributes nothing and the burial it should have had is silently
    missing -- which decides the dSASA half of the strict criterion. The
    package exposes the check; it just does not perform it.

    The table matches on atom name with an element-level fallback, so an
    unfamiliar *residue* built from standard atoms resolves fine. What does not
    resolve is an unfamiliar *atom*: a metal cofactor (FE in HEM) and every
    nucleic-acid atom (C1', O5') come back unknown, which is precisely when a
    target would otherwise bury nothing at all without saying so.
    """
    import protein_interface as pi

    try:
        atoms = pi.load_atoms(pdb_path, chains, include_hetatm=include_hetatm)
    except Exception as exc:
        raise InterfaceError(f"could not read {pdb_path}: {exc}") from exc
    unknown = pi.unknown_sasa_radius_atoms(atoms.atom_names, atoms.residue_names)
    if unknown:
        sample = sorted({tuple(u) if isinstance(u, (list, tuple)) else u for u in unknown})[:6]
        raise InterfaceError(
            f"{len(unknown)} atoms in {pdb_path} have no radius in protein-interface's "
            f"table (e.g. {sample}); their burial would be silently zero"
        )


def interface_residues(
    pdb_path: str,
    binder_chains: list[str],
    target_chains: list[str],
    contact_cutoff: float = DEFAULT_CONTACT_CUTOFF,
    include_hetatm: bool = False,
) -> tuple[list[InterfaceResidue], list[InterfaceResidue]]:
    """Interface residues on each side, as ``(binder_side, target_side)``.

    *target_chains* is a list, so a multi-chain target needs no special case --
    the previous all-atom implementation took a single chain id and raised
    KeyError on the comma-joined form its own callers built.

    Set *include_hetatm* for a ligand target, whose atoms are HETATM and would
    otherwise be absent from the calculation entirely.
    """
    import protein_interface as pi

    if not binder_chains or not target_chains:
        raise InterfaceError(f"need both binder and target chains, got {binder_chains} and {target_chains}")
    overlap = set(binder_chains) & set(target_chains)
    if overlap:
        raise InterfaceError(f"chains {sorted(overlap)} are named as both binder and target in {pdb_path}")

    if contact_cutoff > SUSPICIOUS_CONTACT_CUTOFF:
        logger.warning(
            f"interface contact_cutoff={contact_cutoff} A is an all-atom distance; values above "
            f"{SUSPICIOUS_CONTACT_CUTOFF} were calibrated for CA-CA and select most of a small "
            f"binder rather than its interface"
        )
    _require_known_radii(pdb_path, list(binder_chains) + list(target_chains), include_hetatm)
    try:
        classified = pi.classify_residues(
            pdb_path,
            groups=[list(target_chains), list(binder_chains)],
            chains=list(target_chains) + list(binder_chains),
            mode=INTERFACE_MODE,
            contact_cutoff=contact_cutoff,
            probe_radius=INTERFACE_PROBE_RADIUS,
            n_points=INTERFACE_N_POINTS,
            include_hetatm=include_hetatm,
        )
    except Exception as exc:
        raise InterfaceError(f"could not classify residues in {pdb_path}: {exc}") from exc

    binder_set = set(binder_chains)
    binder_side: list[InterfaceResidue] = []
    target_side: list[InterfaceResidue] = []
    for r in classified.records:
        if r.category != "interface":
            continue
        entry = InterfaceResidue(
            chain=r.chain,
            resseq=int(r.resseq),
            icode=r.icode or "",
            seq_index=int(r.seq_index),
            resname=r.resname,
            dsasa=float(r.dsasa),
            min_interchain_dist=float(r.min_interchain_dist),
        )
        (binder_side if r.chain in binder_set else target_side).append(entry)
    return (
        sorted(binder_side, key=lambda e: (e.chain, e.resseq)),
        sorted(target_side, key=lambda e: (e.chain, e.resseq)),
    )


def sequence_indices(residues: list[InterfaceResidue]) -> list[int]:
    """0-based positions for indexing a chain's sequence string.

    From the classifier's own ``seq_index`` rather than ``res_id - min(res_id)``,
    so a gap in residue numbering does not shift every position.
    """
    return sorted(r.seq_index - 1 for r in residues)


def resseqs(residues: list[InterfaceResidue]) -> set[int]:
    """PDB residue numbers, for joining against anything keyed that way."""
    return {r.resseq for r in residues}
