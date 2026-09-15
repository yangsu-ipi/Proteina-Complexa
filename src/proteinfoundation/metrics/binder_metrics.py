import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
from atomworks.ml.transforms.encoding import atom_array_to_encoding
from biotite.structure import AtomArray
from loguru import logger
from torch import Tensor
from transformers import logging as hf_logging

from proteinfoundation.metrics.interface import (
    interface_residues as find_interface_residues,
)
from proteinfoundation.metrics.interface import resseqs, sequence_indices
from proteinfoundation.metrics.inverse_folding_models import (
    inverse_fold,
    resolve_inverse_folding_model,
)
from proteinfoundation.metrics.metric_utils import (
    get_interface_residues_atomistic,
    replace_seq_in_generated_pdb,
    rmsd_metric,
)
from proteinfoundation.metrics.redesign_set import shared_redesign_set
from proteinfoundation.metrics.seeding import (
    MPNN_OMIT_AAS,
    MPNN_SAMPLING_TEMP,
    mpnn_seed,
    redesign_context_chains,
)
from proteinfoundation.utils.align_utils import kabsch_align_ind, kabsch_align_ligand
from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb, pdb_name_from_path, sort_AtomArray_by_chain_id

hf_logging.set_verbosity_error()


def complex_mpnn_chains(gen_target_chain: list[str], binder_chain: str) -> list[str]:
    """The chains ProteinMPNN conditions on when redesigning a binder for refolding.

    The target is visible, so these redesigns are interface-aware -- which is the
    point, and the reason the holo gate judges them. Named so the
    ``redesign_conditioning`` provenance reported for a run is derived from the
    same list the redesigns were actually generated from, rather than asserted
    separately and left to drift. See
    ``docs/design-notes/apo-holo-redesign-sharing.md``.
    """
    return redesign_context_chains(gen_target_chain, binder_chain)


def updated_structure_path(pdb_file_path: str | Path, is_target_ligand: bool) -> str:
    """The ``_updated`` view of a design: real target sequence, C-alpha only.

    ProteinMPNN's input, and nothing else's. A ligand target has no such view --
    LigandMPNN is all-atom and reads the design PDB -- so the path collapses to
    the design itself there.
    """
    name = pdb_name_from_path(pdb_file_path)
    updated = os.path.join(os.path.dirname(pdb_file_path), name + "_updated.pdb")
    return updated.replace("_updated.pdb", ".pdb") if is_target_ligand else updated


def interface_structure_path(pdb_file_path: str | Path) -> str:
    """The structure a design's interface is measured on: the design, all atoms.

    Not the ``_updated`` view. That file is C-alpha only for a protein target, so
    measuring there made the all-atom contact criterion a CA-CA one and computed
    the burial half from C-alpha spheres -- while the bioinformatics track, asking
    the same question of the same design, read the all-atom file. Two answers to
    one question, differing by side chains that are exactly what an interface is
    made of.

    The two files agree on everything else by construction: same chains, same
    residue numbering, same sequences (checked on CBLN1 production designs,
    136 + 42 residues, identical in both). So the resseqs this yields still key
    ``fix_pos`` in the numbering ProteinMPNN reads.
    """
    name = pdb_name_from_path(pdb_file_path)
    return os.path.join(os.path.dirname(pdb_file_path), name + ".pdb")


def interface_positions(
    structure_path: str,
    binder_chain: str,
    gen_target_chain: list[str],
    is_target_ligand: bool,
    interface_cutoff: float,
) -> tuple[list[int], list[int]]:
    """Binder interface positions on one structure: sequence indices and resseqs.

    The single definition behind both things the cutoff decides -- which
    positions ``mpnn_fixed`` holds fixed, and which residues are counted into
    ``aa_interface_counts``. Module level so the refresh path asks the question
    exactly the way the folding path did.

    Ligand targets keep the all-atom path: protein-interface's radius table is
    protein-only, so every ligand atom comes back without a radius and the burial
    half of the strict criterion would be silently zero for exactly the atoms
    that matter.
    """
    if is_target_ligand:
        idx = get_interface_residues_atomistic(structure_path, binder_chain, interface_cutoff)
        # The atomistic path returns sequence positions only; fix_pos has always
        # assumed resseq == position + 1 for it, and that is unchanged here.
        return idx, [i + 1 for i in idx]
    binder_side, _ = find_interface_residues(
        structure_path,
        binder_chains=[binder_chain],
        target_chains=list(gen_target_chain),
        contact_cutoff=interface_cutoff,
    )
    return sequence_indices(binder_side), sorted(resseqs(binder_side))


@dataclass
class BinderSequenceSet:
    """Everything about a design that does not depend on which folder runs.

    The sequences to evaluate, where the interface is, and what each sequence is
    made of. None of it is a prediction, so none of it should be recomputed once
    per folder -- and while it lived inside ``run_binder_eval`` beside the
    folding dispatch, a second complex folder could only be reached by a separate
    mechanism that had to assemble its own.

    ``aa_stats`` is per sequence type and parallel to that type's entry in
    ``sequences_dict``, and each row records the sequence it was computed from so
    the join is checkable rather than assumed -- see
    ``binder_eval.sequences_for_type``.
    """

    name: str
    binder_chain: str
    gen_target_chain: list
    target_pdb_chain: list
    updated_pdb_path: str
    sequences_dict: dict
    all_sequences: list
    sequence_types_list: list
    all_interface_residues: list
    interface_seq_indices: list
    interface_resseqs: list
    binder_length: int
    aa_stats: dict


def composition_of(sequence: str, interface_residues) -> dict:
    """One sequence's amino-acid composition, whole and at the interface.

    Not a measurement of any structure: it is what the sequence is made of, and
    it is the same whichever folder predicts it.
    """
    all_counts = Counter(sequence)
    if interface_residues is not None and len(interface_residues) > 0:
        # Interface indices match sequence indices; adjust here if that changes.
        interface_counts = Counter("".join(sequence[i] for i in interface_residues))
    else:
        interface_counts = {}
    return {"residue_counts": dict(all_counts), "interface_counts": dict(interface_counts)}


def assemble_binder_sequences(
    pdb_file_path,
    target_pdb_path,
    tmp_path: str = "./tmp/metrics/",
    target_pdb_chain: list = None,
    sequence_types: list = None,
    inverse_folding_model: str = None,
    is_target_ligand: bool = False,
    interface_cutoff: float = None,
    gen_target_chain: list = None,
    binder_chain: str = None,
    num_redesign_seqs: int = None,
    shared_redesign_count: int | None = None,
    fixed_residues_override: list | None = None,
) -> BinderSequenceSet:
    """Build the sequences for one design, folding nothing.

    Split out of :func:`run_binder_eval` so that every complex folder can be
    reached the same way. Inverse folding, the interface query and the
    composition counts are properties of the DESIGN; folding is the only part
    that is a property of a folder, and it is the only part that belongs in a
    per-folder pass.
    """
    name = pdb_name_from_path(pdb_file_path)
    updated_pdb_path = updated_structure_path(pdb_file_path, is_target_ligand)
    # Determine chain IDs
    # sort target_pdb_chain alphabetically to be sure that the first chain is the starting chain
    target_pdb_chain = sorted(target_pdb_chain)
    # If gen_target_chain is not provided, use target_pdb_chain as gen_target_chain
    if gen_target_chain is None:
        gen_target_chain = target_pdb_chain
    starting_chain_id = target_pdb_chain[0]
    # If binder_chain is not provided, use the last chain id in the refolded complex
    if binder_chain is None:
        all_chain_ids = [
            chr(ord(starting_chain_id) + i) for i in range(len(target_pdb_chain) + 1)
        ]  # target chains + binder chain
        binder_chain = all_chain_ids[-1]

    # Check if sequence types are valid
    valid_types = {"mpnn", "mpnn_fixed", "self"}
    invalid_types = set(sequence_types) - valid_types
    if invalid_types:
        raise ValueError(f"Invalid sequence types: {invalid_types}. Valid types are: {valid_types}")

    name = pdb_name_from_path(pdb_file_path)
    updated_pdb_path = updated_structure_path(pdb_file_path, is_target_ligand)
    # Determine chain IDs
    # sort target_pdb_chain alphabetically to be sure that the first chain is the starting chain
    target_pdb_chain = sorted(target_pdb_chain)
    # If gen_target_chain is not provided, use target_pdb_chain as gen_target_chain
    if gen_target_chain is None:
        gen_target_chain = target_pdb_chain
    starting_chain_id = target_pdb_chain[0]
    # If binder_chain is not provided, use the last chain id in the refolded complex
    if binder_chain is None:
        all_chain_ids = [
            chr(ord(starting_chain_id) + i) for i in range(len(target_pdb_chain) + 1)
        ]  # target chains + binder chain
        binder_chain = all_chain_ids[-1]

    if not is_target_ligand:
        ### Updated pdb file has 2 changes:
        ### 1. Replace the sequence of target chains with the original target sequence
        ### 2. Only contains C-alpha atoms
        replace_seq_in_generated_pdb(
            target_pdb_path=target_pdb_path,
            target_pdb_chain=target_pdb_chain,
            gen_pdb_path=pdb_file_path,
            gen_pdb_target_chain=gen_target_chain,
            output_path=updated_pdb_path,
        )

    logger.info(f"Binder chain ID: {binder_chain}")
    logger.info(f"Target chain IDs: {target_pdb_chain}, Is ligand: {is_target_ligand}")

    # Prepare sequences for evaluation
    all_sequences = []
    sequence_types_list = []  # Track the type of each sequence
    sequences_dict = defaultdict(list)
    all_interface_residues = []

    ## Defining target-specific inverse folding arguments
    inverse_folding_model = resolve_inverse_folding_model(inverse_folding_model, is_target_ligand)
    # ProteinMPNN runs CA-only and takes the _updated view; the others are
    # all-atom models and take the design PDB, which carries backbone atoms they
    # can use. Keyed on the resolved model rather than on is_target_ligand, so an
    # explicitly configured ligand_mpnn on a protein target also gets all-atom
    # input instead of a CA-only file it cannot make use of.
    if inverse_folding_model == "protein_mpnn":
        mpnn_input_pdb = updated_pdb_path
    else:
        mpnn_input_pdb = updated_pdb_path.replace("_updated.pdb", ".pdb")

    if num_redesign_seqs is None:
        num_redesign_seqs = 8 if not is_target_ligand else 1
    # Computed once and reused. The four call sites below asked the same question of
    # the same file, and the answer now costs a SASA pass rather than a KD-tree query.
    interface_seq_indices, interface_resseqs = interface_positions(
        interface_structure_path(pdb_file_path), binder_chain, gen_target_chain, is_target_ligand, interface_cutoff
    )

    if "mpnn" in sequence_types:
        logger.info(f"Running inverse folding: {inverse_folding_model}")
        # Use a unique output directory for mpnn
        mpnn_tmp_path = os.path.join(tmp_path, "mpnn")
        os.makedirs(mpnn_tmp_path, exist_ok=True)

        # One set for the whole design, shared with the designability track: both
        # redesign this backbone with the same folder, context, alphabet,
        # temperature and seed, so the second run reproduced the first's work.
        # Generated at the SHARED size and sliced here -- asking for the same
        # count from both sides is what keeps the prefix property confined to one
        # function instead of assumed at every use.
        shared = shared_redesign_set(
            design_name=name,
            mpnn_input_pdb=mpnn_input_pdb,
            out_dir_root=mpnn_tmp_path,
            cache_dir=tmp_path,
            context_chains=complex_mpnn_chains(gen_target_chain, binder_chain),
            chains_to_design=[binder_chain],
            count=max(int(num_redesign_seqs), int(shared_redesign_count or num_redesign_seqs)),
            inverse_folding_model=inverse_folding_model,
        )
        mpnn_sequences = shared[:num_redesign_seqs]
        sequences_dict["mpnn"].extend(mpnn_sequences)
        all_sequences.extend(mpnn_sequences)
        sequence_types_list.extend(["mpnn"] * len(mpnn_sequences))
        all_interface_residues.extend([interface_seq_indices] * len(mpnn_sequences))

    if "mpnn_fixed" in sequence_types:
        # Create a separate directory for mpnn_fixed to avoid conflicts
        mpnn_fixed_tmp_path = os.path.join(tmp_path, "mpnn_fixed")
        os.makedirs(mpnn_fixed_tmp_path, exist_ok=True)

        # Determine fixed positions: use override (e.g. motif residues) or
        # fall back to interface-based detection.
        if fixed_residues_override is not None:
            fix_pos = fixed_residues_override
            logger.info(
                f"Running inverse folding: {inverse_folding_model} with "
                f"overridden fixed residues ({len(fix_pos)} positions)"
            )
        else:
            # Default: fix interface residues (standard binder eval)
            logger.info(
                f"Running inverse folding: {inverse_folding_model} with "
                f"{len(interface_resseqs)} interface residues fixed"
            )
            # Real PDB residue numbers, not position + 1: ProteinMPNN keys fix_pos on the
            # numbering in the file, and the two coincide only while it starts at 1 with
            # no gaps.
            fix_pos = [f"{binder_chain}{r}" for r in interface_resseqs]

        # Composition tracking uses the same set even when fix_pos was overridden.

        mpnn_fixed_sequences = inverse_fold(
            model_type=inverse_folding_model,
            pdb_file_path=mpnn_input_pdb,
            out_dir_root=mpnn_fixed_tmp_path,
            all_chains=complex_mpnn_chains(gen_target_chain, binder_chain),
            pdb_path_chains=[binder_chain],
            fix_pos=fix_pos,
            num_seq_per_target=num_redesign_seqs,
            omit_AAs=MPNN_OMIT_AAS,
            sampling_temp=MPNN_SAMPLING_TEMP,
            seed=mpnn_seed(name, complex_mpnn_chains(gen_target_chain, binder_chain), [binder_chain], variant="fixed"),
            verbose=False,
        )
        sequences_dict["mpnn_fixed"].extend(mpnn_fixed_sequences)
        all_sequences.extend(mpnn_fixed_sequences)
        sequence_types_list.extend(["mpnn_fixed"] * len(mpnn_fixed_sequences))
        all_interface_residues.extend([interface_seq_indices] * len(mpnn_fixed_sequences))

    if "self" in sequence_types:
        logger.info("Running inverse folding: self-generated sequences")
        self_sequence = {"seq": extract_seq_from_pdb(pdb_file_path, chain_id=binder_chain)}
        sequences_dict["self"].append(self_sequence)
        all_sequences.append(self_sequence)
        sequence_types_list.append("self")
        all_interface_residues.append(interface_seq_indices)

    if not all_sequences:
        raise ValueError("No sequences to evaluate. Please specify at least one sequence type.")
    binder_length = len(all_sequences[0]["seq"])

    logger.info("Inverse folding finished")


    aa_stats: dict = {t: [] for t in set(sequence_types_list)}
    for sequence_entry, seq_type, interface_residues in zip(
        all_sequences, sequence_types_list, all_interface_residues, strict=False
    ):
        sequence = sequence_entry["seq"]
        aa_stats[seq_type].append(
            {
                # The sequence this row's metrics were computed from. Downstream
                # code used to recover it by indexing sequences_dict in parallel,
                # which is only correct while both lists stay in append order and
                # fails silently otherwise. Recording it makes that join checkable.
                "sequence": sequence,
                **composition_of(sequence, interface_residues),
                "binder_length": binder_length,
            }
        )

    return BinderSequenceSet(
        name=name,
        binder_chain=binder_chain,
        gen_target_chain=gen_target_chain,
        target_pdb_chain=target_pdb_chain,
        updated_pdb_path=updated_pdb_path,
        sequences_dict=dict(sequences_dict),
        all_sequences=all_sequences,
        sequence_types_list=sequence_types_list,
        all_interface_residues=all_interface_residues,
        interface_seq_indices=interface_seq_indices,
        interface_resseqs=interface_resseqs,
        binder_length=binder_length,
        aa_stats=aa_stats,
    )


def atomarray_to_atom37_coords(
    atomarray: AtomArray,
    chains: list[str],
    return_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Convert an atomarray (selected chains) to atom37 coordinates.

    With *return_mask*, also returns which of the 37 slots each residue actually
    fills. The encoding pads every residue to 37 and the absent slots are NaN,
    turned into the origin here -- so an all-atom comparison that does not carry
    this mask is comparing about four phantom atoms at (0, 0, 0) for every real
    one.
    """
    for chain in chains:
        assert chain in atomarray.chain_id, f"Chain {chain} not found in atomarray.chain_id"
    masks = [atomarray.chain_id == chain for chain in chains]
    # create a mask that's the "or" of all boolean masks in the "masks" list
    if len(masks) == 0:
        subset_atomarray = atomarray
    else:
        mask = masks[0].copy()
        for m in masks[1:]:
            mask = mask | m
        subset_atomarray = atomarray[mask]
    encoded = atom_array_to_encoding(subset_atomarray, encoding=AF2_ATOM37_ENCODING)
    atom37_coords = torch.tensor(encoded["xyz"], dtype=torch.float32).nan_to_num(0.0)
    if return_mask:
        return atom37_coords, torch.tensor(encoded["mask"], dtype=torch.bool)
    return atom37_coords


def calculate_prot_prot_binder_rmsd(
    refolded_complex: AtomArray,
    gen_complex: AtomArray,
    label: str = "",
) -> dict[str, float]:
    """
    Calculate the RMSD related metrics for protein-protein complex.

    Computes CA, backbone-3, backbone-3+O, and all-atom RMSD modes for both
    the binder and the full complex, using mode-suffixed key names that are
    consistent with ``calculate_ligand_binder_rmsd``.

    Args:
        refolded_complex: The refolded complex.
        gen_complex: The generated complex.
        label: Identifier for log messages (e.g. "mpnn_fixed_seq_1").
    Returns:
        A dictionary containing the RMSD related metrics.
        Keys include (for each of binder / complex):
            "binder_scRMSD_ca", "binder_scRMSD_bb3", "binder_scRMSD_bb3o",
            "binder_scRMSD_allatom", "complex_scRMSD_ca", etc.
        Mode-suffixed only: the unsuffixed aliases are gone, because the
        advisory backends never emitted them and a column set that depends on
        which folder produced it is the distinction the migration removed.
    """
    # Set the occupancy to 1.0 for all chains
    refolded_complex.set_annotation(
        "occupancy",
        np.array(
            [1.0 for chain in refolded_complex.chain_id],
            dtype=refolded_complex.coord.dtype,
        ),
    )
    gen_complex.set_annotation(
        "occupancy",
        np.array([1.0 for chain in gen_complex.chain_id], dtype=gen_complex.coord.dtype),
    )

    # We always assume the last chain is the binder
    refolded_complex_chains = sorted(list(set(refolded_complex.chain_id)))
    gen_complex_chains = sorted(list(set(gen_complex.chain_id)))
    refolded_binder_chain = refolded_complex_chains[-1]
    gen_binder_chain = gen_complex_chains[-1]

    # Get the atom coordinates of the complex and the binder in the refolded complex
    refolded_complex_coors = atomarray_to_atom37_coords(refolded_complex, refolded_complex_chains)
    refolded_binder_coors, refolded_binder_mask = atomarray_to_atom37_coords(
        refolded_complex, [refolded_binder_chain], return_mask=True
    )
    # Get the atom coordinates of the generated complex and binder
    gen_complex_coors = atomarray_to_atom37_coords(gen_complex, gen_complex_chains)
    gen_binder_coors, gen_binder_mask = atomarray_to_atom37_coords(
        gen_complex, [gen_binder_chain], return_mask=True
    )

    # Compute binder RMSD in all modes
    binder_scRMSD_ca = rmsd_metric(gen_binder_coors, refolded_binder_coors, mode="ca")
    # The atoms both structures actually have, the way the designability track
    # builds it (gen_mask * rec_mask). This was torch.ones: with 69 residues that
    # is 2553 slots against 540 real atoms, so all-atom RMSD was fitting ~2000
    # phantom atoms parked at the origin, far from the protein. The error scales
    # with how much the two structures really differ, which is what made it look
    # sane where it mattered least -- on one EFNB3 design the AF2 number read 0.475
    # against a true 1.006, while ESMFold2's read 11.505 against a true 1.177.
    mask_binder = gen_binder_mask & refolded_binder_mask
    binder_scRMSD_allatom = rmsd_metric(
        gen_binder_coors,
        refolded_binder_coors,
        mode="all_atom",
        mask_atom_37=mask_binder,
    )
    binder_scRMSD_bb3o = rmsd_metric(gen_binder_coors, refolded_binder_coors, mode="bb3o", mask_atom_37=mask_binder)
    binder_scRMSD_bb3 = rmsd_metric(gen_binder_coors, refolded_binder_coors, mode="bb3", mask_atom_37=mask_binder)

    # Compute complex RMSD (CA only — full-complex all-atom is rarely useful)
    complex_scRMSD_ca = rmsd_metric(gen_complex_coors, refolded_complex_coors, mode="ca")

    # Binder RMSD after superimposing on the TARGET.
    #
    # The third of three ways to superimpose a refolded complex, and the only one
    # that sees placement: binder-aligned RMSD is blind to where the binder sits
    # because it fits the binder onto itself, and complex-aligned RMSD dilutes
    # placement error across whichever chain has more residues. Fitting on the
    # target -- effectively the same structure in both complexes -- leaves the
    # binder's deviation as fold *and* placement together. Subtracting
    # binder_scRMSD_ca is what isolates placement; this number alone does not.
    #
    # Emitted, not gated: there is no distribution to pick a threshold from yet.
    binder_scRMSD_target_aligned_ca = float("nan")
    gen_target_chains = [c for c in gen_complex_chains if c != gen_binder_chain]
    refolded_target_chains = [c for c in refolded_complex_chains if c != refolded_binder_chain]
    if gen_target_chains and refolded_target_chains:
        gen_target_coors = atomarray_to_atom37_coords(gen_complex, gen_target_chains)
        refolded_target_coors = atomarray_to_atom37_coords(refolded_complex, refolded_target_chains)
        n_target, n_binder = gen_target_coors.shape[0], gen_binder_coors.shape[0]
        if refolded_target_coors.shape[0] == n_target and refolded_binder_coors.shape[0] == n_binder:
            # Concatenated explicitly rather than sliced out of the complex arrays,
            # so the target/binder split is stated here instead of resting on the
            # chain ordering that produced them.
            gen_ca = torch.cat([gen_target_coors[:, 1, :], gen_binder_coors[:, 1, :]], dim=0)
            refolded_ca = torch.cat([refolded_target_coors[:, 1, :], refolded_binder_coors[:, 1, :]], dim=0)
            target_mask = torch.zeros(n_target + n_binder, dtype=torch.bool)
            target_mask[:n_target] = True
            aligned_gen, centered_refolded = kabsch_align_ligand(
                gen_ca.unsqueeze(0),
                refolded_ca.unsqueeze(0),
                mask=target_mask.unsqueeze(0),
            )
            deviation = aligned_gen[0][~target_mask] - centered_refolded[0][~target_mask]
            binder_scRMSD_target_aligned_ca = deviation.pow(2).sum(dim=-1).mean().sqrt().item()
        else:
            logger.warning(
                f"{'[' + label + '] ' if label else ''}Target-aligned binder RMSD skipped: residue counts "
                f"differ between generated and refolded complex "
                f"(target {n_target} vs {refolded_target_coors.shape[0]}, "
                f"binder {n_binder} vs {refolded_binder_coors.shape[0]})"
            )

    tag = f"[{label}] " if label else ""
    logger.info(
        f"{tag}binder scRMSD — CA: {binder_scRMSD_ca:.4f}, "
        f"bb3: {binder_scRMSD_bb3:.4f}, bb3o: {binder_scRMSD_bb3o:.4f}, "
        f"all-atom: {binder_scRMSD_allatom:.4f}"
    )
    logger.info(
        f"{tag}complex scRMSD CA: {complex_scRMSD_ca:.4f}, "
        f"binder scRMSD CA on target alignment: {binder_scRMSD_target_aligned_ca:.4f}"
    )

    rmsd_result = {
        # Mode-suffixed keys (consistent with calculate_ligand_binder_rmsd)
        "binder_scRMSD_ca": binder_scRMSD_ca,
        "binder_scRMSD_bb3": binder_scRMSD_bb3,
        "binder_scRMSD_bb3o": binder_scRMSD_bb3o,
        "binder_scRMSD_allatom": binder_scRMSD_allatom,
        "complex_scRMSD_ca": complex_scRMSD_ca,
        # Aligned on the target, measured over the binder -- see above.
        "binder_scRMSD_target_aligned_ca": binder_scRMSD_target_aligned_ca,
        # No unsuffixed "binder_scRMSD"/"complex_scRMSD" aliases. They equalled
        # the CA values and existed for frames written before the modes were
        # named -- and only this side ever emitted them, so a campaign's af2
        # columns carried two names for one number while its esmfold2 columns
        # carried one. That is the last of the primary/advisory distinction the
        # folder migration set out to remove: a folder's column set must not
        # depend on which folder it is. consensus_folding.CONSENSUS_RMSD_SUFFIXES
        # already refused them for the same reason.
    }

    return rmsd_result


def calculate_ligand_binder_rmsd(
    refolded_complex: AtomArray,
    gen_complex: AtomArray,
    ligand_chain_id: str = "A",
    binder_chain_id: str = "B",
    label: str = "",
) -> dict[str, float]:
    """
    Calculate the RMSD related metrics for ligand-binder complex. Now we only support 1 ligand chain and 1 binder chain.
    Args:
        refolded_complex: The refolded complex.
        gen_complex: The generated complex.
        ligand_chain_id: The ligand chain id.
        binder_chain_id: The binder chain id.
    Returns:
        A dictionary containing the RMSD related metrics.
        Keys:
            "binder_scRMSD_ca": The RMSD between the generated binder and the refolded binder based on CA atoms.
            "binder_scRMSD_allatom": The RMSD between the generated binder and the refolded binder based on all atoms.
            "ligand_scRMSD": The RMSD between the ligand in the generated complex and the ligand in the refolded complex.
            "ligand_scRMSD_aligned_ca": Align complexes based on binder backbone atoms, then compute RMSD between the ligands in the aligned complexes.
            "ligand_scRMSD_aligned_allatom": Align complexes based on binder all-atoms, then compute RMSD between the ligands in the aligned complexes.
    """
    tag = f"[{label}] " if label else ""

    # Sort the refolded complex and gen complex by chain id
    refolded_complex = sort_AtomArray_by_chain_id(refolded_complex)
    gen_complex = sort_AtomArray_by_chain_id(gen_complex)
    # Set the occupancy to 1.0 for all chains
    refolded_complex.set_annotation(
        "occupancy",
        np.array(
            [1.0 for chain in refolded_complex.chain_id],
            dtype=refolded_complex.coord.dtype,
        ),
    )
    gen_complex.set_annotation(
        "occupancy",
        np.array([1.0 for chain in gen_complex.chain_id], dtype=gen_complex.coord.dtype),
    )

    refolded_binder_coords, refolded_binder_mask = atomarray_to_atom37_coords(
        refolded_complex, [binder_chain_id], return_mask=True
    )

    gen_binder_coords, gen_binder_mask = atomarray_to_atom37_coords(
        gen_complex, [binder_chain_id], return_mask=True
    )

    # Get the protein binder RMSD (both all-atom and CA)
    binder_scRMSD_ca = rmsd_metric(gen_binder_coords, refolded_binder_coords, mode="ca")
    # Real atoms only -- see calculate_prot_prot_binder_rmsd for what torch.ones
    # was measuring here.
    mask_atom_37 = gen_binder_mask & refolded_binder_mask
    binder_scRMSD_allatom = rmsd_metric(
        gen_binder_coords,
        refolded_binder_coords,
        mode="all_atom",
        mask_atom_37=mask_atom_37,
    )
    binder_scRMSD_bb3o = rmsd_metric(
        gen_binder_coords,
        refolded_binder_coords,
        mode="bb3o",
        mask_atom_37=mask_atom_37,
    )
    binder_scRMSD_bb3 = rmsd_metric(
        gen_binder_coords,
        refolded_binder_coords,
        mode="bb3",
        mask_atom_37=mask_atom_37,
    )
    logger.info(
        f"{tag}binder scRMSD — CA: {binder_scRMSD_ca:.4f}, "
        f"bb3: {binder_scRMSD_bb3:.4f}, bb3o: {binder_scRMSD_bb3o:.4f}, all-atom: {binder_scRMSD_allatom:.4f}"
    )

    # Calculate ligand RMSD if applicable ligand_rmsd
    try:
        ### Firstly remove OXT atoms in the protein binder for both refolded and generated complexes
        gen_complex = gen_complex[
            ((gen_complex.chain_id == binder_chain_id) & (gen_complex.atom_name != "OXT"))
            | (gen_complex.chain_id == ligand_chain_id)
        ]
        refolded_complex = refolded_complex[
            ((refolded_complex.chain_id == binder_chain_id) & (refolded_complex.atom_name != "OXT"))
            | (refolded_complex.chain_id == ligand_chain_id)
        ]

        ### For ligand RMSD
        ligand_atoms = torch.tensor(
            refolded_complex[refolded_complex.chain_id == ligand_chain_id].coord,
            dtype=torch.float32,
        )
        gen_ligand_atoms = torch.tensor(
            gen_complex[gen_complex.chain_id == ligand_chain_id].coord,
            dtype=torch.float32,
        )
        coors_1, coors_2 = kabsch_align_ind(ligand_atoms, gen_ligand_atoms, ret_both=True)
        sq_err = (coors_1 - coors_2) ** 2
        ligand_scRMSD = sq_err.sum(dim=-1).mean().sqrt().item()
        logger.info(f"{tag}ligand scRMSD: {ligand_scRMSD:.4f}")

        ### For ligand RMSD in aligned complexes based on backbone atoms
        ## Select the ligand and binder backbone atoms from the refolded complex
        refolded_complex_bb = refolded_complex[
            ((refolded_complex.chain_id == binder_chain_id) & (refolded_complex.atom_name == "CA"))
            | (refolded_complex.chain_id == ligand_chain_id)
        ]
        gen_complex_bb = gen_complex[
            ((gen_complex.chain_id == binder_chain_id) & (gen_complex.atom_name == "CA"))
            | (gen_complex.chain_id == ligand_chain_id)
        ]
        refolded_complex_bb_coord = torch.tensor(
            refolded_complex_bb.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        gen_complex_bb_coord = torch.tensor(
            gen_complex_bb.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        binder_mask = torch.tensor(
            refolded_complex_bb.chain_id == binder_chain_id,
            dtype=torch.bool,
        )
        ## Align the generated complex to the refolded complex, based only on the binder backbone atoms
        aligned_binder_centered_gen_complex, refolded_binder_centered_complex = kabsch_align_ligand(
            gen_complex_bb_coord.unsqueeze(0),  # [b, n, 3], b=1
            refolded_complex_bb_coord.unsqueeze(0),  # [b, n, 3]
            mask=binder_mask.unsqueeze(0),  # [b, n]
        )
        aligned_binder_centered_gen_complex, refolded_binder_centered_complex = (
            aligned_binder_centered_gen_complex[0],
            refolded_binder_centered_complex[0],
        )
        ## Compute the rmsd between the ligands in the aligned complexes.
        bb_aligned_ligand = aligned_binder_centered_gen_complex[~binder_mask]
        bb_aligned_refolded_ligand = refolded_binder_centered_complex[~binder_mask]
        sq_err = (bb_aligned_ligand - bb_aligned_refolded_ligand) ** 2
        ligand_scRMSD_aligned_bb = sq_err.sum(dim=-1).mean().sqrt().item()
        logger.info(f"{tag}ligand scRMSD aligned CA backbone: {ligand_scRMSD_aligned_bb:.4f}")

        ### For ligand RMSD in aligned complexes based on all atoms
        refolded_complex_allatom_coord = torch.tensor(
            refolded_complex.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        gen_complex_allatom_coord = torch.tensor(
            gen_complex.coord,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        binder_mask = torch.tensor(
            refolded_complex.chain_id == binder_chain_id,
            dtype=torch.bool,
        )
        ## If number of atoms are different (sequence different after redesign), use the aligned backbone rmsd
        ligand_scRMSD_aligned_allatom = ligand_scRMSD_aligned_bb
        logger.debug(
            f"{tag}Atom count: gen={gen_complex_allatom_coord.shape[0]}, "
            f"refolded={refolded_complex_allatom_coord.shape[0]}"
        )
        if gen_complex_allatom_coord.shape[0] == refolded_complex_allatom_coord.shape[0]:
            ## Align the generated complex to the refolded complex, based only on the binder all atoms
            aligned_binder_centered_gen_complex, refolded_binder_centered_complex = kabsch_align_ligand(
                gen_complex_allatom_coord.unsqueeze(0),  # [b, n, 3], b=1
                refolded_complex_allatom_coord.unsqueeze(0),  # [b, n, 3]
                mask=binder_mask.unsqueeze(0),  # [b, n]
            )
            aligned_binder_centered_gen_complex, refolded_binder_centered_complex = (
                aligned_binder_centered_gen_complex[0],
                refolded_binder_centered_complex[0],
            )
            ## Compute the rmsd between the ligands in the aligned complexes.
            allatom_aligned_ligand = aligned_binder_centered_gen_complex[~binder_mask]
            allatom_aligned_refolded_ligand = refolded_binder_centered_complex[~binder_mask]
            sq_err = (allatom_aligned_ligand - allatom_aligned_refolded_ligand) ** 2
            ligand_scRMSD_aligned_allatom = sq_err.sum(dim=-1).mean().sqrt().item()
            logger.info(f"{tag}ligand scRMSD aligned all atoms: {ligand_scRMSD_aligned_allatom:.4f}")

    except Exception as e:
        logger.error(f"{tag}Could not calculate ligand RMSD: {e}")
        ligand_scRMSD = float("inf")
        ligand_scRMSD_aligned_allatom = float("inf")
        ligand_scRMSD_aligned_bb = float("inf")

    rmsd_result = {
        "binder_scRMSD_ca": binder_scRMSD_ca,
        "binder_scRMSD_bb3": binder_scRMSD_bb3,
        "binder_scRMSD_bb3o": binder_scRMSD_bb3o,
        "binder_scRMSD_allatom": binder_scRMSD_allatom,
        "ligand_scRMSD": ligand_scRMSD,
        "ligand_scRMSD_aligned_allatom": ligand_scRMSD_aligned_allatom,
        "ligand_scRMSD_aligned_ca": ligand_scRMSD_aligned_bb,
    }

    return rmsd_result
