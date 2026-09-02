import os
import shutil
import subprocess
from typing import Literal

import torch
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

hf_logging.set_verbosity_error()


def create_individual_fasta_files(
    sequences: list[str],
    output_dir: str,
    format_type: Literal["simple"] = "simple",
    name_prefix: str = "seq",
) -> str:
    """
    Creates individual FASTA files for each sequence with appropriate headers for different folding models.

    Args:
        sequences: List of protein sequences
        output_dir: Directory where individual FASTA files will be written
        format_type: Header format to use:
            - "simple": >seq_1, >seq_2, ... (for ESMFold/ColabFold)
        name_prefix: Prefix for sequence names

    Returns:
        Path to the directory containing the individual FASTA files
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, seq in enumerate(sequences):
        seq_name = f"{name_prefix}_{i + 1}"
        fasta_path = os.path.join(output_dir, f"{seq_name}.fasta")

        if format_type == "simple":
            header = f">{seq_name}"
        else:
            raise ValueError(f"Unknown format_type: {format_type}")

        with open(fasta_path, "w") as f:
            f.write(f"{header}\n{seq}\n")

    return output_dir


def run_esmfold(
    sequences: list[str],
    path_to_esmfold_out: str,
    name: str,
    suffix: str,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
) -> list[str]:
    """
    Runs ESMFold on sequences and stores results as PDB files.

    For now, runs with a single GPU, though not a big deal if we parallelie jobs (easily
    done with our inference pipeline).

    Args:
        sequences: List of protein sequences to predict
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs
        name: name to use when storing
        suffix: to use as suffix when storing files
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep individual output directories after processing.
            If False (default), temporary directories are deleted to save space.

    Returns:
        List of paths (list of str) to PDB files
    """
    is_cluster_run = os.environ.get("SLURM_JOB_ID") is not None

    # Use provided cache_dir or fallback to environment/cluster logic
    final_cache_dir = cache_dir
    if final_cache_dir is None and is_cluster_run:
        final_cache_dir = os.environ.get("CACHE_DIR")
    if final_cache_dir:
        final_cache_dir = os.path.expanduser(final_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=final_cache_dir)
    esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir=final_cache_dir)
    esm_model = esm_model.cuda()

    # Run ESMFold
    list_of_strings_pdb = []
    if len(sequences) == 8:
        max_nres = max([len(x) for x in sequences])
        if max_nres > 700:
            batch_size = 1
            num_batches = 8
        elif max_nres > 500:
            batch_size = 2
            num_batches = 4
        elif max_nres > 200:
            batch_size = 4
            num_batches = 2
        else:
            batch_size = 8
            num_batches = 1
    elif len(sequences) == 1:
        batch_size = 8
        num_batches = 1
    else:
        raise OSError("We can only run ESMFold with 1 or 8 sequences... We should fix this...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        inputs = tokenizer(
            sequences[start_idx:end_idx],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {k: inputs[k].cuda() for k in inputs}

        with torch.no_grad():
            _outputs = esm_model(**inputs)

        _list_of_strings_pdb = _convert_esm_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)

    # Create out directory if not there
    if not os.path.exists(path_to_esmfold_out):
        os.makedirs(path_to_esmfold_out)

    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        fname = f"esm_{i + 1}.pdb_esm_{suffix}"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
            out_esm_paths.append(fdir)

    if not keep_outputs:
        # Clean up individual FASTA files directory
        try:
            shutil.rmtree(os.path.dirname(os.path.dirname(fdir)))
        except Exception as e:
            logger.warning(f"Could not clean up FASTA directory: {e}")

    return out_esm_paths


def folding_model_identity(model: str) -> str:
    """Which weights a monomer folding backend will actually use.

    Goes in the refold cache key so switching checkpoints recomputes instead of
    serving structures from a different model. esmfold2 resolves through the
    shared loader, which honours ESMFOLD2_MONOMER_MODEL.
    """
    if model == "esmfold2":
        from proteinfoundation.metrics.esmfold2_loader import monomer_model_id

        return monomer_model_id()
    if model == "esmfold":
        return "facebook/esmfold_v1"
    return model


def run_esmfold2(
    sequences: list[str],
    path_to_esmfold_out: str,
    name: str,
    suffix: str,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
    seed: int | None = None,
) -> list[str]:
    """Runs ESMFold2 on sequences and stores results as PDB files.

    Same contract as :func:`run_esmfold` -- one PDB per input sequence, returned
    in input order -- so ``monomer_eval`` can dispatch to either. Differences
    worth knowing:

    * Single-chain, single-sequence. ESMFold2 accepts an MSA, but a de novo
      binder has no meaningful alignment, so this path never passes one. The
      Fast checkpoint is the default here for the same reason.
    * No 1-or-8 sequence restriction. ``run_esmfold`` raises OSError for any
      other count because its batch schedule is hardcoded; ``fold_batch``
      length-buckets to a token budget and retries on OOM, so any count works.
    * The model is cached per process by ``esmfold2_loader``, not reloaded per
      call.

    ``cache_dir`` is accepted for signature compatibility and ignored: ESMFold2
    weights resolve through HF_HOME/HF_HUB_CACHE, and an explicit cache_dir would
    override the variable the shared caches are keyed on.

    Untested against real weights, which live in a private repo.
    """
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

    from proteinfoundation.metrics.esmfold2_loader import deterministic_seed, load_esmfold2, monomer_model_id

    if not sequences:
        return []
    if cache_dir:
        logger.debug("run_esmfold2 ignores cache_dir; ESMFold2 resolves weights via HF_HOME/HF_HUB_CACHE")

    model_id = monomer_model_id()
    model = load_esmfold2(model_id)
    builder = ESMFold2InputBuilder()

    inputs = [StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq, msa=None)]) for seq in sequences]
    # fold_batch takes one seed for the whole call, not one per sequence, so this
    # is keyed on the design and the sequence set that is being folded. Same
    # design, same sequences, same structures -- which is what makes a resumed
    # run agree with the cached values it reuses. ESMFold2 is a diffusion
    # sampler; unseeded, two runs disagree.
    if seed is None:
        seed = deterministic_seed(name, suffix, *sequences)
    logger.info(f"Running ESMFold2 ({model_id}) on {len(sequences)} sequence(s) for {name} (seed {seed})")
    results = builder.fold_batch(model, inputs, seed=seed)

    os.makedirs(path_to_esmfold_out, exist_ok=True)
    out_paths = []
    for i, result in enumerate(results):
        # fold_batch returns one entry per input, but an entry is itself a list
        # when num_diffusion_samples > 1. Take the first sample: this path wants a
        # structure to measure scRMSD against, not a ranked best-of-N.
        single = result[0] if isinstance(result, list) else result
        if single is None:
            logger.warning(f"ESMFold2 returned nothing for sequence {i + 1}/{len(sequences)}; skipping")
            continue
        # Filename pattern mirrors run_esmfold's so anything downstream that
        # inspects names sees the same shape. Outputs land in a per-model
        # directory, so there is no collision between backends.
        # The seed goes in the name: each seed folds a different structure, and
        # without it the last overwrites the rest and answers for all of them.
        fname = f"esm_{i + 1}_seed{seed}.pdb_esm_{suffix}"
        fdir = os.path.join(path_to_esmfold_out, fname)
        single.complex.to_protein_complex().to_pdb(fdir)
        out_paths.append(fdir)

    if not keep_outputs:
        # Deliberately not deleting anything. run_esmfold removes two directory
        # levels above its own output here, which would take the paths it just
        # returned with it; monomer_eval passes keep_outputs=True for folding
        # backends anyway.
        logger.debug("run_esmfold2 keeps its outputs; the returned paths must stay readable")
    return out_paths


# I got this function from hugging face's ESM notebook example
def _convert_esm_outputs_to_pdb(outputs) -> list[str]:
    """Takes ESMFold outputs and converts them to a list of PDBs (as strings)."""
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def run_colabfold(
    sequences: list[str],
    path_to_colabfold_out: str,
    suffix: str = "",
    relax: bool = False,
    cache_dir: str | None = None,
    keep_outputs: bool = False,
) -> list[str]:
    """
    Runs ColabFold batch on sequences using individual FASTA files and returns paths to top-ranked PDB files.

    Args:
        sequences (List[str]): List of protein sequences to predict
        path_to_colabfold_out (str): Output directory path for ColabFold results.
        suffix (str): Suffix to add to output files to indicate source (e.g. "mpnn" or "pdb")
        relax (bool): whether to relax the structure afterwards
        cache_dir (Optional[str]): Cache directory for model weights
        keep_outputs (bool): Whether to keep individual output directories after processing.
            If False (default), temporary directories are deleted to save space.

    Returns:
        list[str]: Paths to top-ranked PDB files by pLDDT (rank_001) in the order of input sequences.

    Raises:
        RuntimeError: If ColabFold command fails.
    """
    # Create output directory if it doesn't exist
    os.makedirs(path_to_colabfold_out, exist_ok=True)
    os.makedirs(os.path.join(path_to_colabfold_out, "structures"), exist_ok=True)

    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.environ["XDG_CACHE_HOME"] = cache_dir

    # Create individual FASTA files using the unified function
    fasta_dir = os.path.join(path_to_colabfold_out, "individual_fastas")
    create_individual_fasta_files(sequences, fasta_dir, format_type="simple")

    # Get sequence names for output parsing
    seq_names = [f"seq_{i + 1}" for i in range(len(sequences))]

    # Run ColabFold batch on the directory containing individual FASTA files
    batch_command = (
        f"colabfold_batch {fasta_dir} {path_to_colabfold_out}/structures --msa-mode single_sequence --data {cache_dir}"
    )
    if relax:
        batch_command = batch_command + " --num-relax 1 --use-gpu-relax"

    try:
        result = subprocess.run(batch_command, shell=True, check=True)
        if result.returncode != 0:
            logger.error(f"ColabFold command failed with error: {result.stderr}")
            raise RuntimeError(f"ColabFold command failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"ColabFold command failed with error: {e.stderr}")
        raise RuntimeError(f"ColabFold command failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error running ColabFold: {e!s}")
        raise RuntimeError(f"Unexpected error running ColabFold: {e!s}")

    # Collect PDB file paths for rank_001 models in the original sequence order
    pdb_file_paths = []
    for seq_name in seq_names:
        found_pdb = False
        for filename in os.listdir(f"{path_to_colabfold_out}/structures"):
            if filename.startswith(seq_name) and "rank_001" in filename and filename.endswith(".pdb"):
                pdb_path = f"{path_to_colabfold_out}/structures/{filename}"
                if suffix:
                    # Add suffix to the filename
                    new_path = pdb_path.replace(".pdb", f"_{suffix}.pdb")
                    shutil.copy(pdb_path, new_path)
                    pdb_file_paths.append(new_path)
                else:
                    pdb_file_paths.append(pdb_path)
                found_pdb = True
                break

        if not found_pdb:
            logger.warning(f"No rank_001 PDB file found for sequence: {seq_name}")
            pdb_file_paths.append(None)

    # Clean up individual FASTA files directory
    if not keep_outputs:
        try:
            shutil.rmtree(fasta_dir)
        except Exception as e:
            logger.warning(f"Could not remove FASTA directory: {e}")

    return pdb_file_paths
