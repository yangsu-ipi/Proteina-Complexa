import glob
import json
import os
import shutil
import subprocess
from typing import Literal

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers import logging as hf_logging
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from proteinfoundation.metrics.pae_store import save_pae

hf_logging.set_verbosity_error()


# =============================================================================
# Fold confidence sidecars
# =============================================================================
#
# pLDDT survives a fold because the backends write it into the B-factor column,
# so anything holding the PDB can read it back. pTM and PAE do not: they exist
# only in the folder's own output, and once the run ends the structure on disk
# cannot answer for them. A monomer fold was therefore recorded as pLDDT and
# nothing else, while the complex track reported the whole confidence family.
#
# So each backend writes what it knows beside the structure it wrote. A tiny
# JSON, not a second PDB convention: it rides along with the kept structure
# through the cache, the resume path and the re-derivation path without any of
# them being taught about it, and a fold that predates it simply has no sidecar,
# which every reader treats as unmeasured rather than as zero.


CONFIDENCE_SIDECAR_SUFFIX = ".confidence.json"


def confidence_sidecar_path(pdb_path: str) -> str:
    """Where one folded structure's folder-reported confidence lives.

    Appended rather than substituted for the extension: these files are named
    ``esm_1_seed7.pdb_esm_apo_mpnn``, so there is no extension to replace.
    """
    return pdb_path + CONFIDENCE_SIDECAR_SUFFIX


def write_fold_confidence(pdb_path: str, ptm=None, pae=None) -> None:
    """Record pTM and mean PAE for one folded structure.

    *pae* is the full predicted-aligned-error matrix in Angstroms, as every
    backend reports it; it is stored reduced to its symmetrised mean and divided
    by :data:`PAE_MAX_BIN`, which is the scale every other PAE column in Complexa
    carries. A monomer has one chain, so there is no interface block to take and
    the whole matrix is the answer.

    Never raises. A sidecar that cannot be written costs a column, and a fold is
    far more expensive than the number it failed to record.
    """
    from proteinfoundation.metrics.ensembling import PAE_MAX_BIN

    payload: dict[str, float] = {}
    try:
        if ptm is not None:
            payload["pTM"] = float(ptm)
        if pae is not None:
            array = np.asarray(pae.detach().cpu() if hasattr(pae, "detach") else pae, dtype=float)
            if array.ndim == 2 and array.size:
                payload["pAE"] = float(((array + array.T) / 2).mean()) / PAE_MAX_BIN
        if not payload:
            return
        with open(confidence_sidecar_path(pdb_path), "w") as handle:
            json.dump(payload, handle)
    except Exception as exc:
        logger.warning(f"Could not record fold confidence for {pdb_path}: {exc}")


def read_fold_confidence(pdb_path: str) -> dict[str, float]:
    """What the folder said about one structure, or ``{}`` if it did not say.

    Absent is unmeasured, not zero: folds cached before sidecars existed, and
    backends that report no PAE, both land here.
    """
    path = confidence_sidecar_path(pdb_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            loaded = json.load(handle)
        return {k: float(v) for k, v in loaded.items() if isinstance(v, (int, float))}
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable confidence sidecar {path}: {exc}")
        return {}


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
    # Positionally aligned with list_of_strings_pdb, so a batch that reports
    # nothing contributes Nones rather than shifting the ones after it.
    confidences: list[dict] = []
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
        confidences.extend(_batch_confidences(_outputs, len(_list_of_strings_pdb)))

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
        recorded = confidences[i] if i < len(confidences) else {}
        if recorded:
            write_fold_confidence(fdir, ptm=recorded.get("ptm"), pae=recorded.get("pae"))
            save_pae(
                fdir,
                recorded.get("pae"),
                chain_lengths=[len(sequences[i])],
                backend="esmfold",
                model="facebook/esmfold_v1",
            )

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
) -> list[str | None]:
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
            # None, not dropped. The caller zips these paths against the input
            # sequences by position, so omitting a failure shifted every later
            # structure onto the wrong sequence -- silently, and only for the
            # designs unlucky enough to have one fail.
            logger.warning(f"ESMFold2 returned nothing for sequence {i + 1}/{len(sequences)}")
            out_paths.append(None)
            continue
        # Filename pattern mirrors run_esmfold's so anything downstream that
        # inspects names sees the same shape. Outputs land in a per-model
        # directory, so there is no collision between backends.
        # The seed goes in the name: each seed folds a different structure, and
        # without it the last overwrites the rest and answers for all of them.
        fname = f"esm_{i + 1}_seed{seed}.pdb_esm_{suffix}"
        fdir = os.path.join(path_to_esmfold_out, fname)
        single.complex.to_protein_complex().to_pdb(fdir)
        # What the folder knows and the PDB cannot carry. Same fields the advisory
        # complex path reads off a MolecularComplexResult.
        write_fold_confidence(fdir, ptm=getattr(single, "ptm", None), pae=getattr(single, "pae", None))
        # And the matrix itself, so a later change to what is computed from it is
        # a re-read rather than a refold.
        save_pae(
            fdir,
            getattr(single, "pae", None),
            chain_lengths=[len(sequences[i])],
            backend="esmfold2",
            model=model_id,
            seed=seed,
        )
        out_paths.append(fdir)

    if not keep_outputs:
        # Deliberately not deleting anything. run_esmfold removes two directory
        # levels above its own output here, which would take the paths it just
        # returned with it; monomer_eval passes keep_outputs=True for folding
        # backends anyway.
        logger.debug("run_esmfold2 keeps its outputs; the returned paths must stay readable")
    return out_paths


# I got this function from hugging face's ESM notebook example
def _batch_confidences(outputs, count: int) -> list[dict]:
    """Per-sequence pTM and PAE out of one ESMFold batch, where they are there.

    Guarded on the leading dimension rather than trusting the field to be
    batched: a scalar pTM for a batch of four says nothing about which of the
    four it describes, and a confidence attributed to the wrong sequence is
    worse than an absent one. Both fields are optional on the HF output, so an
    entry is simply ``{}`` when the model did not report them.
    """
    picked: list[dict] = [{} for _ in range(count)]
    for field, key in (("ptm", "ptm"), ("predicted_aligned_error", "pae")):
        value = outputs.get(field) if hasattr(outputs, "get") else getattr(outputs, field, None)
        if value is None:
            continue
        try:
            if len(value) != count:
                continue
        except TypeError:
            continue
        for i in range(count):
            picked[i][key] = value[i]
    return picked


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


def colabfold_data_dir(cache_dir: str | None = None) -> str:
    """Where ``colabfold_batch`` finds its AlphaFold parameters.

    It wants ``<dir>/params/params_model_*.npz``, which is exactly the tree
    build_blackwell.sh already creates at ``community_models/ckpts/AF2`` from the
    2022-12-06 release -- monomer, ptm and multimer sets together. So the weights
    are normally on disk already and the only question is pointing at them.

    Order: an explicit argument, then COLABFOLD_DATA_DIR, then AF2_DIR, then
    CACHE_DIR. Raising beats falling through, because the previous code assigned
    ``os.environ.get("CACHE_DIR")`` OVER its own argument and, with that unset,
    ran ``colabfold_batch ... --data None`` -- four gigabytes downloaded into a
    directory literally named None, per design, on a compute node.
    """
    for value in (cache_dir, os.environ.get("COLABFOLD_DATA_DIR"), os.environ.get("AF2_DIR"),
                  os.environ.get("CACHE_DIR")):
        if value:
            resolved = os.path.expanduser(value)
            if glob.glob(os.path.join(resolved, "params", "params_model_*.npz")):
                _warn_if_download_would_retrigger(resolved)
                return resolved
    for value in (cache_dir, os.environ.get("COLABFOLD_DATA_DIR"), os.environ.get("CACHE_DIR")):
        if value:
            # Nothing on disk yet, but a named directory to download into.
            return os.path.expanduser(value)
    raise RuntimeError(
        "ColabFold needs a parameter directory holding params/params_model_*.npz. "
        "Set COLABFOLD_DATA_DIR, or AF2_DIR to the tree build_blackwell.sh creates at "
        "community_models/ckpts/AF2, or CACHE_DIR to a writable directory to download into."
    )


def _warn_if_download_would_retrigger(data_dir: str) -> None:
    """Say so when a complete parameter store lacks ColabFold's success sentinel.

    ColabFold's download step is skipped only if ``params/download_finished.txt``
    exists. Without it, every ``colabfold_batch`` re-triggers a 3.47 GB download:
    into a read-only store that dies with PermissionError, and into a writable
    one -- which the Complexa tree is -- it quietly succeeds, duplicating weights
    that were already there, once per box and unnoticed until a disk fills.

    Warned rather than fixed here. Creating the file would have this library
    write into a shared model store, and provisioning belongs to
    ``_install/install-colabfold.sh``, which creates the same sentinel under the
    same guard: only when all five ptm sets are present, so an incomplete store
    is never marked finished.
    """
    params = os.path.join(data_dir, "params")
    if os.path.exists(os.path.join(params, "download_finished.txt")):
        return
    complete = all(
        os.path.getsize(os.path.join(params, f"params_model_{i}_ptm.npz")) > 0
        if os.path.exists(os.path.join(params, f"params_model_{i}_ptm.npz"))
        else False
        for i in (1, 2, 3, 4, 5)
    )
    if not complete:
        return
    logger.warning(
        f"{params} holds all five AF2 ptm parameter sets but no download_finished.txt, so "
        f"colabfold_batch will re-download 3.47 GB over them. Create the sentinel once: "
        f"touch {params}/download_finished.txt"
    )


def colabfold_batch_command() -> str:
    """The ``colabfold_batch`` to run, honouring ``COLABFOLD_EXEC_PATH``.

    ColabFold lives in its OWN conda environment by design -- installed by
    ``_install/install-colabfold.sh`` into ``envs/colabfold`` -- because its
    ``[alphafold]`` extra downgrades absl-py, biopython and chex, and pins a jax
    that a Blackwell card cannot use. None of that may reach the environment
    Complexa's primary AF2 refold runs in.

    Complexa cannot ``conda activate`` a second environment mid-process, and it
    does not need to: this is a subprocess, so an absolute path to the other
    env's binary is enough. Named for the executable the way ``RF3_EXEC_PATH``
    is, rather than relying on PATH -- prepending another env's bin directory
    would shadow ``python`` itself.
    """
    return os.environ.get("COLABFOLD_EXEC_PATH") or "colabfold_batch"


def _record_colabfold_confidence(structures_dir: str, seq_name: str, pdb_path: str) -> None:
    """Copy pTM and PAE out of ColabFold's own scores file into a sidecar.

    ColabFold writes ``{job}_scores_rank_001_*.json`` beside the structure, with
    ``ptm`` and the full ``pae`` matrix in Angstroms. Reading the rank-001 file
    rather than any of them matters: the returned PDB is rank 001, and a
    confidence from a different ranked model would describe a structure nobody
    kept.
    """
    matches = sorted(glob.glob(os.path.join(structures_dir, f"{seq_name}_scores_rank_001*.json")))
    if not matches:
        logger.debug(f"No ColabFold scores file for {seq_name}; its pTM and PAE stay unmeasured")
        return
    try:
        with open(matches[0]) as handle:
            scored = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Ignoring unusable ColabFold scores file {matches[0]}: {exc}")
        return
    write_fold_confidence(pdb_path, ptm=scored.get("ptm"), pae=scored.get("pae"))
    save_pae(pdb_path, scored.get("pae"), backend="colabfold", model=os.path.basename(matches[0]))


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

    data_dir = colabfold_data_dir(cache_dir)
    if os.environ.get("CACHE_DIR"):
        os.environ["XDG_CACHE_HOME"] = os.path.expanduser(os.environ["CACHE_DIR"])

    # Create individual FASTA files using the unified function
    fasta_dir = os.path.join(path_to_colabfold_out, "individual_fastas")
    create_individual_fasta_files(sequences, fasta_dir, format_type="simple")

    # Get sequence names for output parsing
    seq_names = [f"seq_{i + 1}" for i in range(len(sequences))]

    # Run ColabFold batch on the directory containing individual FASTA files
    batch_command = (
        f"{colabfold_batch_command()} {fasta_dir} {path_to_colabfold_out}/structures "
        f"--msa-mode single_sequence --data {data_dir}"
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
                _record_colabfold_confidence(
                    f"{path_to_colabfold_out}/structures", seq_name, pdb_file_paths[-1]
                )
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
