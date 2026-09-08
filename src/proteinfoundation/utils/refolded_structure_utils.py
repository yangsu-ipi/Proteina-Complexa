#!/usr/bin/env python3
"""
Utilities for handling refolded structure paths and computing metrics on them.
This module provides functions to extract paths to refolded structures from
binder evaluation results and compute force field and bioinformatics metrics
on successful samples only.
"""

import glob
import os

import pandas as pd
from loguru import logger

from proteinfoundation.metrics.column_names import rename
from proteinfoundation.result_analysis.analysis_utils import SEQUENCE_TYPES
from proteinfoundation.result_analysis.binder_analysis_utils import complex_backend_of


def extract_best_refolded_structure_paths_from_df(
    df: pd.DataFrame, sequence_types: list[str] = None
) -> dict[str, dict[str, str]]:
    """
    Extract paths to the best refolded structures from binder evaluation dataframe.

    For Protenix evaluation, the dataframe contains columns with the paths to the best
    refolded structures for each sequence type:
    - For mpnn and mpnn_fixed: uses the single "best" path column (not the "_all" column)
    - For self: uses the single path column

    Args:
        df: Binder evaluation results dataframe
        sequence_types: List of sequence types to consider

    Returns:
        Dictionary mapping sample names to best structure paths:
        {
            'sample_name': {
                'mpnn': 'path_to_best_mpnn_structure',
                'mpnn_fixed': 'path_to_best_mpnn_fixed_structure',
                'self': 'path_to_self_structure'
            }
        }
    """
    if sequence_types is None:
        sequence_types = SEQUENCE_TYPES

    # Built through the one naming rule rather than guessed at, and resolved from
    # the frame's own provenance column so a run folded by RF3 looks for RF3's
    # columns rather than AF2's.
    backend = complex_backend_of(df) or "af2"
    path_columns = {t: rename(f"{t}_complex_pdb_path", backend) for t in sequence_types}

    best_paths = {}

    for _, row in df.iterrows():
        # Extract sample name from pdb_path
        pdb_path = row["pdb_path"]
        sample_name = os.path.basename(pdb_path).replace(".pdb", "").replace("tmp_", "")

        if sample_name not in best_paths:
            best_paths[sample_name] = {}

        for seq_type in sequence_types:
            col = path_columns[seq_type]
            structure_path = row[col] if col in row.index and pd.notna(row[col]) and row[col] != "" else None
            if structure_path and os.path.exists(structure_path):
                best_paths[sample_name][seq_type] = structure_path
            else:
                logger.debug(f"No valid structure path found for {sample_name} {seq_type} in {col}")

    # Loud, because the failure mode is silence. This used to try four candidate
    # names, three of which never existed in any frame; when the rename turned the
    # real one into {seq}_complex_{backend}_pdb_path, finding nothing looked
    # exactly like the ordinary "this design has no refold" miss. The refolded
    # interface metrics were then simply absent from a run that asked for them,
    # with nothing above debug level to say so.
    if not any(best_paths.values()):
        logger.error(
            f"No refolded structure paths found in any of {len(df)} rows. Looked for "
            f"{sorted(path_columns.values())}; the frame has "
            f"{sorted(c for c in df.columns if c.endswith('pdb_path'))}. Any metric computed on "
            f"refolded structures will be absent."
        )
    return best_paths


def extract_refolded_paths_from_evaluation_output(
    evaluation_output_dir: str, folding_method: str, sample_names: list[str]
) -> dict[str, dict[str, list[str]]]:
    """
    Alternative method to extract refolded structure paths directly from evaluation output directories.

    This function is useful when the evaluation has already been run and you want to
    extract the structure paths for post-processing.

    Args:
        evaluation_output_dir: Directory containing evaluation outputs
        folding_method: Folding method used ('colabdesign', 'protenix', etc.)
        sample_names: List of sample names to look for

    Returns:
        Dictionary mapping sample names to structure paths
    """
    refolded_paths = {}

    for sample_name in sample_names:
        sample_dir = os.path.join(evaluation_output_dir, f"tmp_{sample_name}")
        if not os.path.exists(sample_dir):
            continue

        refolded_paths[sample_name] = {"mpnn": [], "mpnn_fixed": [], "self": []}

        if folding_method == "colabdesign":
            # Look for ColabDesign output structure files
            complex_dir = os.path.join(sample_dir, "MPNN", "Complex")
            if os.path.exists(complex_dir):
                complex_files = glob.glob(os.path.join(complex_dir, "*.pdb"))
                complex_files = sorted(complex_files)

                seq_per_type = 8
                if len(complex_files) >= seq_per_type:
                    refolded_paths[sample_name]["mpnn"] = complex_files[:seq_per_type]
                if len(complex_files) >= 2 * seq_per_type:
                    refolded_paths[sample_name]["mpnn_fixed"] = complex_files[seq_per_type : 2 * seq_per_type]
                if len(complex_files) >= 2 * seq_per_type + 1:
                    refolded_paths[sample_name]["self"] = [complex_files[2 * seq_per_type]]

    return refolded_paths
