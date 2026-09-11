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


def extract_refolded_structure_paths_from_df(
    df: pd.DataFrame, sequence_types: list[str] = None
) -> dict[str, dict[str, list[str | None]]]:
    """Every refold's path per sequence type, not the one a ranking chose.

    Returns ``{sample: {seq_type: [path or None per redesign]}}``, positionally
    aligned with the row's other per-sequence lists: a redesign whose structure
    is missing holds ``None`` rather than being dropped, so slot *i* here is the
    sequence in ``{seq_type}_sequence_all[i]``.

    This replaced a lookup of the headline scalar ``{seq}_complex_{backend}_pdb_path``,
    which had two problems at once. Evaluate stopped writing that scalar when
    ranking moved to analyze, so the lookup found nothing and every metric
    computed on a refolded structure went silently missing from the run. And
    reading it at all meant one redesign's interface was measured and the rest
    were not -- so nothing downstream could re-rank on an interface number, which
    is the whole point of emitting per-sequence lists.

    The ``_all`` column is the source; the scalar is accepted as a fallback so a
    frame written before the split still resolves.
    """
    if sequence_types is None:
        sequence_types = SEQUENCE_TYPES

    # Built through the one naming rule rather than guessed at, and resolved from
    # the frame's own provenance column so a run folded by RF3 looks for RF3's
    # columns rather than AF2's.
    backend = complex_backend_of(df) or "af2"
    path_columns = {t: rename(f"{t}_complex_pdb_path", backend) for t in sequence_types}

    paths: dict[str, dict[str, list[str | None]]] = {}

    for _, row in df.iterrows():
        sample_name = os.path.basename(row["pdb_path"]).replace(".pdb", "").replace("tmp_", "")
        found = paths.setdefault(sample_name, {})

        for seq_type in sequence_types:
            column = path_columns[seq_type]
            values = row.get(f"{column}_all")
            if not isinstance(values, (list, tuple)):
                # A frame from before evaluate emitted lists.
                single = row.get(column)
                values = [single] if isinstance(single, str) and single else []
            slots = [p if isinstance(p, str) and p and os.path.exists(p) else None for p in values]
            if any(slot is not None for slot in slots):
                found[seq_type] = slots
            else:
                logger.debug(f"No valid structure paths for {sample_name} {seq_type} in {column}_all")

    # Loud, because the failure mode is silence. This used to try four candidate
    # names, three of which never existed in any frame; when the rename turned the
    # real one into {seq}_complex_{backend}_pdb_path, finding nothing looked
    # exactly like the ordinary "this design has no refold" miss. The refolded
    # interface metrics were then simply absent from a run that asked for them,
    # with nothing above debug level to say so.
    if not any(paths.values()):
        logger.error(
            f"No refolded structure paths found in any of {len(df)} rows. Looked for "
            f"{sorted(c + '_all' for c in path_columns.values())}; the frame has "
            f"{sorted(c for c in df.columns if 'pdb_path' in c)}. Any metric computed on "
            f"refolded structures will be absent."
        )
    return paths


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
