"""One set of redesigns per design, generated once and used by both tracks.

The binder track and the designability track both redesign the same binder
backbone with the same inverse folder, the same context chains, the same
alphabet, the same temperature and -- because :func:`mpnn_seed` deliberately
excludes the sequence COUNT from its hash -- the same seed. So the binder
track's ``num_redesign_seqs`` sequences are a prefix of designability's
``designability_num_seq``, and the second ProteinMPNN run reproduces work the
first already did. Verified directly: both tracks derive seed 3926876749 for the
same design, and ``script_utils/bioinformatic/verify_mpnn_prefix.py`` checks the
prefix property against the real model.

That prefix is a property of the current seeding, not a law. Relying on it at
every use would make it load-bearing everywhere and checkable nowhere, so both
tracks ask for the SAME count -- ``max(num_redesign_seqs,
designability_num_seq)`` -- and the binder track slices. The prefix is then used
in exactly one place, guarded by a fingerprint that records the count that was
generated, and the moment MPNN-score ranking makes the subset something other
than a prefix, this is the only function that has to change.

``mpnn_fixed`` never shares. It holds interface positions fixed and seeds with
``variant="fixed"``, so it is a different draw by construction; it gets its own
cache file and asks only for the count it uses.
"""

from __future__ import annotations

import hashlib
import json
import os

from loguru import logger

from proteinfoundation.metrics.seeding import (
    MPNN_OMIT_AAS,
    MPNN_SAMPLING_TEMP,
    SEED_DERIVATION_VERSION,
    mpnn_seed,
)

REDESIGN_SET_SCHEMA = 1
REDESIGN_SET_TEMPLATE = "redesign_set_{variant}.json"


def redesign_set_path(cache_dir: str, variant: str = "") -> str:
    return os.path.join(cache_dir, REDESIGN_SET_TEMPLATE.format(variant=variant or "mpnn"))


def redesign_set_fingerprint(
    *,
    design_name: str,
    context_chains: list[str],
    chains_to_design: list[str],
    count: int,
    inverse_folding_model: str,
    variant: str = "",
    fixed_positions=None,
) -> str:
    """Everything that determines the sequences, including how many were made.

    The count is in the key even though it is NOT in the seed. That asymmetry is
    the point: the seed excludes it so a shorter run is a prefix of a longer one,
    and the key includes it so a cache holding two sequences cannot answer a
    request for eight. Without that, asking for more would silently return fewer.
    """
    canonical = json.dumps(
        {
            "design": design_name,
            "context_chains": sorted(context_chains),
            "chains_to_design": sorted(chains_to_design),
            "count": int(count),
            "inverse_folding_model": inverse_folding_model,
            "variant": variant,
            "omit_AAs": sorted(MPNN_OMIT_AAS),
            "temperature": MPNN_SAMPLING_TEMP,
            "seed_derivation": SEED_DERIVATION_VERSION,
            # Which positions were held fixed decides the sequences, so it decides
            # the key. Absent for the unfixed variant, which is what keeps its key
            # independent of any interface derivation.
            "fixed_positions": sorted(fixed_positions) if fixed_positions else None,
            "schema": REDESIGN_SET_SCHEMA,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_redesign_set(cache_dir: str, fingerprint: str, count: int, variant: str = "") -> list[dict] | None:
    """The cached set, or None. Never raises."""
    path = redesign_set_path(cache_dir, variant)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            cached = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Ignoring unusable redesign set {path}: {exc}")
        return None
    if cached.get("fingerprint") != fingerprint:
        return None
    sequences = cached.get("sequences") or []
    if len(sequences) < count:
        # A shorter cached set cannot answer a longer request: the missing
        # sequences are not derivable from the ones present.
        return None
    # The one place the prefix property is relied on, and the fingerprint above
    # records the count that produced it.
    return sequences[:count]


def write_redesign_set(cache_dir: str, fingerprint: str, sequences: list[dict], variant: str = "") -> None:
    """Persist the set. Never raises: losing a cache costs time, not correctness."""
    path = redesign_set_path(cache_dir, variant)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(
                {"fingerprint": fingerprint, "schema": REDESIGN_SET_SCHEMA, "sequences": sequences},
                handle,
            )
    except OSError as exc:
        logger.warning(f"Could not write the redesign set for {cache_dir}: {exc}")


def shared_redesign_set(
    *,
    design_name: str,
    mpnn_input_pdb: str,
    out_dir_root: str,
    cache_dir: str,
    context_chains: list[str],
    chains_to_design: list[str],
    count: int,
    inverse_folding_model: str,
    variant: str = "",
    fix_pos=None,
    reuse_cache: bool = True,
) -> list[dict]:
    """The design's redesigns, generated once and shared by every track that wants them.

    Returns the full dicts ProteinMPNN produced -- ``seq`` and its score -- not
    just the strings, because the binder track reports the score as
    ``{seq_type}_redesign_score_all`` and would otherwise have to regenerate the
    set to recover it.
    """
    fingerprint = redesign_set_fingerprint(
        design_name=design_name,
        context_chains=context_chains,
        chains_to_design=chains_to_design,
        count=count,
        inverse_folding_model=inverse_folding_model,
        variant=variant,
        fixed_positions=fix_pos,
    )
    if reuse_cache:
        cached = read_redesign_set(cache_dir, fingerprint, count, variant)
        if cached is not None:
            logger.info(f"Reusing {len(cached)} cached redesigns for {design_name} ({variant or 'mpnn'})")
            return cached

    from proteinfoundation.metrics.inverse_folding_models import inverse_fold

    sequences = inverse_fold(
        model_type=inverse_folding_model,
        pdb_file_path=mpnn_input_pdb,
        out_dir_root=out_dir_root,
        all_chains=context_chains,
        pdb_path_chains=chains_to_design,
        fix_pos=fix_pos,
        num_seq_per_target=count,
        omit_AAs=MPNN_OMIT_AAS,
        sampling_temp=MPNN_SAMPLING_TEMP,
        seed=mpnn_seed(design_name, context_chains, chains_to_design, variant=variant),
        verbose=False,
    )
    # Recorded in the payload rather than the key: which view of the design the
    # inverse folder read is worth being able to see, but keying on it would stop
    # a protein_mpnn campaign sharing at all, and the two views are measured
    # equivalent under --ca_only.
    write_redesign_set(cache_dir, fingerprint, list(sequences), variant)
    logger.info(f"Generated {len(sequences)} redesigns for {design_name} ({variant or 'mpnn'})")
    return list(sequences)


def redesign_set_size(cfg_metric) -> int:
    """How many redesigns to generate, independent of which metrics are enabled.

    ``max`` of what either track wants, ALWAYS -- not conditioned on
    ``compute_designability``. A count that moved with an unrelated flag would
    change which sequences a campaign produces, not merely which are measured,
    and the apo scRMSD of sequence *i* would stop being a property of the design.
    """
    cfg_metric = cfg_metric if cfg_metric is not None else {}
    redesigns = int(cfg_metric.get("num_redesign_seqs") or 0)
    designability = int(cfg_metric.get("designability_num_seq") or 0) or 8
    return max(redesigns, designability, 1)
