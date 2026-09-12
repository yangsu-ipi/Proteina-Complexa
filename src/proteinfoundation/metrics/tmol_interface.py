"""TMOL force-field interface metrics, read off one structure.

One definition, two callers. ``binder_eval`` computes these on the generated
complex and on the primary backend's refolds; ``consensus_folding`` reads them
off a kept advisory structure. Before this module the mapping from TMOL's reward
dictionary to column names lived only in ``binder_eval``, so the advisory track
could not have them without copying it -- and a second copy of "which key is the
hydrogen-bond count" is exactly the kind of near-duplicate that drifts.

TMOL scores *interchain* pairs and works out the chains itself, so it needs no
binder/target argument: any complex whose chains are separate chains is scored
the same way, whoever folded it.

Nothing here raises. An environment without TMOL, or a structure it cannot load,
yields an empty dict, which every caller reads as "unmeasured" and writes as an
absent or NaN column.
"""

from typing import Any

from loguru import logger

# The reward keys, in the column names they are reported under. Ordered so the
# hydrogen-bond pair and the electrostatics pair read together.
TMOL_METRICS: dict[str, str] = {
    "n_interface_hbonds": "n_interface_hbonds_tmol",
    "total_interface_hbond_energy": "total_interface_hbond_energy_tmol",
    "total_interface_elec_energy": "total_interface_elec_energy_tmol",
    "n_interface_elec_interactions": "n_interface_elec_interactions_tmol",
}

TMOL_METRIC_COLS: list[str] = list(TMOL_METRICS.values())


def tmol_available() -> bool:
    """Whether TMOL can be imported in this environment.

    An import, not a flag: TMOL pulls a compiled extension, and whether it loads
    is a property of the machine rather than of the config.
    """
    try:
        from proteinfoundation.rewards.tmol_reward import TmolRewardModel  # noqa: F401
    except (ImportError, RuntimeError, OSError) as exc:
        logger.debug(f"TMOL unavailable: {exc}")
        return False
    return True


# Process-cached, because constructing it loads a force field. The advisory
# derivation re-reads thousands of structures in one run and must not pay that
# per structure. None means "tried and failed" -- distinct from "not tried yet".
_MODEL: Any | None = None
_TRIED = False


def tmol_model() -> Any | None:
    """The shared scorer, constructed once per process, or None."""
    global _MODEL, _TRIED
    if _TRIED:
        return _MODEL
    _TRIED = True
    try:
        from proteinfoundation.rewards.tmol_reward import TmolRewardModel

        _MODEL = TmolRewardModel(enable_hbond=True, enable_elec=True)
    except Exception as exc:
        logger.warning(f"Failed to initialize TMOL model: {exc}. TMOL metrics will be absent.")
        _MODEL = None
    return _MODEL


def tmol_interface_metrics(pdb_path: str, model: Any | None = None) -> dict[str, float]:
    """The four TMOL metrics for one complex, or ``{}`` if they cannot be had.

    *model* lets a caller that already built a scorer pass it in rather than
    reaching for the shared one; both end up in the same place.
    """
    from proteinfoundation.rewards.base_reward import REWARD_KEY

    scorer = model if model is not None else tmol_model()
    if scorer is None:
        return {}
    try:
        scored = scorer.score(pdb_path=pdb_path, requires_grad=False)[REWARD_KEY]
    except Exception as exc:
        logger.error(f"TMOL error for {pdb_path}: {exc}")
        return {}
    return {column: float(scored[key].item()) for key, column in TMOL_METRICS.items() if key in scored}
