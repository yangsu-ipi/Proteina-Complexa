"""Which folders refold, resolved once from config for every track.

Four keys used to answer this question in four places -- ``monomer_folding_models``
with ``designability_folding_models`` and ``codesignability_folding_models``
overriding it, ``apo_folding_models`` beside them, and ``consensus_backends`` on
the complex side with ``binder_folding_method`` naming the one folder allowed to
decide anything. A campaign could therefore refold with ESMFold v1 for
designability, ESMFold2 and AF2 for apo, AF2 for the complex, and nothing at all
for the complex cross-check -- which is what the shipped configs actually did,
without any of it being visible in one place.

One key now. ``metric.folding_models`` names the folders a campaign uses, and
every track gets the members that can serve it. What a folder can serve is a
capability (see ``_FOLDER_CAPABILITIES``); whether its columns decide a verdict
is a threshold question, answered in analyze.

The legacy keys still work, for one release, because a campaign mid-flight must
not change what it folds when the code under it moves. Setting both is refused:
two sources of truth for what folded is not a thing precedence should resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from proteinfoundation.metrics.column_names import (
    FOLD_TRACKS,
    folder_can,
    folder_family,
    folders_for_track,
)

# The keys this replaces, in the order a reader should look for them.
LEGACY_FOLDER_KEYS: tuple[str, ...] = (
    "monomer_folding_models",
    "designability_folding_models",
    "codesignability_folding_models",
    "apo_folding_models",
    "consensus_backends",
    "binder_folding_method",
)

FOLDING_MODELS_KEY = "folding_models"


class FolderSelectionError(ValueError):
    """A folder configuration that cannot be resolved into what will refold."""


@dataclass
class ResolvedFolders:
    """Which folders serve which track, for one run."""

    monomer: list[str] = field(default_factory=list)
    complex: list[str] = field(default_factory=list)
    # Recorded per row so a result says which folders produced it, the way
    # redesign_conditioning records how its redesigns were made.
    source: str = FOLDING_MODELS_KEY

    def for_track(self, track: str) -> list[str]:
        return list(self.complex if track == "complex" else self.monomer)

    def describe(self) -> str:
        return f"monomer={self.monomer or '-'} complex={self.complex or '-'} (from {self.source})"


def _named_legacy_keys(cfg_metric) -> list[str]:
    return [key for key in LEGACY_FOLDER_KEYS if cfg_metric.get(key) not in (None, [], "")]


def resolve_folding_models(cfg_metric, *, is_target_ligand: bool = False) -> ResolvedFolders:
    """The folders each track will use, from ``metric`` config.

    Warns rather than silently skipping when a named folder cannot serve a track:
    "every folder refolds everything" is a claim, and a claim has to be checkable
    from the run log. Raises when a folder is unknown entirely, which is a typo
    that would otherwise mint columns no threshold and no detector will match.
    """
    cfg_metric = cfg_metric if cfg_metric is not None else {}
    named = _named_legacy_keys(cfg_metric)
    requested = list(cfg_metric.get(FOLDING_MODELS_KEY) or [])

    if requested and named:
        raise FolderSelectionError(
            f"metric.{FOLDING_MODELS_KEY} is set alongside the key(s) it replaces ({', '.join(named)}). "
            f"Two sources of truth for what refolded is not something precedence should resolve -- "
            f"remove the old key(s), or remove {FOLDING_MODELS_KEY}."
        )

    if not requested:
        return _resolve_legacy(cfg_metric, named, is_target_ligand=is_target_ligand)

    unknown = [name for name in requested if not any(folder_can(name, t) for t in FOLD_TRACKS)]
    if unknown:
        raise FolderSelectionError(
            f"metric.{FOLDING_MODELS_KEY} names {unknown}, which no track can refold. A folder that "
            f"nothing recognises mints columns no threshold will ever match, so this is refused "
            f"rather than warned about."
        )

    resolved = ResolvedFolders(
        monomer=folders_for_track(requested, "monomer"),
        complex=folders_for_track(requested, "complex"),
    )
    if is_target_ligand and "af2" in resolved.complex:
        # Not a capability limit of AlphaFold2 -- a limit of the harness that runs
        # it for a complex here, which refuses a ligand target outright.
        logger.warning(
            "af2 cannot fold a ligand complex (ColabDesign refuses one), so the complex track runs "
            f"without it: {[m for m in resolved.complex if m != 'af2'] or 'nothing'}"
        )
        resolved.complex = [m for m in resolved.complex if m != "af2"]

    for name in (folder_family(n) or n for n in requested):
        missing = [t for t in FOLD_TRACKS if not folder_can(name, t)]
        if missing:
            logger.warning(
                f"{name} does not refold {', '.join(missing)}; those tracks run without it. "
                f"This is a capability of the folder, not a choice in this config."
            )
    logger.info(f"Folding models resolved: {resolved.describe()}")
    return resolved


def _resolve_legacy(cfg_metric, named: list[str], *, is_target_ligand: bool) -> ResolvedFolders:
    """Honour the per-track keys exactly as they behaved, and say so once.

    Deliberately NOT a merge into one list: these keys legitimately held different
    values per track, and a campaign that resumes must fold what it folded before.
    """
    shared = list(cfg_metric.get("monomer_folding_models", ["esmfold"]) or [])
    monomer: list[str] = []
    for key in ("designability_folding_models", "codesignability_folding_models"):
        monomer.extend(cfg_metric.get(key, shared) or [])
    monomer.extend(cfg_metric.get("apo_folding_models", []) or [])

    complex_folders = list(cfg_metric.get("consensus_backends", []) or [])
    primary = cfg_metric.get("binder_folding_method")
    if primary:
        complex_folders.insert(0, primary)

    resolved = ResolvedFolders(
        monomer=folders_for_track(monomer or shared, "monomer"),
        complex=folders_for_track(complex_folders, "complex"),
        source=", ".join(named) if named else "defaults",
    )
    if is_target_ligand:
        resolved.complex = [m for m in resolved.complex if m != "af2"]
    if named:
        logger.warning(
            f"metric.{', metric.'.join(named)} still name the folders for this run. They are "
            f"replaced by one metric.{FOLDING_MODELS_KEY} listing every folder the campaign uses; "
            f"each track then gets the members that can serve it. Resolved: {resolved.describe()}"
        )
    return resolved
