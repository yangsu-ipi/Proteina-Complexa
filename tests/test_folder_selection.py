"""Which folders refold, and what happens when a config says something that
cannot be honoured.

Four keys used to answer this in four places, and the shipped configs genuinely
disagreed across them: ESMFold v1 for designability, ESMFold2 + AF2 for apo, AF2
for the complex, nothing at all for the complex cross-check. None of that was
visible anywhere in one place, which is the problem these tests pin shut.
"""

import pytest
from loguru import logger

from proteinfoundation.metrics.folder_selection import (
    FolderSelectionError,
    resolve_folding_models,
)


def _warnings(fn):
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        return fn(), seen
    finally:
        logger.remove(sink)


def test_one_list_serves_every_track_it_can():
    resolved = resolve_folding_models({"folding_models": ["af2", "esmfold2"]})
    assert resolved.designability == ["af2", "esmfold2"]
    assert resolved.codesignability == ["af2", "esmfold2"]
    assert resolved.apo == ["af2", "esmfold2"]
    assert resolved.complex == ["af2", "esmfold2"], "both fold a complex; only gating differed"


def test_a_folder_that_cannot_serve_a_track_is_named_not_dropped_in_silence():
    """'Every folder refolds everything' is a claim, and a claim has to be
    checkable from the run log."""
    resolved, seen = _warnings(
        lambda: resolve_folding_models({"folding_models": ["af2", "esmfold", "rf3"]})
    )
    assert resolved.apo == ["af2", "esmfold"], "rf3 folds no isolated monomer here"
    assert resolved.complex == ["af2", "rf3"], "esmfold v1 folds no complex"
    assert any("esmfold does not refold complex" in m for m in seen)
    assert any("rf3 does not refold monomer" in m for m in seen)
    assert any("capability of the folder, not a choice in this config" in m for m in seen)


def test_an_unknown_folder_is_refused_rather_than_warned_about():
    """A typo mints columns no threshold and no detector will ever match, and the
    monomer gate is OR by default -- so it would loosen the gate rather than fail."""
    with pytest.raises(FolderSelectionError) as raised:
        resolve_folding_models({"folding_models": ["af2", "esmfold3"]})
    assert "esmfold3" in str(raised.value)


def test_a_ligand_target_loses_af2_from_the_complex_track_with_a_reason():
    """Not a limit of AlphaFold2 -- a limit of the harness that runs it for a
    complex, which refuses a ligand target outright."""
    resolved, seen = _warnings(
        lambda: resolve_folding_models({"folding_models": ["af2", "esmfold2"]}, is_target_ligand=True)
    )
    assert resolved.complex == ["esmfold2"]
    assert resolved.apo == ["af2", "esmfold2"], "the monomer tracks are unaffected"
    assert any("ColabDesign refuses one" in m for m in seen)


def test_the_legacy_keys_still_decide_what_a_running_campaign_folds():
    """A campaign mid-flight must not change what it folds because the code under
    it moved. Honoured per-track exactly as before -- deliberately NOT merged into
    one list, since these keys legitimately held different values per track."""
    legacy = {
        "monomer_folding_models": ["esmfold"],
        "apo_folding_models": ["esmfold2", "colabfold"],
        "binder_folding_method": "colabdesign",
        "consensus_backends": ["esmfold2"],
    }
    resolved, seen = _warnings(lambda: resolve_folding_models(legacy))
    # Per track, NOT unioned: these keys hold different values on purpose, and a
    # resuming campaign must refold what it refolded.
    assert resolved.designability == ["esmfold"], "monomer_folding_models still governs designability"
    assert resolved.codesignability == ["esmfold"]
    assert resolved.apo == ["esmfold2", "af2"], "and apo keeps its own, in the new vocabulary"
    assert resolved.complex == ["af2", "esmfold2"], "colabdesign is how af2 folds a complex"
    assert any("replaced by one metric.folding_models" in m for m in seen)


def test_both_forms_are_refused_when_they_disagree():
    """Whichever lost would be invisible in the results, so precedence must not
    settle it. The message has to name WHICH track disagrees and show both
    resolutions, because the two keys usually live in different files."""
    with pytest.raises(FolderSelectionError) as raised:
        resolve_folding_models({"folding_models": ["af2"], "apo_folding_models": ["esmfold2"]})
    message = str(raised.value)
    assert "apo_folding_models" in message
    assert "apo" in message and "disagree about" in message
    assert "pipeline.yaml" in message, "and say where to make the edit"


def test_both_forms_are_allowed_when_they_agree():
    """The ordinary shape of a half-migrated Hydra composition: a base config
    supplies the new key while a campaign's own pipeline.yaml still overrides the
    old ones. That is not two people disagreeing, and refusing it would break
    every in-flight campaign the moment the base config migrated."""
    resolved, seen = _warnings(
        lambda: resolve_folding_models(
            {
                "folding_models": ["af2", "esmfold2"],
                "apo_folding_models": ["af2", "esmfold2"],
                "monomer_folding_models": ["af2", "esmfold2"],
                "binder_folding_method": "colabdesign",
                "consensus_backends": ["esmfold2"],
            }
        )
    )
    assert resolved.apo == ["af2", "esmfold2"]
    assert any("resolve to the same folders" in m for m in seen), "allowed, but still say so"


def test_a_config_naming_nothing_still_resolves_to_what_it_used_to():
    """No keys at all is the shipped default of several configs, and it must not
    become an error or a silently wider set."""
    resolved, _ = _warnings(lambda: resolve_folding_models({}))
    assert resolved.designability == ["esmfold"], "the historical default"
    assert resolved.apo == ["esmfold"], "apo's own historical default, not the shared one"
    assert resolved.complex == [], "and no complex folder without one named"


def test_the_complex_constructor_accepts_the_model_and_the_harness():
    """Regression. `colabdesign` is the harness `af2` runs in for a complex, and
    it resolves through the FAMILY table, not the alias map -- canonical_backend
    leaves it alone. Getting that wrong made initialize_folding_model reject
    'colabdesign', which is the shipped default of every binder config, so every
    binder campaign died at the first design.

    The suite did not catch it because nothing called this constructor by name.
    A real run did, immediately."""
    from proteinfoundation.evaluation.binder_eval import initialize_folding_model

    for name in ("colabdesign", "af2"):
        specs = initialize_folding_model(name, ["A"], "task", False)
        assert specs["model_name"] == "colabdesign", f"{name} must construct the AF2 complex folder"

    with pytest.raises(ValueError, match="does not support ligand"):
        initialize_folding_model("colabdesign", ["A"], "task", True)

    with pytest.raises(ValueError, match="not supported"):
        initialize_folding_model("no_such_folder", ["A"], "task", False)

    # A folder that needs no construction is not an unsupported folder. This
    # raised for esmfold2, which every binder campaign configures and
    # CONSENSUS_BACKENDS folds -- so the pass naming it as its only complex
    # folder died on the folder it was created to run.
    assert initialize_folding_model("esmfold2", ["A"], "task", False)["model_name"] == "esmfold2"


def test_every_folder_name_a_config_may_use_resolves_somewhere():
    """The three ways a folder can be named -- model, monomer implementation,
    complex harness, versioned harness -- must all land on the same family, or a
    config that says one thing and a constructor that expects another disagree
    silently until a campaign dies."""
    from proteinfoundation.metrics.column_names import folder_family

    assert folder_family("af2") == "af2"
    assert folder_family("colabfold") == "af2", "the monomer implementation"
    assert folder_family("colabdesign") == "af2", "the complex harness"
    assert folder_family("rf3_latest") == "rf3", "a versioned harness"
    assert folder_family("esmfold2") == "esmfold2"
    assert folder_family("nonsense") is None


def test_the_legacy_path_never_widens_a_track_by_union():
    """The trap in wiring this: unioning the per-track keys would make a campaign
    whose designability used ESMFold v1 and whose apo used ESMFold2 + AF2 suddenly
    fold designability with all three -- changing what a resuming campaign refolds,
    which is the one thing the legacy path exists to prevent."""
    resolved = resolve_folding_models(
        {
            "monomer_folding_models": ["esmfold"],
            "apo_folding_models": ["esmfold2", "colabfold"],
        }
    )
    assert resolved.designability == ["esmfold"]
    assert resolved.apo == ["esmfold2", "af2"]
    assert resolved.monomer == ["esmfold", "esmfold2", "af2"], "the union is for REPORTING only"
