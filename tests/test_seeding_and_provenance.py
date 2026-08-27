"""Seeds and the provenance columns that describe what they produced.

Seeding exists so the apo and holo tracks can be joined per sequence: a redesign
must be reproducible across a resume, and both tracks must derive the *same* seed
for the same design. The provenance columns exist because the metrics keep their
names while their meaning changes, so two eras must not be silently averaged.

The load-bearing property in both cases is what the value does *not* depend on --
a seed independent of how many sequences were asked for, a column name that
survives the groupby exclusion list.
"""

import re

import pytest

from proteinfoundation.evaluation.utils import (
    REDESIGN_CONDITIONING_BINDER_ONLY,
    REDESIGN_CONDITIONING_COMPLEX,
    redesign_conditioning,
)
from proteinfoundation.metrics.inverse_folding_models import (
    REDESIGN_SCORE_KIND,
    SCORE_KIND_CONFIDENCE,
    SCORE_KIND_NLL,
    resolve_inverse_folding_model,
)
from proteinfoundation.metrics.seeding import (
    MPNN_OMIT_AAS,
    SEED_DERIVATION_VERSION,
    deterministic_seed,
    mpnn_seed,
)

# ------------------------------------------------------------------- seeding


def test_same_request_same_seed():
    assert mpnn_seed("design_0", ["A", "B"], ["B"]) == mpnn_seed("design_0", ["A", "B"], ["B"])


def test_seed_takes_no_sequence_count():
    """The point of Option D: the complex track wants num_redesign_seqs and
    designability wants designability_num_seq, and they must be one draw used
    twice. A count in the seed would make them two different draws, so the
    function must not be able to see one."""
    import inspect

    params = set(inspect.signature(mpnn_seed).parameters)
    assert params == {"design_name", "context_chains", "chains_to_design", "variant"}


@pytest.mark.parametrize(
    "a,b",
    [
        (("d0", ["A", "B"], ["B"], ""), ("d1", ["A", "B"], ["B"], "")),  # design
        (("d0", ["A", "B"], ["B"], ""), ("d0", ["B"], ["B"], "")),  # conditioning
        (("d0", ["A", "B"], ["B"], ""), ("d0", ["A", "B"], ["A"], "")),  # chain designed
        (("d0", ["A", "B"], ["B"], ""), ("d0", ["A", "B"], ["B"], "fixed")),  # variant
    ],
)
def test_seed_separates_genuinely_different_requests(a, b):
    assert mpnn_seed(*a) != mpnn_seed(*b)


def test_chain_order_does_not_change_the_seed():
    assert mpnn_seed("d", ["B", "A"], ["B"]) == mpnn_seed("d", ["A", "B"], ["B"])


def test_seed_is_never_zero(monkeypatch):
    """protein_mpnn_run.py reads its seed as `if args.seed:` and draws a random one
    when that is falsy, so a derived 0 would make exactly one design in 2**32
    unreproducible -- and unreproducible in the way hardest to notice, since every
    other design in the run would be fine."""
    import proteinfoundation.metrics.seeding as seeding

    monkeypatch.setattr(seeding, "deterministic_seed", lambda *parts: 0)
    assert seeding.mpnn_seed("d", ["A"], ["A"]) != 0


def test_seed_is_in_the_range_the_tools_accept():
    for name in ("a", "bb", "design_0_beam"):
        seed = mpnn_seed(name, ["A", "B"], ["B"])
        assert 0 < seed < 2**32


def test_derivation_version_travels_with_the_seed():
    """The constant means "the seeds actually applied changed", not just "the hash
    changed" -- a --seed that never reaches the subprocess breaks the same
    guarantee. Cache fingerprints carry it so such a change invalidates."""
    assert isinstance(SEED_DERIVATION_VERSION, int)
    assert deterministic_seed("a", "b") != deterministic_seed("b", "a")


def test_omit_list_is_shared_so_both_tracks_draw_from_one_alphabet():
    assert MPNN_OMIT_AAS == ["C"]


# -------------------------------------------------------------- provenance


def test_conditioning_is_derived_from_what_the_model_saw():
    assert redesign_conditioning(["B"]) == REDESIGN_CONDITIONING_BINDER_ONLY
    assert redesign_conditioning(["A", "B"]) == REDESIGN_CONDITIONING_COMPLEX
    assert redesign_conditioning(["A", "B", "C"]) == REDESIGN_CONDITIONING_COMPLEX


GROUPBY_EXCLUDED_SUBSTRINGS = ("_res_", "complex_", "binder_", "refolded_", "generated_", "_tmol", "_ligand")
GROUPBY_EXCLUDED_PREFIXES = ("self_", "mpnn_", "mpnn_fixed_")


@pytest.mark.parametrize("column", ["redesign_conditioning", "redesign_model", "redesign_score_kind"])
def test_run_level_provenance_columns_can_group(column):
    """These exist so results from two eras split into separate rows instead of
    averaging two different quantities. A column excluded from groupby is carried
    along as a passive string and cannot do that -- which is why the obvious names
    (_res_mpnn_conditioning, mpnn_score_kind) were both wrong."""
    assert not any(s in column for s in GROUPBY_EXCLUDED_SUBSTRINGS)
    assert not column.startswith(GROUPBY_EXCLUDED_PREFIXES)


@pytest.mark.parametrize("column", ["mpnn_redesign_score", "self_pass_all", "mpnn_apo_scRMSD_ca_esmfold"])
def test_per_sequence_columns_do_not_group(column):
    """Per-sample values must not become grouping keys."""
    assert any(s in column for s in GROUPBY_EXCLUDED_SUBSTRINGS) or column.startswith(GROUPBY_EXCLUDED_PREFIXES)


def test_groupby_exclusion_lists_match_the_source():
    """Guards the two lists above against drifting from analyze.py."""
    import pathlib

    source = pathlib.Path("src/proteinfoundation/analyze.py").read_text()
    block = source[source.index("def get_groupby_columns") : source.index("# Interface Metrics Aggregation")]
    substrings = re.findall(r'"([^"]+)",\s*#', block.split("exclude_substr = [")[1].split("]")[0])
    prefixes = re.findall(r'"([^"]+)",\s*#', block.split("exclude_prefixes = [")[1].split("]")[0])
    assert tuple(substrings) == GROUPBY_EXCLUDED_SUBSTRINGS
    assert tuple(prefixes) == GROUPBY_EXCLUDED_PREFIXES


# ------------------------------------------------------ inverse folder choice


def test_score_conventions_point_opposite_ways():
    """ProteinMPNN reports an NLL, LigandMPNN/SolubleMPNN a confidence, and both
    arrive under the key "score". A ranking hardcoded to one sorts correctly for
    one model and selects the worst sequences for the other."""
    assert REDESIGN_SCORE_KIND["protein_mpnn"] == SCORE_KIND_NLL
    assert REDESIGN_SCORE_KIND["soluble_mpnn"] == SCORE_KIND_CONFIDENCE
    assert REDESIGN_SCORE_KIND["ligand_mpnn"] == SCORE_KIND_CONFIDENCE
    assert SCORE_KIND_NLL != SCORE_KIND_CONFIDENCE


@pytest.mark.parametrize(
    "configured,is_ligand,expected,warns",
    [
        ("protein_mpnn", False, "protein_mpnn", False),
        ("soluble_mpnn", False, "soluble_mpnn", False),
        ("ligand_mpnn", False, "ligand_mpnn", False),
        ("ligand_mpnn", True, "ligand_mpnn", False),
        ("protein_mpnn", True, "ligand_mpnn", True),
        ("soluble_mpnn", True, "ligand_mpnn", True),
    ],
)
def test_ligand_targets_force_ligand_mpnn_and_say_so(configured, is_ligand, expected, warns):
    """ProteinMPNN does not fail on a ligand -- it finds no CA atoms in that chain
    and drops it, redesigning against a pocket that is not there. Overriding
    silently would leave the mistake in the config to surprise someone later."""
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(m), level="WARNING")
    try:
        assert resolve_inverse_folding_model(configured, is_ligand) == expected
    finally:
        logger.remove(sink)
    assert bool(seen) is warns


def test_an_unknown_inverse_folder_raises():
    with pytest.raises(ValueError, match="not supported"):
        resolve_inverse_folding_model("esm_if", False)


# ------------------------------------------- MPNN failures must say what failed


def mpnn_failure():
    """The helper, without importing the module's heavy dependencies."""
    import pathlib
    import subprocess as sp

    src = pathlib.Path("src/proteinfoundation/metrics/inverse_folding_models.py").read_text()
    i = src.index("def _mpnn_failure(")
    ns = {"subprocess": sp}
    exec(src[i : src.index("\ndef ", i + 10)], ns)
    return ns["_mpnn_failure"]


def test_the_command_is_no_longer_silenced_before_it_can_be_read():
    """`capture_output=True` pipes both streams, and the command then appended
    `> /dev/null 2>&1`, which discarded them first. Every failure arrived as
    "SolubleMPNN command failed: " with nothing after the colon -- which is exactly
    what the CBLN1 smoke test produced."""
    import pathlib

    src = pathlib.Path("src/proteinfoundation/metrics/inverse_folding_models.py").read_text()
    # The code pattern, not any mention: the docstring explaining this bug names
    # /dev/null too, and a test that cannot survive its own fix being documented
    # is a test that will be deleted rather than kept.
    assert '+= " > /dev/null' not in src, "a redirect here destroys the diagnostic the error message asks for"
    assert "2>&1" not in src.replace("``> /dev/null 2>&1``", ""), "no stream may be discarded before capture_output"


def test_a_failure_reports_both_streams_and_the_command():
    import subprocess as sp

    text = str(
        mpnn_failure()(
            "SolubleMPNN",
            "python run.py --model_type soluble_mpnn",
            sp.CalledProcessError(1, "c", output="on stdout", stderr="on stderr"),
        )
    )
    assert "exit 1" in text
    assert "on stderr" in text
    assert "on stdout" in text, "these tools disagree about where errors go, so both are reported"
    assert "--model_type soluble_mpnn" in text, "the command is built from config and is usually the answer"


def test_a_silent_failure_says_so_rather_than_trailing_off():
    import subprocess as sp

    text = str(mpnn_failure()("LigandMPNN", "python run.py", sp.CalledProcessError(1, "c", output=None, stderr=None)))
    assert "no output on either stream" in text, "silence must be reported as silence, not as an empty message"


def test_long_output_is_tailed_not_dropped():
    import subprocess as sp

    err = "\n".join(f"line {i}" for i in range(2000))
    text = str(mpnn_failure()("ProteinMPNN", "python run.py", sp.CalledProcessError(1, "c", output="", stderr=err)))
    assert "line 1999" in text, "the end of a traceback is the part that names the error"
    assert len(text) < 6000
