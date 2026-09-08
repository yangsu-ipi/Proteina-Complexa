"""Sizing a follow-up run from what production actually produced.

Yield cannot be predicted before a production run -- how many raw designs
survive trimming, dedup and the gate is a property of the target -- so a
shortfall is the normal outcome and follow-ups are the normal remedy. These
tests pin the arithmetic that turns "I want 700 more designs" into the four
sizing variables, and the audit record that makes a run reconstructable.
"""

import importlib.util
import itertools
import json
import pathlib

import pytest

TEMPLATES = pathlib.Path(__file__).resolve().parents[1] / ".claude/skills/complexa-target-setup/templates"


def _load():
    spec = importlib.util.spec_from_file_location("plan_followup", TEMPLATES / "plan_followup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan_followup = _load()


# The CBLN1 production run, exactly as it recorded itself.
PROD_OUTPUTS = {
    "status": "passed",
    "raw_generation_rows": 512,
    "retained_before_global_dedup": 500,
    "live_after_global_dedup": 340,
}
PROD_TRIM = {
    "per_shard": 250,
    "shards": {"0": {"generated_rows": 256, "retained": 250}, "1": {"generated_rows": 256, "retained": 250}},
}


def campaign(tmp_path, outputs=None, trim=None) -> pathlib.Path:
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "run_outputs_production.json").write_text(json.dumps(outputs or PROD_OUTPUTS))
    (meta / "shard_trim_production.json").write_text(json.dumps(trim or PROD_TRIM))
    return tmp_path


def observed(tmp_path):
    return plan_followup.observed_yield(campaign(tmp_path), "production", 64)


def test_asking_for_what_production_produced_reproduces_production(tmp_path):
    """The arithmetic's own regression test. If inverting production's yield does
    not return production's parameters, every follow-up is skewed by the same
    factor and nothing in the output would say so."""
    got = plan_followup.plan(2, 5, 2, observed(tmp_path), want_designs=340)
    assert (got["seeds"], got["raw"], got["keep"], got["expect"]) == (64, 512, 250, 500)


def test_a_bigger_ask_scales_the_whole_sizing_set(tmp_path):
    got = plan_followup.plan(2, 5, 2, observed(tmp_path), want_designs=700)
    assert got["seeds"] == 132
    assert got["raw"] == got["seeds"] * 8, "seeds x the observed beam expansion"
    assert got["expect"] == got["keep"] * 2, "keep is per shard"
    assert got["keep"] <= got["raw"] // 2, "cannot retain more than a shard produces"


def test_the_ask_is_never_rounded_down(tmp_path):
    """Asking for 700 and planning for 699 is the failure this exists to remove."""
    obs = observed(tmp_path)
    for want in (1, 7, 341, 700, 1739):
        got = plan_followup.plan(2, 5, 2, obs, want_designs=want)
        assert got["projected_designs"] >= want, f"{want} designs planned short"


def test_seeds_divide_evenly_across_shards(tmp_path):
    """split_by_job hands each shard ceil(n/shards), so an uneven count makes the
    last shard a different size than the trim ratio assumes."""
    obs = observed(tmp_path)
    for want in (1, 100, 700):
        for shards in (2, 3, 4):
            got = plan_followup.plan(shards, 5, 2, obs, want_designs=want)
            assert got["seeds"] % shards == 0, f"{got['seeds']} seeds over {shards} shards"


def test_each_followup_draws_a_seed_no_earlier_run_used(tmp_path):
    """The seed reaching generation is base + job_id, so runs closer together
    than SHARDS would have one follow-up's shard 0 redraw another's shard 1."""
    obs = observed(tmp_path)
    seeds = [plan_followup.plan(2, 5, n, obs, want_designs=100)["rng_seed"] for n in range(1, 7)]
    assert len(set(seeds)) == 6, "every run number draws its own"
    assert seeds[0] == 5, "run 1 IS the campaign's base seed -- the old `production` kind's"
    gaps = {b - a for a, b in itertools.pairwise(seeds)}
    assert gaps == {plan_followup.SEED_STRIDE}, "evenly spaced"
    assert min(gaps) > 64, "spaced far wider than any plausible shard count"


def test_the_index_comes_from_the_records_not_a_config(tmp_path):
    """A campaign's own history is the only state. A follow-up planned but never
    run still consumes its index, which is what stops a seed being reused."""
    root = campaign(tmp_path)
    assert plan_followup.next_index(root) == 1
    (root / "metadata" / "followup_1.json").write_text("{}")
    (root / "metadata" / "followup_2.json").write_text("{}")
    assert plan_followup.next_index(root) == 3


def test_a_stray_metadata_file_does_not_break_numbering(tmp_path):
    root = campaign(tmp_path)
    (root / "metadata" / "followup_notanumber.json").write_text("{}")
    (root / "metadata" / "followup_4.json").write_text("{}")
    assert plan_followup.next_index(root) == 5


def test_no_production_run_is_refused_with_the_reason(tmp_path):
    """Rather than defaulting to some sizing nobody chose."""
    (tmp_path / "metadata").mkdir()
    with pytest.raises(SystemExit, match="production run"):
        plan_followup.observed_yield(tmp_path, "production", 64)


def test_a_production_run_that_produced_nothing_cannot_size_anything(tmp_path):
    """Zero designs means a seed is worth zero, and dividing by it would be a
    crash at best and an absurd sizing at worst."""
    empty = {**PROD_OUTPUTS, "live_after_global_dedup": 0}
    with pytest.raises(SystemExit, match="cannot say what a seed is worth"):
        plan_followup.observed_yield(campaign(tmp_path, outputs=empty), "production", 64)


def test_a_nonsense_ask_is_refused(tmp_path):
    obs = observed(tmp_path)
    for want in (0, -5):
        with pytest.raises(SystemExit):
            plan_followup.plan(2, 5, 2, obs, want_designs=want)


def test_the_record_holds_everything_needed_to_reconstruct_the_run(tmp_path):
    """Including the seed, which is the one parameter that cannot be recovered
    from the outputs afterwards."""
    got = plan_followup.plan(2, 5, 4, observed(tmp_path), want_designs=700)
    for key in (
        "seeds",
        "raw",
        "keep",
        "expect",
        "rng_seed",
        "shards",
        "want_designs",
        "reference_kind",
        "reference_seeds",
        "reference_designs",
        "designs_per_seed",
    ):
        assert key in got, f"{key} missing from the audit record"
    assert json.loads(json.dumps(got)), "and it has to serialise"


def test_a_worse_yielding_target_needs_more_seeds(tmp_path):
    """The derivation reads the target's actual yield, so a target that dedups
    harder asks for proportionally more."""
    generous = observed(tmp_path)
    stingy = plan_followup.observed_yield(
        campaign(tmp_path / "b", outputs={**PROD_OUTPUTS, "live_after_global_dedup": 170}), "production", 64
    )
    assert plan_followup.plan(2, 5, 2, stingy, want_designs=700)["seeds"] > plan_followup.plan(2, 5, 2, generous, want_designs=700)["seeds"]


# ---------------------------------------------------------------------------
# Cross-run deduplication. A follow-up samples the same target from the same
# model as the run it follows, so it regenerates designs that run already has.
# ---------------------------------------------------------------------------


def _run_dir(campaign, config, task, name, aatypes):
    d = campaign / "inference" / f"{config}_{task}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    rows = "aatype,total_reward\n" + "".join(f'"{a}",1.0\n' for a in aatypes)
    (d / f"top_samples_{config}.csv").write_text(rows)
    return d


def test_the_pool_is_production_and_earlier_followups(tmp_path):
    root = campaign(tmp_path)
    for name in ("p_production", "p_followup1", "p_followup2", "p_smoke"):
        _run_dir(root, "cfg", "TASK", name, ["1,2", "3,4"])
    pool = plan_followup.pool_dirs(root, "cfg", "TASK", "p", before_run=4)
    names = [pathlib.Path(d).name for d in pool]
    assert names == ["cfg_TASK_p_production", "cfg_TASK_p_followup1", "cfg_TASK_p_followup2"]


def test_the_smoke_run_never_claims_a_sequence(tmp_path):
    """Smoke designs are a throwaway check. Letting one claim a sequence would
    make a production design vanish because a test drew it first."""
    root = campaign(tmp_path)
    _run_dir(root, "cfg", "TASK", "p_production", ["1,2"])
    _run_dir(root, "cfg", "TASK", "p_smoke", ["9,9"])
    pool = plan_followup.pool_dirs(root, "cfg", "TASK", "p", before_run=2)
    # basenames, because pytest puts this test's own name in tmp_path
    assert all("smoke" not in pathlib.Path(d).name for d in pool)
    assert len(pool) == 1


def test_a_run_that_never_filtered_is_refused_not_skipped(tmp_path):
    """Skipping would under-deduplicate silently, and the duplicates it let
    through could not be identified afterwards."""
    root = campaign(tmp_path)
    _run_dir(root, "cfg", "TASK", "p_production", ["1,2"])
    (root / "inference" / "cfg_TASK_p_followup1").mkdir(parents=True)
    with pytest.raises(SystemExit, match="is missing"):
        plan_followup.pool_dirs(root, "cfg", "TASK", "p", before_run=3)


def test_no_completed_run_is_refused(tmp_path):
    root = campaign(tmp_path)
    (root / "inference").mkdir(exist_ok=True)
    with pytest.raises(SystemExit, match="no completed run"):
        plan_followup.pool_dirs(root, "cfg", "TASK", "p", before_run=2)


def test_the_pooled_keys_are_the_union_of_what_each_run_kept(tmp_path):
    from proteinfoundation.utils.run_pooling import pooled_aatypes, retained_aatypes

    root = campaign(tmp_path)
    a = _run_dir(root, "cfg", "TASK", "p_production", ["1,2", "3,4"])
    b = _run_dir(root, "cfg", "TASK", "p_followup1", ["3,4", "5,6"])
    assert retained_aatypes(str(a), "cfg") == {"1,2", "3,4"}
    assert pooled_aatypes([str(a), str(b)], "cfg") == {"1,2", "3,4", "5,6"}


def test_a_missing_filter_output_is_loud(tmp_path):
    from proteinfoundation.utils.run_pooling import retained_aatypes

    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="is missing"):
        retained_aatypes(str(tmp_path / "empty"), "cfg")


def test_the_wrong_column_is_caught_rather_than_matching_nothing(tmp_path):
    """aatype is comma-joined integers, not the letter sequence. Comparing the
    wrong representation would deduplicate nothing and look like it worked."""
    from proteinfoundation.utils.run_pooling import retained_aatypes

    d = tmp_path / "r"
    d.mkdir()
    (d / "top_samples_cfg.csv").write_text("binder_sequence,total_reward\nMKV,1.0\n")
    with pytest.raises(KeyError, match="aatype"):
        retained_aatypes(str(d), "cfg")


def test_the_manifest_round_trips(tmp_path):
    from proteinfoundation.utils.run_pooling import read_pool_manifest

    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"for_run": "x", "inference_dirs": ["/a", "/b"]}))
    assert read_pool_manifest(str(path)) == ["/a", "/b"]
    path.write_text(json.dumps(["/a"]))
    assert read_pool_manifest(str(path)) == ["/a"], "a bare list is accepted too"
    path.write_text(json.dumps({"inference_dirs": "not a list"}))
    with pytest.raises(ValueError, match="list of inference directories"):
        read_pool_manifest(str(path))


# ---------------------------------------------------------------------------
# The pooled report. A campaign reaches its number over several runs, so the
# campaign total is not any one run's analyze output.
# ---------------------------------------------------------------------------


def test_the_pool_membership_rule_is_shared_not_duplicated():
    """The planner deduplicates against these runs and the report counts over
    them. Two copies of the rule would drift, and the failure is silent both
    ways: designs deduplicated against a run the report ignores, or counted from
    a run the dedup never saw."""
    from proteinfoundation.utils.run_pooling import is_pooled_run

    for suffix, expected in (
        ("production", True),
        ("followup1", True),
        ("followup12", True),
        ("smoke", False),
        ("smoke_bw8", False),
        ("followupX", False),
        ("", False),
    ):
        name = f"cfg_TASK_p_{suffix}" if suffix else "cfg_TASK_p_"
        assert is_pooled_run(name, "cfg", "TASK", "p") is expected, suffix


def test_an_unanticipated_smoke_variant_is_excluded_by_default():
    """Matched rather than pattern-excluded: _smoke_bw8 exists in a real campaign
    and nobody wrote a rule for it."""
    from proteinfoundation.utils.run_pooling import is_pooled_run

    assert not is_pooled_run("cfg_TASK_p_smoke_bw8", "cfg", "TASK", "p")


def test_pooled_runs_are_ordered_production_first(tmp_path):
    from proteinfoundation.utils.run_pooling import pooled_run_dirs

    root = tmp_path / "evaluation_results"
    for suffix in ("followup2", "production", "followup10", "followup1", "smoke"):
        (root / f"cfg_TASK_p_{suffix}").mkdir(parents=True)
    got = [pathlib.Path(d).name.split("_p_")[-1] for d in pooled_run_dirs(str(root), "cfg", "TASK", "p")]
    assert got == ["production", "followup1", "followup2", "followup10"]


def test_a_sequence_in_two_runs_is_reported_as_an_overcount():
    """Cross-run dedup should make this impossible. If it happens, a run was
    filtered without its pool manifest and every pooled count is too high."""
    import pandas as pd

    from proteinfoundation.analyze_pooled import duplicate_audit

    df = pd.DataFrame(
        {"binder_sequence": ["AAA", "BBB", "AAA"], "pooled_run": ["r_production", "r_production", "r_followup1"]}
    )
    audit = duplicate_audit(df)
    assert audit["sequences_in_more_than_one_run"] == 1
    assert audit["unique_sequences"] == 2 and audit["rows"] == 3


def test_a_clean_pool_reports_no_duplicates():
    import pandas as pd

    from proteinfoundation.analyze_pooled import duplicate_audit

    df = pd.DataFrame({"binder_sequence": ["AAA", "BBB"], "pooled_run": ["r_production", "r_followup1"]})
    assert duplicate_audit(df)["sequences_in_more_than_one_run"] == 0


def test_the_summary_counts_sequences_not_designs():
    """A design carries several sequences, and it is sequences that get ordered."""
    import pandas as pd

    from proteinfoundation.analyze_pooled import summarise

    df = pd.DataFrame(
        {
            "pooled_run": ["r_production", "r_followup1"],
            "self_pass_all": ["[1]", "[0]"],
            "mpnn_pass_all": ["[1, 0]", "[1, 1]"],
            "self_pass": [1, 0],
            "mpnn_pass": [1, 1],
        }
    )
    got = summarise(df, ["self", "mpnn"])
    assert got["pooled"]["sequences"] == 6, "1 self + 2 mpnn per design"
    assert got["pooled"]["orderable_sequences"] == 4
    assert got["pooled"]["designs_with_a_passing_sequence"] == 2
    assert set(got["per_run"]) == {"r_production", "r_followup1"}
    assert got["per_run"]["r_production"]["orderable_sequences"] == 2


def test_a_design_counts_when_only_its_redesign_passes():
    """The headline {seq_type}_pass column is the first sequence's verdict alone.

    A design whose original sequence fails and whose redesign passes still has
    an orderable sequence, so it belongs in the design count -- reading the
    headline column instead of the per-sequence vector undercounted exactly
    those designs, while orderable_sequences (read from the vector) was right.
    """
    import pandas as pd

    from proteinfoundation.analyze_pooled import summarise

    df = pd.DataFrame(
        {
            "pooled_run": ["r"],
            "self_pass_all": ["[0]"],
            "mpnn_pass_all": ["[0, 1]"],
            "self_pass": [0],
            "mpnn_pass": [0],  # what the headline says: nothing passed
        }
    )
    got = summarise(df, ["self", "mpnn"])["pooled"]
    assert got["orderable_sequences"] == 1
    assert got["designs_with_a_passing_sequence"] == 1


def test_the_design_count_agrees_with_the_sequence_count():
    """Neither can be nonzero while the other is zero: both are read off the
    same per-sequence vectors."""
    import pandas as pd

    from proteinfoundation.analyze_pooled import summarise

    none_pass = pd.DataFrame({"pooled_run": ["r", "r"], "self_pass_all": ["[0, 0]", "[0]"]})
    got = summarise(none_pass, ["self"])["pooled"]
    assert got["orderable_sequences"] == 0
    assert got["designs_with_a_passing_sequence"] == 0


def test_stringified_verdicts_survive_the_csv_round_trip():
    """The pooled frame is read back with pd.read_csv, which returns the text of
    a list column. Counting its characters is a bug this codebase has already
    shipped once."""
    import pandas as pd

    from proteinfoundation.analyze_pooled import summarise

    text = pd.DataFrame({"pooled_run": ["r"], "self_pass_all": ["[1, 0, 1]"]})
    real = pd.DataFrame({"pooled_run": ["r"], "self_pass_all": [[1, 0, 1]]})
    assert summarise(text, ["self"])["pooled"] == summarise(real, ["self"])["pooled"]
    assert summarise(text, ["self"])["pooled"]["sequences"] == 3


# ---------------------------------------------------------------------------
# One numbered sequence, three spellings.
#
# `production`, `followup{K}` and `production{N}` all name runs in one sequence:
# production is run 1 and followup{K} is run K+1. The merge is a rename, not a
# renumbering -- seeds are base + (number - 1) * stride, which is exactly what
# the two old kinds produced, so no campaign on disk has to move.
# ---------------------------------------------------------------------------


def test_the_three_spellings_number_one_sequence():
    for suffix, number in [
        ("production", 1),
        ("followup1", 2),
        ("followup2", 3),
        ("production1", 1),
        ("production7", 7),
    ]:
        assert plan_followup.run_number(suffix) == number, suffix


def test_a_smoke_run_is_not_part_of_the_sequence():
    for suffix in ("smoke", "smoke_bw8", "followupX", "productionX", ""):
        assert plan_followup.run_number(suffix) is None, suffix


def test_the_planner_and_the_pipeline_agree_on_every_spelling():
    """The planner carries its own copy of the numbering, because
    submit_campaign.sh runs it with plain python3 before any conda environment
    exists, so it cannot import the pipeline's. A forced duplicate is only safe
    while something compares the two."""
    from proteinfoundation.utils import run_pooling

    for suffix in [
        "production", "production1", "production2", "production40",
        "followup1", "followup2", "followup10",
        "smoke", "smoke_bw8", "followup", "productionX", "", "prod",
    ]:
        assert plan_followup.run_number(suffix) == run_pooling.run_index(suffix), suffix
    for number in (1, 2, 3, 17):
        assert plan_followup.run_suffix(number) == run_pooling.run_suffix(number)


def test_a_run_keeps_the_seed_its_old_kind_drew(tmp_path):
    """The rename must not redraw anything. production used the base seed and
    followup{K} used base + K * stride; under the numbering those are runs 1 and
    K+1, and base + (number - 1) * stride reproduces both."""
    obs = observed(tmp_path)
    base = 5
    assert plan_followup.plan(2, base, 1, obs, seeds=64)["rng_seed"] == base
    for legacy_k in (1, 2, 3):
        planned = plan_followup.plan(2, base, legacy_k + 1, obs, seeds=64)
        assert planned["rng_seed"] == base + legacy_k * plan_followup.SEED_STRIDE
        assert planned["index"] == legacy_k, "and still records the legacy index its files are named by"


def test_a_run_is_sized_by_seeds_or_by_designs_but_not_both(tmp_path):
    obs = observed(tmp_path)
    assert plan_followup.plan(2, 5, 2, obs, seeds=200)["seeds"] == 200
    with pytest.raises(SystemExit, match="not both and not neither"):
        plan_followup.plan(2, 5, 2, obs, seeds=200, want_designs=700)
    with pytest.raises(SystemExit, match="not both and not neither"):
        plan_followup.plan(2, 5, 2, obs)
