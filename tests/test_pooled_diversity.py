"""Pooled structural diversity: the subset and cross-run-cluster logic.

Two clusterings cannot be added together, so a campaign's backbone diversity
needs one clustering over every run. These test the parts that decide what gets
clustered and how the result is attributed -- the foldseek call itself needs the
binary and is exercised on the cluster.
"""

import importlib.util
import pathlib

import pandas as pd

TEMPLATES = pathlib.Path(__file__).resolve().parents[1] / ".claude/skills/complexa-target-setup/templates"
spec = importlib.util.spec_from_file_location("pooled_diversity", TEMPLATES / "pooled_diversity.py")
pooled_diversity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pooled_diversity)


def frame():
    return pd.DataFrame(
        {
            "pooled_run": ["a", "a", "b", "b"],
            "self_pass_all": ["[1]", "[0]", "[0]", "[1]"],
            "mpnn_pass_all": ["[0, 0]", "[1, 0]", "[0, 0]", "[0, 0]"],
        }
    )


def test_verdict_vectors_are_read_as_lists_not_text():
    """The pooled CSV comes back from read_csv as text, and treating '[0, 0]'
    as truthy would put every design in every successful subset."""
    assert pooled_diversity._passes("[1, 0]")
    assert not pooled_diversity._passes("[0, 0]")
    assert not pooled_diversity._passes("[]")
    assert pooled_diversity._passes([1])
    assert not pooled_diversity._passes("not a list")


def test_the_subsets_are_all_plus_what_survives_the_gate():
    got = dict(pooled_diversity.subsets(frame(), ["self", "mpnn"]))
    assert set(got) == {"all_generated", "successful_self", "successful_mpnn", "successful_any"}
    assert len(got["all_generated"]) == 4
    assert len(got["successful_self"]) == 2
    assert len(got["successful_mpnn"]) == 1
    assert len(got["successful_any"]) == 3, "a design counts once however many of its sequences pass"


def test_an_empty_subset_is_dropped_rather_than_clustered():
    """FoldSeek on nothing is an error, and a zero-structure cluster count would
    read as perfect redundancy rather than as no data."""
    df = frame()
    df["mpnn_pass_all"] = "[0, 0]"
    assert "successful_mpnn" not in dict(pooled_diversity.subsets(df, ["self", "mpnn"]))


def test_a_missing_sequence_type_is_skipped():
    df = frame().drop(columns=["mpnn_pass_all"])
    got = dict(pooled_diversity.subsets(df, ["self", "mpnn"]))
    assert "successful_mpnn" not in got and "successful_self" in got


def test_clusters_are_attributed_to_the_runs_that_contributed_them(tmp_path):
    """The question the pooled clustering exists to answer: a follow-up that
    only resamples production's folds shows up as shared clusters and none of
    its own."""
    df = pd.DataFrame({"pooled_run": ["prod", "prod", "follow", "follow"]})
    csv = tmp_path / "cluster_assignments.csv"
    csv.write_text(
        "cluster_index,sample_index,path_name\n"
        "0,0,x\n0,2,x\n"  # a fold both runs found
        "1,1,x\n"  # production only
        "2,3,x\n"  # follow-up only
    )
    got = pooled_diversity.cluster_split_by_run(str(csv), df)
    assert got["clusters"] == 3
    assert got["shared_by_runs"] == 1
    assert got["exclusive_to"] == {"prod": 1, "follow": 1}


def test_a_missing_assignments_file_reports_unavailable(tmp_path):
    df = pd.DataFrame({"pooled_run": ["a"]})
    assert pooled_diversity.cluster_split_by_run(str(tmp_path / "absent.csv"), df) == {"available": False}


def test_the_settings_match_what_analyze_uses_for_binder_diversity():
    """So a pooled number is comparable with the per-run ones already on disk
    rather than being a second, differently-calibrated measurement."""
    analyze = (pathlib.Path(__file__).resolve().parents[1] / "src/proteinfoundation/analyze.py").read_text()
    block = analyze[analyze.index('diversity_mode="binder"') - 300 : analyze.index('diversity_mode="binder"')]
    assert f"min_seq_id={pooled_diversity.MIN_SEQ_ID}" in block
    assert f"alignment_type={pooled_diversity.ALIGNMENT_TYPE}" in block
