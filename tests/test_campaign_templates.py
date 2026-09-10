"""The campaign script templates, run rather than read.

These live in `.claude/skills/complexa-target-setup/templates/` and are copied
verbatim into every new campaign package. That makes their failure mode worse than
prose: a wrong template is copied confidently into every future campaign, and the
one thing that stops it is exercising them here.

The bugs they encode fixes for were all found the expensive way, on a GPU box, in
the CBLN1 campaign:

  * a trim step that counted designs in the output root alone, and stopped a run
    whose generation had correctly skipped, because the filter had moved them
  * a preflight that demanded ESMFold2 imports of every campaign
  * a verifier that assumed exactly two shards and a magic retained-count of 8
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

TEMPLATES = Path(__file__).resolve().parents[1] / ".claude/skills/complexa-target-setup/templates"


def run(script, *args):
    return subprocess.run([sys.executable, str(TEMPLATES / script), *map(str, args)], capture_output=True, text=True)


def design(root: Path, name: str, reward: float):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.pdb").write_text("ATOM\n")
    return {"pdb_path": str(d / f"{name}.pdb"), "total_reward": str(reward), "aatype": name}


def rewards_csv(root: Path, shard: int, rows):
    p = root / f"rewards_pipeline_{shard}.csv"
    p.write_text(
        "pdb_path,total_reward,aatype\n" + "".join(f"{r['pdb_path']},{r['total_reward']},{r['aatype']}\n" for r in rows)
    )
    return p


def test_every_template_is_syntactically_runnable():
    """The cheapest guard, and it would have caught a bad copy-paste."""
    for script in sorted(TEMPLATES.glob("*.py")):
        r = subprocess.run([sys.executable, "-c", f"compile(open({str(script)!r}).read(), '{script.name}', 'exec')"])
        assert r.returncode == 0, f"{script.name} does not compile"


def test_trim_keeps_n_per_shard_and_sets_the_rest_aside(tmp_path):
    inf = tmp_path / "inference"
    for shard in (0, 1):
        rewards_csv(inf, shard, [design(inf, f"job_{shard}_n_100_id_{i}", 10 - i) for i in range(4)])
    out = tmp_path / "trim.json"
    r = run("trim_shards.py", "--inference-dir", inf, "--per-shard", 2, "--shards", 2, "--output", out)
    assert r.returncode == 0, r.stderr

    report = json.loads(out.read_text())
    assert sum(v["retained"] for v in report["shards"].values()) == 4
    aside = inf / "filtered_out_samples" / "pre_filter_shard_trim"
    assert len(list(aside.iterdir())) == 4, "the other four are set aside, not deleted"


def test_trim_is_resumable_after_the_filter_has_moved_things(tmp_path):
    """The bug that stopped a real campaign. Generation had correctly skipped both
    shards; trim then counted designs in the root, found 2 of the 4 it expected --
    the filter having moved the rest -- and exited."""
    inf = tmp_path / "inference"
    rows = {s: [design(inf, f"job_{s}_n_100_id_{i}", 10 - i) for i in range(4)] for s in (0, 1)}
    for shard, rs in rows.items():
        rewards_csv(inf, shard, rs)
    out = tmp_path / "trim.json"
    assert (
        run("trim_shards.py", "--inference-dir", inf, "--per-shard", 2, "--shards", 2, "--output", out).returncode == 0
    )

    # what the filter does next: retained designs move out of the root too
    filtered = inf / "filtered_out_samples"
    for kept in [p for p in inf.iterdir() if p.is_dir() and p.name.startswith("job_")]:
        kept.rename(filtered / kept.name)

    second = run("trim_shards.py", "--inference-dir", inf, "--per-shard", 2, "--shards", 2, "--output", out)
    assert second.returncode == 0, f"a resumed trim must not fail: {second.stderr}"
    assert "require" not in second.stderr


def test_preflight_asks_for_esmfold2_only_when_the_config_uses_it(tmp_path):
    """A plain-ESMFold campaign should not be failed for lacking ESMFold2, which
    the original could not express."""
    report = tmp_path / "preflight.json"
    report.write_text(
        json.dumps(
            {
                "gpu": {"available": True, "vram_gb": 80},
                "checkpoints": {"complexa.ckpt": {"exists": True}, "complexa_ae.ckpt": {"exists": True}},
                "community_models": {"AF2_DIR": {"exists": True}},
                "tools": {"foldseek": {"exists": True}, "mmseqs": {"exists": True}},
                "disk": {"cwd_free_gb": 9999},
                "env": {},
            }
        )
    )
    cfg = tmp_path / "resolved.yaml"
    cfg.write_text(
        yaml.safe_dump({"metric": {"binder_folding_method": "colabdesign", "apo_folding_models": ["esmfold"]}})
    )

    r = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 100)
    assert "ESMC/ESMFold2 imports failed" not in r.stdout, r.stdout
    assert "HF cache lacks" not in r.stdout, "no ESM model configured, so none should be required"


def test_preflight_reports_a_low_vram_card_against_the_configured_floor(tmp_path):
    report = tmp_path / "preflight.json"
    report.write_text(
        json.dumps({"gpu": {"available": True, "vram_gb": 24}, "checkpoints": {}, "tools": {}, "disk": {}, "env": {}})
    )
    cfg = tmp_path / "resolved.yaml"
    cfg.write_text(yaml.safe_dump({"metric": {}}))

    assert "<40 GB VRAM" in run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 1).stdout
    ok = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 1, "--min-vram-gb", 16)
    assert "VRAM" not in ok.stdout, "24 GB clears a 16 GB floor"


def test_checksums_cover_the_package_and_skip_what_a_run_produces(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "scripts").mkdir(parents=True)
    (pkg / "scripts" / "refresh_checksums.py").write_text((TEMPLATES / "refresh_checksums.py").read_text())
    (pkg / "pipeline.yaml").write_text("a: 1\n")
    (pkg / "inference").mkdir()
    (pkg / "inference" / "design.pdb").write_text("ATOM\n")

    assert subprocess.run([sys.executable, str(pkg / "scripts" / "refresh_checksums.py")]).returncode == 0
    listed = (pkg / "CHECKSUMS.sha256").read_text()
    assert "pipeline.yaml" in listed
    assert "inference" not in listed, "run output is not part of the package"


@pytest.mark.parametrize("shards,retained", [(2, 4), (4, 8)])
def test_the_verifier_is_not_pinned_to_two_shards(tmp_path, shards, retained):
    """It globbed rewards_pipeline_[01].csv and derived its trim report from a
    magic retained-count of 8."""
    inf, ev = tmp_path / "inference", tmp_path / "evaluation"
    inf.mkdir()
    ev.mkdir()
    for s in range(shards):
        rewards_csv(inf, s, [design(inf, f"job_{s}_n_100_id_{i}", 1.0) for i in range(retained // shards)])
    trim = tmp_path / "trim.json"
    trim.write_text(
        json.dumps(
            {
                "per_shard": retained // shards,
                "shards": {str(s): {"retained": retained // shards} for s in range(shards)},
            }
        )
    )
    cfg = tmp_path / "resolved.yaml"
    cfg.write_text(yaml.safe_dump({"metric": {"compute_binder_metrics": True, "compute_monomer_metrics": False}}))

    r = run(
        "verify_run_outputs.py",
        "--inference-dir",
        inf,
        "--evaluation-dir",
        ev,
        "--expected-retained",
        retained,
        "--resolved-config",
        cfg,
        "--output",
        tmp_path / "out.json",
        "--shards",
        shards,
        "--trim-report",
        trim,
    )
    # It fails later (no timing/results here), but never on shard arithmetic.
    assert "--shards says" not in r.stderr
    assert "unequal shard retention" not in r.stderr
    assert "retained" not in r.stderr or "expected" not in r.stderr


# ----------------------------- the runner and its config must agree


RUNNER = TEMPLATES / "run_campaign.sh"
CONFIG_EXAMPLE = TEMPLATES / "campaign.env.example"


def shell_vars(text):
    """Variables a shell script reads, and the ones it assigns."""
    read = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]{2,})\b", text))
    # Anywhere on a line, not only at its start: the runner's `case` arms pack
    # several assignments onto one line with semicolons, which is legitimate shell
    # and invisible to a line-anchored pattern.
    assigned = set(re.findall(r"(?:^|;|\s)(?:export\s+|local\s+)?([A-Z][A-Z0-9_]{2,})=", text, re.M))
    return read, assigned


def test_the_runner_reads_nothing_the_config_does_not_define():
    """The load-bearing structural check. If the runner reads a variable that
    campaign.env.example does not set, the next campaign discovers it by crashing
    -- or worse, by an agent editing the template, which is what templating was
    meant to stop."""
    runner = RUNNER.read_text()
    read, assigned = shell_vars(runner)
    provided, _ = shell_vars(CONFIG_EXAMPLE.read_text())
    _, config_sets = shell_vars(CONFIG_EXAMPLE.read_text())

    environmental = {
        "BASH_SOURCE",
        "SLURM_JOB_ID",
        "USER",
        "HOME",
        "PATH",
        "KIND",
        "STAGE",
        "COMMUNITY_MODELS_PATH",
        "CUDA_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "CCD_MIRROR_PATH",
        "PDB_MIRROR_PATH",
        # Assigned by `eval "$PLAN"` from plan_followup.py's output, so the
        # parser cannot see the assignment. Listed rather than ignored, because
        # a typo in one of these names is exactly what this test is for.
        "FOLLOWUP_RUN_NAME",
        # The run's position in the campaign and the name it goes by. Both come
        # from the planner, which is the single authority on them -- the runner
        # rebuilding either would let a run's metadata files disagree with its
        # inference directory.
        "RUN_NUMBER",
        "RUN_TAG",
        "FOLLOWUP_SEEDS",
        "FOLLOWUP_RAW",
        "FOLLOWUP_KEEP",
        "FOLLOWUP_EXPECT",
        "FOLLOWUP_RNG_SEED",
        "FOLLOWUP_INDEX",
        "FOLLOWUP_RECORD",
        "FOLLOWUP_POOL_MANIFEST",
    }
    unresolved = read - assigned - config_sets - environmental
    assert not unresolved, f"runner reads variables nothing defines: {sorted(unresolved)}"


def test_the_config_example_is_valid_shell():
    assert subprocess.run(["bash", "-n", str(CONFIG_EXAMPLE)]).returncode == 0


@pytest.mark.parametrize("script", ["run_campaign.sh", "campaign.sbatch"])
def test_shell_templates_parse(script):
    assert subprocess.run(["bash", "-n", str(TEMPLATES / script)]).returncode == 0


def test_the_runner_carries_no_campaign_identity():
    """Everything specific lives in campaign.env. The provenance comment naming the
    campaign that validated this is the one allowed exception, and it is a comment."""
    for line in RUNNER.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert not re.search(r"cbln1|5kc5|glud2", line, re.I), f"campaign identity leaked into: {line.strip()}"


def test_the_runner_does_not_reintroduce_the_output_directory_guard():
    """The single line that disabled resume, and which the documentation used to
    ask for. It is easy to add back while 'tidying'."""
    text = RUNNER.read_text()
    live = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    assert "! -e " not in live, "an existence guard on the output directory disables resume"
    assert "refusing to generate over" not in live


def test_the_runner_pins_one_shard_per_gpu():
    """Both shards on card 0 with the other idle, twice, on a real box."""
    text = RUNNER.read_text()
    assert 'CUDA_VISIBLE_DEVICES="$shard"' in text
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in text, "JAX preallocates 75% of the card otherwise"


def source_config(**env):
    """Source campaign.env.example under a given environment and report the result."""
    probe = "; ".join(
        f"echo {v}=${v}" for v in ("XLA_MEM_FRACTION_GENERATE", "XLA_MEM_FRACTION_EVALUATE", "MIN_VRAM_GB", "TASK_NAME")
    )
    r = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", f"source {CONFIG_EXAMPLE}; {probe}"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "USER": "u", **env},
    )
    assert r.returncode == 0, r.stderr
    return dict(line.split("=", 1) for line in r.stdout.strip().splitlines())


def test_the_gpu_knobs_take_an_environment_override():
    """So a fraction can be tried for one run without editing the file -- which is
    exactly the experiment these numbers came from."""
    assert source_config()["XLA_MEM_FRACTION_EVALUATE"] == "0.3"
    assert source_config(XLA_MEM_FRACTION_EVALUATE="0.25")["XLA_MEM_FRACTION_EVALUATE"] == "0.25"
    assert source_config(MIN_VRAM_GB="24")["MIN_VRAM_GB"] == "24"


def test_identity_values_do_not_take_an_override():
    """Overriding the task name from the environment would silently run a different
    campaign against the same package, so those stay plain assignments."""
    assert source_config(TASK_NAME="SOMETHING_ELSE")["TASK_NAME"] == "CBLN1_5KC5_GLUD2"


def test_the_runner_checks_the_configs_env_vars_before_spending_anything():
    """An unset ${oc.env:VAR} used to surface as an omegaconf KeyError after the
    checkpoint had loaded. This runs the template's own detection pipeline rather
    than a copy of it, so the test cannot drift away from what ships."""
    runner = RUNNER.read_text()
    m = re.search(r"< <\((grep -oE .*?\| sort -u)\)", runner, re.S)
    assert m, "the env-var scan is missing from run_campaign.sh"

    probe = Path(__file__).parent / "_probe.yaml"
    probe.write_text(
        "root: ${oc.env:LEGACY_CAMPAIGN_DIR}\nok:   ${oc.env:HOME}\ndef:  ${oc.env:HAS_DEFAULT,/fallback}\n"
    )
    try:
        script = f'''set -euo pipefail
CONFIG="{probe}"
required_env=()
while read -r var; do [[ -z "$var" ]] || [[ -n "${{!var:-}}" ]] || required_env+=("$var"); done < <({m.group(1)})
echo "${{required_env[*]:-}}"'''
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        named = r.stdout.split()
        assert named == ["LEGACY_CAMPAIGN_DIR"], f"expected only the unset no-default var, got {named}"
    finally:
        probe.unlink(missing_ok=True)


def test_the_prepare_hook_survives_an_older_campaign_env():
    """A campaign.env predating PREPARE_STEPS leaves it unset, and the runner uses
    `set -u`. The guard must tolerate unset, empty, and populated alike."""
    guard = "if [[ ${PREPARE_STEPS+set} == set ]] && ((${#PREPARE_STEPS[@]})); then echo RUN; else echo SKIP; fi"
    for setup, expected in [("true", "SKIP"), ("PREPARE_STEPS=()", "SKIP"), ("PREPARE_STEPS=(a b)", "RUN")]:
        # `true` rather than "" for the unset case: an empty fragment leaves a
        # doubled semicolon, which is a shell syntax error and not the thing under test.
        r = subprocess.run(["bash", "-c", f"set -euo pipefail; {setup}; {guard}"], capture_output=True, text=True)
        assert r.returncode == 0, f"{setup!r} broke under set -u: {r.stderr}"
        assert r.stdout.strip() == expected, f"{setup!r} -> {r.stdout.strip()}, wanted {expected}"


def test_the_runner_actually_uses_that_guard():
    """The test above is only worth anything if it guards the shipped code."""
    assert "${PREPARE_STEPS+set} == set" in RUNNER.read_text()


def test_preparation_runs_before_anything_validates():
    """An MSA that a later step reads has to exist by then."""
    text = RUNNER.read_text()
    assert text.index("PREPARE_STEPS") < text.index("validate_resolved_config.py")
    assert text.index("PREPARE_STEPS") < text.index("required_env=()")


def test_the_templates_have_no_undefined_names():
    """`verify_run_outputs.py` reached its final line -- after a full cold run --
    and died on `NameError: required`, left behind when its hardcoded column list
    became an argument. Compiling catches syntax; only a name check catches a
    reference that no longer resolves on a path taken once per run.

    ruff's F821 is exactly this check. The reason it did not fire is duller than a
    missing tool: nothing ever ran ruff over this directory. It does now.
    """
    r = subprocess.run(
        ["ruff", "check", "--select", "F821", "--no-cache", *[str(p) for p in sorted(TEMPLATES.glob("*.py"))]],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout


# ---------------------------------------------------------------------------
# The scale kind. A larger draw than production, with its own RNG seed.
# ---------------------------------------------------------------------------


def test_the_runner_accepts_exactly_the_kinds_it_documents():
    runner = RUNNER.read_text()
    case = runner[runner.index('case "$KIND" in') : runner.index("esac")]
    arms = {a.strip() for line in case.splitlines() if ")" in line for a in [line.split(")")[0]] if a.strip().isalpha()}
    assert arms == {"smoke", "production", "followup", "pooled"}
    usage = runner[: runner.index('case "$KIND" in')]
    for kind in arms:
        assert kind in usage, f"{kind} missing from the usage line"


def test_followup_takes_a_design_count_and_nothing_else():
    """The one number that cannot be predicted before a production run. Anything
    else on the command line would be a number the user had to work out."""
    runner = RUNNER.read_text()
    assert 'WANT_DESIGNS="${1:?' in runner, "required, with a message"
    assert 'STAGE="all"' in runner and 'STAGE="$1"' in runner, "the stage is optional and shifts along"


def test_a_followup_gets_its_own_run_name_and_metadata():
    """RUN_NAME feeds both output directories and KIND_TAG feeds the metadata, so
    two follow-ups sharing either would overwrite each other -- and generation
    refuses to write into a shard whose marker disagrees, so the second would
    simply not run."""
    runner = RUNNER.read_text()
    assert 'RUN_NAME="$FOLLOWUP_RUN_NAME"' in runner
    # The tag comes from the planner rather than being rebuilt here, so a run's
    # metadata files carry the same name as its inference directory whichever
    # spelling it uses -- `followup1` for a run written before the kinds merged,
    # `production4` for one written after.
    assert 'KIND_TAG="$RUN_TAG"' in runner
    for path in ("resolved_config_", "shard_trim_", "preflight_", "run_outputs_"):
        assert f"{path}${{KIND_TAG}}" in runner, f"{path} is not per-run"


def test_the_run_name_exists_before_the_paths_built_from_it():
    """INF and EVAL interpolate RUN_NAME, and the follow-up derives it at run
    time -- so the derivation has to come first or `set -u` aborts every
    follow-up."""
    runner = RUNNER.read_text()
    assert runner.index('RUN_NAME="$FOLLOWUP_RUN_NAME"') < runner.index('INF="$CAMPAIGN_DIR/inference')


def test_nothing_needs_editing_to_size_a_followup():
    """The whole point: no SCALE_SEEDS to compute, no config to touch."""
    config = CONFIG_EXAMPLE.read_text()
    assert "SCALE_" not in config
    assert "FOLLOWUP_SEEDS=" not in config, "a follow-up derives its size, it is not configured"


def test_a_retrofitted_campaign_env_is_told_what_to_add():
    """An older campaign.env predates the seed variables. set -u alone would say
    only 'unbound variable'; the guard names the file to edit. PRODUCTION_SEEDS
    and PRODUCTION_RNG_SEED are load-bearing for follow-ups, which read them as
    the reference run's parameters."""
    runner = RUNNER.read_text()
    for var in ("SMOKE_RNG_SEED", "PRODUCTION_RNG_SEED", "PRODUCTION_SEEDS"):
        assert f"${{{var}:?" in runner, f"{var} is read without a message naming campaign.env"


# ---------------------------------------------------------------------------
# Submitting a run as a dependency chain.
# ---------------------------------------------------------------------------

SUBMIT = TEMPLATES / "submit_campaign.sh"
GENERIC_SBATCH = TEMPLATES / "campaign.sbatch.generic"


def test_the_submitter_and_generic_sbatch_parse():
    for script in (SUBMIT, GENERIC_SBATCH):
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0, script.name


def test_every_stage_waits_on_the_one_before_it():
    """afterok, not afterany: a stage that runs on the output of a job that
    failed produces a result nobody can trust, and the failure is upstream."""
    text = SUBMIT.read_text()
    assert "afterok:" in text and "--dependency=" in text
    assert "afterany" not in text


def test_only_the_folding_stages_ask_for_gpus():
    """filter, analyze and the pooled report read files the GPU stages wrote.
    Holding two GPUs idle through them is hours of a shared machine."""
    text = SUBMIT.read_text()
    stages = text[text.index("STAGES=(") : text.index("\n", text.index("STAGES=("))]
    assert "generate:gpu" in stages and "evaluate:gpu" in stages
    assert "filter:cpu" in stages and "analyze:cpu" in stages and "pooled:cpu" in stages


def test_the_followup_index_is_pinned_across_the_chain():
    """Every stage re-plans, and an unpinned index comes from the records on
    disk -- so generate and evaluate would take consecutive indices and become
    two different runs, the second reading a directory the first never wrote."""
    submitter = SUBMIT.read_text()
    assert "export FOLLOWUP_INDEX=" in submitter
    assert "FOLLOWUP_INDEX=${FOLLOWUP_INDEX}" in submitter, "and reaches the job environment"
    runner = RUNNER.read_text()
    assert '${FOLLOWUP_INDEX:+--index "$FOLLOWUP_INDEX"}' in runner, "and the runner honours it"


def test_a_followup_is_planned_before_anything_is_queued():
    """So a chain that sits in the queue for a day is already auditable, and so
    the index exists before the jobs that share it. After the stages are chosen,
    not before: which stages run decides whether this is a new follow-up or a
    re-run of one that exists."""
    text = SUBMIT.read_text()
    # Anchored on the invocation, not on any mention of the script: a comment
    # naming it appears earlier, where the sized form points at it for converting
    # a design target into a seed count.
    invocation = 'python3 "$CAMPAIGN_DIR/scripts/plan_followup.py"'
    assert text.index("STAGES=(") < text.index(invocation)
    assert text.index(invocation) < text.index('for entry in "${SELECTED[@]}"')


def test_the_pooled_report_runs_last_and_not_for_smoke():
    """It reads every run's results, so before analyze it would report a number
    that predates the run just submitted. Smoke is not part of the pool.

    Ordering is now a property of the stage list rather than of an append after
    the loop, which is what lets a re-run start at any stage and still finish
    with the total."""
    text = SUBMIT.read_text()
    stages = text[text.index("STAGES=(") : text.index("\n", text.index("STAGES=("))]
    assert stages.rstrip(")").endswith("pooled:cpu"), "last in the chain"
    smoke = text[text.index("  smoke) STAGES=(") :][:120]
    assert "pooled" not in smoke


def metadata_state(pkg):
    """Every metadata file and its contents, for comparing before and after."""
    meta = pkg / "metadata"
    return {p.name: p.read_bytes() for p in sorted(meta.glob("*"))} if meta.is_dir() else {}


def estimate(pkg, *args):
    return subprocess.run(
        ["bash", str(pkg / "scripts" / "estimate_run.sh"), *map(str, args)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "CAMPAIGN_DIR": str(pkg), "HOME": str(pkg)},
    )


def test_a_design_target_converts_to_a_seed_count(tmp_path):
    """The question a campaign actually asks between runs. It was answerable only
    by invoking the planner with seven arguments campaign.env already holds, and
    reading shell-variable output."""
    pkg = campaign_package(tmp_path)
    proc = estimate(pkg, 900)
    assert proc.returncode == 0, proc.stderr
    assert "seeds ->" in proc.stdout and "designs" in proc.stdout
    assert "designs per seed, measured over" in proc.stdout, "it says what the estimate rests on"
    assert "submit_campaign.sh production" in proc.stdout, "and what to do with the answer"


def test_a_seed_count_converts_back_to_designs(tmp_path):
    """Both directions, because the submit path takes seeds and a campaign's
    target is designs."""
    pkg = campaign_package(tmp_path)
    proc = estimate(pkg, "--seeds", 200)
    assert proc.returncode == 0, proc.stderr
    assert "200 seeds" in proc.stdout


def test_an_orderable_target_converts_to_a_seed_count(tmp_path):
    """The target a campaign actually cares about: sequences past the gate, not
    structures produced."""
    pkg = campaign_package(tmp_path, pooled=True)
    proc = estimate(pkg, "--orderable", 500)
    assert proc.returncode == 0, proc.stderr
    assert "orderable" in proc.stdout
    assert "orderable per design (95% CI" in proc.stdout


def test_the_orderable_rate_carries_a_clustered_interval(tmp_path):
    """Beam search expands one nres draw into several candidates, so designs
    sharing a root are not independent -- and binder length, which dominates
    whether a design passes, is drawn per root. Treating designs as independent
    understated the variance 4-7 fold on CBLN1: the naive interval after the
    first run was [0.391, 0.574] and excluded the 0.319 the third run delivered,
    while the clustered [0.292, 0.673] covered both later runs."""
    pkg = campaign_package(tmp_path, pooled=True)
    out = estimate(pkg, "--orderable", 500).stdout
    assert "orderable per design (95% CI" in out
    assert "clusters)" in out
    assert "per run so far:" in out
    assert "wider than treating designs as independent draws" in out


def test_an_orderable_target_is_sized_on_the_low_end(tmp_path):
    """Sizing on the mean is what over-promised the first two follow-ups by ~40%.
    The low end over-delivers instead, which is the failure direction to prefer --
    and the line says what the mean would have given, so the choice is visible."""
    pkg = campaign_package(tmp_path, pooled=True)
    out = estimate(pkg, "--orderable", 500).stdout
    assert "sized on" in out and "the low end" in out
    assert "at the mean it would be" in out
    assert "at least" in out, "the projection is a floor, not a point estimate"


def test_sizing_by_orderable_needs_the_analysis(tmp_path):
    """Orderable counts come from applying the success thresholds, which is
    analysis, and the interval needs them per design. A campaign that has not run
    the analyze stage can still size by designs or by seeds."""
    pkg = campaign_package(tmp_path)
    proc = estimate(pkg, "--orderable", 500)
    assert proc.returncode != 0
    assert "RAW_" in proc.stderr and "analyze stage" in proc.stderr
    assert estimate(pkg, 900).returncode == 0, "and the other forms still work"


def test_a_run_is_sized_by_exactly_one_target(tmp_path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("pf", TEMPLATES / "plan_followup.py")
    pf = importlib.util.module_from_spec(spec)
    sys.modules["pf"] = pf
    spec.loader.exec_module(pf)
    obs = {"designs_per_seed": 5.0, "expansion_per_seed": 8.0, "trim_ratio": 0.97,
           "orderable_per_design": 0.4, "orderable_per_design_lower": 0.4}
    with pytest.raises(SystemExit, match="exactly one"):
        pf.plan(2, 5, 2, obs, want_designs=700, want_orderable=500)
    with pytest.raises(SystemExit, match="exactly one"):
        pf.plan(2, 5, 2, obs)
    # 500 orderable / (0.4 per design x 5.0 designs per seed) = 250 seeds
    assert pf.plan(2, 5, 2, obs, want_orderable=500)["seeds"] == 250


def test_estimating_writes_nothing(tmp_path):
    """It answers a question. Reserving a run number or rewriting the audit trail
    to answer one is the failure this whole flag family had."""
    pkg = campaign_package(tmp_path, followups=[(1, 900)])
    before = metadata_state(pkg)
    assert estimate(pkg, 900).returncode == 0
    assert metadata_state(pkg) == before


def test_the_estimate_is_not_presented_as_a_promise(tmp_path):
    pkg = campaign_package(tmp_path)
    assert "not a promise" in estimate(pkg, 900).stdout


def test_a_dry_run_writes_nothing(tmp_path):
    """DRY_RUN promised "submit nothing" and delivered "queue nothing" -- planning
    still wrote. Previewing a chain against a finished campaign rewrote the
    absolute paths in its follow-up records and pool manifests to wherever the
    preview was pointed, which would have left its provenance naming a directory
    that does not exist on the cluster."""
    pkg = campaign_package(tmp_path, followups=[(1, 900), (2, 1110)])
    before = metadata_state(pkg)
    proc, _ = submit(pkg, "followup", "900", "evaluate..analyze")
    assert proc.returncode == 0, proc.stderr
    assert metadata_state(pkg) == before, "a preview must not touch the audit trail"


def test_a_dry_run_does_not_burn_a_run_number(tmp_path):
    """A run planned but never executed still consumes its number, which is what
    keeps seeds from being reused. That is right for a submission and wrong for a
    preview: nothing was submitted, so nothing should be reserved."""
    pkg = campaign_package(tmp_path, followups=[(1, 900), (2, 1110)])
    proc, _ = submit(pkg, "followup", "700")
    assert proc.returncode == 0, proc.stderr
    assert "run #4" in proc.stdout, "it still reports what it would do"
    assert not (pkg / "metadata" / "run_4.json").exists(), "but reserves nothing"


def test_a_dry_run_says_the_record_was_not_written(tmp_path):
    """Otherwise the line reads as a statement that it was."""
    pkg = campaign_package(tmp_path, followups=[(1, 900)])
    proc, _ = submit(pkg, "followup", "900", "evaluate..analyze")
    assert "not written: this is a dry run" in proc.stdout


def test_the_submitter_can_be_previewed_without_submitting():
    """A chain of five jobs against a shared cluster is worth reading first."""
    assert "DRY_RUN" in SUBMIT.read_text()


def test_the_generic_sbatch_takes_the_stage_rather_than_hardcoding_it():
    """One entry point, because five sbatch files that differ only in their last
    word are five files that drift apart."""
    text = GENERIC_SBATCH.read_text()
    assert 'run_campaign.sh" "$@"' in text
    assert "--job-name" not in text, "the name differs per stage and comes from the submitter"
    assert "--gres" not in text, "so does the GPU request"


def test_a_single_run_can_differ_from_the_campaign_config():
    """Editing pipeline.yaml to change one run changes every run that ever
    re-reads it, including a re-evaluation of an earlier one."""
    runner = RUNNER.read_text()
    assert 'EXTRA_OVERRIDES=("$@")' in runner
    assert 'OVERRIDES+=("${EXTRA_OVERRIDES[@]}")' in runner


def test_caller_overrides_win_over_the_runners_own():
    """Appended last, so `++generation.dataloader.dataset.nres.nsamples=N` from a
    caller beats the sizing the runner derived, rather than being silently
    ignored because Hydra took the first."""
    runner = RUNNER.read_text()
    appended = runner.index('OVERRIDES+=("${EXTRA_OVERRIDES[@]}")')
    assert runner.index('OVERRIDES=("++run_name=') < appended, "after the runner's own"
    assert runner.index("dedup_against_manifest") < appended, "and after the follow-up's"


def test_overrides_reach_every_stage_of_a_chain():
    """A redesign count set for generate but not evaluate would refold a
    different number of sequences than were designed."""
    submitter = SUBMIT.read_text()
    stage_call = submitter[submitter.index('dep="$(submit "${TAG}-${stage}"') :][:220]
    assert "EXTRA_OVERRIDES" in stage_call


def test_the_pooled_report_takes_no_metric_overrides():
    """It reads finished CSVs and applies thresholds; a metric override there
    would describe folding that already happened."""
    submitter = SUBMIT.read_text()
    pooled_call = submitter[submitter.index('submit "${TAG}-pooled"') :][:120]
    assert "EXTRA_OVERRIDES" not in pooled_call


def test_the_stage_is_still_optional_with_overrides_present():
    """`followup 900 -- ++x=1` must not read `--` as the stage name."""
    runner = RUNNER.read_text()
    assert '"${1}" != --*' in runner, "a leading -- is not mistaken for a stage"


# ---------------------------------------------------------------------------
# Re-running a chain from a stage. Driven with DRY_RUN rather than read, because
# the failure this guards against -- a follow-up re-run silently becoming a new
# follow-up -- is a property of what gets submitted, not of what the script says.
# ---------------------------------------------------------------------------


def campaign_package(tmp_path, *, followups=(), pooled=False):
    """The smallest package submit_campaign.sh will act on."""
    pkg = tmp_path / "camp"
    (pkg / "slurm").mkdir(parents=True)
    (pkg / "scripts").mkdir(parents=True)
    (pkg / "metadata").mkdir(parents=True)
    (pkg / "slurm" / "campaign.sbatch").write_text("#!/usr/bin/env bash\n")
    for template in ("plan_followup.py", "submit_campaign.sh", "estimate_run.sh"):
        # Copied in rather than run from the templates directory: the submitter
        # locates campaign.env relative to itself, which is what makes a package
        # self-contained.
        (pkg / "scripts" / template).write_text((TEMPLATES / template).read_text())
    (pkg / "campaign.env").write_text(
        "TASK_NAME=T\nCONFIG_NAME=pipeline\nRUN_PREFIX=pfx\n"
        f'CAMPAIGN_DIR="${{CAMPAIGN_DIR:-{pkg}}}"\n'
        "SHARDS=2\nPRODUCTION_SEEDS=64\nPRODUCTION_RNG_SEED=5\n"
    )
    # What a follow-up is sized from: production's actual yield, and the runs it
    # must not duplicate.
    (pkg / "metadata" / "run_outputs_production.json").write_text(
        json.dumps({"raw_generation_rows": 512, "live_after_global_dedup": 340})
    )
    (pkg / "metadata" / "shard_trim_production.json").write_text(
        json.dumps({"shards": {"0": {"generated_rows": 256, "retained": 250}, "1": {"generated_rows": 256, "retained": 250}}})
    )
    inf = pkg / "inference" / "pipeline_T_pfx_production"
    inf.mkdir(parents=True)
    (inf / "top_samples_pipeline.csv").write_text("x\n")
    if pooled:
        # Per-design verdicts, because the interval is clustered by beam root and
        # so needs each design rather than a per-run total. Names carry the
        # `_n_{nres}_` and `beam_orig{k}` tags the clustering reads, and the
        # outcome is made to vary by root so a clustered interval is wider than
        # a naive one -- which is the property under test.
        run_dir = pkg / "evaluation_results" / "pipeline_T_pfx_production"
        run_dir.mkdir(parents=True, exist_ok=True)
        lines = ["pdb_path,self_pass_all,mpnn_pass_all"]
        for root in range(12):
            for i in range(8):
                name = f"job_0_n_{180 + root}_id_{i}_beam_orig{root % 3}_bm1"
                verdict = "1" if root % 2 else "0"
                lines.append(f"/x/{name}/{name}.pdb,\"[{verdict}]\",\"[{verdict}, 0]\"")
        (run_dir / "RAW_protein_binder_results_pipeline_combined.csv").write_text("\n".join(lines) + "\n")
    for index, wanted in followups:
        # Full records, because a run that already exists is now replayed rather
        # than re-derived: its size is history, and re-deriving would move it as
        # the campaign's calibration set grows.
        (pkg / "metadata" / f"followup_{index}.json").write_text(
            json.dumps(
                {
                    "index": index,
                    "number": index + 1,
                    "want_designs": wanted,
                    "run_name": f"pfx_followup{index}",
                    "seeds": 170,
                    "raw": 1360,
                    "keep": 664,
                    "expect": 1328,
                    "rng_seed": 5 + index * 1000,
                }
            )
        )
        d = pkg / "inference" / f"pipeline_T_pfx_followup{index}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "top_samples_pipeline.csv").write_text("x\n")
    return pkg


def submit(pkg, *args, **env):
    proc = subprocess.run(
        ["bash", str(pkg / "scripts" / "submit_campaign.sh"), *map(str, args)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "DRY_RUN": "1",
            "CAMPAIGN_DIR": str(pkg),
            "HOME": str(pkg),
            **env,
        },
    )
    # DRY_RUN prints the planned sbatch lines to stderr; each ends with the
    # arguments run_campaign.sh would receive.
    planned = [line for line in proc.stderr.splitlines() if line.startswith("sbatch ")]
    stages = [line.rsplit("campaign.sbatch ", 1)[1].split() for line in planned]
    return proc, stages


def test_the_default_is_still_the_whole_chain(tmp_path):
    proc, stages = submit(campaign_package(tmp_path), "production")
    assert proc.returncode == 0, proc.stderr
    assert [s[-1] for s in stages] == ["generate", "filter", "evaluate", "analyze", "pooled"]


def test_a_stage_argument_reruns_from_there_to_the_end(tmp_path):
    """Not that stage alone. A stage whose inputs were just rewritten and whose
    outputs were not is a results CSV that disagrees with the per-job CSVs it was
    built from, with nothing saying so."""
    proc, stages = submit(campaign_package(tmp_path), "production", "evaluate")
    assert proc.returncode == 0, proc.stderr
    assert [s[-1] for s in stages] == ["evaluate", "analyze", "pooled"]


def test_a_range_stops_at_the_end_stage(tmp_path):
    """Several runs re-evaluated together share one pooled report. Chaining a
    pooled to each queues a campaign total per run, and every one but the last
    reads a half-re-evaluated campaign and writes it under the same name."""
    proc, stages = submit(campaign_package(tmp_path), "production", "evaluate..analyze")
    assert proc.returncode == 0, proc.stderr
    assert [s[-1] for s in stages] == ["evaluate", "analyze"]


def test_an_open_ended_range_degrades_to_the_plain_forms(tmp_path):
    pkg = campaign_package(tmp_path)
    _, to_end = submit(pkg, "production", "evaluate..")
    _, from_start = submit(pkg, "production", "..analyze")
    assert [s[-1] for s in to_end] == ["evaluate", "analyze", "pooled"]
    assert [s[-1] for s in from_start] == ["generate", "filter", "evaluate", "analyze"]


def test_omitting_the_pooled_report_says_so(tmp_path):
    """The campaign total is the headline number and this chain does not produce
    it. Stopping early is only right if it is followed up."""
    proc, _ = submit(campaign_package(tmp_path), "production", "evaluate..analyze")
    assert "no pooled report is queued" in proc.stdout
    assert "submit_campaign.sh production pooled" in proc.stdout


def test_the_suggested_pooled_command_is_one_that_runs(tmp_path):
    """It said `followup pooled` for a follow-up chain, which is not a runnable
    spelling: followup demands a design count, and the pooled report has no run
    of its own. A note naming a command that errors is worse than no note."""
    pkg = campaign_package(tmp_path, followups=[(1, 900)])
    proc, _ = submit(pkg, "followup", "900", "evaluate..analyze")
    assert "no pooled report is queued" in proc.stdout
    assert "followup pooled" not in proc.stdout
    suggested = "production pooled"
    assert suggested in proc.stdout
    # And it does run.
    rerun, stages = submit(pkg, *suggested.split())
    assert rerun.returncode == 0, rerun.stderr
    assert stages == [["pooled"]]


def test_a_whole_chain_does_not_warn_about_the_pooled_report(tmp_path):
    proc, _ = submit(campaign_package(tmp_path), "production")
    assert "no pooled report" not in proc.stdout


def test_a_range_that_ends_before_it_starts_says_which_way_round(tmp_path):
    """'analyze..evaluate' is a real mistake with an empty result, and 'no stages
    selected' would not tell you which half was wrong."""
    proc, _ = submit(campaign_package(tmp_path), "production", "analyze..evaluate")
    assert proc.returncode == 2
    assert "ends before it starts" in proc.stderr
    assert "evaluate runs before analyze" in proc.stderr


def test_an_unknown_end_stage_is_distinguished_from_a_backwards_one(tmp_path):
    proc, _ = submit(campaign_package(tmp_path), "production", "evaluate..refold")
    assert proc.returncode == 2
    assert "unknown end stage 'refold'" in proc.stderr


def test_the_pooled_report_is_reachable_on_its_own(tmp_path):
    """Re-deriving the campaign total costs a comparison, not a re-evaluation."""
    proc, stages = submit(campaign_package(tmp_path), "production", "pooled")
    assert proc.returncode == 0, proc.stderr
    assert stages == [["pooled"]], "no run kind and no stage word -- it is campaign-wide"


def test_smoke_has_no_pooled_report(tmp_path):
    """Those designs are a throwaway check, not part of the deliverable."""
    pkg = campaign_package(tmp_path)
    proc, stages = submit(pkg, "smoke")
    assert proc.returncode == 0, proc.stderr
    assert "pooled" not in [s[-1] for s in stages]
    proc, _ = submit(pkg, "smoke", "pooled")
    assert proc.returncode == 2
    assert "unknown stage 'pooled'" in proc.stderr


def test_an_unknown_stage_names_the_ones_that_exist(tmp_path):
    proc, _ = submit(campaign_package(tmp_path), "production", "refold")
    assert proc.returncode == 2
    assert "generate filter evaluate analyze pooled" in proc.stderr


def test_a_followup_rerun_reuses_its_index_rather_than_becoming_a_new_run(tmp_path):
    """The bug the manual sbatch workaround existed to avoid. An unpinned re-plan
    allocates the next index, so `followup 900 evaluate` would evaluate an
    inference directory nothing ever wrote -- and burn a seed on a run that never
    happens."""
    pkg = campaign_package(tmp_path, followups=[(1, 900), (2, 1110)])
    proc, stages = submit(pkg, "followup", "900", "evaluate")
    assert proc.returncode == 0, proc.stderr
    assert "run #2 (followup1)" in proc.stdout, "production is run 1, so followup1 is run 2 -- keeping its name"
    assert [s[-1] for s in stages] == ["evaluate", "analyze", "pooled"]
    assert not (pkg / "metadata" / "followup_3.json").exists(), "no new follow-up was planned"
    assert "FOLLOWUP_INDEX=1" in proc.stderr, "and the index reaches the job environment"


def test_a_followup_with_no_stage_still_plans_a_new_one(tmp_path):
    """Re-running from a stage is the exception; asking for more designs is the
    normal case and must keep allocating."""
    pkg = campaign_package(tmp_path, followups=[(1, 900), (2, 1110)])
    proc, stages = submit(pkg, "followup", "700")
    assert proc.returncode == 0, proc.stderr
    # Run 4 of the campaign: production is 1, followup1 and followup2 are 2 and 3.
    # It gets the current spelling, while the runs already on disk keep theirs.
    # No record to check: the harness previews, and a preview reserves nothing --
    # see test_a_dry_run_does_not_burn_a_run_number.
    assert "run #4 (production4)" in proc.stdout, "a new run gets the current spelling"
    assert "1005" not in proc.stdout, "planned fresh, not resumed from followup1"
    assert [s[-1] for s in stages] == ["generate", "filter", "evaluate", "analyze", "pooled"]


def test_resuming_a_count_no_followup_asked_for_is_refused(tmp_path):
    """Rather than resolved by picking the newest: guessing which follow-up was
    meant re-evaluates the wrong designs."""
    pkg = campaign_package(tmp_path, followups=[(1, 900)])
    proc, _ = submit(pkg, "followup", "1234", "evaluate")
    assert proc.returncode != 0
    assert "no record" in proc.stderr and "#1 wanted 900" in proc.stderr


def test_two_followups_wanting_the_same_count_are_refused_not_guessed(tmp_path):
    """The count is the caller's handle on a follow-up, and it stops being one
    the moment two runs share it. Picking the newest would re-evaluate a
    different set of designs than the caller named."""
    pkg = campaign_package(tmp_path, followups=[(1, 900), (2, 900)])
    proc, _ = submit(pkg, "followup", "900", "evaluate")
    assert proc.returncode != 0
    assert "[1, 2] all asked for that many" in proc.stderr
    assert "FOLLOWUP_INDEX=<n>" in proc.stderr, "and the message names the way out"


def test_an_explicit_index_settles_an_ambiguous_count(tmp_path):
    """The way out the message names has to exist. It did not: the error said
    --index, which submit_campaign.sh had no way to forward."""
    pkg = campaign_package(tmp_path, followups=[(1, 900), (2, 900)])
    proc, stages = submit(pkg, "followup", "900", "evaluate", FOLLOWUP_INDEX="2")
    assert proc.returncode == 0, proc.stderr
    assert "run #3 (followup2)" in proc.stdout
    assert [s[-1] for s in stages] == ["evaluate", "analyze", "pooled"]
    assert "FOLLOWUP_INDEX=2" in proc.stderr, "and it reaches the job environment"


def test_a_generate_rerun_is_a_new_followup_not_a_resumed_one(tmp_path):
    """`followup 900 generate` regenerates, which is a new run by definition --
    reusing the index would write into a directory another run already owns."""
    pkg = campaign_package(tmp_path, followups=[(1, 900)])
    proc, _ = submit(pkg, "followup", "900", "generate")
    assert proc.returncode == 0, proc.stderr
    # A new run, so the current spelling -- production is run 1, followup1 run 2,
    # and this one run 3. The runs already on disk keep the names they were
    # written under; only new ones are named the new way.
    assert "run #3 (production3)" in proc.stdout


def test_overrides_still_reach_a_partial_chain(tmp_path):
    proc, stages = submit(campaign_package(tmp_path), "production", "evaluate", "--", "++metric.x=1")
    assert proc.returncode == 0, proc.stderr
    assert stages[0] == ["production", "evaluate", "++metric.x=1"]
    assert stages[-1] == ["pooled"], "except the pooled report, which folds nothing"


def test_what_a_run_actually_used_stays_recoverable():
    """The resolved config is written from the same override list, so a run that
    differed from pipeline.yaml still says how."""
    runner = RUNNER.read_text()
    resolved = runner[runner.index("validate_resolved_config.py") :][:260]
    assert '"${OVERRIDES[@]}"' in resolved


# ---------------------------------------------------------------------------
# Tool provenance and config-aware tool gating
# ---------------------------------------------------------------------------

PREFLIGHT_SH = Path(__file__).resolve().parents[1] / ".claude/skills/_shared/scripts/preflight.sh"


def preflight_report(tmp_path, tools, metric=None, **overrides):
    """A minimal report that clears every gate except the one under test."""
    report = tmp_path / "preflight.json"
    body = {
        "gpu": {"available": True, "vram_gb": 80},
        "checkpoints": {"complexa.ckpt": {"exists": True}, "complexa_ae.ckpt": {"exists": True}},
        "community_models": {"AF2_DIR": {"exists": True}},
        "tools": tools,
        "disk": {"cwd_free_gb": 9999},
        "env": {},
    }
    body.update(overrides)
    report.write_text(json.dumps(body))
    cfg = tmp_path / "resolved.yaml"
    cfg.write_text(yaml.safe_dump({"metric": metric or {}}))
    return report, cfg


PRESENT = {"foldseek": {"exists": True}, "mmseqs": {"exists": True}}


def test_an_absent_sc_binary_is_no_longer_anyones_business(tmp_path):
    """Shape complementarity runs in process now. A stale sc entry in a preflight
    report -- every CBLN1 run recorded exists:false -- must not fail anything."""
    report, cfg = preflight_report(
        tmp_path,
        {**PRESENT, "sc": {"path": "/nope/sc", "exists": False}},
        metric={"compute_refolded_structure_metrics": True, "refolded": {"bioinformatics": True}},
    )
    r = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 100)
    assert "missing sc" not in r.stdout, r.stdout


@pytest.mark.parametrize(
    "metric",
    [
        {"compute_pre_refolding_metrics": True, "pre_refolding": {"bioinformatics": True}},
        {"compute_refolded_structure_metrics": True, "refolded": {"bioinformatics": True}},
        # No sub-block at all: evaluate.py defaults the flag to on, so this
        # config will call sc and the gate has to say so.
        {"compute_refolded_structure_metrics": True},
    ],
)
def test_bioinformatics_requires_the_engine_to_import(tmp_path, metric):
    """The requirement moved from a file on a path to an importable extension."""
    report, cfg = preflight_report(tmp_path, {**PRESENT, "sc": {"path": "/nope/sc", "exists": False}}, metric=metric)
    r = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 100)
    assert "missing sc" not in r.stdout, "no binary is involved any more"
    # protein_interface is a declared dependency, so it imports here and the gate passes.
    assert "protein_interface will not import" not in r.stdout, r.stdout


def test_an_explicitly_disabled_sub_flag_does_not_require_the_engine(tmp_path):
    report, cfg = preflight_report(
        tmp_path,
        {**PRESENT, "sc": {"exists": False}},
        metric={"compute_refolded_structure_metrics": True, "refolded": {"bioinformatics": False}},
    )
    out = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 1).stdout
    assert "protein_interface will not import" not in out


def test_a_generation_config_that_rewards_on_sc_is_still_detected(tmp_path):
    """The reward path reaches shape complementarity without going through the
    metric flags. There is no binary to require any more, but the detection is
    what decides whether the engine import is checked at all."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("chk", TEMPLATES / "check_preflight.py")
    chk = importlib.util.module_from_spec(spec)
    sys.modules["chk"] = chk
    spec.loader.exec_module(chk)

    reward_cfg = {
        "metric": {},
        "reward_models": {
            "bioinformatics": {
                "_target_": "proteinfoundation.rewards.bioinformatics_reward.BioinformaticsRewardModel"
            }
        },
    }
    assert chk.needs_protein_interface(reward_cfg, reward_cfg["metric"])
    assert not chk.needs_protein_interface({"metric": {}}, {})


def test_a_protein_binder_run_needs_the_engine_whatever_the_metric_flags_say():
    """The gate used to ask only whether the bioinformatics COLUMNS were on. But
    the interface definition runs on every protein-target design regardless --
    it is what aa_interface_counts counts and what mpnn_fixed holds fixed -- so a
    run with those columns off still dies at the first design without it."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("chk2", TEMPLATES / "check_preflight.py")
    chk = importlib.util.module_from_spec(spec)
    sys.modules["chk2"] = chk
    spec.loader.exec_module(chk)

    bare = {"compute_binder_metrics": True, "compute_pre_refolding_metrics": False}
    assert chk.needs_protein_interface({"result_type": "protein_binder", "metric": bare}, bare)
    # A ligand target takes the atomistic path and never reaches the extension.
    assert not chk.needs_protein_interface({"result_type": "ligand_binder", "metric": bare}, bare)
    # And a campaign that runs no binder track at all is not asked for it.
    monomer = {"compute_binder_metrics": False}
    assert not chk.needs_protein_interface({"result_type": "monomer", "metric": monomer}, monomer)


def preflight_module(name="chk_hs"):
    """The template imported as a module, so the judgement calls can be read directly."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, TEMPLATES / "check_preflight.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def binder_cfg(tmp_path, hotspots, target_input="A58-193", task="MINE", **entry):
    """A resolved config shaped like the one validate_resolved_config.py writes."""
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM\n")
    e = {"target_path": str(pdb), "target_input": target_input, "hotspot_residues": hotspots}
    e.update(entry)
    return {
        "result_type": "protein_binder",
        "metric": {},
        "generation": {"task_name": task, "target_dict_cfg": {"33_TrkA": {"hotspot_residues": ["X294"]}, task: e}},
    }


def test_hotspots_are_checked_against_the_entry_task_name_selects(tmp_path):
    """An unpinned task_name inheriting 33_TrkA is a documented way to get a clean
    run against the wrong target. Scanning the dict instead of reading task_name
    would check somebody else's hotspots and pass."""
    chk = preflight_module()
    cfg = binder_cfg(tmp_path, ["A70", "A71"])
    assert chk.target_hotspots(cfg)[2] == ["A70", "A71"]
    cfg["generation"]["task_name"] = "33_TrkA"
    assert chk.target_hotspots(cfg)[2] == ["X294"]


@pytest.mark.parametrize(
    "cfg_mod, why",
    [
        (lambda c: c.pop("generation"), "a monomer run declares no target"),
        (lambda c: c.update(result_type="ligand_binder"), "the ligand path never reads hotspots"),
        (lambda c: c["generation"]["target_dict_cfg"]["MINE"].update(ligand="FAD"), "ligand entry"),
        (lambda c: c["generation"]["target_dict_cfg"]["MINE"].update(hotspot_residues=[]), "[] means no focus"),
        (lambda c: c["generation"]["target_dict_cfg"]["MINE"].update(hotspot_residues=[None]), "ligand entries ship [null]"),
    ],
)
def test_campaigns_that_do_not_use_hotspots_are_not_gated_on_them(tmp_path, cfg_mod, why):
    """Every skip here is a real campaign shape. A gate that fires on these would
    be worse than no gate: it fails runs that are correct."""
    chk = preflight_module()
    cfg = binder_cfg(tmp_path, ["A70"])
    cfg_mod(cfg)
    assert chk.target_hotspots(cfg) is None, why
    assert chk.hotspot_failures(cfg) == [], why


def test_an_unresolvable_hotspot_fails_and_names_it(tmp_path):
    """The whole point. Generation would match no residue, warn about nothing, and
    design against no epitope for the length of the campaign."""
    chk = preflight_module()
    cfg = binder_cfg(tmp_path, ["A70", "B999"])
    fails = chk.hotspot_failures(cfg, read=lambda path, spec: {"A70", "A71"})
    assert len(fails) == 1 and "B999" in fails[0] and "A70" not in fails[0].split(":")[-1], fails


def test_hotspots_that_all_resolve_pass(tmp_path):
    chk = preflight_module()
    cfg = binder_cfg(tmp_path, ["A70", "A71"])
    assert chk.hotspot_failures(cfg, read=lambda path, spec: {"A70", "A71", "A72"}) == []


def test_the_contig_is_handed_to_the_reader_not_re_parsed_here(tmp_path):
    """Generation masks on target_input before matching, so a hotspot outside it is
    dropped. The reader gets the spec; nothing in this template re-implements it."""
    chk = preflight_module()
    seen = {}

    def reader(path, spec):
        seen["spec"] = spec
        return {"A70"}

    chk.hotspot_failures(binder_cfg(tmp_path, ["A70"], target_input="A58-193"), read=reader)
    assert seen["spec"] == "A58-193"


def test_declared_hotspots_with_no_target_input_fail(tmp_path):
    chk = preflight_module()
    cfg = binder_cfg(tmp_path, ["A70"], target_input=None)
    assert "target_input is unset" in chk.hotspot_failures(cfg)[0]


def test_a_target_pdb_that_is_not_there_fails(tmp_path):
    chk = preflight_module()
    cfg = binder_cfg(tmp_path, ["A70"], target_path=str(tmp_path / "gone.pdb"))
    assert "target PDB not found" in chk.hotspot_failures(cfg)[0]


BUNDLED_TARGET = Path(__file__).resolve().parents[1] / "assets/target_data/bindcraft_targets/PD-L1.pdb"


def has_atomworks():
    import importlib.util

    return importlib.util.find_spec("atomworks") is not None


needs_env = pytest.mark.skipif(
    not has_atomworks() or not BUNDLED_TARGET.is_file(),
    reason="the real reader needs atomworks and the bundled targets",
)


@needs_env
def test_the_reader_selects_what_generation_selects():
    """Pinned against a bundled target whose config ships its own hotspots:
    02_PDL1 is A1-115 with A37/A39/A49/A98, so the reader agreeing with it is
    the same claim as generation resolving them."""
    chk = preflight_module("chk_real")
    ids = chk.ca_ids_from_pdb(str(BUNDLED_TARGET), "A1-115")
    assert {"A37", "A39", "A49", "A98"} <= ids
    # The mask comes first, so the contig -- not the chain -- bounds what a
    # hotspot can address. check_target_pdb.py matches over the whole chain and
    # would call A115 resolved here.
    narrowed = chk.ca_ids_from_pdb(str(BUNDLED_TARGET), "A5-100")
    assert narrowed == {f"A{i}" for i in range(5, 101)}
    assert "A115" in ids and "A115" not in narrowed


@needs_env
def test_a_contig_that_does_not_fit_the_file_is_reported_not_swallowed(tmp_path):
    """get_mask raises on the first absent residue rather than returning a short
    mask, and generation uses the same selector. Reporting it here turns an
    exception that lands after the checkpoint loads into one that lands now."""
    chk = preflight_module("chk_real2")
    cfg = binder_cfg(tmp_path, ["A37"], target_input="A1-116", target_path=str(BUNDLED_TARGET))
    fails = chk.hotspot_failures(cfg)
    assert len(fails) == 1 and "cannot apply target_input A1-116" in fails[0], fails
    assert "A/*/116" in fails[0], "the failure should name the residue that is missing"


@needs_env
def test_a_hotspot_outside_a_valid_contig_is_caught(tmp_path):
    """The silent case that survives every other check: the contig fits, the file
    is fine, and the hotspot simply is not in the selection."""
    chk = preflight_module("chk_real3")
    cfg = binder_cfg(tmp_path, ["A37", "A115"], target_input="A5-100", target_path=str(BUNDLED_TARGET))
    fails = chk.hotspot_failures(cfg)
    assert len(fails) == 1 and "A115" in fails[0] and "no epitope" in fails[0], fails


def test_an_unreadable_target_fails_rather_than_passing_quietly(tmp_path):
    """A parse error must not read as "no misses"."""
    chk = preflight_module()

    def boom(path, spec):
        raise ValueError("bad contig")

    fails = chk.hotspot_failures(binder_cfg(tmp_path, ["A70"]), read=boom)
    assert "cannot apply target_input" in fails[0] and "bad contig" in fails[0]


def test_the_hotspot_gate_is_wired_into_the_script(tmp_path):
    """The functions above are only worth anything if main() calls them."""
    report, cfg = preflight_report(tmp_path, PRESENT)
    body = binder_cfg(tmp_path, ["A70"], target_path=str(tmp_path / "gone.pdb"))
    cfg.write_text(yaml.safe_dump(body))
    r = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 1)
    assert r.returncode == 1 and "target PDB not found" in r.stdout, r.stdout


def msa_module(name="msa_mod"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, TEMPLATES / "prepare_target_msa.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_msa_template_imports_nothing_heavy():
    """The point of the template. A campaign package is assembled where ColabFold,
    esm and proteinfoundation are usually all absent, so importing it and reading
    --help must work with the standard library alone."""
    import ast

    tree = ast.parse((TEMPLATES / "prepare_target_msa.py").read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
             for n in top for a in (n.names if isinstance(n, ast.Import) else [None]) or [None]}
    assert names <= {"__future__", "argparse", "hashlib", "json", "sys", "datetime", "pathlib"}, names
    r = run("prepare_target_msa.py", "--help")
    assert r.returncode == 0 and "--host-url" in r.stdout


def test_a_deferred_dependency_is_named_not_traced(tmp_path):
    """A missing ColabFold must read as an instruction, not an ImportError."""
    chk = msa_module()
    exc = chk.missing("colabfold", "fetching an MSA", "pip install colabfold")
    assert "colabfold" in str(exc) and "pip install colabfold" in str(exc)
    assert "built on a machine that has neither" in str(exc)


def test_the_public_server_is_queried_once_not_once_per_stage(tmp_path, monkeypatch):
    """PREPARE_STEPS runs before every stage, so a campaign that fetched on each
    would hit a shared free service four times per run."""
    chk = msa_module("msa_skip")
    out = tmp_path / "t_A.a3m"
    out.write_text(">q\nAAA\n>h\nAAB\n")
    calls = []
    monkeypatch.setattr(chk, "target_sequence", lambda pdb, chain: "AAA")
    monkeypatch.setattr(chk, "validate", lambda p, s, m: type("M", (), {"depth": 2})())
    monkeypatch.setattr(chk, "fetch_a3m", lambda *a, **k: calls.append(a) or ">q\nAAA\n")
    args = argparse.Namespace(force=False, max_sequences=16384, host_url="h", user_agent="u", no_env=False)
    chk.prepare_one("t.pdb", "A", out, args)
    assert calls == [], "a valid alignment already on disk must not be refetched"


def test_an_alignment_that_no_longer_matches_the_target_is_refetched(tmp_path, monkeypatch):
    """The case that matters: someone re-cropped the target PDB. The stale a3m is
    for the old sequence, and evaluation would reject it hours later."""
    chk = msa_module("msa_stale")
    out = tmp_path / "t_A.a3m"
    out.write_text(">q\nAAA\n")
    calls = []
    monkeypatch.setattr(chk, "target_sequence", lambda pdb, chain: "CCC")
    monkeypatch.setattr(chk, "write_provenance", lambda *a, **k: None)

    def validate(path, seq, maxn):
        if path.read_text().split("\n")[1] != seq:
            raise ValueError("query does not match target chain 0")
        return type("M", (), {"depth": 2})()

    monkeypatch.setattr(chk, "validate", validate)
    monkeypatch.setattr(chk, "fetch_a3m", lambda *a, **k: calls.append(1) or ">q\nCCC\n>h\nCCG\n")
    chk.prepare_one("t.pdb", "A", out, argparse.Namespace(
        force=False, max_sequences=16384, host_url="h", user_agent="u", no_env=False))
    assert len(calls) == 1 and out.read_text().startswith(">q\nCCC")


def test_what_was_written_is_what_gets_validated(tmp_path, monkeypatch):
    """A short write or a server answering with something else must fail here, not
    at evaluate time."""
    chk = msa_module("msa_written")
    out = tmp_path / "t_A.a3m"
    seen = {}
    monkeypatch.setattr(chk, "target_sequence", lambda pdb, chain: "AAA")
    monkeypatch.setattr(chk, "fetch_a3m", lambda *a, **k: ">q\nAAA\n>h\nAAB\n")
    monkeypatch.setattr(chk, "write_provenance", lambda *a, **k: None)

    def validate(path, seq, maxn):
        seen["from_disk"] = Path(path).read_text()
        return type("M", (), {"depth": 2})()

    monkeypatch.setattr(chk, "validate", validate)
    chk.prepare_one("t.pdb", "A", out, argparse.Namespace(
        force=False, max_sequences=16384, host_url="h", user_agent="u", no_env=False))
    assert seen["from_disk"] == ">q\nAAA\n>h\nAAB\n"


def test_multi_chain_prints_the_plural_config_key(tmp_path, monkeypatch, capsys):
    """target_msa_paths takes one entry per target chain or it raises with the
    counts, so the single-chain shorthand is wrong for a two-chain target."""
    chk = msa_module("msa_multi")
    monkeypatch.setattr(chk, "prepare_one", lambda pdb, c, out, args: out)
    monkeypatch.setattr(sys, "argv", ["p", "--pdb", "x/tgt.pdb", "--chain", "A", "--chain", "B",
                                      "--out-dir", str(tmp_path)])
    chk.main()
    printed = capsys.readouterr().out
    assert "target_msa_paths: [" in printed and "tgt_A.a3m" in printed and "tgt_B.a3m" in printed
    assert "target_msa:" not in printed


def test_campaign_env_passes_the_target_to_the_step():
    """PREPARE_STEPS entries are word-split, and nothing in campaign.env is
    exported -- a step that expected $TARGET_PDB from the environment would get
    nothing. The example has to show the arguments."""
    env = (TEMPLATES / "campaign.env.example").read_text()
    line = [ln for ln in env.splitlines()
            if "prepare_target_msa.py" in ln and not ln.lstrip().startswith("#")]
    assert len(line) == 1, line
    assert "--pdb $TARGET_PDB" in line[0] and "--chain $TARGET_CHAIN" in line[0]
    assert line[0].strip().startswith('"'), "must be one quoted element, or the array splits it"


def msa_cfg(path=None, backends=("esmfold2",), paths=None, result_type="protein_binder"):
    consensus = {}
    if path is not None:
        consensus["target_msa"] = str(path)
    if paths is not None:
        consensus["target_msa_paths"] = [str(p) if p else None for p in paths]
    return {"result_type": result_type,
            "metric": {"consensus_backends": list(backends), "consensus_cfg": consensus}}


def a3m(tmp_path, name="t.a3m", records=2):
    p = tmp_path / name
    p.write_text("".join(f">s{i}\nAAAA\n" for i in range(records)))
    return p


def test_a_named_target_msa_that_is_absent_fails(tmp_path):
    """_load_msa raises FileNotFoundError deep in evaluate, on a run whose generation
    already finished. The path is knowable at submit time."""
    chk = preflight_module("chk_msa")
    cfg = msa_cfg(tmp_path / "nope.a3m")
    fails = chk.target_msa_failures(cfg, cfg["metric"])
    assert len(fails) == 1 and "not there" in fails[0] and "TARGET_MSA" in fails[0], fails


def test_a_truncated_target_msa_fails_on_depth(tmp_path):
    """A prepare step killed mid-write leaves a file that exists. Folding needs two."""
    chk = preflight_module("chk_msa_depth")
    cfg = msa_cfg(a3m(tmp_path, records=1))
    fails = chk.target_msa_failures(cfg, cfg["metric"])
    assert len(fails) == 1 and "1 record(s)" in fails[0], fails
    ok = msa_cfg(a3m(tmp_path, "good.a3m", records=2))
    assert chk.target_msa_failures(ok, ok["metric"]) == []


@pytest.mark.parametrize(
    "cfg, why",
    [
        (msa_cfg(Path("/nope/x.a3m"), backends=()), "no backend reads it"),
        (msa_cfg(Path("/nope/x.a3m"), result_type="ligand_binder"), "ligand skips consensus folding"),
        (msa_cfg(None), "a campaign may fold with no alignment at all"),
        (msa_cfg(None, paths=[None]), "null per chain is legal"),
    ],
)
def test_configs_that_do_not_use_an_msa_are_not_gated(cfg, why):
    assert cfg["metric"]["consensus_cfg"] is not None
    chk = preflight_module("chk_msa_skip")
    assert chk.target_msa_failures(cfg, cfg["metric"]) == [], why


def test_every_entry_of_target_msa_paths_is_checked(tmp_path):
    """One entry per chain, null where a chain has none -- so a list must be walked,
    not just its first element."""
    chk = preflight_module("chk_msa_multi")
    cfg = msa_cfg(None, paths=[a3m(tmp_path, "a.a3m"), None, tmp_path / "gone.a3m"])
    fails = chk.target_msa_failures(cfg, cfg["metric"])
    assert len(fails) == 1 and "gone.a3m" in fails[0], fails


def test_the_msa_gate_is_wired_into_the_script(tmp_path):
    report, cfg = preflight_report(tmp_path, PRESENT)
    body = msa_cfg(tmp_path / "missing.a3m")
    body["metric"].update(compute_binder_metrics=False)
    cfg.write_text(yaml.safe_dump(body))
    r = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 1)
    assert r.returncode == 1 and "target MSA that is not there" in r.stdout, r.stdout


def test_one_variable_names_the_msa_for_both_sides():
    """The whole point of TARGET_MSA: the step that writes the alignment and the config
    that reads it must not be two conventions that agree by hand."""
    env = (TEMPLATES / "campaign.env.example").read_text()
    assert re.search(r"^TARGET_MSA=", env, re.M), "campaign.env must define it"
    step = [ln for ln in env.splitlines()
            if "prepare_target_msa.py" in ln and not ln.lstrip().startswith("#")]
    assert len(step) == 1 and "--out $TARGET_MSA" in step[0], step
    runner = (TEMPLATES / "run_campaign.sh").read_text()
    assert 'if [[ -n "${TARGET_MSA:-}" ]]; then export TARGET_MSA; fi' in runner, (
        "must be exported, and only when set -- an empty export makes ${oc.env:TARGET_MSA} "
        "resolve to '' instead of raising")


def test_out_must_line_up_with_chain(tmp_path):
    """Zipping a short --out list against --chain would write one chain's alignment to
    another's path, and it would validate -- both are real alignments of real chains."""
    r = run("prepare_target_msa.py", "--pdb", "x.pdb", "--chain", "A", "--chain", "B",
            "--out", str(tmp_path / "a.a3m"))
    assert r.returncode != 0 and "one --out per --chain" in r.stderr, r.stderr


def test_a_missing_tool_failure_names_what_needs_it(tmp_path):
    report, cfg = preflight_report(tmp_path, {"foldseek": {"path": "/nope/fs", "exists": False}})
    out = run("check_preflight.py", report, "--resolved-config", cfg, "--expected-designs", 1).stdout
    assert "missing foldseek, needed for diversity clustering" in out, out
    assert "missing mmseqs, needed for sequence clustering" in out, "an absent entry is a missing tool"


def sourced_file_stamp(tmp_path, *paths):
    """file_stamp lifted out of preflight.sh and run on its own.

    The script itself needs bash 4 for `declare -A`, which macOS does not ship,
    so running it whole is not portable. The stamping is, and it is the part
    that has to be right.
    """
    src = PREFLIGHT_SH.read_text()
    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -euo pipefail\n"
        + re.search(r"^json_str\(\) \{.*?^\}", src, re.S | re.M).group(0)
        + "\n"
        + re.search(r"^file_stamp\(\) \{.*?^\}", src, re.S | re.M).group(0)
        + "\n"
        + "".join(f'echo "{{$(file_stamp "{p}")}}"\n' for p in paths)
    )
    out = subprocess.run(["bash", str(harness)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return [json.loads(line) for line in out.stdout.splitlines()]


def test_a_tool_is_stamped_with_its_size_and_hash(tmp_path):
    """Checkpoints were stamped and tools were not, so a report could say which
    weights ran but not which build of sc-rs wrote a column."""
    import hashlib

    binary = tmp_path / "sc"
    binary.write_bytes(b"\x7fELF fake binary")
    (stamp,) = sourced_file_stamp(tmp_path, binary)
    assert stamp["size"] == binary.stat().st_size
    assert stamp["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()[:16]


def test_stamping_degrades_to_null_rather_than_failing(tmp_path):
    """Every probe in preflight.sh degrades rather than aborting the report."""
    missing, empty, directory = sourced_file_stamp(tmp_path, tmp_path / "nope", "", tmp_path)
    for stamp in (missing, empty, directory):
        assert stamp == {"size": None, "sha256": None}, stamp


def test_the_script_stamps_both_checkpoints_and_tools(tmp_path):
    """One stamping rule, two callers -- the divergence this replaced is how the
    tool entries came to lack hashes in the first place."""
    src = PREFLIGHT_SH.read_text()
    assert src.count("$(file_stamp ") == 2, "checkpoints and tools should both stamp"
    assert "CKPT_ITEMS+=" in src and "TOOL_ITEMS+=" in src
