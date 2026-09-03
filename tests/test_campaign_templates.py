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
    assert 'WANT_DESIGNS="${2:?' in runner, "required, with a message"
    assert 'STAGE="${3:-all}"' in runner, "the stage shifts along"


def test_a_followup_gets_its_own_run_name_and_metadata():
    """RUN_NAME feeds both output directories and KIND_TAG feeds the metadata, so
    two follow-ups sharing either would overwrite each other -- and generation
    refuses to write into a shard whose marker disagrees, so the second would
    simply not run."""
    runner = RUNNER.read_text()
    assert 'RUN_NAME="$FOLLOWUP_RUN_NAME"' in runner
    assert 'KIND_TAG="followup${FOLLOWUP_INDEX}"' in runner
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
