"""Shard markers: when a rerun may skip, continue, or must refuse.

The review found that a config mismatch warned and continued. The code's own
comment justified that -- fresh designs land under beam-suffixed directory names
rather than overwriting -- but ``meta`` is empty whenever there is no metadata
tag, and then names are ``job_{job}_n_{n}_id_{counter}`` with the counter
restarting each run, against PDB writes that use overwrite=True.

Aborting needed two things beyond the fix itself, and both are asserted here:
operational flags must not change the digest (or the documented force-rerun flag
would be refused as a config change), and a marker written by an older digest
formula must not be read as a mismatch.

Requires the full environment: generate.py imports torch and the atomworks stack.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch", reason="generate.py needs the full runtime")
from omegaconf import DictConfig, ListConfig, OmegaConf

from proteinfoundation.generate import (
    GENERATION_DIGEST_IGNORED_KEYS,
    GENERATION_DIGEST_VERSION,
    ShardOutputContract,
    assert_contract_produced,
    assign_motif_csv_path,
    digest_v1_candidates,
    generation_config_digest,
    generation_save_branch,
    motif_features_entry,
    motif_info_csv_name,
    motif_info_csv_required,
    shard_already_complete,
    shard_marker_path,
    shard_output_contract,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(**overrides):
    base = {"nsamples": 16, "skip_completed_shards": True, "args": {"nsteps": 400}}
    base.update(overrides)
    return OmegaConf.create(base)


def write_marker(tmp_path, digest="a" * 64, *, version=GENERATION_DIGEST_VERSION, **extra):
    path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"generation_config_sha256": digest, "nsamples": 16, "sample_dirs": [], **extra}
    if version is not None:
        payload["generation_config_digest_version"] = version
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return path


# ------------------------------------------------------------------- digest


def test_the_documented_force_rerun_flag_is_not_a_config_change():
    """skip_completed_shards says *how* to run, not *what* to generate. Hashing it
    would make using it look like a different config -- and now that a mismatch
    aborts, would refuse to do the thing the flag exists for."""
    assert generation_config_digest(cfg(skip_completed_shards=True)) == generation_config_digest(
        cfg(skip_completed_shards=False)
    )


def test_a_real_setting_still_changes_the_digest():
    assert generation_config_digest(cfg()) != generation_config_digest(cfg(args={"nsteps": 200}))
    assert generation_config_digest(cfg()) != generation_config_digest(cfg(nsamples=32))


def test_absent_operational_key_matches_present_one():
    """A config that never set the flag and one that set it to the default are the
    same request."""
    without = OmegaConf.create({"nsamples": 16, "args": {"nsteps": 400}})
    assert generation_config_digest(cfg()) == generation_config_digest(without)


def test_digesting_does_not_mutate_the_callers_config():
    config = cfg()
    generation_config_digest(config)
    assert all(key in config for key in GENERATION_DIGEST_IGNORED_KEYS)


def test_digest_is_stable_across_calls():
    assert generation_config_digest(cfg()) == generation_config_digest(cfg())


# ------------------------------------------------------------ skip decisions


def test_no_marker_means_generate(tmp_path):
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_matching_digest_skips(tmp_path):
    write_marker(tmp_path, "a" * 64)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True


def test_forcing_a_rerun_clears_what_it_is_about_to_redo(tmp_path):
    """The digest matches, so this is a request to redo exactly this shard.
    Leaving the old directories made the result a mix of two attempts, with the
    colliding names overwritten and the rest left beside them -- the branch even
    warned that names would differ, which the mismatch path above it corrects."""
    produced = tmp_path / "job_0_n_100_id_0"
    produced.mkdir()
    (produced / "job_0_n_100_id_0.pdb").write_text("ATOM\n")
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"])

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=False) is False
    assert not produced.exists(), "the recorded output should be gone before regenerating"
    assert not os.path.exists(shard_marker_path(str(tmp_path), 0)), "the marker should go too"


def test_a_forced_rerun_refuses_when_the_marker_names_no_directories(tmp_path):
    """The shape the reviewer specified: a matching legacy marker with no
    sample_dirs, an existing sample directory, skip disabled.

    sample_dirs was added after markers themselves, so a campaign finished between
    those two commits has a valid digest-matching marker that names nothing.
    Clearing it removes nothing and reports nothing remaining, which let the run
    proceed over output it never identified -- the same overwrite this branch
    exists to prevent, through the one marker shape that cannot be checked.
    """
    produced = tmp_path / "job_0_n_100_id_0"
    produced.mkdir()
    (produced / "job_0_n_100_id_0.pdb").write_text("ATOM\n")
    marker_path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as handle:
        json.dump(
            {
                "generation_config_sha256": "a" * 64,
                "generation_config_digest_version": GENERATION_DIGEST_VERSION,
                "nsamples": 16,
            },  # no sample_dirs, as markers between d40066a and c09a82c were written
            handle,
        )

    with pytest.raises(SystemExit, match="records no sample directories"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=False)

    assert produced.exists(), "the output must survive a refusal"
    assert os.path.exists(marker_path), "so must the marker"


def test_the_default_skip_path_is_unaffected_by_a_legacy_marker(tmp_path):
    """Skipping writes nothing, so an unidentifiable marker is only a problem for
    a forced rerun. Refusing here would break resume for those campaigns."""
    marker_path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as handle:
        json.dump(
            {
                "generation_config_sha256": "a" * 64,
                "generation_config_digest_version": GENERATION_DIGEST_VERSION,
                "nsamples": 16,
            },
            handle,
        )
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True


@pytest.mark.parametrize(
    "marker,identifiable",
    [
        ({"sample_dirs": ["a"], "nsamples": 1}, True),
        ({"sample_dirs": [], "nsamples": 0}, True),  # produced nothing; nothing to clear
        ({"sample_dirs": [], "nsamples": 16}, False),  # a marker disagreeing with itself
        ({"nsamples": 16}, False),  # predates sample_dirs
        ({"sample_dirs": "job_0", "nsamples": 1}, False),  # not a list
    ],
)
def test_identifiable_output_is_the_narrow_case(marker, identifiable):
    """An empty list is only safe when the marker also says nothing was produced.
    Everything else the caller cannot act on."""
    from proteinfoundation.generate import shard_output_is_identifiable

    assert shard_output_is_identifiable(marker) is identifiable


# --------------------------------------------------------- file-level checks


def write_output_marker(tmp_path, outputs, *, nsamples=None, digest="a" * 64, schema=None, extra_dirs=()):
    """A marker recording the files a shard produced.

    ``schema`` writes an older payload version deliberately: markers from before
    reward CSVs joined ``outputs`` are the upgrade state, not a hypothetical.
    """
    from proteinfoundation.generate import MARKER_SCHEMA_VERSION

    path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "generation_config_sha256": digest,
                "generation_config_digest_version": GENERATION_DIGEST_VERSION,
                "marker_schema_version": MARKER_SCHEMA_VERSION if schema is None else schema,
                "nsamples": nsamples if nsamples is not None else len(outputs),
                "sample_dirs": sorted({os.path.dirname(o) for o in outputs if os.path.dirname(o)} | set(extra_dirs)),
                "outputs": sorted(outputs),
            },
            handle,
        )
    return path


def make_sample(tmp_path, dir_name, *file_names, content="ATOM\n"):
    """A sample directory holding the given files, returned as relative paths."""
    (tmp_path / dir_name).mkdir(parents=True, exist_ok=True)
    for name in file_names:
        (tmp_path / dir_name / name).write_text(content)
    return [f"{dir_name}/{name}" for name in file_names]


@pytest.mark.parametrize(
    "dir_name,files",
    [
        ("job_0_n_100_id_0", ["job_0_n_100_id_0.pdb"]),
        ("job_0_n_100_id_0", ["job_0_n_100_id_0_binder.pdb", "job_0_n_100_id_0.pdb"]),
        ("job_0_id_0_motif_M0024", ["job_0_id_0_motif_M0024.pdb"]),
    ],
)
def test_a_complete_sample_still_skips(tmp_path, dir_name, files):
    """Standard, ligand and motif shapes, all intact -- the common path must stay
    cheap, or file-level verification would cost every resume."""
    outputs = make_sample(tmp_path, dir_name, *files)
    write_output_marker(tmp_path, outputs)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True


def test_a_deleted_pdb_is_detected_though_its_directory_remains(tmp_path):
    """The directory outlives its contents. sample_dirs alone reported this shard
    complete, and evaluation was where the shortfall surfaced."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_output_marker(tmp_path, outputs)
    (tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0.pdb").unlink()
    assert (tmp_path / "job_0_n_100_id_0").is_dir(), "the directory is still there, which is the point"

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False
    assert not (tmp_path / "job_0_n_100_id_0").exists(), "the partial sample is cleared before regenerating"


def test_a_zero_byte_pdb_counts_as_missing(tmp_path):
    """An interrupted write leaves the file in place. A shard whose PDB is empty
    is not one to skip."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb", content="")
    write_output_marker(tmp_path, outputs)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_a_ligand_sample_missing_its_complex_pdb_is_incomplete(tmp_path):
    """save_protein_ligand_predictions writes a complex PDB beside every binder,
    and only pdb_paths used to reach the marker -- so the complex file was
    unverifiable however fine-grained the check became."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0_binder.pdb", "job_0_n_100_id_0.pdb")
    write_output_marker(tmp_path, outputs, nsamples=1)
    (tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0.pdb").unlink()
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_output_relocated_by_the_filter_still_counts(tmp_path):
    """The filter stage moves designs it did not keep into filtered_out_samples/;
    they are still this shard's output, so a relocated file is not a missing one."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_output_marker(tmp_path, outputs)
    (tmp_path / "filtered_out_samples").mkdir()
    (tmp_path / "job_0_n_100_id_0").rename(tmp_path / "filtered_out_samples" / "job_0_n_100_id_0")
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True


def test_a_legacy_marker_with_intact_output_still_resumes(tmp_path):
    """Markers written before per-file records must keep resuming -- they are the
    state every existing campaign meets on upgrade."""
    make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"])
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True


def test_a_legacy_marker_does_not_fail_open_on_a_deleted_pdb(tmp_path):
    """This replaces a test that codified the opposite. A legacy marker cannot
    name the file it expects, but it can require that the directory still holds a
    usable design -- which is the deletion it was previously blind to. Logging
    that verification is weaker does not restore the artifact."""
    make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"])
    (tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0.pdb").unlink()
    assert (tmp_path / "job_0_n_100_id_0").is_dir()

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_a_legacy_marker_treats_an_empty_pdb_as_missing(tmp_path):
    make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb", content="")
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"])
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_a_marker_claiming_samples_but_recording_no_outputs_refuses(tmp_path):
    """marker_schema_version was written and never consulted. A current-schema
    marker with a positive sample count and no outputs disagrees with itself, so
    nothing it says can be acted on -- distinct from a legacy marker, which is
    silent rather than contradictory."""
    from proteinfoundation.generate import MARKER_SCHEMA_VERSION

    path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "generation_config_sha256": "a" * 64,
                "generation_config_digest_version": GENERATION_DIGEST_VERSION,
                "marker_schema_version": MARKER_SCHEMA_VERSION,
                "nsamples": 4,
                "sample_dirs": [],
                "outputs": [],
            },
            handle,
        )
    with pytest.raises(SystemExit, match="records no output files"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)


# ------------------------------------------------------------- reward CSVs


def test_a_missing_reward_csv_stops_the_shard_being_skipped(tmp_path):
    """The filter stage reads rewards_{config}_{job}.csv and raises "No reward
    files found!" without any. Across several shards it is quieter and worse: it
    processes the ones it can see, so evaluation covers fewer designs than
    generation claimed, with nothing reporting the shortfall."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / "rewards_cfg_0.csv").write_text("sample,reward\n")
    write_output_marker(tmp_path, outputs + ["rewards_cfg_0.csv"], nsamples=1)

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True
    (tmp_path / "rewards_cfg_0.csv").unlink()
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


# ------------------------------------------------------------- readability


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads regardless of mode")
def test_an_unreadable_pdb_counts_as_missing(tmp_path):
    """isfile() and getsize() both read metadata and neither opens anything, so a
    mode-000 file passed a check whose docstring claimed to detect unreadable
    output."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_output_marker(tmp_path, outputs)
    target = tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0.pdb"
    target.chmod(0)
    try:
        assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False
    finally:
        # Detecting the unusable file makes this a damaged shard, so the caller
        # clears it -- the file this restores is gone by design. Guarded rather
        # than removed: if the detection ever stops happening, the mode must
        # still be restored or pytest cannot clean up its own tmp_path.
        if target.exists():
            target.chmod(0o644)

    assert not target.parent.exists(), "the damaged shard is cleared before regenerating"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads regardless of mode")
def test_is_usable_output_separates_the_three_failures(tmp_path):
    from proteinfoundation.generate import is_usable_output

    good = tmp_path / "good.pdb"
    good.write_text("ATOM\n")
    empty = tmp_path / "empty.pdb"
    empty.write_text("")
    locked = tmp_path / "locked.pdb"
    locked.write_text("ATOM\n")
    locked.chmod(0)
    try:
        assert is_usable_output(str(good)) is True
        assert is_usable_output(str(empty)) is False
        assert is_usable_output(str(locked)) is False
        assert is_usable_output(str(tmp_path / "absent.pdb")) is False
        assert is_usable_output(str(tmp_path)) is False  # a directory is not an output
    finally:
        locked.chmod(0o644)


def test_the_marker_records_every_file_including_ligand_complexes(tmp_path):
    """Guards the write side: nsamples counts designs, outputs counts files."""
    from proteinfoundation.generate import write_shard_marker

    binder = tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0_binder.pdb"
    complex_pdb = tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0_complex.pdb"
    binder.parent.mkdir(parents=True)
    for f in (binder, complex_pdb):
        f.write_text("ATOM\n")

    path = write_shard_marker(str(tmp_path), 0, 1, "a" * 64, [str(binder)], extra_output_paths=[str(complex_pdb)])
    with open(path) as handle:
        marker = json.load(handle)
    assert marker["nsamples"] == 1, "the complex PDB is not a second sample"
    assert marker["outputs"] == sorted(
        ["job_0_n_100_id_0/job_0_n_100_id_0_binder.pdb", "job_0_n_100_id_0/job_0_n_100_id_0_complex.pdb"]
    )
    assert marker["sample_dirs"] == ["job_0_n_100_id_0"]


def test_write_marker_returns_its_path(tmp_path):
    """Guards the fixture the cleanup tests rely on."""
    path = write_marker(tmp_path, "a" * 64)
    assert os.path.exists(path)


@pytest.mark.parametrize("skip_enabled", [True, False])
def test_a_different_config_refuses_rather_than_overwriting(tmp_path, skip_enabled):
    """Continuing would overwrite structures wherever directory names coincide and
    leave the previous run's evaluation files beside the new designs -- results
    describing designs that no longer exist."""
    write_marker(tmp_path, "a" * 64)
    with pytest.raises(SystemExit, match="Refusing to generate"):
        shard_already_complete(str(tmp_path), 0, "b" * 64, skip_enabled=skip_enabled)


def test_the_refusal_names_both_recoveries(tmp_path):
    write_marker(tmp_path, "a" * 64)
    with pytest.raises(SystemExit) as excinfo:
        shard_already_complete(str(tmp_path), 0, "b" * 64, skip_enabled=True)
    message = str(excinfo.value)
    assert "run_name" in message and "clear" in message.lower()


@pytest.mark.parametrize("version", [None, 1])
def test_an_unrecognisable_legacy_marker_refuses(tmp_path, version):
    """This test previously asserted the opposite, and was wrong to.

    Continuing let the first post-upgrade resume of every existing campaign write
    into a populated root whose config could not be checked -- overwriting
    structures, since the counters restart, and leaving the previous run's
    evaluation files beside the new designs. Being wrong that way loses data;
    being wrong the other way costs one command.
    """
    write_marker(tmp_path, "a" * 64, version=version)
    with pytest.raises(SystemExit, match="Refusing to generate"):
        shard_already_complete(str(tmp_path), 0, "b" * 64, skip_enabled=True)


@pytest.mark.parametrize("version", [None, 1])
def test_a_legacy_marker_whose_v1_digest_matches_is_the_same_request(tmp_path, version):
    """So the upgrade does not force a rename on campaigns that did not change."""
    config = cfg()
    legacy = digest_v1_candidates(config)
    write_marker(tmp_path, sorted(legacy)[0], version=version)
    assert (
        shard_already_complete(
            str(tmp_path),
            0,
            generation_config_digest(config),
            skip_enabled=True,
            legacy_digests=legacy,
        )
        is True
    )


def test_v1_candidates_cover_the_operational_key_it_used_to_hash(tmp_path):
    """v1 hashed skip_completed_shards, v2 does not, so an unchanged config has
    several possible v1 digests. All of them must be recognised or the flag's
    value at marker-write time decides whether a resume is refused."""
    for value in (True, False):
        assert digest_v1_candidates(cfg(skip_completed_shards=value)) == digest_v1_candidates(cfg())
    assert len(digest_v1_candidates(cfg())) >= 2


def test_a_changed_config_is_still_refused_under_the_legacy_path(tmp_path):
    """Recomputing v1 must not become a way to wave through a real change."""
    write_marker(tmp_path, "a" * 64, version=1)
    with pytest.raises(SystemExit, match="Refusing to generate"):
        shard_already_complete(
            str(tmp_path),
            0,
            generation_config_digest(cfg()),
            skip_enabled=True,
            legacy_digests=digest_v1_candidates(cfg(args={"nsteps": 999})),
        )


def test_an_unreadable_marker_refuses(tmp_path):
    """This test previously asserted the opposite, and was wrong to -- for the
    second time in this file.

    A marker exists only because a shard finished, so its directory holds output;
    an unreadable one is most likely a write interrupted after the structures were
    produced. Continuing restarts the deterministic counters over that output, and
    since the marker cannot be read there is no list of directories to clear
    either.
    """
    path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("{not json")
    with pytest.raises(SystemExit, match="cannot be read"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)


def test_an_absent_marker_over_an_empty_directory_is_an_ordinary_new_run(tmp_path):
    """Still the common case, and it must stay cheap: no marker, nothing on disk."""
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_an_absent_marker_over_existing_output_refuses(tmp_path):
    """The docstring on the test this replaces claimed absence means nothing was
    generated here. That does not follow from the write ordering: sample
    directories are created inside the save loop and the marker only after every
    design is handled, so a kill in between leaves populated output with no
    marker. Retrying restarted the deterministic counters over it.
    """
    produced = tmp_path / "job_0_n_100_id_0"
    produced.mkdir()
    (produced / "job_0_n_100_id_0.pdb").write_text("ATOM\n")

    with pytest.raises(SystemExit, match="no completion marker"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)
    assert produced.exists(), "an interrupted run's output must survive the refusal"


def test_output_moved_to_filtered_out_still_counts_as_existing(tmp_path):
    """The filter stage relocates designs it did not keep; they are still this
    shard's output and still collide on a rerun."""
    filtered = tmp_path / "filtered_out_samples" / "job_0_n_100_id_0"
    filtered.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no completion marker"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)


def test_another_jobs_output_does_not_block_this_one(tmp_path):
    """Shards are generated in parallel into one root, so job 0 must not refuse
    because job 1 has written directories. The _n_ separator also keeps job 1 from
    matching job 10."""
    for name in ("job_1_n_100_id_0", "job_10_n_100_id_0"):
        (tmp_path / name).mkdir()

    # job 0 owns nothing here, so it proceeds
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False
    # job 1 does, so it refuses -- and not because of job 10's directory
    with pytest.raises(SystemExit, match="job_1_n_100_id_0"):
        shard_already_complete(str(tmp_path), 1, "a" * 64, skip_enabled=True)


@pytest.mark.parametrize(
    "dir_name,save_path",
    [
        ("job_0_n_100_id_0", "save_predictions"),
        ("job_0_n_100_id_0_beam_bm0", "save_predictions with a metadata tag"),
        ("job_0_n_100_id_0_binder", "save_protein_ligand_predictions"),
        ("job_0_id_0_motif_M0024_1nzy", "save_motif_predictions"),
    ],
)
def test_every_save_path_is_recognised_as_this_jobs_output(tmp_path, dir_name, save_path):
    """Three save paths, two naming schemes. The motif one carries no _n_ at all,
    so a scan written against the standard format missed interrupted motif output
    entirely -- which is the shape this parametrisation exists to prevent
    recurring when a fourth format appears."""
    (tmp_path / dir_name).mkdir()
    with pytest.raises(SystemExit, match="no completion marker"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)


@pytest.mark.parametrize("dir_name", ["job_0_n_100_id_0", "job_0_id_0_motif_M0024"])
def test_filtered_out_output_is_recognised_for_every_format(tmp_path, dir_name):
    (tmp_path / "filtered_out_samples" / dir_name).mkdir(parents=True)
    with pytest.raises(SystemExit, match="no completion marker"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)


@pytest.mark.parametrize("dir_name", ["job_10_n_100_id_0", "job_10_id_0_motif_M0024"])
def test_job_prefixes_do_not_collide_in_either_format(tmp_path, dir_name):
    """job_1_ must not prefix-match job_10_, whichever naming scheme follows."""
    from proteinfoundation.generate import existing_shard_dirs

    (tmp_path / dir_name).mkdir()
    assert existing_shard_dirs(str(tmp_path), 1) == []
    assert len(existing_shard_dirs(str(tmp_path), 10)) == 1


def test_stray_files_are_not_mistaken_for_output(tmp_path):
    """The scan is a prefix match over directories, so two things must hold: other
    root-level files do not begin with job_{id}_ (timing is timing_*), and a file
    that happens to share the prefix is not a sample directory. Either would block
    every new run in that directory.

    Deliberately no shard_*_complete.json here -- writing one is writing a marker,
    which takes a different branch entirely and would make this test pass for the
    wrong reason.
    """
    (tmp_path / "timing_0.csv").write_text("job_id\n")
    (tmp_path / "job_0_n_100_id_0.pdb").write_text("ATOM\n")  # a file, not a directory
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_a_partial_clear_refuses_and_keeps_the_marker(tmp_path, monkeypatch):
    """Cleanup catches OSError per directory, so a permission failure part way
    through used to leave the caller regenerating over what survived -- with the
    marker already deleted, taking the record of which directories belonged to the
    shard with it."""
    from proteinfoundation import generate as gen

    kept = tmp_path / "job_0_n_100_id_0"
    gone = tmp_path / "job_0_n_100_id_1"
    for d in (kept, gone):
        d.mkdir()
    marker_path = write_marker(tmp_path, "a" * 64, sample_dirs=[kept.name, gone.name])

    # Capture the *function*, not the module. gen.shutil is the global shutil
    # module, so `import shutil as real_shutil` aliases the same object and
    # real_shutil.rmtree resolves to whatever is patched at call time -- which is
    # this wrapper, recursing until the stack ends.
    real_rmtree = gen.shutil.rmtree

    def refuse_one(path, *args, **kwargs):
        if path.endswith(kept.name):
            raise OSError(13, "Permission denied")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(gen.shutil, "rmtree", refuse_one)
    with pytest.raises(SystemExit, match="could not be cleared"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=False)

    assert kept.exists(), "the directory that could not be removed is still there"
    assert os.path.exists(marker_path), "the marker must survive so recovery is possible"


def test_clear_reports_what_survived(tmp_path, monkeypatch):
    """Checked on disk after the attempt rather than inferred from which calls
    raised -- a delete that reports success but leaves the directory is the case
    that matters."""
    from proteinfoundation import generate as gen

    d = tmp_path / "job_0_n_100_id_0"
    d.mkdir()
    marker = {"sample_dirs": [d.name]}

    removed, remaining = gen.clear_shard_output(str(tmp_path), marker)
    assert (removed, remaining) == (1, [])

    d.mkdir()
    monkeypatch.setattr(gen.shutil, "rmtree", lambda *a, **k: None)  # silent no-op
    removed, remaining = gen.clear_shard_output(str(tmp_path), marker)
    assert removed == 1, "the no-op reported success"
    assert remaining == [str(d)], "...but the directory is still there, which is what counts"
    # monkeypatch undoes the patch at teardown; restoring it here by reading
    # gen.shutil.rmtree would read back the no-op.


# ------------------------------------------- legacy markers: the exact design


@pytest.mark.parametrize(
    "survivor,why",
    [
        ("job_0_n_100_id_0_binder.pdb", "an evaluation sidecar, which is derived and not the design"),
        ("job_0_n_100_id_0_binder.pdb", "a ligand binder, whose complex PDB is the file that went"),
    ],
    ids=["evaluation-sidecar", "ligand-binder"],
)
def test_another_pdb_in_the_directory_does_not_answer_for_the_design(tmp_path, survivor, why):
    """Any-.pdb was too weak. All three save paths write {dir}/{dir}.pdb, so the
    expected name is reconstructable from the directory name -- and the files that
    were vouching for it are exactly the ones that must not.

    binder_eval.py:771 writes a {dir}_binder.pdb sidecar into ordinary sample
    directories; it outlives the design it was extracted from, while evaluation
    itself looks for the base PDB and skips the design when that is gone. A ligand
    sample holds both files for real, and deleting the complex left the binder to
    vouch for it."""
    make_sample(tmp_path, "job_0_n_100_id_0", survivor)
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"])
    assert (tmp_path / "job_0_n_100_id_0" / survivor).exists(), why

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


# ------------------------------------------------- required root-level output


REWARDS = "rewards_binder_generate_0.csv"
REWARD_CONTRACT = shard_output_contract("standard", "binder_generate", 0)
LIGAND_CONTRACT = shard_output_contract("ligand", "binder_generate", 0)


def test_a_reward_csv_missing_from_a_reward_unaware_marker_blocks_the_skip(tmp_path):
    """The upgrade state this exists for: a marker written when `outputs` held only
    PDBs, claiming the same schema version as one written after reward CSVs joined
    it. Its recorded files all survive, so every marker-derived check says complete
    -- while the filter, which globs rewards_{config}_*.csv in the output root and
    nowhere else, would silently evaluate the other shards."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_output_marker(tmp_path, outputs, schema=2)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True, (
        "the marker's own records are intact, which is why asking it cannot settle this"
    )

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=REWARD_CONTRACT) is False
    assert not (tmp_path / "job_0_n_100_id_0").exists(), "the shard is cleared, not just refused"


def test_a_recorded_and_present_reward_csv_still_skips(tmp_path):
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / REWARDS).write_text("pdb_path,total_reward\n")
    write_output_marker(tmp_path, outputs + [REWARDS], nsamples=1)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=REWARD_CONTRACT) is True


def test_a_zero_sample_shard_needs_no_reward_csv(tmp_path):
    """save_rewards_to_csv runs only for a nonempty reward frame, so requiring one
    of a shard that generated nothing would refuse a resume over a file that was
    never meant to exist."""
    write_output_marker(tmp_path, [], nsamples=0)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=REWARD_CONTRACT) is True


def test_a_reward_csv_under_filtered_out_samples_does_not_count(tmp_path):
    """The relocation fallback is right for sample directories -- the filter moves
    those -- and wrong for root-level files, which it does not move and does not
    look under. Accepting one there reports the shard complete while filtering
    cannot see its rewards."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / "filtered_out_samples").mkdir()
    (tmp_path / "filtered_out_samples" / REWARDS).write_text("pdb_path,total_reward\n")
    write_output_marker(tmp_path, outputs + [REWARDS], nsamples=1)

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_clearing_a_damaged_shard_removes_its_reward_csv(tmp_path):
    """Clearing removed sample directories only, which cleared the whole shard
    while every output lived inside one. Reward CSVs are root-level: the CSV
    survived, generation repeated the GPU run, and writing over the file it had not
    removed was where it failed."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / REWARDS).write_text("pdb_path,total_reward\n")
    write_output_marker(tmp_path, outputs + [REWARDS], nsamples=1)
    (tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0.pdb").unlink()

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False
    assert not (tmp_path / REWARDS).exists(), "the root-level output is cleared with the rest of the shard"


def test_a_root_output_that_survives_cleanup_aborts_and_keeps_the_marker(tmp_path, monkeypatch):
    """The same rule a surviving directory gets. Regenerating would meet the file
    it could not remove, and the marker is the only record of what the shard owns,
    so it stays."""
    import proteinfoundation.generate as gen

    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / REWARDS).write_text("pdb_path,total_reward\n")
    marker = write_output_marker(tmp_path, outputs + [REWARDS], nsamples=1)
    (tmp_path / "job_0_n_100_id_0" / "job_0_n_100_id_0.pdb").unlink()

    real_remove = gen.os.remove  # captured, so the patch cannot call itself

    def refuse_the_csv(path, *args, **kwargs):
        if str(path).endswith(".csv"):
            raise PermissionError(f"refusing {path}")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(gen.os, "remove", refuse_the_csv)

    with pytest.raises(SystemExit, match="could not be cleared"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)
    assert (tmp_path / REWARDS).exists(), "the file that could not be removed is still there"
    assert os.path.exists(marker), "the marker records what this shard owns; a failed clear must not take it"


def test_the_outputs_schema_floor_survives_a_version_bump(tmp_path):
    """A marker at the schema that introduced `outputs`, recording none beside a
    positive sample count, contradicts itself and must still refuse. Bumping
    MARKER_SCHEMA_VERSION to 3 would have made this fall through to the legacy
    path, undoing the check a commit after it was added -- which is why the floor
    is a separate constant."""
    from proteinfoundation.generate import MARKER_SCHEMA_WITH_OUTPUTS

    make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    write_marker(
        tmp_path,
        "a" * 64,
        marker_schema_version=MARKER_SCHEMA_WITH_OUTPUTS,
        nsamples=16,
        sample_dirs=["job_0_n_100_id_0"],
    )
    with pytest.raises(SystemExit, match="records no output files"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True)


# ------------------------------------------------ which save path this config takes


def ame_conditional_features():
    """The feature classes the shipped AME config actually declares.

    Read from the file rather than asserted as a boolean, because the finding was
    not that the expression was wrong in the abstract -- it was that no config was
    consulted about which branch it takes. Resolution is off: the config is full of
    interpolations over campaign variables that are unset here, and none of them
    affect which _target_ classes are listed.
    """
    ame = OmegaConf.load(os.path.join(REPO_ROOT, "configs", "pipeline", "ame", "ame_generate.yaml"))
    found = []

    def walk(node):
        if isinstance(node, DictConfig):
            for key in node.keys():
                if key == "_target_":
                    found.append(str(node._get_node(key)).split(".")[-1])
                else:
                    walk(node._get_node(key))
        elif isinstance(node, ListConfig):
            for i in range(len(node)):
                walk(node._get_node(i))

    walk(ame)
    return found


def test_the_shipped_ame_config_declares_both_motif_and_ligand_features():
    """The premise of the bug, checked against the file. If AME ever stops setting
    both, the test below is testing nothing and should be revisited rather than
    quietly passing."""
    features = ame_conditional_features()
    assert "MotifFeatures" in features
    assert "LigandFeatures" in features


def test_a_config_setting_both_features_saves_on_the_ligand_branch():
    """main dispatches `if ligand_cond ... elif motif_cond ...`, so a config with
    both saves ligand output and writes a reward CSV. The resume check asked
    "motif?" first and so expected no rewards from every AME shard -- each site
    reading correctly on its own, in incompatible orders."""
    features = ame_conditional_features()
    branch = generation_save_branch("LigandFeatures" in features, "MotifFeatures" in features)
    assert branch == "ligand"
    assert shard_output_contract(branch, "ame_generate", 0).root_outputs == ("rewards_ame_generate_0.csv",)


def test_only_a_motif_run_without_a_ligand_expects_no_rewards():
    assert generation_save_branch(False, True) == "motif"
    assert shard_output_contract("motif", "ame_generate", 0).root_outputs == ()
    assert generation_save_branch(False, False) == "standard"


def test_an_ame_shard_missing_its_reward_csv_is_not_skipped(tmp_path):
    """The whole point, end to end: a reward-unaware marker whose recorded PDBs all
    survive, under a config that sets both features."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb", "job_0_n_100_id_0_binder.pdb")
    write_output_marker(tmp_path, outputs, schema=2, nsamples=1)
    contract = shard_output_contract(generation_save_branch(True, True), "ame_generate", 0)

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=contract) is False


# --------------------------------------- the per-sample contract, for old markers


def test_a_legacy_ligand_sample_missing_its_binder_is_not_intact(tmp_path):
    """The inverse of the sidecar case, and the residual I documented instead of
    fixing last round. A ligand sample needs both files; a marker without `outputs`
    does not record that it was a ligand run. That uncertainty is a reason to ask
    the config, not to read the surviving complex as completeness."""
    make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")  # complex kept, binder gone
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"])

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True, (
        "under a non-ligand config that directory is complete, which is why the contract decides"
    )
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=LIGAND_CONTRACT) is False


def test_a_legacy_ligand_sample_with_both_files_still_resumes(tmp_path):
    make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb", "job_0_n_100_id_0_binder.pdb")
    (tmp_path / REWARDS).write_text("pdb_path,total_reward\n")
    write_marker(tmp_path, "a" * 64, sample_dirs=["job_0_n_100_id_0"], nsamples=1)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=LIGAND_CONTRACT) is True


def test_the_default_contract_asks_for_the_design_alone(tmp_path):
    """Callers that pass no contract keep the previous rule, so the parameter adds a
    requirement where a config states one rather than everywhere."""
    assert ShardOutputContract().files_for("job_0_n_100_id_0") == ["job_0_n_100_id_0.pdb"]


# --------------------------------- cleanup owns required output the marker omits


def test_an_unrecorded_reward_csv_is_cleared_with_the_shard(tmp_path):
    """Detected via the contract, so it must be cleared via the contract too --
    clearing from marker.outputs alone left the file whose absence triggered the
    clear, and reported success because it was equally absent from the check."""
    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / REWARDS).write_text("")  # present but empty: unusable, so it triggers the clear
    write_output_marker(tmp_path, outputs, schema=2, nsamples=1)

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=REWARD_CONTRACT) is False
    assert not (tmp_path / REWARDS).exists(), "the required file is cleared even though the marker never named it"
    assert not (tmp_path / "job_0_n_100_id_0").exists()


def test_an_unrecorded_reward_csv_that_cannot_be_removed_aborts(tmp_path, monkeypatch):
    """And if it will not go, the run stops with the marker kept -- rather than
    repeating the GPU work only to fail writing over it."""
    import proteinfoundation.generate as gen

    outputs = make_sample(tmp_path, "job_0_n_100_id_0", "job_0_n_100_id_0.pdb")
    (tmp_path / REWARDS).write_text("")
    marker = write_output_marker(tmp_path, outputs, schema=2, nsamples=1)

    real_remove = gen.os.remove

    def refuse_the_csv(path, *args, **kwargs):
        if str(path).endswith(".csv"):
            raise PermissionError(f"refusing {path}")
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(gen.os, "remove", refuse_the_csv)

    with pytest.raises(SystemExit, match="could not be cleared"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=REWARD_CONTRACT)
    assert (tmp_path / REWARDS).exists()
    assert os.path.exists(marker), "a failed clear must not take the record of what the shard owns"


# ------------------------------------------------- the motif contig table


def conditional_features_of(*config_path):
    """The `conditional_features` node of a shipped config, unresolved.

    Unresolved on purpose: these configs interpolate over campaign variables that
    are unset here, and the production helper has to cope with exactly that.
    """
    cfg = OmegaConf.load(os.path.join(REPO_ROOT, "configs", *config_path))
    found = []

    def walk(node):
        if isinstance(node, DictConfig):
            for key in node.keys():
                if key == "conditional_features":
                    found.append(node._get_node(key))
                else:
                    walk(node._get_node(key))
        elif isinstance(node, ListConfig):
            for i in range(len(node)):
                walk(node._get_node(i))

    walk(cfg)
    assert found, "the config declares no conditional_features"
    return found[0]


def test_the_motif_info_csv_name_follows_the_convention_evaluation_documents():
    """motif_eval's FileNotFoundError tells the user to place the file at
    {task_name}_{job_id}_motif_info.csv, so that is what generation must write and
    what the contract must ask for."""
    assert motif_info_csv_name("1YCR_AA", 3) == "1YCR_AA_3_motif_info.csv"
    assert motif_info_csv_name(None, 0) == "motif_0_motif_info.csv"


def test_the_shipped_indexed_motif_config_requires_its_contig_table():
    """idx_motif_generate.yaml sets contig_string and no motif_atom_spec, so
    MotifFeatures writes the CSV -- and indexed evaluation cannot run without it."""
    features = conditional_features_of("pipeline", "motif", "idx_motif_generate.yaml")
    assert motif_info_csv_required(features, True, "1YCR_AA", 0) == "1YCR_AA_0_motif_info.csv"


def test_an_atom_spec_config_requires_no_contig_table():
    """gen_dataset writes the CSV only when motif_atom_spec is None. Requiring it
    of an atom-spec run would clear and regenerate a healthy shard over a file that
    was never meant to exist."""
    features = OmegaConf.create(
        [{"_target_": "proteinfoundation.datasets.gen_dataset.MotifFeatures", "motif_atom_spec": "A64: [O, CG]"}]
    )
    assert motif_info_csv_required(features, True, "T", 0) is None
    assert motif_info_csv_required(features, False, "T", 0) is None, "not a motif run at all"
    assert motif_info_csv_required(None, True, "T", 0) is None


def test_an_unresolvable_atom_spec_does_not_demand_a_contig_table():
    """AME interpolates motif_atom_spec over a campaign variable that is unset
    outside a real run. Reading it raises, and the fail-open direction is chosen
    deliberately: guessing wrong here costs a loud FileNotFoundError from
    evaluation that names the file, while guessing wrong the other way clears a
    completed shard's GPU work over a file that never existed."""
    features = conditional_features_of("pipeline", "ame", "ame_generate.yaml")
    assert motif_info_csv_required(features, True, "some_task", 0) is None


def test_a_ligand_save_can_still_owe_a_contig_table():
    """The requirement hangs off the MotifFeatures mode, not off the save branch --
    a config can set both features, save ligand output, and have written the CSV.
    Folding this into the branch would repeat the AME mistake in a new place."""
    contract = shard_output_contract("ligand", "ame_generate", 0, "some_task_0_motif_info.csv")
    assert contract.root_outputs == ("rewards_ame_generate_0.csv", "some_task_0_motif_info.csv")
    assert shard_output_contract("motif", "idx_motif_generate", 0).root_outputs == ()


MOTIF_DIR = "job_0_id_0_motif_1YCR_AA"
MOTIF_CSV = "1YCR_AA_0_motif_info.csv"
MOTIF_CONTRACT = shard_output_contract("motif", "idx_motif_generate", 0, MOTIF_CSV)


def test_a_motif_shard_missing_its_contig_table_is_not_skipped(tmp_path):
    """The marker recorded intact designs, so every marker-derived check said
    complete -- and indexed evaluation then failed deterministically on the file
    generation never claimed to owe."""
    outputs = make_sample(tmp_path, MOTIF_DIR, f"{MOTIF_DIR}.pdb")
    write_output_marker(tmp_path, outputs, nsamples=1)

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is True, (
        "the recorded output all survives, which is why the marker cannot settle this"
    )
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=MOTIF_CONTRACT) is False
    assert not (tmp_path / MOTIF_DIR).exists(), "the shard is cleared, not just refused"


def test_a_motif_shard_with_its_contig_table_skips(tmp_path):
    outputs = make_sample(tmp_path, MOTIF_DIR, f"{MOTIF_DIR}.pdb")
    (tmp_path / MOTIF_CSV).write_text("sample,contig\n")
    write_output_marker(tmp_path, outputs + [MOTIF_CSV], nsamples=1)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=MOTIF_CONTRACT) is True


def test_clearing_a_motif_shard_removes_its_contig_table(tmp_path):
    """Recorded or not, it is required output, so it goes with the rest of the
    shard -- otherwise regeneration meets a stale contig table from the run before."""
    outputs = make_sample(tmp_path, MOTIF_DIR, f"{MOTIF_DIR}.pdb")
    (tmp_path / MOTIF_CSV).write_text("sample,contig\n")
    write_output_marker(tmp_path, outputs, nsamples=1)  # an older marker: CSV not recorded
    (tmp_path / MOTIF_DIR / f"{MOTIF_DIR}.pdb").unlink()

    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True, contract=MOTIF_CONTRACT) is False
    assert not (tmp_path / MOTIF_CSV).exists()


# ------------------------------- getting the contig table actually written


MOTIF_CONFIGS = [("pipeline", "motif", "idx_motif_generate.yaml"), ("pipeline", "motif", "uidx_motif_generate.yaml")]


@pytest.mark.parametrize("config_path", MOTIF_CONFIGS, ids=["indexed", "unindexed"])
def test_no_shipped_motif_config_declares_the_csv_path(config_path):
    """The premise of the bug. main guarded the assignment on the key already being
    there, and no config puts it there -- so the contig table was never written and
    indexed evaluation raised FileNotFoundError every time."""
    entry = motif_features_entry(conditional_features_of(*config_path))
    assert entry is not None
    assert "motif_csv_path" not in entry


def test_hasattr_stays_false_for_an_undeclared_key_inside_open_dict():
    """The mechanism, asserted directly because it is what made the old guard look
    reasonable. open_dict permits *adding* a key; it does not make hasattr true for
    one that is absent."""
    from omegaconf import open_dict

    entry = OmegaConf.create({"_target_": "x.MotifFeatures"})
    assert hasattr(entry, "motif_csv_path") is False
    with open_dict(entry):
        assert hasattr(entry, "motif_csv_path") is False, "which is why the guarded assignment never ran"
        entry.motif_csv_path = "/out/t_0_motif_info.csv"
    assert entry.motif_csv_path == "/out/t_0_motif_info.csv", "assigning directly does work"


@pytest.mark.parametrize("config_path", MOTIF_CONFIGS, ids=["indexed", "unindexed"])
def test_the_path_assignment_lands_on_a_shipped_motif_config(config_path):
    """What main now does, against the real configs. Asserted through the list node
    as well, since it is the list that gets instantiated, not the entry we held."""
    features = conditional_features_of(*config_path)
    assert assign_motif_csv_path(features, "/out/1YCR_AA_0_motif_info.csv") is True
    assert features[0].get("motif_csv_path") == "/out/1YCR_AA_0_motif_info.csv"


def test_the_assignment_reports_when_there_is_no_motif_features_entry():
    """Reported rather than silent: a motif run whose features declare no
    MotifFeatures writes no table, and the caller warns instead of assuming."""
    assert assign_motif_csv_path(OmegaConf.create([{"_target_": "x.LigandFeatures"}]), "/out/x.csv") is False
    assert assign_motif_csv_path(None, "/out/x.csv") is False


def test_motif_features_really_takes_the_parameter_we_set():
    """Setting a key hydra would reject is the same bug in a new place -- the
    assignment would land and instantiation would fail."""
    import inspect

    from proteinfoundation.datasets.gen_dataset import MotifFeatures

    assert "motif_csv_path" in inspect.signature(MotifFeatures.__init__).parameters


# ----------------------------- a marker is not written over an unfinished shard


def test_a_shard_that_did_not_produce_what_it_owes_gets_no_marker(tmp_path):
    """Without this the missing file is a loop, not a failure: the run writes a
    marker that omits the file because the file is not there, and the next run's
    contract check clears the designs and regenerates them to the same end."""
    with pytest.raises(SystemExit, match=MOTIF_CSV):
        assert_contract_produced(str(tmp_path), MOTIF_CONTRACT, 0)
    assert not os.path.exists(shard_marker_path(str(tmp_path), 0)), "no marker, so nothing is skipped later"


def test_a_shard_that_produced_everything_passes_the_check(tmp_path):
    (tmp_path / MOTIF_CSV).write_text("sample,contig\n")
    assert_contract_produced(str(tmp_path), MOTIF_CONTRACT, 0)  # does not raise


def test_an_empty_but_present_required_output_is_not_produced(tmp_path):
    """Same rule as verification uses: an interrupted write leaves the file there."""
    (tmp_path / MOTIF_CSV).write_text("")
    with pytest.raises(SystemExit, match="did not produce"):
        assert_contract_produced(str(tmp_path), MOTIF_CONTRACT, 0)
