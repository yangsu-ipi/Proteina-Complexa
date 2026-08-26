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
from omegaconf import OmegaConf

from proteinfoundation.generate import (
    GENERATION_DIGEST_IGNORED_KEYS,
    GENERATION_DIGEST_VERSION,
    digest_v1_candidates,
    generation_config_digest,
    shard_already_complete,
    shard_marker_path,
)


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


def test_an_absent_marker_is_still_an_ordinary_new_run(tmp_path):
    """Absence means nothing has been generated here; only an unreadable marker
    is evidence of output that cannot be accounted for."""
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False


def test_a_partial_clear_refuses_and_keeps_the_marker(tmp_path, monkeypatch):
    """Cleanup catches OSError per directory, so a permission failure part way
    through used to leave the caller regenerating over what survived -- with the
    marker already deleted, taking the record of which directories belonged to the
    shard with it."""
    import shutil as real_shutil

    from proteinfoundation import generate as gen

    kept = tmp_path / "job_0_n_100_id_0"
    gone = tmp_path / "job_0_n_100_id_1"
    for d in (kept, gone):
        d.mkdir()
    marker_path = write_marker(tmp_path, "a" * 64, sample_dirs=[kept.name, gone.name])

    def refuse_one(path, *args, **kwargs):
        if path.endswith(kept.name):
            raise OSError(13, "Permission denied")
        return real_shutil.rmtree(path, *args, **kwargs)

    monkeypatch.setattr(gen.shutil, "rmtree", refuse_one)
    with pytest.raises(SystemExit, match="could not be cleared"):
        shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=False)

    assert kept.exists(), "the directory that could not be removed is still there"
    assert os.path.exists(marker_path), "the marker must survive so recovery is possible"


def test_clear_reports_what_survived(tmp_path, monkeypatch):
    """Checked on disk after the attempt rather than inferred from which calls
    raised -- a delete that reports success but leaves the directory is the case
    that matters."""
    import shutil as real_shutil

    from proteinfoundation import generate as gen

    d = tmp_path / "job_0_n_100_id_0"
    d.mkdir()
    marker = {"sample_dirs": [d.name]}

    removed, remaining = gen.clear_shard_output(str(tmp_path), marker)
    assert (removed, remaining) == (1, [])

    d.mkdir()
    monkeypatch.setattr(gen.shutil, "rmtree", lambda *a, **k: None)  # silent no-op
    removed, remaining = gen.clear_shard_output(str(tmp_path), marker)
    assert remaining == [str(d)], "a no-op delete must still be reported as remaining"

    monkeypatch.setattr(gen.shutil, "rmtree", real_shutil.rmtree)
