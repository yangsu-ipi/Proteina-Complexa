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


def test_matching_digest_regenerates_when_skipping_is_disabled(tmp_path):
    write_marker(tmp_path, "a" * 64)
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=False) is False


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
def test_an_older_digest_formula_is_incomparable_not_a_mismatch(tmp_path, version):
    """Aborting a resume because *our* hash changed would be a false alarm on a
    hard failure -- worse than the warning it replaced."""
    write_marker(tmp_path, "a" * 64, version=version)
    assert shard_already_complete(str(tmp_path), 0, "b" * 64, skip_enabled=True) is False


def test_an_unreadable_marker_does_not_stop_the_run(tmp_path):
    path = shard_marker_path(str(tmp_path), 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("{not json")
    assert shard_already_complete(str(tmp_path), 0, "a" * 64, skip_enabled=True) is False
