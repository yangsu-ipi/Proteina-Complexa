"""Recovering matrices stored beside a structure's other name.

The advisory store copied a harness's structure and left the sidecar behind, so
the matrix sits under a name no cache entry records. The copy was a copy, so
content matches the two names -- which is the only thing that does, since the
advisory name is a digest of the binder sequence and the harness name is a
sequence index, and neither inverts to the other.
"""

import os

import numpy as np

from proteinfoundation.metrics.pae_backfill import backfill, backfill_job_dir
from proteinfoundation.metrics.pae_store import (
    CONFIDENCE_KEPT_SUFFIX,
    has_stored_pae,
    load_pae,
    save_pae,
)


def realistic_pae(target_len=6, binder_len=4):
    """Enough structure to be a real matrix; the compression is pinned in
    test_pae_store, so this only has to be something a reader can come back."""
    n = target_len + binder_len
    idx = np.arange(n)
    same_chain = (idx[:, None] < target_len) == (idx[None, :] < target_len)
    pae = 2.0 + 0.06 * np.abs(idx[:, None] - idx[None, :]) + np.where(same_chain, 0.0, 12.0)
    return np.clip(pae, 0.2, 31.5).astype(np.float32)


def _job_dir(tmp_path, name="job_0_n_1"):
    job = tmp_path / "evaluation_results" / "run" / name
    (job / "AF2").mkdir(parents=True)
    (job / "af2_complex").mkdir(parents=True)
    return job


def _harness_fold(job, stem, body, with_sidecar=True):
    path = job / "AF2" / f"{stem}.pdb"
    path.write_text(body)
    if with_sidecar:
        save_pae(str(path), realistic_pae(target_len=6, binder_len=4), chain_lengths=[6, 4], backend="af2")
        (job / "AF2" / f"{stem}.pdb{CONFIDENCE_KEPT_SUFFIX}").write_text('{"pTM": 0.7}')
    return path


def test_a_matrix_is_matched_to_its_copy_by_content(tmp_path):
    """The names do not encode each other; the bytes do."""
    job = _job_dir(tmp_path)
    body = "ATOM      1  N   ALA A   1\n"
    _harness_fold(job, "run_self_seq_0_model1", body)
    advisory = job / "af2_complex" / "0c444edd5255_model1.pdb"
    advisory.write_text(body)
    assert not has_stored_pae(str(advisory))

    recovered, missing = backfill_job_dir(str(job))

    assert (recovered, missing) == (1, 0)
    assert load_pae(str(advisory)) is not None, "the reader must find the matrix beside the recorded name"
    assert os.path.exists(str(advisory) + CONFIDENCE_KEPT_SUFFIX), "confidences travel with it"


def test_a_structure_with_no_surviving_copy_is_reported_not_invented(tmp_path):
    """A harness directory is rewritten per sequence while the advisory one
    accumulates, so most originals are gone. Those must be counted, never
    matched to a different structure that happens to be nearby."""
    job = _job_dir(tmp_path)
    _harness_fold(job, "run_self_seq_9_model1", "ATOM      1  N   GLY A   1\n")
    orphan = job / "af2_complex" / "deadbeef0001_model1.pdb"
    orphan.write_text("ATOM      1  N   TRP A   1\n")

    recovered, missing = backfill_job_dir(str(job))

    assert (recovered, missing) == (0, 1)
    assert not has_stored_pae(str(orphan)), "a near-miss must never be treated as the same fold"


def test_running_it_twice_is_running_it_once(tmp_path):
    """Idempotent and additive: it writes what is absent and replaces nothing."""
    job = _job_dir(tmp_path)
    body = "ATOM      1  N   ALA A   1\n"
    _harness_fold(job, "run_mpnn_seq_0_model2", body)
    advisory = job / "af2_complex" / "abc123456789_model2.pdb"
    advisory.write_text(body)

    assert backfill_job_dir(str(job))[0] == 1
    first = load_pae(str(advisory))["pae"].copy()
    # Second pass has nothing left to want, so it recovers nothing and changes nothing.
    assert backfill_job_dir(str(job)) == (0, 0)
    assert (load_pae(str(advisory))["pae"] == first).all()


def test_a_dry_run_reports_without_writing(tmp_path):
    job = _job_dir(tmp_path)
    body = "ATOM      1  N   ALA A   1\n"
    _harness_fold(job, "run_self_seq_0_model1", body)
    advisory = job / "af2_complex" / "0c444edd5255_model1.pdb"
    advisory.write_text(body)

    assert backfill_job_dir(str(job), dry_run=True) == (1, 0)
    assert not has_stored_pae(str(advisory)), "a dry run must write nothing"


def test_matching_never_crosses_designs(tmp_path):
    """Two designs can hold identical structures -- the same binder reached twice
    is not a bug. Matching across job directories would carry one design's matrix
    onto another's structure, and the entry pointing at it would be wrong about
    which fold it describes."""
    body = "ATOM      1  N   ALA A   1\n"
    first = _job_dir(tmp_path, "job_0_n_1")
    _harness_fold(first, "run_self_seq_0_model1", body)
    second = _job_dir(tmp_path, "job_1_n_2")
    lonely = second / "af2_complex" / "0c444edd5255_model1.pdb"
    lonely.write_text(body)

    totals = backfill(str(tmp_path / "evaluation_results"))

    assert totals["job_dirs"] == 2
    assert totals["recovered"] == 0, "the donor lives under a different design"
    assert totals["still_missing"] == 1
    assert not has_stored_pae(str(lonely))


def test_the_esmfold2_track_needs_nothing_and_is_left_alone(tmp_path):
    """That folder writes its own structure and its own sidecar, so it was never
    affected. The utility must report it as nothing to do rather than churn."""
    job = tmp_path / "evaluation_results" / "run" / "job_0_n_1"
    (job / "esmfold2_complex").mkdir(parents=True)
    kept = job / "esmfold2_complex" / "abc123456789_seed1.pdb"
    kept.write_text("ATOM      1  N   ALA A   1\n")
    save_pae(str(kept), realistic_pae(target_len=6, binder_len=4), chain_lengths=[6, 4], backend="esmfold2")

    assert backfill_job_dir(str(job)) == (0, 0)
