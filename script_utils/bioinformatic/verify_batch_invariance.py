#!/usr/bin/env python
"""Does a design's denoising output depend on what it was batched with?

Generation batches designs of different lengths together (``nres: 40-70`` in the
CBLN1 campaign), so every batch is padded to its longest member. If masking is
exact, a design's network output is the same whatever it shares a batch with, and
batch size is pure parallelism -- lowering it on OOM would then draw from the same
distribution and only reproducibility would be at stake. If masking leaks, batch
composition is a systematic input: the same design batched with a 70-mer differs
from itself batched with 45-mers, and batch size becomes a scientific variable.

That question is asked here directly, with the sampler's randomness removed rather
than controlled for. A real batch is captured mid-sampling and then re-run through
``call_nn`` in different compositions. No noise is drawn, no trajectory advances,
so any difference is the network's, not the RNG's -- which is what makes a naive
"generate it twice and compare" test useless: the draws differ with batch size
whether masking is exact or not.

Three comparisons, each isolating one thing:

  same        the identical batch twice. Establishes the numerical noise floor --
              nonzero under TF32 and autotuned kernels, and every other number is
              meaningless without it.
  alone       the design by itself, padded to the same width. Differs from the
              full batch only in whether other designs are present, so this
              isolates leakage ACROSS the batch dimension.
  padded      the design by itself at its own length vs at the batch's padded
              width. Isolates sensitivity to the padding itself, independent of
              neighbours. The width is the real batch's, not a synthetic one:
              padding further would mean inventing fill values per key, and a
              wrong one (collate_fn uses -1 for chains) would report a leak that
              is only my bad padding.

A leak in either direction scales with what caused it; float noise does not. That
is the discriminator, and it is why `same` is measured rather than assumed to be
zero.

Exit codes: 0 invariant, 1 composition-dependent, 2 could not test.
"""

import argparse
import sys

import torch


class _CapturedError(Exception):
    """Raised inside the call_nn hook to stop sampling at the first batch."""


def slice_batch(obj, idx, n_res=None, bsz=None):
    """One sample from a collated batch, optionally trimmed to n_res residues.

    Recursive because the batch nests -- ``x_recycle`` is a dict of tensors. The
    residue axis is identified by size rather than by name: any axis after the
    batch that matches the batch's padded width is a residue axis, which covers
    [b, n, ...] and pair features [b, n, n, ...] without a per-key table.

    A list whose length is the batch size is per-sample metadata -- generation
    indexes ``metadata_tag`` and ``sample_type`` that way -- so one element is
    selected. Mapping over it instead left eight names beside one design's
    tensors, which a model that indexes by batch position would read as a
    different sample. Any other list is a structure to recurse into.
    """
    if isinstance(obj, dict):
        return {k: slice_batch(v, idx, n_res, bsz) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if bsz is not None and len(obj) == bsz:
            return type(obj)([obj[idx]])
        return type(obj)(slice_batch(v, idx, n_res, bsz) for v in obj)
    if not torch.is_tensor(obj) or obj.dim() == 0:
        return obj
    out = obj[idx : idx + 1]
    if n_res is not None:
        width = obj.shape[1] if obj.dim() > 1 else None
        for axis in range(1, out.dim()):
            if out.shape[axis] == width:
                out = out.narrow(axis, 0, n_res)
    return out


def flatten_out(nn_out):
    """The output tensors, in a stable order, as one list."""
    if torch.is_tensor(nn_out):
        return [nn_out]
    if isinstance(nn_out, dict):
        return [t for k in sorted(nn_out) for t in flatten_out(nn_out[k])]
    if isinstance(nn_out, (list, tuple)):
        return [t for v in nn_out for t in flatten_out(v)]
    return []


def max_abs_diff(a, b, n_res):
    """Largest elementwise difference over the valid (unpadded) region."""
    worst = 0.0
    for ta, tb in zip(flatten_out(a), flatten_out(b), strict=True):
        if ta.shape != tb.shape:
            # Different padded widths: compare the region both cover.
            trim = [slice(None)] * ta.dim()
            for axis in range(1, ta.dim()):
                trim[axis] = slice(0, min(ta.shape[axis], tb.shape[axis], n_res))
            ta, tb = ta[tuple(trim)], tb[tuple(trim)]
        worst = max(worst, (ta.float() - tb.float()).abs().max().item())
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-path", required=True, help="directory holding the pipeline yaml")
    ap.add_argument("--config-name", default="pipeline")
    ap.add_argument(
        "--tolerance", type=float, default=10.0, help="multiple of the noise floor that still counts as noise"
    )
    ap.add_argument("overrides", nargs="*", help="hydra overrides, e.g. ++job_id=0")
    args = ap.parse_args()

    import hydra
    from omegaconf import open_dict

    from proteinfoundation.generate import load_ckpt_n_configure_inference, split_by_job

    with hydra.initialize_config_dir(config_dir=args.config_path, version_base=None):
        cfg = hydra.compose(config_name=args.config_name, overrides=list(args.overrides))

    if not torch.cuda.is_available():
        print("no CUDA device; this test is about kernels and padding, so it needs the real one")
        return 2

    model = load_ckpt_n_configure_inference(cfg).cuda().eval()
    cfg_gen = split_by_job(cfg.generation, cfg.get("job_id", 0), cfg.get("gen_njobs", 1))
    dataloader = hydra.utils.instantiate(cfg_gen.dataloader)

    # Stops predict_step from building the AF2 reward: it is JAX, it preallocates
    # most of the card, and nothing here scores anything.
    model.reward_model = object()
    with open_dict(model.inf_cfg):
        model.inf_cfg.reward_model = None

    captured = {}
    original_call_nn = type(model).call_nn

    def capture(self, batch, n_recycle=0):
        captured["batch"] = batch
        raise _CapturedError

    type(model).call_nn = capture
    try:
        batch = next(iter(dataloader))
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        try:
            model.predict_step(batch, 0)
        except _CapturedError:
            pass
    finally:
        type(model).call_nn = original_call_nn

    full = captured.get("batch")
    if full is None:
        print("call_nn was never reached; nothing to compare")
        return 2

    mask = full.get("mask")
    if mask is None:
        print("the captured batch has no `mask`; cannot tell padding from data")
        return 2
    lengths = mask.sum(dim=1).long().tolist()
    b, width = mask.shape[0], mask.shape[1]
    if b < 2:
        print(f"batch of {b}: composition cannot vary. Re-run with a larger generation.dataloader.batch_size")
        return 2
    print(f"captured a batch of {b}, padded width {width}, lengths {lengths}")

    with torch.no_grad():
        out_full_a = model.call_nn(full)
        out_full_b = model.call_nn(full)

        floor = 0.0
        for i in range(b):
            floor = max(
                floor, max_abs_diff(slice_batch(out_full_a, i, bsz=b), slice_batch(out_full_b, i, bsz=b), lengths[i])
            )
        print(f"\nnoise floor (same batch, twice): {floor:.3e}")
        limit = max(floor * args.tolerance, 1e-6)

        print(f"\n{'design':>6}  {'len':>4}  {'alone':>12}  {'padded':>12}   verdict")
        failures = []
        for i in range(b):
            alone_in = slice_batch(full, i, bsz=b)
            d_alone = max_abs_diff(slice_batch(out_full_a, i, bsz=b), model.call_nn(alone_in), lengths[i])

            trimmed = slice_batch(full, i, n_res=lengths[i], bsz=b)
            wide = slice_batch(full, i, n_res=None, bsz=b)
            d_pad = max_abs_diff(model.call_nn(trimmed), model.call_nn(wide), lengths[i])

            ok = d_alone <= limit and d_pad <= limit
            failures.append(None if ok else (i, lengths[i], d_alone, d_pad))
            print(f"{i:>6}  {lengths[i]:>4}  {d_alone:>12.3e}  {d_pad:>12.3e}   {'ok' if ok else 'DIFFERS'}")

    bad = [f for f in failures if f]
    print()
    if not bad:
        print(f"INVARIANT: every design's output matched within {limit:.3e} regardless of composition.")
        print("Batch size is parallelism here, so lowering it on OOM samples the same distribution.")
        return 0

    print(f"COMPOSITION-DEPENDENT: {len(bad)} of {b} designs changed by more than {limit:.3e}.")
    print("`alone` above the floor means the batch dimension leaks; `padded` means the padding does.")
    print("Either way batch size is a scientific variable, not just a resource knob, and an")
    print("OOM retry at a smaller batch would not be sampling the same distribution.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
