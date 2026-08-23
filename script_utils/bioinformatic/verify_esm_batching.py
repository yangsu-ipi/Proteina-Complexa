#!/usr/bin/env python3
"""Prove the batched ESM scoring path matches the per-residue reference path.

Batched masked pseudo-perplexity is arithmetically identical to the one-forward-
per-residue loop: every masked copy is independent, and all copies of one
sequence have the same length so the batch needs no padding. This script checks
that claim instead of assuming it, because a wrong mask position or gather index
would still produce a plausible number.

Three modes:

  --synthetic (default)
      Tiny randomly-initialised ESM model on CPU. Needs no weights and no
      network, so it runs anywhere -- and it exercises the part that can
      actually be wrong: mask placement, chunking, and the log-prob gather.

  --model facebook/esm2_t33_650M_UR50D
      Real ESM2 weights. Runs the unbatched reference and the batched path over
      the same sequences and reports max deviation plus the speedup.

  --model esmc_600m --backend esmc
      ESMC has no unbatched reference (the reference is ESM2-specific), so this
      checks budget-invariance instead: two different batch budgets must agree,
      which catches chunk-boundary errors.

Exit code is non-zero if any check exceeds tolerance.
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, "src")

from proteinfoundation.evaluation.esm_eval import (
    EsmBackend,
    compute_pseudo_perplexity,
    compute_pseudo_perplexity_batched,
    get_esm_backend,
    resolve_backend,
)

# ESM-2's real vocabulary order, so the synthetic check mirrors production
# indexing (cls=0, pad=1, eos=2, unk=3, ..., mask=32).
_ESM2_VOCAB_SPEC = "<cls> <pad> <eos> <unk> L A G V S E R T I D P K Q N F Y M H W C X B U Z O . - <null_1> <mask>"
ESM2_VOCAB = _ESM2_VOCAB_SPEC.split()

DEFAULT_SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "GSHMSEQELLRQAKEWLESHPEAKALFEKALRLLEEGKDPLALLALLQAL",
    "MAGWNAYIDNLMADGTCQDAAIVGYKDSPSVWAAVPGKTFVNITPAEVGVLVGKDRSSFYVNGLTLGGQKCSVIRDSLLQDGEFSMDLRTKSTGGAPTFNVTVTKTDKTLVLLMGKEGVHGGLINKKCYEMASHLRRSQY",
]


class _Encoding(dict):
    """Minimal stand-in for a HuggingFace BatchEncoding."""

    def to(self, _device):
        return self


class SyntheticTokenizer:
    """Character-level tokenizer with ESM-2's vocabulary layout."""

    def __init__(self):
        self.vocab = {tok: i for i, tok in enumerate(ESM2_VOCAB)}
        self.mask_token_id = self.vocab["<mask>"]
        self.pad_token_id = self.vocab["<pad>"]
        self.cls_token_id = self.vocab["<cls>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.inverse = {i: tok for tok, i in self.vocab.items()}

    def _ids(self, sequence: str) -> list[int]:
        body = [self.vocab.get(c.upper(), self.unk_token_id) for c in sequence]
        return [self.cls_token_id] + body + [self.eos_token_id]

    def __call__(self, sequences, return_tensors=None, padding=False):
        if isinstance(sequences, str):
            sequences = [sequences]
        rows = [self._ids(s) for s in sequences]
        width = max(len(r) for r in rows)
        ids, mask = [], []
        for row in rows:
            pad = width - len(row)
            ids.append(row + [self.pad_token_id] * pad)
            mask.append([1] * len(row) + [0] * pad)
        return _Encoding(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(mask, dtype=torch.long),
        )

    def convert_ids_to_tokens(self, ids):
        return [self.inverse[int(i)] for i in ids]


def build_synthetic_backend(seed: int = 0) -> EsmBackend:
    """A tiny real ESM architecture with random weights, on CPU."""
    from transformers import EsmConfig, EsmForMaskedLM

    torch.manual_seed(seed)
    config = EsmConfig(
        vocab_size=len(ESM2_VOCAB),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=1026,
        pad_token_id=1,
        position_embedding_type="rotary",
    )
    model = EsmForMaskedLM(config)
    model.eval()
    tokenizer = SyntheticTokenizer()
    return EsmBackend(
        kind="esm2",
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        mask_token_id=tokenizer.mask_token_id,
    )


def compare_reference(backend: EsmBackend, sequences: list[str], budgets: list[int], tol: float) -> bool:
    """Reference (unbatched) vs batched, over several batch budgets."""
    ok = True
    for seq in sequences:
        t0 = time.perf_counter()
        ref_ppl, ref_ll = compute_pseudo_perplexity(backend.model, backend.tokenizer, seq, backend.device)
        ref_secs = time.perf_counter() - t0

        for budget in budgets:
            t0 = time.perf_counter()
            new_ppl, new_ll = compute_pseudo_perplexity_batched(backend, seq, max_batch_tokens=budget)
            new_secs = time.perf_counter() - t0

            d_ll = abs(new_ll - ref_ll)
            d_ppl = abs(new_ppl - ref_ppl) / max(abs(ref_ppl), 1e-12)
            good = d_ll < tol and d_ppl < tol
            ok &= good
            print(
                f"  L={len(seq):>4}  budget={budget:>6}  "
                f"ll {ref_ll:+.8f} -> {new_ll:+.8f} (d={d_ll:.2e})  "
                f"ppl {ref_ppl:.6f} -> {new_ppl:.6f} (rel={d_ppl:.2e})  "
                f"{ref_secs:.2f}s -> {new_secs:.2f}s ({ref_secs / max(new_secs, 1e-9):.1f}x)  "
                f"{'OK' if good else 'FAIL'}"
            )
    return ok


def compare_budgets(backend: EsmBackend, sequences: list[str], budgets: list[int], tol: float) -> bool:
    """Batched path against itself at different budgets (backend-agnostic)."""
    ok = True
    for seq in sequences:
        results = []
        for budget in budgets:
            t0 = time.perf_counter()
            ppl, ll = compute_pseudo_perplexity_batched(backend, seq, max_batch_tokens=budget)
            results.append((budget, ppl, ll, time.perf_counter() - t0))
        base_ll = results[0][2]
        for budget, ppl, ll, secs in results:
            d_ll = abs(ll - base_ll)
            good = d_ll < tol
            ok &= good
            print(
                f"  L={len(seq):>4}  budget={budget:>6}  ll {ll:+.8f} (d vs first={d_ll:.2e})  "
                f"ppl {ppl:.6f}  {secs:.2f}s  {'OK' if good else 'FAIL'}"
            )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Real model to load (HF name for esm2, registry name for esmc)")
    ap.add_argument("--backend", default="auto", help="auto, esm2, or esmc")
    ap.add_argument("--seq", action="append", default=None, help="Sequence to score (repeatable)")
    ap.add_argument("--budgets", default="64,1024,16384", help="Comma-separated token budgets to test")
    ap.add_argument("--tol", type=float, default=1e-4, help="Max allowed deviation")
    ap.add_argument("--allow-download", action="store_true", help="Permit network fetch of weights")
    args = ap.parse_args()

    sequences = args.seq or DEFAULT_SEQUENCES
    budgets = [int(b) for b in args.budgets.split(",")]

    if args.model is None:
        print("Mode: synthetic (tiny random-weight ESM on CPU, no weights required)")
        backend = build_synthetic_backend()
        print(f"Reference vs batched over {len(sequences)} sequences, budgets={budgets}, tol={args.tol}")
        ok = compare_reference(backend, sequences, budgets, args.tol)
    else:
        kind = resolve_backend(args.model, args.backend)
        print(f"Mode: real weights ({kind}:{args.model})")
        backend = get_esm_backend(args.model, backend=args.backend, force_offline=not args.allow_download)
        if kind == "esm2":
            print(f"Reference vs batched over {len(sequences)} sequences, budgets={budgets}, tol={args.tol}")
            ok = compare_reference(backend, sequences, budgets, args.tol)
        else:
            print(f"Budget-invariance over {len(sequences)} sequences, budgets={budgets}, tol={args.tol}")
            print("(no unbatched reference exists for esmc; this checks chunk boundaries)")
            ok = compare_budgets(backend, sequences, budgets, args.tol)

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
