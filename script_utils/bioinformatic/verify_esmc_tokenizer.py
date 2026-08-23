#!/usr/bin/env python3
"""Check that Biohub's ESMC tokenizer behaves identically across transformers versions.

ESMC's entire coupling to transformers is one line -- ``PreTrainedTokenizerFast``
as a base class in ``esm/tokenization/sequence_tokenizer.py`` -- and that is the
only transformers reference across the 166 modules reachable from
``esm.models.esmc``. Everything downstream of tokenization is pure torch. So the
transformers version can change an ESMC score only by changing tokenization,
which makes comparing token ids a *sufficient* check: no weights, no GPU, and no
ESMC CLI required.

This matters when deciding whether Complexa's environment needs Biohub's
transformers fork. ESMFold2 does require it (``ESMFold2Model`` exists nowhere
else). ESMC does not -- if this check passes under both, that is the evidence.

Usage:

    # in each environment, record what tokenization looks like there
    python verify_esmc_tokenizer.py --emit /tmp/tok_pypi.json
    python verify_esmc_tokenizer.py --emit /tmp/tok_fork.json    # fork's env

    # compare two recordings
    python verify_esmc_tokenizer.py --compare /tmp/tok_pypi.json /tmp/tok_fork.json

    # or check against the golden recording committed beside this script
    python verify_esmc_tokenizer.py

The tokenizer is loaded directly from source when the ``esm`` package is not
importable, so this runs in an environment where ``esm``'s heavier dependencies
(cloudpathlib, msgpack, accelerate, ...) are absent.

Exit codes: 0 match, 1 mismatch, 2 could not run.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import types

GOLDEN = pathlib.Path(__file__).with_name("esmc_tokenizer_golden.json")

# Fixed inputs. Deliberately boring and self-contained: a short peptide, a
# binder-length sequence, and one carrying residues that are their own tokens
# (X, B, U, Z, O) plus a lowercase run, since case handling and rare-residue
# mapping are where a tokenizer change would first show up.
SEQUENCES = [
    "MKTAYIAKQR",
    "GSHMSEQELLRQAKEWLESHPEAKALFEKALRLLEEGKDPLALLALLQALESHPEAKALFEK",
    "MXBUZOACDEFGHIKLMNPQRSTVWYmktayiakqr",
]
PAD_BATCH = ["MKTAYIAKQR", "MKT", "GSHMSEQELLRQAKEW"]


def _load_tokenizer_class(esm_src: str | None):
    """Return Biohub's EsmSequenceTokenizer, from the package or from source."""
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        return EsmSequenceTokenizer, "installed package"
    except Exception:
        pass

    root = esm_src or os.environ.get("ESM_SRC") or os.environ.get("ESMFOLD2_SRC")
    if not root:
        raise RuntimeError(
            "esm is not importable and no source tree was given. Pass --esm-src /path/to/esmfold2 or set ESM_SRC."
        )
    root_path = pathlib.Path(root).expanduser()
    if not (root_path / "esm" / "tokenization" / "sequence_tokenizer.py").exists():
        raise RuntimeError(f"no esm/tokenization/sequence_tokenizer.py under {root_path}")

    # Stub the packages so the submodules import without running esm's own
    # __init__ chain, which reaches dependencies this check does not need.
    sys.path.insert(0, str(root_path))
    for name in ("esm", "esm.tokenization"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(root_path / name.replace(".", "/"))]
            sys.modules[name] = stub

    for mod in (
        "esm.utils.constants.esm3",
        "esm.tokenization.tokenizer_base",
        "esm.tokenization.sequence_tokenizer",
    ):
        spec = importlib.util.spec_from_file_location(mod, root_path / (mod.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod] = module
        spec.loader.exec_module(module)

    return sys.modules["esm.tokenization.sequence_tokenizer"].EsmSequenceTokenizer, str(root_path)


def _source_commit(where: str) -> str | None:
    """Best-effort git SHA of the esm source, for provenance in the recording."""
    path = pathlib.Path(where)
    if not path.is_dir():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _ids(encoded, tokenizer):
    """Pull token ids out of an encoding, whatever the input name is.

    Order matters: this tokenizer sets model_input_names to
    ["sequence_tokens", "attention_mask"], so iterating that list blindly can
    return the attention mask -- all ones for an unpadded sequence, which would
    compare equal between any two environments and make this check vacuous.
    Named token-id keys come first, and mask-like keys are never accepted.
    """
    for key in ("sequence_tokens", "input_ids"):
        if key in encoded:
            return encoded[key]
    for key in tokenizer.model_input_names:
        if "mask" not in key and key in encoded:
            return encoded[key]
    raise RuntimeError(f"no token ids in encoding keys {list(encoded)}")


def fingerprint(esm_src: str | None) -> dict:
    """Everything about tokenization that could change an ESMC score."""
    import transformers

    cls, where = _load_tokenizer_class(esm_src)
    tokenizer = cls()

    vocab = tokenizer.get_vocab()
    specials = {
        name: getattr(tokenizer, f"{name}_token_id", None)
        for name in ("mask", "pad", "cls", "eos", "unk", "bos", "sep")
    }

    per_sequence = {}
    for seq in SEQUENCES:
        ids = _ids(tokenizer(seq), tokenizer)
        per_sequence[seq] = {
            "ids": list(ids),
            "n_tokens": len(ids),
            # The offset assumption the scoring code relies on: residue i at
            # token prefix_len + i.
            "residue_tokens": list(ids[1 : 1 + len(seq)]),
        }

    padded = tokenizer(PAD_BATCH, padding=True)
    batch_ids = _ids(padded, tokenizer)

    return {
        # Informational only -- deliberately excluded from the comparison, since
        # differing across environments is the whole point.
        "env": {
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
            # Home-relative so a committed recording carries no user path.
            "esm_source": where.replace(str(pathlib.Path.home()), "~"),
            "esm_commit": _source_commit(where),
        },
        # This block is what must match.
        "tokenization": {
            "class": cls.__name__,
            "model_input_names": list(tokenizer.model_input_names),
            "vocab_size": len(vocab),
            "vocab_sha256": hashlib.sha256(
                json.dumps(sorted(vocab.items()), separators=(",", ":")).encode()
            ).hexdigest(),
            "vocab": dict(sorted(vocab.items(), key=lambda kv: kv[1])),
            "special_token_ids": specials,
            "chain_break_token": getattr(tokenizer, "cb_token", None),
            "sequences": per_sequence,
            "padded_batch": {
                "inputs": PAD_BATCH,
                "ids": [list(row) for row in batch_ids],
                "attention_mask": [list(row) for row in padded["attention_mask"]]
                if "attention_mask" in padded
                else None,
            },
        },
    }


def compare(a: dict, b: dict, label_a: str, label_b: str) -> bool:
    """Report differences in the tokenization block; True when identical."""
    left, right = a["tokenization"], b["tokenization"]
    print(f"  {label_a}: transformers {a['env']['transformers']}, python {a['env']['python']}")
    print(f"  {label_b}: transformers {b['env']['transformers']}, python {b['env']['python']}")
    if a["env"].get("esm_commit") and a["env"]["esm_commit"] != b["env"].get("esm_commit"):
        print(
            "  NOTE: different esm source commits "
            f"({str(a['env']['esm_commit'])[:12]} vs {str(b['env']['esm_commit'])[:12]}) "
            "-- a mismatch below may be the source tree, not transformers"
        )

    if left == right:
        print("\n  tokenization identical")
        return True

    print("\n  MISMATCH:")
    for key in sorted(set(left) | set(right)):
        lv, rv = left.get(key, "<absent>"), right.get(key, "<absent>")
        if lv == rv:
            continue
        if key == "sequences" and isinstance(lv, dict) and isinstance(rv, dict):
            for seq in sorted(set(lv) | set(rv)):
                if lv.get(seq) != rv.get(seq):
                    print(f"    sequence {seq[:24]}...")
                    print(f"      {label_a}: {lv.get(seq, {}).get('ids')}")
                    print(f"      {label_b}: {rv.get(seq, {}).get('ids')}")
        elif key == "vocab":
            only_a = {k: v for k, v in lv.items() if rv.get(k) != v}
            only_b = {k: v for k, v in rv.items() if lv.get(k) != v}
            print(f"    vocab differs: {label_a} {only_a} vs {label_b} {only_b}")
        else:
            print(f"    {key}:")
            print(f"      {label_a}: {lv}")
            print(f"      {label_b}: {rv}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esm-src", help="path to the esmfold2 source tree (if esm is not installed)")
    ap.add_argument("--emit", metavar="PATH", help="write this environment's recording and exit")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two recordings")
    ap.add_argument("--update-golden", action="store_true", help="overwrite the committed golden")
    args = ap.parse_args()

    if args.compare:
        a, b = (json.loads(pathlib.Path(p).read_text()) for p in args.compare)
        print("Comparing two recordings")
        return 0 if compare(a, b, *args.compare) else 1

    try:
        current = fingerprint(args.esm_src)
    except Exception as exc:
        print(f"could not run: {exc}", file=sys.stderr)
        return 2

    if args.emit:
        pathlib.Path(args.emit).write_text(json.dumps(current, indent=2) + "\n")
        print(f"recording written to {args.emit}")
        print(f"  transformers {current['env']['transformers']}, esm from {current['env']['esm_source']}")
        return 0

    if args.update_golden:
        GOLDEN.write_text(json.dumps(current, indent=2) + "\n")
        print(f"golden updated: {GOLDEN}")
        return 0

    if not GOLDEN.exists():
        print(f"no golden at {GOLDEN}; create one with --update-golden", file=sys.stderr)
        return 2

    print(f"Checking this environment against {GOLDEN.name}")
    ok = compare(json.loads(GOLDEN.read_text()), current, "golden", "this env")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
