"""
ESM sequence scoring: masked pseudo-perplexity and log-likelihood.

Lower pseudo-perplexity indicates a more "natural" sequence according to the
language model.

Two backends are supported, selected by ``backend`` (or auto-detected from the
model name):

* ``esm2`` -- HuggingFace ``AutoModelForMaskedLM`` (default,
  ``facebook/esm2_t33_650M_UR50D``).
* ``esmc`` -- a transformers-format ESMC repo, loaded by the same
  ``AutoModelForMaskedLM``. This is how ESMFold2 itself loads ESMC.
* ``esmc_pkg`` -- the ``esm`` package's own ``ESMC`` class, which currently
  cannot load (its builders leave parameters on the meta device).

Both go through one batched scoring core: pseudo-perplexity needs one masked
forward *per residue*, and those L forwards are independent, so they are run as
a batch instead of a Python loop. The metric definition is unchanged --
:func:`compute_pseudo_perplexity_reference` is the unbatched reference for any
backend, and :func:`compute_pseudo_perplexity` the older ESM2-only one that
bypasses the backend adapter; ``script_utils/bioinformatic/verify_esm_batching.py``
uses them to prove equivalence.

Models are cached globally, keyed on (backend, model name, device), so they load
once per session. ESM2 honours ``ESM_DIR``/``CACHE_DIR``; ESMC resolves weights
through ``HF_HOME``/``HF_HUB_CACHE`` instead (it calls ``snapshot_download``
internally and accepts no ``cache_dir``). Set ``HF_HUB_OFFLINE=1`` to force
fully offline mode; that requires an already-warm cache.
"""

import hashlib
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch
from loguru import logger

# =============================================================================
# Safe Imports
# =============================================================================

ESM_AVAILABLE = False
try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    ESM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ESM/transformers import failed: {e}. ESM metrics will return NaN.")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"

# Transformers-format ESMC repo, loaded through AutoModelForMaskedLM. This is the
# ESMC that works: ESMFold2 itself loads ESMC this way (its load_esmc() imports
# transformers.models.esmc), and the class is registered in the Auto mappings.
DEFAULT_ESMC_MODEL = "biohub/ESMC-6B"

# Registry name understood by the installed ``esm`` package, for the esmc_pkg
# backend. A wrong name fails loudly and the error lists the valid keys from that
# package's LOCAL_MODEL_REGISTRY.
DEFAULT_ESMC_PKG_MODEL = "esmc_600m"

BACKEND_ESM2 = "esm2"
BACKEND_ESMC = "esmc"
BACKEND_ESMC_PKG = "esmc_pkg"

# Backends served by AutoTokenizer + AutoModelForMaskedLM. They share one code
# path, so anything true of ESM2 loading is true of ESMC loading -- including the
# raw-HuggingFace reference that gives the batching an independent cross-check.
HF_BACKENDS = (BACKEND_ESM2, BACKEND_ESMC)

# Budget for one batched forward, in tokens (rows x padded length). A 100-residue
# sequence scores in ceil(100 / (16384 // 102)) = 1 forward instead of 100.
# Lower it if a long-sequence batch runs out of memory; the code also halves on
# OOM by itself.
DEFAULT_ESM_BATCH_TOKENS = 16384

# Per-design cache of sequence scores, written beside binder_eval_cache.json.
# Deliberately a separate file: adding the sequence model to
# binder_eval_fingerprint would change every fingerprint and invalidate the
# refolding caches already on disk, forcing a full refold to gain ESM caching.
ESM_CACHE_FILENAME = "esm_eval_cache.json"

ESM_METRIC_COLS = [
    "esm_pseudo_perplexity",
    "esm_log_likelihood",
]

# =============================================================================
# Global Model Cache
# =============================================================================

# Keyed on (backend, model_name, device) so an ESM2 entry cannot be served for
# an ESMC request. ESMC has no model cache of its own -- its ``from_pretrained``
# rebuilds the model on every call -- so this cache is what makes it usable
# from a per-design evaluation loop.
_ESM_BACKEND_CACHE: dict[tuple[str, str, str], "EsmBackend"] = {}

# Hold one model at a time, as the previous single-slot cache did: these are
# ~2.6 GB on the GPU, and a sweep over esm_model would otherwise keep every
# variant resident. Loading a second model evicts the first, and says so, so
# thrashing is visible in the log rather than silent.
_ESM_CACHE_MAXSIZE = 1


# =============================================================================
# Backend Abstraction
# =============================================================================


class EsmBackend:
    """Uniform masked-LM surface over ESM2 and ESMC.

    The scoring core needs only four things from a model: how to encode a batch
    of sequences, which token id means "mask", how many special tokens precede
    the first residue, and how to get logits. Everything backend-specific lives
    here.

    Attributes:
        kind: ``esm2`` or ``esmc``.
        model: The loaded model, in eval mode on ``device``.
        tokenizer: The matching tokenizer.
        device: Torch device string.
        mask_token_id: Token id used to mask a position.
        prefix_len: Special tokens before residue 0 (BOS).
        suffix_len: Special tokens after the last residue (EOS).
    """

    def __init__(
        self,
        kind: str,
        model,
        tokenizer,
        device: str,
        mask_token_id: int,
        prefix_len: int = 1,
        suffix_len: int = 1,
    ):
        self.kind = kind
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.mask_token_id = mask_token_id
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len

    def encode(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Tokenize sequences into a padded ``[B, T]`` batch.

        Returns:
            ``(input_ids, attention_mask)``. ``attention_mask`` is None for
            backends that derive padding themselves.
        """
        if self.kind == BACKEND_ESMC_PKG:
            # The esm package's ESMC._tokenize takes a list and pads internally.
            # Passing sequence_id=None to forward lets it derive the pad mask,
            # which also avoids its flash-attention assert on the mask dtype.
            ids = self.model._tokenize(sequences)
            return ids.to(self.device), None

        encoded = self.tokenizer(sequences, return_tensors="pt", padding=True)
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        return ids, mask

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Run a forward pass and return ``[B, T, V]`` logits."""
        if self.kind == BACKEND_ESMC_PKG:
            return self.model.forward(sequence_tokens=input_ids).sequence_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def resolve_backend(model_name: str, backend: str = "auto") -> str:
    """Resolve the backend for a model name.

    Three backends:

    * ``esm2`` -- HuggingFace ``AutoModelForMaskedLM``.
    * ``esmc`` -- the same loader, for a transformers-format ESMC repo. Biohub's
      transformers fork registers ``ESMCForMaskedLM`` in the Auto mappings, and
      its tokenizer uses standard ``input_ids`` naming, so ESMC needs no special
      handling here.
    * ``esmc_pkg`` -- the ``esm`` package's own ``ESMC`` class. Kept for
      completeness and currently unusable: that package's builders construct
      under ``init_empty_weights()`` and then load via huggingface_hub's
      ``load_torch_model``, which never passes ``assign=True``, so parameters
      stay on the meta device and moving the model raises. ESMFold2 does not use
      this implementation either.

    ``auto`` sends any name containing "esmc" to the ``esmc`` backend and
    everything else to ``esm2``. Note that resolves ESMC to the *working* route:
    reaching the package implementation requires asking for ``esmc_pkg``
    explicitly.
    """
    backend = (backend or "auto").lower()
    if backend in (BACKEND_ESM2, BACKEND_ESMC, BACKEND_ESMC_PKG):
        return backend
    if backend != "auto":
        raise ValueError(
            f"Unknown ESM backend '{backend}'. Expected one of: auto, "
            f"{BACKEND_ESM2}, {BACKEND_ESMC}, {BACKEND_ESMC_PKG}"
        )
    return BACKEND_ESMC if "esmc" in model_name.lower() else BACKEND_ESM2


# =============================================================================
# Core Computation
# =============================================================================


def compute_pseudo_perplexity(
    model,
    tokenizer,
    sequence: str,
    device: str = "cuda",
) -> tuple[float, float]:
    """Compute pseudo-perplexity for a single sequence, one forward per residue.

    Unbatched reference implementation, kept so the batched path can be checked
    against it. Production callers go through
    :func:`compute_esm_ppl_for_sequences`, which batches. ESM2 only -- it takes
    a raw HuggingFace model rather than an :class:`EsmBackend`.
    """
    if not sequence or len(sequence) == 0:
        return np.nan, np.nan

    try:
        inputs = tokenizer(sequence, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_length = input_ids.size(1) - 2  # Exclude BOS and EOS tokens

        log_probs = []
        for i in range(1, seq_length + 1):
            masked_input = input_ids.clone()
            masked_input[0, i] = tokenizer.mask_token_id

            with torch.no_grad():
                outputs = model(masked_input)
                logits = outputs.logits
                log_prob = torch.log_softmax(logits[0, i], dim=-1)
                true_token = input_ids[0, i]
                log_probs.append(log_prob[true_token].item())

        avg_log_likelihood = sum(log_probs) / len(log_probs)
        pseudo_ppl = np.exp(-avg_log_likelihood)

        return pseudo_ppl, avg_log_likelihood

    except Exception as e:
        logger.error(f"ESM computation failed: {e}")
        return np.nan, np.nan


def compute_pseudo_perplexity_reference(
    backend: EsmBackend,
    sequence: str,
) -> tuple[float, float]:
    """Unbatched pseudo-perplexity for any backend: the equivalence reference.

    One masked forward per residue at batch size one, no chunking -- the metric
    written the slow, obvious way. :func:`compute_pseudo_perplexity_batched`
    must agree with this for every backend, which is what
    ``script_utils/bioinformatic/verify_esm_batching.py`` checks.

    What an agreement here does and does not prove: this drives the same
    :class:`EsmBackend` ``encode``/``logits`` as the batched path, so it
    validates the batching itself -- mask placement, chunk boundaries, the
    log-prob gather -- but not the adapter beneath it. A wrong ``encode`` would
    be invisible to the comparison because both sides share it. For ESM2,
    :func:`compute_pseudo_perplexity` drives a raw HuggingFace model and
    tokenizer instead, bypassing the adapter, so running both there covers the
    adapter too. ESMC has no such independent path, so its guarantee stops at
    the batching.
    """
    if not sequence:
        return np.nan, np.nan

    try:
        input_ids, attention_mask = backend.encode([sequence])
        n_tokens = input_ids.size(1)
        n_residues = n_tokens - backend.prefix_len - backend.suffix_len
        if n_residues != len(sequence):
            raise RuntimeError(
                f"ESM tokenizer produced {n_tokens} tokens for a {len(sequence)}-residue sequence, "
                f"implying {n_residues} residues with prefix_len={backend.prefix_len} / "
                f"suffix_len={backend.suffix_len}"
            )
        _verify_residue_alignment(backend, sequence, input_ids[0])

        log_probs = []
        for offset in range(n_residues):
            position = backend.prefix_len + offset
            masked_ids = input_ids.clone()
            masked_ids[0, position] = backend.mask_token_id
            with torch.no_grad():
                logits = backend.logits(masked_ids, attention_mask)
                # float32 for the softmax, matching the batched path, so any
                # difference between them is a real one and not a dtype artefact.
                log_prob = torch.log_softmax(logits[0, position].float(), dim=-1)
                log_probs.append(log_prob[input_ids[0, position]].item())

        avg_log_likelihood = sum(log_probs) / len(log_probs)
        return float(np.exp(-avg_log_likelihood)), float(avg_log_likelihood)

    except Exception as e:
        logger.error(f"ESM computation failed: {e}")
        return np.nan, np.nan


def _verify_residue_alignment(backend: EsmBackend, sequence: str, input_ids: torch.Tensor) -> None:
    """Check that residue i sits at token ``prefix_len + i``.

    The offset is an assumption about the tokenizer's special tokens. Rather
    than trust it silently -- a wrong offset would score shifted positions and
    still return a plausible number -- confirm it by converting the residue
    token ids back to characters. Skipped when the tokenizer cannot do that.
    """
    convert = getattr(backend.tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return
    start = backend.prefix_len
    stop = start + len(sequence)
    try:
        tokens = convert(input_ids[start:stop].tolist())
    except Exception:  # tokenizer does not support round-tripping; skip the check
        return
    if len(tokens) != len(sequence):
        raise RuntimeError(
            f"ESM tokenizer round-trip returned {len(tokens)} tokens for a {len(sequence)}-residue sequence"
        )
    for i, (residue, token) in enumerate(zip(sequence, tokens, strict=True)):
        # Multi-character tokens are unknown/special markers (e.g. <unk>), which
        # legitimately stand in for a residue; only a concrete mismatch is a bug.
        if len(str(token)) == 1 and str(token).upper() != residue.upper():
            raise RuntimeError(
                f"ESM residue alignment check failed at position {i}: "
                f"expected '{residue}', tokenizer reports '{token}'. "
                f"The {backend.kind} tokenizer's special-token layout is not "
                f"prefix_len={backend.prefix_len}."
            )


def _masked_log_probs(
    backend: EsmBackend,
    sequence: str,
    max_batch_tokens: int = DEFAULT_ESM_BATCH_TOKENS,
) -> list[float]:
    """Log-probability of each residue under masking, computed in batches.

    Builds the L single-position-masked copies of one sequence and runs them as
    batches rather than one at a time. Every row has the same length, so the
    batch needs no padding and the result is arithmetically identical to the
    per-residue loop.

    Returns:
        One log-probability per residue, in sequence order.
    """
    input_ids, attention_mask = backend.encode([sequence])
    n_tokens = input_ids.size(1)
    n_residues = n_tokens - backend.prefix_len - backend.suffix_len

    if n_residues != len(sequence):
        raise RuntimeError(
            f"ESM tokenizer produced {n_tokens} tokens for a {len(sequence)}-residue sequence, "
            f"implying {n_residues} residues with prefix_len={backend.prefix_len} / "
            f"suffix_len={backend.suffix_len}"
        )
    _verify_residue_alignment(backend, sequence, input_ids[0])

    positions = torch.arange(backend.prefix_len, backend.prefix_len + n_residues, device=input_ids.device)
    true_ids = input_ids[0, positions]

    batch_size = max(1, min(int(max_batch_tokens) // max(1, n_tokens), n_residues))
    log_probs: list[float] = []
    start = 0
    while start < n_residues:
        chunk = positions[start : start + batch_size]
        try:
            log_probs.extend(
                _score_chunk(backend, input_ids, attention_mask, chunk, true_ids[start : start + len(chunk)])
            )
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            logger.warning(f"ESM batch OOM; retrying at batch_size={batch_size}")
            continue
        start += len(chunk)

    logger.debug(
        f"ESM scored {n_residues} residues in {int(np.ceil(n_residues / batch_size))} forward(s) "
        f"at batch_size={batch_size} ({backend.kind})"
    )
    return log_probs


def _score_chunk(
    backend: EsmBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    chunk_positions: torch.Tensor,
    chunk_true_ids: torch.Tensor,
) -> list[float]:
    """Score one batch of masked copies; returns their log-probabilities."""
    n = chunk_positions.numel()
    rows = torch.arange(n, device=input_ids.device)

    batch_ids = input_ids.repeat(n, 1)
    batch_ids[rows, chunk_positions] = backend.mask_token_id
    batch_mask = attention_mask.repeat(n, 1) if attention_mask is not None else None

    with torch.no_grad():
        logits = backend.logits(batch_ids, batch_mask)
        # ESMC runs in bfloat16; take the softmax in float32 so the reduction is
        # not the limiting factor on precision.
        masked_logits = logits[rows, chunk_positions].float()
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        picked = log_probs[rows, chunk_true_ids]

    return picked.tolist()


def compute_pseudo_perplexity_batched(
    backend: EsmBackend,
    sequence: str,
    max_batch_tokens: int = DEFAULT_ESM_BATCH_TOKENS,
) -> tuple[float, float]:
    """Batched pseudo-perplexity and mean log-likelihood for one sequence.

    Same metric as :func:`compute_pseudo_perplexity`, same NaN-on-failure
    contract, but the per-residue masked forwards are batched.
    """
    if not sequence:
        return np.nan, np.nan
    try:
        log_probs = _masked_log_probs(backend, sequence, max_batch_tokens=max_batch_tokens)
        if not log_probs:
            return np.nan, np.nan
        avg_log_likelihood = sum(log_probs) / len(log_probs)
        return float(np.exp(-avg_log_likelihood)), float(avg_log_likelihood)
    except Exception as e:
        logger.error(f"ESM computation failed: {e}")
        return np.nan, np.nan


# =============================================================================
# Weight Resolution
# =============================================================================


def resolve_esm_dtype(model_name: str, configured: str = "auto"):
    """Which dtype to materialise ESM weights in.

    ``"auto"`` here does NOT mean transformers' ``"auto"``. That one follows the
    checkpoint's declared ``torch_dtype``, and biohub/ESMC-6B declares none -- so it
    materialised float32 and the CBLN1 smoke test logged
    ``6.35B parameters, torch.float32, 23.7 GiB of weights``, roughly twice what
    the model computes in.

    ESMC is scored in bfloat16 by this module either way (see the softmax in
    ``_score_chunk``, taken in float32 precisely because the logits are not), so
    float32 *weights* buy no precision that survives to the number we keep. They
    cost ~12 GiB on a card that also holds ESMFold2 and, when the folding backend
    is colabdesign, JAX's preallocation -- which is what pushed both ESMFold2's
    advisory refolding and the ProteinMPNN subprocess out of memory.

    So ESMC defaults to bfloat16 and everything else to the checkpoint's own dtype:
    ESM2 650M is small enough that its footprint has never been the constraint, and
    silently halving its precision would change published numbers for no gain.
    ``float32`` or ``bfloat16`` may be set explicitly via ``metric.esm_dtype``.
    """
    if configured and configured != "auto":
        resolved = getattr(torch, configured, None)
        if resolved is None:
            raise ValueError(f"metric.esm_dtype={configured!r} is not a torch dtype (try bfloat16, float16, float32)")
        return resolved
    return torch.bfloat16 if "esmc" in model_name.lower() else "auto"


def load_masked_lm(model_name: str, dtype="auto", **kwargs):
    """``AutoModelForMaskedLM.from_pretrained`` that keeps the checkpoint's dtype.

    Without an explicit dtype, transformers materialises weights in float32 whatever
    the checkpoint stores. For ESMC 6B that is ~24 GB of parameters where the
    bfloat16 the model actually computes in needs ~12 GB -- and the evaluation
    process shares its card with JAX, so the difference decides whether ESMFold2's
    advisory refolding fits. On the CBLN1 smoke test it did not: torch held
    38.85 GiB of live tensors and a 38 MiB allocation failed.

    ``"auto"`` rather than a hardcoded bfloat16, so a checkpoint that genuinely is
    float32 stays float32 and only the ones that declare a smaller dtype shrink.

    The keyword was renamed ``torch_dtype`` -> ``dtype`` in transformers 4.56.
    This repo's pyproject admits >=4.57,<6 and the Biohub fork pins 4.57.6, so both
    spellings are in range; the new one is tried first and the old is the fallback.
    """
    # The module-level symbol, not a local import: it doubles as the ESM_AVAILABLE
    # probe, and re-importing here would leave that import unused and removable.
    try:
        return AutoModelForMaskedLM.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


def log_model_footprint(model, model_name: str) -> None:
    """Say what was actually loaded, so a dtype surprise is visible on the next run.

    Reading this off the model beats inferring it from an OOM message, which is how
    it was found the first time.
    """
    try:
        params = sum(p.numel() for p in model.parameters())
        dtype = next(model.parameters()).dtype
        gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        logger.info(f"{model_name}: {params / 1e9:.2f}B parameters, {dtype}, {gib:.1f} GiB of weights")
    except (StopIteration, RuntimeError) as exc:  # a meta-device or empty model
        logger.debug(f"Could not measure {model_name} footprint: {exc}")


def _resolve_esm_dir() -> str | None:
    """Resolve ESM_DIR as a direct local model path.

    ESM_DIR should point to a HuggingFace hub cache directory that already
    contains the downloaded model (e.g. ``community_models/ckpts/ESM2``).
    Returns None if ESM_DIR is not set or the directory doesn't exist.

    ESM2 only: ESMC resolves weights via snapshot_download and takes no
    cache_dir, so it follows HF_HOME/HF_HUB_CACHE instead.
    """
    esm_dir = os.environ.get("ESM_DIR")
    if esm_dir:
        esm_dir = os.path.expanduser(esm_dir)
        if os.path.isdir(esm_dir):
            logger.debug(f"Using ESM_DIR from environment: {esm_dir}")
            return esm_dir
    return None


def _resolve_cache_dir() -> str | None:
    """Resolve the HuggingFace cache directory for ESM models, or None.

    Priority:
    1. CACHE_DIR environment variable
    2. None -- let HuggingFace resolve it (HF_HOME / HF_HUB_CACHE, else
       ~/.cache/huggingface/hub)

    Returning None rather than a path matters twice. It used to return
    ``~/.cache``, which is one level above the real hub cache
    (``~/.cache/huggingface/hub``), so weights already present were reported
    missing and re-downloaded into a second, parallel tree. And because an
    explicit ``cache_dir=`` overrides HF_HOME entirely, any value here silently
    disabled HF_HOME -- the very variable the shared weight caches are keyed on.
    ``run_esmfold`` already passes None unless CACHE_DIR is set; this matches it.

    Note: ESM_DIR is handled separately as a local model path, not as a
    download cache. This prevents HuggingFace from downloading model files
    into the project tree.
    """
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        logger.debug(f"Using CACHE_DIR from environment: {cache_dir}")
        return cache_dir

    logger.debug("No CACHE_DIR; deferring to HuggingFace (HF_HOME / HF_HUB_CACHE)")
    return None


# =============================================================================
# Loading
# =============================================================================


def _load_hf_masked_lm(
    model_name: str, device: str, force_offline: bool, kind: str = BACKEND_ESM2, dtype="auto"
) -> EsmBackend:
    """Load a HuggingFace masked LM from ESM_DIR, then the HF cache.

    Serves both ``esm2`` and ``esmc``: Biohub's transformers fork registers
    ESMCForMaskedLM in the Auto mappings, so a transformers-format ESMC repo
    loads through exactly this path.
    """
    if not ESM_AVAILABLE:
        raise RuntimeError("ESM/transformers not available. Install with: pip install transformers")

    esm_dir = _resolve_esm_dir()
    cache_dir = _resolve_cache_dir()

    load_locations = []
    if esm_dir:
        load_locations.append(("ESM_DIR", esm_dir))
    load_locations.append(("CACHE_DIR" if cache_dir else "HF default cache", cache_dir))

    model = None
    tokenizer = None

    for label, loc in load_locations:
        logger.info(f"Loading ESM model: {model_name} ({label}: {loc})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=loc,
                local_files_only=True,
            )
            model = load_masked_lm(model_name, dtype=dtype, cache_dir=loc, local_files_only=True)
            logger.info(f"Loaded ESM model from {label} (offline)")
            log_model_footprint(model, model_name)
            break
        except Exception:
            logger.debug(f"ESM model not found in {label}: {loc}")
            continue

    if model is None:
        if force_offline:
            # Only name cache_dir in the hint when one is actually in force;
            # otherwise the download lands wherever HF_HOME points, which is
            # where the loader will look next time.
            cache_kwarg = f", cache_dir='{cache_dir}'" if cache_dir else ""
            search_paths = ", ".join(
                f"{label}={loc}" if loc else f"{label}=HF_HOME/HF_HUB_CACHE" for label, loc in load_locations
            )
            logger.error(
                f"Failed to load ESM model from local paths: {search_paths}\n"
                f"The model may not be downloaded yet. To download, run:\n"
                f'  python -c "from transformers import AutoTokenizer, AutoModelForMaskedLM; '
                f"AutoTokenizer.from_pretrained('{model_name}'{cache_kwarg}); "
                f"AutoModelForMaskedLM.from_pretrained('{model_name}'{cache_kwarg})\""
            )
            raise RuntimeError(f"ESM model not found in local paths: {search_paths}")

        # If not forcing offline, download to cache_dir (not ESM_DIR)
        logger.info(f"Downloading ESM model from HuggingFace to {cache_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = load_masked_lm(model_name, dtype=dtype, cache_dir=cache_dir)
        log_model_footprint(model, model_name)

    model = model.to(device)
    model.eval()

    return EsmBackend(
        kind=kind,
        model=model,
        tokenizer=tokenizer,
        device=device,
        mask_token_id=tokenizer.mask_token_id,
    )


def _load_esmc_pkg(model_name: str, device: str) -> EsmBackend:
    """Load ESMC from the ``esm`` package (the non-working implementation).

    Prefer ``esm_backend=esmc`` with a transformers-format repo. This path is
    kept because the package registry carries sizes the transformers repos may
    not, but as of 2026-08-24 it cannot actually load: see resolve_backend.

    Weights come from snapshot_download, so they follow HF_HOME/HF_HUB_CACHE --
    ESM_DIR and CACHE_DIR do not apply here. The repos are gated: an HF token
    and an accepted licence are required, and HF_HUB_OFFLINE=1 needs a warm
    cache.
    """
    if importlib.util.find_spec("esm") is None:
        raise RuntimeError(
            "ESMC backend requested but the 'esm' package is not installed. "
            "Install the Biohub/EvolutionaryScale esm package, or set "
            "metric.esm_backend=esm2."
        )

    try:
        from esm.models.esmc import ESMC
    except ImportError as exc:
        # Facebook's fair-esm claims the same top-level module name as the
        # Biohub/EvolutionaryScale package, so only one can be installed at a
        # time -- and designability.run_esmfold_multimer imports fair-esm's.
        # Convert to RuntimeError so the caller's NaN fallback still applies
        # instead of the ImportError escaping and failing the whole evaluation.
        raise RuntimeError(
            f"An 'esm' module is installed but provides no ESMC ({exc}). "
            "Facebook's fair-esm uses the same module name and shadows the "
            "Biohub/EvolutionaryScale package; they cannot coexist, and "
            "run_esmfold_multimer needs fair-esm. Set metric.esm_backend=esm2 "
            "to score with ESM2 instead."
        ) from exc

    if os.environ.get("ESM_DIR") or os.environ.get("CACHE_DIR"):
        logger.info(
            "ESMC resolves weights through HF_HOME/HF_HUB_CACHE; ESM_DIR and CACHE_DIR are ignored for this backend."
        )

    logger.info(f"Loading ESMC model: {model_name} (device={device})")
    try:
        model = ESMC.from_pretrained(model_name, device=torch.device(device))
    except TypeError:
        # Older signatures take no device kwarg.
        model = ESMC.from_pretrained(model_name).to(device)
    except Exception as exc:
        available = _esmc_registry_names()
        hint = f" Known local model names: {available}." if available else ""
        raise RuntimeError(f"Failed to load ESMC model '{model_name}': {exc}.{hint}") from exc

    model.eval()

    tokenizer = getattr(model, "tokenizer", None)
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        raise RuntimeError(
            f"ESMC model '{model_name}' exposes no tokenizer.mask_token_id; cannot compute masked pseudo-perplexity"
        )

    return EsmBackend(
        kind=BACKEND_ESMC_PKG,
        model=model,
        tokenizer=tokenizer,
        device=device,
        mask_token_id=mask_token_id,
    )


def _esmc_registry_names() -> list[str]:
    """Local model names the installed ``esm`` package knows, for error messages."""
    try:
        from esm.pretrained import LOCAL_MODEL_REGISTRY

        return sorted(str(k) for k in LOCAL_MODEL_REGISTRY)
    except Exception:
        return []


def get_esm_backend(
    model_name: str = DEFAULT_ESM_MODEL,
    backend: str = "auto",
    device: str | None = None,
    force_offline: bool = True,
    dtype: str = "auto",
) -> EsmBackend:
    """Get or load a scoring backend (cached globally).

    Cached on (backend, model name, device), so switching models or backends
    within a session loads the new one rather than serving the old one.

    Args:
        model_name: HF model name for ESM2, or a registry name for ESMC.
        backend: ``auto``, ``esm2``, or ``esmc``.
        device: Device to load on (default: auto-detect cuda/cpu).
        force_offline: If True, load from local caches only and never download.
        dtype: ``auto`` (bfloat16 for ESMC, the checkpoint's own dtype otherwise),
            or an explicit torch dtype name. See :func:`resolve_esm_dtype`.

    Returns:
        A loaded :class:`EsmBackend` in eval mode.
    """
    kind = resolve_backend(model_name, backend)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # force_offline is expressed per call, via local_files_only= on each
    # from_pretrained below, and never by setting HF_HUB_OFFLINE. That env var is
    # process-global: this function used to set it and never restore it, so a run
    # that scored with ESM left every later download disabled -- including
    # ESMFold2's, which loads through an ordinary from_pretrained in the same
    # process, a few lines later in the same evaluation loop. A valid token and a
    # cold cache then failed for a reason nothing in the traceback pointed at.
    key = (kind, model_name, device)
    cached = _ESM_BACKEND_CACHE.get(key)
    if cached is not None:
        logger.debug(f"Using cached ESM backend: {kind}:{model_name} on {device}")
        return cached

    while len(_ESM_BACKEND_CACHE) >= _ESM_CACHE_MAXSIZE:
        evicted, _ = _ESM_BACKEND_CACHE.popitem()
        logger.info(f"Evicting cached ESM backend {evicted[0]}:{evicted[1]} to load {kind}:{model_name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if kind == BACKEND_ESMC_PKG:
        loaded = _load_esmc_pkg(model_name, device)
    else:
        loaded = _load_hf_masked_lm(model_name, device, force_offline, kind, resolve_esm_dtype(model_name, dtype))

    _ESM_BACKEND_CACHE[key] = loaded
    logger.info(f"ESM backend loaded ({kind}:{model_name}) on {device} and cached for reuse")
    return loaded


def get_esm_model(
    model_name: str = DEFAULT_ESM_MODEL,
    device: str | None = None,
    force_offline: bool = True,
):
    """Get or load the ESM model and tokenizer (cached globally).

    Back-compatible shim over :func:`get_esm_backend` for callers that want the
    raw ``(model, tokenizer, device)`` triple.
    """
    loaded = get_esm_backend(model_name=model_name, device=device, force_offline=force_offline)
    return loaded.model, loaded.tokenizer, loaded.device


def clear_esm_cache():
    """Clear cached ESM models to free GPU memory."""
    if not _ESM_BACKEND_CACHE:
        return
    n = len(_ESM_BACKEND_CACHE)
    _ESM_BACKEND_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Cleared {n} cached ESM backend(s)")


# =============================================================================
# Score Cache
# =============================================================================


def esm_cache_fingerprint(model_name: str, backend: str = "auto") -> str:
    """Identity of the scorer: everything that changes the numbers.

    Covers the model name and the resolved backend, so an ESM2-to-ESMC switch
    cannot serve stale scores. Deliberately excludes ``max_batch_tokens``: the
    batched and unbatched paths are verified equivalent
    (``script_utils/bioinformatic/verify_esm_batching.py``), so the batch budget
    is a performance knob and keying on it would discard a valid cache.

    It also does not cover kernel availability. Installing transformer_engine or
    xformers shifts values by what the fork calls rounding noise, and cached
    entries will predate that; clear the caches if that matters for a comparison.
    """
    canonical = json.dumps(
        {"model_name": model_name, "backend": resolve_backend(model_name, backend)},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _esm_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, ESM_CACHE_FILENAME)


def read_esm_cache(cache_dir: str, fingerprint: str) -> dict[str, list[float]]:
    """Cached ``{sequence: [pppl, log_likelihood]}``, or empty.

    Returns empty rather than raising for anything unexpected -- a cache is an
    optimisation, and a bad one must not stop an evaluation.
    """
    path = _esm_cache_path(cache_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            cached = json.load(handle)
        if cached.get("fingerprint") != fingerprint:
            logger.info(
                f"ESM cache at {path} was produced by a different scorer "
                f"({str(cached.get('fingerprint'))[:12]} != {fingerprint[:12]}); recomputing"
            )
            return {}
        scores = cached["scores"]
        return {k: v for k, v in scores.items() if isinstance(v, list) and len(v) == 2}
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(f"Ignoring unusable ESM cache {path}: {exc}")
        return {}


def write_esm_cache(cache_dir: str, fingerprint: str, scores: dict[str, list[float]]) -> None:
    """Persist sequence scores. Never raises."""
    try:
        # Serialise first so a non-encodable payload leaves no half-written file.
        blob = json.dumps({"fingerprint": fingerprint, "scores": scores})
        with open(_esm_cache_path(cache_dir), "w") as handle:
            handle.write(blob)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"Could not write ESM cache for {cache_dir}: {exc}")


# =============================================================================
# Public API
# =============================================================================


def compute_esm_ppl_for_sequences(
    sequences: list[str],
    model_name: str = DEFAULT_ESM_MODEL,
    force_offline: bool = True,
    backend: str = "auto",
    max_batch_tokens: int = DEFAULT_ESM_BATCH_TOKENS,
    cache_dir: str | None = None,
    reuse_cache: bool = True,
    dtype: str = "auto",
) -> pd.DataFrame:
    """Compute ESM pseudo-perplexity for a list of sequences.

    Args:
        sequences: List of protein sequences.
        model_name: HF model name, or a registry name for the esmc_pkg backend.
        force_offline: If True, only load from local cache (no network requests).
        backend: ``auto``, ``esm2``, ``esmc``, or ``esmc_pkg``.
        max_batch_tokens: Token budget for one batched forward.
        cache_dir: Directory holding a per-design score cache. When every
            sequence is already cached the model is never loaded at all, which is
            the point: scoring is a per-residue masked forward per sequence, and
            a resumed evaluation would otherwise repay it in full while the
            refolding it accompanies costs nothing.
        reuse_cache: Set False to rescore and overwrite.

    Returns:
        DataFrame with sequences and their ESM metrics, one row per input in
        input order.
    """
    if not ESM_AVAILABLE:
        logger.error("ESM/transformers not available. Install with: pip install transformers")
        return pd.DataFrame(columns=["sequence"] + ESM_METRIC_COLS)

    if not sequences:
        logger.warning("No sequences provided for ESM evaluation")
        return pd.DataFrame(columns=["sequence"] + ESM_METRIC_COLS)

    fingerprint = esm_cache_fingerprint(model_name, backend)
    scores: dict[str, list[float]] = {}
    if cache_dir and reuse_cache:
        scores = read_esm_cache(cache_dir, fingerprint)

    # Deduplicate: the same sequence can appear more than once in a request, and
    # the metric depends only on (model, sequence).
    pending = [s for s in dict.fromkeys(sequences) if s and s not in scores]
    n_reused = len(set(sequences)) - len(pending)

    if pending:
        logger.info(
            f"Computing ESM pseudo-perplexity for {len(pending)} sequences"
            + (f" ({n_reused} reused from cache)" if n_reused else "")
        )
        try:
            esm_backend = get_esm_backend(model_name, backend=backend, force_offline=force_offline, dtype=dtype)
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to load ESM model: {e}")
            esm_backend = None

        if esm_backend is not None:
            fresh = {}
            for seq in pending:
                pppl, log_ll = compute_pseudo_perplexity_batched(esm_backend, seq, max_batch_tokens=max_batch_tokens)
                # Never cache a failure: a NaN here means the model or the
                # sequence failed, and persisting it would make one bad run
                # permanent for every later resume.
                if pppl == pppl and log_ll == log_ll:
                    fresh[seq] = [float(pppl), float(log_ll)]
            scores.update(fresh)
            if cache_dir and fresh:
                write_esm_cache(cache_dir, fingerprint, scores)
    else:
        logger.info(f"ESM pseudo-perplexity for {len(sequences)} sequences served entirely from cache")

    results = [
        {
            "sequence": seq,
            "esm_pseudo_perplexity": scores.get(seq, [np.nan, np.nan])[0],
            "esm_log_likelihood": scores.get(seq, [np.nan, np.nan])[1],
        }
        for seq in sequences
    ]
    return pd.DataFrame(results)


def compute_esm_ppl_for_pdbs(
    pdb_paths: list[str],
    protein_type: str = "binder",
    model_name: str = DEFAULT_ESM_MODEL,
    force_offline: bool = True,
    backend: str = "auto",
    max_batch_tokens: int = DEFAULT_ESM_BATCH_TOKENS,
) -> pd.DataFrame:
    """
    Compute ESM pseudo-perplexity for sequences extracted from PDB files.

    Args:
        pdb_paths: List of PDB file paths
        protein_type: "binder" (last chain) or "monomer" (all chains)
        model_name: ESM model to use
        force_offline: If True, only load from local cache (no network requests)
        backend: ``auto``, ``esm2``, or ``esmc``.
        max_batch_tokens: Token budget for one batched forward.

    Returns:
        DataFrame with pdb_path, sequence, and ESM metrics
    """
    from proteinfoundation.evaluation.binder_eval_utils import get_binder_chain_from_complex
    from proteinfoundation.utils.pdb_utils import extract_seq_from_pdb

    if not ESM_AVAILABLE:
        logger.error("ESM/transformers not available. Install with: pip install transformers")
        return pd.DataFrame(columns=["pdb_path", "sequence"] + ESM_METRIC_COLS)

    if not pdb_paths:
        logger.warning("No PDB paths provided for ESM evaluation")
        return pd.DataFrame(columns=["pdb_path", "sequence"] + ESM_METRIC_COLS)

    logger.info(f"Computing ESM pseudo-perplexity for {len(pdb_paths)} PDB files (type: {protein_type})")

    try:
        esm_backend = get_esm_backend(model_name, backend=backend, force_offline=force_offline)
    except (RuntimeError, ValueError) as e:
        logger.error(f"Failed to load ESM model: {e}")
        return pd.DataFrame(
            [
                {
                    "pdb_path": p,
                    "sequence": None,
                    "esm_pseudo_perplexity": np.nan,
                    "esm_log_likelihood": np.nan,
                }
                for p in pdb_paths
            ]
        )

    results = []
    failed_count = 0
    for pdb_path in pdb_paths:
        try:
            if protein_type == "binder":
                binder_chain, _ = get_binder_chain_from_complex(pdb_path)
                sequence = extract_seq_from_pdb(pdb_path, chain_id=binder_chain)
            else:  # monomer
                sequence = extract_seq_from_pdb(pdb_path, chain_id=None)

            pppl, log_ll = compute_pseudo_perplexity_batched(esm_backend, sequence, max_batch_tokens=max_batch_tokens)

            results.append(
                {
                    "pdb_path": pdb_path,
                    "sequence": sequence,
                    "esm_pseudo_perplexity": pppl,
                    "esm_log_likelihood": log_ll,
                }
            )

        except Exception as e:
            logger.warning(f"Failed to process {pdb_path}: {e}")
            failed_count += 1
            results.append(
                {
                    "pdb_path": pdb_path,
                    "sequence": "FAIL",
                    "esm_pseudo_perplexity": np.nan,
                    "esm_log_likelihood": np.nan,
                }
            )

    if failed_count > 0:
        logger.warning(f"ESM evaluation failed for {failed_count}/{len(pdb_paths)} PDB files")
    logger.info(f"ESM evaluation complete for {len(pdb_paths) - failed_count}/{len(pdb_paths)} PDB files")

    return pd.DataFrame(results)
