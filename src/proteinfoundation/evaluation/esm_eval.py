"""
ESM sequence scoring: masked pseudo-perplexity and log-likelihood.

Lower pseudo-perplexity indicates a more "natural" sequence according to the
language model.

Two backends are supported, selected by ``backend`` (or auto-detected from the
model name):

* ``esm2`` -- HuggingFace ``AutoModelForMaskedLM`` (default,
  ``facebook/esm2_t33_650M_UR50D``).
* ``esmc`` -- the ``esm`` package's ``ESMC`` (EvolutionaryScale, or the Biohub
  fork that also ships ESMFold2; the two are the same model).

Both go through one batched scoring core: pseudo-perplexity needs one masked
forward *per residue*, and those L forwards are independent, so they are run as
a batch instead of a Python loop. The metric definition is unchanged --
:func:`compute_pseudo_perplexity` is kept as the unbatched reference used by
``script_utils/bioinformatic/verify_esm_batching.py`` to prove equivalence.

Models are cached globally, keyed on (backend, model name, device), so they load
once per session. ESM2 honours ``ESM_DIR``/``CACHE_DIR``; ESMC resolves weights
through ``HF_HOME``/``HF_HUB_CACHE`` instead (it calls ``snapshot_download``
internally and accepts no ``cache_dir``). Set ``HF_HUB_OFFLINE=1`` to force
fully offline mode; that requires an already-warm cache.
"""

import importlib.util
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

# Registry name understood by the installed ``esm`` package. Kept as a default
# only; a wrong name fails loudly at load time and the error lists the valid
# keys from that package's LOCAL_MODEL_REGISTRY.
DEFAULT_ESMC_MODEL = "esmc_600m"

BACKEND_ESM2 = "esm2"
BACKEND_ESMC = "esmc"

# Budget for one batched forward, in tokens (rows x padded length). A 100-residue
# sequence scores in ceil(100 / (16384 // 102)) = 1 forward instead of 100.
# Lower it if a long-sequence batch runs out of memory; the code also halves on
# OOM by itself.
DEFAULT_ESM_BATCH_TOKENS = 16384

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
        if self.kind == BACKEND_ESMC:
            # ESMC._tokenize takes a list and pads internally. Passing
            # sequence_id=None to forward lets it derive the pad mask, which
            # also avoids its flash-attention assert on the mask dtype.
            ids = self.model._tokenize(sequences)
            return ids.to(self.device), None

        encoded = self.tokenizer(sequences, return_tensors="pt", padding=True)
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        return ids, mask

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Run a forward pass and return ``[B, T, V]`` logits."""
        if self.kind == BACKEND_ESMC:
            return self.model.forward(sequence_tokens=input_ids).sequence_logits
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def resolve_backend(model_name: str, backend: str = "auto") -> str:
    """Resolve the backend for a model name.

    ``auto`` treats any name containing "esmc" as ESMC and everything else as
    ESM2, so ``esmc_600m`` and ``biohub/esmc-600m-2024-12`` both route to ESMC
    while ``facebook/esm2_t33_650M_UR50D`` does not.
    """
    backend = (backend or "auto").lower()
    if backend in (BACKEND_ESM2, BACKEND_ESMC):
        return backend
    if backend != "auto":
        raise ValueError(f"Unknown ESM backend '{backend}'. Expected one of: auto, {BACKEND_ESM2}, {BACKEND_ESMC}")
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


def _load_esm2(model_name: str, device: str, force_offline: bool) -> EsmBackend:
    """Load an ESM2 masked-LM from ESM_DIR, then the HF cache."""
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
            model = AutoModelForMaskedLM.from_pretrained(
                model_name,
                cache_dir=loc,
                local_files_only=True,
            )
            logger.info(f"Loaded ESM model from {label} (offline)")
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
        os.environ.pop("HF_HUB_OFFLINE", None)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

    model = model.to(device)
    model.eval()

    return EsmBackend(
        kind=BACKEND_ESM2,
        model=model,
        tokenizer=tokenizer,
        device=device,
        mask_token_id=tokenizer.mask_token_id,
    )


def _load_esmc(model_name: str, device: str) -> EsmBackend:
    """Load ESMC from the ``esm`` package.

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
        kind=BACKEND_ESMC,
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
) -> EsmBackend:
    """Get or load a scoring backend (cached globally).

    Cached on (backend, model name, device), so switching models or backends
    within a session loads the new one rather than serving the old one.

    Args:
        model_name: HF model name for ESM2, or a registry name for ESMC.
        backend: ``auto``, ``esm2``, or ``esmc``.
        device: Device to load on (default: auto-detect cuda/cpu).
        force_offline: If True, set HF_HUB_OFFLINE=1 and load locally only.

    Returns:
        A loaded :class:`EsmBackend` in eval mode.
    """
    kind = resolve_backend(model_name, backend)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    key = (kind, model_name, device)
    cached = _ESM_BACKEND_CACHE.get(key)
    if cached is not None:
        logger.debug(f"Using cached ESM backend: {kind}:{model_name} on {device}")
        return cached

    if force_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.debug("Set HF_HUB_OFFLINE=1 to force offline mode")

    while len(_ESM_BACKEND_CACHE) >= _ESM_CACHE_MAXSIZE:
        evicted, _ = _ESM_BACKEND_CACHE.popitem()
        logger.info(f"Evicting cached ESM backend {evicted[0]}:{evicted[1]} to load {kind}:{model_name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if kind == BACKEND_ESMC:
        loaded = _load_esmc(model_name, device)
    else:
        loaded = _load_esm2(model_name, device, force_offline)

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
# Public API
# =============================================================================


def compute_esm_ppl_for_sequences(
    sequences: list[str],
    model_name: str = DEFAULT_ESM_MODEL,
    force_offline: bool = True,
    backend: str = "auto",
    max_batch_tokens: int = DEFAULT_ESM_BATCH_TOKENS,
) -> pd.DataFrame:
    """Compute ESM pseudo-perplexity for a list of sequences.

    Args:
        sequences: List of protein sequences.
        model_name: HF model name (ESM2) or registry name (ESMC).
        force_offline: If True, only load from local cache (no network requests).
        backend: ``auto``, ``esm2``, or ``esmc``.
        max_batch_tokens: Token budget for one batched forward.

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

    logger.info(f"Computing ESM pseudo-perplexity for {len(sequences)} sequences")

    try:
        esm_backend = get_esm_backend(model_name, backend=backend, force_offline=force_offline)
    except (RuntimeError, ValueError) as e:
        logger.error(f"Failed to load ESM model: {e}")
        return pd.DataFrame(
            [
                {
                    "sequence": seq,
                    "esm_pseudo_perplexity": np.nan,
                    "esm_log_likelihood": np.nan,
                }
                for seq in sequences
            ]
        )

    results = []
    for seq in sequences:
        pppl, log_ll = compute_pseudo_perplexity_batched(esm_backend, seq, max_batch_tokens=max_batch_tokens)
        results.append(
            {
                "sequence": seq,
                "esm_pseudo_perplexity": pppl,
                "esm_log_likelihood": log_ll,
            }
        )

    logger.info(f"ESM evaluation complete for {len(sequences)} sequences")
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
