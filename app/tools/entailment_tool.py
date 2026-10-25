"""
KnowledgeMesh — HHEM Grounding / Factual Consistency Tool (batched)

Uses Vectara's HHEMv2 hallucination evaluation model directly through
Transformers rather than SentenceTransformers CrossEncoder.

What changed vs. the one-pair-at-a-time version
-----------------------------------------------
1. score_entailment_batch(): scores many (premise, hypothesis) pairs in a
   few forward passes instead of one forward pass per claim.
2. LRU cache keyed on (premise, hypothesis): a claim already scored
   against the same evidence is never re-scored.
3. warmup_hhem(): load the model and run one realistic batch at startup
   so the first real request does not pay the cold-start cost.
4. Truncation fix: the premise is truncated, never the hypothesis.
5. Tuning switches (environment variables, all optional):
    HHEM_BATCH_SIZE       pairs per forward pass (default 8)
    HHEM_PREMISE_TOKENS   max premise tokens per pair (default 512)
    HHEM_QUANTIZE         "1" = int8 dynamic quantization (CPU only)
    HHEM_NUM_THREADS      torch CPU threads (default: PyTorch's own)

Input:
    premise    = retrieved evidence
    hypothesis = generated claim

Output:
    float in [0.0, 1.0]

Interpretation:
    ~1.0 -> strongly supported
    ~0.5 -> uncertain
    ~0.0 -> unsupported / likely hallucinated

Public API:
    score_entailment(premise, hypothesis) -> float
    score_entailment_batch(pairs)         -> list[float]
    warmup_hhem()                         -> None
    get_hhem_profile / start_hhem_profile / stop_hhem_profile
"""

from __future__ import annotations


import hashlib
import logging
import os
import re
import threading
from collections import OrderedDict
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Sequence


import logfire
import torch
from transformers import AutoModelForSequenceClassification


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


MODEL_NAME = "vectara/hallucination_evaluation_model"


# Deliberate latency/memory cap.
#
# HHEM-2.1-Open inherits the "> 512 tokens" tokenizer warning from its
# Flan-T5-base foundation and supports longer inputs, but we keep prompts
# within 512 tokens by default for predictable batching and latency.
HHEM_MAX_LENGTH = 512


# Pairs per forward pass. On CPU 8 is a good default; on a GPU raise it
# (16-32). Pairs are length-sorted first so padding waste stays small.
HHEM_BATCH_SIZE = int(os.getenv("HHEM_BATCH_SIZE", "8"))


# Max tokens of premise per pair. 512 means "no extra cap": the premise
# then only has to fit the model window, as before.
HHEM_PREMISE_TOKENS = int(os.getenv("HHEM_PREMISE_TOKENS", "512"))


# "1" = int8 dynamic quantization of the T5 component (CPU only).
HHEM_QUANTIZE = os.getenv("HHEM_QUANTIZE", "0") == "1"


# 0 = leave PyTorch's default thread count alone.
_HHEM_THREADS = int(os.getenv("HHEM_NUM_THREADS", "0"))


if _HHEM_THREADS > 0:
    torch.set_num_threads(_HHEM_THREADS)


# Cached (premise, hypothesis) scores kept in memory.
HHEM_CACHE_SIZE = 4096


# Tokens kept free for special tokens when fitting the premise.
_TOKEN_MARGIN = 4


# Smallest premise budget worth scoring; below this the hypothesis itself
# is too long and gets truncated as a last resort.
_MIN_PREMISE_TOKENS = 32


# ============================================================
# Profiling (process-wide + per grounding pass)
# ============================================================


# Process-wide counters. The lock protects snapshots and updates; these
# cumulative counters are not request-scoped.
_hhem_call_count = 0
_hhem_total_ms = 0.0
_hhem_profile_lock = threading.Lock()
_active_hhem_profile: ContextVar[dict[str, float] | None] = ContextVar(
    "active_hhem_profile",
    default=None,
)


def get_hhem_profile() -> dict[str, float | int]:
    """Return a thread-safe snapshot of process-wide scorer metrics."""
    with _hhem_profile_lock:
        return {
            "call_count": int(_hhem_call_count),
            "total_ms": round(float(_hhem_total_ms), 2),
        }


def start_hhem_profile() -> tuple[object, dict[str, float]]:
    """Start a context-local profile for one grounding pass."""
    profile: dict[str, float] = {"call_count": 0, "total_ms": 0.0}
    token = _active_hhem_profile.set(profile)
    return token, profile


def stop_hhem_profile(token: object) -> None:
    """Restore the previous context-local profile."""
    _active_hhem_profile.reset(token)


def _record_profile(pairs_scored: int, elapsed_ms: float) -> None:
    """Attribute model work to the global and the active-pass profiles."""
    global _hhem_call_count, _hhem_total_ms

    profile = _active_hhem_profile.get()

    if profile is not None:
        profile["call_count"] += pairs_scored
        profile["total_ms"] += elapsed_ms

    with _hhem_profile_lock:
        _hhem_call_count += pairs_scored
        _hhem_total_ms += elapsed_ms


# ============================================================
# Model loading / warmup
# ============================================================


_model: Any | None = None
_model_lock = threading.Lock()


def _get_model() -> Any:
    """Lazily load the HHEMv2 model once per Python process."""

    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logfire.info(
                "Loading HHEMv2 grounding model",
                model=MODEL_NAME,
                max_length=HHEM_MAX_LENGTH,
            )

            _model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
            )

            _model.eval()

            if HHEM_QUANTIZE:
                try:
                    _model.t5 = torch.quantization.quantize_dynamic(
                        _model.t5,
                        {torch.nn.Linear},
                        dtype=torch.qint8,
                    )
                    logger.info("HHEMv2 int8 dynamic quantization enabled")
                except Exception as exc:
                    logger.warning("HHEMv2 quantization skipped: %s", exc)

            logfire.info(
                "HHEMv2 grounding model ready",
                model=MODEL_NAME,
                max_length=HHEM_MAX_LENGTH,
            )

    return _model


def warmup_hhem() -> None:
    """
    Load the model and run one realistic batch.

    Call once at application startup so the first real request does not
    pay model-load and first-inference cost. The dummy batch uses
    near-limit prompts so the first large real batch is not cold.
    Failures are logged and swallowed: scoring still works lazily.
    """

    started = perf_counter()

    try:
        model = _get_model()

        long_premise = " ".join(
            ["Paris is the capital of France and its largest city."] * 40
        )

        dummy = [
            (f"{long_premise} Item {i}.", "Paris is a city in France.")
            for i in range(HHEM_BATCH_SIZE)
        ]

        _score_uncached(model, dummy)

        logger.info(
            "HHEMv2 warmup complete in %.0f ms",
            (perf_counter() - started) * 1000,
        )

    except Exception as exc:
        logger.warning("HHEMv2 warmup failed (will load lazily): %s", exc)


# ============================================================
# Cache
# ============================================================


_cache: "OrderedDict[str, float]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(premise: str, hypothesis: str) -> str:
    digest = hashlib.sha1()
    digest.update(premise.encode("utf-8", errors="ignore"))
    digest.update(b"\x00")
    digest.update(hypothesis.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def _cache_get(key: str) -> float | None:
    with _cache_lock:
        value = _cache.get(key)

        if value is not None:
            _cache.move_to_end(key)

        return value


def _cache_put(key: str, value: float) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)

        while len(_cache) > HHEM_CACHE_SIZE:
            _cache.popitem(last=False)


def clear_hhem_cache() -> None:
    """Drop all cached scores (tests, or after swapping the model)."""
    with _cache_lock:
        _cache.clear()


# ============================================================
# Prompt building and premise-side truncation
# ============================================================


def _normalize_score(score: float) -> float:
    """Clamp the factual-consistency score to [0, 1]."""
    return max(0.0, min(1.0, float(score)))


def _prompt_template(model: Any) -> str:
    template = getattr(model, "prompt", None)

    if not template:
        raise ValueError("HHEMv2 model does not expose its expected prompt template.")

    return template


def _tokenizer(model: Any) -> Any:
    # The upstream HHEMv2 remote code really spells this attribute
    # `tokenzier`; keep the typo.
    tokenizer = getattr(model, "tokenzier", None)

    if tokenizer is None:
        raise ValueError("HHEMv2 tokenizer is unavailable on model.tokenzier.")

    return tokenizer


def _select_relevant_sentences(
    premise: str,
    hypothesis: str,
    tokenizer: Any,
    budget: int,
) -> str:
    """
    Keep the premise sentences that overlap most with the claim, in
    original order, until the token budget is used.

    Used only when the premise is longer than the budget. Plain
    head-truncation can drop the sentence that actually supports the
    claim; this keeps the likeliest supporting sentences instead.
    """

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", premise) if s.strip()
    ]

    if len(sentences) <= 1:
        return premise

    hyp_terms = set(re.findall(r"\w+", hypothesis.lower()))

    def overlap(sentence: str) -> int:
        return len(hyp_terms & set(re.findall(r"\w+", sentence.lower())))

    ranked = sorted(
        range(len(sentences)),
        key=lambda i: overlap(sentences[i]),
        reverse=True,
    )

    # Tokenize each sentence once instead of once per selection attempt.
    token_counts = [
        len(
            tokenizer(
                sentence,
                add_special_tokens=False,
                verbose=False,
            )["input_ids"]
        )
        for sentence in sentences
    ]

    kept: set[int] = set()
    used = 0

    for i in ranked:
        sentence_tokens = token_counts[i]

        if used + sentence_tokens > budget:
            continue

        kept.add(i)
        used += sentence_tokens

    if not kept:
        return premise

    return " ".join(sentences[i] for i in sorted(kept))


def _build_prompt(model: Any, premise: str, hypothesis: str) -> str:
    """
    Build the HHEMv2 prompt, fitting the PREMISE into the context window.

    The prompt ends with the hypothesis, and tokenizer truncation cuts
    from the right, so truncating the whole prompt would chop the claim
    being verified. Instead, measure everything except the premise and
    trim the premise to the remaining token budget.

    When the premise exceeds the budget, keep its most claim-relevant
    sentences; fall back to head truncation only if that still does not
    fit.
    """

    template = _prompt_template(model)
    tokenizer = _tokenizer(model)

    overhead_ids = tokenizer(
        template.format(text1="", text2=hypothesis),
        add_special_tokens=True,
        verbose=False,
    )["input_ids"]

    budget = HHEM_MAX_LENGTH - len(overhead_ids) - _TOKEN_MARGIN

    if budget < _MIN_PREMISE_TOKENS:
        # Pathologically long hypothesis: trim it, keep a usable premise.
        hyp_ids = tokenizer(
            hypothesis,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]

        keep = max(
            _MIN_PREMISE_TOKENS,
            len(hyp_ids) - (_MIN_PREMISE_TOKENS - budget),
        )

        hypothesis = tokenizer.decode(
            hyp_ids[:keep],
            skip_special_tokens=True,
        )

        overhead_ids = tokenizer(
            template.format(text1="", text2=hypothesis),
            add_special_tokens=True,
            verbose=False,
        )["input_ids"]

        budget = max(
            _MIN_PREMISE_TOKENS,
            HHEM_MAX_LENGTH - len(overhead_ids) - _TOKEN_MARGIN,
        )

    # Optional extra cap for speed.
    budget = min(budget, HHEM_PREMISE_TOKENS)

    premise_ids = tokenizer(
        premise,
        add_special_tokens=False,
        verbose=False,
    )["input_ids"]

    if len(premise_ids) > budget:
        premise = _select_relevant_sentences(
            premise,
            hypothesis,
            tokenizer,
            budget,
        )

        premise_ids = tokenizer(
            premise,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]

        # Still too long, for example one giant sentence: head-truncate.
        if len(premise_ids) > budget:
            premise = tokenizer.decode(
                premise_ids[:budget],
                skip_special_tokens=True,
            )

    return template.format(text1=premise, text2=hypothesis)


# ============================================================
# Batched inference
# ============================================================


def _infer_prompts(model: Any, prompts: Sequence[str]) -> list[float]:
    """One forward pass over already-built prompts."""

    tokenizer = _tokenizer(model)

    t5_model = getattr(model, "t5", None)

    if t5_model is None:
        raise ValueError("HHEMv2 model does not expose the expected `.t5` component.")

    tokenize_started = perf_counter()

    # Truncation remains enabled as a safety net; _build_prompt already
    # fits the premise, so in practice nothing should be cut here.
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=HHEM_MAX_LENGTH,
    )

    tokenize_ms = round((perf_counter() - tokenize_started) * 1000, 2)

    device = t5_model.device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    t5_model.eval()

    inference_started = perf_counter()

    with torch.inference_mode():
        outputs = t5_model(**inputs)

    inference_ms = round((perf_counter() - inference_started) * 1000, 2)

    logger.info(
        "HHEMv2 batch inference: pairs=%s tokenize_ms=%s inference_ms=%s",
        len(prompts),
        tokenize_ms,
        inference_ms,
    )

    # Match the upstream HHEMv2 implementation:
    # logits[:, 0, :] -> classification token, softmax, class 1.
    logits = outputs.logits[:, 0, :]
    probs = torch.softmax(logits, dim=-1)[:, 1]

    return [_normalize_score(value) for value in probs.detach().cpu().tolist()]


def _score_uncached(
    model: Any,
    pairs: Sequence[tuple[str, str]],
) -> list[float | None]:
    """
    Score pairs in length-sorted chunks of HHEM_BATCH_SIZE.

    Order of returned scores matches `pairs`.

    A successful model score is a float, including a genuine 0.0.
    A failed pair is None, so callers can distinguish failure from an
    actual unsupported score.
    """

    prompts = [_build_prompt(model, p, h) for p, h in pairs]

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    scores: list[float | None] = [None] * len(prompts)

    for start in range(0, len(order), HHEM_BATCH_SIZE):
        chunk_idx = order[start : start + HHEM_BATCH_SIZE]
        chunk_prompts = [prompts[i] for i in chunk_idx]

        try:
            chunk_scores = _infer_prompts(model, chunk_prompts)
        except Exception as exc:
            logger.warning(
                "HHEMv2 batch of %s failed (%s); retrying pair-by-pair.",
                len(chunk_prompts),
                exc,
            )

            chunk_scores = []

            for prompt in chunk_prompts:
                try:
                    chunk_scores.append(_infer_prompts(model, [prompt])[0])
                except Exception:
                    logger.exception("HHEMv2 single-pair scoring failed.")
                    chunk_scores.append(None)

        if len(chunk_scores) != len(chunk_idx):
            raise RuntimeError("HHEM inference returned an unexpected number of scores")

        for i, score in zip(chunk_idx, chunk_scores):
            scores[i] = score

    return scores


# ============================================================
# Public API
# ============================================================


def score_entailment_batch(
    pairs: Sequence[tuple[str, str]],
) -> list[float]:
    """
    Score many (premise, hypothesis) pairs at once.

    Returns one float in [0.0, 1.0] per input pair, in input order.

    - Empty premise/hypothesis -> 0.0 (not sent to the model).
    - Cached pairs are returned without inference.
    - Duplicate pairs inside the batch are scored once.
    - Any model failure scores the affected pair 0.0, so the RAG graph
      can continue and treat the claim as unsupported. Failures are
      never cached.
    """

    total = len(pairs)

    if total == 0:
        return []

    started = perf_counter()

    results: list[float] = [0.0] * total

    # key -> (premise, hypothesis, [result indices])
    pending: dict[str, tuple[str, str, list[int]]] = {}
    cache_hits = 0

    for i, (premise, hypothesis) in enumerate(pairs):
        premise = str(premise or "").strip()
        hypothesis = str(hypothesis or "").strip()

        if not premise or not hypothesis:
            results[i] = 0.0
            continue

        key = _cache_key(premise, hypothesis)

        cached = _cache_get(key)

        if cached is not None:
            results[i] = cached
            cache_hits += 1
            continue

        if key in pending:
            pending[key][2].append(i)
        else:
            pending[key] = (premise, hypothesis, [i])

    model_pairs = 0

    if pending:
        try:
            model = _get_model()

            keys = list(pending.keys())
            batch = [(pending[k][0], pending[k][1]) for k in keys]
            model_pairs = len(batch)

            with logfire.span(
                "🔬 HHEMv2 Grounding Batch",
                pairs=total,
                cache_hits=cache_hits,
                model_pairs=model_pairs,
                batch_size=HHEM_BATCH_SIZE,
                max_input_tokens=HHEM_MAX_LENGTH,
            ):
                scores = _score_uncached(model, batch)

            if len(scores) != len(keys):
                raise RuntimeError(
                    "HHEM scorer returned an unexpected number of scores"
                )

            for key, score in zip(keys, scores):
                for i in pending[key][2]:
                    if score is None:
                        results[i] = 0.0
                    else:
                        results[i] = _normalize_score(score)

                # Cache every successful model output, including a
                # genuine 0.0. Failed pairs remain None and are not cached.
                if score is not None:
                    _cache_put(key, _normalize_score(score))

        except Exception as exc:
            logger.exception("HHEMv2 grounding scoring failed: %s", exc)

            try:
                logfire.exception(
                    "❌ HHEMv2 grounding scoring failed",
                    error_type=type(exc).__name__,
                )
            except Exception:
                pass

    elapsed_ms = round((perf_counter() - started) * 1000, 2)

    _record_profile(pairs_scored=model_pairs, elapsed_ms=elapsed_ms)

    logger.info(
        "HHEMv2 batch done: pairs=%s cache_hits=%s model_pairs=%s elapsed_ms=%s",
        total,
        cache_hits,
        model_pairs,
        elapsed_ms,
    )

    return results


def score_entailment(
    premise: str,
    hypothesis: str,
) -> float:
    """
    Score whether `hypothesis` is factually supported by `premise`.

    Single-pair convenience wrapper over score_entailment_batch, so it
    shares the cache and the premise-side truncation fix. Prefer the
    batch API when scoring several claims.

    Fail-safe: any model/evaluation failure returns 0.0.
    """

    return score_entailment_batch([(premise, hypothesis)])[0]
