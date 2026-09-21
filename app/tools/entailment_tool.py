"""
KnowledgeMesh — HHEM Grounding / Factual Consistency Tool

Uses Vectara's HHEMv2 hallucination evaluation model directly through
Transformers rather than SentenceTransformers CrossEncoder.

Important:
    The upstream HHEMv2 remote `predict()` implementation tokenizes without
    explicit truncation. Long evidence + hypothesis pairs can therefore
    exceed the T5 model's 512-token context window.

This wrapper performs the same HHEMv2 inference path explicitly while
enforcing the tokenizer's maximum sequence length.

Input:
    premise    = retrieved evidence
    hypothesis = generated claim

Output:
    float in [0.0, 1.0]

Interpretation:
    ~1.0 -> strongly supported
    ~0.5 -> uncertain
    ~0.0 -> unsupported / likely hallucinated
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import logfire
import torch
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_NAME = "vectara/hallucination_evaluation_model"

# HHEMv2 uses a T5 tokenizer with a 512-token context window.
# Keep this explicit rather than relying on tokenizer defaults.
HHEM_MAX_LENGTH = 512

_model: Any | None = None
_model_lock = threading.Lock()


def _get_model() -> Any:
    """
    Lazily load the HHEMv2 model once per Python process.
    """

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

            logfire.info(
                "HHEMv2 grounding model ready",
                model=MODEL_NAME,
                max_length=HHEM_MAX_LENGTH,
            )

    return _model


def _to_float(value: Any) -> float:
    """
    Convert tensors / numpy scalars / nested single-value outputs
    into a Python float.
    """

    if hasattr(value, "detach"):
        value = value.detach()

    if hasattr(value, "cpu"):
        value = value.cpu()

    if hasattr(value, "numpy"):
        value = value.numpy()

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Empty model output.")

        return _to_float(value[0])

    try:
        return float(value)
    except Exception as exc:
        raise ValueError(
            f"Unable to convert model output to float: {type(value).__name__}"
        ) from exc


def _extract_score(raw_output: Any) -> float:
    """
    Extract the factual-consistency score from HHEMv2 output.
    """

    if raw_output is None:
        raise ValueError("HHEMv2 returned None.")

    # Tensor / numpy-like output.
    if hasattr(raw_output, "shape"):
        try:
            if len(raw_output.shape) == 0:
                return _to_float(raw_output)

            return _to_float(raw_output[0])
        except Exception:
            pass

    # List / tuple output.
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            raise ValueError("HHEMv2 returned an empty result.")

        return _to_float(raw_output[0])

    return _to_float(raw_output)


def _normalize_score(score: float) -> float:
    """
    Clamp the factual-consistency score to [0, 1].
    """

    return max(0.0, min(1.0, float(score)))


def _build_prompt(
    model: Any,
    premise: str,
    hypothesis: str,
) -> str:
    """
    Build the exact HHEMv2 prompt used by its remote implementation.
    """

    prompt_template = getattr(model, "prompt", None)

    if not prompt_template:
        raise ValueError("HHEMv2 model does not expose its expected prompt template.")

    return prompt_template.format(
        text1=premise,
        text2=hypothesis,
    )


def _predict_with_safe_truncation(
    model: Any,
    premise: str,
    hypothesis: str,
) -> Any:
    """
    Run HHEMv2 inference while explicitly enforcing the tokenizer's
    maximum sequence length.

    This mirrors the upstream HHEMv2 `predict()` implementation:

        prompt -> tokenizer -> T5 -> logits -> softmax -> class 1

    but adds:

        truncation=True
        max_length=512

    to prevent context-window overflow.
    """

    tokenizer = getattr(model, "tokenzier", None)

    if tokenizer is None:
        raise ValueError("HHEMv2 tokenizer is unavailable on model.tokenzier.")

    prompt = _build_prompt(
        model=model,
        premise=premise,
        hypothesis=hypothesis,
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=HHEM_MAX_LENGTH,
    )

    # The upstream implementation uses the T5 component exposed as
    # `model.t5`, so preserve that behavior exactly.
    t5_model = getattr(model, "t5", None)

    if t5_model is None:
        raise ValueError("HHEMv2 model does not expose the expected `.t5` component.")

    # Move tokenizer outputs to the same device as the T5 model.
    try:
        device = t5_model.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
    except Exception:
        # CPU is a safe fallback if device discovery is unavailable.
        pass

    t5_model.eval()

    with torch.no_grad():
        outputs = t5_model(**inputs)

    logits = outputs.logits

    # Match the upstream HHEMv2 implementation:
    # logits[:, 0, :] -> classification token
    logits = logits[:, 0, :]

    transformed_probs = torch.softmax(logits, dim=-1)

    # Probability of class 1 = factual consistency score.
    raw_scores = transformed_probs[:, 1]

    return raw_scores


def score_entailment(
    premise: str,
    hypothesis: str,
) -> float:
    """
    Score whether `hypothesis` is factually supported by `premise`.

    The premise is the trusted/retrieved evidence.
    The hypothesis is the generated claim.

    Returns:
        float in [0.0, 1.0]

    Fail-safe behavior:
        Any model/evaluation failure returns 0.0 so the RAG graph
        can continue and treat the claim as unsupported.
    """

    premise = str(premise or "").strip()
    hypothesis = str(hypothesis or "").strip()

    if not premise or not hypothesis:
        return 0.0

    try:
        model = _get_model()

        with logfire.span(
            "🔬 HHEMv2 Grounding Score",
            premise_chars=len(premise),
            hypothesis_chars=len(hypothesis),
            max_input_tokens=HHEM_MAX_LENGTH,
        ):
            raw_output = _predict_with_safe_truncation(
                model=model,
                premise=premise,
                hypothesis=hypothesis,
            )

            score = _extract_score(raw_output)
            score = _normalize_score(score)

            logfire.info(
                "HHEMv2 grounding score computed",
                score=round(score, 4),
                max_input_tokens=HHEM_MAX_LENGTH,
            )

            return score

    except Exception as exc:
        logger.exception(
            "HHEMv2 grounding scoring failed: %s",
            exc,
        )

        try:
            logfire.exception(
                "❌ HHEMv2 grounding scoring failed",
                error_type=type(exc).__name__,
            )
        except Exception:
            pass

        return 0.0
