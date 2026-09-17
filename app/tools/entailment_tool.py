"""
KnowledgeMesh — HHEM Grounding / Factual Consistency Tool

Uses Vectara's HHEMv2 hallucination evaluation model directly through
Transformers rather than SentenceTransformers CrossEncoder.

Why:
    HHEM-2.1/Open provides its own custom model implementation and
    preprocessing/prediction logic. Loading it through
    sentence_transformers.CrossEncoder can fail because the repository
    does not expose the processor metadata expected by that loader.

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
            )

            _model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
            )

            _model.eval()

            logfire.info(
                "HHEMv2 grounding model ready",
                model=MODEL_NAME,
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

    HHEMv2's custom implementation exposes `predict()` and normally
    returns a tensor/array of scores in [0, 1].
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
        ):
            pairs = [(premise, hypothesis)]

            # HHEMv2 exposes a custom predict() method.
            raw_output = model.predict(pairs)

            score = _extract_score(raw_output)
            score = _normalize_score(score)

            logfire.info(
                "HHEMv2 grounding score computed",
                score=round(score, 4),
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
