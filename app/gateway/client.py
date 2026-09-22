from __future__ import annotations

import random
import time
from typing import Any, Callable, Optional, TypeVar

from langchain_openai import ChatOpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, Portkey, createHeaders

from app.config import settings


# ============================================================
# Portkey Configuration
# ============================================================

PORTKEY_CONFIG_SLUG = settings.PORTKEY_CONFIG_SLUG

# Primary / fallback models.
PORTKEY_PRIMARY_MODEL = "@rag/openai/gpt-oss-120b"
PORTKEY_FALLBACK_MODEL = "@rag/openai/gpt-oss-20b"

# Application-level bounded retry policy.
#
# Portkey may already retry/fallback internally, but KnowledgeMesh
# also needs a deterministic application-level safety net.
LLM_MAX_RETRIES = 1
LLM_BASE_BACKOFF_SECONDS = 1.0
LLM_MAX_BACKOFF_SECONDS = 4.0

# Errors which should normally trigger retry/fallback.
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Rate Limited
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}


# ============================================================
# Native Portkey Client
# ============================================================

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_SLUG,
)


# ============================================================
# Exceptions
# ============================================================


class LLMGatewayError(RuntimeError):
    """Base error for KnowledgeMesh LLM gateway failures."""


class LLMRateLimitError(LLMGatewayError):
    """Raised when the LLM provider remains rate limited."""


class LLMProviderError(LLMGatewayError):
    """Raised when the provider/gateway remains unavailable."""


# ============================================================
# Error Classification
# ============================================================


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """
    Best-effort extraction of an HTTP status code from SDK errors.

    Portkey/OpenAI-compatible exceptions can expose the status in
    different attributes depending on SDK version.
    """

    for attr in (
        "status_code",
        "http_status",
        "status",
        "code",
    ):
        value = getattr(exc, attr, None)

        if isinstance(value, int):
            return value

        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)

    if response is not None:
        value = getattr(response, "status_code", None)

        if isinstance(value, int):
            return value

    return None


def _is_retryable_error(exc: BaseException) -> bool:
    """
    Determine whether an LLM error is transient and should trigger
    bounded retry/fallback behavior.
    """

    status_code = _extract_status_code(exc)

    if status_code in RETRYABLE_STATUS_CODES:
        return True

    error_text = str(exc).lower()

    retryable_terms = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "connection reset",
        "connection error",
        "server error",
    )

    return any(term in error_text for term in retryable_terms)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when an exception appears to represent rate limiting."""

    status_code = _extract_status_code(exc)

    if status_code == 429:
        return True

    error_text = str(exc).lower()

    return (
        "rate limit" in error_text
        or "rate_limit" in error_text
        or "too many requests" in error_text
        or "429" in error_text
    )


# ============================================================
# Backoff
# ============================================================


def _backoff_seconds(attempt: int) -> float:
    """
    Exponential backoff with small jitter.

    attempt=0 -> ~1 sec
    attempt=1 -> ~2 sec
    """

    exponential = LLM_BASE_BACKOFF_SECONDS * (2**attempt)

    bounded = min(
        exponential,
        LLM_MAX_BACKOFF_SECONDS,
    )

    jitter = random.uniform(0.0, 0.25)

    return bounded + jitter


# ============================================================
# Native Portkey Completion
# ============================================================


def _create_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    **kwargs: Any,
):
    """
    Execute one native Portkey completion.

    This function intentionally performs exactly ONE provider call.
    Retry/fallback decisions belong to the wrapper below.
    """

    return portkey_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


# ============================================================
# Production LLM Completion
# ============================================================


T = TypeVar("T")


def create_chat_completion(
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    primary_model: str = PORTKEY_PRIMARY_MODEL,
    fallback_model: str = PORTKEY_FALLBACK_MODEL,
    max_retries: int = LLM_MAX_RETRIES,
    on_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
    **kwargs: Any,
):
    """
    Production-safe LLM completion.

    Execution policy:

        primary
          ↓
        retry primary once on transient failure
          ↓
        fallback model
          ↓
        retry fallback once on transient failure
          ↓
        controlled LLMGatewayError

    This prevents an LLM provider failure from creating an
    unbounded retry loop inside the graph.

    Returns:
        Native Portkey completion response.
    """

    models = [
        ("primary", primary_model),
        ("fallback", fallback_model),
    ]

    last_error: Optional[BaseException] = None

    for provider_role, model in models:
        for attempt in range(max_retries + 1):
            try:
                if on_event is not None:
                    on_event(
                        "llm_attempt",
                        {
                            "role": provider_role,
                            "model": model,
                            "attempt": attempt + 1,
                            "max_attempts": max_retries + 1,
                        },
                    )

                response = _create_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )

                if on_event is not None:
                    on_event(
                        "llm_success",
                        {
                            "role": provider_role,
                            "model": model,
                            "attempt": attempt + 1,
                        },
                    )

                return response

            except Exception as exc:
                last_error = exc

                retryable = _is_retryable_error(exc)
                rate_limited = _is_rate_limit_error(exc)

                if on_event is not None:
                    on_event(
                        "llm_failure",
                        {
                            "role": provider_role,
                            "model": model,
                            "attempt": attempt + 1,
                            "retryable": retryable,
                            "rate_limited": rate_limited,
                            "status_code": _extract_status_code(exc),
                            "error_type": type(exc).__name__,
                        },
                    )

                # Non-transient errors should not be repeatedly retried.
                if not retryable:
                    break

                # Retry current model if retry budget remains.
                if attempt < max_retries:
                    delay = _backoff_seconds(attempt)

                    if on_event is not None:
                        on_event(
                            "llm_backoff",
                            {
                                "role": provider_role,
                                "model": model,
                                "delay_seconds": round(delay, 3),
                                "attempt": attempt + 1,
                            },
                        )

                    time.sleep(delay)

        # Current model exhausted.
        #
        # Continue to fallback model rather than allowing the
        # provider exception to immediately terminate the graph.
        if on_event is not None:
            on_event(
                "llm_fallback",
                {
                    "from_role": provider_role,
                    "next_role": "fallback",
                },
            )

    # --------------------------------------------------------
    # Controlled final failure
    # --------------------------------------------------------

    if last_error is not None:
        if _is_rate_limit_error(last_error):
            raise LLMRateLimitError(
                "All configured LLM attempts were rate limited."
            ) from last_error

        raise LLMProviderError("All configured LLM attempts failed.") from last_error

    raise LLMGatewayError("LLM gateway failed without returning an exception.")


# ============================================================
# LangChain LLM
# ============================================================


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Return a Portkey-backed LangChain ChatOpenAI instance.

    This remains available for existing LangChain-based nodes.

    Native Portkey calls that require deterministic application-level
    fallback should use create_chat_completion().
    """

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PORTKEY_PRIMARY_MODEL,
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG_SLUG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        ),
    )


# ============================================================
# Portkey Cache Status
# ============================================================


def extract_cache_status(response) -> str:
    """
    Extract the Portkey cache status from a response.

    Returns:
        HIT  -> response served from Portkey cache
        MISS -> response was not served from cache
    """

    for attr in (
        "_raw_response",
        "_response",
        "_http_response",
    ):
        raw = getattr(response, attr, None)

        if raw is not None:
            headers = getattr(raw, "headers", {})

            status = headers.get(
                "x-portkey-cache-status",
                "",
            )

            if status:
                return status.upper()

    return "MISS"
