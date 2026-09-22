from __future__ import annotations

from unittest.mock import patch

from app.gateway.client import (
    PORTKEY_FALLBACK_MODEL,
    PORTKEY_PRIMARY_MODEL,
    create_chat_completion,
)


def _response():
    return {
        "choices": [
            {
                "message": {
                    "content": "fallback success",
                }
            }
        ]
    }


def test_429_retries_primary_then_uses_fallback():
    """
    Verify:

        primary -> 429
        primary -> 429
        fallback -> success

    The important assertion is that the fallback model is actually
    called after the primary model exhausts its retry budget.
    """

    calls = []

    def fake_completion(*, model, messages, temperature=0.1, **kwargs):
        calls.append(model)

        if model == PORTKEY_PRIMARY_MODEL:
            error = RuntimeError("429 Too Many Requests")
            error.status_code = 429
            raise error

        return _response()

    with patch(
        "app.gateway.client._create_completion",
        side_effect=fake_completion,
    ):
        response = create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "test fallback",
                }
            ],
            temperature=0.1,
        )

    assert response == _response()

    assert calls == [
        PORTKEY_PRIMARY_MODEL,
        PORTKEY_PRIMARY_MODEL,
        PORTKEY_FALLBACK_MODEL,
    ]


def test_success_does_not_call_fallback():
    """
    Verify that a successful primary request does not unnecessarily
    invoke the fallback model.
    """

    calls = []

    def fake_completion(*, model, messages, temperature=0.1, **kwargs):
        calls.append(model)
        return _response()

    with patch(
        "app.gateway.client._create_completion",
        side_effect=fake_completion,
    ):
        response = create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "primary success",
                }
            ],
            temperature=0.1,
        )

    assert response == _response()
    assert calls == [PORTKEY_PRIMARY_MODEL]


def test_non_retryable_error_does_not_retry_primary():
    """
    A non-transient error should immediately move to fallback rather
    than repeatedly retrying the same primary request.
    """

    calls = []

    def fake_completion(*, model, messages, temperature=0.1, **kwargs):
        calls.append(model)

        if model == PORTKEY_PRIMARY_MODEL:
            error = RuntimeError("400 Bad Request")
            error.status_code = 400
            raise error

        return _response()

    with patch(
        "app.gateway.client._create_completion",
        side_effect=fake_completion,
    ):
        response = create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "fallback after non retryable",
                }
            ],
            temperature=0.1,
        )

    assert response == _response()

    assert calls == [
        PORTKEY_PRIMARY_MODEL,
        PORTKEY_FALLBACK_MODEL,
    ]
