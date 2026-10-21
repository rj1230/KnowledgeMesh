import pytest
from fastapi.testclient import TestClient

import app.guardrails.rails as rails
import app.main as main


@pytest.mark.parametrize(
    ("classifier_result", "expected_text"),
    [
        ("NON_TECHNICAL", "outside"),
        ("AMBIGUOUS", "more technical context"),
    ],
)
def test_coffee_request_is_blocked_before_rag_graph(
    monkeypatch,
    classifier_result,
    expected_text,
):
    monkeypatch.setattr(
        rails,
        "_run_topic_classifier",
        lambda message: classifier_result,
    )

    graph_calls = []

    def forbidden_graph_invoke(*args, **kwargs):
        graph_calls.append((args, kwargs))
        raise AssertionError(
            "RAG graph must not run for a blocked request"
        )

    monkeypatch.setattr(
        main.rag_agent,
        "invoke",
        forbidden_graph_invoke,
    )

    client = TestClient(main.app)
    response = client.post(
        "/query",
        json={"q": "Will you teach me how to make coffee?"},
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["validation_status"] == "blocked"
    assert body["answer_source"] == "guardrails"
    assert body["retrieval_used"] is False
    assert body["web_search_used"] is False
    assert body["sources"] == []
    assert "Retrieval: Skipped" in body["thought_process"]
    assert expected_text in body["answer"].lower()
    assert graph_calls == []
