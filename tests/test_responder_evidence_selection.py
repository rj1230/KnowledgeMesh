from app.agents.nodes import responder


def test_approved_generation_documents_are_used(monkeypatch):
    monkeypatch.setattr(
        responder.logfire, "warning", lambda *a, **k: None
    )

    approved = [{"id": "approved-1"}]
    raw_web = [{"id": "unapproved-web-1"}]

    result = responder._get_documents_for_generation(
        {
            "grader_status": "success",
            "grading_mode": "graded",
            "generation_documents": approved,
            "web_search_used": True,
            "web_documents": raw_web,
        }
    )

    assert result == approved


def test_unapproved_web_documents_are_not_fallback_evidence(monkeypatch):
    monkeypatch.setattr(
        responder.logfire, "warning", lambda *a, **k: None
    )

    result = responder._get_documents_for_generation(
        {
            "grader_status": "success",
            "grading_mode": "graded",
            "generation_documents": [],
            "web_search_used": True,
            "web_documents": [{"id": "unapproved-web-1"}],
        }
    )

    assert result == []


def test_grader_failure_blocks_stale_generation_documents(monkeypatch):
    monkeypatch.setattr(
        responder.logfire, "warning", lambda *a, **k: None
    )

    result = responder._get_documents_for_generation(
        {
            "grader_status": "failed",
            "grading_mode": "unavailable",
            "generation_documents": [{"id": "stale-1"}],
            "web_search_used": True,
            "web_documents": [{"id": "unapproved-web-1"}],
        }
    )

    assert result == []

def test_revision_prompt_is_included_in_generation_prompt():
    revision_prompt = (
        "STRICT GROUNDING REVISION REQUIRED. "
        "Remove the unsupported claim that memory is always required."
    )

    prompt = responder._build_generation_prompt(
        query="What does the evidence establish?",
        technical_context="[chunk_1] The evidence describes a trigger and a stopping rule.",
        conversation_history="",
        previous_answer=(
            "The process uses a trigger. "
            "Memory is always required. [chunk_1]"
        ),
        revision_feedback=[
            "Remove the unsupported memory claim."
        ],
        revision_prompt=revision_prompt,
        conversational=False,
        revision_count=1,
    )

    assert revision_prompt in prompt
    assert "Remove the unsupported memory claim." in prompt
    assert "PREVIOUS ANSWER:" in prompt
    assert "RETRIEVED EVIDENCE:" in prompt
    assert "Do not introduce new facts" in prompt
    assert "If a factual claim is unsupported, remove it." in prompt


def test_revision_prompt_helper_preserves_claim_specific_instructions():
    revision_prompt = (
        "STRICT GROUNDING REVISION REQUIRED. "
        "Remove the unsupported claim that memory is always required."
    )

    result = responder._get_revision_prompt(
        {"revision_prompt": revision_prompt}
    )

    assert result == revision_prompt

def test_revision_prompt_is_included_in_generation_prompt():
    revision_prompt = (
        "STRICT GROUNDING REVISION REQUIRED. "
        "Remove the unsupported claim that memory is always required."
    )

    prompt = responder._build_generation_prompt(
        query="What does the evidence establish?",
        technical_context="[chunk_1] The evidence describes a trigger and a stopping rule.",
        conversation_history="",
        previous_answer=(
            "The process uses a trigger. "
            "Memory is always required. [chunk_1]"
        ),
        revision_feedback=[
            "Remove the unsupported memory claim."
        ],
        revision_prompt=revision_prompt,
        conversational=False,
        revision_count=1,
    )

    assert revision_prompt in prompt
    assert "Remove the unsupported memory claim." in prompt
    assert "PREVIOUS ANSWER:" in prompt
    assert "RETRIEVED EVIDENCE:" in prompt
    assert "Do not introduce new facts" in prompt
    assert "If a factual claim is unsupported, remove it." in prompt


def test_revision_prompt_helper_preserves_claim_specific_instructions():
    revision_prompt = (
        "STRICT GROUNDING REVISION REQUIRED. "
        "Remove the unsupported claim that memory is always required."
    )

    result = responder._get_revision_prompt(
        {"revision_prompt": revision_prompt}
    )

    assert result == revision_prompt
