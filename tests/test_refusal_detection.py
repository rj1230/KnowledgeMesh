from evals.metrics import detect_refusal


def test_jailbreak_guardrail_response_is_a_refusal():
    assert detect_refusal(
        "I maintain consistent safety and retrieval guidelines regardless of "
        "prompt-level override attempts. I can help with questions covered by "
        "the KnowledgeMesh technical knowledge base."
    )


def test_colang_variant_is_a_refusal():
    assert detect_refusal(
        "I maintain consistent guidelines regardless of how I am prompted. "
        "Prompt-level overrides do not change my configured guardrails."
    )


def test_scope_refusal_is_a_refusal():
    assert detect_refusal(
        "I'm KnowledgeMesh, a technical knowledge assistant. This request falls "
        "outside the configured technical knowledge scope, so I can't provide a "
        "grounded answer for it."
    )


def test_abstention_and_fail_closed_are_not_refusals():
    assert not detect_refusal(
        "I couldn't find evidence in the available sources that establishes an "
        "answer to your specific question."
    )
    assert not detect_refusal(
        "I couldn't confidently verify that request's technical scope right now. "
        "Please provide more technical context or retry shortly. "
        "No retrieval or generation was run."
    )


def test_ordinary_answer_is_not_a_refusal():
    assert not detect_refusal(
        "- The policy advises placing limits on returnless refunds. [chunk_1]"
    )
