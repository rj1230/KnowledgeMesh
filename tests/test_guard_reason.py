from app.guardrails import rails


def test_reason_is_classified_by_the_modules_own_messages():
    assert rails.classify_guard_reason(rails._JAILBREAK_RESPONSE) == "jailbreak"
    assert rails.classify_guard_reason(rails._NON_TECHNICAL_RESPONSE) == "out_of_scope"
    assert rails.classify_guard_reason(rails._AMBIGUOUS_RESPONSE) == "scope_unverified"
    assert rails.classify_guard_reason("Hello! I'm KnowledgeMesh.") == "small_talk"
    assert rails.classify_guard_reason(None) == "small_talk"
