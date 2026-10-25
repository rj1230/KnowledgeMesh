from app.agents import graph as graph_module

ABSTENTION = (
    "I couldn't find evidence in the available sources that "
    "establishes an answer to your specific question."
)


def _run_exhausted_validation_graph(monkeypatch, *, failure_stage, thread_id):
    responder_calls = []
    validation_calls = []
    trajectory_inputs = []

    def fake_planner(state):
        return {"route": "conversation", "retrieval_required": False}

    def fake_responder(state):
        responder_calls.append(dict(state))
        return {
            "generation_failed": False,
            "final_answer": "UNSUPPORTED ANSWER THAT MUST NOT ESCAPE",
            "candidate_answer": "UNSUPPORTED ANSWER THAT MUST NOT ESCAPE",
        }

    def fake_citation_check(state):
        validation_calls.append("citation")
        failed = failure_stage == "citation"
        return {
            "citation_valid": not failed,
            "citation_feedback": ["Synthetic citation failure"] if failed else [],
        }

    def fake_grounding_critic(state):
        validation_calls.append("grounding")
        return {
            "is_grounded": False,
            "grounding_valid": False,
            "answer_supported": False,
            "grounding_feedback": ["Synthetic grounding failure"],
        }

    def fake_answer_relevance(state):
        raise AssertionError("Relevance must not run after validation failure.")

    def fake_evaluate_trajectory(state):
        trajectory_inputs.append(dict(state))
        return {
            "trajectory_status": "failed",
            "hop_count": 0,
            "required_hops": 0,
            "failure_taxonomy": ["VALIDATION_FAILURE"],
            "metrics": {"answer_useful": 0.0},
            "evaluation_trace": state.get("evaluation_trace", []),
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "generate_node", fake_responder)
    monkeypatch.setattr(graph_module, "citation_check_node", fake_citation_check)
    monkeypatch.setattr(graph_module, "grounding_critic_node", fake_grounding_critic)
    monkeypatch.setattr(graph_module, "answer_relevance_node", fake_answer_relevance)
    monkeypatch.setattr(graph_module, "evaluate_trajectory", fake_evaluate_trajectory)

    compiled_graph = graph_module.build_graph()
    initial_state = {
        "route": "conversation",
        "retrieval_required": False,
        "original_query": "What does the evidence establish?",
        "current_query": "What does the evidence establish?",
        "final_answer": "UNSUPPORTED ANSWER THAT MUST NOT ESCAPE",
        "candidate_answer": "UNSUPPORTED ANSWER THAT MUST NOT ESCAPE",
        "citation_valid": failure_stage != "citation",
        "is_grounded": failure_stage != "grounding",
        "grounding_valid": failure_stage != "grounding",
        "answer_supported": failure_stage != "grounding",
        "answer_useful": False,
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": graph_module.MAX_ANSWER_REVISIONS,
        "evaluation_trace": [],
    }

    result = compiled_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    return result, responder_calls, validation_calls, trajectory_inputs


def test_compiled_graph_abstains_after_citation_revision_budget_exhausted(monkeypatch):
    result, responders, validations, trajectory_inputs = (
        _run_exhausted_validation_graph(
            monkeypatch,
            failure_stage="citation",
            thread_id="fail-closed-citation-regression",
        )
    )

    assert len(responders) == graph_module.MAX_ANSWER_REVISIONS + 1
    assert validations.count("citation") == len(responders)
    assert "grounding" not in validations
    assert result["final_answer"] == ABSTENTION
    assert result["candidate_answer"] == ABSTENTION
    assert result["citation_valid"] is False
    assert result["answer_supported"] is False
    assert result["is_grounded"] is False
    assert result["trajectory_status"] == "failed"

    events = trajectory_inputs[-1]["evaluation_trace"]
    assert any(
        event.get("step") == "answer_abstention"
        and event.get("status") == "abstained"
        and (event.get("details") or {}).get("reason") == "citation_validation_revision_budget_exhausted"
        for event in events
    )


def test_compiled_graph_abstains_after_grounding_revision_budget_exhausted(monkeypatch):
    result, responders, validations, trajectory_inputs = (
        _run_exhausted_validation_graph(
            monkeypatch,
            failure_stage="grounding",
            thread_id="fail-closed-grounding-regression",
        )
    )

    assert len(responders) == graph_module.MAX_ANSWER_REVISIONS + 1
    assert validations.count("citation") == len(responders)
    assert validations.count("grounding") == len(responders)
    assert result["final_answer"] == ABSTENTION
    assert result["candidate_answer"] == ABSTENTION
    assert result["citation_valid"] is False
    assert result["answer_supported"] is False
    assert result["is_grounded"] is False
    assert result["trajectory_status"] == "failed"

    events = trajectory_inputs[-1]["evaluation_trace"]
    assert any(
        event.get("step") == "answer_abstention"
        and event.get("status") == "abstained"
        and (event.get("details") or {}).get("reason") == "grounding_validation_revision_budget_exhausted"
        for event in events
    )


def test_compiled_graph_fails_trajectory_when_revision_generation_fails(monkeypatch):
    responder_calls = []
    grounding_calls = []
    trajectory_inputs = []

    def fake_planner(state):
        return {
            "route": "conversation",
            "retrieval_required": False,
        }

    def fake_responder(state):
        responder_calls.append(dict(state))

        if len(responder_calls) == 1:
            return {
                "generation_failed": False,
                "final_answer": "Initial answer. [chunk_1]",
                "candidate_answer": "Initial answer. [chunk_1]",
            }

        # Simulate a provider failure during revision. The previous answer
        # remains present, but this generation attempt has failed.
        return {
            "generation_failed": True,
            "final_answer": state.get("final_answer", ""),
            "candidate_answer": state.get("final_answer", ""),
        }

    def fake_citation_check(state):
        return {
            "citation_valid": True,
            "citation_feedback": "",
        }

    def fake_grounding_critic(state):
        grounding_calls.append(dict(state))
        return {
            "is_grounded": False,
            "answer_supported": False,
            "grounding_feedback": ["The answer lacks sufficient evidence."],
        }

    def fake_evaluate_trajectory(state):
        trajectory_inputs.append(dict(state))
        assert state.get("generation_failed") is True
        return {
            "trajectory_status": "failed",
            "hop_count": 0,
            "required_hops": 1,
            "failure_taxonomy": ["GENERATION_FAILURE"],
            "metrics": {"answer_useful": 0.0},
            "evaluation_trace": state.get("evaluation_trace", []),
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "generate_node", fake_responder)
    monkeypatch.setattr(
        graph_module, "citation_check_node", fake_citation_check
    )
    monkeypatch.setattr(
        graph_module, "grounding_critic_node", fake_grounding_critic
    )
    monkeypatch.setattr(
        graph_module, "evaluate_trajectory", fake_evaluate_trajectory
    )

    compiled_graph = graph_module.build_graph()
    result = compiled_graph.invoke(
        {
            "route": "conversation",
            "retrieval_required": False,
            "original_query": "What does the evidence establish?",
            "final_answer": "",
            "citation_valid": True,
            "is_grounded": False,
            "answer_supported": False,
            "answer_useful": False,
            "answer_revision_count": 0,
            "revision_count": 0,
            "max_revisions": graph_module.MAX_ANSWER_REVISIONS,
            "grounding_feedback": [],
            "evaluation_trace": [],
        },
        config={
            "configurable": {
                "thread_id": "revision-generation-failure-regression"
            }
        },
    )

    assert len(responder_calls) == 2
    assert len(grounding_calls) == 1
    assert len(trajectory_inputs) == 1
    assert result["generation_failed"] is True
    assert result["trajectory_status"] == "failed"
    assert result["failure_taxonomy"] == ["GENERATION_FAILURE"]


def test_compiled_graph_passes_critic_revision_prompt_to_next_generation(monkeypatch):
    responder_calls = []
    grounding_calls = []

    critic_prompt = (
        "STRICT GROUNDING REVISION REQUIRED. "
        "Remove the unsupported claim that memory is always required."
    )

    def fake_planner(state):
        return {"route": "conversation", "retrieval_required": False}

    def fake_responder(state):
        responder_calls.append(dict(state))
        if len(responder_calls) == 1:
            return {
                "generation_failed": False,
                "final_answer": "The process uses a trigger. Memory is always required.",
                "candidate_answer": "The process uses a trigger. Memory is always required.",
            }
        return {
            "generation_failed": False,
            "final_answer": "The process uses a trigger. [chunk_1]",
            "candidate_answer": "The process uses a trigger. [chunk_1]",
        }

    def fake_citation_check(state):
        return {"citation_valid": True, "citation_feedback": []}

    def fake_grounding_critic(state):
        grounding_calls.append(dict(state))
        if len(grounding_calls) == 1:
            return {
                "is_grounded": False,
                "grounding_valid": False,
                "answer_supported": False,
                "grounding_feedback": [
                    "Remove the unsupported memory claim."
                ],
                "revision_prompt": critic_prompt,
                "unsupported_atomic_claims": [
                    {"claim": "Memory is always required.", "score": 0.16}
                ],
            }
        return {
            "is_grounded": True,
            "grounding_valid": True,
            "answer_supported": True,
            "grounding_feedback": [],
            "revision_prompt": "",
            "unsupported_atomic_claims": [],
        }

    def fake_answer_relevance(state):
        return {"answer_useful": True}

    def fake_evaluate_trajectory(state):
        return {
            "trajectory_status": "passed",
            "hop_count": 0,
            "required_hops": 0,
            "failure_taxonomy": [],
            "metrics": {"answer_useful": 1.0},
            "evaluation_trace": state.get("evaluation_trace", []),
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "generate_node", fake_responder)
    monkeypatch.setattr(graph_module, "citation_check_node", fake_citation_check)
    monkeypatch.setattr(graph_module, "grounding_critic_node", fake_grounding_critic)
    monkeypatch.setattr(graph_module, "answer_relevance_node", fake_answer_relevance)
    monkeypatch.setattr(graph_module, "evaluate_trajectory", fake_evaluate_trajectory)

    compiled_graph = graph_module.build_graph()
    initial_state = {
        "route": "conversation",
        "retrieval_required": False,
        "original_query": "What does the evidence establish?",
        "current_query": "What does the evidence establish?",
        "final_answer": "",
        "candidate_answer": "",
        "citation_valid": True,
        "is_grounded": True,
        "grounding_valid": True,
        "answer_supported": True,
        "answer_useful": False,
        "answer_revision_count": 0,
        "revision_count": 0,
        "max_revisions": graph_module.MAX_ANSWER_REVISIONS,
        "evaluation_trace": [],
    }

    result = compiled_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "revision-prompt-handoff-regression"}},
    )

    assert len(responder_calls) == 2
    assert len(grounding_calls) == 2
    assert responder_calls[1]["revision_requested"] is True
    assert responder_calls[1]["revision_prompt"] == critic_prompt
    assert "Remove the unsupported memory claim." in str(
        responder_calls[1].get("grounding_feedback")
    )
    assert result["final_answer"] == "The process uses a trigger. [chunk_1]"
