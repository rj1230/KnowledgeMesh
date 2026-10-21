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
