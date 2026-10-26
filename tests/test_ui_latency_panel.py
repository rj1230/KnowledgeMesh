import importlib.util
from pathlib import Path

UI_PATH = Path(__file__).resolve().parents[1] / "ui.py"
SPEC = importlib.util.spec_from_file_location("knowledgemesh_latency_ui_test_module", UI_PATH)
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def test_latency_panel_shows_grounding_and_request_hhem_without_double_counting():
    trace = {
        "backend_latency_ms": 1000,
        "latency": 1.2,
        "retrieval_latency_ms": 100,
        "rerank_latency_ms": 50,
        "grader_latency_ms": 100,
        "generation_latency_ms": 250,
        "grounding_ms": 400,
        "request_hhem_total_ms": 300,
        "request_hhem_call_count": 4,
    }

    rendered = ui.render_latency_panel(trace)

    assert "Grounding critic pass" in rendered
    assert "400" in rendered
    assert "HHEM across request" in rendered
    assert "300" in rendered
    assert "4 HHEM calls" in rendered

    # The bar continues to account for the four existing stages only:
    # 100 + 50 + 100 + 250 = 500 ms; remainder = 500 ms.
    assert "Not itemised" in rendered
    assert "500" in rendered


def test_latency_panel_works_when_grounding_diagnostics_are_missing():
    rendered = ui.render_latency_panel({
        "backend_latency_ms": 500,
        "retrieval_latency_ms": 100,
        "rerank_latency_ms": 50,
        "grader_latency_ms": 50,
        "generation_latency_ms": 100,
    })

    assert rendered
    assert "Grounding critic pass" not in rendered
    assert "HHEM across request" not in rendered

def test_parse_response_preserves_grounding_and_hhem_metrics():
    data = {
        "answer": "The answer is supported.",
        "status": "Completed successfully.",
        "retrieval_used": True,
        "citation_valid": True,
        "is_grounded": True,
        "latency_ms": 1000,
        "retrieval_latency_ms": 100,
        "rerank_latency_ms": 50,
        "grader_latency_ms": 100,
        "generation_latency_ms": 250,
        "grounding_ms": 400,
        "hhem_call_count": 3,
        "hhem_total_ms": 275.0,
        "request_hhem_call_count": 4,
        "request_hhem_total_ms": 300.0,
        "grounding_pass_count": 2,
    }

    answer, trace, completion_state, completion_label = ui.parse_response(
        data, elapsed=1.2, http_status=200
    )

    assert answer == "The answer is supported."
    assert trace["grounding_ms"] == 400
    assert trace["request_hhem_total_ms"] == 300.0
    assert trace["request_hhem_call_count"] == 4
    assert trace["grounding_pass_count"] == 2

    rendered = ui.render_latency_panel(trace)
    assert "Grounding critic pass" in rendered
    assert "HHEM across request" in rendered
    assert "4 HHEM calls" in rendered
