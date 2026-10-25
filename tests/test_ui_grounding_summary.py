import importlib.util
from pathlib import Path

import pytest


UI_PATH = Path(__file__).resolve().parents[1] / "ui.py"
SPEC = importlib.util.spec_from_file_location("knowledgemesh_ui_test_module", UI_PATH)
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def test_quality_summary_keeps_min_entailment_separate_from_grounding():
    rendered = ui.render_quality_summary(
        {
            "support_score": 0.92,
            "is_grounded": False,
            "answer_supported": False,
            "citation_valid": True,
        }
    )

    assert "Min entailment" in rendered
    assert "0.92" in rendered
    assert "Not grounded" in rendered
    assert "Not supported" in rendered


def test_quality_summary_shows_grounded_status_and_min_entailment():
    rendered = ui.render_quality_summary(
        {
            "support_score": 0.92,
            "is_grounded": True,
            "answer_supported": True,
            "citation_valid": True,
        }
    )

    assert "Min entailment" in rendered
    assert "0.92" in rendered
    assert "Grounded" in rendered
    assert "Not grounded" not in rendered
