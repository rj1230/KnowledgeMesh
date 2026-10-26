import importlib.util
import sys
from pathlib import Path

import app.guardrails.rails as rails


ROOT = Path(__file__).resolve().parents[1]
UI_PATH = ROOT / "ui.py"
SPEC = importlib.util.spec_from_file_location(
    "knowledgemesh_inprocess_lifecycle_test_module",
    UI_PATH,
)
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def test_inprocess_backend_enters_and_releases_fastapi_lifespan(monkeypatch):
    import app.main as main

    initialized = []

    def fake_initialize_rails():
        initialized.append(True)
        monkeypatch.setattr(rails, "_rails", object())
        monkeypatch.setattr(rails, "_check_topic_llm", object())

    monkeypatch.setattr(ui, "_secret_or_env", lambda name: None)
    monkeypatch.setattr(main, "get_embedding_dim", lambda: 384)
    monkeypatch.setattr(main, "initialize_rails", fake_initialize_rails)
    monkeypatch.setattr(main, "warmup_hhem", lambda: None)
    monkeypatch.setattr(main.logfire, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_guardrails_initialized", False)

    backend_factory = getattr(ui.get_inprocess_backend, "__wrapped__", None)
    assert backend_factory is not None, (
        "Expected Streamlit's cached function to expose its wrapped function."
    )

    client = backend_factory()
    try:
        assert initialized == [True]
        assert main._guardrails_initialized is True
        assert rails._rails is not None
        assert rails._check_topic_llm is not None

        response = client.get("/health")
        assert response.status_code == 200
    finally:
        ui._release_inprocess_backend(client)
