"""Focused tests for thread-safe embedding model initialization."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace
import sys

import pytest

from app.services.retrieval import embedding


@pytest.fixture
def reset_embedding_state(monkeypatch):
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setattr(embedding, "_embedding_dimension", None)


def test_concurrent_first_calls_initialize_model_once(
    monkeypatch, reset_embedding_state
):
    init_lock = Lock()
    init_count = 0
    workers = 8
    start_barrier = Barrier(workers)

    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            nonlocal init_count
            with init_lock:
                init_count += 1
            # Widen the race window to exercise concurrent first calls.
            sleep(0.05)

        def get_sentence_embedding_dimension(self):
            return embedding.CANONICAL_EMBEDDING_DIMENSION

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    def load_after_barrier():
        start_barrier.wait(timeout=5)
        return embedding._load_model()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        models = list(executor.map(lambda _: load_after_barrier(), range(workers)))

    assert init_count == 1
    assert all(model is models[0] for model in models)
    assert embedding._model is models[0]
    assert embedding._embedding_dimension == embedding.CANONICAL_EMBEDDING_DIMENSION


def test_invalid_dimension_fails_closed(
    monkeypatch, reset_embedding_state
):
    invalid_dimension = embedding.CANONICAL_EMBEDDING_DIMENSION + 1

    class WrongDimensionModel:
        def __init__(self, model_name, device=None):
            pass

        def get_sentence_embedding_dimension(self):
            return invalid_dimension

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=WrongDimensionModel),
    )

    with pytest.raises(RuntimeError, match="unexpected dimension"):
        embedding._load_model()

    assert embedding._model is None
    assert embedding._embedding_dimension is None


def test_modern_dimension_api_is_preferred(
    monkeypatch, reset_embedding_state
):
    class ModernApiModel:
        def __init__(self, model_name, device=None):
            self.modern_calls = 0
            self.legacy_calls = 0

        def get_embedding_dimension(self):
            self.modern_calls += 1
            return embedding.CANONICAL_EMBEDDING_DIMENSION

        def get_sentence_embedding_dimension(self):
            self.legacy_calls += 1
            return -1

    instances = []

    class FakeSentenceTransformer(ModernApiModel):
        def __init__(self, model_name, device=None):
            super().__init__(model_name, device)
            instances.append(self)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    model = embedding._load_model()

    assert model is instances[0]
    assert model.modern_calls == 1
    assert model.legacy_calls == 0
    assert embedding._embedding_dimension == embedding.CANONICAL_EMBEDDING_DIMENSION


def test_legacy_dimension_api_fallback(
    monkeypatch, reset_embedding_state
):
    class LegacyApiModel:
        def __init__(self, model_name, device=None):
            pass

        def get_sentence_embedding_dimension(self):
            return embedding.CANONICAL_EMBEDDING_DIMENSION

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=LegacyApiModel),
    )

    model = embedding._load_model()

    assert isinstance(model, LegacyApiModel)
    assert embedding._embedding_dimension == embedding.CANONICAL_EMBEDDING_DIMENSION


def test_validate_embedding_accepts_finite_vector():
    vector = [0.25] * embedding.CANONICAL_EMBEDDING_DIMENSION

    result = embedding._validate_embedding(vector)

    assert len(result) == embedding.CANONICAL_EMBEDDING_DIMENSION
    assert all(value == 0.25 for value in result)


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_validate_embedding_rejects_non_finite_values(invalid_value):
    vector = [0.0] * embedding.CANONICAL_EMBEDDING_DIMENSION
    vector[-1] = invalid_value

    with pytest.raises(RuntimeError, match="non-finite"):
        embedding._validate_embedding(vector)


def test_validate_embedding_rejects_wrong_dimension():
    vector = [0.0] * (embedding.CANONICAL_EMBEDDING_DIMENSION - 1)

    with pytest.raises(RuntimeError, match="dimension mismatch"):
        embedding._validate_embedding(vector)
