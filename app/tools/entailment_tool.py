from sentence_transformers import CrossEncoder

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("vectara/hallucination_evaluation_model")
    return _model


def score_entailment(premise: str, hypothesis: str) -> float:
    """
    Returns a 0-1 score for whether `premise` supports `hypothesis`.
    Higher = more grounded / less hallucinated.
    """
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    model = _get_model()
    score = model.predict([(premise, hypothesis)])[0]
    return float(score)
