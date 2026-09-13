import re
import logfire
from app.agents.state import AgentState
from app.tools.entailment_tool import score_entailment

CLAIM_PATTERN = re.compile(r"(.+?\.)\s*\[chunk_(\d+)\]", re.DOTALL)


def _extract_claims(answer: str) -> list[dict]:
    claims = []
    for match in CLAIM_PATTERN.finditer(answer):
        claims.append(
            {"text": match.group(1).strip(), "cited_chunk_id": int(match.group(2))}
        )
    return claims


def _get_chunk_text(merged_context: str, chunk_id: int) -> str:
    marker = f"[chunk_{chunk_id}]"
    idx = merged_context.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = merged_context.find("[chunk_", start)
    return merged_context[start : end if end != -1 else None].strip()


def grounding_critic_node(state: AgentState):
    """
    Decomposes the answer into cited claims and checks each one against the
    chunk it claims to be supported by. An answer is grounded only if every
    claim is entailed by its cited chunk.
    """
    if state["route"] == "simple":
        return {"claims": [], "grounding_scores": [], "is_grounded": True}

    answer = state.get("final_answer", "")
    context = state.get("merged_context", "")

    with logfire.span("🔬 Grounding Critic"):
        claims = _extract_claims(answer)
        scores = []
        for claim in claims:
            chunk_text = _get_chunk_text(context, claim["cited_chunk_id"])
            score = score_entailment(premise=chunk_text, hypothesis=claim["text"])
            scores.append(
                {
                    "claim": claim["text"],
                    "cited_chunk_id": claim["cited_chunk_id"],
                    "score": score,
                    "supported": score >= 0.5,
                }
            )

        is_grounded = bool(scores) and all(s["supported"] for s in scores)
        logfire.info(
            f"Grounding check: {sum(s['supported'] for s in scores)}/{len(scores)} claims supported"
        )

    return {"claims": claims, "grounding_scores": scores, "is_grounded": is_grounded}
