import logfire
from app.agents.state import AgentState


def citation_check_node(state: AgentState):
    """
    Verifies every citation in the answer points to a chunk that actually
    exists in the retrieved context — catches invented chunk IDs, distinct
    from the semantic entailment check done in grounding_critic.
    """
    if state["route"] == "simple":
        return {"citation_valid": True}

    context = state.get("merged_context", "")
    claims = state.get("claims", [])

    with logfire.span("📎 Citation Check"):
        if not claims:
            logfire.warning("No citations found in the answer.")
            valid = False
        else:
            valid = True
            for claim in claims:
                marker = f"[chunk_{claim['cited_chunk_id']}]"
                if marker not in context:
                    logfire.warning(
                        f"Citation {marker} does not exist in retrieved context"
                    )
                    valid = False
                    break

        logfire.info(f"Citation check: {'passed' if valid else 'failed'}")

    return {
        "citation_valid": valid,
        "revision_count": state.get("revision_count", 0) + 1,
    }
