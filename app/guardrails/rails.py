import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.actions import action

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None
_check_topic_llm: ChatGroq | None = None


@action(name="check_topic")
async def check_topic(context: dict | None = None) -> bool:
    """
    Custom topic-check action. Explicit prompt, explicit yes/no parsing —
    catches off-topic questions that don't closely match a canonical
    example in colang_rules.py. Returns True (on-topic) on any ambiguity
    or LLM error, so a flaky call here fails open rather than blocking
    legitimate enterprise questions.

    Scope note: this prompt's definition of "on-topic" must stay in sync
    with colang_rules.py's YAML_CONTENT instructions and capabilities
    response, and with what planner_node actually routes to RAG (technical
    infra docs, not just HR/policy). A prior version of this prompt only
    mentioned HR/policy topics, which caused legitimate technical questions
    (e.g. "what do you mean by aws") to be misclassified as off-topic.
    """
    if _check_topic_llm is None:
        logfire.warning("⚠️ check_topic called before rails initialised — allowing.")
        return True

    ctx = context or {}
    # NeMo's built-in self_check_input uses "user_message"; some versions/
    # configs surface "last_user_message" instead. Try both — and log which
    # keys were actually present if neither hits, so we can see the real
    # key name instead of silently allowing everything through.
    user_message = ctx.get("user_message") or ctx.get("last_user_message") or ""

    if not user_message:
        logfire.warning(
            f"⚠️ check_topic got no user_message in context (keys={list(ctx.keys())}) — allowing."
        )
        return True

    prompt = (
        "You are a topic classifier for Knowldemina, an enterprise knowledge "
        "assistant that covers company information, policies, HR, benefits, "
        "onboarding, workplace procedures, AND technical infrastructure "
        "documentation — including Kubernetes, networking, cloud platforms "
        "(AWS), Intel-based systems, and related enterprise engineering "
        "topics.\n"
        f'User message: "{user_message}"\n\n'
        "Is this message on-topic for that assistant? "
        "Answer with exactly one word: YES or NO."
    )

    try:
        response = await _check_topic_llm.ainvoke(prompt)
        answer = response.content.strip().upper()
    except Exception as exc:
        logfire.warning(f"⚠️ check_topic LLM call failed: {exc} — allowing.")
        return True

    if answer.startswith("NO"):
        return False
    if answer.startswith("YES"):
        return True

    logfire.warning(f"⚠️ check_topic got unparseable answer: '{answer}' — allowing.")
    return True


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-20b (via Groq) for fast intent classification and
    the check_topic catch-all at the gate.
    """
    global _rails, _check_topic_llm

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY, model="openai/gpt-oss-20b", temperature=0
    )
    _check_topic_llm = guard_llm

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    _rails.register_action(check_topic, "check_topic")
    logfire.info(
        "🛡️ NeMo Guardrails initialised (openai/gpt-oss-20b, check_topic registered)."
    )


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
        except Exception as exc:
            logfire.error(f"🛡️ Guardrails raised, failing open: {exc}")
            return False, None

        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
