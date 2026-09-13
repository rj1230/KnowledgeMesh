import logfire
from app.agents.state import AgentState
from app.gateway import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using either Documentation Context or Web Context
    (whichever source the grader picked) AND Conversation History.
    """
    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if state["route"] == "simple":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        source_docs = (
            state["web_results"]
            if state.get("context_source") == "web"
            else state["documents"]
        )

        for i, doc in enumerate(source_docs):
            tagged = f"[chunk_{i}] {doc}\n\n"
            if len(full_context) + len(tagged) < max_context_chars:
                full_context += tagged
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.
        Each context block is tagged with a chunk ID like [chunk_0].
        For every sentence in your answer that uses information from the context,
        cite the chunk ID it came from, in the form [chunk_N], right after that sentence.
        Do not cite a chunk unless that specific sentence is actually supported by it.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            update = {
                "final_answer": content,
                "merged_context": full_context if state["route"] != "simple" else "",
                "messages": [{"role": "assistant", "content": content}],
            }

            if is_cache_hit:
                logfire.info(
                    "⚡ Gateway Cache Hit — response served from Portkey cache."
                )
                update["status"] = "Cache hit — instant response."
                update["plan"] = ["Cache: Hit ⚡"]
            else:
                logfire.info("✅ Response synthesised via LLM.")
                update["status"] = "Response generated."
                # no "plan" key -> reducer has nothing to append

            return update

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
