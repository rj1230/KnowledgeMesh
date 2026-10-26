# Paste into ui.py where the final answer is rendered.
# Adjust `result` to whatever dict your /query response is stored in.
import streamlit as st

def render_partial_answer_banner(result: dict) -> None:
    removed = result.get("pruned_claims") or []
    if not removed:
        return
    st.info(
        f"Partial answer: {len(removed)} point(s) could not be verified "
        "against the sources and were removed."
    )
    with st.expander("What was removed"):
        for claim in removed:
            st.write(f"- {claim}")

# usage, just above the answer text:
# render_partial_answer_banner(result)
