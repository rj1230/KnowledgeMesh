from typing import TypedDict, List, Literal, Optional, Annotated
import operator


class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]

    current_query: str
    original_query: str
    plan: Annotated[List[str], operator.add]
    status: str
    final_answer: str

    # --- Routing ---
    route: Literal["simple", "knowledge"]

    # --- KB retrieval + grading ---
    documents: List[str]
    kb_grade: Literal["good", "weak"]

    # --- Web fallback + grading ---
    web_results: List[str]
    web_grade: Literal["good", "weak"]

    # --- Merge / loop control ---
    context_source: Optional[Literal["kb", "web"]]
    merged_context: str
    retrieval_loops: int

    # --- Grounding + citation verification ---
    claims: List[dict]
    grounding_scores: List[dict]
    is_grounded: bool
    citation_valid: bool
    revision_count: int
