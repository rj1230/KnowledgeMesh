# app/grounding/claim_grounding.py

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SupportType(str, Enum):
    DIRECT = "DIRECT"
    INFERRED = "INFERRED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    support_type: SupportType = SupportType.UNSUPPORTED
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str | None = None
    confidence: float = 0.0


class AtomicClaimsResponse(BaseModel):
    claims: list[AtomicClaim]


async def extract_atomic_claims(
    answer: str,
    llm_client,
) -> list[AtomicClaim]:
    prompt = f"""
Split the answer below into atomic factual claims.

Rules:
- One claim must contain one independently verifiable assertion.
- Keep numbers, dates, names, conditions, and qualifications.
- Do not create claims for headings, questions, or opinions.
- Preserve citation markers attached to each claim.
- Do not add facts.
- Return an empty list if there are no factual claims.

Return JSON matching this schema:
{{
  "claims": [
    {{
      "claim_id": "c-001",
      "text": "...",
      "citation_ids": []
    }}
  ]
}}

Answer:
{answer}
"""

    result = await llm_client.generate_structured(
        prompt=prompt,
        schema=AtomicClaimsResponse,
    )

    if isinstance(result, AtomicClaimsResponse):
        return result.claims

    if isinstance(result, dict):
        parsed = AtomicClaimsResponse.model_validate(result)
        return parsed.claims

    raise TypeError(f"Unexpected structured-output type: {type(result).__name__}")
