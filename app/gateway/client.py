from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Portkey saved configuration
PORTKEY_CONFIG_SLUG = settings.PORTKEY_CONFIG_SLUG


# Native Portkey client
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_SLUG,
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI.

    Portkey acts as the LLM gateway while exposing an
    OpenAI-compatible API endpoint to LangChain.

    Routing, fallback, retry, and other gateway behavior
    are controlled by the saved Portkey configuration.
    """

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/llama-3.3-70b-versatile",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG_SLUG,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production",
            },
        ),
    )


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey response headers.
    Returns MISS if the header cannot be found.
    """

    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)

        if raw is not None:
            status = getattr(raw, "headers", {}).get(
                "x-portkey-cache-status",
                "",
            )

            if status:
                return status.upper()

    return "MISS"
