from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# ============================================================
# Portkey Configuration
# ============================================================

PORTKEY_CONFIG_SLUG = settings.PORTKEY_CONFIG_SLUG

# Primary model configured in Portkey.
# Portkey itself handles fallback to the secondary target.
PORTKEY_PRIMARY_MODEL = "@rag/openai/gpt-oss-120b"


# ============================================================
# Native Portkey Client
# ============================================================

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=PORTKEY_CONFIG_SLUG,
)


# ============================================================
# LangChain LLM
# ============================================================


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Return a Portkey-backed LangChain ChatOpenAI instance.

    Portkey is responsible for:
        - Provider routing
        - Fallback
        - Retry
        - Caching
        - Gateway-level observability

    The active Portkey configuration is controlled by
    PORTKEY_CONFIG_SLUG in the application settings.
    """

    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=PORTKEY_PRIMARY_MODEL,
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


# ============================================================
# Portkey Cache Status
# ============================================================


def extract_cache_status(response) -> str:
    """
    Extract the Portkey cache status from a response.

    Returns:
        HIT  -> response served from Portkey cache
        MISS -> response was not served from cache
    """

    for attr in (
        "_raw_response",
        "_response",
        "_http_response",
    ):
        raw = getattr(response, attr, None)

        if raw is not None:
            headers = getattr(raw, "headers", {})

            status = headers.get(
                "x-portkey-cache-status",
                "",
            )

            if status:
                return status.upper()

    return "MISS"
