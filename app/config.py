import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# REQUIRED ENVIRONMENT
# ============================================================

_REQUIRED_ENV_VARS = (
    "QDRANT_CLUSTER_ENDPOINT",
    "QDRANT_API_KEY",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid at startup."""


def _require(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {var_name}. "
            f"Set it in your .env file or the process environment before "
            f"starting the app."
        )
    return value


def _require_int(var_name: str, default: str) -> int:
    raw = os.getenv(var_name, default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"Environment variable {var_name}={raw!r} must be an integer."
        ) from exc


def _validate_all_required() -> None:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            f"{', '.join(missing)}. Set them in your .env file or the "
            "process environment before starting the app."
        )


_validate_all_required()


class Settings:

    # ========================================================
    # EMBEDDINGS
    # ========================================================
    #
    # Canonical embedding configuration is enforced by
    # app.services.retrieval.embedding.
    #
    # Qdrant enterprise_rag currently contains 384-dimensional
    # BGE-small vectors. These values are therefore kept aligned
    # with the canonical embedding service.

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )

    embedding_dimension: int = _require_int(
        "EMBEDDING_DIMENSION",
        "384",
    )

    # ========================================================
    # VECTOR DB (QDRANT)
    # ========================================================

    QDRANT_URL = _require("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = _require("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv(
        "QDRANT_COLLECTION",
        "enterprise_rag",
    )

    # ========================================================
    # REASONING ENGINE (GROQ)
    # ========================================================

    GROQ_API_KEY = _require("GROQ_API_KEY")

    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    GROQ_FALLBACK_API_KEY = os.getenv(
        "GROQ_FALLBACK_API_KEY",
        "",
    )

    # ========================================================
    # LLM GATEWAY (PORTKEY)
    # ========================================================

    PORTKEY_API_KEY = os.getenv(
        "PORTKEY_API_KEY",
        "",
    )

    PORTKEY_CONFIG_SLUG = os.getenv(
        "PORTKEY_CONFIG_SLUG",
        "",
    )

    GROQ_SLUG = os.getenv(
        "GROQ_SLUG",
        "rag",
    )

    GROQ_SLUG_2 = os.getenv(
        "GROQ_SLUG_2",
        "brag",
    )

    # ========================================================
    # WEB SEARCH FALLBACK (TAVILY)
    # ========================================================

    TAVILY_API_KEY = _require(
        "TAVILY_API_KEY",
    )

    # ========================================================
    # SELF-RAG CONTROLS
    # ========================================================

    TOP_K = _require_int(
        "TOP_K",
        "5",
    )

    MAX_SUPPORT_RETRIES = _require_int(
        "MAX_SUPPORT_RETRIES",
        "2",
    )

    MAX_RETRIEVAL_REWRITES = _require_int(
        "MAX_RETRIEVAL_REWRITES",
        "2",
    )

    MAX_WEB_REWRITES = _require_int(
        "MAX_WEB_REWRITES",
        "2",
    )

    # ========================================================
    # AUDIT PERSISTENCE
    # ========================================================

    DATABASE_PATH = os.getenv(
        "DATABASE_PATH",
        "data/audit.db",
    )

    @property
    def database_file(self) -> Path:
        p = Path(self.DATABASE_PATH)

        return (
            p
            if p.is_absolute()
            else ROOT / p
        )

    # ========================================================
    # OBSERVABILITY
    # ========================================================

    LANGSMITH_TRACING = os.getenv(
        "LANGSMITH_TRACING",
        "true",
    )

    LANGSMITH_API_KEY = os.getenv(
        "LANGSMITH_API_KEY",
        "",
    )

    LANGSMITH_PROJECT = os.getenv(
        "LANGSMITH_PROJECT",
        "rag_scale_test",
    )

    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT",
        "https://api.smith.langchain.com",
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate_embedding_dimension(self) -> None:
        """
        Validate the deployment configuration against the
        canonical KnowledgeMesh embedding invariants.
        """

        expected_model = "BAAI/bge-small-en-v1.5"
        expected_dimension = 384

        if self.embedding_model != expected_model:
            raise ConfigError(
                "Embedding configuration mismatch.\n"
                f"Expected model: {expected_model}\n"
                f"Configured model: {self.embedding_model}\n\n"
                "The existing enterprise_rag Qdrant collection uses "
                "the canonical BGE-small embedding model."
            )

        if self.embedding_dimension != expected_dimension:
            raise ConfigError(
                "Embedding dimension configuration mismatch.\n"
                f"Expected dimension: {expected_dimension}\n"
                f"Configured dimension: {self.embedding_dimension}\n\n"
                "The existing enterprise_rag Qdrant collection uses "
                f"{expected_dimension}-dimensional vectors."
            )

    # ========================================================
    # SAFE CONFIG SUMMARY
    # ========================================================

    def safe_summary(self) -> dict:
        """
        Return a non-secret configuration snapshot.

        API keys are intentionally never included.
        """

        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "qdrant_collection": self.QDRANT_COLLECTION,
            "groq_model": self.GROQ_MODEL,
            "groq_fallback_configured": bool(
                self.GROQ_FALLBACK_API_KEY
            ),
            "portkey_configured": bool(
                self.PORTKEY_API_KEY
            ),
            "top_k": self.TOP_K,
            "max_support_retries": self.MAX_SUPPORT_RETRIES,
            "max_retrieval_rewrites": self.MAX_RETRIEVAL_REWRITES,
            "max_web_rewrites": self.MAX_WEB_REWRITES,
            "database_file": str(self.database_file),
            "langsmith_tracing": self.LANGSMITH_TRACING,
        }

    def __repr__(self) -> str:
        return f"Settings({self.safe_summary()})"


settings = Settings()

settings.validate_embedding_dimension()


# ============================================================
# LANGCHAIN TRACING BRIDGE
# ============================================================

os.environ["LANGCHAIN_TRACING_V2"] = (
    settings.LANGSMITH_TRACING
)

os.environ["LANGCHAIN_API_KEY"] = (
    settings.LANGSMITH_API_KEY
)

os.environ["LANGCHAIN_PROJECT"] = (
    settings.LANGSMITH_PROJECT
)

os.environ["LANGCHAIN_ENDPOINT"] = (
    settings.LANGSMITH_ENDPOINT
)


logger.info(
    "Config loaded: %s",
    settings.safe_summary(),
)
