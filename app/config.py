import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Required for the pipeline to function at all. Everything else has a safe
# default and is allowed to be unset.
_REQUIRED_ENV_VARS = (
    "GEMINI_API_KEY",
    "QDRANT_CLUSTER_ENDPOINT",
    "QDRANT_API_KEY",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",  # required: web search is load-bearing in the Self-RAG fallback path
)

# Known output dimensions for supported Gemini embedding models, used to
# catch a mismatched EMBEDDING_DIMENSION override before it silently
# corrupts a Qdrant collection.
_KNOWN_GEMINI_EMBEDDING_DIMS = {
    "models/gemini-embedding-2-preview": 3072,
    "models/text-embedding-004": 768,
}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid at startup."""


def _require(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {var_name}. "
            f"Set it in your .env file or the process environment before starting the app."
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
            f"{', '.join(missing)}. Set them in your .env file or the process "
            "environment before starting the app."
        )


_validate_all_required()


class Settings:
    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = _require("GEMINI_API_KEY")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "models/gemini-embedding-2-preview"
    )
    embedding_dimension: int = _require_int("EMBEDDING_DIMENSION", "3072")

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = _require("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = _require("QDRANT_API_KEY")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_rag")

    # --- REASONING ENGINE (GROQ) ---
    # Single source of truth for Groq creds/model — do not add a second,
    # differently-cased copy of these two lines elsewhere in this class.
    GROQ_API_KEY = _require("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY", "")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "")
    PORTKEY_CONFIG_SLUG = os.getenv("PORTKEY_CONFIG_SLUG", "")
    GROQ_SLUG = os.getenv("GROQ_SLUG", "rag")  # primary: @rag/llama-3.3-70b-versatile
    GROQ_SLUG_2 = os.getenv(
        "GROQ_SLUG_2", "brag"
    )  # fallback: @brag/llama-3.1-8b-instant

    # --- WEB SEARCH FALLBACK (TAVILY) ---
    TAVILY_API_KEY = _require("TAVILY_API_KEY")

    # --- SELF-RAG CONTROLS ---
    # Bounds on the grade -> rewrite -> retry cycle, so a stubborn query
    # can't loop indefinitely.
    TOP_K = _require_int("TOP_K", "5")
    MAX_SUPPORT_RETRIES = _require_int(
        "MAX_SUPPORT_RETRIES", "2"
    )  # IsSUP-triggered regeneration
    MAX_RETRIEVAL_REWRITES = _require_int(
        "MAX_RETRIEVAL_REWRITES", "2"
    )  # query rewrites against Qdrant
    MAX_WEB_REWRITES = _require_int(
        "MAX_WEB_REWRITES", "2"
    )  # query rewrites against Tavily

    # --- AUDIT PERSISTENCE ---
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/audit.db")

    @property
    def database_file(self) -> Path:
        p = Path(self.DATABASE_PATH)
        return p if p.is_absolute() else ROOT / p

    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )

    def validate_embedding_dimension(self) -> None:
        """Warn (don't crash) if EMBEDDING_DIMENSION doesn't match the known
        output size for embedding_model. A silent mismatch here is what
        actually breaks Qdrant upserts/queries, so this is checked
        explicitly rather than left to fail deep in the retrieval code."""
        expected = _KNOWN_GEMINI_EMBEDDING_DIMS.get(self.embedding_model)
        if expected is not None and expected != self.embedding_dimension:
            logger.warning(
                "EMBEDDING_DIMENSION=%s does not match the known output "
                "dimension (%s) for embedding_model=%s. Qdrant collection "
                "sizing and vector search will break unless this is "
                "intentional.",
                self.embedding_dimension,
                expected,
                self.embedding_model,
            )

    def safe_summary(self) -> dict:
        """Non-secret snapshot of config for startup logs/health checks.
        Never log the Settings object directly — it holds API keys."""
        return {
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "qdrant_collection": self.QDRANT_COLLECTION,
            "groq_model": self.GROQ_MODEL,
            "groq_fallback_configured": bool(self.GROQ_FALLBACK_API_KEY),
            "portkey_configured": bool(self.PORTKEY_API_KEY),
            "top_k": self.TOP_K,
            "max_support_retries": self.MAX_SUPPORT_RETRIES,
            "max_retrieval_rewrites": self.MAX_RETRIEVAL_REWRITES,
            "max_web_rewrites": self.MAX_WEB_REWRITES,
            "database_file": str(self.database_file),
            "langsmith_tracing": self.LANGSMITH_TRACING,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Settings({self.safe_summary()})"


settings = Settings()
settings.validate_embedding_dimension()

# Bridge LANGSMITH_* naming into the LANGCHAIN_* names LangChain's SDK
# actually reads for automatic tracing.
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGSMITH_TRACING
os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

logger.info("Config loaded: %s", settings.safe_summary())
