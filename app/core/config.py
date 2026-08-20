# app/core/config.py

"""
Configuration Management
Centralized settings with environment variable support
"""

import torch
from functools import cached_property, lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration with tunable parameters"""

    # =========================
    # Device Configuration (GPU/CPU)
    # =========================
    # Two 2GB fp32 models oversubscribe a 4GB card, and concurrent calls then evict
    # each other instead of computing. One at a time is dramatically faster; raise
    # this only on a card that fits both models with headroom.
    INFERENCE_CONCURRENCY: int = 1

    # fp16 halves resident model size and is faster on any Ampere-or-later GPU.
    # Ignored on CPU, where half precision is emulated and slower.
    USE_FP16_ON_GPU: bool = True

    @cached_property
    def MODEL_DTYPE(self):
        """Half precision on GPU, full precision on CPU."""
        if self.DEVICE == "cuda" and self.USE_FP16_ON_GPU:
            return torch.float16
        return torch.float32

    @cached_property
    def DEVICE(self) -> str:
        """Auto-detect GPU availability (Cached, runs once)"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Hardware Accelerator Detected: {device.upper()}")

        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning(
                "⚠️ Running on CPU. For heavy RAG workloads, a GPU is recommended."
            )
        return device

    # =========================
    # Model Caching
    # =========================
    HF_HOME: str = "/app/.cache/huggingface"  # Cache directory
    MODEL_CACHE_DIR: str = "/app/model_cache"

    # =========================
    # API Keys
    # =========================
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # =========================
    # Infrastructure
    # =========================
    REDIS_URL: str = "redis://localhost:6379/0"
    # Wildcard origins are invalid with credentials enabled; list the frontend explicitly
    CORS_ORIGINS: list = ["http://localhost:8501", "http://127.0.0.1:8501"]
    LOG_LEVEL: str = "INFO"
    # Debug logging includes document content and tool arguments; off by default
    DEBUG_MODE: bool = False

    # =========================
    # LLM Configuration
    # =========================
    # "provider:model" routes explicitly; a bare name goes to OpenAI. Both must appear
    # in SUPPORTED_MODELS — /chat and /models/select reject anything else.
    DEFAULT_MODEL: str = "gemini:gemini-3.6-flash"
    # Deliberately the other provider: a fallback sharing the primary's key and quota dies
    # with it. A Gemini day-quota wall took out both halves of a Gemini-to-Gemini pair.
    FALLBACK_MODEL: str = "gpt-5-mini"
    # Zero, because the same question asked twice should give the same answer. At 0.1 it
    # did not: one run took "typical order size" as the median value and the next as the
    # median unit count, one run summed two files' totals and the next correctly refused,
    # one applied a subset filter and the next reported the unfiltered figure. Every
    # number was right in both — the variance was in interpretation and rule-following,
    # which is exactly what sampling noise costs an agent that must obey a dozen rules
    # per turn. Raise it only if you want variety over repeatability.
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 8192
    AGENT_MAX_TURNS: int = 10

    # The allow-list /models serves and both endpoints validate against, so a typo is
    # rejected rather than answered by the fallback. Routing is by prefix, not by this
    # list: "gemini:" goes to Google and anything else to OpenAI, so adding a name here
    # is all it takes to select any model either provider offers.
    #
    # Reasoning-grade models first — this agent plans over ten turns, picks its own tools
    # and writes structured output, which is where the cheaper models thin out.
    # Every Gemini name here was confirmed by an actual generate call, not by appearing in
    # the model list: gemini-2.5-pro is still listed and answers a 404 saying it is no
    # longer available to new keys.
    SUPPORTED_MODELS: list = [
        # Reasoning-grade — what a ten-turn agentic workload deserves
        "gemini:gemini-3.1-pro-preview",
        "gemini:gemini-pro-latest",
        "gpt-5.2",
        "gpt-5-pro",
        # Current generation, fast and cheap enough to leave running
        "gemini:gemini-3.6-flash",
        "gemini:gemini-3.5-flash",
        "gemini:gemini-flash-latest",
        "gpt-5-mini",
        # Cheapest, and the previous generation kept for comparison
        "gemini:gemini-3.5-flash-lite",
        "gemini:gemini-2.5-flash",
        "gpt-4o",
    ]

    # Retried on HTTP status alone (429, 503, timeouts) before failing over. Kept low
    # because each attempt adds latency; an exhausted account is refused before any
    # completion is generated, so retrying it costs time rather than money.
    LLM_MAX_RETRIES: int = 2
    LLM_RETRY_BASE_DELAY: float = 1.0

    # =========================
    # RAG Configuration
    # =========================
    # Embedding model (from sentence-transformers / HuggingFace)
    # MULTILINGUAL (Default - supports 100+ languages):
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    # Alternatives:
    # EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"  # English only, faster (768 dim)
    # EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"  # English only, best quality (1024 dim)

    # Multi-query expansion: total phrasings searched per query, including the original
    USE_MULTI_QUERY: bool = True
    NUM_QUERY_VARIANTS: int = 3

    # Chunking strategy
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_SIZE: int = 100
    # Presets by content type:
    # Research papers/books (preserve context): CHUNK_SIZE=1500, CHUNK_OVERLAP=300
    # Emails/short docs: CHUNK_SIZE=500, CHUNK_OVERLAP=100
    # Technical/code docs: CHUNK_SIZE=800, CHUNK_OVERLAP=150

    # Multi-document support
    MAX_DOCUMENTS_PER_SESSION: int = 50
    DOCUMENT_INDEX_TTL: int = 3600  # 1 hours

    # Retrieval parameters
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_CANDIDATES: int = 25  # Higher for multilingual
    # Presets:
    # High recall (research): TOP_K=7, CANDIDATES=30
    # Balanced: TOP_K=5, CANDIDATES=20
    # High precision: TOP_K=3, CANDIDATES=15

    # Reranking (MULTILINGUAL Default)
    USE_RERANKING: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_THRESHOLD: float = 0.35
    # Alternatives:
    # RERANKER_MODEL: str = "BAAI/bge-reranker-v2-gemma"  # Best accuracy (needs good GPU)
    # RERANKER_MODEL: str = "BAAI/bge-reranker-large"  # English only, faster
    # Threshold: 0.50 (precision), 0.35 (balanced), 0.25 (recall)

    # Hybrid search weights (should sum to ~1.0)
    # Multilingual default: semantic-heavy (concepts over keywords)
    BM25_WEIGHT: float = 0.3
    SEMANTIC_WEIGHT: float = 0.7
    RRF_K: int = 60
    # Presets:
    # Semantic-heavy (multilingual, concepts): BM25=0.2, SEMANTIC=0.8
    # Balanced (general): BM25=0.4, SEMANTIC=0.6
    # Keyword-heavy (codes, technical): BM25=0.7, SEMANTIC=0.3

    # =========================
    # Session & Storage
    # =========================
    SESSION_TTL: int = 3600  # 1 hour
    MAX_CHAT_HISTORY: int = 20
    MAX_FILE_SIZE_MB: int = 50

    # =========================
    # Auto-ingestion
    # =========================
    KB_FOLDER: str = "/app/kb"
    AUTO_INGEST_ON_STARTUP: bool = True
    KB_SESSION_ID: str = "kb_default"  # shared corpus every session may retrieve from

    # =========================
    # Index Persistence
    # =========================
    PERSIST_INDEX: bool = True
    INDEX_DIR: str = "/app/data/index"

    # =========================
    # Request Limits
    # =========================
    MAX_QUERY_CHARS: int = 4000
    MAX_SESSION_ID_CHARS: int = 64
    MAX_CUSTOM_DATA_ROWS: int = 5000

    # =========================
    # Performance
    # =========================
    MAX_CONCURRENT_TASKS: int = 10

    # =========================
    # Data Processing
    # =========================
    # Universal data type handling
    # Name tokens that mark a column as an identifier rather than a measure. Matched as
    # whole `_`-separated parts, so "id" catches customer_id while leaving notes alone.
    # Zero-padded values are detected independently, so this list need not be exhaustive.
    FORCE_STRING_COLUMNS: list = ["id", "code", "ref", "reference", "sku", "employee_code"]
    NUMERIC_CONVERSION_THRESHOLD: float = 0.7  # 70% valid numbers to convert column
    DATE_FORMATS: list = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%b-%y",
        "%B %Y",  # Nov-19, November 2019
    ]

    # One .env serves both this class and Docker Compose interpolation, so it
    # legitimately carries keys the application does not own — GPU_COUNT is read by
    # compose to decide whether to request a device. Ignore rather than reject.
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    settings = Settings()

    # Log configuration on first load
    logger.info("=" * 60)
    logger.info("📋 CONFIGURATION LOADED")
    logger.info("=" * 60)
    logger.info(f"LLM Model: {settings.DEFAULT_MODEL}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Reranker Model: {settings.RERANKER_MODEL}")
    logger.info(f"Device: {settings.DEVICE}")
    logger.info(f"Debug Mode: {settings.DEBUG_MODE}")
    logger.info(f"Redis: {settings.REDIS_URL}")
    logger.info(f"Session TTL: {settings.SESSION_TTL}s")
    logger.info(f"Max Chat History: {settings.MAX_CHAT_HISTORY}")
    logger.info("=" * 60)
    return settings
