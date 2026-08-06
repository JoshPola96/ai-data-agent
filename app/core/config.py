# app/core/config.py

"""
Configuration Management
Centralized settings with environment variable support
"""

import torch
from functools import cached_property, lru_cache
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration with tunable parameters"""

    # =========================
    # Device Configuration (GPU/CPU)
    # =========================
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
    DEBUG_MODE: bool = True  # Enable comprehensive debug logging

    # =========================
    # LLM Configuration
    # =========================
    # Model format: "provider:model" or just "model" (defaults to OpenAI)
    # Gemini options: gemini-2.0-flash-001 (stable, GA), gemini-2.5-flash-001 (latest), gemini-2.0-pro-exp (best)
    # OpenAI options: gpt-4o (best), gpt-3.5-turbo (cheap)
    DEFAULT_MODEL: str = "gpt-4o"
    FALLBACK_MODEL: str = "gemini:gemini-flash-latest"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 4000
    AGENT_MAX_TURNS: int = 10

    # Single source of truth for /models and /models/select; "gemini:" prefix routes to Google
    SUPPORTED_MODELS: list = [
        "gemini:gemini-2.5-flash",
        "gemini:gemini-flash-latest",
        "gemini:gemini-3-flash-preview",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-4o",
    ]

    # Transient faults (429 rate limit, 503 overload) are retried before failing over.
    # Kept low on purpose: a funding error is never retried, and each attempt costs.
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
    FORCE_STRING_COLUMNS: list = ["id", "code", "reference", "employee_code"]
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

    class Config:
        env_file = ".env"
        case_sensitive = True


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
