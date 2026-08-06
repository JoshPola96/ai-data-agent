# app/preload.py

"""
Model Preloading Script
Downloads and caches models during Docker build for faster startup.
Updated with Smart Caching: Checks existence before downloading.
"""

import sys
import os
import logging

from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read from the environment rather than app.core.config: this layer costs ~4GB to
# rebuild, and importing config would make every settings edit invalidate it.
# Defaults mirror Settings.EMBEDDING_MODEL / Settings.RERANKER_MODEL.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def load_model_smart(build, model_name, model_type="Model"):
    """
    Load from the local cache when possible, downloading only when it is missing.

    Offline mode is passed as an argument, not an environment variable: huggingface_hub
    reads HF_HUB_OFFLINE into a module constant at import time, so setting it here —
    after sentence_transformers is imported — would be silently ignored.
    """
    logger.info(f"🔍 Checking cache for {model_type}: {model_name}")

    try:
        model = build(local_files_only=True)
        logger.info(f"✅ [CACHE] {model_name} loaded from cache, no download")
        return model
    except Exception as e:
        logger.info(f"⚠️ [MISSING] {model_name} not cached ({type(e).__name__}), downloading...")

    try:
        model = build()
        logger.info(f"✅ [DOWNLOAD] {model_name} ready")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {e}")
        raise


def preload():
    """Preload all models into cache"""
    # Cache directory
    hf_cache = os.environ.get("HF_HOME", "/app/.cache/huggingface")
    logger.info("=" * 80)
    logger.info("📦 SMART MODEL PRELOADING")
    logger.info("=" * 80)
    logger.info(f"Cache directory: {hf_cache}")

    device = "cpu"  # Always use CPU for preloading build step

    try:
        # 1. Embedding Model
        logger.info("")
        embedding_model = load_model_smart(
            lambda **kw: SentenceTransformer(
                EMBEDDING_MODEL, device=device, cache_folder=hf_cache, **kw
            ),
            EMBEDDING_MODEL,
            "Embedding",
        )
        logger.info(
            f"   Dimension: {embedding_model.get_sentence_embedding_dimension()}"
        )

        # 2. Reranker Model
        logger.info("")
        reranker = load_model_smart(
            lambda **kw: CrossEncoder(RERANKER_MODEL, device=device, **kw),
            RERANKER_MODEL,
            "Reranker",
        )

        # 3. Verification
        logger.info("")
        logger.info("🧪 Verifying models...")

        # Test embedding
        test_embed = embedding_model.encode(["test"], show_progress_bar=False)
        logger.info(f"   ✓ Embedding functional (Shape: {test_embed.shape})")

        # Test reranker
        test_scores = reranker.predict([["test query", "test document"]])
        logger.info(f"   ✓ Reranker functional (Score: {test_scores[0]:.3f})")

        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ PRELOAD COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ PRELOAD CRITICAL FAILURE: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    preload()
