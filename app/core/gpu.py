# app/core/gpu.py

"""
Serialised model inference.

The embedder and reranker together need roughly twice what a 4GB card holds, so
concurrent calls do not share the GPU — they evict each other. Measured on five
parallel retrievals, one reranker batch took 51-102 seconds; the identical batch
run alone takes 0.05. Queueing the calls is what makes them fast.

Inference also gets its own thread pool. The default executor is shared with
document parsing and BM25 scoring, so a burst of uploads would otherwise stall
every embedding call behind them.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

_pool = ThreadPoolExecutor(
    max_workers=settings.INFERENCE_CONCURRENCY, thread_name_prefix="infer"
)
_gate = asyncio.Semaphore(settings.INFERENCE_CONCURRENCY)


async def infer(fn, label: str = "inference"):
    """Run a blocking model call on the inference pool, bounded by the gate."""
    if _gate.locked():
        logger.debug(f"⏳ {label} queued behind in-flight inference")

    async with _gate:
        return await asyncio.get_running_loop().run_in_executor(_pool, fn)
