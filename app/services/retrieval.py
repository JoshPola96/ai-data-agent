# app/services/retrieval.py

"""
Hybrid Retrieval Service
BM25 (lexical) + FAISS (semantic) + Cross-encoder reranking with comprehensive logging
"""

import asyncio
import logging
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.core.config import get_settings
from app.core.gpu import infer
from app.core.vectorstore import VectorStore
from asyncio import Lock

settings = get_settings()
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Advanced hybrid retrieval:
    1. BM25 lexical search
    2. FAISS semantic search
    3. RRF (Reciprocal Rank Fusion)
    4. Cross-encoder reranking
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.reranker: CrossEncoder = None
        self._reranker_initialized = False
        self._lock = Lock()
        # BM25 tokenises the whole corpus to build, so it is cached against the store's version rather than rebuilt for every request
        self._bm25 = None
        self._bm25_version = None
        self._bm25_lock = Lock()

    async def _initialize_reranker(self):
        """Lazy load reranker model"""
        if not settings.USE_RERANKING or self._reranker_initialized:
            return

        async with self._lock:
            if self._reranker_initialized:
                return

            logger.info("=" * 60)
            logger.info("🔧 INITIALIZING RERANKER")
            logger.info("=" * 60)
            logger.info(f"Model: {settings.RERANKER_MODEL}")
            logger.info(f"Device: {settings.DEVICE}")

            try:
                self.reranker = await infer(
                    lambda: CrossEncoder(
                        settings.RERANKER_MODEL,
                        device=settings.DEVICE,
                        model_kwargs={"torch_dtype": settings.MODEL_DTYPE},
                    ),
                    "load reranker",
                )

                logger.info("✅ Reranker loaded successfully")
                logger.info("=" * 60)
                self._reranker_initialized = True

            except Exception as e:
                logger.error(f"❌ Reranker initialization failed: {e}")
                raise

    async def retrieve(
        self,
        query: str,
        expanded_queries: List[str] = None,
        session_ids: set = None,
    ) -> List[Dict]:
        """
        Main retrieval pipeline.

        Args:
            query: Original user query
            expanded_queries: Optional pre-expanded queries
            session_ids: Restrict retrieval to these sessions; None searches everything

        Returns:
            List of documents with relevance scores
        """
        start_time = asyncio.get_event_loop().time()
        top_k = settings.RETRIEVAL_TOP_K

        queries = expanded_queries if expanded_queries else [query]

        logger.info("=" * 60)
        logger.info("🔍 HYBRID RETRIEVAL")
        logger.info("=" * 60)
        logger.info(f"Query: {query}")
        logger.info(f"Expanded queries: {len(queries)}")
        if len(queries) > 1:
            for i, q in enumerate(queries[:3], 1):
                logger.info(f"  {i}. {q}")

        # Get all documents
        all_docs = await self.vector_store.get_documents()
        if not all_docs:
            logger.warning("⚠️ No documents in vector store")
            return []

        # Scope to the documents this session may see; positions stay global so BM25 and the vector index agree and the cached index is shared by every session
        allowed = (
            None
            if session_ids is None
            else {
                i
                for i, d in enumerate(all_docs)
                if d.get("session_id") in session_ids
            }
        )

        if allowed is not None and not allowed:
            logger.warning("⚠️ No documents visible to this session")
            return []

        visible = len(all_docs) if allowed is None else len(allowed)
        logger.info(f"Searching {visible} of {len(all_docs)} documents")

        bm25 = await self._corpus_index(all_docs)

        # Search with all queries in parallel
        search_tasks = [self._search_single_query(q, bm25, allowed) for q in queries]
        all_query_results = await asyncio.gather(*search_tasks)

        # Combine results
        combined_indices = []
        for fused_indices in all_query_results:
            combined_indices.extend(fused_indices)

        # Deduplicate while preserving order
        unique_indices = list(dict.fromkeys(combined_indices))

        logger.debug(f"Combined results: {len(unique_indices)} unique documents")

        # Get top candidates for reranking
        candidate_indices = unique_indices[: settings.RETRIEVAL_CANDIDATES]
        candidates = [all_docs[i] for i in candidate_indices if i < len(all_docs)]

        if not candidates:
            logger.warning("⚠️ No candidates found")
            return []

        logger.info(f"Reranking {len(candidates)} candidates")

        # Reranking
        if settings.USE_RERANKING:
            await self._initialize_reranker()
            ranked_docs = await self._rerank(query, candidates, top_k)
        else:
            ranked_docs = candidates[:top_k]
            for doc in ranked_docs:
                doc["relevance_score"] = 1.0

        # Filter by threshold
        final_docs = [
            doc
            for doc in ranked_docs
            if doc.get("relevance_score", 0) >= settings.RERANK_THRESHOLD
        ]

        elapsed = asyncio.get_event_loop().time() - start_time

        logger.info("=" * 60)
        logger.info(f"✅ RETRIEVAL COMPLETE in {elapsed:.2f}s")
        logger.info(f"   Retrieved: {len(final_docs)} documents")
        if final_docs:
            logger.info(f"   Top score: {final_docs[0].get('relevance_score', 0):.3f}")
        logger.info("=" * 60)

        return final_docs

    async def _corpus_index(self, all_docs: List[Dict]) -> BM25Okapi:
        """
        BM25 over the whole corpus, rebuilt only when the corpus changes.

        Building it tokenises every document, which was previously paid on every
        request — and once per query variant before that. Keying the cache on the
        store's version also lets all sessions share one index: scoping is applied
        when selecting hits, not by re-indexing a subset.
        """
        version = self.vector_store.version

        if self._bm25 is not None and self._bm25_version == version:
            return self._bm25

        async with self._bm25_lock:
            if self._bm25 is not None and self._bm25_version == version:
                return self._bm25

            self._bm25 = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: BM25Okapi(
                    [d.get("content", "").lower().split() for d in all_docs]
                ),
            )
            self._bm25_version = version
            logger.info(f"🔤 BM25 indexed {len(all_docs)} documents (v{version})")

        return self._bm25

    async def _search_single_query(
        self, query: str, bm25: BM25Okapi, allowed: set = None
    ) -> List[int]:
        """Hybrid search for single query"""
        logger.debug(f"Searching: {query[:50]}...")

        # BM25 and FAISS in parallel
        bm25_task = self._bm25_search(
            query, bm25, settings.RETRIEVAL_CANDIDATES, allowed
        )
        faiss_task = self.vector_store.search(
            query, settings.RETRIEVAL_CANDIDATES, allowed=allowed
        )

        bm25_results, faiss_results = await asyncio.gather(bm25_task, faiss_task)

        logger.debug(f"  BM25: {len(bm25_results)} results")
        logger.debug(f"  FAISS: {len(faiss_results)} results")

        # Fusion
        fused_indices = self._rrf_fusion(bm25_results, faiss_results)

        logger.debug(f"  Fused: {len(fused_indices)} results")

        return fused_indices

    async def _bm25_search(
        self, query: str, bm25: BM25Okapi, top_k: int, allowed: set = None
    ) -> List[Tuple[int, float]]:
        """BM25 lexical search against the cached corpus index, scoped on selection."""

        def _bm25_compute():
            scores = bm25.get_scores(query.lower().split())
            hits = []
            for i in np.argsort(scores)[::-1]:
                index = int(i)
                if allowed is None or index in allowed:
                    hits.append((index, float(scores[index])))
                    if len(hits) >= top_k:
                        break
            return hits

        return await asyncio.get_event_loop().run_in_executor(None, _bm25_compute)

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        faiss_results: List[Tuple[int, float]],
        k: int = None,
    ) -> List[int]:
        """Reciprocal Rank Fusion"""
        k = k or settings.RRF_K
        scores = {}

        # BM25 scores
        for rank, (idx, _) in enumerate(bm25_results, 1):
            scores[idx] = scores.get(idx, 0) + (settings.BM25_WEIGHT / (k + rank))

        # FAISS scores
        for rank, (idx, _) in enumerate(faiss_results, 1):
            scores[idx] = scores.get(idx, 0) + (settings.SEMANTIC_WEIGHT / (k + rank))

        # Sort
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_indices

    async def _rerank(
        self, query: str, documents: List[Dict], top_k: int
    ) -> List[Dict]:
        """Cross-encoder reranking"""
        if not documents:
            return []

        logger.debug(f"Reranking {len(documents)} documents")

        # Prepare pairs
        pairs = [[query, doc["content"]] for doc in documents]

        # Run reranker
        scores = await infer(
            lambda: self.reranker.predict(pairs, show_progress_bar=False), "rerank"
        )

        # Normalize
        scores = np.array(scores)
        min_score, max_score = scores.min(), scores.max()

        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.ones_like(scores)

        # Sort
        ranked_indices = np.argsort(normalized_scores)[::-1][:top_k]

        # Add scores
        ranked_docs = []
        for idx in ranked_indices:
            doc = documents[idx].copy()
            doc["relevance_score"] = float(normalized_scores[idx])
            ranked_docs.append(doc)

        logger.debug(f"Reranked: top score={ranked_docs[0]['relevance_score']:.3f}")

        return ranked_docs
