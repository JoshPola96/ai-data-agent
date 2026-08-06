# app/core/vectorstore.py

"""
In-Memory Vector Store
High-performance FAISS-based vector database with GPU support and comprehensive logging
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class VectorStore:
    """
    In-memory vector database with FAISS indexing.
    Supports GPU acceleration when available.
    Thread-safe with async support for concurrent operations.
    """

    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.index: Optional[faiss.Index] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.dim = settings.EMBEDDING_DIM
        self._lock = asyncio.Lock()
        self._initialized = False

        logger.debug(f"🔧 VectorStore.__init__: Dimension={self.dim}")

    async def initialize(self):
        """Initialize embedding model with GPU support if available"""
        if self._initialized:
            logger.debug("✓ VectorStore already initialized, skipping")
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info("=" * 60)
            logger.info("🚀 INITIALIZING VECTOR STORE")
            logger.info("=" * 60)
            logger.info(f"Model: {settings.EMBEDDING_MODEL}")
            logger.info(f"Device: {settings.DEVICE}")
            logger.info(f"Dimension: {settings.EMBEDDING_DIM}")

            # Load model in thread pool
            loop = asyncio.get_event_loop()

            try:
                self.embedder = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(
                        settings.EMBEDDING_MODEL,
                        device=settings.DEVICE,
                        cache_folder=settings.HF_HOME,
                    ),
                )

                # Verify model dimension
                test_embed = self.embedder.encode(["test"], convert_to_numpy=True)
                actual_dim = test_embed.shape[1]

                if actual_dim != settings.EMBEDDING_DIM:
                    logger.warning(
                        f"⚠️ Model dimension mismatch! Expected {settings.EMBEDDING_DIM}, got {actual_dim}"
                    )
                    self.dim = actual_dim
                    logger.info(f"   Updated dimension to: {self.dim}")

                # Initialize FAISS index
                self.index = faiss.IndexFlatIP(self.dim)
                self._restore()

                logger.info("✅ VectorStore initialized successfully")
                logger.info(f"   Embedding Model: {settings.EMBEDDING_MODEL}")
                logger.info(f"   Device: {settings.DEVICE}")
                logger.info(f"   Dimension: {self.dim}")
                logger.info("=" * 60)

                self._initialized = True

            except Exception as e:
                logger.error(
                    f"❌ VectorStore initialization failed: {e}", exc_info=True
                )
                raise

    def _paths(self) -> Tuple[Path, Path]:
        base = Path(settings.INDEX_DIR)
        return base / "index.faiss", base / "documents.json"

    def _restore(self):
        """Load a previously persisted index so uploads survive a restart."""
        if not settings.PERSIST_INDEX:
            return

        index_path, docs_path = self._paths()
        if not (index_path.exists() and docs_path.exists()):
            logger.info("📂 No persisted index found, starting empty")
            return

        try:
            documents = json.loads(docs_path.read_text(encoding="utf-8"))
            index = faiss.read_index(str(index_path))

            # A mismatch means the pair was written by a different model or a torn
            # write; rebuilding from documents is safer than serving wrong vectors
            if index.d != self.dim or index.ntotal != len(documents):
                logger.warning(
                    f"⚠️ Persisted index does not match documents "
                    f"(dim {index.d}/{self.dim}, vectors {index.ntotal}/{len(documents)}), discarding"
                )
                return

            self.documents = documents
            self.index = index
            logger.info(f"📂 Restored {index.ntotal} vectors from {index_path}")

        except Exception as e:
            logger.warning(f"⚠️ Could not restore index ({e}), starting empty")

    def _persist(self):
        """Write index and documents atomically so a crash cannot leave a torn pair."""
        if not settings.PERSIST_INDEX or self.index is None:
            return

        index_path, docs_path = self._paths()

        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)

            faiss.write_index(self.index, str(index_path) + ".tmp")
            docs_path.with_suffix(".json.tmp").write_text(
                json.dumps(self.documents), encoding="utf-8"
            )

            os.replace(str(index_path) + ".tmp", index_path)
            os.replace(docs_path.with_suffix(".json.tmp"), docs_path)

            logger.debug(f"💾 Persisted {self.index.ntotal} vectors")

        except Exception as e:
            logger.error(f"❌ Index persistence failed: {e}")

    async def prune_sessions(self, keep) -> int:
        """Drop documents whose session no longer exists; keep(session_id) decides."""
        async with self._lock:
            before = len(self.documents)
            self.documents = [d for d in self.documents if keep(d.get("session_id"))]
            removed = before - len(self.documents)

            if removed:
                await self._build_index()
                self._persist()
                logger.info(f"🧹 Pruned {removed} documents from expired sessions")

        return removed

    async def add_documents(
        self, documents: List[Dict[str, str]], rebuild_index: bool = True
    ) -> int:
        """
        Add documents to the store and rebuild index.

        Args:
            documents: List of dicts with 'content', 'source', 'doc_id', 'metadata'
            rebuild_index: Whether to rebuild FAISS index immediately

        Returns:
            Number of documents added
        """
        if not documents:
            logger.debug("⊘ add_documents: No documents provided")
            return 0

        await self.initialize()

        logger.info("=" * 60)
        logger.info(f"📥 ADDING {len(documents)} DOCUMENTS TO VECTOR STORE")
        logger.info("=" * 60)

        async with self._lock:
            start_count = len(self.documents)

            # Log document sources
            sources = {}
            for doc in documents:
                source = doc.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1

            logger.info("Documents by source:")
            for source, count in sources.items():
                logger.info(f"  • {source}: {count} chunks")

            self.documents.extend(documents)

            logger.info(f"Total documents in store: {len(self.documents)}")

            if rebuild_index:
                if self.index is not None and start_count > 0:
                    logger.info("🔨 Performing incremental index update...")
                    await self._incremental_index_update(documents)
                else:
                    logger.info("🔨 Building initial index...")
                    await self._build_index()

                self._persist()

        logger.info("=" * 60)
        return len(documents)

    async def _incremental_index_update(self, new_documents: List[Dict[str, str]]):
        """Add new embeddings to existing index without rebuilding"""
        start_time = asyncio.get_event_loop().time()

        logger.debug(f"🔧 Incremental update for {len(new_documents)} documents")

        # Extract and embed
        contents = [doc["content"] for doc in new_documents]

        logger.debug(
            f"   Content lengths: min={min(len(c) for c in contents)}, "
            f"max={max(len(c) for c in contents)}, "
            f"avg={sum(len(c) for c in contents) // len(contents)}"
        )

        loop = asyncio.get_event_loop()

        logger.debug(
            f"   Embedding {len(contents)} texts with batch_size={settings.EMBEDDING_BATCH_SIZE}"
        )

        embeddings = await loop.run_in_executor(
            None,
            lambda: self.embedder.encode(
                contents,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=settings.DEBUG_MODE,
                device=settings.DEVICE,
            ).astype("float32"),
        )

        logger.debug(f"   Embedding shape: {embeddings.shape}")

        # Normalize
        faiss.normalize_L2(embeddings)

        # Add to index
        self.index.add(embeddings)

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"✅ Index updated in {elapsed:.2f}s: "
            f"added {len(new_documents)} vectors, total {self.index.ntotal}"
        )

    async def _build_index(self):
        """Build FAISS index from current documents with GPU support"""
        if not self.documents:
            logger.warning("⚠️ No documents to index")
            self.index = None
            return

        start_time = asyncio.get_event_loop().time()
        logger.info(f"🔨 Building FAISS index for {len(self.documents)} documents")

        if self.embedder is None:
            logger.error("❌ Embedder not initialized!")
            raise RuntimeError("VectorStore not initialized")

        # Extract content
        contents = [doc["content"] for doc in self.documents]

        logger.debug("   Document content stats:")
        logger.debug(f"     Min length: {min(len(c) for c in contents)}")
        logger.debug(f"     Max length: {max(len(c) for c in contents)}")
        logger.debug(
            f"     Avg length: {sum(len(c) for c in contents) // len(contents)}"
        )

        # Embed in batches
        loop = asyncio.get_event_loop()

        logger.info(
            f"   Encoding {len(contents)} texts (batch_size={settings.EMBEDDING_BATCH_SIZE})"
        )

        embeddings = await loop.run_in_executor(
            None,
            lambda: self.embedder.encode(
                contents,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=settings.DEBUG_MODE,
                device=settings.DEVICE,
            ).astype("float32"),
        )

        logger.debug(f"   Generated embeddings shape: {embeddings.shape}")

        # Normalize
        faiss.normalize_L2(embeddings)

        # The index always lives on CPU: this project ships faiss-cpu, which has no
        # GPU support. Embedding and reranking still run on the GPU via torch.
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        logger.info(
            f"✅ FAISS index built in {asyncio.get_event_loop().time() - start_time:.2f}s"
        )

        logger.info(f"   Total vectors in index: {self.index.ntotal}")

    async def search(
        self, query: str, top_k: int = None, allowed: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Semantic search using FAISS.

        Args:
            query: Search query
            top_k: Number of results
            allowed: Restrict hits to these document indices

        Returns:
            List of (document_index, similarity_score) tuples
        """
        await self.initialize()

        if not self.index or self.index.ntotal == 0:
            logger.warning("⚠️ No documents indexed for search")
            return []

        if self.embedder is None:
            logger.error("❌ Embedder not available for search")
            return []

        top_k = top_k or settings.RETRIEVAL_CANDIDATES

        logger.debug(f"🔍 Searching: query='{query[:100]}...', top_k={top_k}")

        # Embed query
        loop = asyncio.get_event_loop()
        query_vec = await loop.run_in_executor(
            None,
            lambda: self.embedder.encode(
                [query], convert_to_numpy=True, device=settings.DEVICE
            ).astype("float32"),
        )
        faiss.normalize_L2(query_vec)

        # Over-fetch when scoping, since foreign hits are discarded after ranking
        fetch_k = min(top_k * 4 if allowed is not None else top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)

        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1 and (allowed is None or int(idx) in allowed)
        ][:top_k]

        logger.debug(f"   Found {len(results)} results")
        if results and settings.DEBUG_MODE:
            logger.debug(f"   Top 3 scores: {[f'{s:.3f}' for _, s in results[:3]]}")

        return results

    async def get_documents(
        self, indices: Optional[List[int]] = None
    ) -> List[Dict[str, str]]:
        """Get documents by indices, or all if None."""
        async with self._lock:
            if indices is None:
                return self.documents.copy()
            return [self.documents[i] for i in indices if i < len(self.documents)]

    async def clear(self):
        """Clear all documents and index"""
        async with self._lock:
            self.documents.clear()
            self.index = faiss.IndexFlatIP(self.dim)
            self._persist()
            logger.info("🗑️ Vector store cleared")

    async def remove_by_session(self, session_id: str) -> int:
        """Remove every document owned by a session and reindex."""
        async with self._lock:
            initial_count = len(self.documents)
            self.documents = [
                d for d in self.documents if d.get("session_id") != session_id
            ]
            removed_count = initial_count - len(self.documents)

            if removed_count > 0:
                await self._build_index()
                self._persist()
                logger.info(
                    f"🗑️ Removed {removed_count} documents for session {session_id[:8]}"
                )

        return removed_count

    async def remove_by_source(self, source: str) -> int:
        """Remove all documents from a specific source."""
        async with self._lock:
            initial_count = len(self.documents)
            self.documents = [d for d in self.documents if d.get("source") != source]
            removed_count = initial_count - len(self.documents)

            if removed_count > 0:
                await self._build_index()
                self._persist()
                logger.info(
                    f"🗑️ Removed {removed_count} documents from source: {source}"
                )

        return removed_count

    def get_stats(self) -> Dict:
        """Get current store statistics"""
        sources = {}
        for doc in self.documents:
            src = doc.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1

        return {
            "total_documents": len(self.documents),
            "indexed_vectors": self.index.ntotal if self.index else 0,
            "sources": sources,
            "embedding_model": settings.EMBEDDING_MODEL,
            "dimension": self.dim,
            "device": settings.DEVICE,
        }
