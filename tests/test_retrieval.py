"""Hybrid retrieval: session scoping, query fan-out, and index reuse."""

import unittest
from unittest.mock import patch

import app.services.retrieval as retrieval_mod
from app.services.retrieval import HybridRetriever

DOCS = [
    {"content": "alice quarterly revenue grew", "source": "a.pdf", "session_id": "alice"},
    {"content": "bob confidential headcount", "source": "b.pdf", "session_id": "bob"},
    {"content": "shared revenue handbook", "source": "kb.pdf", "session_id": "kb_default"},
    {"content": "bob revenue projections", "source": "d.pdf", "session_id": "bob"},
]


class FakeVectorStore:
    version = 1
    """Returns every document as a hit so scoping, not ranking, is what's under test."""

    def __init__(self):
        self.search_calls = []

    async def get_documents(self):
        return DOCS

    async def search(self, query, top_k=None, allowed=None):
        self.search_calls.append(allowed)
        hits = [(i, 1.0 - i / 100) for i in range(len(DOCS))]
        if allowed is not None:
            hits = [h for h in hits if h[0] in allowed]
        return hits[:top_k] if top_k else hits


class RetrievalTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = FakeVectorStore()
        self.retriever = HybridRetriever(self.store)
        self.settings_patch = patch.multiple(
            retrieval_mod.settings,
            USE_RERANKING=False,
            RERANK_THRESHOLD=0.0,
            RETRIEVAL_TOP_K=10,
            RETRIEVAL_CANDIDATES=25,
        )
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    async def sources(self, **kwargs):
        docs = await self.retriever.retrieve("revenue", **kwargs)
        return sorted(d["source"] for d in docs)

    async def test_session_sees_only_own_documents_and_shared_kb(self):
        self.assertEqual(
            await self.sources(session_ids={"alice", "kb_default"}),
            ["a.pdf", "kb.pdf"],
        )

    async def test_other_sessions_documents_never_leak(self):
        docs = await self.retriever.retrieve("revenue", session_ids={"alice", "kb_default"})
        self.assertFalse(any(d["session_id"] == "bob" for d in docs))

    async def test_multi_query_does_not_widen_scope(self):
        self.assertEqual(
            await self.sources(
                expanded_queries=["revenue", "income", "earnings"],
                session_ids={"alice", "kb_default"},
            ),
            ["a.pdf", "kb.pdf"],
        )

    async def test_unknown_session_retrieves_nothing(self):
        self.assertEqual(await self.sources(session_ids={"nobody"}), [])

    async def test_omitting_scope_searches_everything(self):
        self.assertEqual(len(await self.sources()), len(DOCS))

    async def test_bm25_index_built_once_regardless_of_variant_count(self):
        with patch.object(
            retrieval_mod, "BM25Okapi", wraps=retrieval_mod.BM25Okapi
        ) as bm25:
            await self.retriever.retrieve(
                "revenue", expanded_queries=["a", "b", "c", "d"]
            )
            self.assertEqual(bm25.call_count, 1)

    async def test_the_index_is_reused_across_requests(self):
        """Tokenising the corpus is the expensive part; it must not repeat per request."""
        with patch.object(
            retrieval_mod, "BM25Okapi", wraps=retrieval_mod.BM25Okapi
        ) as bm25:
            for _ in range(3):
                await self.retriever.retrieve("revenue")
            self.assertEqual(bm25.call_count, 1)

    async def test_a_corpus_change_invalidates_the_index(self):
        with patch.object(
            retrieval_mod, "BM25Okapi", wraps=retrieval_mod.BM25Okapi
        ) as bm25:
            await self.retriever.retrieve("revenue")
            self.retriever.vector_store.version += 1
            await self.retriever.retrieve("revenue")
            self.assertEqual(bm25.call_count, 2)

    async def test_scope_is_passed_down_to_vector_search(self):
        await self.retriever.retrieve("revenue", session_ids={"alice"})
        self.assertTrue(all(a is not None for a in self.store.search_calls))


if __name__ == "__main__":
    unittest.main()
