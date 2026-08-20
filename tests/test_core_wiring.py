"""
The two modules nothing referenced: the inference gate, and the singletons.

A coverage sweep found `gpu.py` and `dependencies.py` named by no test. The gate is not
incidental — it is the reason a 4GB card works at all. Concurrent embedding and reranking
evict each other, and a reranker batch that takes 0.05s alone took 51-102s when five
retrievals ran at once. Everything downstream depends on it queueing, and nothing checked
that it did.

The singletons matter for a duller reason: they are constructed at import, so a circular
import or a constructor that needs a running event loop breaks the whole application at
startup rather than in a request, where it would at least be visible.
"""

import asyncio
import threading
import time
import unittest


class TheInferenceGateSerialises(unittest.IsolatedAsyncioTestCase):
    async def test_calls_do_not_overlap_at_concurrency_one(self):
        """The whole point: two calls queue rather than share the card."""
        from app.core import gpu

        overlap, live = [], []

        def work():
            live.append(1)
            overlap.append(len(live))
            time.sleep(0.05)
            live.pop()
            return "done"

        results = await asyncio.gather(*(gpu.infer(work, "t") for _ in range(4)))
        self.assertEqual(results, ["done"] * 4)
        self.assertEqual(max(overlap), 1, f"observed {max(overlap)} concurrent calls")

    async def test_the_result_is_returned_to_the_caller(self):
        from app.core import gpu

        self.assertEqual(await gpu.infer(lambda: 6 * 7, "answer"), 42)

    async def test_a_failing_call_raises_rather_than_returning_none(self):
        """Swallowing this would surface as an empty embedding, not as an error."""
        from app.core import gpu

        def explode():
            raise RuntimeError("CUDA out of memory")

        with self.assertRaises(RuntimeError):
            await gpu.infer(explode, "boom")

    async def test_the_gate_is_released_after_a_failure(self):
        """A gate held by a crashed call would deadlock every later inference."""
        from app.core import gpu

        def explode():
            raise ValueError("nope")

        for _ in range(3):
            with self.assertRaises(ValueError):
                await gpu.infer(explode, "boom")

        self.assertEqual(
            await asyncio.wait_for(gpu.infer(lambda: "alive", "after"), timeout=5),
            "alive",
        )

    async def test_work_runs_off_the_event_loop(self):
        """A blocking call on the loop would stall every other request."""
        from app.core import gpu

        loop_thread = threading.current_thread().name
        where = await gpu.infer(lambda: threading.current_thread().name, "thread")
        self.assertNotEqual(where, loop_thread)
        self.assertTrue(where.startswith("infer"), where)

    def test_the_pool_is_sized_from_settings(self):
        from app.core import gpu
        from app.core.config import get_settings

        self.assertEqual(
            gpu._pool._max_workers, get_settings().INFERENCE_CONCURRENCY
        )


class TheSingletonsConstructAtImport(unittest.TestCase):
    """
    Built at import time to dodge circular imports, which means a mistake here is a
    startup crash rather than a failed request.
    """

    def setUp(self):
        from app.core import dependencies

        self.deps = dependencies

    def test_every_service_is_present(self):
        for name in ("vector_store", "session_store", "llm_service", "chart_service",
                     "ingestion_service", "retriever"):
            self.assertTrue(hasattr(self.deps, name), name)

    def test_the_services_share_one_vector_store(self):
        """Two stores would mean an upload the retriever cannot see."""
        self.assertIs(self.deps.ingestion_service.vector_store, self.deps.vector_store)
        self.assertIs(self.deps.retriever.vector_store, self.deps.vector_store)

    def test_ingestion_shares_the_one_session_store(self):
        self.assertIs(self.deps.ingestion_service.session_store, self.deps.session_store)

    def test_nothing_was_initialised_at_import(self):
        """Loading 4GB of weights on import would block the event loop at startup."""
        self.assertFalse(self.deps.vector_store._initialized)
        self.assertIsNone(self.deps.vector_store.embedder)

    def test_importing_twice_returns_the_same_objects(self):
        import importlib

        again = importlib.import_module("app.core.dependencies")
        self.assertIs(again.vector_store, self.deps.vector_store)
