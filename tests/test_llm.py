"""Query expansion: parsing, deduplication, and degradation when the provider fails."""

import unittest
from unittest.mock import AsyncMock, patch

from app.services.llm import LLMService


class QueryExpansionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.svc = LLMService()
        self.svc._initialized = True

    def respond(self, payload):
        return patch.object(self.svc, "_complete_json", AsyncMock(return_value=payload))

    async def test_original_query_leads_the_list(self):
        with self.respond('{"queries": ["total revenue", "how much revenue"]}'):
            out = await self.svc.generate_query_variants("What is the revenue?", 3)
        self.assertEqual(out[0], "What is the revenue?")
        self.assertEqual(len(out), 3)

    async def test_variant_count_is_a_hard_cap(self):
        with self.respond('{"queries": ["a", "b", "c", "d", "e"]}'):
            out = await self.svc.generate_query_variants("q", 3)
        self.assertEqual(len(out), 3)

    async def test_duplicates_are_dropped_case_insensitively(self):
        with self.respond('{"queries": ["Revenue", "  REVENUE ", "profit"]}'):
            out = await self.svc.generate_query_variants("revenue", 5)
        self.assertEqual(out, ["revenue", "profit"])

    async def test_malformed_json_falls_back_to_original(self):
        with self.respond("not json"):
            out = await self.svc.generate_query_variants("revenue", 3)
        self.assertEqual(out, ["revenue"])

    async def test_provider_failure_falls_back_to_original(self):
        with patch.object(
            self.svc, "_complete_json", AsyncMock(side_effect=RuntimeError("down"))
        ):
            out = await self.svc.generate_query_variants("revenue", 3)
        self.assertEqual(out, ["revenue"])

    async def test_single_variant_skips_the_llm_entirely(self):
        with patch.object(self.svc, "_complete_json", AsyncMock()) as call:
            out = await self.svc.generate_query_variants("revenue", 1)
        self.assertEqual(out, ["revenue"])
        call.assert_not_called()


class FallbackTest(unittest.IsolatedAsyncioTestCase):
    """A dead fallback model must terminate rather than recurse."""

    async def test_fallback_does_not_loop_when_it_is_the_model_that_failed(self):
        svc = LLMService()
        svc._initialized = True
        svc.model = "gpt-5.2"
        svc.fallback_model = "gemini:gemini-flash-latest"

        with patch.object(svc, "chat", AsyncMock()) as chat:
            result = await svc._try_fallback(
                [], None, None, 0.1, True, failed_model="gemini:gemini-flash-latest"
            )

        chat.assert_not_called()
        self.assertIn("unavailable", result["content"])

    async def test_fallback_hops_once_when_a_different_model_failed(self):
        svc = LLMService()
        svc._initialized = True
        svc.model = "gpt-5.2"
        svc.fallback_model = "gemini:gemini-flash-latest"

        with patch.object(svc, "chat", AsyncMock(return_value={"content": "ok"})) as chat:
            await svc._try_fallback([], None, None, 0.1, True, failed_model="gpt-5.2")

        chat.assert_awaited_once()
        self.assertEqual(
            chat.await_args.kwargs["model_override"], "gemini:gemini-flash-latest"
        )


if __name__ == "__main__":
    unittest.main()
