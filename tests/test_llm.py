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
        # The dead end names the model and the reason: a silent "temporarily unavailable"
        # hid an exhausted key behind a message that suggested waiting would help.
        self.assertIn("gemini:gemini-flash-latest", result["content"])
        self.assertIn("no usable fallback", result["content"])
        self.assertIn("no fallback available", result["degraded"])

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


class TemperatureHasOneSourceOfTruth(unittest.TestCase):
    """
    The same twelve questions run twice diverged on five of them — a subset filter applied
    once and skipped once, two files' totals summed once and correctly refused once. Every
    figure was right in both runs; what varied was interpretation. An agent obeying a dozen
    rules a turn cannot afford sampling noise, so the default is zero and lives in exactly
    one place.
    """

    def test_the_shipped_default_is_deterministic(self):
        """The class default, not the resolved value — a local .env may override it."""
        from app.core.config import Settings

        self.assertEqual(Settings.model_fields["LLM_TEMPERATURE"].default, 0.0)

    def test_the_chat_default_comes_from_settings_not_a_literal(self):
        import inspect

        from app.core.config import get_settings
        from app.services.llm import LLMService

        default = inspect.signature(LLMService.chat).parameters["temperature"].default
        self.assertEqual(default, get_settings().LLM_TEMPERATURE)

    def test_query_expansion_keeps_its_own_named_temperature(self):
        """Variety is the point there, so it differs on purpose and says so."""
        from app.services.llm import _EXPANSION_TEMPERATURE

        self.assertGreater(_EXPANSION_TEMPERATURE, 0.0)

    def test_no_bare_temperature_literal_survives_in_the_service(self):
        import inspect
        import re

        import app.services.llm as mod

        source = inspect.getsource(mod)
        literals = re.findall(r"temperature\s*=\s*(0\.\d+|[1-9])", source)
        self.assertEqual(literals, [], f"hardcoded temperature(s): {literals}")
