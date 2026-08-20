"""
The /chat handler itself, with the provider and session store mocked.

Every other suite exercises the tools directly, which let a NameError in the
handler's context assembly reach production: the import was updated and the call
site was not, and nothing executed that line until a real request arrived.
"""

import json
import unittest
from unittest.mock import AsyncMock, patch

import pandas as pd

import app.main as main
from app.utils import prompts
from app.utils.schemas import ChatRequest

DF = pd.DataFrame(
    {
        "region": ["North", "South", "East", "West"],
        "revenue": [120000, 98000, 156000, 87000],
    }
)

FINAL = json.dumps(
    {
        "answer": "East leads on revenue at 156,000.",
        "visualizations": [],
        "key_insights": ["East is highest", "West is lowest"],
        "sources_used": ["sales"],
    }
)


def answer(content=FINAL, tool_calls=()):
    return AsyncMock(return_value={"content": content, "tool_calls": list(tool_calls)})


class ChatEndpointTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tables = {"sales": DF}
        patches = [
            patch.object(main.session_store, "get_all_dataframes", AsyncMock(return_value=self.tables)),
            patch.object(main.session_store, "get_chat_history", AsyncMock(return_value=[])),
            patch.object(main.session_store, "get_session_documents", AsyncMock(return_value=[])),
            patch.object(main.session_store, "add_chat_turn", AsyncMock()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    async def ask(self, query="revenue by region", **kw):
        return await main.chat(ChatRequest(session_id="uitest", query=query, **kw))

    async def test_a_plain_question_is_answered(self):
        with patch.object(main.llm_service, "chat", answer()):
            resp = await self.ask()
        self.assertIn("East", resp.response)
        self.assertEqual(resp.session_id, "uitest")

    async def test_insights_reach_the_response_metadata(self):
        with patch.object(main.llm_service, "chat", answer()):
            resp = await self.ask()
        self.assertEqual(len(resp.metadata["key_insights"]), 2)

    async def test_the_system_prompt_names_every_table(self):
        """Regression: the handler referenced a helper it no longer imported."""
        spy = answer()
        with patch.object(main.llm_service, "chat", spy):
            await self.ask()

        system_prompt = spy.await_args.args[1]
        self.assertIn("sales", system_prompt)
        self.assertIn("revenue", system_prompt)

    async def test_metadata_reports_turns_and_model(self):
        with patch.object(main.llm_service, "chat", answer()):
            resp = await self.ask()
        self.assertEqual(resp.metadata["turns_taken"], 1)
        self.assertIn("processing_time", resp.metadata)

    async def test_tools_are_withheld_on_the_final_turn(self):
        """
        A model that only ever calls tools must still be made to answer, or the loop
        spends its budget and returns an apology.
        """
        calls = []

        async def always_tools(messages, system_prompt, tools, *a, **kw):
            calls.append(tools)
            if tools:
                return {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "name": "calculate_statistics",
                         "arguments": {"table_name": "sales", "operation": "sum", "column": "revenue"}}
                    ],
                }
            return {"content": FINAL, "tool_calls": []}

        with patch.object(main.llm_service, "chat", side_effect=always_tools):
            resp = await self.ask()

        self.assertIsNone(calls[-1], "final turn must be called without tools")
        self.assertIn("East", resp.response, "final turn must produce a real answer")

    async def test_a_produced_chart_is_attached_to_the_response(self):
        emitted = []

        async def chart_then_answer(messages, system_prompt, tools, *a, **kw):
            if tools and not emitted:
                emitted.append(1)
                return {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "name": "generate_chart",
                         "arguments": {"table_name": "sales", "chart_type": "bar",
                                       "x_column": "region", "y_column": "revenue",
                                       "title": "Revenue"}}
                    ],
                }
            return {"content": FINAL, "tool_calls": []}

        with patch.object(main.llm_service, "chat", side_effect=chart_then_answer):
            resp = await self.ask()

        charts = [v for v in resp.metadata["visualizations"] if v["type"] == "chart"]
        self.assertEqual(len(charts), 1)
        self.assertIn("chart_json", charts[0]["chart_data"])

    async def test_a_produced_diagram_is_attached_to_the_response(self):
        emitted = []

        async def diagram_then_answer(messages, system_prompt, tools, *a, **kw):
            if tools and not emitted:
                emitted.append(1)
                return {
                    "content": "",
                    "tool_calls": [
                        {"id": "d1", "name": "generate_diagram",
                         "arguments": {"mermaid": "flowchart TD\n  A[Start] --> B[End]",
                                       "title": "Flow"}}
                    ],
                }
            return {"content": FINAL, "tool_calls": []}

        with patch.object(main.llm_service, "chat", side_effect=diagram_then_answer):
            resp = await self.ask()

        diagrams = [v for v in resp.metadata["visualizations"] if v["type"] == "diagram"]
        self.assertEqual(len(diagrams), 1)
        self.assertIn("flowchart", diagrams[0]["mermaid"])
        self.assertEqual(diagrams[0]["caption"], "Flow")

    async def test_the_agent_trace_records_tools_and_results(self):
        emitted = []

        async def one_tool(messages, system_prompt, tools, *a, **kw):
            if tools and not emitted:
                emitted.append(1)
                return {
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "name": "calculate_statistics",
                         "arguments": {"table_name": "sales", "operation": "sum", "column": "revenue"}}
                    ],
                }
            return {"content": FINAL, "tool_calls": []}

        with patch.object(main.llm_service, "chat", side_effect=one_tool):
            resp = await self.ask()

        kinds = [s.type for s in resp.agent_trace]
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_result", kinds)

    async def test_a_chart_spec_the_model_invents_is_discarded(self):
        """
        Observed in production: instead of calling the tool, the model put
        {"chart_type": "bar", ...} in `visualizations`. That is a spec, not a figure, and
        the frontend drew an empty panel for it — which read as "no chart was produced".
        """
        echoed = json.dumps(
            {
                "answer": "East leads on revenue.",
                "visualizations": [
                    {"chart_type": "bar", "x_column": "region", "y_column": "revenue",
                     "title": "Revenue by Region"},
                    {"type": "table", "data": [{"region": "East", "revenue": 156000}]},
                ],
                "key_insights": ["East is highest"],
                "sources_used": ["sales"],
            }
        )
        with patch.object(main.llm_service, "chat", answer(content=echoed)):
            resp = await self.ask()

        kinds = [v.get("type") for v in resp.metadata["visualizations"]]
        self.assertEqual(kinds, ["table"], "only drawable blocks may reach the frontend")

    async def test_malformed_model_output_degrades_instead_of_raising(self):
        with patch.object(main.llm_service, "chat", answer(content="not json at all")):
            resp = await self.ask()
        self.assertTrue(resp.response)


if __name__ == "__main__":
    unittest.main()


class TheStreamNarratesTheSameRun(unittest.IsolatedAsyncioTestCase):
    """
    /chat and /chat/stream are one loop with two consumers.

    Duplicating the handler would let them drift, and the streamed one is the one nobody
    would notice going stale. The generator is therefore the only implementation: /chat
    keeps the `final` event, /chat/stream forwards every event as it happens.
    """

    def setUp(self):
        self.tables = {"sales": DF}
        patches = [
            patch.object(main.session_store, "get_all_dataframes", AsyncMock(return_value=self.tables)),
            patch.object(main.session_store, "get_chat_history", AsyncMock(return_value=[])),
            patch.object(main.session_store, "get_session_documents", AsyncMock(return_value=[])),
            patch.object(main.session_store, "add_chat_turn", AsyncMock()),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    async def events(self, side_effect=None, content=FINAL):
        chat = (AsyncMock(side_effect=side_effect) if side_effect
                else answer(content=content))
        with patch.object(main.llm_service, "chat", chat):
            return [
                event
                async for event in main.run_agent(
                    ChatRequest(session_id="stream", query="revenue by region")
                )
            ]

    async def test_a_plain_answer_still_ends_in_one_final(self):
        events = await self.events()
        self.assertEqual([e["type"] for e in events][-1], "final")
        self.assertEqual(sum(1 for e in events if e["type"] == "final"), 1)

    async def test_the_context_event_names_the_tables_in_scope(self):
        context = next(e for e in await self.events() if e["type"] == "context")
        self.assertEqual(context["tables"], ["sales"])

    async def test_every_turn_reports_its_number_and_budget(self):
        turns = [e for e in await self.events() if e["type"] == "turn"]
        self.assertEqual(turns[0]["turn"], 1)
        self.assertEqual(turns[0]["of"], main.settings.AGENT_MAX_TURNS)
        self.assertTrue(turns[0]["tools_offered"])

    async def test_tool_calls_and_results_are_narrated_in_order(self):
        emitted = []

        async def stats_then_answer(messages, system_prompt, tools, *a, **kw):
            if tools and not emitted:
                emitted.append(1)
                return {"content": "Let me total these up.", "tool_calls": [
                    {"id": "c1", "name": "calculate_statistics",
                     "arguments": {"table_name": "sales", "operation": "sum",
                                   "column": "revenue"}}]}
            return {"content": FINAL, "tool_calls": []}

        events = await self.events(side_effect=stats_then_answer)
        kinds = [e["type"] for e in events]

        self.assertLess(kinds.index("tool_call"), kinds.index("tool_result"))
        self.assertIn("thinking", kinds, "reasoning before a tool call is worth showing")

        call = next(e for e in events if e["type"] == "tool_call")
        result = next(e for e in events if e["type"] == "tool_result")
        self.assertEqual(call["name"], "calculate_statistics")
        self.assertEqual(result["name"], "calculate_statistics")
        self.assertFalse(result["failed"])
        self.assertTrue(result["summary"])

    async def test_a_failing_tool_is_marked_as_such(self):
        emitted = []

        async def bad_tool(messages, system_prompt, tools, *a, **kw):
            if tools and not emitted:
                emitted.append(1)
                return {"content": "", "tool_calls": [
                    {"id": "c1", "name": "calculate_statistics",
                     "arguments": {"table_name": "nope", "operation": "sum",
                                   "column": "revenue"}}]}
            return {"content": FINAL, "tool_calls": []}

        results = [e for e in await self.events(side_effect=bad_tool)
                   if e["type"] == "tool_result"]
        self.assertTrue(results)
        self.assertIn("not found", results[0]["summary"])

    async def test_a_failover_notice_is_streamed_when_it_happens(self):
        degraded = {"content": FINAL, "tool_calls": [],
                    "degraded": "gpt-4o failed (HTTP 429); answered with gemini instead"}
        notices = [e for e in await self.events(side_effect=AsyncMock(return_value=degraded))
                   if e["type"] == "notice"]
        self.assertTrue(notices)
        self.assertIn("429", notices[0]["text"])

    async def test_the_final_event_carries_what_chat_returns(self):
        """If these diverge, the streamed answer is a different answer."""
        with patch.object(main.llm_service, "chat", answer()):
            direct = await main.chat(ChatRequest(session_id="stream", query="revenue by region"))

        streamed = next(e for e in await self.events() if e["type"] == "final")["response"]
        self.assertEqual(streamed.response, direct.response)
        self.assertEqual(streamed.metadata["key_insights"], direct.metadata["key_insights"])
        self.assertEqual(streamed.session_id, direct.session_id)

    async def test_every_event_is_json_serialisable(self):
        """The stream is NDJSON; an unserialisable event would tear the response."""
        for event in await self.events():
            with self.subTest(kind=event["type"]):
                if event["type"] == "final":
                    event = {"type": "final", "response": event["response"].model_dump()}
                json.loads(json.dumps(event, default=str))


class ATruncatedAnswerSaysSo(unittest.IsolatedAsyncioTestCase):
    """
    Asked for a hundred rows as a table, the model emitted JSON longer than the token
    limit. It was cut mid-structure, the salvage path recovered the prose by regex, and
    the table went with it — leaving "here are the first 100 sales figures" above an empty
    panel, with no field declaring the language either.

    Salvage is the right behaviour; doing it silently is not. The reply now carries the
    fact that it was recovered, so the reader knows the figure it mentions is missing.
    """

    def test_a_salvaged_response_is_flagged(self):
        _, meta = main.extract_structured_response(
            '{"answer": "Here are the first 100 rows", "visualizations": [{"type"'
        )
        self.assertTrue(meta.get("salvaged"))
        self.assertEqual(meta["visualizations"], [])

    def test_a_well_formed_response_is_not_flagged(self):
        _, meta = main.extract_structured_response(
            json.dumps({"question_language": "English", "answer": "Total is 5.",
                        "key_insights": [], "visualizations": [], "sources_used": []})
        )
        self.assertFalse(meta.get("salvaged"))

    def test_the_answer_text_still_survives_salvage(self):
        answer, _ = main.extract_structured_response(
            '{"answer": "Revenue is 1,714,947.65 SAR", "visualizations": [{"typ'
        )
        self.assertIn("1,714,947.65", answer)

    def test_the_prompt_caps_a_composed_table(self):
        text = " ".join(prompts.get_system_prompt("", "", [], "", "q").lower().split())
        self.assertIn("cap it at 25 rows", text)
        self.assertIn("say how many rows exist in total", text)
