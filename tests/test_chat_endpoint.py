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

    async def test_malformed_model_output_degrades_instead_of_raising(self):
        with patch.object(main.llm_service, "chat", answer(content="not json at all")):
            resp = await self.ask()
        self.assertTrue(resp.response)


if __name__ == "__main__":
    unittest.main()
