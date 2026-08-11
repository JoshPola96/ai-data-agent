"""Charting values lifted out of prose, and describing many small tables."""

import json
import unittest

import pandas as pd

from app.services.tools import build_table_context, execute_tool

# Shape the agent produces after reading figures out of a document: numbers arrive
# as JSON numbers or strings, and the x-axis is a period label, not a number.
PAYSLIPS = [
    {"date": "2019-06", "salary": 5561, "currency": "INR"},
    {"date": "2019-07", "salary": "11917", "currency": "INR"},
    {"date": "2022-07", "salary": 1392.1, "currency": "EUR"},
    {"date": "2022-08", "salary": "4426.92", "currency": "EUR"},
]


class CustomDataChartTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression: pandas 3 removed errors="ignore" from to_numeric.

    Passing it raised ValueError before any chart was drawn, so every figure built
    from custom_data failed — the whole "chart numbers found in a PDF" feature.
    """

    async def chart(self, **args):
        raw = await execute_tool("generate_chart", {"custom_data": PAYSLIPS, **args}, {})
        return json.loads(raw)

    async def test_line_chart_from_extracted_values(self):
        out = await self.chart(
            chart_type="line", x_column="date", y_column="salary", title="Salary"
        )
        self.assertTrue(out["success"], out.get("error"))

    async def test_numeric_strings_are_summed_as_numbers(self):
        out = await self.chart(
            chart_type="bar", x_column="currency", y_column="salary",
            aggregation="sum", title="Salary",
        )
        self.assertTrue(out["success"], out.get("error"))
        ys = out["chart_json"]["data"][0]["y"]
        ys = list(ys) if isinstance(ys, list) else None
        if ys:
            # 5561 + 11917 concatenated as strings would not equal the numeric sum
            self.assertIn(17478, [round(v) for v in ys])

    async def test_period_label_axis_is_not_blanked(self):
        """errors="coerce" alone would turn "2019-06" into NaN and empty the chart."""
        out = await self.chart(
            chart_type="line", x_column="date", y_column="salary", title="Salary"
        )
        self.assertTrue(out["success"], out.get("error"))
        xs = out["chart_json"]["data"][0]["x"]
        self.assertTrue(any("2019" in str(v) for v in xs), f"x-axis lost its labels: {xs}")

    async def test_split_by_currency_survives(self):
        out = await self.chart(
            chart_type="line", x_column="date", y_column="salary",
            color_column="currency", title="By currency",
        )
        self.assertTrue(out["success"], out.get("error"))
        names = sorted(t.get("name") or "" for t in out["chart_json"]["data"])
        self.assertEqual(names, ["EUR", "INR"])

    async def test_non_numeric_y_reports_cleanly(self):
        out = await self.chart(
            chart_type="line", x_column="date", y_column="currency", title="Bad"
        )
        self.assertIn("success", out)


class TableContextTest(unittest.TestCase):
    """
    Regression: only the newest five tables were described.

    With twenty-four one-row payslip tables the agent could not see the rest, so it
    probed them with query_data and exhausted its turn budget before answering.
    """

    def many(self, n):
        return {
            f"Payslip{i:02d}": pd.DataFrame({"Earnings": ["Gross"], "Amount": [100 + i]})
            for i in range(n)
        }

    def test_every_table_is_named_when_there_are_many(self):
        ctx = build_table_context(self.many(24))
        for i in range(24):
            self.assertIn(f"Payslip{i:02d}", ctx)

    def test_detailed_profiles_are_limited(self):
        ctx = build_table_context(self.many(24), profile_limit=5)
        self.assertEqual(ctx.count("**Preview**"), 5)

    def test_row_counts_are_reported_in_the_index(self):
        ctx = build_table_context(self.many(10))
        self.assertIn("1 rows", ctx)

    def test_few_tables_skip_the_index_and_profile_all(self):
        ctx = build_table_context(self.many(3))
        self.assertNotIn("All 3 tables", ctx)
        self.assertEqual(ctx.count("**Preview**"), 3)

    def test_no_tables_is_stated_plainly(self):
        self.assertIn("No structured data", build_table_context({}))

    def test_wide_tables_truncate_the_column_list(self):
        wide = {f"T{i}": pd.DataFrame({f"c{j}": [1] for j in range(20)}) for i in range(6)}
        ctx = build_table_context(wide)
        self.assertIn("more]", ctx)


if __name__ == "__main__":
    unittest.main()
