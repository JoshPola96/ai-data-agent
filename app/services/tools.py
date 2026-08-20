# app/services/tools.py

"""
Tool Execution Service - Enhanced with Comprehensive Logging
Supports: Chart generation, Statistics, Data queries, Knowledge search
"""

import logging
import json
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from app.core.config import get_settings
from app.services import diagram
from app.services.chart import ChartService
from app.utils.helpers import canonical_labels, to_numeric, totals_row_index

settings = get_settings()
logger = logging.getLogger(__name__)


def _casing_variants(series: pd.Series, limit: int = 200) -> str:
    """
    Report values that are the same label written differently, if any are.

    "North", "north", "NORTH" and " North" group as four categories, so a four-region
    dataset charts as eleven bars and the totals are wrong by construction. The signal
    is cheap to compute and impossible to see from a two-row sample.
    """
    # Text arrives as object or as pandas' str dtype depending on the build, so the test
    # is what the column is not: numbers and dates have no spelling variants.
    if series.dtype.kind in "ifbcmM":
        return ""

    text = series.dropna().astype(str)
    if text.empty or text.nunique() > limit:
        return ""

    raw = text.nunique()
    folded = text.str.strip().str.casefold().nunique()
    if folded >= raw:
        return ""

    canonical = sorted(text.str.strip().str.casefold().unique())[:6]
    return (
        f"{raw} spellings of {folded} distinct values (case/whitespace only): "
        f"{canonical}. Normalise before grouping, or the totals split across variants."
    )


def _redundant_measures(df: pd.DataFrame, tolerance: float = 1e-4) -> str:
    """
    Report numeric columns that are the same measure written in different units.

    A currency column and its converted twin hold the same money twice, and nothing in the
    schema says so — asked for "total revenue", the agent summed both. The test is a
    *constant ratio* between the columns, which is what a unit conversion is. Correlation
    would seem the obvious check and is the wrong one: a single large outlier pushes
    Pearson's r to 1.00 between quantities as unrelated as units sold and revenue.
    """
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2 or len(numeric) < 3:
        return ""

    columns = list(numeric.columns)
    pairs = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            ratio = (numeric[a] / numeric[b]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) < 3 or not ratio.any():
                continue
            # Relative spread, so the test is scale-free. The threshold sits in a wide
            # empty band: a currency pair rounded to two decimals measures 1.3e-06, while
            # the closest genuinely different pair in the same data — revenue against units
            # — measures 2.8e-01, five orders of magnitude away.
            spread = ratio.std(ddof=0) / abs(ratio.mean())
            if pd.notna(spread) and spread < tolerance:
                pairs.append(f"{a} = {b} × {ratio.mean():.4g}")

    if not pairs:
        return ""

    return (
        f"{'; '.join(pairs[:4])} — the same measure in different units. "
        "Report one, never the sum of both."
    )


def _totals_row(df: pd.DataFrame) -> str:
    """Describe a totals row still present in the data, for tables ingestion never saw."""
    row = totals_row_index(df)
    if row is None:
        return ""

    numeric = set(df.select_dtypes(include="number").columns)
    label = next(
        (str(df.loc[row, c]) for c in df.columns
         if c not in numeric and pd.notna(df.loc[row, c])),
        f"row {row}",
    )
    return (
        f"Row {row} ('{label}') is the sum of the other rows — a totals line loaded as "
        "data. Exclude it before aggregating, or every total counts twice."
    )


def _quality_warnings(df: pd.DataFrame) -> List[str]:
    """
    The measured flaws in a table, as short lines.

    Split out because a preview is expensive and a warning is not. Only the newest few
    tables get a full profile, and with ten loaded the oldest kept nothing but a column
    list — so the duplicate rows and the eight spellings of four regions vanished from the
    prompt entirely. Asked what was wrong with that file the agent said it looked fine,
    and it was right to: it had never been told otherwise. Warnings now travel with every
    table however many there are.
    """
    out = []
    for column in df.columns:
        variants = _casing_variants(df[column])
        if variants:
            out.append(f"'{column}': {variants}")

    totals = _totals_row(df)
    if totals:
        out.append(totals)

    redundant = _redundant_measures(df)
    if redundant:
        out.append(redundant)

    try:
        duplicates = int(df.duplicated().sum())
    except Exception:
        duplicates = 0
    if duplicates:
        out.append(
            f"{duplicates} exactly duplicated row(s): every total counts them twice."
        )
    return out


def _create_data_profile(df: pd.DataFrame, table_name: str) -> str:
    """
    Generate comprehensive data profile for LLM context.
    """
    try:
        lines = [
            f"### Table: '{table_name}' ({len(df)} rows, {len(df.columns)} columns)"
        ]

        # Remove duplicate columns
        df_clean = df.loc[:, ~df.columns.duplicated()]

        # Preview
        try:
            preview = df_clean.head(3).to_markdown(index=False)
            lines.append(f"\n**Preview**:\n{preview}\n")
        except Exception as e:
            logger.debug(f"Preview generation failed: {e}")

        # Column types
        numeric = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        datetime = df_clean.select_dtypes(include=["datetime64"]).columns.tolist()
        text = df_clean.select_dtypes(
            exclude=[np.number, "datetime64"]
        ).columns.tolist()

        lines.append("**Column Types**:")
        lines.append(f"  Numeric ({len(numeric)}): {numeric}")
        lines.append(f"  Datetime ({len(datetime)}): {datetime}")
        lines.append(f"  Text ({len(text)}): {text}\n")

        # Column details (first 10)
        lines.append("**Column Details**:")
        for col in df_clean.columns[:10]:
            dtype = str(df_clean[col].dtype)
            unique = df_clean[col].nunique()
            nulls = df_clean[col].isna().sum()

            # Two samples of a three-value column read as the whole set: shown
            # ['closed', 'open'], the agent described the column as "open vs. closed" and
            # a third of the orders vanished from its account of the data. Small
            # categoricals are cheap to state in full, so they are stated in full.
            if 0 < unique <= 12 and df_clean[col].dtype.kind not in "ifcmM":
                values = sorted(df_clean[col].dropna().astype(str).unique())
                detail = f"Values ({unique}): {values}"
            else:
                samples = df_clean[col].dropna().astype(str).head(2).tolist()
                detail = f"Samples: {samples}"

            lines.append(
                f"  • {col} ({dtype}) | Unique: {unique} | Nulls: {nulls} | {detail}"
            )

            # Grouping on raw strings turned four regions into eleven bars, and the model
            # normalised on some questions and not others. It cannot notice reliably from
            # two samples, so the collapse is measured here and stated as a fact.
            warning = _casing_variants(df_clean[col])
            if warning:
                lines.append(f"      ⚠ {warning}")

        if len(df_clean.columns) > 10:
            lines.append(f"  ... and {len(df_clean.columns) - 10} more columns")

        totals = _totals_row(df_clean)
        if totals:
            lines.append(f"\n**⚠ Totals row**: {totals}")

        redundant = _redundant_measures(df_clean)
        if redundant:
            lines.append(f"\n**⚠ Redundant measures**: {redundant}")

        # Two identical rows inflate every total by one order's worth, and no aggregate
        # reveals it. Counting them costs nothing and the reader deserves to know.
        try:
            duplicates = int(df_clean.duplicated().sum())
        except Exception:
            duplicates = 0
        if duplicates:
            lines.append(
                f"\n**⚠ {duplicates} exactly duplicated row(s)**: every total counts them "
                "twice. Say so when reporting one, and deduplicate if it is an error."
            )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Data profile generation failed: {e}")
        return f"Table: {table_name} (Error generating profile: {e})"


def build_table_context(dfs: Dict[str, pd.DataFrame], profile_limit: int = 5) -> str:
    """
    Describe every available table, profiling only the most recent few.

    A full profile costs real tokens, but omitting a table entirely is worse: the
    agent cannot know it exists and burns turns probing with query_data to find out.
    Twenty-four one-row payslip tables exhausted a ten-turn budget this way. Every
    table therefore gets a one-line schema; the newest get the detail.
    """
    if not dfs:
        return "No structured data available."

    newest_first = list(dfs.items())[::-1]
    lines = []

    if len(newest_first) > profile_limit:
        lines.append(f"**All {len(newest_first)} tables** (name: rows × columns):")
        for name, df in newest_first:
            cols = ", ".join(map(str, df.columns[:8]))
            more = f", +{len(df.columns) - 8} more" if len(df.columns) > 8 else ""
            lines.append(f"  • {name}: {len(df)} rows [{cols}{more}]")
            # Warnings ride with every table, profiled or not. They are what stops a
            # wrong answer, and they cost a line each.
            for warning in _quality_warnings(df):
                lines.append(f"      ⚠ {warning}")
        lines.append(f"\n**Detailed profiles** (newest {profile_limit}):")

    lines.extend(
        _create_data_profile(df, name) for name, df in newest_first[:profile_limit]
    )

    return "\n".join(lines)


def _parse_custom_data(custom_data: Any) -> List[Dict]:
    """
    Parse custom_data which may be list of dicts OR list of JSON strings (Gemini format).
    Preserves existing functionality - just adds JSON string support.
    """
    if not custom_data:
        return []

    # Already correct format - return as-is
    if isinstance(custom_data, list) and all(
        isinstance(item, dict) for item in custom_data
    ):
        return custom_data

    # List of JSON strings (Gemini sometimes does this)
    if isinstance(custom_data, list) and any(
        isinstance(item, str) for item in custom_data
    ):
        logger.debug("  🔧 Detected JSON strings in custom_data, parsing...")
        parsed = []
        for item in custom_data:
            if isinstance(item, dict):
                parsed.append(item)
            elif isinstance(item, str):
                try:
                    parsed.append(json.loads(item))
                except json.JSONDecodeError:
                    logger.warning(f"    ⚠️ Failed to parse: {item[:50]}...")
                    continue
        return parsed

    # Fallback - return as-is
    return custom_data


# Keys that select the data, not the chart; everything else is chart vocabulary
_DATA_SOURCE_KEYS = {
    "table_name",
    "custom_data",
    "charts",
    "filter_column",
    "filter_value",
    "join_table",
    "join_on",
}


def _filter_rows(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """
    Restrict rows to one or more values of a column.

    Only query_data could filter, so "just for branch B" produced filtered prose beside a
    chart of every branch — and the model then read its narrative off the unfiltered
    chart. Every tool that resolves a table now takes the same two arguments, so the
    numbers and the picture describe one population or neither does.

    Exact match wins, case- and space-insensitive; a comma-separated list is tried next
    so two subsets can be compared; substring is a last resort for a single value,
    because matching loosely lets 'card' also select 'discard'.
    """
    resolved = ChartService._fuzzy_col_match(df, str(column))
    if not resolved:
        raise KeyError(
            f"Cannot filter on '{column}': no such column. Available: {list(df.columns)}"
        )

    series = df[resolved]
    whole = str(value).strip()
    parts = [p.strip() for p in whole.split(",") if p.strip()]
    attempts = [[whole]] if len(parts) <= 1 else [[whole], parts]

    if pd.api.types.is_numeric_dtype(series):
        for wanted in attempts:
            numbers = to_numeric(pd.Series(wanted)).dropna()
            subset = df[series.isin(numbers)]
            if not subset.empty:
                return subset
    else:
        text = series.astype(str).str.strip().str.casefold()
        for wanted in attempts:
            subset = df[text.isin({w.casefold() for w in wanted})]
            if not subset.empty:
                return subset
        if len(parts) <= 1:
            subset = df[text.str.contains(whole.casefold(), regex=False, na=False)]
            if not subset.empty:
                return subset

    # An empty frame charts as an empty panel and aggregates to zero, both of which read
    # as findings. The values actually present turn that into a correctable mistake.
    present = series.astype(str).drop_duplicates().head(15).tolist()
    raise ValueError(
        f"No rows where {resolved} is {value!r}. Values present: {present}"
    )


def _group_keys(df: pd.DataFrame, group_by) -> List[str]:
    """
    Resolve one or several grouping columns.

    Accepting a comma-separated list lets "by region and category" be one call. With a
    single key the agent had to issue one call per combination, which cost fourteen
    tool calls and forty-seven seconds on a five-by-four grid.
    """
    if not group_by:
        return []

    requested = group_by if isinstance(group_by, list) else str(group_by).split(",")
    keys, missing = [], []
    for name in requested:
        name = name.strip()
        if not name:
            continue
        resolved = ChartService._fuzzy_col_match(df, name)
        if not resolved:
            missing.append(name)
        elif resolved not in keys:
            keys.append(resolved)

    # Silently dropping an unknown key returned the ungrouped total, and "Sum of revenue: 1,075,384" is indistinguishable from a real answer — the model reported one number as a breakdown. A grouping that cannot be honoured is an error, not a total.
    if missing:
        raise KeyError(
            f"Cannot group by {missing}: no such column. Available: {list(df.columns)}"
        )
    return keys


def _top_groups(result: pd.Series, top_n: Any) -> tuple:
    """
    Trim a grouped result to a ranking, and say that is what it is.

    "Top 5" printed without its denominator reads as "there are 5" — the same trap as a
    page of query rows read as the whole population.
    """
    if not top_n or len(result) <= int(top_n):
        return result, ""
    return result.nlargest(int(top_n)), f" (top {int(top_n)} of {len(result)})"


def _resolve_dataframe(args: Dict, dfs: Dict) -> tuple:
    """Resolve a tool's data source to (dataframe, error); exactly one of the two is None."""
    custom_data = args.get("custom_data")
    table_name = args.get("table_name")

    if custom_data:
        if len(custom_data) > settings.MAX_CUSTOM_DATA_ROWS:
            return None, (
                f"custom_data has {len(custom_data)} rows, limit is "
                f"{settings.MAX_CUSTOM_DATA_ROWS}. Aggregate before charting."
            )
        try:
            df = pd.DataFrame(_parse_custom_data(custom_data))
        except Exception as e:
            return None, f"Invalid custom_data: {e}"
        if df.empty:
            return None, "custom_data produced an empty table"
        df = df.loc[:, ~df.columns.duplicated()]
    elif table_name and table_name != "__none__":
        df = dfs.get(table_name)
        if df is None or df.empty:
            return None, f"Table '{table_name}' not found. Available: {list(dfs.keys())}"
        df = df.loc[:, ~df.columns.duplicated()]
    else:
        return None, "Must provide either 'table_name' or 'custom_data'"

    # Two sheets sharing a key had no numeric form: asked for cost per ticket by month, the
    # agent fetched spend by month and tickets by month, then did twelve divisions in its
    # head and passed the results in as custom_data. Eleven were right. June came out as
    # 720.08 against a true 719.50 — a figure indistinguishable from the correct one, in a
    # chart the reader has no way to check. Joining here means pandas divides, not the model.
    other, key = args.get("join_table"), args.get("join_on")
    if other and key:
        right = dfs.get(other)
        if right is None or right.empty:
            return None, f"Join table '{other}' not found. Available: {list(dfs.keys())}"

        left_key = ChartService._fuzzy_col_match(df, str(key))
        right_key = ChartService._fuzzy_col_match(right, str(key))
        if not left_key or not right_key:
            return None, (
                f"Cannot join on '{key}': present in "
                f"{'both' if left_key and right_key else 'only one table'}. "
                f"Left has {list(df.columns)}, right has {list(right.columns)}."
            )

        right = right.loc[:, ~right.columns.duplicated()]
        overlap = [c for c in right.columns if c in df.columns and c != right_key]
        merged = df.merge(
            right.rename(columns={right_key: left_key}).drop(columns=overlap),
            on=left_key,
            how="inner",
        )
        if merged.empty:
            return None, (
                f"Joining on '{left_key}' matched no rows. Left values look like "
                f"{df[left_key].astype(str).head(3).tolist()}, right like "
                f"{right[right_key].astype(str).head(3).tolist()}."
            )
        logger.info(
            f"  ⋈ Joined {other} on {left_key}: {len(merged)} rows, "
            f"{len(merged.columns)} columns"
        )
        df = merged

    column, value = args.get("filter_column"), args.get("filter_value")
    if column and value not in (None, ""):
        try:
            rows = len(df)
            df = _filter_rows(df, column, value)
            logger.info(f"  🔍 Filter {column}={value}: {len(df)} of {rows} rows")
        except (KeyError, ValueError) as e:
            return None, str(e)

    return df, None


def get_tool_definitions(
    dataframes: Dict[str, pd.DataFrame], query: str = ""
) -> List[Dict]:
    """
    Generate tool definitions with clear guidance on data sources.
    """
    table_names = list(dataframes.keys()) if dataframes else []

    # Stating the rule was not enough: three separate descriptions said to match the
    # user's language and two English questions still came back with Spanish chart
    # titles. Quoting the actual question beside the parameter that writes the title
    # gives the model something to match rather than a rule to remember — the same
    # reason the reply-language rule is repeated inside the final user turn.
    title_language = (
        f"Write it in the language of this question: \"{query[:200]}\"."
        if query.strip()
        else "Write it in the same language as the user's question."
    )
    title_rule = (
        f"{title_language} The reader sees it beside the answer, and a title in "
        "another language reads as a bug."
    )

    data_guidance = f"""
**AVAILABLE TABLES:** {", ".join(table_names) if table_names else "NONE"}

**CRITICAL DATA SOURCE RULES:**
1. IF data exists in a TABLE → use `table_name` parameter
2. IF data is in PDF/Document → extract to list of dicts, use `custom_data` parameter
3. NEVER use a filename as `table_name` (filenames are NOT tables)

**Example - Structured Data:**
{{"table_name": "sales_data", "chart_type": "bar", ...}}

**Example - Unstructured Data (from PDF):**
{{"custom_data": [{{"month": "Jan", "value": 100}}, ...], "chart_type": "line", ...}}
"""

    # One definition shared by every tool that reads a table: a filter the chart honours
    # but the statistics do not is how a figure and its picture come to disagree.
    # Shared by every tool that reads a table, for the same reason the filter is: a join the
    # chart honours but the statistics do not is a figure and a picture of two populations.
    join_params = {
        "join_table": {
            "type": "string",
            "description": (
                "Combine table_name with a second table before computing. Use whenever the "
                "answer needs columns from two tables — a spend column against a volume "
                "column, a rate against a headcount. Never divide two tools' results "
                "yourself; join them and let the ratio be computed."
            ),
            "enum": table_names if table_names else ["__none__"],
        },
        "join_on": {
            "type": "string",
            "description": (
                "The column both tables share, matched exactly — 'month', 'site', 'branch'. "
                "Rows present in only one table are dropped."
            ),
        },
    }

    filter_params = {
        "filter_column": {
            "type": "string",
            "description": (
                "Restrict to rows matching filter_value before anything else happens. "
                "Required whenever the question is about a subset — one branch, one "
                "site, one period."
            ),
        },
        "filter_value": {
            "type": "string",
            "description": (
                "Value to keep, matched exactly and ignoring case. Comma-separate to "
                "keep several, e.g. 'B, C' for a two-way comparison."
            ),
        },
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "generate_chart",
                "description": f"""
Create Plotly charts from tabular data.

{data_guidance}

**Supported Chart Types:**
- bar: categorical comparison; set `color_column` to break series out side by side
- line: trend over time or ordered categories
- pie: proportions of a whole (top 10 slices)
- scatter: relationship between two numeric columns
- histogram: distribution of one numeric column
- box: distribution and outliers, optionally split by a category on x
- heatmap: correlation matrix across all numeric columns (no x_column needed)

Use `top_n` for ranking questions ("top 10 products by revenue").
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to visualize (use ONLY if table exists)",
                            "enum": table_names if table_names else ["__none__"],
                        },
                        "custom_data": {
                            "type": "array",
                            "description": "Custom data as list of dicts (use for extracted PDF data)",
                            "items": {"type": "object"},
                        },
                        "chart_type": {
                            "type": "string",
                            "description": "Type of chart to create",
                            "enum": ["bar", "line", "pie", "scatter", "histogram", "box", "heatmap"],
                        },
                        "x_column": {
                            "type": "string",
                            "description": "Column name for X-axis",
                        },
                        "y_column": {
                            "type": "string",
                            "description": (
                                "Column for the Y-axis (optional for histogram/heatmap). "
                                "Accepts a ratio of two numeric columns as 'a/b' — use this "
                                "for rates and per-unit values, e.g. 'returns/units'."
                            ),
                        },
                        "aggregation": {
                            "type": "string",
                            "description": "Aggregation method",
                            "enum": ["sum", "mean", "count", "min", "max", "none"],
                            "default": "none",
                        },
                        "title": {
                            "type": "string",
                            "description": f"Chart title. {title_rule}",
                        },
                        "color_column": {
                            "type": "string",
                            "description": "Column to use for color grouping (optional)",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Keep only the highest N rows by y_column, for ranking questions",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "y2_column": {
                            "type": "string",
                            "description": (
                                "Second measure drawn against its own right-hand axis. Use "
                                "when two series belong together but not on one scale, e.g. "
                                "revenue with margin percentage."
                            ),
                        },
                        "normalize_keys": {
                            "type": "boolean",
                            "description": (
                                "Collapse x/colour values that differ only in case or "
                                "surrounding space before grouping. Set it when the profile "
                                "warns of several spellings of one value."
                            ),
                        },
                        "y2_aggregation": {
                            "type": "string",
                            "enum": ["mean", "sum", "min", "max", "median"],
                            "description": (
                                "How to aggregate y2_column. Defaults to mean, which is what a "
                                "percentage or rate needs — summing one is meaningless. Pass "
                                "'sum' only when the second measure is a true total."
                            ),
                        },
                        "reference": {
                            "type": "string",
                            "description": (
                                "Draw a baseline: 'mean', 'median', or a number. Gives a "
                                "comparison something to be measured against."
                            ),
                        },
                        "resample": {
                            "type": "string",
                            "description": (
                                "Bucket a date x_column before aggregating: D, W, M, Q or Y. "
                                "Use for daily rows answering a monthly question."
                            ),
                            "enum": ["D", "W", "M", "Q", "Y"],
                        },
                        **filter_params,
                        **join_params,
                    },
                    "required": ["chart_type", "title"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_dashboard",
                "description": f"""
Create several related charts from one dataset in a single call.

Use when a question is better answered by a set of views than by one chart — for
example a trend over time alongside a breakdown by category and a distribution.
Prefer this over calling generate_chart repeatedly.

{data_guidance}
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to visualize (use ONLY if table exists)",
                            "enum": table_names if table_names else ["__none__"],
                        },
                        "custom_data": {
                            "type": "array",
                            "description": "Custom data as list of dicts (use for extracted PDF data)",
                            "items": {"type": "object"},
                        },
                        "charts": {
                            "type": "array",
                            "description": "Two to six chart specifications, all drawn from the same dataset",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chart_type": {
                                        "type": "string",
                                        "enum": [
                                            "bar",
                                            "line",
                                            "pie",
                                            "scatter",
                                            "histogram",
                                        ],
                                    },
                                    "x_column": {"type": "string"},
                                    "y_column": {"type": "string"},
                                    "aggregation": {
                                        "type": "string",
                                        "enum": [
                                            "sum",
                                            "mean",
                                            "count",
                                            "min",
                                            "max",
                                            "none",
                                        ],
                                    },
                                    "title": {
                                        "type": "string",
                                        "description": f"Panel title. {title_rule}",
                                    },
                                    "color_column": {"type": "string"},
                                    "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
                                    "y2_column": {"type": "string"},
                                    "y2_aggregation": {"type": "string", "enum": ["mean", "sum", "min", "max", "median"]},
                                    "normalize_keys": {"type": "boolean"},
                                    "reference": {"type": "string"},
                                    "resample": {"type": "string", "enum": ["D", "W", "M", "Q", "Y"]},
                                },
                                "required": ["chart_type", "title"],
                            },
                        },
                        **filter_params,
                        **join_params,
                    },
                    "required": ["charts"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_diagram",
                "description": f"""
Draw a diagram from Mermaid source: flowcharts, sequence diagrams, state machines,
entity relationships, timelines.

Use this whenever a document describes a process, an API exchange, a state machine or
a hierarchy — a flow is far clearer drawn than narrated. Read the document first with
search_knowledge_base, then express what it describes.

The first line must declare the type: {", ".join(diagram.SUPPORTED_TYPES)}.

Write labels plainly; punctuation is quoted for you. If the source is rejected, the
error names the line so you can correct it and call again.

**Example:**
flowchart TD
    A[Agent] -->|phone number| B[Backend checks registry]
    B --> C{{Known?}}
    C -->|yes| D[Send login link]
    C -->|no| E[Send signup link]
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mermaid": {
                            "type": "string",
                            "description": "Complete Mermaid source, starting with the diagram type",
                        },
                        "title": {
                            "type": "string",
                            "description": f"Short caption shown above the diagram. {title_rule}",
                        },
                    },
                    "required": ["mermaid", "title"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": """
Search the knowledge base for relevant document content.

Use this to retrieve information from uploaded PDFs, documents, or other text sources.
Especially useful for extracting data that needs to be visualized with custom_data.
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_statistics",
                "description": f"""
Calculate statistical measures on data.

{data_guidance}

**Operations:**
- sum, mean, median, std, min, max: Basic statistics
- count: Count rows/values
- describe: Full statistical summary
- correlation: Correlation matrix (numeric columns only)
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Table name (use ONLY if table exists)",
                            "enum": table_names if table_names else ["__none__"],
                        },
                        "custom_data": {
                            "type": "array",
                            "description": "Custom data as list of dicts",
                            "items": {"type": "object"},
                        },
                        "operation": {
                            "type": "string",
                            "description": "Statistical operation",
                            "enum": [
                                "sum",
                                "mean",
                                "median",
                                "std",
                                "min",
                                "max",
                                "count",
                                "describe",
                                "correlation",
                            ],
                        },
                        "column": {
                            "type": "string",
                            "description": (
                                "Column name (required for most operations). Accepts a "
                                "ratio of two numeric columns as 'a/b' for rates, e.g. "
                                "'returns/units' — aggregated as sum over sum."
                            ),
                        },
                        "normalize_keys": {"type": "boolean", "description": "Collapse values differing only in case or surrounding space before grouping. Set it when the profile warns of several spellings of one value, and set it on every call in the turn so the numbers and the chart agree."},
                        "group_by": {
                            "type": "string",
                            "description": (
                                "Group by one column, or several separated by commas for "
                                "combinations — 'region, category' answers 'by region and "
                                "category' in one call instead of one call per pair."
                            ),
                        },
                        "top_n": {
                            "type": "integer",
                            "description": (
                                "Return only the highest N groups. Use for ranking "
                                "questions instead of reading the winner out of a long "
                                "listing yourself."
                            ),
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "resample": {
                            "type": "string",
                            "description": (
                                "Bucket a date group_by column into D, W, M, Q or Y "
                                "periods before aggregating. Daily rows need this to "
                                "answer a monthly question numerically."
                            ),
                            "enum": ["D", "W", "M", "Q", "Y"],
                        },
                        **filter_params,
                        **join_params,
                    },
                    "required": ["operation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_data",
                "description": f"""
Filter and retrieve specific records from data.

{data_guidance}
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Table name (use ONLY if table exists)",
                            "enum": table_names if table_names else ["__none__"],
                        },
                        "custom_data": {
                            "type": "array",
                            "description": "Custom data as list of dicts",
                            "items": {"type": "object"},
                        },
                        **filter_params,
                        **join_params,
                        "sort_by": {
                            "type": "string",
                            "description": "Column to sort by",
                        },
                        "ascending": {
                            "type": "boolean",
                            "description": "Sort direction",
                            "default": True,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum rows to return",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                },
            },
        },
    ]

    logger.info(f"🔧 Generated {len(tools)} tool definitions")
    logger.info(f"  📊 Available tables: {table_names if table_names else 'None'}")

    return tools


async def execute_tool(
    name: str, args: Dict[str, Any], dfs: Dict[str, pd.DataFrame]
) -> str:
    """
    Execute a tool with comprehensive error handling and logging.
    """
    logger.info("=" * 60)
    logger.info(f"🔧 TOOL EXECUTION: {name}")
    logger.info(f"  Args: {json.dumps(args, indent=2, default=str)}")
    logger.info("=" * 60)

    try:
        if name == "generate_chart":
            result = await _generate_chart(args, dfs)
        elif name == "generate_dashboard":
            result = await _generate_dashboard(args, dfs)
        elif name == "generate_diagram":
            result = json.dumps(
                diagram.build(args.get("mermaid", ""), args.get("title", "Diagram"))
            )
        elif name == "calculate_statistics":
            result = await _calculate_statistics(args, dfs)
        elif name == "query_data":
            result = await _query_data(args, dfs)
        else:
            error_msg = f"Unknown tool: {name}"
            logger.error(f"  ❌ {error_msg}")
            result = json.dumps({"success": False, "error": error_msg})

        logger.info("  ✅ Tool executed successfully")
        logger.info("=" * 60)
        return result

    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        logger.error(f"  ❌ {error_msg}", exc_info=True)
        logger.info("=" * 60)
        return json.dumps({"success": False, "error": error_msg})


async def _generate_chart(args: Dict, dfs: Dict) -> str:
    """Generate chart with enhanced logging"""

    chart_type = args.get("chart_type")
    table_name = args.get("table_name")
    custom_data = args.get("custom_data")
    x_col = args.get("x_column")
    y_col = args.get("y_column")

    logger.info(f"  📊 Chart type: {chart_type}")
    logger.info(f"  📋 Table: {table_name}")
    logger.info(f"  📦 Custom data: {len(custom_data) if custom_data else 0} records")
    logger.info(f"  📏 Columns: X={x_col}, Y={y_col}")

    df, error = _resolve_dataframe(args, dfs)
    if error:
        logger.error(f"  ❌ {error}")
        return json.dumps({"success": False, "error": error})

    # Values lifted out of prose arrive as strings, so the plotted axes are coerced — but only when every value converts. pandas 3 removed errors="ignore", and errors="coerce" alone would blank a label column like "2019-06" into NaN.
    if custom_data:
        for col in (x_col, y_col):
            if col and col in df.columns:
                converted = to_numeric(df[col])
                if converted.notna().sum() == df[col].notna().sum():
                    df[col] = converted

    logger.info(f"  ✅ Data resolved: {df.shape}")

    # The chart service speaks the same vocabulary as the tool schema, so options pass straight through — a new one needs no plumbing here
    result = ChartService.generate_chart(
        df=df, **{k: v for k, v in args.items() if k not in _DATA_SOURCE_KEYS}
    )

    # Convert numpy types for JSON serialization
    if result.get("success") and result.get("chart_json"):
        result["chart_json"] = _convert_numpy_types(result["chart_json"])

    return json.dumps(result)


async def _generate_dashboard(args: Dict, dfs: Dict) -> str:
    """Build a set of related charts from one dataset."""
    charts_spec = args.get("charts") or []

    logger.info(f"  📊 Dashboard: {len(charts_spec)} charts requested")

    if not charts_spec:
        return json.dumps({"success": False, "error": "No chart specifications given"})

    df, error = _resolve_dataframe(args, dfs)
    if error:
        logger.error(f"  ❌ {error}")
        return json.dumps({"success": False, "error": error})

    result = ChartService.generate_multiple_charts(df, charts_spec)

    for chart in result.get("charts", []):
        chart["chart_json"] = _convert_numpy_types(chart["chart_json"])

    return json.dumps(result)


async def _calculate_statistics(args: Dict, dfs: Dict) -> str:
    """Calculate statistics with enhanced logging"""

    operation = args.get("operation")
    table_name = args.get("table_name")
    custom_data = args.get("custom_data")
    column = args.get("column")
    group_by = args.get("group_by")

    logger.info(f"  📊 Operation: {operation}")
    logger.info(f"  📋 Table: {table_name}")
    logger.info(f"  📦 Custom data: {len(custom_data) if custom_data else 0} records")
    logger.info(f"  📏 Column: {column}, Group by: {group_by}")

    df, error = _resolve_dataframe(args, dfs)
    if error:
        logger.error(f"  ❌ {error}")
        return error

    try:
        # Count operation
        keys = _group_keys(df, group_by)

        # Bucketing and ranking were chart-only, so a monthly total or a top-ten had no
        # numeric form — the model had to read one off its own picture, or eyeball the
        # winner out of every group. Both are the same tool asymmetry as the filter was.
        if args.get("resample") and keys:
            df = ChartService.bucket_dates(df, keys[0], args["resample"])

        # The same switch the chart tool takes, so a turn that normalises for the numbers
        # normalises for the picture too. Without it the agent cleaned the data for one
        # and not the other, and reported a leader its own chart contradicted.
        if args.get("normalize_keys"):
            df = df.copy()
            for key in keys:
                if df[key].dtype.kind not in "ifbcmM":
                    before = df[key].nunique()
                    df[key] = canonical_labels(df[key])
                    if df[key].nunique() < before:
                        logger.info(
                            f"  ⇢ Normalised '{key}': {before} labels to {df[key].nunique()}"
                        )

        if operation == "count":
            if keys:
                result, note = _top_groups(df.groupby(keys).size(), args.get("top_n"))
                output = f"Count by {keys}{note}:\n{result.to_string()}"
            else:
                output = f"Total rows: {len(df)}"

            logger.info(f"  ✅ {output}")
            return output

        # Describe operation
        if operation == "describe":
            result = df.describe()
            output = f"Statistical Summary:\n{result.to_string()}"
            logger.info("  ✅ Generated summary")
            return output

        # Correlation operation
        if operation == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                error_msg = "No numeric columns found for correlation"
                logger.error(f"  ❌ {error_msg}")
                return error_msg

            result = numeric_df.corr()
            output = f"Correlation Matrix:\n{result.to_string()}"
            logger.info("  ✅ Calculated correlation")
            return output

        # Column-based operations
        if not column:
            error_msg = f"Column required for '{operation}' operation"
            logger.error(f"  ❌ {error_msg}")
            return error_msg

        # "returns/units" is accepted here exactly as it is by the chart tools, and aggregates the same way: sum both sides, divide once.
        ratio = ChartService.resolve_ratio(df, column)
        if ratio:
            if keys:
                totals = df.groupby(keys)[[ratio["num"], ratio["den"]]].sum()
                rates = totals[ratio["num"]] / totals[ratio["den"]].replace(0, np.nan)
                result, note = _top_groups(rates, args.get("top_n"))
                output = (
                    f"Rate {ratio['num']}/{ratio['den']} by {keys}{note}:"
                    f"\n{result.to_string()}"
                )
            else:
                total = df[ratio["den"]].sum()
                rate = df[ratio["num"]].sum() / total if total else float("nan")
                output = f"Overall {ratio['num']}/{ratio['den']}: {rate}"

            logger.info(f"  ✅ {output[:120]}")
            return output

        column = ChartService._fuzzy_col_match(df, column) or column
        if column not in df.columns:
            return f"Column '{column}' not found. Available: {list(df.columns)}"

        # Convert to numeric
        df[column] = to_numeric(df[column])

        if keys:
            grouped = df.groupby(keys)[column].agg(operation)
            result, note = _top_groups(grouped, args.get("top_n"))
            output = (
                f"{operation.capitalize()} of {column} by {keys}{note}:"
                f"\n{result.to_string()}"
            )
        else:
            result = df[column].agg(operation)
            output = f"{operation.capitalize()} of {column}: {result}"

        logger.info(f"  ✅ {output}")
        return output

    except Exception as e:
        error_msg = f"Statistics calculation failed: {e}"
        logger.error(f"  ❌ {error_msg}")
        return error_msg

async def _query_data(args: Dict, dfs: Dict) -> str:
    """Query data with enhanced logging"""

    table_name = args.get("table_name")
    filter_column = args.get("filter_column")
    filter_value = args.get("filter_value")
    sort_by = args.get("sort_by")
    ascending = args.get("ascending", True)
    limit = min(args.get("limit", 10), 100)

    logger.info(f"  📋 Table: {table_name}")
    if filter_column:
        logger.info(f"  🔍 Filter: {filter_column}={filter_value}")
    logger.info(f"  📊 Sort: {sort_by} ({'ASC' if ascending else 'DESC'})")
    logger.info(f"  🔢 Limit: {limit}")

    df, error = _resolve_dataframe(args, dfs)
    if error:
        logger.error(f"  ❌ {error}")
        return error

    try:
        result = df.copy()

        # Apply sort
        if sort_by:
            result = result.sort_values(by=sort_by, ascending=ascending)
            logger.info(f"  📊 Sorted by: {sort_by}")

        # Apply limit
        matched = len(result)
        result = result.head(limit)

        if result.empty:
            output = "No results found."
            logger.info(f"  ⚠️  {output}")
            return output

        # "Query Results (10 rows)" over sixty matches invites the answer "there are ten". The page and the population are different numbers, so both are stated.
        header = (
            f"Query Results (showing {len(result)} of {matched} matching rows)"
            if matched > len(result)
            else f"Query Results ({len(result)} rows)"
        )
        output = f"{header}:\n{result.to_string(index=False)}"
        logger.info(f"  ✅ Returned {len(result)} rows")
        return output

    except Exception as e:
        error_msg = f"Query failed: {e}"
        logger.error(f"  ❌ {error_msg}")
        return error_msg


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    elif pd.isna(obj):
        return None
    else:
        return obj
