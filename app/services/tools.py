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

settings = get_settings()
logger = logging.getLogger(__name__)


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
            samples = df_clean[col].dropna().astype(str).head(2).tolist()

            lines.append(
                f"  • {col} ({dtype}) | Unique: {unique} | Nulls: {nulls} | Samples: {samples}"
            )

        if len(df_clean.columns) > 10:
            lines.append(f"  ... and {len(df_clean.columns) - 10} more columns")

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
_DATA_SOURCE_KEYS = {"table_name", "custom_data", "charts"}


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
        return df.loc[:, ~df.columns.duplicated()], None

    if table_name and table_name != "__none__":
        df = dfs.get(table_name)
        if df is None or df.empty:
            return None, f"Table '{table_name}' not found. Available: {list(dfs.keys())}"
        return df.loc[:, ~df.columns.duplicated()], None

    return None, "Must provide either 'table_name' or 'custom_data'"


def get_tool_definitions(dataframes: Dict[str, pd.DataFrame]) -> List[Dict]:
    """
    Generate tool definitions with clear guidance on data sources.
    """
    table_names = list(dataframes.keys()) if dataframes else []

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
                            "description": "Chart title",
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
                                    "title": {"type": "string"},
                                    "color_column": {"type": "string"},
                                    "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
                                    "y2_column": {"type": "string"},
                                    "reference": {"type": "string"},
                                    "resample": {"type": "string", "enum": ["D", "W", "M", "Q", "Y"]},
                                },
                                "required": ["chart_type", "title"],
                            },
                        },
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
                            "description": "Short caption shown above the diagram",
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
                        "group_by": {
                            "type": "string",
                            "description": "Group by column (optional)",
                        },
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
                        "filter_column": {
                            "type": "string",
                            "description": "Column to filter on",
                        },
                        "filter_value": {
                            "type": "string",
                            "description": "Value to filter for",
                        },
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

    # Values lifted out of prose arrive as strings, so the plotted axes are coerced —
    # but only when every value converts. pandas 3 removed errors="ignore", and
    # errors="coerce" alone would blank a label column like "2019-06" into NaN.
    if custom_data:
        for col in (x_col, y_col):
            if col and col in df.columns:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() == df[col].notna().sum():
                    df[col] = converted

    logger.info(f"  ✅ Data resolved: {df.shape}")

    # The chart service speaks the same vocabulary as the tool schema, so options
    # pass straight through — a new one needs no plumbing here
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

    if custom_data:
        try:
            df = pd.DataFrame(_parse_custom_data(custom_data))
            logger.info(f"  ✅ Loaded custom data: {df.shape}")
        except Exception as e:
            error_msg = f"Invalid custom_data: {e}"
            logger.error(f"  ❌ {error_msg}")
            return error_msg
    elif table_name and table_name != "__none__":
        df = dfs.get(table_name)
        if df is None:
            error_msg = f"Table '{table_name}' not found. Available: {list(dfs.keys())}"
            logger.error(f"  ❌ {error_msg}")
            return error_msg
    else:
        error_msg = "Must provide either 'table_name' or 'custom_data'"
        logger.error(f"  ❌ {error_msg}")
        return error_msg

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    try:
        # Count operation
        if operation == "count":
            if group_by:
                result = df.groupby(group_by).size()
                output = f"Count by {group_by}:\n{result.to_string()}"
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

        # "returns/units" is accepted here exactly as it is by the chart tools, and
        # aggregates the same way: sum both sides, divide once.
        ratio = ChartService.resolve_ratio(df, column)
        if ratio:
            if group_by:
                totals = df.groupby(group_by)[[ratio["num"], ratio["den"]]].sum()
                result = totals[ratio["num"]] / totals[ratio["den"]].replace(0, np.nan)
                output = f"Rate {ratio['num']}/{ratio['den']} by {group_by}:\n{result.to_string()}"
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
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if group_by:
            group_by = ChartService._fuzzy_col_match(df, group_by) or group_by
            result = df.groupby(group_by)[column].agg(operation)
            output = f"{operation.capitalize()} of {column} by {group_by}:\n{result.to_string()}"
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
    custom_data = args.get("custom_data")
    filter_column = args.get("filter_column")
    filter_value = args.get("filter_value")
    sort_by = args.get("sort_by")
    ascending = args.get("ascending", True)
    limit = min(args.get("limit", 10), 100)

    logger.info(f"  📋 Table: {table_name}")
    logger.info(f"  🔍 Filter: {filter_column}={filter_value}")
    logger.info(f"  📊 Sort: {sort_by} ({'ASC' if ascending else 'DESC'})")
    logger.info(f"  🔢 Limit: {limit}")

    # Get data source
    if table_name and table_name != "__none__":
        df = dfs.get(table_name)
        if df is None:
            error_msg = f"Table '{table_name}' not found"
            logger.error(f"  ❌ {error_msg}")
            return error_msg
    elif custom_data:
        try:
            # Parse JSON strings if present
            parsed_data = _parse_custom_data(custom_data)
            df = pd.DataFrame(parsed_data)
            logger.info(f"  ✅ Loaded custom data: {df.shape}")
        except Exception as e:
            error_msg = f"Invalid custom_data: {e}"
            logger.error(f"  ❌ {error_msg}")
            return error_msg
    else:
        error_msg = "Must provide either 'table_name' or 'custom_data'"
        logger.error(f"  ❌ {error_msg}")
        return error_msg

    df = df.loc[:, ~df.columns.duplicated()]

    try:
        result = df.copy()

        # Apply filter
        if filter_column and filter_value:
            if pd.api.types.is_numeric_dtype(df[filter_column]):
                try:
                    result = result[result[filter_column] == float(filter_value)]
                except (ValueError, TypeError):
                    result = result[
                        result[filter_column]
                        .astype(str)
                        .str.contains(str(filter_value), case=False, na=False)
                    ]
            else:
                result = result[
                    result[filter_column]
                    .astype(str)
                    .str.contains(str(filter_value), case=False, na=False)
                ]

            logger.info(f"  🔍 Filtered: {len(result)} rows")

        # Apply sort
        if sort_by:
            result = result.sort_values(by=sort_by, ascending=ascending)
            logger.info(f"  📊 Sorted by: {sort_by}")

        # Apply limit
        result = result.head(limit)

        if result.empty:
            output = "No results found."
            logger.info(f"  ⚠️  {output}")
            return output

        output = f"Query Results ({len(result)} rows):\n{result.to_string(index=False)}"
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
