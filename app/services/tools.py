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

from app.services.chart import ChartService

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
- bar: Bar chart (categorical comparisons)
- line: Line chart (trends over time)
- pie: Pie chart (proportions, max 10 categories)
- scatter: Scatter plot (correlations)
- histogram: Distribution histogram
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
                            "enum": ["bar", "line", "pie", "scatter", "histogram"],
                        },
                        "x_column": {
                            "type": "string",
                            "description": "Column name for X-axis",
                        },
                        "y_column": {
                            "type": "string",
                            "description": "Column name for Y-axis (optional for histogram)",
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
                    },
                    "required": ["chart_type", "x_column", "title"],
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
                            "description": "Column name (required for most operations)",
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

    df = None

    # Source 1: Custom data (from PDFs, etc.)
    if custom_data:
        try:
            # Parse JSON strings if present
            parsed_data = _parse_custom_data(custom_data)
            df = pd.DataFrame(parsed_data)
            logger.info(f"  ✅ Loaded custom data: {df.shape}")

            # Convert numeric columns
            for col in [x_col, y_col]:
                if col and col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                        logger.info(f"    📊 Converted {col} to numeric")
                    except (ValueError, TypeError):
                        logger.info(f"    📝 Kept {col} as-is (non-numeric)")

        except Exception as e:
            error_msg = f"Invalid custom_data: {e}"
            logger.error(f"  ❌ {error_msg}")
            return json.dumps({"success": False, "error": error_msg})

    # Source 2: Table (from structured files)
    elif table_name and table_name != "__none__":
        df = dfs.get(table_name)

        if df is None or df.empty:
            error_msg = f"Table '{table_name}' not found. Available: {list(dfs.keys())}"
            logger.error(f"  ❌ {error_msg}")
            return json.dumps({"success": False, "error": error_msg})

        logger.info(f"  ✅ Loaded table '{table_name}': {df.shape}")

    else:
        error_msg = "Must provide either 'table_name' or 'custom_data'"
        logger.error(f"  ❌ {error_msg}")
        return json.dumps({"success": False, "error": error_msg})

    # Generate chart
    result = ChartService.generate_chart(
        df=df,
        chart_type=chart_type,
        x_column=x_col,
        y_column=y_col,
        aggregation=args.get("aggregation", "none"),
        title=args.get("title", "Chart"),
        color_column=args.get("color_column"),
    )

    # Convert numpy types for JSON serialization
    if result.get("success") and result.get("chart_json"):
        result["chart_json"] = _convert_numpy_types(result["chart_json"])

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

        # Convert to numeric
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if group_by:
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
