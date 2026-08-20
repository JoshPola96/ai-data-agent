# utils/helpers.py

import re
import logging
import unicodedata

import pandas as pd

# Commas are stripped only where they sit in thousands positions.
_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _strip_currency(text: str) -> str:
    """Drop leading currency marks, whichever currency it is."""
    # Unicode's currency-symbol category rather than a hand-picked list, so a rand,
    # zloty or shekel sign works without anyone having remembered to add it.
    return text.lstrip("".join(
        ch for ch in set(text) if unicodedata.category(ch) == "Sc"
    ) + " ").strip()


def clean_number(value):
    """Strip human formatting from one value, leaving something pd.to_numeric can read."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    # Accounting notation: (300) is negative three hundred
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()

    text = _strip_currency(text)
    if text.startswith("-"):
        negative, text = True, text[1:].strip()
    text = text.rstrip("%").strip().replace(" ", "")

    if _THOUSANDS.match(text):
        text = text.replace(",", "")

    return f"-{text}" if negative and text else text


def canonical_labels(series: "pd.Series") -> "pd.Series":
    """
    Collapse values that differ only in case or surrounding space.

    The surviving label is the most common original spelling rather than a title-cased
    guess, so "iPhone" does not become "Iphone" on the way through. Without this the
    agent had to hand-roll normalisation per call: it cleaned the data for the
    statistics and charted the raw table, then reported a leader the chart contradicted.
    """
    text = series.astype(str).str.strip()
    key = text.str.casefold()
    winner = text.groupby(key).agg(lambda group: group.value_counts().idxmax())
    return key.map(winner)


def to_numeric(series: "pd.Series") -> "pd.Series":
    """
    Coerce a column to numbers the way spreadsheets and PDFs actually write them.

    Handing "1,200.50" straight to pd.to_numeric returns NaN. On a four-row payslip
    column that left exactly one parseable value, and the tool reported it as the total:
    900.25 where the answer was 5,501.50, with nothing on screen to suggest a problem.
    Model-extracted `custom_data` is the worst case, since figures lifted out of a
    document arrive formatted for a human reader.
    """
    if series.dtype.kind in "ifb":
        return series
    return pd.to_numeric(series.map(clean_number), errors="coerce")


def setup_logger(name: str, level=logging.INFO):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def clean_filename(filename: str) -> str:
    """Removes special characters to make filenames safe for Redis keys/IDs."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)


def format_docs_for_prompt(docs: list) -> str:
    """Formats retrieved documents into a clean string for the LLM."""
    return "\n\n".join(
        [f"[Source: {d.get('source', 'Unknown')}]\n{d['content']}" for d in docs]
    )


# Only consulted when a single numeric column leaves the arithmetic uncorroborated. Kept
# short and literal: a wide list would start deleting rows about totals rather than totals.
_TOTAL_WORDS = (
    "total", "totals", "subtotal", "sub-total", "grand total", "sum", "overall",
    "aggregate", "المجموع", "الإجمالي", "总计", "合計", "gesamt", "totaal", "totale",
)


def _reads_as_total(label: str) -> bool:
    text = label.strip().lower().strip(":-— ")
    return any(text == w or text.startswith(w + " ") or text.endswith(" " + w)
               for w in _TOTAL_WORDS)


def totals_row_index(frame: "pd.DataFrame", tolerance: float = 1e-6):
    """
    The index of a row that is the sum of the other rows, if there is one.

    Exported spreadsheets end with a TOTAL line. Loaded as data it is one more row, so
    every sum doubles. The test is arithmetic rather than a search for the word "total":
    such a row satisfies `column total == 2 x its own value`, whatever language its label
    is written in, and it has to agree across at least two numeric columns before it
    counts — one column alone can coincide.
    """
    numeric = frame.select_dtypes(include="number")
    if numeric.empty or len(numeric) < 3:
        return None

    # Only the final row is a candidate. Arithmetic alone is not evidence of a totals
    # line, however many columns agree: in a real headcount sheet the largest site had
    # 42 engineers and 18 support against 27+15 and 11+7 for the other two, so both
    # columns concurred and Riyadh was deleted from the data. The site simply stopped
    # existing, and the agent reported two sites where there were three. An export writes
    # its TOTAL at the end, so position is the corroboration the arithmetic lacks.
    last = frame.index[-1]

    votes = {}
    for column in numeric.columns:
        values = numeric[column].dropna()
        if len(values) < 3 or last not in values.index:
            continue
        total = values.sum()
        if not total:
            continue
        if abs(total - 2 * values.loc[last]) <= abs(total) * tolerance:
            votes[last] = votes.get(last, 0) + 1

    if not votes:
        return None
    row, agreeing = max(votes.items(), key=lambda pair: pair[1])
    if agreeing >= 2:
        return row

    # One numeric column offers no corroboration, and the arithmetic alone is far too
    # weak: in any three-row table where the last value happens to equal the first two,
    # a legitimate row was being deleted as a totals line. It cost a real answer twice —
    # scores of 1, 2, 3 averaged 1.5 because the 3 was dropped, and quantities of 5, 10,
    # 15 totalled 15. So a lone numeric column has to be backed by the row's own labels,
    # which on a genuine totals line are either blank or say so.
    # An entirely empty column is an absent column, not a blank label. Counting it as one
    # made every last row look labelled-blank: a table of 1, 2, 3 beside a column of nulls
    # lost its third row, and the total fell from 6 to 3.
    labels = [
        str(frame.loc[row, c]).strip()
        for c in frame.columns
        if c not in numeric.columns and frame[c].notna().any()
    ]
    if not labels:
        return None
    if all(v == "" or v.lower() in {"nan", "none", "nat"} for v in labels):
        return row
    return row if any(_reads_as_total(v) for v in labels) else None
