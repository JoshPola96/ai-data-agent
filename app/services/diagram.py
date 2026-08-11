# app/services/diagram.py

"""
Mermaid diagram generation.

The agent writes mermaid fluently and escapes it badly. An unquoted parenthesis
inside a node label ends the node early and aborts the entire diagram — which is
how four consecutive flowcharts came back as nothing but parse errors:

    A[GET /v2/auth/agent/phone-status (phone_number, x-agent-secret)]
                                      ^ mermaid reads this as a new node shape

Labels are therefore quoted here rather than trusted, and a diagram whose brackets
still do not balance is reported as an error instead of rendered as a broken image.
"""

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = (
    "flowchart",
    "graph",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
    "stateDiagram-v2",
    "erDiagram",
    "journey",
    "gantt",
    "mindmap",
    "timeline",
)

# An identifier followed immediately by a shape opener, closed before whitespace,
# an arrow, or the end of the line. Lazy so the close bracket is the first one that
# actually terminates the node, not the first one that merely looks like it.
_NODE = re.compile(
    r"(?P<id>\b[A-Za-z_][\w-]*)"
    r"(?P<open>\[\[|\(\(|\[|\(|\{)"
    r'(?!")'
    r"(?P<text>.+?)"
    r"(?P<close>\]\]|\)\)|\]|\)|\})"
    r"(?=\s|$|-|=|\||;|,|&)"
)

# Edge labels: A -->|text| B
_EDGE_LABEL = re.compile(r"\|(?!\")(?P<text>[^|\n]+)\|")

# Characters that end a label early if left unquoted
_BREAKS_PARSING = re.compile(r'[()\[\]{}:;#"]')

_FENCE = re.compile(r"^```(?:mermaid)?\s*|\s*```$", re.DOTALL)

# A label containing an arrow means a bracket was left open and the lazy match ran
# past it. Quoting that would hide a real error behind a valid-looking diagram.
_ARROW_IN_LABEL = re.compile(r"--+>|==+>|-\.->|--+\s|==+\s")


def _quote(text: str) -> str:
    """Mermaid has no backslash escape; a literal quote becomes its entity."""
    return text.replace('"', "#quot;")


def _sanitize_line(line: str) -> str:
    """Quote node and edge labels whose punctuation would otherwise end them."""

    def node(match: re.Match) -> str:
        text = match.group("text")
        if not _BREAKS_PARSING.search(text) or _ARROW_IN_LABEL.search(text):
            return match.group(0)
        return (
            f"{match.group('id')}{match.group('open')}"
            f'"{_quote(text)}"'
            f"{match.group('close')}"
        )

    def edge(match: re.Match) -> str:
        text = match.group("text")
        if not _BREAKS_PARSING.search(text):
            return match.group(0)
        return f'|"{_quote(text)}"|'

    return _EDGE_LABEL.sub(edge, _NODE.sub(node, line))


def _unbalanced(line: str) -> bool:
    """True when brackets outside quoted spans do not close."""
    depth = {"[": 0, "(": 0, "{": 0}
    closers = {"]": "[", ")": "(", "}": "{"}
    quoted = False

    for char in line:
        if char == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif char in depth:
            depth[char] += 1
        elif char in closers:
            depth[closers[char]] -= 1

    return any(v != 0 for v in depth.values())


def build(mermaid: str, title: str = "Diagram") -> Dict[str, Any]:
    """
    Validate and repair a mermaid source.

    Returns {"success", "diagram_type", "mermaid", "title", "summary"} or
    {"success": False, "error"} with a message the agent can act on.
    """
    source = _FENCE.sub("", (mermaid or "").strip()).strip()

    if not source:
        return {"success": False, "error": "No diagram source provided"}

    header = next((ln.strip() for ln in source.splitlines() if ln.strip()), "")
    kind = header.split()[0] if header else ""

    if kind not in SUPPORTED_TYPES:
        return {
            "success": False,
            "error": (
                f"First line must declare a diagram type, got '{header[:40]}'. "
                f"Supported: {', '.join(SUPPORTED_TYPES)}"
            ),
        }

    lines = [_sanitize_line(ln) for ln in source.splitlines()]

    for number, line in enumerate(lines, 1):
        if _unbalanced(line):
            return {
                "success": False,
                "error": f"Unbalanced brackets on line {number}: {line.strip()[:80]}",
            }

    repaired = sum(1 for a, b in zip(source.splitlines(), lines) if a != b)
    if repaired:
        logger.info(f"  🧹 Quoted labels on {repaired} line(s)")

    logger.info(f"  📐 {kind} diagram, {len(lines)} lines")

    return {
        "success": True,
        "diagram_type": kind,
        "title": title,
        "mermaid": "\n".join(lines),
        "summary": f"{kind} diagram, {len(lines)} lines"
        + (f", {repaired} label(s) quoted" if repaired else ""),
    }
