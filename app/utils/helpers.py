# utils/helpers.py

import re
import logging


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


# Coarse script detection. The point is not to name the language exactly but to pin
# the reply to the script the user wrote in: a general "match the user's language"
# instruction loses to a prompt carrying thousands of characters of Arabic context.
_SCRIPTS = {
    "Arabic": ("؀", "ۿ"),
    "Cyrillic": ("Ѐ", "ӿ"),
    "Devanagari": ("ऀ", "ॿ"),
    "Hebrew": ("֐", "׿"),
    "Greek": ("Ͱ", "Ͽ"),
    "Thai": ("฀", "๿"),
    "CJK": ("぀", "鿿"),
}


def dominant_script(text: str) -> str:
    """Name the script the text is mostly written in, or 'Latin'."""
    counts = {
        name: sum(1 for c in text if low <= c <= high)
        for name, (low, high) in _SCRIPTS.items()
    }
    latin = sum(1 for c in text if c.isascii() and c.isalpha())

    script, hits = max(counts.items(), key=lambda item: item[1])
    return script if hits > latin else "Latin"


def clean_filename(filename: str) -> str:
    """Removes special characters to make filenames safe for Redis keys/IDs."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)


def format_docs_for_prompt(docs: list) -> str:
    """Formats retrieved documents into a clean string for the LLM."""
    return "\n\n".join(
        [f"[Source: {d.get('source', 'Unknown')}]\n{d['content']}" for d in docs]
    )
