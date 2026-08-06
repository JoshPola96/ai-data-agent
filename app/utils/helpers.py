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


def clean_filename(filename: str) -> str:
    """Removes special characters to make filenames safe for Redis keys/IDs."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)


def format_docs_for_prompt(docs: list) -> str:
    """Formats retrieved documents into a clean string for the LLM."""
    return "\n\n".join(
        [f"[Source: {d.get('source', 'Unknown')}]\n{d['content']}" for d in docs]
    )
