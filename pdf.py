"""
PDF text extraction for RAG using pypdf.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF file. Returns concatenated page text.
    On failure (e.g. encrypted or corrupt) raises or returns empty string and logs.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed; cannot read PDF")
        return ""

    try:
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t and t.strip():
                parts.append(t.strip())
        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        logger.exception("PDF extract failed for %s: %s", path, e)
        raise
