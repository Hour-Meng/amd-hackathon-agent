"""Production text preprocessing for cache embeddings and similarity."""

from __future__ import annotations

import re
import unicodedata

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def preprocess_for_cache(text: str) -> str:
    """
    Production preprocessing pipeline:
    lowercase → NFC normalize → strip punctuation → collapse whitespace.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text.strip().lower())
    no_punct = _PUNCTUATION_PATTERN.sub(" ", normalized)
    return _WHITESPACE_PATTERN.sub(" ", no_punct).strip()
