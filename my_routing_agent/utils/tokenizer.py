"""Instant local token estimation without API round-trips."""

from __future__ import annotations

import functools
from typing import Iterable

import tiktoken

from my_routing_agent.config import TokenizerConfig


@functools.lru_cache(maxsize=8)
def _get_encoding(name: str) -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


class TokenCounter:
    """CPU-only token counter backed by tiktoken encodings."""

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self._config = config or TokenizerConfig()
        self._default = _get_encoding(self._config.default_encoding)
        self._local = _get_encoding(self._config.local_model_encoding)
        self._remote = _get_encoding(self._config.remote_model_encoding)

    def count(self, text: str, *, target: str = "default") -> int:
        if not text:
            return 0
        encoding = self._select_encoding(target)
        return len(encoding.encode(text, disallowed_special=()))

    def count_messages(self, messages: Iterable[dict[str, str]], *, target: str = "default") -> int:
        total = 0
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            total += self.count(role, target=target)
            total += self.count(content, target=target)
            total += 4  # OpenAI-style per-message overhead
        return total + 2

    def _select_encoding(self, target: str) -> tiktoken.Encoding:
        if target == "local":
            return self._local
        if target == "remote":
            return self._remote
        return self._default


def estimate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Convenience function for one-off token estimates."""
    return len(_get_encoding(encoding_name).encode(text or "", disallowed_special=()))
