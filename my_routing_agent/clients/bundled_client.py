"""In-process GGUF inference via llama-cpp-python (no Ollama server required)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from my_routing_agent.clients.local_client import InferenceResponse
from my_routing_agent.config import LocalConfig

logger = logging.getLogger("bundled_client")

_LLM_LOCK = threading.Lock()
_LLM_CACHE: dict[str, Any] = {}


def _load_llama(model_path: str, *, n_ctx: int = 2048, n_threads: int = 2) -> Any:
    """Load (and cache) a llama-cpp Llama instance for the given GGUF path."""
    key = str(Path(model_path).resolve())
    with _LLM_LOCK:
        cached = _LLM_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required for bundled local inference. "
                "Install it or unset LOCAL_GGUF_PATH to use Ollama."
            ) from exc
        llm = Llama(
            model_path=key,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        _LLM_CACHE[key] = llm
        return llm


class BundledModelClient:
    """Local inference against a GGUF file bundled in the container/image."""

    def __init__(
        self,
        model_path: str,
        *,
        config: LocalConfig | None = None,
        n_ctx: int = 2048,
        n_threads: int = 2,
    ) -> None:
        self._config = config or LocalConfig()
        self.model_path = str(Path(model_path).expanduser())
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm: Any | None = None

    @property
    def model(self) -> str:
        return self._config.model or Path(self.model_path).name

    def health_check(self) -> bool:
        path = Path(self.model_path)
        if not path.is_file():
            return False
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:
            logger.warning("Bundled model health check failed: %s", exc)
            return False

    def _ensure_loaded(self) -> Any:
        if self._llm is None:
            self._llm = _load_llama(
                self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
            )
        return self._llm

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> InferenceResponse:
        del json_mode  # llama-cpp chat path does not enforce JSON mode here
        started = time.perf_counter()
        try:
            llm = self._ensure_loaded()
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens or self._config.max_tokens,
                temperature=temperature,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            usage = result.get("usage") or {}
            return InferenceResponse(
                content=content,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                latency_ms=latency_ms,
                model=str(result.get("model") or self.model),
                raw=result if isinstance(result, dict) else {},
                success=bool(content),
                error=None if content else "empty bundled response",
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.warning("Bundled chat failed: %s", exc)
            return InferenceResponse(
                content="",
                latency_ms=latency_ms,
                model=self.model,
                success=False,
                error=str(exc),
            )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> InferenceResponse:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
