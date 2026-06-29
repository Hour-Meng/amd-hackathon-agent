"""OpenAI-compatible client for local Ollama / CPU inference with logprobs streaming."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator

import requests

from my_routing_agent.config import LocalConfig

logger = logging.getLogger("local_client")


@dataclass
class InferenceResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    parsed_json: dict[str, Any] | None = None


class LocalClient:
    """Handles chat completions against a local OpenAI-compatible server."""

    def __init__(self, config: LocalConfig | None = None) -> None:
        self._config = config or LocalConfig()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            }
        )

    @property
    def model(self) -> str:
        return self._config.model

    def health_check(self) -> bool:
        try:
            response = self._session.get(
                self._config.base_url.rstrip("/") + "/models",
                timeout=5,
            )
            return response.status_code < 500
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> InferenceResponse:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = self._config.base_url.rstrip("/") + "/chat/completions"
        started = time.perf_counter()

        try:
            response = self._session.post(
                url,
                data=json.dumps(payload),
                timeout=self._config.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            body = response.json()
            choice = body.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = body.get("usage") or {}
            content = message.get("content") or ""

            return InferenceResponse(
                content=content,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                latency_ms=latency_ms,
                model=body.get("model", self._config.model),
                raw=body,
                success=True,
            )
        except requests.RequestException as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            detail = str(exc)
            if exc.response is not None:
                try:
                    detail = exc.response.text[:500]
                except Exception:
                    pass
            return InferenceResponse(
                content="",
                latency_ms=latency_ms,
                model=self._config.model,
                success=False,
                error=detail,
            )

    def stream_with_logprobs(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_logprobs: int = 5,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream chat completions with logprobs. Yields dicts with keys:
          - token: str (the delta content)
          - logprobs: list[dict] | None (top logprobs for this token)
          - token_index: int
          - finish_reason: str | None
        """
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "stream": True,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        try:
            resp = self._session.post(url, data=json.dumps(payload), timeout=self._config.timeout_seconds, stream=True)
            resp.raise_for_status()
            token_index = -1
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk_data = line[6:].strip()
                if chunk_data == "[DONE]":
                    yield {"token": "", "logprobs": None, "token_index": token_index, "finish_reason": "stop"}
                    return
                try:
                    chunk = json.loads(chunk_data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content", "") or ""
                if content:
                    token_index += 1
                logprobs_info = choices[0].get("logprobs")
                top_logprobs_list = None
                if logprobs_info:
                    top_logprobs_list = logprobs_info.get("top_logprobs")
                finish = choices[0].get("finish_reason")
                yield {
                    "token": content,
                    "logprobs": top_logprobs_list,
                    "token_index": token_index,
                    "finish_reason": finish,
                }
                if finish:
                    return
        except Exception as exc:
            logger.warning("stream_with_logprobs error: %s", exc)
            yield {"token": "", "logprobs": None, "token_index": -1, "finish_reason": "error"}

