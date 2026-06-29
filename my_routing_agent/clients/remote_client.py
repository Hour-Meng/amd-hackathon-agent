"""Fireworks AI client with strict JSON schema mapping."""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from my_routing_agent.clients.local_client import InferenceResponse
from my_routing_agent.config import RemoteConfig


class RemoteClient:
    """OpenAI-compatible Fireworks AI chat completions with JSON mode."""

    DEFAULT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Direct answer without filler."},
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0 and 1.",
            },
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    def __init__(self, config: RemoteConfig | None = None) -> None:
        self._config = config or RemoteConfig()
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
        if not self._config.api_key:
            return False
        try:
            response = self._session.get(
                self._config.base_url.rstrip("/") + "/models",
                timeout=8,
            )
            return response.status_code < 500
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        json_mode: bool = True,
        schema: dict[str, Any] | None = None,
    ) -> InferenceResponse:
        if not self._config.api_key:
            return InferenceResponse(
                content="",
                model=self._config.model,
                success=False,
                error="FIREWORKS_API_KEY is not configured.",
            )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._config.max_tokens,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "routing_agent_response",
                    "schema": schema or self.DEFAULT_SCHEMA,
                    "strict": True,
                },
            }

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

            parsed: dict[str, Any] | None = None
            if json_mode and content:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None

            return InferenceResponse(
                content=content,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
                latency_ms=latency_ms,
                model=body.get("model", self._config.model),
                raw=body,
                success=True,
                parsed_json=parsed,
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

    @staticmethod
    def build_schema(
        properties: dict[str, Any],
        *,
        required: list[str] | None = None,
        name: str = "custom_response",
    ) -> dict[str, Any]:
        """Helper to construct strict JSON schemas for structured remote outputs."""
        req = required or list(properties.keys())
        return {
            "type": "object",
            "properties": properties,
            "required": req,
            "additionalProperties": False,
        }
