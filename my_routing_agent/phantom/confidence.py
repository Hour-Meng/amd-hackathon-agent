"""PHANTOM A — Confidence Predictor with Entropy Early Abort.

Measures Shannon entropy of the local model's next-token probability distribution
at generation token N. If the model is already confused (high entropy), abort
before wasting compute and escalate to remote.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

import requests

from my_routing_agent.config import PhantomConfig

logger = logging.getLogger("phantom_confidence")

LOCAL_CHAT_ENDPOINT = "http://localhost:11434/v1/chat/completions"

ENTROPY_SYSTEM_PROMPT = (
    "You are a concise, direct answering agent. "
    "Answer in under 15 words. No greetings, no filler."
)


class ConfidencePredictor:
    """Stream local model output, measure entropy at checkpoint, abort if confused."""

    def __init__(self, config: PhantomConfig | None = None) -> None:
        cfg = config or PhantomConfig()
        self._check_at: int = cfg.entropy_check_token
        self._abort_threshold: float = cfg.entropy_abort_threshold

    def speculative_execute_local(
        self,
        prompt: str,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        system_prompt: str | None = None,
        timeout: int = 30,
    ) -> tuple[str | None, str, float]:
        """
        Stream local model output token by token via OpenAI-compatible endpoint.
        At token CHECK_AT, compute H(Y) from top logprobs.
        If H(Y) > ENTROPY_ABORT_THRESHOLD, abort and return (None, "entropy_abort").
        Otherwise return (output, "local_complete").

        Returns (output, status, entropy_at_check).
        """
        url = base_url.rstrip("/") + "/chat/completions"
        sys_msg = system_prompt or ENTROPY_SYSTEM_PROMPT
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 128,
            "temperature": 0.0,
        }
        tokens_seen: list[str] = []
        entropy_at_check: float = 0.0
        started = time.perf_counter()

        try:
            resp = requests.post(url, json=payload, timeout=timeout, stream=True)
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                chunk_data = line[6:].strip()
                if chunk_data == "[DONE]":
                    break
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
                    tokens_seen.append(content)

                logprobs_info = choices[0].get("logprobs", None)
                token_index = len(tokens_seen) - 1

                if token_index == self._check_at and logprobs_info:
                    top_logprobs = logprobs_info.get("top_logprobs", []) or []
                    if top_logprobs:
                        logprobs_list = top_logprobs[0] if isinstance(top_logprobs, list) else top_logprobs
                        probs = []
                        for lp_entry in logprobs_list:
                            if isinstance(lp_entry, dict):
                                lp_val = lp_entry.get("logprob", -20)
                            elif isinstance(lp_entry, (int, float)):
                                lp_val = lp_entry
                            else:
                                continue
                            p = math.exp(lp_val) if lp_val < 0 else lp_val
                            probs.append(p)
                        if probs:
                            total_p = sum(probs) or 1.0
                            normalized = [p / total_p for p in probs]
                            entropy_at_check = -sum(p * math.log2(p) for p in normalized if p > 0)

                    if entropy_at_check > self._abort_threshold:
                        elapsed = (time.perf_counter() - started) * 1000
                        logger.info(
                            "ENTROPY ABORT at token %d H=%.4f threshold=%.2f elapsed=%.1fms",
                            self._check_at, entropy_at_check, self._abort_threshold, elapsed,
                        )
                        return None, "entropy_abort", entropy_at_check

            output = "".join(tokens_seen)
            elapsed = (time.perf_counter() - started) * 1000
            logger.info(
                "LOCAL COMPLETE tokens=%d entropy_check=%.4f elapsed=%.1fms",
                len(tokens_seen), entropy_at_check, elapsed,
            )
            return output, "local_complete", entropy_at_check

        except requests.Timeout:
            logger.warning("Local streaming timed out")
            return None, "timeout", entropy_at_check
        except requests.ConnectionError:
            logger.warning("Local connection failed")
            return None, "connection_error", entropy_at_check
        except requests.RequestException as exc:
            logger.warning("Local streaming error: %s", exc)
            return None, "error", entropy_at_check
        except Exception as exc:
            logger.warning("Unexpected error in speculative execution: %s", exc)
            return None, "error", entropy_at_check
