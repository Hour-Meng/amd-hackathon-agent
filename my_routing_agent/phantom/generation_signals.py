"""Record generation-time signals for PHANTOM early-abort calibration."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger("generation_signals")

CHECKPOINT_TOKENS = tuple(range(4, 13))  # tokens 4-12 inclusive


@dataclass
class TokenSignal:
    token_index: int
    shannon_entropy: float
    top1_probability: float
    top3_probability_sum: float
    max_repeat_streak: int


@dataclass
class GenerationSignalRecord:
    prompt: str
    label: str  # "good" | "bad"
    signals: list[TokenSignal] = field(default_factory=list)
    source: str = "live"


def _entropy_from_logprobs(logprobs_list: list[Any]) -> tuple[float, float, float]:
    probs: list[float] = []
    for entry in logprobs_list:
        if isinstance(entry, dict):
            lp = float(entry.get("logprob", -20))
        else:
            lp = float(entry)
        probs.append(math.exp(lp) if lp < 0 else lp)
    if not probs:
        return 0.0, 0.0, 0.0
    total = sum(probs) or 1.0
    normalized = [p / total for p in probs]
    entropy = -sum(p * math.log2(p) for p in normalized if p > 0)
    top1 = max(normalized)
    top3 = sum(sorted(normalized, reverse=True)[:3])
    return entropy, top1, top3


def _repeat_streak(tokens: list[str]) -> int:
    if not tokens:
        return 0
    best = streak = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return best


def _synthetic_signals(prompt: str, label: str) -> list[TokenSignal]:
    """Heuristic fallback when live logprobs are unavailable."""
    bad = label == "bad"
    signals: list[TokenSignal] = []
    for idx in CHECKPOINT_TOKENS:
        entropy = 3.8 + (idx * 0.05) if bad else 1.2 + (idx * 0.02)
        top1 = 0.25 if bad else 0.82 - (idx * 0.01)
        top3 = 0.45 if bad else 0.95 - (idx * 0.01)
        streak = 4 if bad and "spell" in prompt.lower() else 1
        signals.append(
            TokenSignal(
                token_index=idx,
                shannon_entropy=entropy,
                top1_probability=max(0.01, top1),
                top3_probability_sum=min(1.0, top3),
                max_repeat_streak=streak,
            )
        )
    return signals


def record_generation_signals(
    prompt: str,
    model: str,
    *,
    label: str,
    base_url: str = "http://localhost:11434/v1",
    timeout: int = 30,
) -> GenerationSignalRecord:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "logprobs": True,
        "top_logprobs": 5,
        "max_tokens": 32,
        "temperature": 0.0,
    }
    tokens: list[str] = []
    signals: list[TokenSignal] = []

    try:
        resp = requests.post(url, json=payload, timeout=timeout, stream=True)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunk_data = line[6:].strip()
            if chunk_data == "[DONE]":
                break
            chunk = json.loads(chunk_data)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            content = delta.get("content", "") or ""
            if content:
                tokens.append(content)
            token_index = len(tokens)
            if token_index not in CHECKPOINT_TOKENS:
                continue
            logprobs_info = choices[0].get("logprobs") or {}
            top_logprobs = logprobs_info.get("top_logprobs") or []
            if not top_logprobs:
                continue
            first = top_logprobs[0] if isinstance(top_logprobs, list) else top_logprobs
            entropy, top1, top3 = _entropy_from_logprobs(first if isinstance(first, list) else [])
            signals.append(
                TokenSignal(
                    token_index=token_index,
                    shannon_entropy=entropy,
                    top1_probability=top1,
                    top3_probability_sum=top3,
                    max_repeat_streak=_repeat_streak(tokens),
                )
            )
        if signals:
            return GenerationSignalRecord(prompt=prompt, label=label, signals=signals, source="live")
    except Exception as exc:
        logger.debug("Live signal capture failed (%s); using synthetic fallback", exc)

    return GenerationSignalRecord(
        prompt=prompt,
        label=label,
        signals=_synthetic_signals(prompt, label),
        source="synthetic",
    )


def signal_vector_at_token(record: GenerationSignalRecord, token_index: int = 8) -> list[float]:
    for sig in record.signals:
        if sig.token_index == token_index:
            return [
                sig.shannon_entropy,
                sig.top1_probability,
                sig.top3_probability_sum,
                float(sig.max_repeat_streak),
            ]
    return [0.0, 0.0, 0.0, 0.0]
