"""Two-tier routing intelligence: static heuristics + semantic complexity scoring."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from my_routing_agent.config import RoutingThresholds
from my_routing_agent.middleware.compressor import ProcessedInput
from my_routing_agent.utils.tokenizer import TokenCounter


class RouteDecision(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class RoutingTier(str, Enum):
    TIER1_HEURISTIC = "tier1_heuristic"
    TIER2_SEMANTIC = "tier2_semantic"


@dataclass(frozen=True)
class RoutingResult:
    destination: RouteDecision
    tier: RoutingTier
    reason: str
    complexity_score: int
    estimated_tokens: int
    confidence: float


MATH_EXPRESSION = re.compile(
    r"^\s*(?:what\s+is\s+)?[\d\s+\-*/().,%^]+(?:=|\?)?\s*$",
    re.IGNORECASE,
)
SIMPLE_FACT = re.compile(
    r"^\s*(?:what|who|when|where|define|list|name)\s+(?:is|are|was|were)?\s*.{1,80}\?\s*$",
    re.IGNORECASE,
)
YES_NO = re.compile(r"^\s*(?:is|are|do|does|can|will|should|was|were)\s+.+?\?\s*$", re.IGNORECASE)

REASONING_KEYWORDS = (
    "analyze",
    "compare",
    "contrast",
    "evaluate",
    "explain why",
    "step by step",
    "prove",
    "derive",
    "debug",
    "root cause",
    "architecture",
    "design pattern",
    "optimize",
    "refactor",
    "implement",
    "algorithm",
    "trade-off",
    "reasoning",
    "multi-step",
)

CODE_KEYWORDS = (
    "python",
    "javascript",
    "typescript",
    "sql",
    "regex",
    "function",
    "class ",
    "api ",
    "code",
    "stack trace",
    "exception",
    "compile",
    "runtime",
    "```",
)

SIMPLE_KEYWORDS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhello\b", re.IGNORECASE),
    re.compile(r"\bhi\b", re.IGNORECASE),
    re.compile(r"\bthanks\b", re.IGNORECASE),
    re.compile(r"\bthank you\b", re.IGNORECASE),
    re.compile(r"\b(?:yes|no|ok)\b", re.IGNORECASE),
    re.compile(r"\bdefine\b", re.IGNORECASE),
    re.compile(r"\bcapital of\b", re.IGNORECASE),
    re.compile(r"\bconvert\b", re.IGNORECASE),
    re.compile(r"\btranslate\b", re.IGNORECASE),
)


class RoutingEngine:
    """Decides whether a task should run locally or on the remote reasoning engine."""

    def __init__(
        self,
        thresholds: RoutingThresholds | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._thresholds = thresholds or RoutingThresholds()
        self._tokens = token_counter or TokenCounter()

    def route(self, processed: ProcessedInput) -> RoutingResult:
        tokens = processed.post_optimization_tokens or self._tokens.count(processed.text)
        text = processed.text.strip()

        tier1 = self._tier1_heuristics(text, tokens, processed)
        if tier1 is not None:
            return tier1

        score = self._tier2_complexity_score(text, processed)
        return self._tier2_decision(score, tokens)

    def _tier1_heuristics(
        self,
        text: str,
        tokens: int,
        processed: ProcessedInput,
    ) -> RoutingResult | None:
        thresholds = self._thresholds

        if processed.has_images and thresholds.force_remote_on_image:
            return RoutingResult(
                destination=RouteDecision.REMOTE,
                tier=RoutingTier.TIER1_HEURISTIC,
                reason="Multimodal input with images requires remote vision-capable model.",
                complexity_score=80,
                estimated_tokens=tokens + thresholds.image_token_penalty,
                confidence=0.95,
            )

        if not text:
            return RoutingResult(
                destination=RouteDecision.LOCAL,
                tier=RoutingTier.TIER1_HEURISTIC,
                reason="Empty text payload; default to lightweight local handler.",
                complexity_score=0,
                estimated_tokens=tokens,
                confidence=0.7,
            )

        if tokens <= thresholds.bypass_remote_max_tokens and self._is_simple_math(text):
            return RoutingResult(
                destination=RouteDecision.LOCAL,
                tier=RoutingTier.TIER1_HEURISTIC,
                reason="Deterministic math expression bypasses remote inference.",
                complexity_score=5,
                estimated_tokens=tokens,
                confidence=0.99,
            )

        if tokens <= thresholds.simple_query_max_tokens:
            if self._is_simple_factual(text) or self._matches_simple_keywords(text):
                return RoutingResult(
                    destination=RouteDecision.LOCAL,
                    tier=RoutingTier.TIER1_HEURISTIC,
                    reason="Short factual query within local token budget.",
                    complexity_score=10,
                    estimated_tokens=tokens,
                    confidence=0.92,
                )
            if YES_NO.match(text):
                return RoutingResult(
                    destination=RouteDecision.LOCAL,
                    tier=RoutingTier.TIER1_HEURISTIC,
                    reason="Binary yes/no question suitable for local model.",
                    complexity_score=12,
                    estimated_tokens=tokens,
                    confidence=0.88,
                )

        if tokens > thresholds.max_local_tokens or len(text) > thresholds.max_local_chars:
            return RoutingResult(
                destination=RouteDecision.REMOTE,
                tier=RoutingTier.TIER1_HEURISTIC,
                reason="Input exceeds local token/character thresholds.",
                complexity_score=70,
                estimated_tokens=tokens,
                confidence=0.9,
            )

        return None

    def _tier2_decision(self, score: int, tokens: int) -> RoutingResult:
        thresholds = self._thresholds

        if score <= thresholds.local_complexity_ceiling:
            destination = RouteDecision.LOCAL
            reason = f"Semantic complexity score {score} ≤ local ceiling ({thresholds.local_complexity_ceiling})."
            confidence = 0.85
        elif score >= thresholds.remote_complexity_floor:
            destination = RouteDecision.REMOTE
            reason = f"Semantic complexity score {score} ≥ remote floor ({thresholds.remote_complexity_floor})."
            confidence = 0.88
        else:
            # Ambiguous band: prefer local when token budget is tight to save cost.
            if tokens <= thresholds.max_local_tokens // 2:
                destination = RouteDecision.LOCAL
                reason = (
                    f"Ambiguous complexity ({score}); token budget favors local "
                    f"({tokens} tokens)."
                )
                confidence = 0.65
            else:
                destination = RouteDecision.REMOTE
                reason = (
                    f"Ambiguous complexity ({score}); longer context favors remote reasoning."
                )
                confidence = 0.72

        return RoutingResult(
            destination=destination,
            tier=RoutingTier.TIER2_SEMANTIC,
            reason=reason,
            complexity_score=score,
            estimated_tokens=tokens,
            confidence=confidence,
        )

    def _tier2_complexity_score(self, text: str, processed: ProcessedInput) -> int:
        lowered = text.lower()
        score = 20

        score += min(25, len(text) // 120)
        score += min(15, text.count("?") * 4)
        score += min(12, text.count("\n") * 2)
        score += self._keyword_score(lowered, REASONING_KEYWORDS, weight=6, cap=24)
        score += self._keyword_score(lowered, CODE_KEYWORDS, weight=5, cap=20)

        if processed.has_images:
            score += self._thresholds.image_token_penalty // 4

        if self._has_multi_step_markers(lowered):
            score += 15
        if self._looks_like_json_or_schema_task(lowered):
            score += 10
        if self._is_simple_math(text):
            score -= 15
        if self._is_simple_factual(text):
            score -= 10

        return int(max(0, min(100, score)))

    @staticmethod
    def _keyword_score(text: str, keywords: Iterable[str], *, weight: int, cap: int) -> int:
        hits = sum(1 for kw in keywords if kw in text)
        return min(cap, hits * weight)

    @staticmethod
    def _matches_simple_keywords(text: str) -> bool:
        return any(pattern.search(text) for pattern in SIMPLE_KEYWORDS)

    @staticmethod
    def _matches_any(text: str, keywords: Iterable[str]) -> bool:
        return any(kw in text for kw in keywords)

    @staticmethod
    def _is_simple_math(text: str) -> bool:
        if not MATH_EXPRESSION.match(text):
            return False
        expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
        allowed = set("0123456789+-*/().,%^ ")
        return bool(expr) and all(ch in allowed for ch in expr)

    @staticmethod
    def _is_simple_factual(text: str) -> bool:
        return bool(SIMPLE_FACT.match(text))

    @staticmethod
    def _has_multi_step_markers(text: str) -> bool:
        markers = (
            "first,",
            "second,",
            "then ",
            "finally",
            "step 1",
            "step by step",
            "1.",
            "2.",
            "outline",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _looks_like_json_or_schema_task(text: str) -> bool:
        markers = ("json", "schema", "field", "extract", "structured", "parse")
        return any(marker in text for marker in markers)

    @staticmethod
    def try_evaluate_math(text: str) -> str | None:
        """Optional deterministic local shortcut for pure arithmetic."""
        if not RoutingEngine._is_simple_math(text):
            return None
        expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
        allowed = set("0123456789+-*/(). ")
        if not expr or not all(ch in allowed for ch in expr):
            return None
        try:
            # Safe evaluation: only digits and basic operators after sanitization.
            value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
            if isinstance(value, float) and math.isfinite(value):
                if value.is_integer():
                    return str(int(value))
                return str(round(value, 10))
            if isinstance(value, int):
                return str(value)
        except Exception:
            return None
        return None
