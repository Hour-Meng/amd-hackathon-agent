"""Two-tier routing intelligence: static heuristics + semantic complexity scoring + ANGKOR 3-zone."""

from __future__ import annotations

import math
import os
import pickle
import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from my_routing_agent.config import AdaptiveThresholdConfig, RoutingThresholds
from my_routing_agent.middleware.compressor import ProcessedInput
from my_routing_agent.middleware.entropy import compute_shannon_entropy
from my_routing_agent.routers.features import FeatureExtractor
from my_routing_agent.utils.tokenizer import TokenCounter

try:
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LogisticRegression = None  # type: ignore[misc]


class RouteDecision(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class RoutingTier(str, Enum):
    TIER1_HEURISTIC = "tier1_heuristic"
    TIER2_SEMANTIC = "tier2_semantic"
    TIER2_ANGKOR = "tier2_angkor"


class PhantomZone(str, Enum):
    CLEAR_LOCAL = "clear_local"
    PHANTOM_RACE = "phantom_race"
    CLEAR_REMOTE = "clear_remote"


@dataclass(frozen=True)
class AngkorRoutingResult:
    destination: RouteDecision
    zone: PhantomZone
    tier: RoutingTier
    reason: str
    complexity_score: float
    estimated_tokens: int
    confidence: float
    theta: float


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
    "analyze", "compare", "contrast", "evaluate", "explain why",
    "step by step", "prove", "derive", "debug", "root cause",
    "architecture", "design pattern", "optimize", "refactor",
    "implement", "algorithm", "trade-off", "reasoning", "multi-step",
)
CODE_KEYWORDS = (
    "python", "javascript", "typescript", "sql", "regex",
    "function", "class ", "api ", "code", "stack trace",
    "exception", "compile", "runtime", "```",
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
    """Original rule-based router — kept as fallback when sklearn unavailable."""

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
        self, text: str, tokens: int, processed: ProcessedInput,
    ) -> RoutingResult | None:
        thresholds = self._thresholds
        if processed.has_images and thresholds.force_remote_on_image:
            return RoutingResult(
                destination=RouteDecision.REMOTE, tier=RoutingTier.TIER1_HEURISTIC,
                reason="Multimodal input with images requires remote vision-capable model.",
                complexity_score=80, estimated_tokens=tokens + thresholds.image_token_penalty, confidence=0.95,
            )
        if not text:
            return RoutingResult(
                destination=RouteDecision.LOCAL, tier=RoutingTier.TIER1_HEURISTIC,
                reason="Empty text payload; default to lightweight local handler.",
                complexity_score=0, estimated_tokens=tokens, confidence=0.7,
            )
        if tokens <= thresholds.bypass_remote_max_tokens and self._is_simple_math(text):
            return RoutingResult(
                destination=RouteDecision.LOCAL, tier=RoutingTier.TIER1_HEURISTIC,
                reason="Deterministic math expression bypasses remote inference.",
                complexity_score=5, estimated_tokens=tokens, confidence=0.99,
            )
        if tokens <= thresholds.simple_query_max_tokens:
            if self._is_simple_factual(text) or self._matches_simple_keywords(text):
                return RoutingResult(
                    destination=RouteDecision.LOCAL, tier=RoutingTier.TIER1_HEURISTIC,
                    reason="Short factual query within local token budget.",
                    complexity_score=10, estimated_tokens=tokens, confidence=0.92,
                )
            if YES_NO.match(text):
                return RoutingResult(
                    destination=RouteDecision.LOCAL, tier=RoutingTier.TIER1_HEURISTIC,
                    reason="Binary yes/no question suitable for local model.",
                    complexity_score=12, estimated_tokens=tokens, confidence=0.88,
                )
        if tokens > thresholds.max_local_tokens or len(text) > thresholds.max_local_chars:
            return RoutingResult(
                destination=RouteDecision.REMOTE, tier=RoutingTier.TIER1_HEURISTIC,
                reason="Input exceeds local token/character thresholds.",
                complexity_score=70, estimated_tokens=tokens, confidence=0.9,
            )
        return None

    def _tier2_decision(self, score: int, tokens: int) -> RoutingResult:
        thresholds = self._thresholds
        if score <= thresholds.local_complexity_ceiling:
            return RoutingResult(
                destination=RouteDecision.LOCAL, tier=RoutingTier.TIER2_SEMANTIC,
                reason=f"Semantic complexity score {score} ≤ local ceiling ({thresholds.local_complexity_ceiling}).",
                complexity_score=score, estimated_tokens=tokens, confidence=0.85,
            )
        if score >= thresholds.remote_complexity_floor:
            return RoutingResult(
                destination=RouteDecision.REMOTE, tier=RoutingTier.TIER2_SEMANTIC,
                reason=f"Semantic complexity score {score} ≥ remote floor ({thresholds.remote_complexity_floor}).",
                complexity_score=score, estimated_tokens=tokens, confidence=0.88,
            )
        if tokens <= thresholds.max_local_tokens // 2:
            return RoutingResult(
                destination=RouteDecision.LOCAL, tier=RoutingTier.TIER2_SEMANTIC,
                reason=f"Ambiguous complexity ({score}); token budget favors local ({tokens} tokens).",
                complexity_score=score, estimated_tokens=tokens, confidence=0.65,
            )
        return RoutingResult(
            destination=RouteDecision.REMOTE, tier=RoutingTier.TIER2_SEMANTIC,
            reason=f"Ambiguous complexity ({score}); longer context favors remote reasoning.",
            complexity_score=score, estimated_tokens=tokens, confidence=0.72,
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
        markers = ("first,", "second,", "then ", "finally", "step 1", "step by step", "1.", "2.", "outline")
        return any(marker in text for marker in markers)

    @staticmethod
    def _looks_like_json_or_schema_task(text: str) -> bool:
        markers = ("json", "schema", "field", "extract", "structured", "parse")
        return any(marker in text for marker in markers)

    @staticmethod
    def try_evaluate_math(text: str) -> str | None:
        if not RoutingEngine._is_simple_math(text):
            return None
        expr = re.sub(r"(?i)^what\s+is\s+", "", text).strip().rstrip("=?")
        allowed = set("0123456789+-*/(). ")
        if not expr or not all(ch in allowed for ch in expr):
            return None
        try:
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


# ---------------------------------------------------------------------------
# ANGKOR 3-zone router with sklearn classifier + adaptive threshold
# ---------------------------------------------------------------------------

MODEL_PKL_PATH = os.path.join(os.path.dirname(__file__), "router_model.pkl")

# Default weights for logistic regression if no .pkl exists.
# Coefficients: [L_norm, H_norm, R_code, R_depth, L_out_norm]
_DEFAULT_COEF = np.array([1.5, 1.2, 2.0, 1.8, 0.8], dtype=np.float64)
_DEFAULT_INTERCEPT = np.array([-2.0], dtype=np.float64)


def _bootstrap_classifier() -> Any | None:
    """Create a LogisticRegression with hand-tuned weights for hackathon."""
    if not SKLEARN_AVAILABLE:
        return None
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )
    model.coef_ = _DEFAULT_COEF.reshape(1, -1)
    model.intercept_ = _DEFAULT_INTERCEPT
    model.classes_ = np.array([0.0, 1.0])
    return model


class AdaptiveThreshold:
    """Self-calibrating threshold θ that shifts based on CPU latency."""

    def __init__(self, config: AdaptiveThresholdConfig | None = None) -> None:
        cfg = config or AdaptiveThresholdConfig()
        self.theta: float = cfg.initial_theta
        self._min: float = cfg.min_theta
        self._max: float = cfg.max_theta
        self._budget_ms: float = cfg.latency_budget_ms
        self._window: deque[float] = deque(maxlen=20)

    def record_latency(self, latency_ms: float) -> None:
        self._window.append(latency_ms)

    def update(self) -> float:
        if len(self._window) < 5:
            return self.theta
        avg = sum(self._window) / len(self._window)
        if avg > self._budget_ms:
            self.theta = max(self._min, self.theta - 0.05)
        elif avg < self._budget_ms * 0.5:
            self.theta = min(self._max, self.theta + 0.03)
        self.theta = max(self._min, min(self.theta, self._max))
        return self.theta

    def reset(self, theta: float | None = None) -> None:
        if theta is not None:
            self.theta = max(self._min, min(theta, self._max))
        self._window.clear()


class SklearnRouter:
    """Tier 2 ANGKOR router: 5-feature sklearn classifier + 3-zone routing + adaptive θ."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        adaptive_config: AdaptiveThresholdConfig | None = None,
    ) -> None:
        self._features = feature_extractor or FeatureExtractor()
        self._theta = AdaptiveThreshold(adaptive_config)
        self._model = self._load_model()
        self._ready = self._model is not None

    def _load_model(self) -> Any | None:
        pkl = Path(MODEL_PKL_PATH)
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    model = pickle.load(f)
                if hasattr(model, "predict_proba"):
                    return model
            except Exception:
                pass
        return _bootstrap_classifier()

    def save_model(self, path: str | None = None) -> None:
        if self._model is None:
            return
        dest = path or MODEL_PKL_PATH
        with open(dest, "wb") as f:
            pickle.dump(self._model, f)

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def theta(self) -> float:
        return self._theta.theta

    def score(self, text: str, entropy_score: float | None = None) -> float:
        if not self._ready:
            return 0.5
        features = self._features.extract(text, entropy_score).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        return float(proba[1])

    def zone(self, complexity: float) -> PhantomZone:
        dead_zone = 0.10
        theta = self._theta.theta
        if complexity < (theta - dead_zone):
            return PhantomZone.CLEAR_LOCAL
        if complexity > (theta + dead_zone):
            return PhantomZone.CLEAR_REMOTE
        return PhantomZone.PHANTOM_RACE

    def route(self, text: str, entropy_score: float | None = None) -> AngkorRoutingResult:
        tokens = len(text.split())
        complexity = self.score(text, entropy_score)
        z = self.zone(complexity)

        if z == PhantomZone.CLEAR_LOCAL:
            return AngkorRoutingResult(
                destination=RouteDecision.LOCAL, zone=z, tier=RoutingTier.TIER2_ANGKOR,
                reason=f"Angkor: C={complexity:.3f} < θ-0.10 ({self._theta.theta - 0.10:.3f}) → clear local",
                complexity_score=complexity, estimated_tokens=tokens, confidence=1.0 - complexity, theta=self._theta.theta,
            )
        if z == PhantomZone.PHANTOM_RACE:
            return AngkorRoutingResult(
                destination=RouteDecision.LOCAL, zone=z, tier=RoutingTier.TIER2_ANGKOR,
                reason=f"Angkor: C={complexity:.3f} within ±0.10 of θ={self._theta.theta:.3f} → PHANTOM race",
                complexity_score=complexity, estimated_tokens=tokens, confidence=0.5, theta=self._theta.theta,
            )
        return AngkorRoutingResult(
            destination=RouteDecision.REMOTE, zone=z, tier=RoutingTier.TIER2_ANGKOR,
            reason=f"Angkor: C={complexity:.3f} > θ+0.10 ({self._theta.theta + 0.10:.3f}) → clear remote",
            complexity_score=complexity, estimated_tokens=tokens, confidence=complexity, theta=self._theta.theta,
        )
