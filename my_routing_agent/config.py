"""Central configuration for the hybrid routing agent with ANGKOR + PHANTOM."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LocalConfig:
    """Ollama / OpenAI-compatible local inference endpoint."""

    base_url: str = field(
        default_factory=lambda: os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "qwen2.5:0.5b"))
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LOCAL_LLM_TIMEOUT", "120"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("LOCAL_LLM_MAX_TOKENS", "512")))


@dataclass(frozen=True)
class RemoteConfig:
    """Fireworks AI remote inference endpoint."""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("FIREWORKS_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv(
            "FIREWORKS_MODEL", "accounts/fireworks/models/qwen3p7-max"
        )
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("FIREWORKS_TIMEOUT", "180"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("FIREWORKS_MAX_TOKENS", "1024")))


@dataclass(frozen=True)
class RemoteModelTier:
    """Maps a complexity score band to a Fireworks remote model."""

    min_score: int
    max_score: int
    model_id: str
    label: str


def _tier_model_env(env_key: str, default: str) -> str:
    value = os.getenv(env_key, default).strip()
    return value or default


def remote_model_tiers() -> tuple[RemoteModelTier, ...]:
    """Tiered remote models by task complexity (overridable via env)."""
    return (
        RemoteModelTier(
            0,
            25,
            _tier_model_env(
                "REMOTE_TIER_FAST_MODEL",
                "accounts/fireworks/models/qwen3p7-plus",
            ),
            "fast",
        ),
        RemoteModelTier(
            26,
            55,
            _tier_model_env(
                "REMOTE_TIER_BALANCED_MODEL",
                "accounts/fireworks/models/minimax-m3",
            ),
            "balanced",
        ),
        RemoteModelTier(
            56,
            100,
            _tier_model_env(
                "REMOTE_TIER_FULL_MODEL",
                "accounts/fireworks/models/qwen3p7-max",
            ),
            "full",
        ),
    )


@dataclass(frozen=True)
class RoutingThresholds:
    """Strict routing thresholds for Tier-1 heuristics and Tier-2 classification."""

    # Tier 1 — static heuristics
    max_local_tokens: int = 512
    max_local_chars: int = 2048
    simple_query_max_tokens: int = 64
    bypass_remote_max_tokens: int = 128

    # Tier 2 — semantic complexity scoring (0–100 scale)
    local_complexity_ceiling: int = 35
    remote_complexity_floor: int = 55
    ambiguous_band_low: int = 36
    ambiguous_band_high: int = 54

    # Vision / multimodal
    force_remote_on_image: bool = True
    image_token_penalty: int = 120


@dataclass(frozen=True)
class CompressorConfig:
    """Image down-sampling and text pruning settings."""

    max_image_dimension: int = 512
    jpeg_quality: int = 85
    png_optimize: bool = True
    strip_system_fluff: bool = True
    collapse_whitespace: bool = True
    max_text_chars: int = 8000
    enable_pos_pruning: bool = field(
        default_factory=lambda: os.getenv("ENABLE_POS_PRUNING", "true").lower() in {"1", "true", "yes"}
    )
    spacy_model: str = field(default_factory=lambda: os.getenv("SPACY_MODEL", "en_core_web_sm"))


@dataclass(frozen=True)
class CacheConfig:
    """Tier 0 — FAISS semantic cache settings."""

    threshold: float = field(
        default_factory=lambda: float(os.getenv("CACHE_THRESHOLD", "0.90"))
    )
    candidate_range_low: float = field(
        default_factory=lambda: float(os.getenv("CACHE_CANDIDATE_RANGE_LOW", "0.88"))
    )
    candidate_range_high: float = field(
        default_factory=lambda: float(os.getenv("CACHE_CANDIDATE_RANGE_HIGH", "0.92"))
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("CACHE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    index_path: str = field(
        default_factory=lambda: os.getenv("CACHE_INDEX_PATH", "faiss_cache.index")
    )
    store_path: str = field(
        default_factory=lambda: os.getenv("CACHE_STORE_PATH", "cache_store.json")
    )
    max_entries: int = field(
        default_factory=lambda: int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
    )


@dataclass(frozen=True)
class PhantomConfig:
    """PHANTOM speculative pre-flight executor settings."""

    entropy_check_token: int = field(
        default_factory=lambda: int(os.getenv("ENTROPY_CHECK_TOKEN", "8"))
    )
    entropy_abort_threshold: float = field(
        default_factory=lambda: float(os.getenv("ENTROPY_ABORT_THRESHOLD", "3.5"))
    )
    ensemble_abort_threshold: float = field(
        default_factory=lambda: float(os.getenv("ENSEMBLE_ABORT_THRESHOLD", "0.25"))
    )
    dead_zone: float = field(
        default_factory=lambda: float(os.getenv("PHANTOM_DEAD_ZONE", "0.10"))
    )
    per_model_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "qwen2.5:0.5b": float(os.getenv("PHANTOM_THRESHOLD_QWEN_0_5B", "0.25")),
            "qwen2.5:32b": float(os.getenv("PHANTOM_THRESHOLD_QWEN_32B", "0.30")),
            "default": float(os.getenv("PHANTOM_THRESHOLD_DEFAULT", "0.25")),
        }
    )


@dataclass(frozen=True)
class VerifierConfig:
    """Tier 3 — Cascade verify settings."""

    coherence_threshold: float = field(
        default_factory=lambda: float(os.getenv("COHERENCE_THRESHOLD", "0.55"))
    )
    escalation_max_rate: float = field(
        default_factory=lambda: float(os.getenv("ESCALATION_MAX_RATE", "0.15"))
    )


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    """Adaptive routing threshold θ settings."""

    initial_theta: float = field(
        default_factory=lambda: float(os.getenv("COMPLEXITY_THRESHOLD", "0.65"))
    )
    latency_budget_ms: float = field(
        default_factory=lambda: float(os.getenv("LATENCY_BUDGET_MS", "800"))
    )
    min_theta: float = field(
        default_factory=lambda: float(os.getenv("MIN_THETA", "0.40"))
    )
    max_theta: float = field(
        default_factory=lambda: float(os.getenv("MAX_THETA", "0.85"))
    )


@dataclass(frozen=True)
class BudgetEnforcerConfig:
    """PHANTOM C — Dynamic max_tokens budget settings."""

    base_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("BASE_MAX_TOKENS", "512"))
    )
    min_token_budget: int = field(
        default_factory=lambda: int(os.getenv("MIN_TOKEN_BUDGET", "20"))
    )


@dataclass(frozen=True)
class TokenizerConfig:
    """Local token estimation settings."""

    default_encoding: str = "cl100k_base"
    local_model_encoding: str = "cl100k_base"
    remote_model_encoding: str = "cl100k_base"


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration."""

    local: LocalConfig = field(default_factory=LocalConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    routing: RoutingThresholds = field(default_factory=RoutingThresholds)
    compressor: CompressorConfig = field(default_factory=CompressorConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    phantom: PhantomConfig = field(default_factory=PhantomConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    adaptive: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    budget: BudgetEnforcerConfig = field(default_factory=BudgetEnforcerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    enable_fallback: bool = field(
        default_factory=lambda: os.getenv("ROUTING_ENABLE_FALLBACK", "true").lower()
        in {"1", "true", "yes"}
    )
    system_prompt: str = (
        "Respond concisely and accurately. For structured tasks, output valid JSON only."
    )

    def validate(self) -> None:
        if not self.remote.api_key:
            pass

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_model": self.local.model,
            "remote_model": self.remote.model,
            "max_local_tokens": self.routing.max_local_tokens,
            "enable_fallback": self.enable_fallback,
        }


def load_config() -> AgentConfig:
    """Load configuration from environment with sane defaults."""
    config = AgentConfig()
    config.validate()
    return config


def parse_allowed_models() -> list[str]:
    """Parse comma-separated ALLOWED_MODELS env var into normalized model ids."""
    raw = os.getenv("ALLOWED_MODELS", "").strip()
    if not raw:
        return []
    models: list[str] = []
    for part in raw.split(","):
        mid = part.strip()
        if mid:
            models.append(mid)
    return models


def skip_local_inference() -> bool:
    """True when SKIP_LOCAL env disables Ollama / canned local paths."""
    return os.getenv("SKIP_LOCAL", "").lower() in {"1", "true", "yes"}


def request_timeout_seconds() -> float:
    """Per-request HTTP timeout; defaults to 30s in batch (SKIP_LOCAL) mode."""
    default = "30" if skip_local_inference() else "180"
    return float(os.getenv("REQUEST_TIMEOUT_SECONDS", default))


def batch_timeout_seconds() -> float:
    """Total batch pipeline timeout (default 600s / 10 minutes)."""
    return float(os.getenv("BATCH_TIMEOUT_SECONDS", "600"))
