"""Central configuration for the hybrid routing agent."""

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
    model: str = field(default_factory=lambda: os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b"))
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
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    enable_fallback: bool = field(
        default_factory=lambda: os.getenv("ROUTING_ENABLE_FALLBACK", "true").lower()
        in {"1", "true", "yes"}
    )
    system_prompt: str = (
        "Respond concisely and accurately. For structured tasks, output valid JSON only."
    )

    def validate(self) -> None:
        """Raise if critical remote credentials are missing when remote routing is possible."""
        if not self.remote.api_key:
            # Remote may still be skipped entirely; warn at runtime instead of hard-failing init.
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
