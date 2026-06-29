"""Core orchestrator: middleware → router → ANGKOR+PHANTOM → inference → telemetry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from my_routing_agent.cache.semantic_cache import SemanticCache
from my_routing_agent.clients.local_client import InferenceResponse, LocalClient
from my_routing_agent.clients.remote_client import RemoteClient
from my_routing_agent.config import AgentConfig, load_config
from my_routing_agent.middleware.compressor import InputCompressor, ProcessedInput
from my_routing_agent.middleware.entropy import compute_shannon_entropy
from my_routing_agent.phantom.budget import BudgetEnforcer
from my_routing_agent.phantom.confidence import ConfidencePredictor
from my_routing_agent.phantom.speculative import SpeculativeRunner
from my_routing_agent.routers.engine import (
    PhantomZone,
    RouteDecision,
    RoutingEngine,
    RoutingResult,
    SklearnRouter,
)
from my_routing_agent.routers.features import FeatureExtractor
from my_routing_agent.utils.tokenizer import TokenCounter
from my_routing_agent.verifier.cascade import CascadeVerifier


@dataclass
class TelemetryMetrics:
    target_destination: str
    routing_tier: str
    routing_reason: str
    complexity_score: int
    input_tokens_pre: int
    input_tokens_post: int
    output_tokens: int
    execution_latency_ms: float
    model: str = ""
    fallback_used: bool = False
    success: bool = True
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def print_report(self) -> None:
        print("\n=== ANGKOR + PHANTOM Telemetry ===")
        print(f"Target Destination   : {self.target_destination.upper()}")
        print(f"Routing Tier         : {self.routing_tier}")
        print(f"Routing Reason       : {self.routing_reason}")
        print(f"Complexity Score     : {self.complexity_score}")
        print(f"Input Tokens (Pre)   : {self.input_tokens_pre}")
        print(f"Input Tokens (Post)  : {self.input_tokens_post}")
        print(f"Output Tokens        : {self.output_tokens}")
        print(f"Execution Latency    : {self.execution_latency_ms:.2f} ms")
        print(f"Model                : {self.model}")
        if self.fallback_used:
            print("Fallback             : YES")
        if self.error:
            print(f"Error                : {self.error}")
        extra = self.extra
        if extra.get("phantom_winner"):
            print(f"PHANTOM Winner       : {extra['phantom_winner']}")
        if extra.get("entropy_at_check") is not None:
            print(f"Entropy at check     : {extra['entropy_at_check']:.3f}")
        if extra.get("cache_hit"):
            print("Cache Hit            : YES (0 tokens)")
        print(f"Success              : {self.success}")
        print("================================\n")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HybridRoutingAgent:
    """End-to-end pipeline for token-efficient hybrid inference routing."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or load_config()
        self.token_counter = TokenCounter(self.config.tokenizer)
        self.compressor = InputCompressor(self.config.compressor, self.token_counter)
        self.router = RoutingEngine(self.config.routing, self.token_counter)
        self.local_client = LocalClient(self.config.local)
        self.remote_client = RemoteClient(self.config.remote)
        self.cache = SemanticCache()
        self.cache.initialize()
        self.sklearn_router = SklearnRouter()
        self.features = FeatureExtractor(self.token_counter)
        self.budget = BudgetEnforcer()
        self.confidence = ConfidencePredictor()
        self.phantom_runner = SpeculativeRunner(self.confidence, self.budget)
        self.verifier = CascadeVerifier()

    def run(
        self,
        user_input: str,
        *,
        image_paths: list[str] | None = None,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
        force_destination: RouteDecision | None = None,
        force_phantom_race: bool = False,
        entropy_trace: bool = False,
    ) -> tuple[str, TelemetryMetrics]:
        pipeline_started = time.perf_counter()

        cache_hit = self.cache.lookup(user_input)
        if cache_hit:
            latency_ms = (time.perf_counter() - pipeline_started) * 1000.0
            telemetry = TelemetryMetrics(
                target_destination="cache",
                routing_tier="tier0_cache",
                routing_reason="Semantic cache hit (zero tokens, zero API calls).",
                complexity_score=0,
                input_tokens_pre=0,
                input_tokens_post=0,
                output_tokens=0,
                execution_latency_ms=latency_ms,
                model="cache",
                success=True,
                extra={"cache_hit": True},
            )
            telemetry.print_report()
            return cache_hit.response, telemetry

        processed = self.compressor.process(user_input, image_paths=image_paths)
        routing = self.router.route(processed)

        if force_destination is not None:
            routing = RoutingResult(
                destination=force_destination,
                tier=routing.tier,
                reason=f"Forced destination override ({force_destination.value}).",
                complexity_score=routing.complexity_score,
                estimated_tokens=routing.estimated_tokens,
                confidence=1.0,
            )

        math_answer = RoutingEngine.try_evaluate_math(processed.text)
        if math_answer and routing.destination == RouteDecision.LOCAL:
            latency_ms = (time.perf_counter() - pipeline_started) * 1000.0
            output_tokens = self.token_counter.count(math_answer)
            telemetry = TelemetryMetrics(
                target_destination=RouteDecision.LOCAL.value,
                routing_tier=routing.tier.value,
                routing_reason="Deterministic local math evaluation (zero inference tokens).",
                complexity_score=routing.complexity_score,
                input_tokens_pre=processed.pre_optimization_tokens,
                input_tokens_post=processed.post_optimization_tokens,
                output_tokens=output_tokens,
                execution_latency_ms=latency_ms,
                model="deterministic-math",
                success=True,
            )
            telemetry.print_report()
            return math_answer, telemetry

        # ANGKOR 3-zone check
        if self.sklearn_router.is_ready or force_phantom_race:
            entropy_score = compute_shannon_entropy(processed.text)
            angkor = self.sklearn_router.route(processed.text, entropy_score=entropy_score)

            if force_phantom_race or angkor.zone == PhantomZone.PHANTOM_RACE:
                feats = self.features.extract(processed.text, entropy_score=entropy_score)
                L_out_norm = float(feats[4])
                confidence = abs(angkor.complexity_score - self.sklearn_router.theta)

                def _remote_fn(text: str, **kw: object) -> str | None:
                    max_tok = int(kw.get("max_tokens", 128))
                    messages = [
                        {"role": "system", "content": self.config.system_prompt},
                        {"role": "user", "content": text},
                    ]
                    resp = self.remote_client.chat(messages, max_tokens=max_tok)
                    return resp.content if resp.success else None

                output, source, phantom_telemetry = self.phantom_runner.phantom_race(
                    prompt=processed.text,
                    L_out_norm=L_out_norm,
                    confidence=confidence,
                    local_model=self.config.local.model,
                    remote_call=_remote_fn,
                )

                latency_ms = (time.perf_counter() - pipeline_started) * 1000.0
                telemetry = TelemetryMetrics(
                    target_destination=f"phantom_{source}",
                    routing_tier="phantom_race",
                    routing_reason=angkor.reason,
                    complexity_score=int(angkor.complexity_score * 100),
                    input_tokens_pre=processed.pre_optimization_tokens,
                    input_tokens_post=processed.post_optimization_tokens,
                    output_tokens=self.token_counter.count(output),
                    execution_latency_ms=latency_ms,
                    model=f"phantom:{source}",
                    success=True,
                    extra={
                        "phantom_winner": source,
                        "phantom_telemetry": phantom_telemetry,
                        "angkor_complexity": angkor.complexity_score,
                        "angkor_theta": angkor.theta,
                    },
                )
                telemetry.print_report()
                return output, telemetry

        messages = self._build_messages(processed)
        response, fallback_used = self._invoke_with_fallback(
            routing, messages, json_mode=json_mode, schema=schema,
        )

        output_tokens = response.completion_tokens or self.token_counter.count(response.content)
        total_latency_ms = (time.perf_counter() - pipeline_started) * 1000.0

        telemetry = TelemetryMetrics(
            target_destination=routing.destination.value,
            routing_tier=routing.tier.value,
            routing_reason=routing.reason,
            complexity_score=routing.complexity_score,
            input_tokens_pre=processed.pre_optimization_tokens,
            input_tokens_post=processed.post_optimization_tokens,
            output_tokens=output_tokens,
            execution_latency_ms=total_latency_ms,
            model=response.model,
            fallback_used=fallback_used,
            success=response.success,
            error=response.error,
            extra={
                "confidence": routing.confidence,
                "total_tokens": response.total_tokens,
                "inference_latency_ms": response.latency_ms,
            },
        )
        telemetry.print_report()

        content = response.content
        if hasattr(response, "parsed_json") and response.parsed_json:
            content = json.dumps(response.parsed_json, ensure_ascii=False)

        if response.success:
            self.cache.store(user_input, content, metadata={"route": routing.destination.value})

        return content, telemetry

    def _build_messages(self, processed: ProcessedInput) -> list[dict[str, Any]]:
        if processed.has_images:
            parts: list[dict[str, Any]] = []
            if processed.text:
                parts.append({"type": "text", "text": processed.text})
            for image_b64 in processed.images:
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
            user_content: str | list[dict[str, Any]] = parts
        else:
            user_content = processed.text
        return [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _invoke_with_fallback(
        self, routing: RoutingResult, messages: list[dict[str, Any]], *, json_mode: bool, schema: dict[str, Any] | None,
    ) -> tuple[InferenceResponse, bool]:
        primary = routing.destination
        fallback = RouteDecision.REMOTE if primary == RouteDecision.LOCAL else RouteDecision.LOCAL

        response = self._dispatch(primary, messages, json_mode=json_mode, schema=schema)
        if response.success:
            return response, False
        if not self.config.enable_fallback:
            return response, False
        print(f"[fallback] Primary '{primary.value}' failed: {response.error}. Retrying via '{fallback.value}'.", file=sys.stderr)
        return self._dispatch(fallback, messages, json_mode=json_mode, schema=schema), True

    def _dispatch(self, destination: RouteDecision, messages: list[dict[str, Any]], *, json_mode: bool, schema: dict[str, Any] | None,) -> InferenceResponse:
        if destination == RouteDecision.LOCAL:
            return self.local_client.chat(messages, json_mode=json_mode)
        return self.remote_client.chat(messages, json_mode=json_mode or True, schema=schema)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ANGKOR Router + PHANTOM Layer — hybrid token-efficient routing agent."
    )
    parser.add_argument("prompt", nargs="?", default="What is 17 * 23?", help="User task / prompt text.")
    parser.add_argument("--image", action="append", default=[], dest="images", help="Optional image path(s) for multimodal routing.")
    parser.add_argument("--json", action="store_true", help="Request structured JSON output (strict on remote).")
    parser.add_argument("--force-local", action="store_true", help="Bypass router and force local inference.")
    parser.add_argument("--force-remote", action="store_true", help="Bypass router and force remote inference.")
    parser.add_argument("--force-phantom-race", action="store_true", help="Force PHANTOM speculative race regardless of zone.")
    parser.add_argument("--entropy-trace", action="store_true", help="Show entropy measurement trace.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    agent = HybridRoutingAgent()

    force: RouteDecision | None = None
    if args.force_local:
        force = RouteDecision.LOCAL
    elif args.force_remote:
        force = RouteDecision.REMOTE

    result, telemetry = agent.run(
        args.prompt,
        image_paths=args.images or None,
        json_mode=args.json,
        force_destination=force,
        force_phantom_race=args.force_phantom_race,
        entropy_trace=args.entropy_trace,
    )

    print("=== Model Output ===")
    print(result)
    return 0 if telemetry.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
