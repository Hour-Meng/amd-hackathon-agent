"""Benchmark suite for ANGKOR Router + PHANTOM Layer token savings & accuracy measurement."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from my_routing_agent.cache.semantic_cache import SemanticCache
from my_routing_agent.config import load_config
from my_routing_agent.middleware.compressor import InputCompressor
from my_routing_agent.middleware.entropy import compute_shannon_entropy
from my_routing_agent.routers.engine import SklearnRouter
from my_routing_agent.routers.features import FeatureExtractor
from my_routing_agent.utils.tokenizer import TokenCounter


@dataclass
class BenchmarkResult:
    query: str = ""
    complexity: float = 0.0
    zone: str = ""
    entropy: float = 0.0
    cache_hit: bool = False
    compression_ratio: float = 0.0
    latency_ms: float = 0.0


@dataclass
class BenchmarkReport:
    total_queries: int = 0
    cache_hits: int = 0
    avg_complexity: float = 0.0
    avg_entropy: float = 0.0
    avg_compression_ratio: float = 0.0
    zone_distribution: dict[str, int] = field(default_factory=dict)
    results: list[BenchmarkResult] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n" + "=" * 55)
        print("  ANGKOR + PHANTOM Benchmark Report")
        print("=" * 55)
        print(f"  Total queries:        {self.total_queries}")
        print(f"  Cache hits:           {self.cache_hits} ({self.cache_hits / max(1, self.total_queries):.1%})")
        print(f"  Avg complexity:       {self.avg_complexity:.3f}")
        print(f"  Avg entropy:          {self.avg_entropy:.3f}")
        print(f"  Avg compression:      {self.avg_compression_ratio:.1%}")
        print(f"  Zone distribution:    {self.zone_distribution}")
        print("=" * 55)


CALIBRATION_QUERIES: list[dict[str, Any]] = [
    {"query": "What is 2 + 2?", "type": "math", "expected": "4"},
    {"query": "Capital of France", "type": "factual", "expected": "Paris"},
    {"query": "Who wrote Hamlet", "type": "factual", "expected": "Shakespeare"},
    {"query": "What is the population of Japan", "type": "factual", "expected": ""},
    {"query": "Write a binary search in Python", "type": "code", "expected": ""},
    {"query": "Explain quantum entanglement", "type": "explanation", "expected": ""},
    {"query": "Derive Euler's identity", "type": "analysis", "expected": ""},
    {"query": "Compare REST and GraphQL", "type": "analysis", "expected": ""},
    {"query": "Hello", "type": "greeting", "expected": ""},
    {"query": "Extract email addresses from this text: john@example.com", "type": "extraction", "expected": ""},
    {"query": "What is the capital of France, who wrote Hamlet, and what is 5 * 13?", "type": "multi", "expected": ""},
    {"query": "spell apple backward", "type": "character", "expected": ""},
    {"query": "What is machine learning?", "type": "factual", "expected": ""},
    {"query": "Write a Python class for a binary tree", "type": "code", "expected": ""},
    {"query": "Explain the intuition behind gradient descent", "type": "explanation", "expected": ""},
]


def run_benchmark(queries: list[dict[str, Any]] | None = None, json_output: bool = False) -> BenchmarkReport:
    if queries is None:
        queries = CALIBRATION_QUERIES

    config = load_config()
    cache = SemanticCache()
    cache.initialize()
    tokenizer = TokenCounter()
    compressor = InputCompressor()
    router = SklearnRouter()
    fe = FeatureExtractor()

    report = BenchmarkReport(total_queries=len(queries))

    for q in queries:
        text = q.get("query", "")
        started = time.perf_counter()

        entry = cache.lookup(text)
        cache_hit = entry is not None
        if not cache_hit:
            entropy_score = compute_shannon_entropy(text)
            processed = compressor.process(text=text)
            pre_tokens = processed.pre_optimization_tokens
            post_tokens = processed.post_optimization_tokens
            compression = 1.0 - (post_tokens / max(1, pre_tokens)) if pre_tokens > 0 else 0.0

            if router.is_ready:
                angkor = router.route(text, entropy_score=entropy_score)
                complexity = angkor.complexity_score
                zone = angkor.zone.value
            else:
                complexity = 0.5
                zone = "unknown"
        else:
            entropy_score = 0.0
            compression = 1.0
            complexity = 0.0
            zone = "cache_hit"

        latency_ms = (time.perf_counter() - started) * 1000.0

        report.results.append(BenchmarkResult(
            query=text,
            complexity=complexity,
            zone=zone,
            entropy=entropy_score,
            cache_hit=cache_hit,
            compression_ratio=compression,
            latency_ms=latency_ms,
        ))

        if cache_hit:
            report.cache_hits += 1
        report.zone_distribution[zone] = report.zone_distribution.get(zone, 0) + 1

    n = len(report.results)
    report.avg_complexity = sum(r.complexity for r in report.results) / max(1, n)
    report.avg_entropy = sum(r.entropy for r in report.results) / max(1, n)
    report.avg_compression_ratio = sum(r.compression_ratio for r in report.results) / max(1, n)

    if json_output:
        print(json.dumps({
            "total_queries": report.total_queries,
            "cache_hits": report.cache_hits,
            "cache_hit_rate": report.cache_hits / max(1, n),
            "avg_complexity": round(report.avg_complexity, 4),
            "avg_entropy": round(report.avg_entropy, 4),
            "avg_compression_ratio": round(report.avg_compression_ratio, 4),
            "zone_distribution": report.zone_distribution,
        }, indent=2))
    else:
        report.print_report()

    return report


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="ANGKOR + PHANTOM benchmark suite")
    parser.add_argument("--queries", type=str, default="", help="Path to JSON queries file")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.queries:
        with open(args.queries) as f:
            queries = json.load(f)
    else:
        queries = CALIBRATION_QUERIES

    run_benchmark(queries, json_output=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
