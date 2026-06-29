"""PHANTOM B — Speculative Pre-Flight Executor (Parallel Race).

When complexity score C is within the dead zone around θ, fire local and remote
in parallel. First valid result wins; the other is discarded.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from my_routing_agent.phantom.confidence import ConfidencePredictor
from my_routing_agent.phantom.budget import BudgetEnforcer

logger = logging.getLogger("phantom_speculative")


class SpeculativeRunner:
    """Race local and remote in parallel. First structurally valid result wins."""

    def __init__(
        self,
        confidence_predictor: ConfidencePredictor | None = None,
        budget_enforcer: BudgetEnforcer | None = None,
    ) -> None:
        self._confidence = confidence_predictor or ConfidencePredictor()
        self._budget = budget_enforcer or BudgetEnforcer()
        self._stop_event = threading.Event()

    def phantom_race(
        self,
        prompt: str,
        L_out_norm: float,
        confidence: float,
        local_model: str,
        remote_call: Callable[..., str | None],
        *,
        local_base_url: str = "http://localhost:11434/v1",
        remote_kwargs: dict[str, Any] | None = None,
        structural_validator: Callable[[str], bool] | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Fire local (entropy-aware) and remote simultaneously.
        Returns (result_text, source, telemetry).
        source is "local" or "remote" or "fallback".
        """
        self._stop_event.clear()
        budget = self._budget.compute_token_budget(L_out_norm, confidence)
        telemetry: dict[str, Any] = {
            "local_status": None,
            "remote_status": None,
            "local_entropy": None,
            "winner": None,
            "budget": budget,
        }

        def _run_local() -> tuple[str | None, str, dict[str, Any]]:
            if self._stop_event.is_set():
                return None, "cancelled", {"entropy": None}
            local_start = time.perf_counter()
            output, status, entropy = self._confidence.speculative_execute_local(
                prompt, local_model, base_url=local_base_url,
            )
            elapsed = (time.perf_counter() - local_start) * 1000
            return output, status, {"entropy": entropy, "latency_ms": elapsed}

        def _run_remote() -> tuple[str | None, str, dict[str, Any]]:
            if self._stop_event.is_set():
                return None, "cancelled", {}
            remote_start = time.perf_counter()
            try:
                kwargs = dict(remote_kwargs or {})
                kwargs.setdefault("max_tokens", budget)
                output = remote_call(prompt, **kwargs)
                elapsed = (time.perf_counter() - remote_start) * 1000
                return output, "complete" if output else "empty", {"latency_ms": elapsed}
            except Exception as exc:
                return None, f"error:{exc}", {}

        validator = structural_validator or (lambda x: bool(x and x.strip()))

        with ThreadPoolExecutor(max_workers=2) as executor:
            local_future = executor.submit(_run_local)
            remote_future = executor.submit(_run_remote)

            for future in as_completed([local_future, remote_future]):
                output, status, meta = future.result()
                source = "local" if future == local_future else "remote"

                if source == "local":
                    telemetry["local_status"] = status
                    telemetry["local_entropy"] = meta.get("entropy")
                else:
                    telemetry["remote_status"] = status

                if status == "entropy_abort":
                    logger.info("PHANTOM: local aborted (entropy), waiting for remote")
                    continue

                if output and validator(output):
                    self._stop_event.set()
                    telemetry["winner"] = source
                    telemetry["latency_ms"] = meta.get("latency_ms", 0)
                    logger.info("PHANTOM: %s wins (budget=%d)", source, budget)
                    return output, source, telemetry

        # Both failed or no valid result — try binary escalation
        logger.warning("PHANTOM: both paths failed, falling back to remote-only")
        try:
            kwargs = dict(remote_kwargs or {})
            kwargs.setdefault("max_tokens", budget * 2)
            fallback = remote_call(prompt, **kwargs)
            if fallback:
                telemetry["winner"] = "fallback_remote"
                return fallback, "fallback_remote", telemetry
        except Exception as exc:
            logger.error("PHANTOM fallback failed: %s", exc)

        return "⚠️ PHANTOM: all execution paths failed.", "failed", telemetry
