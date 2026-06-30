"""Dead-zone speculative race with cancellable remote + telemetry."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from my_routing_agent.calibration.phantom_calibrator import should_abort
from my_routing_agent.phantom.budget import BudgetEnforcer
from my_routing_agent.phantom.generation_signals import (
    record_generation_signals,
    signal_vector_at_token,
)

logger = logging.getLogger("deadzone_runner")

MODEL_UNAVAILABLE_MARKERS = (
    "not found",
    "model_not_found",
    "does not exist",
    "not deployed",
    "invalid model",
)


@dataclass
class DeadZoneTelemetry:
    races_started: int = 0
    races_cancelled_local_win: int = 0
    races_cancelled_remote_win: int = 0
    races_cost_delta: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "races_started": self.races_started,
            "races_cancelled_local_win": self.races_cancelled_local_win,
            "races_cancelled_remote_win": self.races_cancelled_remote_win,
            "races_cost_delta": round(self.races_cost_delta, 4),
        }


@dataclass
class _RemoteState:
    future: Future | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    tokens_emitted: int = 0
    model_index: int = 0
    started_at: float = 0.0
    cost_estimate: float = 0.0


class DeadZoneRunner:
    """
    For complexity C in [θ-0.10, θ+0.10]:
    - local generation with early-abort monitor
    - cancellable remote with budgeted max_tokens
    - remote model-not-found → next validated model
    - local abort cancels remote only if remote hasn't passed 2x initial token window
    """

    def __init__(
        self,
        *,
        budget_enforcer: BudgetEnforcer | None = None,
        ensemble_report: dict[str, Any] | None = None,
        initial_token_window: int = 16,
    ) -> None:
        self._budget = budget_enforcer or BudgetEnforcer()
        self._ensemble = ensemble_report or {}
        self._initial_token_window = initial_token_window
        self.telemetry = DeadZoneTelemetry()

    @staticmethod
    def _is_model_unavailable(error_text: str) -> bool:
        low = (error_text or "").lower()
        return any(marker in low for marker in MODEL_UNAVAILABLE_MARKERS)

    def run_race(
        self,
        prompt: str,
        *,
        local_model: str,
        validated_remote_models: list[str],
        remote_call: Callable[..., tuple[str | None, str, int]],
        L_out_norm: float,
        confidence: float,
        local_base_url: str = "http://localhost:11434/v1",
    ) -> tuple[str, str, dict[str, Any]]:
        self.telemetry.races_started += 1
        budget = self._budget.compute_token_budget(L_out_norm, confidence)
        remote_state = _RemoteState(started_at=time.perf_counter())
        local_abort = threading.Event()
        meta: dict[str, Any] = {
            "budget": budget,
            "remote_models_tried": [],
            "winner": None,
            "local_abort": False,
        }

        def _local_worker() -> tuple[str | None, dict[str, Any]]:
            record = record_generation_signals(
                prompt, local_model, label="unknown", base_url=local_base_url
            )
            vector = signal_vector_at_token(record, token_index=8)
            abort = should_abort(vector, self._ensemble) if self._ensemble else False
            if abort:
                local_abort.set()
                return None, {"abort": True, "vector": vector}
            # Lightweight placeholder generation path for race integration
            return "".join(record.prompt.split()[:8]), {"abort": False, "vector": vector}

        def _remote_worker() -> tuple[str | None, str, int]:
            models = validated_remote_models or [""]
            for idx, model_id in enumerate(models):
                if remote_state.cancel_event.is_set():
                    return None, "cancelled", remote_state.tokens_emitted
                remote_state.model_index = idx
                meta["remote_models_tried"].append(model_id)
                try:
                    answer, status, tokens = remote_call(
                        prompt,
                        model_id=model_id,
                        max_tokens=budget,
                        cancel_event=remote_state.cancel_event,
                        token_counter=remote_state,
                    )
                    remote_state.tokens_emitted = max(remote_state.tokens_emitted, tokens)
                    if self._is_model_unavailable(status):
                        continue
                    if answer:
                        remote_state.cost_estimate = tokens * 0.0001
                        return answer, status, tokens
                except Exception as exc:
                    if self._is_model_unavailable(str(exc)):
                        continue
                    return None, f"error:{exc}", remote_state.tokens_emitted
            return None, "unavailable", remote_state.tokens_emitted

        local_start_cost = 0.01
        with ThreadPoolExecutor(max_workers=2) as pool:
            local_future = pool.submit(_local_worker)
            remote_state.future = pool.submit(_remote_worker)

            local_answer, local_meta = local_future.result()
            meta["local"] = local_meta

            if local_meta.get("abort"):
                meta["local_abort"] = True
                window_limit = self._initial_token_window * 2
                if remote_state.tokens_emitted <= window_limit:
                    remote_state.cancel_event.set()
                    if remote_state.future:
                        remote_state.future.cancel()
                    self.telemetry.races_cancelled_local_win += 1
                    meta["winner"] = "remote_after_local_abort"
                else:
                    self.telemetry.races_cancelled_remote_win += 1

            remote_answer, remote_status, remote_tokens = remote_state.future.result()
            meta["remote_status"] = remote_status
            meta["remote_tokens"] = remote_tokens

        if local_answer and not local_meta.get("abort"):
            winner = "local"
            answer = local_answer
            self.telemetry.races_cost_delta += local_start_cost - remote_state.cost_estimate
        elif remote_answer:
            winner = "remote"
            answer = remote_answer
            self.telemetry.races_cost_delta += remote_state.cost_estimate - local_start_cost
        else:
            winner = "failed"
            answer = "⚠️ Dead-zone race failed."

        meta["winner"] = winner
        meta["telemetry"] = self.telemetry.to_dict()
        return answer, winner, meta
