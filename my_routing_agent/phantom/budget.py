"""PHANTOM C — Dynamic max_tokens budget enforcer."""

from __future__ import annotations

from my_routing_agent.config import BudgetEnforcerConfig


class BudgetEnforcer:
    """Sets max_tokens per API call based on predicted output length and confidence."""

    def __init__(self, config: BudgetEnforcerConfig | None = None) -> None:
        cfg = config or BudgetEnforcerConfig()
        self._base_max: int = cfg.base_max_tokens
        self._min_budget: int = cfg.min_token_budget

    def compute_token_budget(self, L_out_norm: float, confidence: float) -> int:
        """
        L_out_norm: predicted output length, normalized 0–1 (from feature extractor)
        confidence: classifier confidence in the routing decision
        """
        raw_budget = int(L_out_norm * self._base_max)
        confidence_scale = 0.8 + (0.4 * (1.0 - confidence))
        budget = int(raw_budget * confidence_scale)
        return max(self._min_budget, min(budget, self._base_max))

    @staticmethod
    def budget_for_task(task_type: str, base_max: int = 512) -> int:
        """Quick heuristic budget by task type."""
        ratios = {
            "qa": 0.05,
            "factual": 0.05,
            "extraction": 0.20,
            "explanation": 0.50,
            "code": 0.85,
            "analysis": 0.70,
            "generation": 0.85,
        }
        ratio = ratios.get(task_type, 0.3)
        return max(20, int(ratio * base_max))
