"""5-feature complexity vector extractor for Tier 2 route gate."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from my_routing_agent.middleware.entropy import compute_shannon_entropy, normalize_entropy
from my_routing_agent.utils.tokenizer import TokenCounter

CODE_PATTERNS: tuple[str, ...] = (
    "python", "javascript", "typescript", "sql", "regex",
    "function", "class ", "api ", "code", "stack trace",
    "exception", "compile", "runtime", "```", "def ",
)

REASONING_MARKERS: tuple[str, ...] = (
    "if", "because", "therefore", "compare", "contrast",
    "analyze", "evaluate", "explain why", "step by step",
    "prove", "derive", "debug", "root cause", "architecture",
    "design pattern", "optimize", "refactor", "implement",
    "algorithm", "trade-off", "reasoning", "multi-step",
)

TASK_VERB_LENGTH: dict[str, float] = {
    "name": 0.05, "what": 0.05, "who": 0.05, "when": 0.05,
    "list": 0.10, "find": 0.15, "extract": 0.20,
    "describe": 0.35, "explain": 0.50, "why": 0.50,
    "summarize": 0.40, "compare": 0.70, "analyze": 0.70,
    "write": 0.85, "create": 0.85, "generate": 0.85,
    "implement": 0.85, "derive": 0.60,
}

MAX_TOKENS_NORM = 2048


class FeatureExtractor:
    """Extracts a 5-dim complexity feature vector from a prompt + entropy score."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self._tokens = token_counter or TokenCounter()

    def extract(self, text: str, entropy_score: float | None = None) -> np.ndarray:
        cleaned = text.strip()
        if not cleaned:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        token_count = self._tokens.count(cleaned)
        lowered = cleaned.lower()

        L_norm = min(1.0, token_count / MAX_TOKENS_NORM)

        if entropy_score is None:
            entropy_score = compute_shannon_entropy(cleaned)
        H_norm = normalize_entropy(entropy_score)

        code_tokens = len(re.findall(r'\b(?:' + '|'.join(re.escape(p) for p in CODE_PATTERNS) + r')\b', lowered))
        R_code = min(1.0, code_tokens / max(1, len(cleaned.split())) * 10)

        reasoning_hits = sum(1 for m in REASONING_MARKERS if m in lowered)
        R_depth = min(1.0, reasoning_hits / 10.0)

        L_out_norm = self._predict_output_length(lowered)

        return np.array([L_norm, H_norm, R_code, R_depth, L_out_norm], dtype=np.float32)

    def _predict_output_length(self, lowered: str) -> float:
        tokens = lowered.split()
        if not tokens:
            return 0.05
        first_word = tokens[0].strip(",.!?")
        for verb, length in TASK_VERB_LENGTH.items():
            if first_word == verb or lowered.startswith(verb):
                question_count = lowered.count("?")
                if question_count > 1:
                    length = min(1.0, length * question_count * 0.8)
                return length
        q_count = lowered.count("?")
        if q_count > 2:
            return 0.6
        if q_count > 0:
            return 0.3
        if len(tokens) > 30:
            return 0.5
        return 0.15

    @staticmethod
    def feature_names() -> list[str]:
        return ["L_norm", "H_norm", "R_code", "R_depth", "L_out_norm"]
