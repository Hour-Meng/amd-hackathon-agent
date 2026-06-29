"""Shannon entropy calculator for prompt complexity measurement."""

from __future__ import annotations

import math
from collections import Counter


def compute_shannon_entropy(text: str) -> float:
    """
    Compute Shannon entropy H of a prompt from its token frequency distribution.

    H = -Σ p(tᵢ) · log₂(p(tᵢ))

    High entropy = high lexical diversity = complex reasoning likely needed.
    Low entropy = repetitive, templated = simple local task.
    """
    if not text or not text.strip():
        return 0.0
    tokens = text.strip().split()
    if not tokens:
        return 0.0
    total = len(tokens)
    freq = Counter(tokens)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def normalize_entropy(entropy: float, max_h: float = 10.0) -> float:
    """Normalize entropy to [0, 1] range. Max H for typical English text ~8-10."""
    return max(0.0, min(1.0, entropy / max_h))
