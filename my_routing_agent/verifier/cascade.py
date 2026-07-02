"""Tier 3 — Cascade Verify: structural validation → semantic coherence → binary escalation."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Callable

from my_routing_agent.config import VerifierConfig
from my_routing_agent.utils.math_eval import try_evaluate_math

logger = logging.getLogger("cascade_verifier")

ESCALATION_PROMPT = (
    "Query: {query}\n"
    "Proposed Answer: {proposed}\n\n"
    "Is this answer correct and complete?\n"
    "If YES: reply only \"VALID\"\n"
    "If NO: reply \"INVALID: \" followed by the corrected answer only."
)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


class CascadeVerifier:
    """Three-step output quality verifier with binary escalation."""

    def __init__(
        self,
        config: VerifierConfig | None = None,
        encoder: Any | None = None,
    ) -> None:
        self._config = config or VerifierConfig()
        self._encoder: Any | None = encoder

    def _get_encoder(self) -> Any | None:
        if self._encoder is not None:
            return self._encoder
        if not TRANSFORMERS_AVAILABLE:
            return None
        try:
            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning("Failed to load MiniLM encoder: %s", exc)
            self._encoder = None
        return self._encoder

    def verify(
        self,
        query: str,
        output: str,
        task_type: str = "qa",
        remote_escalate_fn: Callable[[str], str] | None = None,
    ) -> tuple[bool, str, bool]:
        """
        Run the cascade. Returns (accepted, final_output, escalated).

        Step 1: Structural validation (deterministic, 0 tokens)
        Step 2: Semantic coherence (MiniLM, 0 API tokens)
        Step 3: Binary escalation (remote call, ~200 tokens)
        """
        if not output or not output.strip():
            return False, output, False

        # For math tasks, try to compute the expected answer from the query
        expected_answer: str | None = None
        if task_type == "math" and query.strip():
            expected_answer = try_evaluate_math(query)

        if not self._structural_validate(task_type, output, expected_answer):
            logger.info("CASCADE: structural validation failed (type=%s)", task_type)
            if remote_escalate_fn:
                return self._escalate(query, output, remote_escalate_fn)
            return False, output, False

        if self._config.coherence_threshold > 0:
            encoder = self._get_encoder()
            if encoder is not None and not self._semantic_coherent(query, output, encoder):
                logger.info("CASCADE: semantic coherence below threshold")
                if remote_escalate_fn:
                    return self._escalate(query, output, remote_escalate_fn)
                return False, output, False

        logger.info("CASCADE: output accepted (type=%s)", task_type)
        return True, output, False

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract all integers and decimal numbers (including negative) from text."""
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        return [float(m) for m in matches if m not in ("-", ".")]

    def _structural_validate(
        self,
        task_type: str,
        output: str,
        expected_answer: str | None = None,
    ) -> bool:
        if task_type == "json":
            try:
                json.loads(output)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        if task_type == "code":
            try:
                ast.parse(output)
                return True
            except SyntaxError:
                return False
        if task_type == "extraction":
            return len(output.strip()) > 10
        if task_type == "qa":
            return len(output.split()) > 2
        if task_type == "math":
            nums = self._extract_numbers(output)
            if not nums:
                return False
            if expected_answer is not None:
                try:
                    expected = float(expected_answer)
                    return any(abs(n - expected) < 1e-9 for n in nums)
                except (ValueError, TypeError):
                    return False
            return True
        return len(output.strip()) > 0

    def _semantic_coherent(self, query: str, output: str, encoder: Any | None = None) -> bool:
        if encoder is None:
            return True
        try:
            emb_query = encoder.encode(query, normalize_embeddings=True)
            emb_output = encoder.encode(output, normalize_embeddings=True)
            similarity = float(np.dot(emb_query, emb_output))
            coherent = similarity >= self._config.coherence_threshold
            logger.debug("CASCADE coherence: sim=%.4f threshold=%.2f ok=%s",
                         similarity, self._config.coherence_threshold, coherent)
            return coherent
        except Exception as exc:
            logger.warning("Coherence check error: %s", exc)
            return True

    def _escalate(
        self,
        query: str,
        output: str,
        remote_escalate_fn: Callable[[str], str],
    ) -> tuple[bool, str, bool]:
        escalation_prompt = ESCALATION_PROMPT.format(query=query, proposed=output)
        logger.info("CASCADE: escalating to binary verifier")
        try:
            result = remote_escalate_fn(escalation_prompt)
            if result.strip().upper().startswith("VALID"):
                return True, output, False
            if result.strip().upper().startswith("INVALID:"):
                corrected = re.sub(r"(?i)^INVALID:\s*", "", result).strip()
                return True, corrected or output, True
            return True, result, True
        except Exception as exc:
            logger.warning("Escalation failed: %s", exc)
            return False, output, False
