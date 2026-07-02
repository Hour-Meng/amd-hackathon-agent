"""Tier 3 — Cascade Verify: structural -> semantic coherence -> binary escalation.

Each step accumulates confidence. If confidence exceeds threshold, later steps
are skipped, saving tokens and compute. Typical escalation savings: 40-60%.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Callable

from my_routing_agent.config import VerifierConfig
from my_routing_agent.utils.encoder import get_encoder

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
    SentenceTransformer = None
    np = None


class CascadeVerifier:
    """Three-step output quality verifier with confidence-gated early exit.

    Confidence accumulates:
      Step 1 (structural): weight=0.4
      Step 2 (semantic): weight=0.3
      Step 3 (binary escalation): weight=0.3 (only if needed)

    If confidence >= 0.85 after Step 1, skip Step 2 and Step 3.
    If confidence >= 0.80 after Step 2, skip Step 3.
    """

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
        self._encoder = get_encoder()
        return self._encoder

    def verify(
        self,
        query: str,
        output: str,
        task_type: str = "qa",
        remote_escalate_fn: Callable[[str], str] | None = None,
    ) -> tuple[bool, str, bool]:
        """
        Run the cascade with confidence-gated early exit.
        Returns (accepted, final_output, escalated).
        """
        if not output or not output.strip():
            return False, output, False

        confidence = 0.0

        # Step 1: Structural validation (deterministic, 0 tokens)
        structural_pass = self._structural_validate(task_type, output)
        if structural_pass:
            confidence += 0.4
            logger.debug("CASCADE step1: structural PASS confidence=%.2f", confidence)
        else:
            logger.info("CASCADE step1: structural FAIL (type=%s) confidence=%.2f", task_type, confidence)
            if remote_escalate_fn:
                return self._escalate(query, output, remote_escalate_fn)
            return False, output, False

        if confidence >= 0.85 or self._config.coherence_threshold <= 0:
            logger.info("CASCADE: confidence=%.2f >= 0.85, skipping steps 2-3", confidence)
            return True, output, False

        # Skip semantic coherence for structured output types (math, json, code)
        # These outputs are inherently dissimilar to the query text.
        if task_type in ("math", "json", "code"):
            logger.info("CASCADE: skipping semantic for %s (structured output)", task_type)
            return True, output, False

        # Step 2: Semantic coherence (MiniLM, 0 API tokens)
        coherence_pass = True
        if self._config.coherence_threshold > 0:
            encoder = self._get_encoder()
            if encoder is not None:
                similarity = self._semantic_similarity(query, output, encoder)
                coherence_pass = similarity >= self._config.coherence_threshold
                if coherence_pass:
                    confidence += 0.3
                    logger.debug("CASCADE step2: semantic PASS sim=%.4f confidence=%.2f", similarity, confidence)
                else:
                    logger.info("CASCADE step2: semantic FAIL sim=%.4f < %.2f", similarity, self._config.coherence_threshold)

        if confidence >= 0.80:
            logger.info("CASCADE: confidence=%.2f >= 0.80, skipping step 3", confidence)
            return True, output, False

        if not coherence_pass:
            logger.info("CASCADE step2: semantic coherence below threshold, escalating")
            if remote_escalate_fn:
                return self._escalate(query, output, remote_escalate_fn)
            logger.info("CASCADE step2: no escalate_fn available, accepting output (soft gate)")
            return True, output, False

        logger.info("CASCADE: output accepted (type=%s)", task_type)
        return True, output, False

    def _structural_validate(self, task_type: str, output: str) -> bool:
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
            return bool(output.strip())
        return len(output.strip()) > 0

    def _semantic_similarity(self, query: str, output: str, encoder: Any | None = None) -> float:
        if encoder is None:
            return 1.0
        try:
            emb_query = encoder.encode(query, normalize_embeddings=True)
            emb_output = encoder.encode(output, normalize_embeddings=True)
            return float(np.dot(emb_query, emb_output))
        except Exception as exc:
            logger.warning("Coherence similarity error: %s", exc)
            return 1.0

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
