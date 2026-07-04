"""Hybrid Token-Efficient Routing Agent — Streamlit chatbot demo."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

import threading

import requests
import streamlit as st
from PIL import Image

from my_routing_agent.cache.semantic_cache import SemanticCache
from my_routing_agent.middleware.entropy import (
    compute_char_entropy,
    compute_shannon_entropy,
    normalize_entropy,
)
from my_routing_agent.routers.engine import (
    AdaptiveThreshold,
    PhantomZone,
    SklearnRouter,
)
from my_routing_agent.routers.features import FeatureExtractor
from my_routing_agent.phantom.budget import BudgetEnforcer
from my_routing_agent.phantom.confidence import ConfidencePredictor
from my_routing_agent.phantom.speculative import SpeculativeRunner
from my_routing_agent.utils.math_eval import (
    extract_arithmetic_expression,
    is_local_arithmetic,
    is_prime_check_prompt,
    is_symbolic_math,
    try_prime_check,
)
from my_routing_agent.verifier.cascade import CascadeVerifier

logger = logging.getLogger("hybrid_router")

# ANGKOR + PHANTOM module-level singletons (lazy-init in main)
_ANGKOR_CACHE: SemanticCache | None = None
_ANGKOR_SKLEARN_ROUTER: SklearnRouter | None = None
_ANGKOR_ADAPTIVE_THETA: AdaptiveThreshold | None = None
_ANGKOR_PHANTOM_RUNNER: SpeculativeRunner | None = None
_ANGKOR_VERIFIER: CascadeVerifier | None = None
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

LOCAL_ENDPOINT = "http://localhost:11434/api/generate"
LOCAL_TAGS_ENDPOINT = "http://localhost:11434/api/tags"
REMOTE_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"

# --- Local-backend safety: health gate, timeouts, memory-aware routing ------------
# A heavy local model (e.g. qwen2.5:32b) can freeze the whole machine. These bounds
# guarantee the Streamlit app never blocks indefinitely on a local request.
LOCAL_HEALTH_TTL_SECONDS = 30.0      # cache a dead/alive verdict to avoid retry storms
LOCAL_HEALTH_TIMEOUT_SECONDS = 2.0   # fast probe of the Ollama backend
LOCAL_INFERENCE_TIMEOUT_SECONDS = 60  # strict read-timeout for a normal local model
LOCAL_HEAVY_INFERENCE_TIMEOUT_SECONDS = 25  # tighter ceiling for heavy local models
LOCAL_DECOMP_TIMEOUT_SECONDS = 20    # planner/distill local calls must stay snappy
MEMORY_PRESSURE_THRESHOLD = 0.85     # bypass heavy local above this RAM utilization
HEAVY_LOCAL_PARAM_BILLIONS = 30      # >= this many params counts as "heavy"
TRIVIAL_PROMPT_MAX_CHARS = 15        # very short prompts are treated as trivial
DISTILL_MIN_CHARS = 80               # skip compression for prompts shorter than this
DEFAULT_COMPLEXITY_THRESHOLD = 30    # complexity score (0-100) above which we escalate

# Routing reasons emitted when a heavy local model is intentionally skipped.
LOCAL_UNAVAILABLE_REASON = "local-backend-unavailable"
MEMORY_PRESSURE_REASON = "memory-pressure:bypass-heavy-local"
HEAVY_LOCAL_BYPASS_REASON = "heavy-local-bypass:trivial"
LOCAL_TIMEOUT_REASON = "local-timeout:fallback-remote"
ROUTER_DEFAULT_REMOTE = "router-default:remote"
REMOTE_FACTUAL_REASON = "remote-required:factual"
REMOTE_CODE_REASON = "remote-required:code"
REMOTE_LONG_REASON = "remote-required:long"
REMOTE_MULTIHOP_REASON = "remote-required:multi-hop"
PRIOR_FAILURE_REASON = "remote-required:prior-local-failure"
LOCAL_GREETING_REASON = "local-allowed:greeting"
CANNED_REPLY_REASON = "local-allowed:canned"
LOCAL_MATH_REASON = "local-allowed:math-python"
REMOTE_SYMBOLIC_MATH_REASON = "remote-required:symbolic-math"
LOCAL_PRIME_REASON = "local-allowed:prime-check"
LOCAL_CREATIVE_REASON = "local-allowed:creative"
ENTROPY_GATE_REASON = "entropy-gate:unclear-input"
LONG_PROMPT_CHARS = 180
MAX_TXT_CONTEXT_CHARS = 4000
DEFAULT_MAX_SUB_AGENTS = 1
HARD_MAX_SUB_AGENTS = 4
MIN_SUBTASK_CHARS = int(os.getenv("MIN_SUBTASK_CHARS", "12"))
MIN_SUBTASK_WORDS = int(os.getenv("MIN_SUBTASK_WORDS", "3"))
SWARM_MAX_CONCURRENT = int(os.getenv("SWARM_MAX_CONCURRENT", "2"))
LONG_CONTEXT_LINE_THRESHOLD = 12
REMOTE_MAX_TOKENS_GREETING = 32
REMOTE_MAX_TOKENS_SIMPLE = 64
REMOTE_MAX_TOKENS_DEFAULT = int(os.getenv("FIREWORKS_MAX_TOKENS", "768"))
REMOTE_MAX_TOKENS_LONG = int(os.getenv("REMOTE_MAX_TOKENS_LONG", "1024"))
REMOTE_MAX_TOKENS_CAP = int(os.getenv("REMOTE_MAX_TOKENS_CAP", "2048"))
RATE_LIMIT_MAX_RETRIES = int(os.getenv("RATE_LIMIT_MAX_RETRIES", "3"))
RATE_LIMIT_BACKOFF_BASE = float(os.getenv("RATE_LIMIT_BACKOFF_BASE", "1.0"))
ENTROPY_INPUT_GATE_THRESHOLD = float(os.getenv("ENTROPY_INPUT_GATE_THRESHOLD", "3.8"))
ENTROPY_CHAR_GATE_THRESHOLD = float(os.getenv("ENTROPY_CHAR_GATE_THRESHOLD", "3.9"))
REMOTE_UI_TIMEOUT_SECONDS = 120
INSTANT_GREETING_MS_BUDGET = 200  # UI must show feedback within this window for greetings
ROOT_DIR = Path(__file__).resolve().parent
UI_EXECUTOR = ThreadPoolExecutor(max_workers=2)

_GREETING_ACK_PATTERN = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|thx|bye|goodbye|good morning|"
    r"good afternoon|good evening|how are you)(?:[!?. ]+.*)?$",
    re.IGNORECASE,
)
CODE_GEN_PATTERNS = (
    "write code",
    "write a function",
    "implement ",
    "debug this",
    "python script",
    "javascript",
    "typescript",
    "def ",
    "class ",
    "```",
)
FORMAT_PATTERNS = ("to uppercase", "to lowercase", "capitalize ")
CREATIVE_PATTERNS = (
    "write a poem",
    "write a story",
    "write me a poem",
    "write me a story",
    "haiku",
    "limerick",
    "creative writing",
    "compose a",
    "imagine a",
    "short story",
    "fictional",
)

# Defaults / catalogs surfaced in the sidebar.
# Local model is a lightweight utility (greetings/math/format only) — NOT the default generator.
DEFAULT_LOCAL_MODEL = "qwen2.5:0.5b"
CUSTOM_MODEL_SENTINEL = "Custom..."
REMOTE_MODEL_OPTIONS = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/qwen3p7-plus",
    CUSTOM_MODEL_SENTINEL,
]
DEFAULT_REMOTE_MODEL = REMOTE_MODEL_OPTIONS[0]
# Vision requires a multimodal model regardless of the text-model selection.
REMOTE_VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"

FIREWORKS_MODEL_PREFIX = "accounts/fireworks/models/"

# Priority-ordered remote text models. The first entry is the default; on
# NOT_FOUND / inaccessible / not-deployed the executor tries the next in order.
REMOTE_MODEL_CANDIDATES = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/qwen3p7-plus",
]

# Provider/model registry: models known to be deployed and accessible on
# Fireworks. The router validates against this BEFORE the first call so an
# invalid/undeployed selection is never used as the first attempt.
KNOWN_DEPLOYED_REMOTE_MODELS = frozenset(
    {
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/qwen3p7-plus",
        "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
    }
)

# Substrings in a Fireworks error body that mean "this model can't be used".
MODEL_UNAVAILABLE_MARKERS = (
    "not_found",
    "not found",
    "does not exist",
    "not deployed",
    "inaccessible",
    "invalid model",
    "no such model",
    "unknown model",
    "model not found",
)


def normalize_model_id(model_id: str) -> str:
    """
    Normalize a Fireworks model id to canonical
    'accounts/fireworks/models/<id>' form. Bare ids and stray slashes are fixed.
    """
    if not model_id:
        return ""
    mid = model_id.strip().strip("/")
    if not mid:
        return ""
    if mid.startswith("accounts/") and "/models/" in mid:
        return mid
    return f"{FIREWORKS_MODEL_PREFIX}{mid.split('/')[-1]}"


def is_known_deployed_model(model_id: str) -> bool:
    """True if the (normalized) model id is in the deployed/accessible registry."""
    return normalize_model_id(model_id) in KNOWN_DEPLOYED_REMOTE_MODELS


def build_remote_candidates(selected_model: str) -> list[str]:
    """
    Ordered, de-duplicated remote attempt list.

    - The user-selected model is tried FIRST only when it is known-deployed.
    - Otherwise the validated candidates lead, and the unknown selection is kept
      as a last-resort attempt (so we still try it, but never first).
    """
    selected = normalize_model_id(selected_model)
    ordered: list[str] = []

    if selected and is_known_deployed_model(selected):
        ordered.append(selected)

    for cand in REMOTE_MODEL_CANDIDATES:
        normalized = normalize_model_id(cand)
        if normalized and normalized not in ordered and is_known_deployed_model(normalized):
            ordered.append(normalized)

    # Unknown selection: still attempt it, but only after the known-good models.
    if selected and selected not in ordered:
        ordered.append(selected)

    return ordered


def _is_model_unavailable(status_code: int, body: str) -> bool:
    """Classify a Fireworks response as 'model is invalid/undeployed/inaccessible'."""
    if status_code == 404:
        return True
    if status_code in (400, 403):
        low = (body or "").lower()
        return any(marker in low for marker in MODEL_UNAVAILABLE_MARKERS)
    return False


def _extract_remote_answer(data: dict) -> str | None:
    """Return assistant text from common chat/reasoning response shapes."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if not isinstance(first, dict):
        return None

    message = first.get("message")
    if isinstance(message, dict):
        for key in ("content", "reasoning_content"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("text", "content", "reasoning_content"):
        value = first.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


RouteName = Literal[
    "MATH_PYTHON",
    "VISION_REMOTE",
    "TEXT_LOCAL",
    "TEXT_REMOTE",
    "FALLBACK_REMOTE",
    "AGENT_SWARM",
    "CACHE_HIT",
    "PHANTOM_RACE",
]

_INVALID_SUBTASK_PATTERN = re.compile(r"^[\d\]\[\)\(]+$")

# Feature-weighted complexity scoring.
COMPLEXITY_BASE_SCORE = 1
COMPLEXITY_CHARS_PER_POINT = 50
COMPLEXITY_KEYWORD_WEIGHT = 50
COMPLEXITY_KEYWORDS = (
    "derive",
    "analyze",
    "explain",
    "computable",
    "algorithm",
    "tree",
    "code",
)

# Goldilocks guardrails: concise answers, permit how-to/general knowledge,
# deny only factual impossibilities and logical contradictions.
SUB_AGENT_SYSTEM_PROMPT = """You are a concise, direct answering agent.
Rules:
1. Provide the direct answer. No greetings, no fluff.
2. "How-to" instructions, general knowledge, facts, and open-ended factual
   requests (e.g. "an amazing thing about X", "a fact about Y") are ALL valid.
   Answer them neutrally and directly with a real fact.
3. NEVER argue with or second-guess a well-formed request. Do not reply that a
   subject is "not amazing", "not interesting", "just a country", or otherwise
   judge the premise of a normal factual question.
4. ONLY output an "Error:" if the premise is factually impossible or a logical
   contradiction (e.g. asking for the capital of a city).
5. For creative tasks, output ONLY the final creative text — no reasoning,
   steps, scratchpad, or meta-commentary.

Examples:
Task: capital of France -> Paris.
Task: capital of London -> Error: London is a city, not a country.
Task: Tell me 1 amazing thing about France -> The Eiffel Tower was the tallest man-made structure in the world until 1930.
Task: How to open a jar of jam -> Twist the lid counter-clockwise. Run under warm water if stuck.
Task: Who is Lebron James -> American professional basketball player."""

DISTILL_SYSTEM_PROMPT = (
    "You are a STRUCTURE-PRESERVING prompt compressor. Rewrite the user's text in "
    "fewer words WITHOUT changing what is being asked.\n"
    "RULES:\n"
    "1. NEVER answer, solve, or explain the prompt. Only rewrite it more tersely.\n"
    "2. Preserve EVERY task, question, instruction, bullet, and numbered step as a "
    "SEPARATE line. Never merge multiple tasks into one generic sentence.\n"
    "3. Keep the SAME number of items as the input. If the input has 3 tasks, output 3.\n"
    "4. Keep all numbers, names, code, equations, units, and constraints verbatim.\n"
    "5. Remove only greetings and filler. Output one task per line, no extra prose.\n\n"
    "EXAMPLE INPUT:\n"
    "Hi! Could you please first spell 'apple' backwards for me, then also tell me "
    "what 9 + 10 is, and finally share one fact about France?\n"
    "EXAMPLE OUTPUT:\n"
    "spell 'apple' backwards\n"
    "what is 9 + 10\n"
    "one fact about France"
)

# Strict few-shot planner: parent context MUST be appended to every sub-task so that
# fragments like "Paris" or "Cambodia" are never emitted as standalone words.
TASK_DECOMPOSITION_SYSTEM = (
    "You are a strict task decomposition planner. Split the user's prompt into a JSON "
    "array of clean, explicit, self-contained sub-tasks.\n"
    "RULES:\n"
    "1. NEVER emit a bare noun or single word (e.g. 'Paris', 'Cambodia'). Each sub-task "
    "must carry the FULL parent context/intent of the original prompt.\n"
    "2. If the prompt shares a verb or qualifier across a list (e.g. 'capital of'), "
    "repeat that qualifier on EVERY sub-task.\n"
    "3. Any arithmetic or mathematical expression must be emitted as 'math: <expression>'.\n"
    "4. Output ONLY a valid JSON array of strings. No markdown, no prose, no keys.\n"
    "5. If the prompt is a single task, return a one-element array.\n\n"
    "EXAMPLE INPUT:\n"
    "tell me the capital of france, london, 2+12\n"
    "EXAMPLE OUTPUT:\n"
    '["capital of France", "capital of London", "math: 2+12"]\n\n'
    "EXAMPLE INPUT:\n"
    "tell me the capital of france, london, paris, cambodia, 2+12\n"
    "EXAMPLE OUTPUT:\n"
    '["capital of France", "capital of London", "capital of Paris", '
    '"capital of Cambodia", "math: 2+12"]\n\n'
    "EXAMPLE INPUT:\n"
    "what is the population of japan and who wrote hamlet\n"
    "EXAMPLE OUTPUT:\n"
    '["population of Japan", "who wrote Hamlet"]\n\n'
    "EXAMPLE INPUT:\n"
    "summarize the french revolution\n"
    "EXAMPLE OUTPUT:\n"
    '["summarize the French Revolution"]'
)


@dataclass
class RouteResult:
    answer: str
    route: RouteName
    tokens: int
    latency_ms: float
    original_prompt: str = ""
    model_used: str = ""
    distilled_prompt: str | None = None
    distillation_chars_saved: int = 0
    distillation_error: str | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    routing_reason: str | None = None
    complexity_score: int | None = None
    confidence_score: float | None = None
    retries: int = 0
    wall_clock_ms: float | None = None
    sub_results: list[RouteResult] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    # Hierarchical orchestration metadata (classifier → planner → executor).
    prompt_type: str | None = None
    decomposition_used: bool = False
    num_agents: int = 1
    escalation_reason: str | None = None
    message_id: str = ""


@dataclass
class RequestTiming:
    """Per-request stage timestamps for routing/dispatch latency diagnosis."""

    origin: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def mark(self, stage: str, **meta: object) -> None:
        self.stages[stage] = time.perf_counter()
        if meta:
            self.meta.update(meta)

    def ms_since_origin(self, stage: str) -> float | None:
        ts = self.stages.get(stage)
        if ts is None:
            return None
        return (ts - self.origin) * 1000.0

    def ms_between(self, start: str, end: str) -> float | None:
        s, e = self.stages.get(start), self.stages.get(end)
        if s is None or e is None:
            return None
        return (e - s) * 1000.0

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {"stages_ms": {}}
        stages_ms: dict[str, float] = {}
        for name, ts in self.stages.items():
            stages_ms[name] = round((ts - self.origin) * 1000.0, 2)
        out["stages_ms"] = stages_ms
        router_ms = self.ms_between("router_start", "route_decision")
        if router_ms is not None:
            out["router_ms"] = round(router_ms, 2)
        model_ms = self.ms_between("model_call_start", "model_call_end")
        if model_ms is not None:
            out["model_call_ms"] = round(model_ms, 2)
        total_ms = self.ms_between("input_received", "final_response")
        if total_ms is not None:
            out["total_ms"] = round(total_ms, 2)
        feedback_ms = self.ms_between("input_received", "ui_feedback_shown")
        if feedback_ms is not None:
            out["ui_feedback_ms"] = round(feedback_ms, 2)
        first_token_ms = self.ms_between("model_call_start", "first_token")
        if first_token_ms is not None:
            out["first_token_ms"] = round(first_token_ms, 2)
        out.update(self.meta)
        return out

    def attach(self, result: RouteResult) -> None:
        result.diagnostics["timing"] = self.to_dict()

    def log_summary(self, prompt: str) -> None:
        logger.info("TIMING prompt=%r %s", prompt.strip()[:60], self.to_dict())


def _render_key(prefix: str, result: RouteResult, suffix: str = "") -> str:
    """Unique Streamlit key for telemetry widgets (stable per chat message)."""
    mid = result.message_id or str(id(result))
    return f"{prefix}_{mid}_{suffix}".rstrip("_")


def _has_ui_context() -> bool:
    """True only on the Streamlit script thread; False inside worker threads."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def compress_image_to_base64(image_file: BinaryIO) -> str:
    """Downscale image to 512px max edge and return a JPEG data URI."""
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((512, 512))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def calculate_complexity(prompt: str) -> int:
    """
    Feature-weighted complexity score.

    Base score + 1 point per 50 chars + heavy weight per analytical keyword.
    Scores above the sidebar threshold route REMOTE; otherwise LOCAL.
    """
    score = COMPLEXITY_BASE_SCORE
    score += len(prompt) // COMPLEXITY_CHARS_PER_POINT
    lowered = prompt.lower()
    for keyword in COMPLEXITY_KEYWORDS:
        if keyword in lowered:
            score += COMPLEXITY_KEYWORD_WEIGHT
    return score


# Character-level / tokenization-sensitive patterns. Small local models tokenize
# poorly, so any of these force REMOTE regardless of complexity score.
CHARACTER_LEVEL_PATTERNS = (
    "spell",
    "reverse",
    "backward",
    "backwards",
    "count letters",
    "count characters",
    "letters of",
    "nth character",
    "how many r's",
    "anagram",
    "scramble",
)
CHARACTER_LEVEL_GUARD_REASON = "character-level tokenization guard"
FACTUAL_RISK_GUARD_REASON = REMOTE_FACTUAL_REASON  # alias for tests / telemetry

# Reasons that must skip local distillation (backend unsafe or exact text required).
LOCAL_DISTILL_UNSAFE_REASONS = frozenset(
    {
        LOCAL_UNAVAILABLE_REASON,
        MEMORY_PRESSURE_REASON,
        HEAVY_LOCAL_BYPASS_REASON,
        CHARACTER_LEVEL_GUARD_REASON,
        REMOTE_FACTUAL_REASON,
        PRIOR_FAILURE_REASON,
    }
)

# Geo/civic/identity/encyclopedia facts — always remote in router-first mode.
FACTUAL_RISK_PATTERNS = (
    "where is ",
    "where are ",
    "capital of",
    "what is the capital",
    "what's the capital",
    "what is the population",
    "population of",
    "who wrote",
    "who is ",
    "what country",
    "what city",
    " a country or a city",
    " a city or a country",
    "located in",
    "border of",
    "amazing thing about",
    "fact about",
)
# Low-risk casual prompts explicitly allowed on weak local models.
LOCAL_TRIVIAL_WHITELIST_PATTERNS = (
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "how are you",
    "good morning",
    "good evening",
)


def is_character_level_task(prompt: str) -> bool:
    """True if the prompt requires exact character-level manipulation."""
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in CHARACTER_LEVEL_PATTERNS)


def is_factual_risk_prompt(prompt: str) -> bool:
    """True for direct geographic, civic, identity, or encyclopedia-style facts."""
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in FACTUAL_RISK_PATTERNS)


def is_weak_local_model(model_id: str) -> bool:
    """Small Ollama models that should not serve high-stakes factual queries."""
    mid = model_id.lower().strip()
    return ":0.5b" in mid or mid in {"qwen2.5:0.5b"}


def is_local_trivial_whitelisted(prompt: str) -> bool:
    """Trivial casual prompts that may stay LOCAL even on a weak model."""
    lowered = prompt.lower().strip()
    if not lowered:
        return False
    for pattern in LOCAL_TRIVIAL_WHITELIST_PATTERNS:
        if lowered == pattern or lowered.startswith(f"{pattern} ") or lowered.startswith(pattern):
            return True
    return False


def is_greeting_or_tiny_chat(prompt: str) -> bool:
    """Fast precheck: greetings, acknowledgements, ultra-short chat."""
    stripped = prompt.strip()
    if not stripped:
        return False
    if len(stripped) <= TRIVIAL_PROMPT_MAX_CHARS and _GREETING_ACK_PATTERN.match(stripped):
        return True
    return is_local_trivial_whitelisted(stripped)


def has_complex_attachment(*, has_image: bool, txt_context_chars: int) -> bool:
    return has_image or txt_context_chars > 0


def get_canned_greeting_reply(prompt: str) -> str:
    lowered = prompt.strip().lower()
    if lowered.startswith(("thank", "thx")):
        return "You're welcome! Let me know if you need anything else."
    if lowered.startswith(("bye", "goodbye")):
        return "Goodbye! Feel free to come back anytime."
    if "how are you" in lowered:
        return "I'm doing well, thanks for asking! How can I help you today?"
    return "Hello! How can I help you today?"


def is_pure_greeting_request(
    prompt: str,
    *,
    has_image: bool = False,
    has_txt_context: bool = False,
) -> bool:
    """True for trivial greetings with no attachments — eligible for instant UI path."""
    if has_image or has_txt_context:
        return False
    return is_greeting_or_tiny_chat(prompt)


def should_skip_expensive_preprocess(
    prompt: str,
    *,
    has_image: bool = False,
    has_txt_context: bool = False,
) -> bool:
    """Skip FAISS cache, PHANTOM race, planner, and cascade verify for trivial prompts."""
    if has_image or has_txt_context:
        return False
    if is_greeting_or_tiny_chat(prompt):
        return True
    if would_math_intercept(prompt):
        return True
    return False


def is_trivial_fast_path(
    prompt: str,
    *,
    has_image: bool = False,
    has_txt_context: bool = False,
) -> bool:
    """UI instant path: greetings and deterministic math."""
    return should_skip_expensive_preprocess(
        prompt, has_image=has_image, has_txt_context=has_txt_context
    )


def build_txt_context(file_bytes: bytes | None, *, max_chars: int = MAX_TXT_CONTEXT_CHARS) -> tuple[str, int]:
    """Return labeled, truncated file context and its character count."""
    if not file_bytes:
        return "", 0
    try:
        raw = file_bytes.decode("utf-8", errors="replace").strip()
    except Exception:
        return "", 0
    if not raw:
        return "", 0
    content_chars = min(len(raw), max_chars)
    truncated = raw[:max_chars]
    if len(raw) > max_chars:
        truncated += "\n...[truncated]"
    block = f"\n\n[Attached context from file]\n{truncated}\n[/Attached context]"
    return block, content_chars


def is_code_generation_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in CODE_GEN_PATTERNS)


def is_simple_format_task(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in FORMAT_PATTERNS) and not is_character_level_task(
        prompt
    )


def is_creative_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in CREATIVE_PATTERNS)


def is_multi_hop_prompt(prompt: str) -> bool:
    """Multi-part or analytical prompts that need remote reasoning."""
    if is_symbolic_math(prompt) or is_local_arithmetic(prompt):
        return False
    lowered = prompt.lower()
    if lowered.count("?") > 1:
        return True
    if " and " in lowered and any(keyword in lowered for keyword in COMPLEXITY_KEYWORDS):
        return True
    if lowered.count(",") >= 2 and any(
        kw in lowered for kw in ("what", "who", "where", "capital", "tell me")
    ):
        return True
    return any(keyword in lowered for keyword in COMPLEXITY_KEYWORDS)


def should_entropy_gate_input(prompt: str, *, threshold: float | None = None) -> bool:
    """Gate gibberish / keyboard-mash prompts before any model call."""
    stripped = prompt.strip()
    if len(stripped) < 8:
        return False
    gate = threshold if threshold is not None else ENTROPY_INPUT_GATE_THRESHOLD
    if compute_char_entropy(stripped) >= ENTROPY_CHAR_GATE_THRESHOLD:
        return True
    return compute_shannon_entropy(stripped) >= gate


def strip_reasoning_traces(text: str) -> str:
    """Remove chain-of-thought / scratchpad leakage from model output."""
    if not text:
        return text
    cleaned = text.strip()
    think_open = "<" + "think" + ">"
    think_close = "<" + "/" + "think" + ">"
    think_pattern = re.escape(think_open) + r"[\s\S]*?" + re.escape(think_close)
    cleaned = re.sub(think_pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", cleaned, flags=re.IGNORECASE)
    lines = cleaned.splitlines()
    kept: list[str] = []
    for line in lines:
        stripped_line = line.strip()
        if re.match(
            r"^(?:step\s*\d+|thought:|reasoning:|analysis:|scratchpad:)",
            stripped_line,
            re.I,
        ):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if "\n\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        if len(parts) > 1:
            cleaned = parts[-1]
    return cleaned.strip()


def is_response_truncated(data: dict) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    return first.get("finish_reason") == "length"


def is_rate_limit_error(status_code: int, body: str) -> bool:
    if status_code == 429:
        return True
    low = (body or "").lower()
    return "rate_limit" in low or "rate limit" in low or "too many requests" in low


def is_valid_subtask(fragment: str) -> bool:
    """Reject garbage fragments from regex splitting."""
    stripped = fragment.strip()
    if not stripped:
        return False
    if len(stripped) < MIN_SUBTASK_CHARS:
        return False
    if len(stripped.split()) < MIN_SUBTASK_WORDS and not is_local_arithmetic(stripped):
        return False
    if _INVALID_SUBTASK_PATTERN.match(stripped):
        return False
    if is_symbolic_math(stripped) and len(stripped) < 20:
        return False
    return True


def merge_short_fragments(parts: list[str]) -> list[str]:
    """Merge fragments that fail is_valid_subtask into neighbors."""
    if not parts:
        return parts
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip() if buffer else part.strip()
        if is_valid_subtask(candidate):
            merged.append(candidate)
            buffer = ""
        elif buffer:
            buffer = candidate
        else:
            buffer = part.strip()
    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        elif is_valid_subtask(buffer) or len(buffer.split()) >= MIN_SUBTASK_WORDS:
            merged.append(buffer)
    return merged[:HARD_MAX_SUB_AGENTS]


def is_local_capable_prompt(prompt: str) -> bool:
    """Prompts within the local capability ceiling (greetings/format — no LLM facts)."""
    if is_local_trivial_whitelisted(prompt):
        return True
    if is_creative_prompt(prompt):
        return True
    return is_simple_format_task(prompt)


def would_math_intercept(prompt: str) -> bool:
    """True when deterministic math eval would handle the prompt (no LLM)."""
    if is_symbolic_math(prompt):
        return False
    return is_local_arithmetic(prompt)


_PRIOR_LOCAL_FAILURES: set[str] = set()
_PRIOR_FAILURE_LOCK = threading.Lock()


def _prompt_failure_key(prompt: str) -> str:
    return prompt.strip().lower()[:300]


def mark_prior_local_failure(prompt: str) -> None:
    with _PRIOR_FAILURE_LOCK:
        _PRIOR_LOCAL_FAILURES.add(_prompt_failure_key(prompt))


def had_prior_local_failure(prompt: str) -> bool:
    with _PRIOR_FAILURE_LOCK:
        return _prompt_failure_key(prompt) in _PRIOR_LOCAL_FAILURES


def reset_prior_local_failures() -> None:
    with _PRIOR_FAILURE_LOCK:
        _PRIOR_LOCAL_FAILURES.clear()


_HEAVY_PARAM_PATTERN = re.compile(r":(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def is_heavy_local_model(model_id: str) -> bool:
    """
    True for large local models (>= HEAVY_LOCAL_PARAM_BILLIONS params, e.g.
    qwen2.5:32b / llama3:70b) that can freeze a laptop on inference.
    """
    if not model_id:
        return False
    match = _HEAVY_PARAM_PATTERN.search(model_id.lower())
    if not match:
        return False
    try:
        return float(match.group(1)) >= HEAVY_LOCAL_PARAM_BILLIONS
    except ValueError:
        return False


def is_trivial_prompt(prompt: str) -> bool:
    """Short/casual prompts that should never warrant a heavy local model."""
    stripped = prompt.strip()
    if not stripped:
        return True
    if is_local_trivial_whitelisted(stripped):
        return True
    return len(stripped) <= TRIVIAL_PROMPT_MAX_CHARS and "?" not in stripped


def get_memory_usage() -> float | None:
    """System RAM utilization in [0, 1], or None when psutil is unavailable."""
    try:
        import psutil

        return float(psutil.virtual_memory().percent) / 100.0
    except Exception:
        return None


def is_memory_constrained(threshold: float = MEMORY_PRESSURE_THRESHOLD) -> bool:
    """True only when memory usage is known AND at/above the pressure threshold."""
    usage = get_memory_usage()
    return usage is not None and usage >= threshold


# Health-gate cache: model_id -> (expiry_epoch, healthy). Guarded for swarm threads.
_LOCAL_HEALTH_LOCK = threading.Lock()
_LOCAL_HEALTH_CACHE: dict[str, tuple[float, bool]] = {}


def reset_local_health_cache() -> None:
    """Clear the cached health verdicts (used by tests and the UI refresh)."""
    with _LOCAL_HEALTH_LOCK:
        _LOCAL_HEALTH_CACHE.clear()


def check_local_health(model_id: str, *, ttl: float = LOCAL_HEALTH_TTL_SECONDS) -> bool:
    """
    Probe the Ollama backend for liveness and that `model_id` is actually pulled.
    The verdict is cached for `ttl` seconds so a dead backend is not hammered.
    """
    key = (model_id or "").strip()
    now = time.time()

    with _LOCAL_HEALTH_LOCK:
        cached = _LOCAL_HEALTH_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]

    healthy = False
    try:
        resp = requests.get(LOCAL_TAGS_ENDPOINT, timeout=LOCAL_HEALTH_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            names = {str(m.get("name", "")) for m in resp.json().get("models", [])}
            if not names:
                # Server is up but reports no catalog; allow the attempt.
                healthy = True
            else:
                healthy = key in names or f"{key}:latest" in names
    except Exception as exc:
        logger.warning("Local health check failed for %s: %s", key, exc)
        healthy = False

    with _LOCAL_HEALTH_LOCK:
        _LOCAL_HEALTH_CACHE[key] = (now + ttl, healthy)
    return healthy


def is_local_unavailable(model_id: str) -> bool:
    """Convenience inverse of check_local_health (cached)."""
    return not check_local_health(model_id)


def get_local_health_for_ui(model_id: str) -> bool | None:
    """
    Non-blocking health read for the sidebar banner.
    Returns cached verdict or None while a background refresh is pending.
    """
    key = (model_id or "").strip()
    now = time.time()
    stale_verdict: bool | None = None
    with _LOCAL_HEALTH_LOCK:
        cached = _LOCAL_HEALTH_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]
        if cached:
            stale_verdict = cached[1]
    schedule_local_health_refresh(key)
    return stale_verdict


def schedule_local_health_refresh(model_id: str) -> None:
    """Probe Ollama on a worker thread so the Streamlit rerun is not blocked."""
    key = (model_id or "").strip()
    if not key:
        return
    try:
        UI_EXECUTOR.submit(check_local_health, key)
    except Exception as exc:
        logger.debug("Could not schedule health refresh for %s: %s", key, exc)


@dataclass
class RouterDecision:
    """Authoritative routing decision produced before any backend is touched."""

    route: Literal["LOCAL", "REMOTE"]
    complexity_score: int
    reason: str
    model_id: str
    confidence_score: float = 0.0


def compute_remote_max_tokens(prompt: str, decision: RouterDecision) -> int:
    if is_greeting_or_tiny_chat(prompt) or decision.reason in {
        LOCAL_GREETING_REASON,
        CANNED_REPLY_REASON,
    }:
        return REMOTE_MAX_TOKENS_GREETING
    if decision.reason in {REMOTE_SYMBOLIC_MATH_REASON} or is_symbolic_math(prompt):
        return REMOTE_MAX_TOKENS_LONG
    if is_synthesis_or_summary_task(prompt) or is_creative_prompt(prompt):
        return REMOTE_MAX_TOKENS_LONG
    if decision.complexity_score < 20:
        return REMOTE_MAX_TOKENS_SIMPLE
    if len(prompt.strip()) > LONG_PROMPT_CHARS:
        return REMOTE_MAX_TOKENS_LONG
    return REMOTE_MAX_TOKENS_DEFAULT


def route_decision(
    prompt: str,
    threshold: int,
    *,
    has_image: bool,
    active_local_model: str,
    active_remote_model: str,
    local_unavailable: bool = False,
    memory_pressure: float | None = None,
    allow_heavy_local: bool = False,
    has_txt_context: bool = False,
) -> RouterDecision:
    """
    Router-first single source of truth. Runs BEFORE any model call.

    Default is REMOTE (Fireworks). LOCAL is allowed only for deterministic math,
    short greetings, and simple formatting when the local backend is healthy.
    Greetings without complex attachments fail closed to LOCAL/canned — never REMOTE.
    """
    score = min(100, calculate_complexity(prompt))
    remote_model = normalize_model_id(active_remote_model) or DEFAULT_REMOTE_MODEL

    if has_image:
        return RouterDecision("REMOTE", score, "vision:remote", remote_model, 0.95)

    # Fast precheck — skip remote/phantom paths for pure greetings.
    if is_greeting_or_tiny_chat(prompt) and not has_complex_attachment(
        has_image=False, txt_context_chars=1 if has_txt_context else 0
    ):
        if would_math_intercept(prompt):
            return RouterDecision("LOCAL", score, LOCAL_MATH_REASON, "python-eval", 1.0)
        if local_unavailable:
            return RouterDecision(
                "LOCAL", score, CANNED_REPLY_REASON, "canned-reply", 0.95
            )
        return RouterDecision(
            "LOCAL", score, LOCAL_GREETING_REASON, active_local_model, 0.95
        )

    if is_character_level_task(prompt):
        return RouterDecision(
            "REMOTE", score, CHARACTER_LEVEL_GUARD_REASON, remote_model, 0.99
        )

    if is_symbolic_math(prompt):
        return RouterDecision(
            "REMOTE", score, REMOTE_SYMBOLIC_MATH_REASON, remote_model, 0.95
        )

    if should_entropy_gate_input(prompt):
        return RouterDecision(
            "LOCAL", score, ENTROPY_GATE_REASON, "entropy-gate", 0.99
        )

    if is_prime_check_prompt(prompt):
        return RouterDecision("LOCAL", score, LOCAL_PRIME_REASON, "python-eval", 1.0)

    if would_math_intercept(prompt):
        return RouterDecision("LOCAL", score, LOCAL_MATH_REASON, "python-eval", 1.0)

    if is_creative_prompt(prompt):
        if local_unavailable:
            return RouterDecision(
                "REMOTE", score, LOCAL_UNAVAILABLE_REASON, remote_model, 0.75
            )
        if is_heavy_local_model(active_local_model) and not allow_heavy_local:
            return RouterDecision(
                "REMOTE", score, HEAVY_LOCAL_BYPASS_REASON, remote_model, 0.80
            )
        return RouterDecision(
            "LOCAL", score, LOCAL_CREATIVE_REASON, active_local_model, 0.88
        )

    if is_factual_risk_prompt(prompt) and not is_local_trivial_whitelisted(prompt):
        return RouterDecision("REMOTE", score, REMOTE_FACTUAL_REASON, remote_model, 0.92)

    if is_code_generation_prompt(prompt):
        return RouterDecision("REMOTE", score, REMOTE_CODE_REASON, remote_model, 0.90)

    if had_prior_local_failure(prompt):
        return RouterDecision("REMOTE", score, PRIOR_FAILURE_REASON, remote_model, 0.85)

    if len(prompt.strip()) > LONG_PROMPT_CHARS or score > threshold:
        return RouterDecision("REMOTE", score, REMOTE_LONG_REASON, remote_model, 0.82)

    if is_multi_hop_prompt(prompt):
        return RouterDecision("REMOTE", score, REMOTE_MULTIHOP_REASON, remote_model, 0.88)

    if is_local_capable_prompt(prompt):
        if local_unavailable:
            return RouterDecision(
                "REMOTE", score, LOCAL_UNAVAILABLE_REASON, remote_model, 0.75
            )
        if memory_pressure is not None and memory_pressure >= MEMORY_PRESSURE_THRESHOLD:
            return RouterDecision("REMOTE", score, MEMORY_PRESSURE_REASON, remote_model, 0.78)
        if is_heavy_local_model(active_local_model) and not allow_heavy_local:
            return RouterDecision(
                "REMOTE", score, HEAVY_LOCAL_BYPASS_REASON, remote_model, 0.80
            )
        return RouterDecision(
            "LOCAL", score, LOCAL_GREETING_REASON, active_local_model, 0.90
        )

    # Router-first default: remote is the generator for everything else.
    return RouterDecision("REMOTE", score, ROUTER_DEFAULT_REMOTE, remote_model, 0.70)


def _log_routing(
    prompt: str,
    decision: RouterDecision,
    *,
    local_healthy: bool,
) -> None:
    preview = prompt.strip().replace("\n", " ")[:60]
    logger.info(
        "ROUTING route=%s model_id=%s reason=%s confidence=%.2f score=%s "
        "local_healthy=%s memory=%s prompt=%r",
        decision.route,
        decision.model_id,
        decision.reason,
        decision.confidence_score,
        decision.complexity_score,
        local_healthy,
        get_memory_usage(),
        preview,
    )


def _log_dispatch_telemetry(
    prompt: str,
    decision: RouterDecision,
    result: RouteResult,
    *,
    token_budget: int,
    file_attached: bool,
    dispatcher_route: str,
) -> None:
    match = (
        (decision.route == "REMOTE" and result.route in {"TEXT_REMOTE", "FALLBACK_REMOTE", "VISION_REMOTE", "PHANTOM_RACE"})
        or (
            decision.route == "LOCAL"
            and result.route in {"TEXT_LOCAL", "MATH_PYTHON", "CACHE_HIT"}
        )
        or decision.reason == CANNED_REPLY_REASON
    )
    logger.info(
        "DISPATCH route_decision=%s complexity=%s token_budget=%s model_id=%s "
        "executed_route=%s latency_ms=%.1f file_attached=%s dispatcher_match=%s prompt=%r",
        decision.route,
        decision.complexity_score,
        token_budget,
        result.model_used or decision.model_id,
        result.route,
        result.latency_ms,
        file_attached,
        match,
        prompt.strip()[:80],
    )
    timing_diag = result.diagnostics.get("timing")
    if timing_diag:
        logger.info("DISPATCH_TIMING %s", timing_diag)


def _log_executed(result: RouteResult) -> None:
    attempts = result.diagnostics.get("remote_attempts")
    local_diag = result.diagnostics.get("local", {})
    logger.info(
        "EXECUTED backend=%s model_id=%s reason=%s confidence=%s fallback=%s "
        "latency_ms=%.1f remote_attempts=%s local_error=%s",
        result.route,
        result.model_used,
        result.routing_reason,
        result.confidence_score,
        result.fallback_used,
        result.latency_ms,
        attempts or [],
        local_diag.get("error_type") if isinstance(local_diag, dict) else None,
    )


def _log_local_attempt(
    *,
    model_name: str,
    backend: str,
    route: str,
    timeout_ms: int,
    fallback_used: bool,
    error_type: str | None,
) -> None:
    """Structured observability for every local-backend attempt."""
    logger.info(
        "LOCAL attempt model_name=%s backend=%s route=%s timeout_ms=%s "
        "memory_usage=%s fallback_used=%s error_type=%s",
        model_name,
        backend,
        route,
        timeout_ms,
        get_memory_usage(),
        fallback_used,
        error_type,
    )


def _local_inference_timeout(model_id: str) -> int:
    """Strict read-timeout (seconds) for a local model, tighter when heavy."""
    return (
        LOCAL_HEAVY_INFERENCE_TIMEOUT_SECONDS
        if is_heavy_local_model(model_id)
        else LOCAL_INFERENCE_TIMEOUT_SECONDS
    )


def _ollama_generate(
    *,
    prompt: str,
    model: str,
    system: str | None = None,
    timeout: int = 120,
    options: dict[str, object] | None = None,
) -> tuple[str, int]:
    """Call local Ollama generate API; return response text and eval token count."""
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options

    response = requests.post(LOCAL_ENDPOINT, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    text = str(data.get("response", "")).strip()
    tokens = int(data.get("eval_count", 0))
    return text, tokens


def safe_math_agent(prompt: str, started: float) -> RouteResult | None:
    """
    Extract embedded mathematical expressions from natural-language prompts.
    Planner-tagged tasks like 'math: 2+12' are stripped and evaluated first.
    Symbolic algebra/calculus is never handled here.
    """
    if is_symbolic_math(prompt):
        return None

    prime_answer = try_prime_check(prompt)
    if prime_answer is not None:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer=prime_answer,
            route="MATH_PYTHON",
            tokens=0,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used="python-eval",
        )

    expr = extract_arithmetic_expression(prompt)
    if expr is None:
        return None
    try:
        result = eval(expr, {"__builtins__": None}, {})  # noqa: S307
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer=str(result),
            route="MATH_PYTHON",
            tokens=0,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used="python-eval",
        )
    except ZeroDivisionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer="Error: Division by zero.",
            route="MATH_PYTHON",
            tokens=0,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used="python-eval",
        )
    except Exception:
        return None


_HEURISTIC_SPLIT = re.compile(
    r",\s*(?=(?:what|who|where|when|why|how|tell|capital|population|math:))",
    re.IGNORECASE,
)

SUMMARY_SYNTHESIS_PATTERNS = (
    "summarize",
    "summarise",
    "summary",
    "tl;dr",
    "tldr",
    "synthesis",
    "synthesize",
    "synthesise",
    "explain this",
    "overview of",
    "key points",
    "main points",
    "in brief",
    "condense",
    "recap",
    "digest",
    "what does this document",
    "what does this text",
)

LOOP_TASK_PATTERNS = (
    "each section",
    "each paragraph",
    "step by step through",
    "iterate over",
    "go through each",
)

PlannerMode = Literal["DIRECT", "LOOP", "SPLIT"]


@dataclass
class PlannerDecision:
    """Single-agent-first planner output — one read of the full prompt."""

    mode: PlannerMode
    num_agents: int
    tasks: list[str]
    reason: str
    preserve_original: bool
    confidence: float
    split_approved: bool = False


def is_synthesis_or_summary_task(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in SUMMARY_SYNTHESIS_PATTERNS)


def is_long_context_prompt(prompt: str) -> bool:
    stripped = prompt.strip()
    if len(stripped) > LONG_PROMPT_CHARS:
        return True
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if len(lines) >= LONG_CONTEXT_LINE_THRESHOLD:
        return True
    if "[Attached context from file]" in prompt:
        return True
    return False


def is_iterative_long_task(prompt: str) -> bool:
    lowered = prompt.lower()
    return is_long_context_prompt(prompt) and any(p in lowered for p in LOOP_TASK_PATTERNS)


def _context_preserving_task(original_prompt: str, subtask: str) -> str:
    """Wrap a sub-task with the full original question so agents never lose context."""
    return (
        "[Original user request — preserve this context]\n"
        f"{original_prompt.strip()}\n\n"
        "[Sub-task to answer now]\n"
        f"{subtask.strip()}"
    )


_TASK_START_KEYWORDS = frozenset(
    {
        "what", "who", "where", "when", "why", "how", "tell", "spell", "write",
        "list", "give", "find", "name", "capital", "population", "explain",
        "count", "reverse", "calculate", "summarize", "describe", "compute",
    }
)


def _starts_with_task_keyword(segment: str) -> bool:
    tokens = segment.strip().split()
    if not tokens:
        return False
    return tokens[0].lower().strip(",.?!:;'\"") in _TASK_START_KEYWORDS


def split_independent_tasks(prompt: str) -> list[str]:
    """
    Conservative splitter — only for short, explicitly multi-query prompts.
    Never splits long pasted documents, summarization context, or math notation.
    """
    cleaned = prompt.strip()
    if not cleaned:
        return [cleaned]
    if (
        is_long_context_prompt(cleaned)
        or is_synthesis_or_summary_task(cleaned)
        or is_symbolic_math(cleaned)
    ):
        return [cleaned]
    if len(cleaned) > 200:
        return [cleaned]

    parts = [part.strip() for part in _HEURISTIC_SPLIT.split(cleaned) if part.strip()]
    if len(parts) >= 2:
        parts = merge_short_fragments(parts)
        if len(parts) >= 2 and all(is_valid_subtask(p) for p in parts):
            return parts[:HARD_MAX_SUB_AGENTS]

    if cleaned.count(",") >= 2:
        comma_parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if 2 <= len(comma_parts) <= HARD_MAX_SUB_AGENTS:
            capital_match = re.match(
                r"^(tell me the capital of)\s+(.+)$",
                comma_parts[0],
                re.IGNORECASE,
            )
            if capital_match:
                prefix = capital_match.group(1)
                expanded = [comma_parts[0]]
                for tail in comma_parts[1:]:
                    if is_local_arithmetic(tail):
                        expanded.append(tail)
                    else:
                        expanded.append(f"{prefix} {tail}")
                expanded = merge_short_fragments(expanded)
                if len(expanded) >= 2 and all(is_valid_subtask(p) for p in expanded):
                    return expanded[:HARD_MAX_SUB_AGENTS]
            query_like = sum(
                1
                for part in comma_parts
                if _starts_with_task_keyword(part) or is_local_arithmetic(part)
            )
            if query_like >= 2 and query_like == len(comma_parts):
                valid = merge_short_fragments(comma_parts)
                if len(valid) >= 2 and all(is_valid_subtask(p) for p in valid):
                    return valid[:HARD_MAX_SUB_AGENTS]

    # Sentence / numbered-list boundaries only.
    sentence_parts = [
        seg.strip()
        for seg in re.split(r"(?<=[?.!])\s+(?=[A-Z0-9\"'])", cleaned)
        if seg.strip()
    ]
    if 2 <= len(sentence_parts) <= HARD_MAX_SUB_AGENTS:
        sentence_parts = merge_short_fragments(sentence_parts)
        if len(sentence_parts) >= 2 and all(is_valid_subtask(p) for p in sentence_parts):
            return sentence_parts[:HARD_MAX_SUB_AGENTS]

    enum_parts = [
        seg.strip()
        for seg in re.split(r"\s*(?:\d+[.)]\s+)", cleaned)
        if seg.strip()
    ]
    if 2 <= len(enum_parts) <= HARD_MAX_SUB_AGENTS:
        enum_parts = merge_short_fragments(enum_parts)
        if len(enum_parts) >= 2 and all(is_valid_subtask(p) for p in enum_parts):
            return enum_parts[:HARD_MAX_SUB_AGENTS]

    return [cleaned]


def heuristic_task_split(prompt: str) -> list[str]:
    """Backward-compatible alias — delegates to the conservative splitter."""
    return split_independent_tasks(prompt)


def should_decompose(prompt: str) -> bool:
    """
    True only when subtasks are clearly independent, short, and self-contained.
    Fail closed to single-agent for summaries, long context, and uncertain cases.
    """
    if is_synthesis_or_summary_task(prompt):
        return False
    if is_long_context_prompt(prompt):
        return False
    if is_direct_answer_prompt(prompt):
        return False
    if is_character_level_task(prompt):
        return False
    if is_symbolic_math(prompt):
        return False
    if len(prompt.strip()) > 200:
        return False

    parts = split_independent_tasks(prompt)
    if len(parts) < 2 or len(parts) > HARD_MAX_SUB_AGENTS:
        return False

    substantial = [
        part
        for part in parts
        if is_valid_subtask(part) and not is_local_trivial_whitelisted(part)
    ]
    if len(substantial) < 2:
        return False

    independent = sum(
        1
        for part in substantial
        if _starts_with_task_keyword(part) or is_local_arithmetic(part)
    )
    return independent >= 2


def decide_mode(prompt: str) -> PlannerDecision:
    """
    Pre-classifier reader: inspect the full prompt once and choose DIRECT, LOOP, or SPLIT.
    Fail closed to single-agent when uncertain.
    """
    cleaned = prompt.strip()
    if not cleaned:
        return PlannerDecision("DIRECT", 1, [cleaned], "empty", True, 1.0)

    if is_synthesis_or_summary_task(cleaned):
        return PlannerDecision(
            "DIRECT", 1, [cleaned], "summary:single_agent", True, 0.98,
        )

    if is_iterative_long_task(cleaned):
        return PlannerDecision(
            "LOOP", 1, [cleaned], "long_context:iterative_single_agent", True, 0.85,
        )

    if is_long_context_prompt(cleaned):
        return PlannerDecision(
            "DIRECT", 1, [cleaned], "long_context:preserve", True, 0.90,
        )

    if is_direct_answer_prompt(cleaned) or is_character_level_task(cleaned):
        return PlannerDecision(
            "DIRECT", 1, [cleaned], "direct_answer", True, 0.95,
        )

    if should_decompose(cleaned):
        raw_parts = split_independent_tasks(cleaned)[:HARD_MAX_SUB_AGENTS]
        if len(raw_parts) >= 2:
            wrapped = [_context_preserving_task(cleaned, part) for part in raw_parts]
            return PlannerDecision(
                "SPLIT",
                len(wrapped),
                wrapped,
                "split:explicit_independent_tasks",
                True,
                0.82,
                split_approved=True,
            )

    return PlannerDecision(
        "DIRECT", 1, [cleaned], "uncertain:single_agent", True, 0.75,
    )


SINGLE_TURN_MAX_CHARS = 120
PromptType = Literal["DIRECT_ANSWER", "LOCAL_DECOMPOSE", "REMOTE_ESCALATE"]


def is_direct_answer_prompt(prompt: str) -> bool:
    """
    Greetings, small talk, one-sentence questions, and one-step responses.
    These must never be decomposed into a swarm.
    """
    stripped = prompt.strip()
    if not stripped:
        return True
    parts = split_independent_tasks(stripped)
    if len(parts) >= 2:
        independent = sum(
            1
            for part in parts
            if _starts_with_task_keyword(part) or would_math_intercept(part)
        )
        if independent >= 2:
            return False
    if is_local_trivial_whitelisted(stripped):
        return True
    if is_simple_format_task(stripped):
        return True
    if would_math_intercept(stripped):
        return True
    if is_character_level_task(stripped):
        return False
    # Single-turn: at most one question mark, bounded length, not a comma-list.
    if stripped.count("?") <= 1 and len(stripped) <= SINGLE_TURN_MAX_CHARS:
        if stripped.count(",") < 2:
            return True
    return False


def is_beneficial_to_decompose(prompt: str) -> bool:
    """Backward-compatible alias for the conservative decomposition gate."""
    return should_decompose(prompt)


@dataclass
class ClassificationResult:
    """Level-1 cheap classifier output — runs on the full prompt before any split."""

    prompt_type: PromptType
    prompt_label: str
    route: Literal["LOCAL", "REMOTE"]
    reason: str
    confidence_score: float
    escalation_reason: str | None = None
    decomposition_used: bool = False
    num_agents: int = 1
    planner_mode: PlannerMode = "DIRECT"
    planner_reason: str = ""
    original_prompt_intact: bool = True
    split_approved: bool = False
    planned_tasks: list[str] = field(default_factory=list)


def classify_prompt(
    prompt: str,
    threshold: int,
    active_local_model: str,
    active_remote_model: str,
    *,
    has_image: bool = False,
) -> ClassificationResult:
    """
    Pre-classifier reader: reads the full prompt once via decide_mode(), then maps
    to execution types. SPLIT is only used when the planner explicitly approves.
    """
    planner = decide_mode(prompt)
    local_unavailable = not has_image and is_local_unavailable(active_local_model)
    decision = route_decision(
        prompt,
        threshold,
        has_image=has_image,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
        local_unavailable=local_unavailable,
        memory_pressure=get_memory_usage() if not has_image else None,
    )

    base_fields = {
        "planner_mode": planner.mode,
        "planner_reason": planner.reason,
        "original_prompt_intact": planner.preserve_original,
        "split_approved": planner.split_approved,
    }

    if has_image:
        return ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="image",
            route="REMOTE",
            reason="vision:remote",
            confidence_score=0.95,
            escalation_reason="vision:remote",
            num_agents=1,
            planned_tasks=[prompt],
            **base_fields,
        )

    if is_character_level_task(prompt):
        return ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="character_level",
            route="REMOTE",
            reason=CHARACTER_LEVEL_GUARD_REASON,
            confidence_score=0.99,
            escalation_reason=CHARACTER_LEVEL_GUARD_REASON,
            num_agents=1,
            planned_tasks=[prompt],
            **base_fields,
        )

    if had_prior_local_failure(prompt):
        return ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="prior_failure",
            route="REMOTE",
            reason=PRIOR_FAILURE_REASON,
            confidence_score=0.85,
            escalation_reason=PRIOR_FAILURE_REASON,
            num_agents=1,
            planned_tasks=[prompt],
            **base_fields,
        )

    if planner.mode == "SPLIT" and planner.split_approved:
        if any(is_character_level_task(part) for part in planner.tasks):
            return ClassificationResult(
                prompt_type="REMOTE_ESCALATE",
                prompt_label="character_level_subtask",
                route="REMOTE",
                reason=CHARACTER_LEVEL_GUARD_REASON,
                confidence_score=0.99,
                escalation_reason=CHARACTER_LEVEL_GUARD_REASON,
                num_agents=1,
                planned_tasks=[prompt],
                **base_fields,
            )
        return ClassificationResult(
            prompt_type="LOCAL_DECOMPOSE",
            prompt_label="multi_task",
            route=decision.route,
            reason=planner.reason,
            confidence_score=planner.confidence,
            decomposition_used=True,
            num_agents=min(planner.num_agents, HARD_MAX_SUB_AGENTS),
            planned_tasks=planner.tasks,
            **base_fields,
        )

    if planner.mode == "LOOP":
        ptype_loop: PromptType = (
            "REMOTE_ESCALATE" if decision.route == "REMOTE" else "DIRECT_ANSWER"
        )
        return ClassificationResult(
            prompt_type=ptype_loop,
            prompt_label="loop",
            route=decision.route,
            reason=planner.reason,
            confidence_score=planner.confidence,
            escalation_reason=decision.reason if decision.route == "REMOTE" else None,
            decomposition_used=False,
            num_agents=1,
            planned_tasks=[prompt],
            **base_fields,
        )

    if is_direct_answer_prompt(prompt):
        if is_local_trivial_whitelisted(prompt):
            label = "greeting"
        elif would_math_intercept(prompt):
            label = "math"
        else:
            label = "single_turn"
        ptype: PromptType = (
            "REMOTE_ESCALATE" if decision.route == "REMOTE" else "DIRECT_ANSWER"
        )
        return ClassificationResult(
            prompt_type=ptype,
            prompt_label=label,
            route=decision.route,
            reason=decision.reason,
            confidence_score=decision.confidence_score,
            escalation_reason=decision.reason if decision.route == "REMOTE" else None,
            decomposition_used=False,
            num_agents=1,
            planned_tasks=[prompt],
            **base_fields,
        )

    if decision.route == "REMOTE":
        return ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="cloud_escalation",
            route="REMOTE",
            reason=decision.reason,
            confidence_score=decision.confidence_score,
            escalation_reason=decision.reason,
            num_agents=1,
            planned_tasks=[prompt],
            **base_fields,
        )

    return ClassificationResult(
        prompt_type="DIRECT_ANSWER",
        prompt_label="single_route",
        route=decision.route,
        reason=planner.reason,
        confidence_score=planner.confidence,
        escalation_reason=None,
        decomposition_used=False,
        num_agents=1,
        planned_tasks=[prompt],
        **base_fields,
    )


def _log_orchestration(
    classification: ClassificationResult,
    prompt: str,
    *,
    latency_ms: float = 0.0,
) -> None:
    preview = prompt.strip().replace("\n", " ")[:60]
    logger.info(
        "ORCHESTRATION route_decision=%s prompt_type=%s prompt_label=%s "
        "planner_mode=%s planner_reason=%s decomposition_used=%s num_agents=%s "
        "original_prompt_intact=%s escalation_reason=%s confidence=%.2f latency_ms=%.1f prompt=%r",
        classification.route,
        classification.prompt_type,
        classification.prompt_label,
        classification.planner_mode,
        classification.planner_reason,
        classification.decomposition_used,
        classification.num_agents,
        classification.original_prompt_intact,
        classification.escalation_reason,
        classification.confidence_score,
        latency_ms,
        preview,
    )


def _attach_orchestration(
    result: RouteResult, classification: ClassificationResult
) -> RouteResult:
    result.prompt_type = classification.prompt_type
    result.decomposition_used = classification.decomposition_used
    result.num_agents = classification.num_agents
    result.escalation_reason = classification.escalation_reason
    result.confidence_score = classification.confidence_score
    result.diagnostics["orchestration"] = {
        "prompt_type": classification.prompt_type,
        "prompt_label": classification.prompt_label,
        "planner_mode": classification.planner_mode,
        "planner_reason": classification.planner_reason,
        "decomposition_used": classification.decomposition_used,
        "num_agents": classification.num_agents,
        "original_prompt_intact": classification.original_prompt_intact,
        "split_approved": classification.split_approved,
        "escalation_reason": classification.escalation_reason,
    }
    result.diagnostics["planner"] = {
        "mode": classification.planner_mode,
        "num_agents": classification.num_agents,
        "reason": classification.planner_reason,
        "original_prompt_intact": classification.original_prompt_intact,
        "split_approved": classification.split_approved,
    }
    return result


_TASK_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_TASK_ENUM_WORDS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "then",
    "also",
    "finally",
    "next",
    "lastly",
)


def count_tasks(text: str) -> int:
    """
    Count distinct user intents for distillation safety checks.

    Long documents and summarization requests are always one task. Short prompts
    may count bullets, enumerators, or explicit independent queries — never raw
    line counts on pasted documents.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return 0

    if is_synthesis_or_summary_task(cleaned) or is_long_context_prompt(cleaned):
        return 1

    if is_symbolic_math(cleaned):
        return 1

    lowered = cleaned.lower()
    candidates = [1]

    bullets = len(_TASK_BULLET_PATTERN.findall(cleaned))
    if bullets >= 2:
        candidates.append(bullets)

    if "?" in cleaned:
        qcount = len([seg for seg in cleaned.split("?") if seg.strip()])
        if qcount >= 2:
            candidates.append(qcount)

    enum_count = sum(1 for w in _TASK_ENUM_WORDS if re.search(rf"\b{w}\b", lowered))
    if enum_count >= 2:
        candidates.append(enum_count)

    if should_decompose(cleaned):
        candidates.append(len(split_independent_tasks(cleaned)))

    parts = split_independent_tasks(cleaned)
    if len(parts) >= 2:
        independent = sum(
            1
            for part in parts
            if _starts_with_task_keyword(part) or would_math_intercept(part)
        )
        if independent >= 2:
            candidates.append(len(parts))

    if len(cleaned) <= 200 and not is_symbolic_math(cleaned):
        lines = len([line for line in cleaned.splitlines() if line.strip()])
        if 2 <= lines <= HARD_MAX_SUB_AGENTS:
            candidates.append(lines)

    best = max(candidates)
    return min(best, HARD_MAX_SUB_AGENTS) if best >= 2 else 1


def adjust_prompt_for_remote(
    user_text: str,
    active_local_model: str,
    *,
    preserve_exact: bool = False,
) -> tuple[str, int, str | None, str]:
    """
    Middle-layer prompt adjustment before remote inference.
    Normalizes whitespace; optionally compresses via local distill when safe.
    """
    cleaned = user_text.strip()
    single_line = re.sub(r"\s+", " ", cleaned)
    if preserve_exact or not cleaned:
        return single_line, 0, None, "preserve"
    if len(cleaned) < DISTILL_MIN_CHARS:
        return single_line, 0, None, "normalize"
    if is_local_unavailable(active_local_model):
        return single_line, 0, None, "normalize-no-local"
    distilled, tokens, err = distill_prompt(cleaned, active_local_model)
    method = "distill" if distilled.strip() != cleaned else "normalize"
    return distilled, tokens, err, method


def distill_prompt(user_text: str, active_local_model: str) -> tuple[str, int, str | None]:
    """
    Structure-preserving compression via local Ollama before remote inference.

    The distiller shortens wording but must preserve every task. If the distilled
    output drops any task (distilled_task_count < original_task_count) or the
    prompt is already short, the ORIGINAL prompt is returned unchanged.
    Returns (text, local_tokens_used, error_message).
    """
    cleaned = user_text.strip()
    if not cleaned:
        return cleaned, 0, None

    # Skip compression for short prompts — nothing meaningful to save.
    if len(cleaned) < DISTILL_MIN_CHARS:
        return cleaned, 0, None

    # Skip local distillation entirely if the local backend is unsafe/heavy — it
    # would freeze exactly like the inference call we are trying to avoid.
    if is_heavy_local_model(active_local_model) and (
        is_local_unavailable(active_local_model) or is_memory_constrained()
    ):
        return cleaned, 0, "Local backend unsafe for distillation; using original prompt."

    original_task_count = count_tasks(cleaned)

    try:
        distilled, tokens = _ollama_generate(
            prompt=cleaned,
            model=active_local_model,
            system=DISTILL_SYSTEM_PROMPT,
            timeout=LOCAL_DECOMP_TIMEOUT_SECONDS,
            options={"temperature": 0.0},
        )
        distilled = distilled.strip()
        if not distilled:
            return cleaned, tokens, "Distillation returned empty output; using original prompt."

        # Safety fallback: never let compression drop a task.
        distilled_task_count = count_tasks(distilled)
        if distilled_task_count < original_task_count:
            logger.warning(
                "Distillation dropped tasks (%s -> %s); using original prompt.",
                original_task_count,
                distilled_task_count,
            )
            return (
                cleaned,
                tokens,
                f"Distillation dropped tasks ({original_task_count} → "
                f"{distilled_task_count}); using original prompt.",
            )
        return distilled, tokens, None
    except requests.ConnectionError:
        return cleaned, 0, "Ollama unavailable for distillation; using original prompt."
    except requests.Timeout:
        return cleaned, 0, "Distillation timed out; using original prompt."
    except requests.RequestException as exc:
        return cleaned, 0, f"Distillation failed: {exc}"


def _parse_question_array(raw: str) -> list[str]:
    """Parse a JSON string array from Ollama output."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty decomposition response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise ValueError("No JSON array found in decomposition response") from None
        parsed = json.loads(match.group())

    if not isinstance(parsed, list):
        raise ValueError("Decomposition output is not a JSON array")

    questions = [str(item).strip() for item in parsed if str(item).strip()]
    if not questions:
        raise ValueError("Decomposition array is empty")
    return questions


def task_dispatcher(
    prompt: str,
    active_local_model: str = "",
    *,
    allow_heavy_local: bool = False,
) -> list[str]:
    """
    Router-first decomposition: heuristic split only — never calls a local LLM
    planner. Keeps the middle layer responsive and remote-oriented.
    """
    _ = active_local_model, allow_heavy_local  # signature kept for callers
    return heuristic_task_split(prompt)


@dataclass
class SwarmPlan:
    """Pre-swarm plan from the hierarchical classifier + planner."""

    tasks: list[str]
    global_route: Literal["LOCAL", "REMOTE"]
    reason: str
    single_route: bool
    classification: ClassificationResult

    @property
    def single_remote(self) -> bool:
        """Backward-compatible alias for single-route plans."""
        return self.single_route


def plan_request(
    prompt: str,
    threshold: int,
    active_local_model: str,
    active_remote_model: str,
    *,
    allow_heavy_local: bool = False,
) -> SwarmPlan:
    """
    Hierarchical orchestration planner.

    Level-1: classify the WHOLE prompt (no split yet).
    Level-2: decompose only when LOCAL_DECOMPOSE is beneficial.
    Level-3: per-task execution handles cloud escalation.
    """
    _ = allow_heavy_local
    started = time.perf_counter()
    classification = classify_prompt(
        prompt,
        threshold,
        active_local_model,
        active_remote_model,
    )
    _log_orchestration(
        classification,
        prompt,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )

    # Single-agent default — SPLIT only when planner explicitly approved decomposition.
    if (
        classification.planner_mode == "SPLIT"
        and classification.split_approved
        and classification.decomposition_used
        and classification.num_agents > DEFAULT_MAX_SUB_AGENTS
    ):
        tasks = classification.planned_tasks or [prompt]
        if len(tasks) > 1:
            return SwarmPlan(
                tasks=tasks[:HARD_MAX_SUB_AGENTS],
                global_route=classification.route,
                reason=classification.reason,
                single_route=False,
                classification=classification,
            )

    return SwarmPlan(
        tasks=[prompt],
        global_route=classification.route,
        reason=classification.reason,
        single_route=True,
        classification=classification,
    )


def dispatch_instant_trivial(
    prompt: str,
    threshold: int,
    active_local_model: str,
    active_remote_model: str,
    *,
    timing: RequestTiming | None = None,
) -> RouteResult:
    """
    Fastest trivial path: math eval or canned greeting — no cache, PHANTOM, verify, or Ollama.
    """
    if would_math_intercept(prompt):
        tracker = timing or RequestTiming()
        tracker.mark("input_received")
        tracker.mark("router_start")
        score = min(100, calculate_complexity(prompt))
        tracker.mark("complexity_computed", complexity_score=score)
        decision = route_decision(
            prompt,
            threshold,
            has_image=False,
            active_local_model=active_local_model,
            active_remote_model=active_remote_model,
            local_unavailable=False,
            has_txt_context=False,
        )
        tracker.mark(
            "route_decision",
            route=decision.route,
            reason=decision.reason,
            complexity_score=decision.complexity_score,
        )
        tracker.mark("model_call_start")
        math_result = safe_math_agent(prompt, tracker.origin)
        tracker.mark("model_call_end")
        tracker.mark("final_response")
        if math_result is not None:
            math_result.routing_reason = LOCAL_MATH_REASON
            math_result.complexity_score = decision.complexity_score
            math_result.confidence_score = decision.confidence_score
            math_result.diagnostics.update({
                "instant_trivial": True,
                "skipped_cache": True,
                "skipped_phantom": True,
                "skipped_verify": True,
            })
            tracker.attach(math_result)
            tracker.log_summary(prompt)
            return math_result
    return dispatch_instant_greeting(
        prompt,
        threshold,
        active_local_model,
        active_remote_model,
        timing=timing,
    )


def dispatch_instant_greeting(
    prompt: str,
    threshold: int,
    active_local_model: str,
    active_remote_model: str,
    *,
    timing: RequestTiming | None = None,
) -> RouteResult:
    """
    Fastest greeting path: router decision only, canned reply, no health probe,
    no FAISS, no PHANTOM, no Ollama round-trip.
    """
    tracker = timing or RequestTiming()
    tracker.mark("input_received")
    tracker.mark("router_start")
    score = min(100, calculate_complexity(prompt))
    tracker.mark("complexity_computed", complexity_score=score)
    decision = route_decision(
        prompt,
        threshold,
        has_image=False,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
        local_unavailable=False,
        has_txt_context=False,
    )
    tracker.mark(
        "route_decision",
        route=decision.route,
        reason=decision.reason,
        complexity_score=decision.complexity_score,
    )
    token_budget = compute_remote_max_tokens(prompt, decision)
    started = tracker.origin

    if would_math_intercept(prompt):
        tracker.mark("model_call_start")
        math_result = safe_math_agent(prompt, started)
        tracker.mark("model_call_end")
        tracker.mark("final_response")
        if math_result is not None:
            math_result.routing_reason = LOCAL_MATH_REASON
            math_result.complexity_score = decision.complexity_score
            math_result.confidence_score = decision.confidence_score
            tracker.attach(math_result)
            tracker.log_summary(prompt)
            return math_result

    tracker.mark("model_call_start")
    answer = get_canned_greeting_reply(prompt)
    tracker.mark("first_token")
    tracker.mark("model_call_end")
    tracker.mark("final_response")
    result = RouteResult(
        answer=answer,
        route="TEXT_LOCAL",
        tokens=0,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        original_prompt=prompt,
        model_used="canned-reply",
        routing_reason=decision.reason if decision.reason != CANNED_REPLY_REASON else CANNED_REPLY_REASON,
        complexity_score=decision.complexity_score,
        confidence_score=decision.confidence_score,
        diagnostics={
            "token_budget": token_budget,
            "instant_greeting": True,
            "skipped_health_probe": True,
            "skipped_phantom": True,
            "skipped_cache": True,
        },
    )
    _log_dispatch_telemetry(
        prompt, decision, result, token_budget=token_budget,
        file_attached=False, dispatcher_route=result.route,
    )
    tracker.attach(result)
    tracker.log_summary(prompt)
    return result


def route_and_execute(
    prompt: str,
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    image_file: BinaryIO | None = None,
    *,
    allow_heavy_local: bool = False,
    has_txt_context: bool = False,
    timing: RequestTiming | None = None,
    fast_greeting: bool = False,
) -> RouteResult:
    """
    Dispatcher: execute exactly what route_decision() authorizes.
    """
    started = time.perf_counter()
    has_image = image_file is not None
    pure_greeting = is_pure_greeting_request(
        prompt, has_image=has_image, has_txt_context=has_txt_context
    )

    if timing:
        timing.mark("router_start")

    if pure_greeting and fast_greeting:
        return dispatch_instant_greeting(
            prompt,
            threshold,
            active_local_model,
            active_remote_model,
            timing=timing,
        )

    if pure_greeting:
        local_healthy = True
        local_unavailable = False
    else:
        local_healthy = not has_image and check_local_health(active_local_model)
        local_unavailable = not has_image and not local_healthy
    memory_pressure = get_memory_usage() if not has_image else None

    if timing:
        timing.mark("complexity_computed", complexity_score=min(100, calculate_complexity(prompt)))

    decision = route_decision(
        prompt,
        threshold,
        has_image=has_image,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
        local_unavailable=local_unavailable,
        memory_pressure=memory_pressure,
        allow_heavy_local=allow_heavy_local,
        has_txt_context=has_txt_context,
    )
    if timing:
        timing.mark(
            "route_decision",
            route=decision.route,
            reason=decision.reason,
            complexity_score=decision.complexity_score,
        )
    token_budget = compute_remote_max_tokens(prompt, decision)
    _log_routing(prompt, decision, local_healthy=local_healthy or has_image)

    if decision.reason == ENTROPY_GATE_REASON:
        result = RouteResult(
            answer=(
                "I couldn't understand that input — it looks like random or garbled text. "
                "Please rephrase your question clearly."
            ),
            route="TEXT_LOCAL",
            tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            original_prompt=prompt,
            model_used="entropy-gate",
            routing_reason=ENTROPY_GATE_REASON,
            complexity_score=decision.complexity_score,
            confidence_score=decision.confidence_score,
            diagnostics={
                "token_budget": token_budget,
                "entropy_gated": True,
                "input_entropy": compute_shannon_entropy(prompt.strip()),
            },
        )
        _log_dispatch_telemetry(
            prompt, decision, result, token_budget=token_budget,
            file_attached=has_txt_context, dispatcher_route=result.route,
        )
        if timing:
            timing.mark("final_response")
            timing.attach(result)
        return result

    if decision.reason == CANNED_REPLY_REASON:
        result = RouteResult(
            answer=get_canned_greeting_reply(prompt),
            route="TEXT_LOCAL",
            tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            original_prompt=prompt,
            model_used="canned-reply",
            routing_reason=CANNED_REPLY_REASON,
            complexity_score=decision.complexity_score,
            confidence_score=decision.confidence_score,
            diagnostics={"token_budget": token_budget, "file_attached": has_txt_context},
        )
        _log_dispatch_telemetry(
            prompt, decision, result, token_budget=token_budget,
            file_attached=has_txt_context, dispatcher_route=result.route,
        )
        if timing:
            timing.mark("final_response")
            timing.attach(result)
        return result

    if decision.reason == LOCAL_GREETING_REASON and pure_greeting:
        if timing:
            timing.mark("model_call_start")
        answer = get_canned_greeting_reply(prompt)
        if timing:
            timing.mark("first_token")
            timing.mark("model_call_end")
        result = RouteResult(
            answer=answer,
            route="TEXT_LOCAL",
            tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            original_prompt=prompt,
            model_used="canned-reply",
            routing_reason=LOCAL_GREETING_REASON,
            complexity_score=decision.complexity_score,
            confidence_score=decision.confidence_score,
            diagnostics={
                "token_budget": token_budget,
                "instant_greeting": True,
                "skipped_local_ollama": True,
            },
        )
        _log_dispatch_telemetry(
            prompt, decision, result, token_budget=token_budget,
            file_attached=has_txt_context, dispatcher_route=result.route,
        )
        if timing:
            timing.mark("final_response")
            timing.attach(result)
        return result

    if has_image:
        answer, route, tokens, latency_ms = _route_vision(prompt, api_key, image_file, started)
        result = RouteResult(
            answer=answer,
            route=route,
            tokens=tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used=REMOTE_VISION_MODEL,
            routing_reason=decision.reason,
            complexity_score=decision.complexity_score,
            diagnostics={"token_budget": token_budget, "file_attached": has_txt_context},
        )
        _log_dispatch_telemetry(
            prompt, decision, result, token_budget=token_budget,
            file_attached=has_txt_context, dispatcher_route=result.route,
        )
        return result

    if decision.route == "REMOTE":
        preserve_text = (
            decision.reason == CHARACTER_LEVEL_GUARD_REASON
            or decision.reason in LOCAL_DISTILL_UNSAFE_REASONS
        )
        if timing:
            timing.mark("model_call_start")
        result = _route_text_remote(
            prompt,
            api_key,
            active_local_model,
            active_remote_model,
            started,
            skip_distillation=preserve_text,
            max_tokens=token_budget,
        )
        if timing and result.answer:
            timing.mark("first_token")
        if timing:
            timing.mark("model_call_end")
        result.routing_reason = decision.reason
        result.complexity_score = decision.complexity_score
        result.confidence_score = decision.confidence_score
        result.diagnostics.setdefault("token_budget", token_budget)
        _log_dispatch_telemetry(
            prompt, decision, result, token_budget=token_budget,
            file_attached=has_txt_context, dispatcher_route=result.route,
        )
        if timing:
            timing.mark("final_response")
            timing.attach(result)
            timing.log_summary(prompt)
        return result

    math_result = safe_math_agent(prompt, started)
    if math_result is not None:
        if timing:
            timing.mark("model_call_start")
            timing.mark("first_token")
            timing.mark("model_call_end")
        math_result.routing_reason = LOCAL_MATH_REASON
        math_result.complexity_score = decision.complexity_score
        math_result.confidence_score = decision.confidence_score
        _log_dispatch_telemetry(
            prompt, decision, math_result, token_budget=token_budget,
            file_attached=has_txt_context, dispatcher_route=math_result.route,
        )
        if timing:
            timing.mark("final_response")
            timing.attach(math_result)
            timing.log_summary(prompt)
        return math_result

    if timing:
        timing.mark("model_call_start")
    result = _route_text_local(
        prompt, api_key, active_local_model, active_remote_model, started
    )
    if timing and result.answer:
        timing.mark("first_token")
    if timing:
        timing.mark("model_call_end")
    if result.fallback_used and is_greeting_or_tiny_chat(prompt) and not has_txt_context:
        result = RouteResult(
            answer=get_canned_greeting_reply(prompt),
            route="TEXT_LOCAL",
            tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            original_prompt=prompt,
            model_used="canned-reply",
            routing_reason=CANNED_REPLY_REASON,
            complexity_score=decision.complexity_score,
            confidence_score=decision.confidence_score,
            diagnostics={"token_budget": token_budget, "fallback_avoided_remote": True},
        )
    else:
        result.routing_reason = decision.reason
        result.complexity_score = decision.complexity_score
        result.confidence_score = decision.confidence_score
    _log_dispatch_telemetry(
        prompt, decision, result, token_budget=token_budget,
        file_attached=has_txt_context, dispatcher_route=result.route,
    )
    if timing:
        timing.mark("final_response")
        timing.attach(result)
        timing.log_summary(prompt)
    return result


# Limit concurrent Fireworks calls during agent swarms.
_REMOTE_CALL_SEMAPHORE = threading.BoundedSemaphore(SWARM_MAX_CONCURRENT)


def execute_agent_swarm(
    tasks: list[str],
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    *,
    allow_heavy_local: bool = False,
    original_prompt: str = "",
) -> RouteResult:
    """Run route_and_execute in parallel for each planner-approved sub-question."""
    swarm_started = time.perf_counter()
    ordered: list[RouteResult | None] = [None] * len(tasks)

    worker_count = max(1, min(len(tasks), SWARM_MAX_CONCURRENT))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                route_and_execute,
                task,
                threshold,
                api_key,
                active_local_model,
                active_remote_model,
                allow_heavy_local=allow_heavy_local,
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                ordered[index] = future.result()
            except Exception as exc:
                ordered[index] = RouteResult(
                    answer=f"⚠️ Sub-agent {index + 1} failed: {exc}",
                    route="TEXT_LOCAL",
                    tokens=0,
                    latency_ms=0.0,
                    original_prompt=tasks[index],
                )

    sub_results: list[RouteResult] = [
        result if result is not None
        else RouteResult(
            answer=f"⚠️ Sub-agent {i + 1} returned no result.",
            route="TEXT_LOCAL",
            tokens=0,
            latency_ms=0.0,
            original_prompt=tasks[i],
        )
        for i, result in enumerate(ordered)
    ]

    sections: list[str] = []
    for index, (task, result) in enumerate(zip(tasks, sub_results, strict=True), start=1):
        sections.append(f"### Sub-Agent {index}\n**Task:** {task}\n\n{result.answer}")

    total_tokens = sum(r.tokens for r in sub_results)
    wall_latency_ms = (time.perf_counter() - swarm_started) * 1000.0
    any_fallback = any(r.fallback_used for r in sub_results)

    return RouteResult(
        answer="\n\n---\n\n".join(sections),
        route="AGENT_SWARM",
        tokens=total_tokens,
        latency_ms=wall_latency_ms,
        original_prompt=original_prompt or " | ".join(tasks),
        fallback_used=any_fallback,
        sub_results=sub_results,
        wall_clock_ms=wall_latency_ms,
        diagnostics={
            "swarm_sub_agents": len(tasks),
            "swarm_tokens_total": total_tokens,
            "original_prompt_intact": bool(original_prompt),
            "swarm_max_concurrent": SWARM_MAX_CONCURRENT,
        },
    )


def _angkor_phantom_execute(
    prompt: str,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    started: float,
) -> RouteResult | None:
    """Try ANGKOR 3-zone + PHANTOM race. Returns None if not ready or not PHANTOM zone."""
    global _ANGKOR_SKLEARN_ROUTER, _ANGKOR_PHANTOM_RUNNER, _ANGKOR_ADAPTIVE_THETA
    router = _ANGKOR_SKLEARN_ROUTER
    if router is None or not router.is_ready:
        return None
    entropy_score = compute_shannon_entropy(prompt)
    angkor_result = router.route(prompt, entropy_score=entropy_score)
    z = angkor_result.zone
    if z == PhantomZone.PHANTOM_RACE:
        runner = _ANGKOR_PHANTOM_RUNNER
        if runner is None:
            return None
        fe = FeatureExtractor()
        features = fe.extract(prompt, entropy_score=entropy_score)
        L_out_norm = float(features[4])
        confidence = 1.0 - abs(angkor_result.complexity_score - router.theta) / 0.5

        token_budget = compute_remote_max_tokens(
            prompt,
            RouterDecision("REMOTE", 50, "phantom-race", active_remote_model, 0.8),
        )

        def _remote_fn(text: str, **kw: object) -> str | None:
            max_tok = int(kw.get("max_tokens", token_budget))
            result = _route_text_remote(
                text, api_key, active_local_model, active_remote_model,
                started, fallback=False, skip_distillation=False,
                max_tokens=max_tok,
            )
            return result.answer if result and result.answer else None

        source_name = "local" if angkor_result.destination == "local" else "remote"
        answer_text, winner, telemetry = runner.phantom_race(
            prompt=prompt,
            L_out_norm=L_out_norm,
            confidence=confidence,
            local_model=active_local_model,
            remote_call=_remote_fn,
        )
        route: RouteName = "PHANTOM_RACE"
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer=answer_text,
            route=route,
            tokens=0,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used=f"phantom:{winner}",
            routing_reason=angkor_result.reason,
            diagnostics={
                "phantom": telemetry,
                "angkor_complexity": angkor_result.complexity_score,
                "angkor_theta": angkor_result.theta,
            },
        )
    return None


def _cache_lookup(prompt: str) -> RouteResult | None:
    global _ANGKOR_CACHE
    cache = _lazy_init_angkor_cache()
    if cache is None:
        return None
    entry = cache.lookup(prompt)
    if entry is None:
        return None
    return RouteResult(
        answer=entry.response,
        route="CACHE_HIT",
        tokens=0,
        latency_ms=0.0,
        original_prompt=prompt,
        model_used="cache",
        routing_reason="cache-hit",
        metadata=entry.metadata,
    )


def _cache_store(prompt: str, result: RouteResult) -> None:
    global _ANGKOR_CACHE
    cache = _lazy_init_angkor_cache()
    if cache is None:
        return
    cache.store(prompt, result.answer, metadata={
        "route": result.route,
        "model_used": result.model_used,
        "tokens": result.tokens,
    })


def _verify_local_result(
    prompt: str,
    result: RouteResult,
    api_key: str,
    started: float,
) -> RouteResult:
    global _ANGKOR_VERIFIER
    # Trivial local paths never need cascade verify (avoids loading MiniLM + remote escalate).
    if result.route in ("MATH_PYTHON",) or result.diagnostics.get("instant_greeting") or result.diagnostics.get("instant_trivial"):
        return result
    if not api_key or not api_key.strip():
        return result
    verifier = _ANGKOR_VERIFIER
    if verifier is None:
        return result
    if result.route not in ("TEXT_LOCAL", "MATH_PYTHON", "PHANTOM_RACE"):
        return result

    def _escalate_fn(q: str) -> str:
        res = _route_text_remote(
            q, api_key, "", DEFAULT_REMOTE_MODEL, started,
            fallback=True, skip_distillation=True,
        )
        return res.answer if res and res.answer else ""

    task_type = "qa"
    if result.route == "MATH_PYTHON":
        task_type = "math"
    accepted, final_output, escalated = verifier.verify(
        prompt, result.answer, task_type=task_type, remote_escalate_fn=_escalate_fn,
    )
    if escalated:
        result.answer = final_output
        result.diagnostics["cascade_escalated"] = True
    if not accepted:
        result.answer = final_output or result.answer
    return result


def process_user_request(
    prompt: str,
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    image_file: BinaryIO | None = None,
    *,
    allow_heavy_local: bool = False,
    has_txt_context: bool = False,
    timing: RequestTiming | None = None,
) -> RouteResult:
    """Top-level orchestrator with ANGKOR + PHANTOM integration."""
    started = time.perf_counter()
    if timing:
        timing.mark("input_received")

    if image_file is not None:
        clf = classify_prompt(
            prompt, threshold, active_local_model, active_remote_model, has_image=True,
        )
        result = route_and_execute(
            prompt, threshold, api_key, active_local_model, active_remote_model,
            image_file=image_file, allow_heavy_local=allow_heavy_local,
            has_txt_context=has_txt_context, timing=timing,
        )
        return _attach_orchestration(result, clf)

    # Fastest path first — skip FAISS, PHANTOM, verify for greetings and math.
    if should_skip_expensive_preprocess(prompt, has_txt_context=has_txt_context):
        result = dispatch_instant_trivial(
            prompt,
            threshold,
            active_local_model,
            active_remote_model,
            timing=timing,
        )
        return result

    if not api_key or not api_key.strip():
        preview_decision = route_decision(
            prompt,
            threshold,
            has_image=False,
            active_local_model=active_local_model,
            active_remote_model=active_remote_model,
            local_unavailable=False,
            has_txt_context=has_txt_context,
        )
        if preview_decision.route == "REMOTE":
            return RouteResult(
                answer=(
                    "❌ **Fireworks API Key required.**\n\n"
                    "Enter your API key in the sidebar to route this prompt remotely."
                ),
                route="TEXT_REMOTE",
                tokens=0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                original_prompt=prompt,
                model_used=active_remote_model,
                routing_reason=preview_decision.reason,
                complexity_score=preview_decision.complexity_score,
                diagnostics={"fail_fast": True, "skipped_phantom": True, "skipped_cache": True},
            )

    if timing:
        timing.mark("cache_lookup_start")
    cache_hit = _cache_lookup(prompt)
    if timing:
        timing.mark("cache_lookup_end", cache_hit=cache_hit is not None)
    if cache_hit is not None:
        if timing:
            timing.mark("final_response")
            timing.attach(cache_hit)
            timing.log_summary(prompt)
        return cache_hit

    if timing:
        timing.mark("phantom_check_start")
    phantom = _angkor_phantom_execute(
        prompt, api_key, active_local_model, active_remote_model, started,
    )
    if timing:
        timing.mark("phantom_check_end", phantom_ran=phantom is not None)
    if phantom is not None:
        phantom = _verify_local_result(prompt, phantom, api_key, started)
        _cache_store(prompt, phantom)
        return phantom

    plan = plan_request(
        prompt, threshold, active_local_model, active_remote_model,
        allow_heavy_local=allow_heavy_local,
    )

    if plan.single_route or plan.classification.num_agents <= 1:
        result = route_and_execute(
            prompt, threshold, api_key, active_local_model, active_remote_model,
            allow_heavy_local=allow_heavy_local, has_txt_context=has_txt_context,
            timing=timing,
        )
        result = _verify_local_result(prompt, result, api_key, started)
        _cache_store(prompt, result)
        return _attach_orchestration(result, plan.classification)

    result = execute_agent_swarm(
        plan.tasks, threshold, api_key, active_local_model, active_remote_model,
        allow_heavy_local=allow_heavy_local,
        original_prompt=prompt,
    )
    result.decomposition_used = True
    result.num_agents = len(plan.tasks)
    result.prompt_type = plan.classification.prompt_type
    result.escalation_reason = plan.classification.escalation_reason
    result.diagnostics["orchestration"] = {
        "prompt_type": plan.classification.prompt_type,
        "prompt_label": plan.classification.prompt_label,
        "decomposition_used": True,
        "num_agents": len(plan.tasks),
        "escalation_reason": plan.classification.escalation_reason,
    }
    _cache_store(prompt, result)
    return result


def _route_vision(
    prompt: str,
    api_key: str,
    image_file: BinaryIO,
    started: float,
) -> tuple[str, RouteName, int, float]:
    if not api_key or not api_key.strip():
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "❌ **Fireworks API Key required.**\n\n"
            "Vision routing needs your API key in the sidebar."
        )
        return message, "VISION_REMOTE", 0, latency_ms

    try:
        image_file.seek(0)
        data_uri = compress_image_to_base64(image_file)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return f"⚠️ Image compression failed: {exc}", "VISION_REMOTE", 0, latency_ms

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    user_text = prompt.strip() or "Describe this image."
    payload = {
        "model": REMOTE_VISION_MODEL,
        "messages": [
            {"role": "system", "content": "Describe this image concisely. /no_think"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "max_tokens": 100,
    }

    try:
        response = requests.post(
            REMOTE_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        tokens = int(data.get("usage", {}).get("total_tokens", 0))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return answer, "VISION_REMOTE", tokens, latency_ms

    except requests.ConnectionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Could not reach Fireworks API.", "VISION_REMOTE", 0, latency_ms

    except requests.Timeout:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Vision request timed out.", "VISION_REMOTE", 0, latency_ms

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text[:400]
        return f"⚠️ Vision inference error:\n\n{detail}", "VISION_REMOTE", 0, latency_ms


def _route_text_local(
    prompt: str,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    started: float,
) -> RouteResult:
    """
    Attempt local Ollama inference exactly ONCE, wrapped in a strict timeout and a
    health gate, then on ANY failure (dead backend, timeout, non-200 status such as
    a 404 NOT_FOUND, or an empty body) automatically reroute to the remote Fireworks
    endpoint. A heavy local model is never retried for the same request.
    """
    timeout_s = _local_inference_timeout(active_local_model)
    timeout_ms = timeout_s * 1000
    fallback_reason: str | None = None
    error_type: str | None = None

    # Health gate: if the backend is already known-unhealthy, skip the call entirely
    # so we never block on a dead/overloaded local model.
    if is_local_unavailable(active_local_model):
        fallback_reason = "Local backend health check failed."
        error_type = "health_check_failed"
    else:
        payload = {
            "model": active_local_model,
            "prompt": prompt,
            "system": SUB_AGENT_SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        }
        try:
            # (connect, read) timeout: read bounds total generation wait for stream=False.
            response = requests.post(
                LOCAL_ENDPOINT,
                json=payload,
                timeout=(LOCAL_HEALTH_TIMEOUT_SECONDS, timeout_s),
            )
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"Ollama returned status {response.status_code}", response=response
                )
            data = response.json()
            answer = data.get("response", "").strip()
            if not answer:
                raise ValueError("Ollama returned an empty response")
            tokens = int(data.get("eval_count", 0))
            latency_ms = (time.perf_counter() - started) * 1000.0
            _log_local_attempt(
                model_name=active_local_model,
                backend="ollama",
                route="TEXT_LOCAL",
                timeout_ms=timeout_ms,
                fallback_used=False,
                error_type=None,
            )
            return RouteResult(
                answer=answer,
                route="TEXT_LOCAL",
                tokens=tokens,
                latency_ms=latency_ms,
                original_prompt=prompt,
                model_used=active_local_model,
                diagnostics={
                    "local": {
                        "model_name": active_local_model,
                        "timeout_ms": timeout_ms,
                        "memory_usage": get_memory_usage(),
                    }
                },
            )

        except requests.ConnectionError:
            fallback_reason = "Ollama is not running on localhost:11434."
            error_type = "connection_error"
        except requests.Timeout:
            # Strict timeout exceeded: stop waiting, mark backend unhealthy, go remote.
            fallback_reason = (
                f"Local model '{active_local_model}' exceeded {timeout_s}s timeout."
            )
            error_type = "timeout"
            with _LOCAL_HEALTH_LOCK:
                _LOCAL_HEALTH_CACHE[active_local_model.strip()] = (
                    time.time() + LOCAL_HEALTH_TTL_SECONDS,
                    False,
                )
        except (requests.RequestException, ValueError) as exc:
            fallback_reason = f"Local model '{active_local_model}' error: {exc}"
            error_type = "request_error"

    logger.warning(
        "LOCAL fallback model_name=%s error_type=%s reason=%s",
        active_local_model,
        error_type,
        fallback_reason,
    )
    mark_prior_local_failure(prompt)
    _log_local_attempt(
        model_name=active_local_model,
        backend="ollama",
        route="FALLBACK_REMOTE",
        timeout_ms=timeout_ms,
        fallback_used=True,
        error_type=error_type,
    )

    # Local-to-Remote fallback. Never fail the user; bind to the UI-selected model.
    if _has_ui_context():
        st.toast(f"⚠️ {fallback_reason} Rerouting to Fireworks (remote).", icon="↩️")

    fallback_result = _route_text_remote(
        prompt,
        api_key,
        active_local_model,
        active_remote_model,
        started,
        fallback=True,
        skip_distillation=True,
    )
    fallback_result.fallback_used = True
    fallback_result.fallback_reason = fallback_reason
    fallback_result.diagnostics.setdefault("local", {}).update(
        {
            "model_name": active_local_model,
            "timeout_ms": timeout_ms,
            "error_type": error_type,
            "memory_usage": get_memory_usage(),
        }
    )
    return fallback_result


def _route_text_remote(
    prompt: str,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    started: float,
    *,
    fallback: bool = False,
    skip_distillation: bool = False,
    max_tokens: int = REMOTE_MAX_TOKENS_DEFAULT,
) -> RouteResult:
    route_name: RouteName = "FALLBACK_REMOTE" if fallback else "TEXT_REMOTE"

    if not api_key or not api_key.strip():
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "❌ **Fireworks API Key required.**\n\n"
            "Enter your API key in the sidebar to route this prompt remotely."
        )
        return RouteResult(
            answer=message,
            route=route_name,
            tokens=0,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used=active_remote_model,
            fallback_used=fallback,
        )

    original_prompt = prompt
    distill_tokens = 0
    distill_error: str | None = None
    adjust_method = "preserve" if skip_distillation else "normalize"

    if skip_distillation:
        distilled = re.sub(r"\s+", " ", original_prompt.strip())
    else:
        distilled, distill_tokens, distill_error, adjust_method = adjust_prompt_for_remote(
            prompt, active_local_model, preserve_exact=False
        )
        if not distilled or distilled.strip() == "":
            distilled = original_prompt
            if _has_ui_context():
                st.warning("Prompt adjustment returned empty. Using original prompt.")

    chars_saved = max(0, len(original_prompt) - len(distilled))

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    # Priority-ordered remote models. The selected model is validated against the
    # registry first; an unknown/undeployed selection is never tried first.
    candidates = build_remote_candidates(active_remote_model)
    if not candidates:
        candidates = [normalize_model_id(active_remote_model) or active_remote_model]

    # Diagnostics: every model_id we attempt, in order, with its outcome.
    remote_attempts: list[dict[str, str]] = []
    rate_limit_hits = 0
    truncated_flag = False

    def _remote_result(
        answer: str,
        tokens: int,
        retries: int,
        model_used: str,
        *,
        extra_diag: dict[str, object] | None = None,
    ) -> RouteResult:
        cleaned_answer = strip_reasoning_traces(answer)
        diag: dict[str, object] = {
            "remote_attempts": remote_attempts,
            "prompt_adjustment": adjust_method,
            "rate_limit_hits": rate_limit_hits,
            "truncated": truncated_flag,
        }
        if extra_diag:
            diag.update(extra_diag)
        return RouteResult(
            answer=cleaned_answer,
            route=route_name,
            tokens=tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            original_prompt=original_prompt,
            model_used=model_used,
            distilled_prompt=None if skip_distillation else distilled,
            distillation_chars_saved=chars_saved,
            distillation_error=distill_error,
            fallback_used=fallback,
            retries=retries,
            diagnostics=diag,
        )

    # A REMOTE decision must be served remotely — we NEVER fall back to local here.
    max_attempts = 2
    last_detail = "unknown error"

    for model_id in candidates:
        current_max_tokens = max_tokens

        unavailable = False
        malformed_response = False
        for attempt in range(max_attempts):
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SUB_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": distilled},
                ],
                "max_tokens": current_max_tokens,
                "temperature": 0.0,
            }
            acquired = False
            try:
                acquired = _REMOTE_CALL_SEMAPHORE.acquire(timeout=180)
                if not acquired:
                    last_detail = "Remote concurrency limit timeout"
                    continue
                response = requests.post(
                    REMOTE_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=180,
                )
                if is_rate_limit_error(response.status_code, response.text):
                    rate_limit_hits += 1
                    last_detail = f"HTTP {response.status_code}: rate limited"
                    remote_attempts.append(
                        {
                            "model_id": model_id,
                            "status": "rate_limited",
                            "detail": response.text[:200],
                        }
                    )
                    for rate_retry in range(RATE_LIMIT_MAX_RETRIES):
                        backoff = RATE_LIMIT_BACKOFF_BASE * (2 ** rate_retry)
                        time.sleep(backoff)
                        response = requests.post(
                            REMOTE_ENDPOINT,
                            headers=headers,
                            json=payload,
                            timeout=180,
                        )
                        if not is_rate_limit_error(response.status_code, response.text):
                            break
                        rate_limit_hits += 1
                    if is_rate_limit_error(response.status_code, response.text):
                        last_detail = (
                            "Fireworks rate limit exceeded. "
                            "Please wait a moment and try again."
                        )
                        return _remote_result(
                            f"⚠️ {last_detail}",
                            distill_tokens,
                            attempt,
                            model_id,
                        )
                if response.status_code >= 500:
                    last_detail = f"HTTP {response.status_code} (transient)"
                    logger.warning(
                        "Remote 5xx for %s on attempt %s: %s",
                        model_id,
                        attempt + 1,
                        last_detail,
                    )
                    continue
                if _is_model_unavailable(response.status_code, response.text):
                    last_detail = f"HTTP {response.status_code}: {response.text[:200]}"
                    unavailable = True
                    break
                response.raise_for_status()
                try:
                    data = response.json()
                except ValueError as exc:
                    last_detail = f"Remote response was not valid JSON: {exc}"
                    malformed_response = True
                    logger.warning(
                        "Remote invalid JSON response for %s: %s",
                        model_id,
                        last_detail,
                    )
                    remote_attempts.append(
                        {
                            "model_id": model_id,
                            "status": "malformed_response",
                            "detail": last_detail,
                        }
                    )
                    break
                answer = _extract_remote_answer(data)
                if not answer:
                    last_detail = (
                        "Remote response missing assistant content: "
                        f"{str(data)[:400]}"
                    )
                    malformed_response = True
                    logger.warning(
                        "Remote malformed success response for %s: %s",
                        model_id,
                        last_detail,
                    )
                    remote_attempts.append(
                        {
                            "model_id": model_id,
                            "status": "malformed_response",
                            "detail": last_detail,
                        }
                    )
                    break
                remote_tokens = int(data.get("usage", {}).get("total_tokens", 0))
                if is_response_truncated(data):
                    truncated_flag = True
                    if current_max_tokens < REMOTE_MAX_TOKENS_CAP:
                        current_max_tokens = min(
                            current_max_tokens * 2, REMOTE_MAX_TOKENS_CAP
                        )
                        remote_attempts.append(
                            {
                                "model_id": model_id,
                                "status": "truncated_retry",
                                "detail": f"retry max_tokens={current_max_tokens}",
                            }
                        )
                        continue
                remote_attempts.append({"model_id": model_id, "status": "ok"})
                return _remote_result(
                    answer, remote_tokens + distill_tokens, attempt, model_id
                )

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_detail = str(exc)
                logger.warning(
                    "Remote transient error for %s on attempt %s: %s",
                    model_id,
                    attempt + 1,
                    last_detail,
                )
                continue

            except requests.RequestException as exc:
                detail = str(exc)
                status_code = exc.response.status_code if exc.response is not None else 0
                if exc.response is not None:
                    detail = exc.response.text[:400]
                last_detail = detail
                if _is_model_unavailable(status_code, detail):
                    unavailable = True
                    break
                # Other non-transient error (not model-related): record and stop;
                # advancing models won't help and we preserve the original payload.
                logger.error("Remote non-transient error (%s): %s", model_id, detail)
                remote_attempts.append(
                    {"model_id": model_id, "status": "error", "detail": detail}
                )
                return _remote_result(
                    "⚠️ Remote service temporarily unavailable. Please try again shortly.",
                    distill_tokens,
                    attempt,
                    model_id,
                )

            finally:
                if acquired:
                    _REMOTE_CALL_SEMAPHORE.release()

        if unavailable:
            logger.warning(
                "Remote model unavailable, trying next candidate: %s (%s)",
                model_id,
                last_detail,
            )
            remote_attempts.append(
                {"model_id": model_id, "status": "unavailable", "detail": last_detail}
            )
            continue

        if malformed_response:
            continue

        # Transient errors exhausted for this model — try the next candidate.
        remote_attempts.append(
            {"model_id": model_id, "status": "transient_failed", "detail": last_detail}
        )

    # No remote candidate worked. Return a clear structured error; the original
    # prompt payload is preserved on the result (original_prompt + diagnostics).
    attempted = ", ".join(a["model_id"] for a in remote_attempts) or "none"
    logger.error("All remote candidates failed. Attempted: %s", attempted)
    return _remote_result(
        "⚠️ **All remote models failed.**\n\n"
        f"Attempted (in order): {attempted}\n\n"
        f"Last error: {last_detail}",
        distill_tokens,
        max_attempts,
        candidates[-1] if candidates else active_remote_model,
    )


BURNED_ROUTES = ("VISION_REMOTE", "TEXT_REMOTE", "FALLBACK_REMOTE", "PHANTOM_RACE")
SAVED_ROUTES = ("MATH_PYTHON", "TEXT_LOCAL", "CACHE_HIT")


def _aggregate_swarm_metrics(result: RouteResult) -> tuple[str, int, float, int]:
    """Return route label, total tokens, parallel runtime, sub-agent count."""
    sub_results = result.sub_results
    total_tokens = sum(item.tokens for item in sub_results)
    # Parallel runtime = wall-clock of the swarm (≈ slowest thread), never the sum.
    if result.wall_clock_ms is not None:
        parallel_latency = result.wall_clock_ms
    else:
        parallel_latency = max((item.latency_ms for item in sub_results), default=0.0)
    route_parts = sorted({item.route for item in sub_results})
    route_label = f"🐝 AGENT_SWARM ({len(sub_results)} agents: {', '.join(route_parts)})"
    return route_label, total_tokens, parallel_latency, len(sub_results)


def render_metrics(result: RouteResult) -> None:
    """Render hackathon demo metrics in three columns above the answer."""
    col1, col2, col3 = st.columns(3)

    route_labels = {
        "MATH_PYTHON": "🧮 MATH_PYTHON",
        "VISION_REMOTE": "👁️ VISION_REMOTE",
        "TEXT_LOCAL": "💻 TEXT_LOCAL",
        "TEXT_REMOTE": "☁️ TEXT_REMOTE",
        "FALLBACK_REMOTE": "☁️ TEXT_REMOTE (fallback)",
        "AGENT_SWARM": "🐝 AGENT_SWARM",
        "CACHE_HIT": "💾 CACHE_HIT",
        "PHANTOM_RACE": "👻 PHANTOM_RACE",
    }

    if result.route == "AGENT_SWARM" and result.sub_results:
        route_label, total_tokens, parallel_latency, agent_count = _aggregate_swarm_metrics(result)
        total_latency = parallel_latency
        burned = any(item.route in BURNED_ROUTES for item in result.sub_results)
        saved = any(item.route in SAVED_ROUTES for item in result.sub_results)
    else:
        route_label = route_labels.get(result.route, result.route)
        total_tokens = result.tokens
        total_latency = result.latency_ms
        burned = result.route in BURNED_ROUTES
        saved = result.route in SAVED_ROUTES
        agent_count = 0

    with col1:
        st.markdown(f"**Route**  \n{route_label}")
        if result.prompt_type:
            mode = {
                "DIRECT_ANSWER": "🎯 Direct answer (1 agent)",
                "LOCAL_DECOMPOSE": "🔀 Decomposed swarm",
                "REMOTE_ESCALATE": "☁️ Cloud escalation (1 agent)",
            }.get(result.prompt_type, result.prompt_type)
            st.caption(mode)
        if result.decomposition_used:
            st.caption(f"Agents spawned: {result.num_agents}")
        if result.route == "FALLBACK_REMOTE":
            st.caption(f"↩️ Actual backend: Fireworks `{result.model_used}`")
        elif result.route == "AGENT_SWARM" and result.fallback_used:
            st.caption("↩️ Fallback to Remote occurred")
        if result.model_used and result.route not in ("AGENT_SWARM", "FALLBACK_REMOTE"):
            st.caption(f"Backend: `{result.model_used}`")
        if result.routing_reason and result.route != "AGENT_SWARM":
            st.caption(f"Reason: {result.routing_reason}")
        if result.escalation_reason and result.route != "AGENT_SWARM":
            st.caption(f"Escalation: {result.escalation_reason}")
        if result.confidence_score is not None and result.route != "AGENT_SWARM":
            st.caption(f"Confidence: {result.confidence_score:.0%}")

    with col2:
        if result.route == "AGENT_SWARM" and result.sub_results:
            if burned and saved:
                st.markdown(f"**Token Usage**  \n🔥 {total_tokens} Total (swarm aggregate)")
            elif burned:
                st.markdown(f"**Token Usage**  \n🔥 {total_tokens} Tokens Burned")
            else:
                st.markdown(f"**Token Usage**  \n✅ {total_tokens} Tokens Saved")
        elif burned:
            st.markdown(f"**Token Usage**  \n🔥 {total_tokens} Tokens Burned")
        else:
            st.markdown(f"**Token Usage**  \n✅ {total_tokens} Tokens Saved")

    with col3:
        if agent_count:
            sequential_sum = sum(item.latency_ms for item in result.sub_results)
            st.markdown(
                f"**Latency**  \n⏱️ {total_latency:.1f} ms parallel "
                f"({agent_count} sub-agents · {sequential_sum:.0f} ms sequential)"
            )
        else:
            st.markdown(f"**Latency**  \n⏱️ {total_latency:.1f} ms")


def render_middleware_telemetry(result: RouteResult) -> None:
    """Show distillation, fallback, and swarm middleware details."""
    sub_distillations = [sub for sub in result.sub_results if sub.distilled_prompt is not None]
    sub_fallbacks = [sub for sub in result.sub_results if sub.fallback_used]
    has_distillation = result.distilled_prompt is not None or bool(sub_distillations)
    has_swarm = bool(result.sub_results)
    has_fallback = result.fallback_used or bool(sub_fallbacks)

    def _attempts(res: RouteResult) -> list[dict]:
        return list(res.diagnostics.get("remote_attempts", []) or [])

    remote_attempts = _attempts(result)
    sub_attempts = [(i, _attempts(s)) for i, s in enumerate(result.sub_results, start=1)]
    has_remote_attempts = (
        len(remote_attempts) > 1
        or any(len(a) > 1 for _, a in sub_attempts)
    )

    has_cache = result.route == "CACHE_HIT"
    has_phantom = result.route == "PHANTOM_RACE" or result.diagnostics.get("phantom")
    has_cascade = result.diagnostics.get("cascade_escalated", False)

    if not has_distillation and not has_swarm and not has_fallback and not has_remote_attempts and not has_cache and not has_phantom:
        return

    with st.expander(
        "ANGKOR + PHANTOM Telemetry",
        expanded=False,
        key=_render_key("telemetry_expander", result),
    ):
        if result.complexity_score is not None:
            st.markdown(f"**Complexity score:** `{result.complexity_score}`")

        if has_cache:
            st.success("💾 **Cache Hit** — Zero tokens, instant response from semantic cache.")
            if _ANGKOR_CACHE:
                st.caption(f"Cache size: {_ANGKOR_CACHE.size} entries · Hit rate: {_ANGKOR_CACHE.hit_rate:.1%}")

        if has_phantom:
            phantom_data = result.diagnostics.get("phantom", {})
            st.markdown("#### 👻 PHANTOM Race")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Local status", phantom_data.get("local_status", "?"))
            with col2:
                st.metric("Remote status", phantom_data.get("remote_status", "?"))
            with col3:
                st.metric("Winner", phantom_data.get("winner", "?"))
            if phantom_data.get("local_entropy") is not None:
                st.metric("Entropy at check", f"{phantom_data['local_entropy']:.3f}")
            budget = phantom_data.get("budget")
            if budget:
                st.metric("Token budget", budget)
            angkor_c = result.diagnostics.get("angkor_complexity")
            angkor_t = result.diagnostics.get("angkor_theta")
            if angkor_c is not None and angkor_t is not None:
                st.markdown(f"**Angkor:** C={angkor_c:.3f} θ={angkor_t:.3f}")

        if has_cascade:
            st.info("🔁 **Cascade Verify** — Binary escalation corrected the output.")

        def _render_attempts(attempts: list[dict], label: str) -> None:
            if len(attempts) <= 1:
                return
            st.markdown(f"#### Remote Model Fallback — {label}")
            for index, att in enumerate(attempts, start=1):
                status = att.get("status", "?")
                icon = "✅" if status == "ok" else "↪️"
                detail = att.get("detail", "")
                suffix = f" — {detail}" if detail else ""
                st.markdown(f"{icon} `{att.get('model_id')}` → **{status}**{suffix}")

        _render_attempts(remote_attempts, "request")
        for index, attempts in sub_attempts:
            _render_attempts(attempts, f"sub-agent {index}")

        if result.route == "FALLBACK_REMOTE" and result.fallback_reason:
            st.warning(
                f"Local→Remote fallback: {result.fallback_reason} "
                f"Routed to `{result.model_used}`."
            )

        if result.distilled_prompt is not None:
            st.markdown("#### Distillation Report")
            st.caption("Original prompt compressed locally before Fireworks inference.")
            col_orig, col_dist = st.columns(2)
            with col_orig:
                st.text_area(
                    "Original prompt",
                    value=result.original_prompt,
                    height=120,
                    disabled=True,
                    key=_render_key("original_prompt", result)
                )
            with col_dist:
                st.text_area(
                    "Distilled prompt (sent to Fireworks)",
                    value=result.distilled_prompt,
                    height=120,
                    disabled=True,
                    key=_render_key("distilled_prompt", result)
                )
            st.metric("Characters saved", result.distillation_chars_saved)
            if result.distillation_error:
                st.warning(result.distillation_error)

        for index, sub in enumerate(sub_distillations, start=1):
            st.markdown(f"#### Distillation Report — Sub-Agent {index}")
            col_orig, col_dist = st.columns(2)
            with col_orig:
                st.text_area(
                    f"Original (sub-agent {index})",
                    value=sub.original_prompt,
                    height=100,
                    disabled=True,
                    key=_render_key("orig_prompt", result, str(index))
                )
            with col_dist:
                st.text_area(
                    f"Distilled (sub-agent {index})",
                    value=sub.distilled_prompt or "",
                    height=100,
                    disabled=True,
                    key=_render_key("dist_prompt", result, str(index))
                )
            st.metric(f"Characters saved (sub-agent {index})", sub.distillation_chars_saved)

        if has_swarm:
            st.markdown("#### Agent Swarm Decomposition")
            worker_count = max(1, min(len(result.sub_results), SWARM_MAX_CONCURRENT))
            st.caption(
                f"Parallel sub-agents via ThreadPoolExecutor "
                f"(max_workers={worker_count}, wall-clock latency)."
            )
            for index, sub in enumerate(result.sub_results, start=1):
                badge = " ↩️ fallback→remote" if sub.fallback_used else ""
                st.markdown(
                    f"**Sub-Agent {index}** → `{sub.route}`{badge} · {sub.latency_ms:.1f} ms"
                )
                st.code(
                    sub.original_prompt,
                    language=None,
                    key=_render_key("swarm_prompt", result, str(index)),
                )
                if sub.fallback_used and sub.fallback_reason:
                    st.caption(f"↩️ {sub.fallback_reason} → `{sub.model_used}`")
            if result.wall_clock_ms is not None:
                st.info(f"Wall-clock swarm latency: {result.wall_clock_ms:.1f} ms")


def render_assistant_message(message: dict) -> None:
    result = message["result"]
    if message.get("message_id") and not result.message_id:
        result.message_id = str(message["message_id"])
    render_metrics(result)
    render_middleware_telemetry(result)
    st.markdown(message["content"])


def _lazy_init_angkor_cache() -> SemanticCache | None:
    """Load FAISS cache once per Streamlit session (not every script rerun)."""
    global _ANGKOR_CACHE
    if _ANGKOR_CACHE is not None:
        return _ANGKOR_CACHE
    if not _has_ui_context():
        return None
    ss = st.session_state
    if ss.get("angkor_cache_initialized"):
        _ANGKOR_CACHE = ss.get("angkor_cache")
        return _ANGKOR_CACHE
    if ss.get("angkor_cache_initializing"):
        return None
    ss.angkor_cache_initializing = True
    cache = SemanticCache()
    initialized = cache.initialize()
    if initialized:
        ss.angkor_cache = cache
        _ANGKOR_CACHE = cache
    else:
        ss.angkor_cache = None
    ss.angkor_cache_initialized = True
    ss.angkor_cache_initializing = False
    return _ANGKOR_CACHE


def _bootstrap_angkor_session() -> None:
    """Persist ANGKOR/PHANTOM singletons across Streamlit reruns via session_state."""
    global _ANGKOR_CACHE, _ANGKOR_SKLEARN_ROUTER, _ANGKOR_ADAPTIVE_THETA
    global _ANGKOR_PHANTOM_RUNNER, _ANGKOR_VERIFIER
    if not _has_ui_context():
        return
    ss = st.session_state
    if not ss.get("angkor_session_bootstrapped"):
        ss.angkor_session_bootstrapped = True
        ss.angkor_cache = None
        ss.angkor_cache_initialized = False
        ss.angkor_cache_initializing = False
        try:
            from sklearn.linear_model import LogisticRegression  # noqa: F401
            ss.angkor_sklearn_router = SklearnRouter()
        except ImportError:
            ss.angkor_sklearn_router = None
        ss.angkor_adaptive_theta = AdaptiveThreshold()
        ss.angkor_phantom_runner = SpeculativeRunner()
        ss.angkor_verifier = CascadeVerifier()
    _ANGKOR_SKLEARN_ROUTER = ss.get("angkor_sklearn_router")
    _ANGKOR_ADAPTIVE_THETA = ss.get("angkor_adaptive_theta")
    _ANGKOR_PHANTOM_RUNNER = ss.get("angkor_phantom_runner")
    _ANGKOR_VERIFIER = ss.get("angkor_verifier")
    _ANGKOR_CACHE = ss.get("angkor_cache")


def _load_saved_api_key() -> str:
    path = ROOT_DIR / ".fireworks_api_key"
    if path.exists():
        try:
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
        except Exception:
            pass
    return ""


def _save_api_key(key: str) -> None:
    path = ROOT_DIR / ".fireworks_api_key"
    try:
        path.write_text(key.strip(), encoding="utf-8")
    except Exception:
        pass


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "attached_txt_bytes" not in st.session_state:
        st.session_state.attached_txt_bytes = None
    if "attached_txt_name" not in st.session_state:
        st.session_state.attached_txt_name = ""
    if "_pending_chat" not in st.session_state:
        st.session_state._pending_chat = None
    if "fireworks_api_key" not in st.session_state:
        st.session_state.fireworks_api_key = _load_saved_api_key()


def _render_orchestration_caption(result: RouteResult) -> None:
    orch = result.diagnostics.get("orchestration")
    if isinstance(orch, dict):
        ptype = orch.get("prompt_type")
        if ptype == "DIRECT_ANSWER":
            st.caption("🎯 Direct answer — single agent, no decomposition.")
        elif ptype == "REMOTE_ESCALATE":
            st.caption("☁️ Cloud escalation — single remote agent.")
        elif orch.get("decomposition_used"):
            st.caption(f"🔀 Decomposed into {orch.get('num_agents', 1)} sub-agents.")
    elif result.prompt_type == "DIRECT_ANSWER":
        st.caption("🎯 Direct answer — single agent, no decomposition.")
    elif result.prompt_type == "REMOTE_ESCALATE":
        st.caption("☁️ Cloud escalation — single remote agent.")
    elif result.decomposition_used:
        st.caption(f"🔀 Decomposed into {result.num_agents} sub-agents.")


def _handle_pending_chat(
    *,
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
) -> None:
    """
    Second-phase chat handler: user message is already in history; show lightweight
    feedback immediately (no st.spinner dimming), then dispatch.
    """
    pending = st.session_state.pop("_pending_chat", None)
    if not pending:
        return

    prompt = str(pending["prompt"])
    full_prompt = str(pending["full_prompt"])
    has_txt_context = bool(pending.get("has_txt_context"))
    instant = bool(pending.get("instant"))
    image_bytes = pending.get("image_bytes")
    image_file: BinaryIO | None = None
    if image_bytes:
        image_file = io.BytesIO(image_bytes)

    timing = RequestTiming()
    timing.mark("input_received")

    with st.chat_message("assistant"):
        status_slot = st.empty()
        if instant:
            status_slot.caption("⚡ Instant reply")
        else:
            status_slot.markdown("_Routing…_")
        timing.mark("ui_feedback_shown")

        if instant:
            result = dispatch_instant_trivial(
                prompt,
                threshold,
                active_local_model,
                active_remote_model,
                timing=timing,
            )
        else:
            result = run_request_nonblocking(
                process_user_request,
                full_prompt,
                threshold,
                api_key,
                active_local_model,
                active_remote_model,
                image_file=image_file,
                has_txt_context=has_txt_context,
                timing=timing,
            )

        status_slot.empty()
        _render_orchestration_caption(result)
        message_id = uuid.uuid4().hex
        result.message_id = message_id
        assistant_message = {
            "role": "assistant",
            "content": result.answer,
            "result": result,
            "message_id": message_id,
        }
        render_assistant_message(assistant_message)
        st.session_state.messages.append(assistant_message)


def run_request_nonblocking(fn, *args, **kwargs) -> RouteResult:
    """Run routing/dispatch off the Streamlit main thread with a hard timeout."""
    future = UI_EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=REMOTE_UI_TIMEOUT_SECONDS)
    except FuturesTimeoutError:
        logger.warning("UI dispatch timed out after %ss", REMOTE_UI_TIMEOUT_SECONDS)
        prompt_preview = str(args[0]) if args else ""
        return RouteResult(
            answer=(
                "⚠️ Request timed out while waiting for the model. "
                "The UI stayed responsive — try again or shorten the prompt."
            ),
            route="TEXT_REMOTE",
            tokens=0,
            latency_ms=REMOTE_UI_TIMEOUT_SECONDS * 1000.0,
            original_prompt=prompt_preview,
            routing_reason="ui-timeout",
            diagnostics={"cancelled": True},
        )


def main() -> None:
    st.set_page_config(
        page_title="Hybrid Routing Agent",
        page_icon="🔀",
        layout="wide",
    )

    st.title("🔀 ANGKOR Router + PHANTOM Layer")
    st.caption(
        "Adaptive Neural Gate for Knowledge-Optimized Routing — "
        "with Predictive Hallucination and Token Management Optimizer. "
        "T0: Cache | T1: Compress | T2: 3-Zone Router | PHANTOM: Speculative Race | T3: Verify"
    )

    init_session_state()

    # -- ANGKOR + PHANTOM session-persistent singletons (survive Streamlit reruns) --
    _bootstrap_angkor_session()

    with st.sidebar:
        st.header("⚙️ ANGKOR + PHANTOM Configuration")

        entered_key = st.text_input(
            "Fireworks API Key",
            type="password",
            placeholder="fw_...",
            value=st.session_state.fireworks_api_key,
            help="Required for remote, vision, and fallback routes. Saved to disk once entered.",
        )
        if entered_key != st.session_state.fireworks_api_key:
            st.session_state.fireworks_api_key = entered_key
            _save_api_key(entered_key)
        api_key = entered_key

        active_local_model = st.text_input(
            "Local Utility Model (Ollama)",
            value=DEFAULT_LOCAL_MODEL,
            help="Lightweight Ollama model for local inference & PHANTOM early abort.",
        ).strip() or DEFAULT_LOCAL_MODEL

        remote_choice = st.selectbox(
            "Remote Fireworks Model",
            options=REMOTE_MODEL_OPTIONS,
            index=0,
            help="Fireworks model for remote inference & PHANTOM race.",
        )
        if remote_choice == CUSTOM_MODEL_SENTINEL:
            active_remote_model = st.text_input(
                "Enter Custom Model ID",
                value="accounts/fireworks/models/",
            ).strip()
        else:
            active_remote_model = remote_choice

        if not active_remote_model:
            active_remote_model = DEFAULT_REMOTE_MODEL
            st.warning("Empty custom model ID — using default remote model.")

        active_remote_model = normalize_model_id(active_remote_model) or DEFAULT_REMOTE_MODEL
        st.caption(f"Active remote model: `{active_remote_model}`")

        candidate_chain = build_remote_candidates(active_remote_model)
        if not is_known_deployed_model(active_remote_model):
            st.warning(
                f"⚠️ `{active_remote_model}` is not in the known-deployed registry. "
                "It will only be tried as a last resort.\n\n"
                f"**Fallback order:** {', '.join(f'`{m}`' for m in candidate_chain)}"
            )
        else:
            st.caption("Fallback order: " + " → ".join(f"`{m}`" for m in candidate_chain))

        st.divider()

        # -- ANGKOR Controls -------------------------------------------------
        st.subheader("🎯 ANGKOR Router")
        cache_threshold = st.slider(
            "Cache Similarity Floor (Tier 0)",
            min_value=0.70, max_value=0.99, value=0.90, step=0.01,
            help="Cosine similarity threshold for FAISS cache hit.",
        )
        if _ANGKOR_CACHE:
            _ANGKOR_CACHE._config = _ANGKOR_CACHE._config.__class__(
                threshold=cache_threshold,
                model_name=_ANGKOR_CACHE._config.model_name,
                index_path=_ANGKOR_CACHE._config.index_path,
                store_path=_ANGKOR_CACHE._config.store_path,
            )

        phantom_dead_zone = st.slider(
            "PHANTOM Dead Zone (±θ)",
            min_value=0.02, max_value=0.25, value=0.10, step=0.01,
            help="Band around θ that triggers PHANTOM speculative race.",
        )

        entropy_threshold = st.slider(
            "Entropy Abort Threshold",
            min_value=2.0, max_value=5.0, value=3.5, step=0.1,
            help="H(Y) above this → abort local generation (PHANTOM A).",
        )
        if _ANGKOR_PHANTOM_RUNNER is not None:
            _ANGKOR_PHANTOM_RUNNER._confidence._abort_threshold = entropy_threshold

        theta_current = _ANGKOR_ADAPTIVE_THETA.theta if _ANGKOR_ADAPTIVE_THETA else 0.65
        st.metric("Adaptive θ", f"{theta_current:.3f}", delta=None)

        uploaded_image = st.file_uploader(
            "Upload Image (optional)",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"],
            help="When attached, the next message routes through the vision pipeline.",
        )
        uploaded_txt = st.file_uploader(
            "Attach .txt context (optional)",
            type=["txt"],
            help="Drag or upload a plain-text file. Its contents are added as context.",
        )
        if uploaded_txt is not None:
            st.session_state.attached_txt_bytes = uploaded_txt.getvalue()
            st.session_state.attached_txt_name = uploaded_txt.name
        if st.session_state.get("attached_txt_name"):
            st.caption(f"Text context: `{st.session_state.attached_txt_name}`")

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Attached image preview", use_container_width=True)

        st.divider()

        st.info(
            "**ANGKOR + PHANTOM Pipeline:**\n"
            "T0: FAISS Cache → T1: Compress → T2: 3-Zone Router → "
            "PHANTOM: Entropy Abort + Speculative Race + Budget Enforcement → "
            "T3: Cascade Verify"
        )

        if _ANGKOR_CACHE:
            st.caption(f"Cache: {_ANGKOR_CACHE.size} entries · {_ANGKOR_CACHE.hit_rate:.1%} hit rate")

        if st.button("Clear Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.attached_txt_bytes = None
            st.session_state.attached_txt_name = ""

    # Local-backend health banner (non-blocking — never stall the UI on Ollama probe)
    local_health = get_local_health_for_ui(active_local_model)
    if local_health is False:
        st.warning(
            f"⚠️ Local utility model `{active_local_model}` is **unavailable**. "
            "Greetings use canned replies; other prompts may route remote."
        )
    elif local_health is True:
        mem_usage = get_memory_usage()
        if mem_usage is not None and mem_usage >= MEMORY_PRESSURE_THRESHOLD:
            st.warning(
                f"⚠️ Memory at {mem_usage:.0%} — local utility model bypassed; "
                "routing to Fireworks remote."
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])
                if message.get("image_preview"):
                    st.image(message["image_preview"], width=200)

    _handle_pending_chat(
        threshold=DEFAULT_COMPLEXITY_THRESHOLD,
        api_key=api_key,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
    )

    threshold = DEFAULT_COMPLEXITY_THRESHOLD

    if prompt := st.chat_input("Ask anything, or attach files in the sidebar..."):
        txt_block, txt_chars = build_txt_context(st.session_state.get("attached_txt_bytes"))
        has_txt_context = txt_chars > 0
        full_prompt = f"{prompt}{txt_block}" if txt_block else prompt

        user_content = prompt
        if uploaded_image is not None:
            user_content += "\n\n📎 *Image attached*"
        if has_txt_context:
            user_content += f"\n\n📎 *Text file attached:* `{st.session_state.get('attached_txt_name', 'file.txt')}`"

        user_message = {
            "role": "user",
            "content": user_content,
            "image_preview": uploaded_image.getvalue() if uploaded_image else None,
            "txt_attached": has_txt_context,
        }
        st.session_state.messages.append(user_message)
        st.session_state._pending_chat = {
            "prompt": prompt,
            "full_prompt": full_prompt,
            "has_txt_context": has_txt_context,
            "instant": is_trivial_fast_path(
                prompt,
                has_image=uploaded_image is not None,
                has_txt_context=has_txt_context,
            ),
            "image_bytes": uploaded_image.getvalue() if uploaded_image else None,
        }
        st.rerun()


if __name__ == "__main__":
    main()
