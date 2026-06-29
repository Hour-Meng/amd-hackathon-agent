"""Hybrid Token-Efficient Routing Agent — Streamlit chatbot demo."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import BinaryIO, Literal

import threading

import requests
import streamlit as st
from PIL import Image

logger = logging.getLogger("hybrid_router")
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
LOCAL_MATH_REASON = "local-allowed:math-python"
LONG_PROMPT_CHARS = 180
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
REMOTE_VISION_MODEL = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

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
]

MATH_EXTRACT_PATTERN = re.compile(r"([\d\s\+\-\*\/\(\)\.]{3,})")
MATH_OPERATOR_PATTERN = re.compile(r"[\+\-\*\/]")
MATH_PREFIX_PATTERN = re.compile(r"^math:\s*", re.IGNORECASE)

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
    # Hierarchical orchestration metadata (classifier → planner → executor).
    prompt_type: str | None = None
    decomposition_used: bool = False
    num_agents: int = 1
    escalation_reason: str | None = None


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


def is_code_generation_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in CODE_GEN_PATTERNS)


def is_simple_format_task(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(pattern in lowered for pattern in FORMAT_PATTERNS) and not is_character_level_task(
        prompt
    )


def is_multi_hop_prompt(prompt: str) -> bool:
    """Multi-part or analytical prompts that need remote reasoning."""
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


def is_local_capable_prompt(prompt: str) -> bool:
    """Prompts within the local capability ceiling (greetings/format — no LLM facts)."""
    if is_local_trivial_whitelisted(prompt):
        return True
    return is_simple_format_task(prompt)


def would_math_intercept(prompt: str) -> bool:
    """True when deterministic math eval would handle the prompt (no LLM)."""
    return safe_math_agent(prompt, time.perf_counter()) is not None


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


@dataclass
class RouterDecision:
    """Authoritative routing decision produced before any backend is touched."""

    route: Literal["LOCAL", "REMOTE"]
    complexity_score: int
    reason: str
    model_id: str
    confidence_score: float = 0.0


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
) -> RouterDecision:
    """
    Router-first single source of truth. Runs BEFORE any model call.

    Default is REMOTE (Fireworks). LOCAL is allowed only for deterministic math,
    short greetings, and simple formatting when the local backend is healthy.
    """
    score = min(100, calculate_complexity(prompt))
    remote_model = normalize_model_id(active_remote_model) or DEFAULT_REMOTE_MODEL

    if has_image:
        return RouterDecision("REMOTE", score, "vision:remote", remote_model, 0.95)

    if is_character_level_task(prompt):
        return RouterDecision(
            "REMOTE", score, CHARACTER_LEVEL_GUARD_REASON, remote_model, 0.99
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

    if would_math_intercept(prompt):
        return RouterDecision("LOCAL", score, LOCAL_MATH_REASON, "python-eval", 1.0)

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


def _math_candidates(prompt: str) -> list[str]:
    """Build candidate strings for math extraction (planner prefix + embedded expr)."""
    stripped = prompt.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
    if MATH_PREFIX_PATTERN.match(stripped):
        expr = MATH_PREFIX_PATTERN.sub("", stripped).strip()
        if expr and expr not in candidates:
            candidates.insert(0, expr)
    return candidates


def safe_math_agent(prompt: str, started: float) -> RouteResult | None:
    """
    Extract embedded mathematical expressions from natural-language prompts.
    Planner-tagged tasks like 'math: 2+12' are stripped and evaluated first.

    Highest-priority interceptor: any sub-task containing a real arithmetic
    expression (must include an operator) is computed locally for zero tokens.
    """
    seen: set[str] = set()
    for text in _math_candidates(prompt):
        if text in seen:
            continue
        seen.add(text)
        for match in MATH_EXTRACT_PATTERN.finditer(text):
            candidate = match.group(1).strip()
            if len(candidate) < 3:
                continue
            if not MATH_OPERATOR_PATTERN.search(candidate):
                continue
            if not any(ch.isdigit() for ch in candidate):
                continue
            try:
                result = eval(candidate, {"__builtins__": None}, {})  # noqa: S307
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
                continue
    return None


_HEURISTIC_SPLIT = re.compile(
    r",\s*(?=(?:what|who|where|when|why|how|tell|capital|population|math:))",
    re.IGNORECASE,
)


def heuristic_task_split(prompt: str) -> list[str]:
    """Heuristic task split without any local LLM planner call."""
    cleaned = prompt.strip()
    if not cleaned:
        return [cleaned]
    parts = [part.strip() for part in _HEURISTIC_SPLIT.split(cleaned) if part.strip()]
    if len(parts) > 1:
        return parts
    if cleaned.count(",") >= 2 and len(cleaned) > 40:
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) > 1:
            return parts
    return [cleaned]


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
    # Multi-segment comma lists are never single-turn direct answers.
    if stripped.count(",") >= 2 and len(stripped) > 35:
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
    """True only when multiple independent substantial tasks justify a swarm."""
    if is_direct_answer_prompt(prompt):
        return False
    if is_character_level_task(prompt):
        return False
    parts = heuristic_task_split(prompt)
    if len(parts) < 2:
        return False
    substantial = [
        part
        for part in parts
        if len(part) > 8 and not is_local_trivial_whitelisted(part)
    ]
    if len(substantial) >= 2:
        return True
    # Comma-list prompts with 3+ segments are multi-task even when segments are short.
    if len(parts) >= 3 and prompt.count(",") >= 2 and len(prompt) > 35:
        return True
    return False


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


def classify_prompt(
    prompt: str,
    threshold: int,
    active_local_model: str,
    active_remote_model: str,
    *,
    has_image: bool = False,
) -> ClassificationResult:
    """
    Hierarchical level-1 reader/classifier (rule-based, no LLM).

    Decides DIRECT_ANSWER | LOCAL_DECOMPOSE | REMOTE_ESCALATE on the WHOLE prompt
    before any decomposition or model call.
    """
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

    if has_image:
        return ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="image",
            route="REMOTE",
            reason="vision:remote",
            confidence_score=0.95,
            escalation_reason="vision:remote",
            num_agents=1,
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
        )

    # Direct-answer guard: greetings, small talk, single-sentence questions.
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
        )

    # Cloud escalation for hard prompts that do not benefit from splitting.
    if decision.route == "REMOTE" and not is_beneficial_to_decompose(prompt):
        return ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="cloud_escalation",
            route="REMOTE",
            reason=decision.reason,
            confidence_score=decision.confidence_score,
            escalation_reason=decision.reason,
            num_agents=1,
        )

    # Level-2: decompose only when multiple independent tasks justify it.
    if is_beneficial_to_decompose(prompt):
        parts = heuristic_task_split(prompt)
        if any(is_character_level_task(part) for part in parts):
            return ClassificationResult(
                prompt_type="REMOTE_ESCALATE",
                prompt_label="character_level_subtask",
                route="REMOTE",
                reason=CHARACTER_LEVEL_GUARD_REASON,
                confidence_score=0.99,
                escalation_reason=CHARACTER_LEVEL_GUARD_REASON,
                num_agents=1,
            )
        return ClassificationResult(
            prompt_type="LOCAL_DECOMPOSE",
            prompt_label="multi_task",
            route=decision.route,
            reason="decompose:beneficial",
            confidence_score=decision.confidence_score,
            decomposition_used=True,
            num_agents=len(parts),
        )

    return ClassificationResult(
        prompt_type="DIRECT_ANSWER" if decision.route == "LOCAL" else "REMOTE_ESCALATE",
        prompt_label="single_route",
        route=decision.route,
        reason=decision.reason,
        confidence_score=decision.confidence_score,
        escalation_reason=decision.reason if decision.route == "REMOTE" else None,
        num_agents=1,
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
        "decomposition_used=%s num_agents=%s escalation_reason=%s "
        "confidence=%.2f latency_ms=%.1f prompt=%r",
        classification.route,
        classification.prompt_type,
        classification.prompt_label,
        classification.decomposition_used,
        classification.num_agents,
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
        "decomposition_used": classification.decomposition_used,
        "num_agents": classification.num_agents,
        "escalation_reason": classification.escalation_reason,
    }
    return result


_TASK_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_TASK_SPLIT_PATTERN = re.compile(
    r",\s*(?=(?:and\s+)?(?:what|who|where|when|why|how|tell|spell|write|list|"
    r"give|find|name|capital|population|explain|count|reverse|calculate|"
    r"summarize|describe|compute|then|also|finally|next))",
    re.IGNORECASE,
)
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


def count_tasks(text: str) -> int:
    """
    Count distinct tasks/questions/instructions in a prompt.

    Strong structural signals win first (bullets/numbered steps, separate lines,
    sentence enumerators like "first/second/third", question marks). An inline
    comma list is only counted as multi-task when it is unambiguous — 3+ items,
    or every segment begins with an imperative/interrogative keyword — so prose
    with mid-sentence commas is not mistaken for several tasks. Always >= 1 for
    non-empty input.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return 0

    strong = [1]
    strong.append(len(_TASK_BULLET_PATTERN.findall(cleaned)))
    strong.append(len([line for line in cleaned.splitlines() if line.strip()]))
    lowered = cleaned.lower()
    strong.append(sum(1 for w in _TASK_ENUM_WORDS if re.search(rf"\b{w}\b", lowered)))
    if "?" in cleaned:
        strong.append(len([seg for seg in cleaned.split("?") if seg.strip()]))

    best = max(strong)
    if best >= 2:
        return best

    # Fallback: inline comma list, only when unambiguous.
    segments = [seg for seg in _TASK_SPLIT_PATTERN.split(cleaned) if seg and seg.strip()]
    if len(segments) >= 3 or (
        len(segments) >= 2 and all(_starts_with_task_keyword(seg) for seg in segments)
    ):
        return len(segments)

    return 1


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

    # Agent-swarm guard: direct answers and escalations are always single-route.
    if (
        classification.num_agents == 1
        or not classification.decomposition_used
        or classification.prompt_type != "LOCAL_DECOMPOSE"
    ):
        return SwarmPlan(
            tasks=[prompt],
            global_route=classification.route,
            reason=classification.reason,
            single_route=True,
            classification=classification,
        )

    parts = heuristic_task_split(prompt)
    if any(is_character_level_task(part) for part in parts):
        escalated = ClassificationResult(
            prompt_type="REMOTE_ESCALATE",
            prompt_label="character_level_subtask",
            route="REMOTE",
            reason=CHARACTER_LEVEL_GUARD_REASON,
            confidence_score=0.99,
            escalation_reason=CHARACTER_LEVEL_GUARD_REASON,
            num_agents=1,
        )
        return SwarmPlan(
            tasks=[prompt],
            global_route="REMOTE",
            reason=CHARACTER_LEVEL_GUARD_REASON,
            single_route=True,
            classification=escalated,
        )

    if len(parts) <= 1:
        return SwarmPlan(
            tasks=[prompt],
            global_route=classification.route,
            reason=classification.reason,
            single_route=True,
            classification=classification,
        )

    return SwarmPlan(
        tasks=parts,
        global_route=classification.route,
        reason=classification.reason,
        single_route=False,
        classification=classification,
    )


def route_and_execute(
    prompt: str,
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    image_file: BinaryIO | None = None,
    *,
    allow_heavy_local: bool = False,
) -> RouteResult:
    """
    Produce an authoritative router decision, then execute the chosen backend.

    Router authority: if the decision is REMOTE, ALL local-model logic (including
    math eval and length/keyword heuristics) is bypassed and the remote model is
    always used. Character-level tasks are hard-routed REMOTE by the pre-router
    override. Local failures may rescue UP to remote, but a REMOTE decision is
    never executed by the local model.

    Heavy-local safety: before committing to a heavy local model the backend is
    health-gated and memory-checked so the app never blocks on a dead/overloaded
    local model.
    """
    started = time.perf_counter()
    has_image = image_file is not None

    # Only probe health/memory for routing — router runs before any model call.
    local_healthy = not has_image and check_local_health(active_local_model)
    local_unavailable = not has_image and not local_healthy
    memory_pressure = get_memory_usage() if not has_image else None

    decision = route_decision(
        prompt,
        threshold,
        has_image=has_image,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
        local_unavailable=local_unavailable,
        memory_pressure=memory_pressure,
        allow_heavy_local=allow_heavy_local,
    )
    _log_routing(prompt, decision, local_healthy=local_healthy or has_image)

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
        )
        _log_executed(result)
        return result

    if decision.route == "REMOTE":
        # Skip local distillation for tokenization-sensitive tasks AND whenever the
        # local backend itself is the reason we are going remote (it would freeze).
        preserve_text = (
            decision.reason == CHARACTER_LEVEL_GUARD_REASON
            or decision.reason in LOCAL_DISTILL_UNSAFE_REASONS
        )
        result = _route_text_remote(
            prompt,
            api_key,
            active_local_model,
            active_remote_model,
            started,
            skip_distillation=preserve_text,
        )
        result.routing_reason = decision.reason
        result.complexity_score = decision.complexity_score
        result.confidence_score = decision.confidence_score
        _log_executed(result)
        return result

    # LOCAL route only: deterministic math eval (0 tokens) is allowed here.
    math_result = safe_math_agent(prompt, started)
    if math_result is not None:
        math_result.routing_reason = LOCAL_MATH_REASON
        math_result.complexity_score = decision.complexity_score
        math_result.confidence_score = decision.confidence_score
        _log_executed(math_result)
        return math_result

    result = _route_text_local(
        prompt, api_key, active_local_model, active_remote_model, started
    )
    result.routing_reason = decision.reason
    result.complexity_score = decision.complexity_score
    result.confidence_score = decision.confidence_score
    _log_executed(result)
    return result


def execute_agent_swarm(
    tasks: list[str],
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    *,
    allow_heavy_local: bool = False,
) -> RouteResult:
    """Run route_and_execute in parallel for each decomposed sub-question."""
    swarm_started = time.perf_counter()
    ordered: list[RouteResult | None] = [None] * len(tasks)

    # One worker per task (capped) so no sub-agent waits in a queue behind another.
    worker_count = max(1, min(len(tasks), 8))
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
        # Parallel runtime: dominated by the slowest thread, not the sequential sum.
        latency_ms=wall_latency_ms,
        original_prompt=" | ".join(tasks),
        fallback_used=any_fallback,
        sub_results=sub_results,
        wall_clock_ms=wall_latency_ms,
    )


def process_user_request(
    prompt: str,
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    image_file: BinaryIO | None = None,
    *,
    allow_heavy_local: bool = False,
) -> RouteResult:
    """Top-level orchestrator: classify → plan → single route or swarm."""
    if image_file is not None:
        clf = classify_prompt(
            prompt,
            threshold,
            active_local_model,
            active_remote_model,
            has_image=True,
        )
        result = route_and_execute(
            prompt,
            threshold,
            api_key,
            active_local_model,
            active_remote_model,
            image_file=image_file,
            allow_heavy_local=allow_heavy_local,
        )
        return _attach_orchestration(result, clf)

    plan = plan_request(
        prompt,
        threshold,
        active_local_model,
        active_remote_model,
        allow_heavy_local=allow_heavy_local,
    )

    if plan.single_route or plan.classification.num_agents <= 1:
        result = route_and_execute(
            prompt,
            threshold,
            api_key,
            active_local_model,
            active_remote_model,
            allow_heavy_local=allow_heavy_local,
        )
        return _attach_orchestration(result, plan.classification)

    result = execute_agent_swarm(
        plan.tasks,
        threshold,
        api_key,
        active_local_model,
        active_remote_model,
        allow_heavy_local=allow_heavy_local,
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

    def _remote_result(
        answer: str, tokens: int, retries: int, model_used: str
    ) -> RouteResult:
        return RouteResult(
            answer=answer,
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
            diagnostics={
                "remote_attempts": remote_attempts,
                "prompt_adjustment": adjust_method,
            },
        )

    # A REMOTE decision must be served remotely — we NEVER fall back to local here.
    # For each candidate model: retry transient errors; on an invalid/undeployed
    # model error, automatically advance to the next candidate.
    max_attempts = 2
    last_detail = "unknown error"

    for model_id in candidates:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SUB_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": distilled},
            ],
            "max_tokens": 128,
            "temperature": 0.0,
        }

        unavailable = False
        malformed_response = False
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    REMOTE_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=180,
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
                    f"⚠️ Remote inference error ({model_id}):\n\n{detail}",
                    distill_tokens,
                    attempt,
                    model_id,
                )

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


BURNED_ROUTES = ("VISION_REMOTE", "TEXT_REMOTE", "FALLBACK_REMOTE")
SAVED_ROUTES = ("MATH_PYTHON", "TEXT_LOCAL")


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

    if not has_distillation and not has_swarm and not has_fallback and not has_remote_attempts:
        return

    with st.expander("Middleware Telemetry"):
        if result.complexity_score is not None:
            st.markdown(f"**Complexity score:** `{result.complexity_score}`")

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
                )
            with col_dist:
                st.text_area(
                    "Distilled prompt (sent to Fireworks)",
                    value=result.distilled_prompt,
                    height=120,
                    disabled=True,
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
                )
            with col_dist:
                st.text_area(
                    f"Distilled (sub-agent {index})",
                    value=sub.distilled_prompt or "",
                    height=100,
                    disabled=True,
                )
            st.metric(f"Characters saved (sub-agent {index})", sub.distillation_chars_saved)

        if has_swarm:
            st.markdown("#### Agent Swarm Decomposition")
            worker_count = max(1, min(len(result.sub_results), 8))
            st.caption(
                f"Parallel sub-agents via ThreadPoolExecutor "
                f"(max_workers={worker_count}, wall-clock latency)."
            )
            for index, sub in enumerate(result.sub_results, start=1):
                badge = " ↩️ fallback→remote" if sub.fallback_used else ""
                st.markdown(
                    f"**Sub-Agent {index}** → `{sub.route}`{badge} · {sub.latency_ms:.1f} ms"
                )
                st.code(sub.original_prompt, language=None)
                if sub.fallback_used and sub.fallback_reason:
                    st.caption(f"↩️ {sub.fallback_reason} → `{sub.model_used}`")
            if result.wall_clock_ms is not None:
                st.info(f"Wall-clock swarm latency: {result.wall_clock_ms:.1f} ms")


def render_assistant_message(message: dict) -> None:
    result = message["result"]
    render_metrics(result)
    render_middleware_telemetry(result)
    st.markdown(message["content"])


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title="Hybrid Routing Agent",
        page_icon="🔀",
        layout="wide",
    )

    st.title("🔀 Router-First Hybrid AI Middleware")
    st.caption(
        "Track 1 middle layer: the router decides LOCAL (trivial/deterministic) vs "
        "REMOTE (Fireworks) before any model call. Remote is the default generator."
    )

    init_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Fireworks API Key",
            type="password",
            placeholder="fw_...",
            help="Required for remote, vision, and fallback routes.",
        )

        active_local_model = st.text_input(
            "Local Utility Model (optional)",
            value=DEFAULT_LOCAL_MODEL,
            help=(
                "Lightweight Ollama model for greetings/format only. "
                "Not the default generator — most prompts route to Fireworks."
            ),
        ).strip() or DEFAULT_LOCAL_MODEL

        remote_choice = st.selectbox(
            "Remote Fireworks Model",
            options=REMOTE_MODEL_OPTIONS,
            index=0,
            help="Fireworks model for remote and fallback inference. Pick 'Custom...' to type your own.",
        )
        if remote_choice == CUSTOM_MODEL_SENTINEL:
            active_remote_model = st.text_input(
                "Enter Custom Model ID",
                value="accounts/fireworks/models/",
                help="Full Fireworks model path, e.g. accounts/fireworks/models/<id>.",
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

        threshold = st.slider(
            "Complexity Threshold (Score)",
            min_value=0,
            max_value=200,
            value=30,
            step=5,
        )
        st.caption(
            "Prompts scoring above this route REMOTE; otherwise LOCAL. "
            "Score = base + length + analytical-keyword weights."
        )

        uploaded_image = st.file_uploader(
            "Upload Image (optional)",
            type=["jpg", "jpeg", "png"],
            help="When attached, the next message routes through the vision pipeline.",
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Attached image preview", use_container_width=True)

        st.divider()

        st.info(
            "Router-first stack:\n"
            "1. **Authoritative router** → LOCAL only for greetings/math/format\n"
            "2. **Default generator** → Fireworks remote API\n"
            "3. **Health gate** → skip dead/slow local backend\n"
            "4. **Prompt adjustment** → compress before remote\n"
            "5. **Agent swarm** → heuristic split + parallel remote sub-agents"
        )

        if st.button("Clear Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

    # Local-backend health banner (router-first: unhealthy local → all remote).
    local_healthy = check_local_health(active_local_model)
    if not local_healthy:
        st.warning(
            f"⚠️ Local utility model `{active_local_model}` is **unavailable**. "
            "All prompts route to Fireworks remote until the backend recovers."
        )
    else:
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

    if prompt := st.chat_input("Ask anything, or attach an image in the sidebar..."):
        user_content = prompt
        if uploaded_image is not None:
            user_content += "\n\n📎 *Image attached*"

        user_message = {
            "role": "user",
            "content": user_content,
            "image_preview": uploaded_image.getvalue() if uploaded_image else None,
        }
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(user_content)
            if uploaded_image is not None:
                st.image(uploaded_image, width=200)

        with st.chat_message("assistant"):
            if uploaded_image is not None:
                clf = classify_prompt(
                    prompt,
                    threshold,
                    active_local_model,
                    active_remote_model,
                    has_image=True,
                )
                plan = SwarmPlan(
                    tasks=[prompt],
                    global_route="REMOTE",
                    reason="vision:remote",
                    single_route=True,
                    classification=clf,
                )
            else:
                plan = plan_request(
                    prompt,
                    threshold,
                    active_local_model,
                    active_remote_model,
                )

            clf = plan.classification
            if clf.prompt_type == "DIRECT_ANSWER":
                st.caption("🎯 Direct answer — single agent, no decomposition.")
            elif clf.prompt_type == "REMOTE_ESCALATE":
                st.caption("☁️ Cloud escalation — single remote agent.")
            elif clf.decomposition_used:
                st.caption(f"🔀 Decomposed into {clf.num_agents} sub-agents.")

            use_swarm = (
                clf.decomposition_used
                and clf.num_agents > 1
                and not plan.single_route
                and len(plan.tasks) > 1
            )

            if use_swarm:
                with st.status("Spawning Agent Swarm...", expanded=True) as status:
                    status.write(f"Decomposed into **{len(plan.tasks)}** parallel sub-agents.")
                    for index, task in enumerate(plan.tasks, start=1):
                        preview = f"{task[:80]}{'…' if len(task) > 80 else ''}"
                        status.write(f"• Sub-agent {index}: {preview}")
                    result = execute_agent_swarm(
                        plan.tasks,
                        threshold,
                        api_key,
                        active_local_model,
                        active_remote_model,
                    )
                    result = _attach_orchestration(result, clf)
                    result.decomposition_used = True
                    result.num_agents = len(plan.tasks)
                    status.update(label="Agent Swarm complete", state="complete")
            else:
                with st.spinner("Routing..."):
                    single_prompt = prompt
                    result = route_and_execute(
                        single_prompt,
                        threshold,
                        api_key,
                        active_local_model,
                        active_remote_model,
                        image_file=uploaded_image,
                    )
                    result = _attach_orchestration(result, clf)

            assistant_message = {
                "role": "assistant",
                "content": result.answer,
                "result": result,
            }
            render_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
