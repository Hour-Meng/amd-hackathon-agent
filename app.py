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
REMOTE_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"

# Defaults / catalogs surfaced in the sidebar.
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
    "You are a prompt compressor. Extract ONLY the core question or instruction from "
    "the user's text. Remove all conversational filler, greetings, or irrelevant context. "
    "Output nothing but the extracted question."
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
    retries: int = 0
    wall_clock_ms: float | None = None
    sub_results: list[RouteResult] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)


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

# Geo/civic/identity/encyclopedia facts that weak local models must not answer.
FACTUAL_RISK_GUARD_REASON = "factual-risk:weak-local"
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


@dataclass
class RouterDecision:
    """Authoritative routing decision produced before any backend is touched."""

    route: Literal["LOCAL", "REMOTE"]
    complexity_score: int
    reason: str
    model_id: str


def route_decision(
    prompt: str,
    threshold: int,
    *,
    has_image: bool,
    active_local_model: str,
    active_remote_model: str,
) -> RouterDecision:
    """
    Single source of truth for LOCAL vs REMOTE. The dispatcher MUST honor this.

    Priority: image → character-level guard → factual-risk (weak local) → complexity.
    """
    score = min(100, calculate_complexity(prompt))
    remote_model = normalize_model_id(active_remote_model) or DEFAULT_REMOTE_MODEL

    if has_image:
        return RouterDecision("REMOTE", 100, "vision:remote", remote_model)

    if is_character_level_task(prompt):
        return RouterDecision(
            "REMOTE", 100, CHARACTER_LEVEL_GUARD_REASON, remote_model
        )

    if (
        is_factual_risk_prompt(prompt)
        and is_weak_local_model(active_local_model)
        and not is_local_trivial_whitelisted(prompt)
    ):
        return RouterDecision("REMOTE", score, FACTUAL_RISK_GUARD_REASON, remote_model)

    if score > threshold:
        return RouterDecision("REMOTE", score, "complexity>threshold", remote_model)

    return RouterDecision("LOCAL", score, "low-complexity-local", active_local_model)


def _log_routing(prompt: str, decision: RouterDecision) -> None:
    preview = prompt.strip().replace("\n", " ")[:60]
    logger.info(
        "ROUTING decision route=%s model_id=%s score=%s reason=%s prompt=%r",
        decision.route,
        decision.model_id,
        decision.complexity_score,
        decision.reason,
        preview,
    )


def _log_executed(result: RouteResult) -> None:
    attempts = result.diagnostics.get("remote_attempts")
    logger.info(
        "EXECUTED backend=%s model_id=%s reason=%s fallback=%s retries=%s "
        "latency_ms=%.1f remote_attempts=%s",
        result.route,
        result.model_used,
        result.routing_reason,
        result.fallback_used,
        result.retries,
        result.latency_ms,
        attempts or [],
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


def distill_prompt(user_text: str, active_local_model: str) -> tuple[str, int, str | None]:
    """
    Compress a prompt via local Ollama before remote inference.
    Returns (distilled_text, local_tokens_used, error_message).
    """
    cleaned = user_text.strip()
    if not cleaned:
        return cleaned, 0, None

    try:
        distilled, tokens = _ollama_generate(
            prompt=cleaned,
            model=active_local_model,
            system=DISTILL_SYSTEM_PROMPT,
            timeout=90,
            options={"temperature": 0.0},
        )
        if not distilled:
            return cleaned, tokens, "Distillation returned empty output; using original prompt."
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


def task_dispatcher(prompt: str, active_local_model: str) -> list[str]:
    """
    Use local Ollama to split a prompt into distinct, context-preserving sub-tasks.
    Falls back to a single-element array on any failure.
    """
    cleaned = prompt.strip()
    if not cleaned:
        return [cleaned]

    try:
        raw, _ = _ollama_generate(
            prompt=cleaned,
            model=active_local_model,
            system=TASK_DECOMPOSITION_SYSTEM,
            timeout=90,
            options={"temperature": 0.0, "top_p": 0.1},
        )
        return _parse_question_array(raw)
    except Exception:
        return [cleaned]


@dataclass
class SwarmPlan:
    """Pre-swarm plan produced by the authoritative global router.

    `single_remote` means the whole prompt must be served as ONE remote task and
    the decomposer is NOT allowed to split it (character-level global override).
    """

    tasks: list[str]
    global_route: Literal["LOCAL", "REMOTE"]
    reason: str
    single_remote: bool


def plan_request(
    prompt: str,
    threshold: int,
    active_local_model: str,
    active_remote_model: str,
) -> SwarmPlan:
    """
    Authoritative pre-swarm routing. Runs a GLOBAL route_decision on the FULL
    prompt BEFORE any decomposition.

    If the full prompt contains character-level content, the entire request is
    pinned REMOTE and kept as a single task — the decomposer may not override the
    global route for the character-level portion. Otherwise the local planner is
    allowed to split the prompt into a sub-agent swarm.
    """
    decision = route_decision(
        prompt,
        threshold,
        has_image=False,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
    )

    if decision.reason == CHARACTER_LEVEL_GUARD_REASON:
        # Global character-level override: one remote task, no decomposition.
        return SwarmPlan(
            tasks=[prompt],
            global_route="REMOTE",
            reason=decision.reason,
            single_remote=True,
        )

    # Non-character-level: decomposition may reduce cost without breaking semantics.
    tasks = task_dispatcher(prompt, active_local_model)
    return SwarmPlan(
        tasks=tasks,
        global_route=decision.route,
        reason=decision.reason,
        single_remote=False,
    )


def route_and_execute(
    prompt: str,
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
    image_file: BinaryIO | None = None,
) -> RouteResult:
    """
    Produce an authoritative router decision, then execute the chosen backend.

    Router authority: if the decision is REMOTE, ALL local-model logic (including
    math eval and length/keyword heuristics) is bypassed and the remote model is
    always used. Character-level tasks are hard-routed REMOTE by the pre-router
    override. Local failures may rescue UP to remote, but a REMOTE decision is
    never executed by the local model.
    """
    started = time.perf_counter()
    has_image = image_file is not None

    decision = route_decision(
        prompt,
        threshold,
        has_image=has_image,
        active_local_model=active_local_model,
        active_remote_model=active_remote_model,
    )
    _log_routing(prompt, decision)

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
        # Preserve exact text for tokenization-sensitive tasks (no local distillation).
        preserve_text = decision.reason == CHARACTER_LEVEL_GUARD_REASON
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
        _log_executed(result)
        return result

    # LOCAL route only: deterministic math eval (0 tokens) is allowed here.
    math_result = safe_math_agent(prompt, started)
    if math_result is not None:
        math_result.routing_reason = "math:python-eval"
        math_result.complexity_score = decision.complexity_score
        _log_executed(math_result)
        return math_result

    result = _route_text_local(
        prompt, api_key, active_local_model, active_remote_model, started
    )
    result.routing_reason = decision.reason
    result.complexity_score = decision.complexity_score
    _log_executed(result)
    return result


def execute_agent_swarm(
    tasks: list[str],
    threshold: int,
    api_key: str,
    active_local_model: str,
    active_remote_model: str,
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
) -> RouteResult:
    """Top-level orchestrator: decomposition → swarm or single pipeline."""
    if image_file is not None:
        return route_and_execute(
            prompt,
            threshold,
            api_key,
            active_local_model,
            active_remote_model,
            image_file=image_file,
        )

    plan = plan_request(prompt, threshold, active_local_model, active_remote_model)

    # Character-level global override: serve the whole prompt as one remote task.
    if plan.single_remote:
        return route_and_execute(
            prompt, threshold, api_key, active_local_model, active_remote_model
        )

    if len(plan.tasks) > 1:
        return execute_agent_swarm(
            plan.tasks, threshold, api_key, active_local_model, active_remote_model
        )

    single_prompt = plan.tasks[0] if plan.tasks else prompt
    return route_and_execute(
        single_prompt, threshold, api_key, active_local_model, active_remote_model
    )


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
    Attempt local Ollama inference; on ANY failure (exception, non-200 status such
    as a 404 NOT_FOUND when the model isn't pulled, or an empty body) automatically
    reroute to the remote Fireworks endpoint using `active_remote_model`.
    """
    payload = {
        "model": active_local_model,
        "prompt": prompt,
        "system": SUB_AGENT_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128},
    }

    fallback_reason: str | None = None
    try:
        response = requests.post(LOCAL_ENDPOINT, json=payload, timeout=120)
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
        return RouteResult(
            answer=answer,
            route="TEXT_LOCAL",
            tokens=tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
            model_used=active_local_model,
        )

    except requests.ConnectionError:
        fallback_reason = "Ollama is not running on localhost:11434."
    except requests.Timeout:
        fallback_reason = "Local Ollama request timed out."
    except (requests.RequestException, ValueError) as exc:
        fallback_reason = f"Local model '{active_local_model}' error: {exc}"

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

    if skip_distillation:
        # Local backend is down — distillation (also local) would fail too.
        distilled = original_prompt
    else:
        distilled, distill_tokens, distill_error = distill_prompt(prompt, active_local_model)
        # Distillation fallback — never send empty/None text to the remote API.
        if not distilled or distilled.strip() == "":
            distilled = original_prompt
            if _has_ui_context():
                st.warning("Distillation returned empty. Falling back to original prompt.")

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
            diagnostics={"remote_attempts": remote_attempts},
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
                data = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
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
        "FALLBACK_REMOTE": "↩️ FALLBACK_REMOTE",
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
        if result.route == "FALLBACK_REMOTE" or (
            result.route == "AGENT_SWARM" and result.fallback_used
        ):
            st.caption("↩️ Fallback to Remote occurred")
        if result.model_used and result.route != "AGENT_SWARM":
            st.caption(f"Model: `{result.model_used}`")
        if result.routing_reason and result.route != "AGENT_SWARM":
            st.caption(f"Reason: {result.routing_reason}")

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

    st.title("🔀 Hybrid Token-Efficient Routing Agent")
    st.caption(
        "Advanced middleware: embedded math extraction, complexity-scored routing, "
        "prompt distillation, and parallel agent cloning with local→remote fallback."
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
            "Local Ollama Model",
            value=DEFAULT_LOCAL_MODEL,
            help="Ollama model used for local inference, distillation, and decomposition.",
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
            "Middleware stack:\n"
            "1. **Task decomposition** (multi-question → agent swarm)\n"
            "2. **Embedded math regex** → Python eval\n"
            "3. **Complexity scorer** → local vs remote\n"
            "4. **Local failure** → automatic Fireworks fallback\n"
            "5. **Prompt distillation** → Fireworks (empty-safe)"
        )

        if st.button("Clear Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

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
            # Authoritative pre-swarm routing: a global character-level prompt is
            # pinned REMOTE as a single task and the decomposer cannot override it.
            if uploaded_image is not None:
                plan = SwarmPlan([prompt], "REMOTE", "vision:remote", single_remote=False)
            else:
                plan = plan_request(
                    prompt, threshold, active_local_model, active_remote_model
                )

            if plan.single_remote:
                st.info(
                    "🔒 Character-level content detected — routing the full prompt "
                    "REMOTE as a single task (decomposition skipped)."
                )

            use_swarm = (
                uploaded_image is None
                and not plan.single_remote
                and len(plan.tasks) > 1
            )

            if use_swarm:
                with st.status("Spawning Agent Swarm...", expanded=True) as status:
                    status.write(f"Decomposed into **{len(plan.tasks)}** parallel sub-agents.")
                    for index, task in enumerate(plan.tasks, start=1):
                        preview = f"{task[:80]}{'…' if len(task) > 80 else ''}"
                        status.write(f"• Sub-agent {index}: {preview}")
                    result = execute_agent_swarm(
                        plan.tasks, threshold, api_key, active_local_model, active_remote_model
                    )
                    status.update(label="Agent Swarm complete", state="complete")
            else:
                with st.spinner("Routing..."):
                    single_prompt = prompt if plan.single_remote else (
                        plan.tasks[0] if plan.tasks else prompt
                    )
                    result = route_and_execute(
                        single_prompt,
                        threshold,
                        api_key,
                        active_local_model,
                        active_remote_model,
                        image_file=uploaded_image,
                    )

            assistant_message = {
                "role": "assistant",
                "content": result.answer,
                "result": result,
            }
            render_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
