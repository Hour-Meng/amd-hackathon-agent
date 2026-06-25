"""Hybrid Token-Efficient Routing Agent — Streamlit chatbot demo."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import BinaryIO, Literal

import requests
import streamlit as st
from PIL import Image

LOCAL_ENDPOINT = "http://localhost:11434/api/generate"
REMOTE_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
LOCAL_MODEL = "llama3.2"
REMOTE_TEXT_MODEL = "accounts/fireworks/models/qwen2p5-72b-instruct"
REMOTE_VISION_MODEL = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

RouteName = Literal["MATH_PYTHON", "VISION_REMOTE", "TEXT_LOCAL", "TEXT_REMOTE", "AGENT_SWARM"]

MATH_EXTRACT_PATTERN = re.compile(r"([\d\s\+\-\*\/\(\)\.]{3,})")
MATH_OPERATOR_PATTERN = re.compile(r"[\+\-\*\/]")

# Core guardrail enforced on every LLM endpoint (local + remote).
NEGATIVE_GUARDRAIL = (
    "You are a data-extraction micro-service, not a conversational assistant. "
    "If an instruction contains a factual error, contradiction, or misconception "
    "(e.g., asking for the capital of a city), deny it immediately in under 5 words. "
    "Do not explain, do not apologize, do not give background context. "
    "Example: 'Error: London is a city.'"
)

AGENT_SYSTEM_PROMPT = (
    "Respond to the user instruction under these constraints:\n"
    "- Be brutally concise.\n"
    "- No conversational filler, greetings, or introductory phrases.\n"
    "- If the request is factually flawed, state the error and stop.\n"
    "- Max limit: 15 words per answer."
)

# Single system prompt injected into both Ollama and Fireworks chat calls.
CORE_SYSTEM_PROMPT = f"{NEGATIVE_GUARDRAIL}\n\n{AGENT_SYSTEM_PROMPT}"

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
    distilled_prompt: str | None = None
    distillation_chars_saved: int = 0
    distillation_error: str | None = None
    wall_clock_ms: float | None = None
    sub_results: list[RouteResult] = field(default_factory=list)


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


def _ollama_generate(
    *,
    prompt: str,
    system: str | None = None,
    timeout: int = 120,
    options: dict[str, object] | None = None,
) -> tuple[str, int]:
    """Call local Ollama generate API; return response text and eval token count."""
    payload: dict[str, object] = {
        "model": LOCAL_MODEL,
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
    If a valid equation is found and safely evaluated, return MATH_PYTHON result.

    Highest-priority interceptor: any sub-task containing a real arithmetic
    expression (must include an operator) is computed locally for zero tokens.
    """
    for match in MATH_EXTRACT_PATTERN.finditer(prompt):
        candidate = match.group(1).strip()
        if len(candidate) < 3:
            continue
        # Require an actual operator so bare numbers/years are not mis-routed as math.
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
            )
        except ZeroDivisionError:
            latency_ms = (time.perf_counter() - started) * 1000.0
            return RouteResult(
                answer="⚠️ Division by zero.",
                route="MATH_PYTHON",
                tokens=0,
                latency_ms=latency_ms,
                original_prompt=prompt,
            )
        except Exception:
            continue
    return None


def distill_prompt(user_text: str) -> tuple[str, int, str | None]:
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
            system=DISTILL_SYSTEM_PROMPT,
            timeout=90,
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


def task_dispatcher(prompt: str) -> list[str]:
    """
    Use local Ollama to split a prompt into distinct questions.
    Falls back to a single-element array on any failure.
    """
    cleaned = prompt.strip()
    if not cleaned:
        return [cleaned]

    try:
        raw, _ = _ollama_generate(
            prompt=cleaned,
            system=TASK_DECOMPOSITION_SYSTEM,
            timeout=90,
            options={"temperature": 0.0, "top_p": 0.1},
        )
        return _parse_question_array(raw)
    except Exception:
        return [cleaned]


def route_and_execute(
    prompt: str,
    threshold: int,
    api_key: str,
    image_file: BinaryIO | None = None,
) -> RouteResult:
    """
    Evaluate input through middleware and execute exactly one of four routes.

    Middleware order: vision → embedded math → text (local / remote + distillation).
    """
    started = time.perf_counter()

    if image_file is not None:
        answer, route, tokens, latency_ms = _route_vision(prompt, api_key, image_file, started)
        return RouteResult(
            answer=answer,
            route=route,
            tokens=tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
        )

    math_result = safe_math_agent(prompt, started)
    if math_result is not None:
        return math_result

    if len(prompt) <= threshold:
        answer, route, tokens, latency_ms = _route_text_local(prompt, started)
        return RouteResult(
            answer=answer,
            route=route,
            tokens=tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
        )

    return _route_text_remote(prompt, api_key, started)


def execute_agent_swarm(
    tasks: list[str],
    threshold: int,
    api_key: str,
) -> RouteResult:
    """Run route_and_execute in parallel for each decomposed sub-question."""
    swarm_started = time.perf_counter()
    ordered: list[RouteResult | None] = [None] * len(tasks)

    # One worker per task (capped) so no sub-agent waits in a queue behind another.
    worker_count = max(1, min(len(tasks), 8))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(route_and_execute, task, threshold, api_key): index
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

    return RouteResult(
        answer="\n\n---\n\n".join(sections),
        route="AGENT_SWARM",
        tokens=total_tokens,
        # Parallel runtime: dominated by the slowest thread, not the sequential sum.
        latency_ms=wall_latency_ms,
        original_prompt=" | ".join(tasks),
        sub_results=sub_results,
        wall_clock_ms=wall_latency_ms,
    )


def process_user_request(
    prompt: str,
    threshold: int,
    api_key: str,
    image_file: BinaryIO | None = None,
) -> RouteResult:
    """Top-level orchestrator: decomposition → swarm or single pipeline."""
    if image_file is not None:
        return route_and_execute(prompt, threshold, api_key, image_file=image_file)

    tasks = task_dispatcher(prompt)
    if len(tasks) > 1:
        return execute_agent_swarm(tasks, threshold, api_key)

    single_prompt = tasks[0] if tasks else prompt
    return route_and_execute(single_prompt, threshold, api_key)


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
    started: float,
) -> tuple[str, RouteName, int, float]:
    payload = {
        "model": LOCAL_MODEL,
        "prompt": prompt,
        "system": CORE_SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 64},
    }

    try:
        response = requests.post(LOCAL_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        tokens = int(data.get("eval_count", 0))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return answer, "TEXT_LOCAL", tokens, latency_ms

    except requests.ConnectionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        warning = (
            "⚠️ **Ollama is not running on localhost:11434.**\n\n"
            "Start the server with `ollama serve`, then pull the model:\n"
            "```bash\nollama pull llama3.2\n```"
        )
        return warning, "TEXT_LOCAL", 0, latency_ms

    except requests.Timeout:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Local request timed out.", "TEXT_LOCAL", 0, latency_ms

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text[:400]
        return f"⚠️ Local inference error:\n\n{detail}", "TEXT_LOCAL", 0, latency_ms


def _route_text_remote(
    prompt: str,
    api_key: str,
    started: float,
) -> RouteResult:
    if not api_key or not api_key.strip():
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "❌ **Fireworks API Key required.**\n\n"
            "Enter your API key in the sidebar to route long prompts remotely."
        )
        return RouteResult(
            answer=message,
            route="TEXT_REMOTE",
            tokens=0,
            latency_ms=latency_ms,
            original_prompt=prompt,
        )

    distilled, distill_tokens, distill_error = distill_prompt(prompt)
    chars_saved = max(0, len(prompt) - len(distilled))

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REMOTE_TEXT_MODEL,
        "messages": [
            {"role": "system", "content": CORE_SYSTEM_PROMPT},
            {"role": "user", "content": distilled},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
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
        remote_tokens = int(data.get("usage", {}).get("total_tokens", 0))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer=answer,
            route="TEXT_REMOTE",
            tokens=remote_tokens + distill_tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
            distilled_prompt=distilled,
            distillation_chars_saved=chars_saved,
            distillation_error=distill_error,
        )

    except requests.ConnectionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer="⚠️ Could not reach Fireworks API.",
            route="TEXT_REMOTE",
            tokens=distill_tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
            distilled_prompt=distilled,
            distillation_chars_saved=chars_saved,
            distillation_error=distill_error,
        )

    except requests.Timeout:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer="⚠️ Remote request timed out.",
            route="TEXT_REMOTE",
            tokens=distill_tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
            distilled_prompt=distilled,
            distillation_chars_saved=chars_saved,
            distillation_error=distill_error,
        )

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text[:400]
        return RouteResult(
            answer=f"⚠️ Remote inference error:\n\n{detail}",
            route="TEXT_REMOTE",
            tokens=distill_tokens,
            latency_ms=latency_ms,
            original_prompt=prompt,
            distilled_prompt=distilled,
            distillation_chars_saved=chars_saved,
            distillation_error=distill_error,
        )


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
        "AGENT_SWARM": "🐝 AGENT_SWARM",
    }

    if result.route == "AGENT_SWARM" and result.sub_results:
        route_label, total_tokens, parallel_latency, agent_count = _aggregate_swarm_metrics(result)
        total_latency = parallel_latency
        burned = any(
            item.route in ("VISION_REMOTE", "TEXT_REMOTE") for item in result.sub_results
        )
        saved = any(
            item.route in ("MATH_PYTHON", "TEXT_LOCAL") for item in result.sub_results
        )
    else:
        route_label = route_labels.get(result.route, result.route)
        total_tokens = result.tokens
        total_latency = result.latency_ms
        burned = result.route in ("VISION_REMOTE", "TEXT_REMOTE")
        saved = result.route in ("MATH_PYTHON", "TEXT_LOCAL")
        agent_count = 0

    with col1:
        st.markdown(f"**Route**  \n{route_label}")

    with col2:
        if result.route == "AGENT_SWARM" and result.sub_results:
            if burned and saved:
                st.markdown(
                    f"**Token Usage**  \n🔥 {total_tokens} Total (swarm aggregate)"
                )
            elif burned:
                st.markdown(f"**Token Usage**  \n🔥 {total_tokens} Tokens Burned")
            else:
                st.markdown(f"**Token Usage**  \n✅ {total_tokens} Tokens Saved")
        elif saved:
            st.markdown(f"**Token Usage**  \n✅ {total_tokens} Tokens Saved")
        else:
            st.markdown(f"**Token Usage**  \n🔥 {total_tokens} Tokens Burned")

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
    """Show distillation and swarm middleware details."""
    sub_distillations = [
        sub for sub in result.sub_results if sub.distilled_prompt is not None
    ]
    has_distillation = result.distilled_prompt is not None or bool(sub_distillations)
    has_swarm = bool(result.sub_results)

    if not has_distillation and not has_swarm:
        return

    with st.expander("Middleware Telemetry"):
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
            st.caption("Parallel sub-agents executed via ThreadPoolExecutor (max_workers=3).")
            for index, sub in enumerate(result.sub_results, start=1):
                st.markdown(f"**Sub-Agent {index}** → `{sub.route}` · {sub.latency_ms:.1f} ms")
                st.code(sub.original_prompt, language=None)
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
        "Advanced middleware: embedded math extraction, prompt distillation, "
        "and parallel agent cloning for multi-question prompts."
    )

    init_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Fireworks API Key",
            type="password",
            placeholder="fw_...",
            help="Required for vision and long-text remote routes.",
        )

        threshold = st.slider(
            "Text Complexity Threshold (Chars)",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
        )
        st.caption(
            "If a prompt exceeds this length, it routes to the remote model "
            "(after local distillation)."
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
            "3. **Prompt distillation** → Fireworks\n"
            "4. **Length threshold** → local vs remote"
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
            tasks = [prompt] if uploaded_image is not None else task_dispatcher(prompt)
            use_swarm = uploaded_image is None and len(tasks) > 1

            if use_swarm:
                with st.status("Spawning Agent Swarm...", expanded=True) as status:
                    status.write(f"Decomposed into **{len(tasks)}** parallel sub-agents.")
                    for index, task in enumerate(tasks, start=1):
                        status.write(f"• Sub-agent {index}: {task[:80]}{'…' if len(task) > 80 else ''}")
                    result = execute_agent_swarm(tasks, threshold, api_key)
                    status.update(label="Agent Swarm complete", state="complete")
            else:
                with st.spinner("Routing..."):
                    result = process_user_request(
                        prompt,
                        threshold,
                        api_key,
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
