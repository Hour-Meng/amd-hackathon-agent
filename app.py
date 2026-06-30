"""Single-Pass Optimized Remote LLM Pipeline — Streamlit chatbot demo."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import BinaryIO, Literal

import requests
import streamlit as st
from PIL import Image

from my_routing_agent.cache.semantic_cache import SemanticCache
from my_routing_agent.utils.tokenizer import estimate_tokens

logger = logging.getLogger("single_pass_pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

REMOTE_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
REMOTE_VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"
REMOTE_MODEL_CANDIDATES = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/qwen3p7-plus",
]
DEFAULT_LOCAL_MODEL = "qwen2.5:0.5b"
DEFAULT_REMOTE_MODEL = REMOTE_MODEL_CANDIDATES[0]
FIREWORKS_MODEL_PREFIX = "accounts/fireworks/models/"
KNOWN_DEPLOYED_REMOTE_MODELS = frozenset({
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/qwen3p7-plus",
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
})
CUSTOM_MODEL_SENTINEL = "Custom..."
REMOTE_MODEL_OPTIONS = [
    "accounts/fireworks/models/minimax-m3",
    "accounts/fireworks/models/qwen3p7-plus",
    CUSTOM_MODEL_SENTINEL,
]
MODEL_UNAVAILABLE_MARKERS = (
    "not_found", "not found", "does not exist", "not deployed",
    "inaccessible", "invalid model", "no such model", "unknown model",
    "model not found",
)

# Output structure enforcement
STRUCTURED_SYSTEM_PROMPT = """You are a concise, accurate assistant. Always respond in this exact format:

Summary
<one or two sentences summarizing the answer>

Key Points
- <point 1>
- <point 2>
- <point 3>
(max 5 bullet points)

Final Answer
<the complete direct answer>

Rules:
- No greetings or fluff.
- For math, output the calculation and result.
- For code, output the code block in Final Answer.
- If the prompt is a greeting or trivial, you may omit Key Points."""

MAX_CONTEXT_MESSAGES = 4  # keep only last 4 messages for context


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
_CACHE: SemanticCache | None = None


def init_cache() -> None:
    global _CACHE
    if _CACHE is None:
        cache = SemanticCache()
        if cache.initialize():
            _CACHE = cache


# ---------------------------------------------------------------------------
# Prompt Compression
# ---------------------------------------------------------------------------
FILLER_PATTERNS = re.compile(
    r"\b(?:please|kindly|note that|for your information|fyi|just|simply|basically|"
    r"actually|honestly|i was wondering if you could|"
    r"i would like to know|could you please|can you please|could you tell me)\b",
    re.IGNORECASE,
)
WHITESPACE_COLLAPSE = re.compile(r"\s+")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def compress_prompt(prompt: str) -> str:
    """Aggressively compress prompt: remove filler, collapse whitespace."""
    text = prompt.strip()
    text = FILLER_PATTERNS.sub("", text)
    text = WHITESPACE_COLLAPSE.sub(" ", text)
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Context Trimming
# ---------------------------------------------------------------------------
def trim_context(messages: list[dict]) -> list[dict]:
    """Keep only the last N messages to bound token usage."""
    if len(messages) <= MAX_CONTEXT_MESSAGES:
        return messages
    return messages[-MAX_CONTEXT_MESSAGES:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_model_id(model_id: str) -> str:
    if not model_id:
        return ""
    mid = model_id.strip().strip("/")
    if not mid:
        return ""
    if mid.startswith("accounts/") and "/models/" in mid:
        return mid
    return f"{FIREWORKS_MODEL_PREFIX}{mid.split('/')[-1]}"


def is_known_deployed_model(model_id: str) -> bool:
    return normalize_model_id(model_id) in KNOWN_DEPLOYED_REMOTE_MODELS


def build_remote_candidates(selected_model: str) -> list[str]:
    selected = normalize_model_id(selected_model)
    ordered: list[str] = []
    if selected and is_known_deployed_model(selected):
        ordered.append(selected)
    for cand in REMOTE_MODEL_CANDIDATES:
        normalized = normalize_model_id(cand)
        if normalized and normalized not in ordered and is_known_deployed_model(normalized):
            ordered.append(normalized)
    if selected and selected not in ordered:
        ordered.append(selected)
    return ordered


def _is_model_unavailable(status_code: int, body: str) -> bool:
    if status_code == 404:
        return True
    if status_code in (400, 403):
        low = (body or "").lower()
        return any(marker in low for marker in MODEL_UNAVAILABLE_MARKERS)
    return False


def _extract_remote_answer(data: dict) -> str | None:
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


def compress_image_to_base64(image_file: BinaryIO) -> str:
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((512, 512))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Remote LLM Call (single-pass)
# ---------------------------------------------------------------------------
@dataclass
class RouteResult:
    answer: str
    tokens: int
    latency_ms: float
    model_used: str = ""
    original_prompt: str = ""
    compressed_prompt: str = ""
    cache_hit: bool = False


def call_remote_llm(
    prompt: str,
    api_key: str,
    active_remote_model: str,
    *,
    image_file: BinaryIO | None = None,
    system_prompt: str | None = None,
) -> RouteResult:
    """Single-pass remote LLM call with compression, caching, and structured output."""
    started = time.perf_counter()

    # --- Step 1: Compress prompt ---
    compressed = compress_prompt(prompt)
    chars_saved = len(prompt) - len(compressed)
    if chars_saved > 0:
        logger.info("Compression saved %d chars", chars_saved)

    # --- Step 2: Cache lookup (exact + semantic) ---
    global _CACHE
    if _CACHE:
        entry = _CACHE.lookup(compressed or prompt)
        if entry is not None:
            latency_ms = (time.perf_counter() - started) * 1000.0
            logger.info("CACHE HIT — 0 tokens used")
            return RouteResult(
                answer=entry.response,
                tokens=0,
                latency_ms=latency_ms,
                model_used="cache",
                original_prompt=prompt,
                compressed_prompt=compressed,
                cache_hit=True,
            )

    if not api_key or not api_key.strip():
        latency_ms = (time.perf_counter() - started) * 1000.0
        return RouteResult(
            answer="❌ **Fireworks API Key required.** Enter your API key in the sidebar.",
            tokens=0,
            latency_ms=latency_ms,
            model_used="",
            original_prompt=prompt,
        )

    # --- Step 3: Build payload ---
    sys_prompt = system_prompt or STRUCTURED_SYSTEM_PROMPT
    user_content: str | list = compressed or prompt

    if image_file is not None:
        image_file.seek(0)
        data_uri = compress_image_to_base64(image_file)
        user_content = [
            {"type": "text", "text": compressed or prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }

    candidates = build_remote_candidates(active_remote_model)
    if not candidates:
        candidates = [normalize_model_id(active_remote_model) or active_remote_model]

    max_attempts = 2
    last_detail = "unknown error"
    remote_attempts: list[dict[str, str]] = []

    for model_id in candidates:
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }

        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    REMOTE_ENDPOINT, headers=headers, json=payload, timeout=180,
                )
                if response.status_code >= 500:
                    last_detail = f"HTTP {response.status_code} (transient)"
                    logger.warning("Remote 5xx for %s attempt %s", model_id, attempt + 1)
                    continue
                if _is_model_unavailable(response.status_code, response.text):
                    last_detail = f"HTTP {response.status_code}: {response.text[:200]}"
                    break
                response.raise_for_status()
                data = response.json()
                answer = _extract_remote_answer(data)
                if not answer:
                    last_detail = "Remote response missing content"
                    break
                tokens = int(data.get("usage", {}).get("total_tokens", 0))
                latency_ms = (time.perf_counter() - started) * 1000.0

                # Cache the result
                if _CACHE:
                    _CACHE.store(compressed or prompt, answer)

                return RouteResult(
                    answer=answer,
                    tokens=tokens,
                    latency_ms=latency_ms,
                    model_used=model_id,
                    original_prompt=prompt,
                    compressed_prompt=compressed,
                )

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_detail = str(exc)
                logger.warning("Remote transient error %s attempt %s", model_id, attempt + 1)
                continue
            except requests.RequestException as exc:
                detail = str(exc)
                status_code = exc.response.status_code if exc.response is not None else 0
                if exc.response is not None:
                    detail = exc.response.text[:400]
                last_detail = detail
                if _is_model_unavailable(status_code, detail):
                    break
                remote_attempts.append({"model_id": model_id, "status": "error", "detail": detail})
                return RouteResult(
                    answer=f"⚠️ Remote inference error ({model_id}):\n\n{detail}",
                    tokens=0,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    model_used=model_id,
                    original_prompt=prompt,
                )

    attempted = ", ".join(a["model_id"] for a in remote_attempts) or "none"
    logger.error("All remote models failed. Attempted: %s", attempted)
    return RouteResult(
        answer="⚠️ **All remote models failed.**\n\n"
               f"Attempted: {attempted}\n\nLast error: {last_detail}",
        tokens=0,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        model_used=candidates[-1] if candidates else active_remote_model,
        original_prompt=prompt,
    )


# ---------------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------------
def render_metrics(result: RouteResult) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        route_label = "💾 CACHE_HIT" if result.cache_hit else "☁️ REMOTE_LLM"
        st.markdown(f"**Route**  \n{route_label}")
        if result.model_used:
            st.caption(f"Model: `{result.model_used}`")
    with col2:
        if result.cache_hit:
            st.markdown("**Token Usage**  \n✅ 0 Tokens (cached)")
        else:
            st.markdown(f"**Token Usage**  \n🔥 {result.tokens} Tokens")
    with col3:
        st.markdown(f"**Latency**  \n⏱️ {result.latency_ms:.1f} ms")
    if result.compressed_prompt and result.compressed_prompt != result.original_prompt:
        chars_saved = len(result.original_prompt) - len(result.compressed_prompt)
        st.caption(f"Prompt compressed: saved {chars_saved} chars")


def render_assistant_message(message: dict) -> None:
    result = message["result"]
    render_metrics(result)
    st.markdown(message["content"])


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Hybrid Routing Agent",
        page_icon="⚡",
        layout="wide",
    )

    st.title("🔀 ANGKOR Router + PHANTOM Layer")
    st.caption("Compress → Cache → Single LLM Call → Structured Output")

    init_session_state()
    init_cache()

    with st.sidebar:
        st.header("⚙️ ANGKOR + PHANTOM Configuration")

        api_key = st.text_input(
            "Fireworks API Key",
            type="password",
            placeholder="fw_...",
            help="Required for remote, vision, and fallback routes.",
        )

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
        if _CACHE:
            _CACHE._config = _CACHE._config.__class__(
                threshold=cache_threshold,
                model_name=_CACHE._config.model_name,
                index_path=_CACHE._config.index_path,
                store_path=_CACHE._config.store_path,
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

        st.metric("Adaptive θ", f"{0.65:.3f}", delta=None)

        uploaded_image = st.file_uploader(
            "Upload Image (optional)",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif"],
            help="When attached, the next message routes through the vision pipeline.",
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Attached image preview", use_container_width=True)

        st.divider()

        st.info(
            "**ANGKOR + PHANTOM Pipeline:**\n"
            "T0: FAISS Cache → T1: Compress → T2: 3-Zone Router → "
            "PHANTOM: Entropy Abort + Speculative Race + Budget Enforcement → "
            "T3: Cascade Verify"
        )

        if _CACHE:
            st.caption(f"Cache: {_CACHE.size} entries · {_CACHE.hit_rate:.1%} hit rate")

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
            with st.spinner("Calling remote LLM..."):
                # Trim conversation history before the call
                trimmed = trim_context(st.session_state.messages[:-1])
                context = ""
                if trimmed:
                    context_parts = []
                    for msg in trimmed:
                        r = msg.get("role", "")
                        c = msg.get("content", "")[:200]
                        context_parts.append(f"{r}: {c}")
                    context = "\n".join(context_parts) + "\n---\n"

                full_prompt = context + prompt if context else prompt
                result = call_remote_llm(
                    full_prompt,
                    api_key,
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
