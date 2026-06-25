"""Hybrid Token-Efficient Routing Agent — Streamlit chatbot demo."""

from __future__ import annotations

import base64
import io
import re
import time
from typing import BinaryIO, Literal

import requests
import streamlit as st
from PIL import Image

LOCAL_ENDPOINT = "http://localhost:11434/api/generate"
REMOTE_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
LOCAL_MODEL = "llama3.2"
REMOTE_TEXT_MODEL = "accounts/fireworks/models/qwen2p5-72b-instruct"
REMOTE_VISION_MODEL = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"

RouteName = Literal["MATH_PYTHON", "VISION_REMOTE", "TEXT_LOCAL", "TEXT_REMOTE"]
MATH_PATTERN = re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$")


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


def is_math_expression(prompt: str) -> bool:
    cleaned = prompt.strip()
    return bool(cleaned) and bool(MATH_PATTERN.match(cleaned))


def route_and_execute(
    prompt: str,
    threshold: int,
    api_key: str,
    image_file: BinaryIO | None = None,
) -> tuple[str, RouteName, int, float]:
    """
    Evaluate input through middleware and execute exactly one of four routes.

    Returns:
        answer, route_name, tokens_used, latency_ms
    """
    started = time.perf_counter()

    if image_file is not None:
        return _route_vision(prompt, api_key, image_file, started)

    if is_math_expression(prompt):
        return _route_math(prompt, started)

    if len(prompt) <= threshold:
        return _route_text_local(prompt, started)

    return _route_text_remote(prompt, api_key, started)


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


def _route_math(
    prompt: str,
    started: float,
) -> tuple[str, RouteName, int, float]:
    cleaned = prompt.strip()
    try:
        result = eval(cleaned, {"__builtins__": None}, {})  # noqa: S307
        latency_ms = (time.perf_counter() - started) * 1000.0
        return str(result), "MATH_PYTHON", 0, latency_ms
    except ZeroDivisionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Division by zero.", "MATH_PYTHON", 0, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return f"⚠️ Math evaluation failed: {exc}", "MATH_PYTHON", 0, latency_ms


def _route_text_local(
    prompt: str,
    started: float,
) -> tuple[str, RouteName, int, float]:
    payload = {"model": LOCAL_MODEL, "prompt": prompt, "stream": False}

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
) -> tuple[str, RouteName, int, float]:
    if not api_key or not api_key.strip():
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "❌ **Fireworks API Key required.**\n\n"
            "Enter your API key in the sidebar to route long prompts remotely."
        )
        return message, "TEXT_REMOTE", 0, latency_ms

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REMOTE_TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
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
        return answer, "TEXT_REMOTE", tokens, latency_ms

    except requests.ConnectionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Could not reach Fireworks API.", "TEXT_REMOTE", 0, latency_ms

    except requests.Timeout:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Remote request timed out.", "TEXT_REMOTE", 0, latency_ms

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text[:400]
        return f"⚠️ Remote inference error:\n\n{detail}", "TEXT_REMOTE", 0, latency_ms


def render_metrics(route: RouteName, tokens: int, latency_ms: float) -> None:
    """Render hackathon demo metrics in three columns above the answer."""
    col1, col2, col3 = st.columns(3)

    route_labels = {
        "MATH_PYTHON": "🧮 MATH_PYTHON",
        "VISION_REMOTE": "👁️ VISION_REMOTE",
        "TEXT_LOCAL": "💻 TEXT_LOCAL",
        "TEXT_REMOTE": "☁️ TEXT_REMOTE",
    }

    with col1:
        st.markdown(f"**Route**  \n{route_labels.get(route, route)}")

    with col2:
        if route in ("MATH_PYTHON", "TEXT_LOCAL"):
            st.markdown(f"**Token Usage**  \n✅ {tokens} Tokens Saved")
        else:
            st.markdown(f"**Token Usage**  \n🔥 {tokens} Tokens Burned")

    with col3:
        st.markdown(f"**Latency**  \n⏱️ {latency_ms:.1f} ms")


def render_assistant_message(message: dict) -> None:
    render_metrics(message["route"], message["tokens"], message["latency_ms"])
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
        "Middleware-first routing: images → vision, math → Python, short text → Ollama, "
        "long text → Fireworks."
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
            "If a prompt exceeds this length, it routes to the remote model. "
            "Otherwise, it routes locally."
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
            f"Routing priority: **Image** → **Math** → **Text ≤ {threshold}** (local) "
            f"→ **Text > {threshold}** (remote)"
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
            with st.spinner("Routing..."):
                answer, route, tokens, latency_ms = route_and_execute(
                    prompt,
                    threshold,
                    api_key,
                    image_file=uploaded_image,
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "route": route,
                "tokens": tokens,
                "latency_ms": latency_ms,
            }
            render_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
