"""Hybrid Token-Efficient Routing Agent — Streamlit chatbot demo."""

from __future__ import annotations

import time
from typing import Literal

import requests
import streamlit as st

LOCAL_ENDPOINT = "http://localhost:11434/api/generate"
REMOTE_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
LOCAL_MODEL = "llama3.2"
REMOTE_MODEL = "accounts/fireworks/models/qwen2p5-72b-instruct"

Destination = Literal["LOCAL", "REMOTE"]


def route_and_execute(
    prompt: str,
    threshold: int,
    api_key: str,
) -> tuple[str, Destination, int, float]:
    """
    Route a prompt by character length and execute against the chosen backend.

    Returns:
        answer: Model response or user-facing error/warning text
        destination: "LOCAL" or "REMOTE"
        tokens: eval_count (local) or usage.total_tokens (remote)
        latency_ms: End-to-end execution time in milliseconds
    """
    started = time.perf_counter()

    if len(prompt) <= threshold:
        return _execute_local(prompt, started)

    return _execute_remote(prompt, api_key, started)


def _execute_local(
    prompt: str,
    started: float,
) -> tuple[str, Destination, int, float]:
    payload = {"model": LOCAL_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(LOCAL_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("response", "").strip()
        tokens = int(data.get("eval_count", 0))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return answer, "LOCAL", tokens, latency_ms

    except requests.ConnectionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        warning = (
            "⚠️ **Ollama is not running on localhost:11434.**\n\n"
            "Start the server with `ollama serve`, then pull the model:\n"
            "```bash\nollama pull llama3.2\n```"
        )
        return warning, "LOCAL", 0, latency_ms

    except requests.Timeout:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Local request timed out. Try a shorter prompt.", "LOCAL", 0, latency_ms

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text[:400]
        return f"⚠️ Local inference error:\n\n{detail}", "LOCAL", 0, latency_ms


def _execute_remote(
    prompt: str,
    api_key: str,
    started: float,
) -> tuple[str, Destination, int, float]:
    if not api_key or not api_key.strip():
        latency_ms = (time.perf_counter() - started) * 1000.0
        message = (
            "❌ **Fireworks API Key required.**\n\n"
            "Enter your API key in the sidebar to route long prompts to the remote model."
        )
        return message, "REMOTE", 0, latency_ms

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REMOTE_MODEL,
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
        return answer, "REMOTE", tokens, latency_ms

    except requests.ConnectionError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Could not reach Fireworks API. Check your network connection.", "REMOTE", 0, latency_ms

    except requests.Timeout:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "⚠️ Remote request timed out.", "REMOTE", 0, latency_ms

    except requests.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text[:400]
        return f"⚠️ Remote inference error:\n\n{detail}", "REMOTE", 0, latency_ms


def render_metrics(destination: Destination, tokens: int, latency_ms: float) -> None:
    """Render hackathon demo metrics in three columns above the answer."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if destination == "LOCAL":
            st.markdown("**Destination**  \n💻 Local (CPU)")
        else:
            st.markdown("**Destination**  \n☁️ Remote (Fireworks)")

    with col2:
        if destination == "LOCAL":
            st.markdown(f"**Token Usage**  \n✅ {tokens} Tokens Saved")
        else:
            st.markdown(f"**Token Usage**  \n🔥 {tokens} Tokens Burned")

    with col3:
        st.markdown(f"**Latency**  \n⏱️ {latency_ms:.1f} ms")


def render_assistant_message(message: dict) -> None:
    render_metrics(message["destination"], message["tokens"], message["latency_ms"])
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
        "Short prompts run locally on Ollama (free). Long prompts route to Fireworks AI (paid)."
    )

    init_session_state()

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Fireworks API Key",
            type="password",
            placeholder="fw_...",
            help="Required when prompts exceed the complexity threshold.",
        )

        threshold = st.slider(
            "Complexity Threshold (Character count)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
        )
        st.caption(
            "If a prompt exceeds this length, it routes to the remote model. "
            "Otherwise, it routes locally."
        )

        st.divider()

        route_hint = "💻 Local (Ollama)" if threshold >= 10 else "☁️ Remote"
        st.info(f"Prompts ≤ **{threshold}** chars → {route_hint}")

        if st.button("Clear Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing..."):
                answer, destination, tokens, latency_ms = route_and_execute(
                    prompt,
                    threshold,
                    api_key,
                )

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "destination": destination,
                "tokens": tokens,
                "latency_ms": latency_ms,
            }
            render_assistant_message(assistant_message)
            st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()
