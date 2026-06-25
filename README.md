# Hybrid Token-Efficient Routing Agent

AMD Hackathon project: a **middleware-first hybrid inference router** that minimizes remote API token spend by sending work to the cheapest capable backend—Python `eval`, local Ollama, or Fireworks AI—based on prompt shape, length, and modality.

The primary demo is a **Streamlit chatbot** in [`app.py`](app.py). A separate, library-style orchestrator lives in [`my_routing_agent/`](my_routing_agent/) for CLI use and deeper experimentation.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [CLI Orchestrator (Optional)](#cli-orchestrator-optional)
- [Routing & Middleware](#routing--middleware)
- [Demo Prompts](#demo-prompts)
- [Repository Layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

---

## What It Does

| Capability | Description |
|------------|-------------|
| **Smart math extraction** | Regex finds arithmetic embedded in natural language (e.g. `"What is 2 + 2?"`) and evaluates it locally with zero LLM tokens. |
| **Prompt distillation** | Long prompts bound for Fireworks are compressed first by local Ollama to strip filler and save remote tokens. |
| **Task decomposition & agent swarm** | Multi-question prompts are split by a local planner into explicit sub-tasks, then executed in parallel via `ThreadPoolExecutor`. |
| **Vision routing** | Sidebar image uploads route to Fireworks vision (`llama-v3p2-11b-vision-instruct`). |
| **Negative guardrails** | Local and remote LLM calls enforce terse, factual answers and immediate denial of flawed requests (e.g. “capital of London”). |
| **Telemetry UI** | Route, token usage, parallel latency, distillation report, and swarm decomposition are shown in the Streamlit dashboard. |

---

## Architecture

```text
User prompt
    │
    ▼
┌─────────────────────────────────────┐
│  Task Decomposition (local Ollama)  │  ← multi-question → JSON sub-task list
└─────────────────────────────────────┘
    │
    ├── len(tasks) > 1 ──► Agent Swarm (parallel route_and_execute per task)
    │
    └── single task ──► route_and_execute
                            │
                            ├── Image attached? ──► VISION_REMOTE (Fireworks)
                            ├── Math regex match? ──► MATH_PYTHON (eval, 0 tokens)
                            ├── len(prompt) ≤ threshold ──► TEXT_LOCAL (Ollama)
                            └── len(prompt) > threshold ──► distill ──► TEXT_REMOTE (Fireworks)
```

**Backends**

| Route | Backend | When |
|-------|---------|------|
| `MATH_PYTHON` | Python `eval` (sandboxed) | Embedded expression with `+ - * /` |
| `TEXT_LOCAL` | Ollama `llama3.2` | Short text (≤ char threshold) |
| `TEXT_REMOTE` | Fireworks `qwen2p5-72b-instruct` | Long text (after distillation) |
| `VISION_REMOTE` | Fireworks `llama-v3p2-11b-vision-instruct` | Image uploaded |
| `AGENT_SWARM` | Parallel sub-agents | Multiple decomposed tasks |

---

## Prerequisites

1. **Python 3.11+** (3.12 tested)
2. **Ollama** running locally with `llama3.2` pulled  
   Required for: local text inference, prompt distillation, and task decomposition.
3. **Fireworks AI API key**  
   Required for: long-text remote routes and vision. Math and short local routes work without it.

### Install Ollama and pull the model

```bash
# Install Ollama: https://ollama.com/download

# Start the server (if not already running)
ollama serve

# In another terminal, pull the model used by app.py
ollama pull llama3.2
```

Verify Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Hour-Meng/amd-hackathon-agent.git
cd amd-hackathon-agent

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Upgrade pip and install dependencies
pip install -U pip
pip install -r my_routing_agent/requirements.txt
```

**Dependencies** (`my_routing_agent/requirements.txt`):

- `streamlit` — web UI
- `requests` — Ollama & Fireworks HTTP calls
- `Pillow` — image downscaling for vision route
- `tiktoken` — token counting (CLI package)

---

## Configuration

### Streamlit app (`app.py`)

Most settings are in the **sidebar** at runtime:

| Setting | Default | Purpose |
|---------|---------|---------|
| **Fireworks API Key** | _(empty)_ | Required for `TEXT_REMOTE` and `VISION_REMOTE` |
| **Text Complexity Threshold** | 50 chars | Prompts longer than this go remote (after distillation) |
| **Image upload** | optional | Forces vision route on next message |

No `.env` file is required for the Streamlit demo—the API key is entered in the UI.

### CLI package (`my_routing_agent`)

The CLI reads environment variables (optional overrides):

```bash
export FIREWORKS_API_KEY="fw_..."
export LOCAL_LLM_BASE_URL="http://localhost:11434/v1"
export LOCAL_LLM_MODEL="llama3.2:3b"
export FIREWORKS_MODEL="accounts/fireworks/models/qwen3p7-max"
```

Get a Fireworks key at [https://fireworks.ai](https://fireworks.ai).

---

## Running the App

From the repository root with your virtual environment activated:

```bash
streamlit run app.py
```

Streamlit opens a browser tab (default `http://localhost:8501`).

**Minimum smoke test (no Fireworks key needed):**

```text
What is 2 + 2?
```

Expected: route `MATH_PYTHON`, answer `4`, 0 tokens burned.

**Full stack test (Ollama + Fireworks key in sidebar):**

```text
tell me the capital of france, london, paris, cambodia, 2+12
```

Expected: agent swarm with context-preserving sub-tasks, math handled locally, capitals answered tersely or denied when flawed.

---

## CLI Orchestrator (Optional)

The `my_routing_agent` package is a separate, config-driven pipeline (compressor → router → local/remote clients) with printed telemetry.

```bash
# From repo root, with venv active
python -m my_routing_agent.main "What is 17 * 23?"

# Force a destination
python -m my_routing_agent.main "Explain quantum tunneling" --force-remote
python -m my_routing_agent.main "Hello" --force-local

# Multimodal (image path)
python -m my_routing_agent.main "Describe this" --image path/to/photo.jpg
```

Set `FIREWORKS_API_KEY` in the environment for remote routes.

---

## Routing & Middleware

### Middleware execution order (single task)

1. **Vision** — if an image is attached in the sidebar  
2. **Embedded math** — regex `([\d\s\+\-\*\/\(\)\.]{3,})` with operator check; safe `eval` with `{"__builtins__": None}`  
3. **Length threshold** — short → local Ollama; long → distill then Fireworks  

### Task decomposition (multi-question)

Local Ollama acts as a **strict few-shot planner**. It must:

- Never emit bare nouns (`"Paris"`, `"Cambodia"`)  
- Repeat shared context on every sub-task (`"capital of France"`, not `"France"`)  
- Tag math as `"math: <expression>"`  

Example:

```text
Input:  tell me the capital of france, london, paris, cambodia, 2+12
Output: ["capital of France", "capital of London", "capital of Paris",
         "capital of Cambodia", "math: 2+12"]
```

### Agent swarm

When decomposition returns more than one task:

- `ThreadPoolExecutor` runs up to `min(len(tasks), 8)` workers in parallel  
- Each sub-task goes through `route_and_execute` independently  
- Failures in one thread do not crash the app  
- Dashboard latency reports **parallel wall-clock time** (slowest thread), not the sum of all threads  

### Prompt distillation

Before Fireworks text inference, local Ollama compresses the prompt. The **Middleware Telemetry** expander shows original vs distilled text and characters saved.

### LLM system constraints

All local and remote text/vision LLM calls receive a combined guardrail prompt:

- Data-extraction micro-service behavior (deny flawed facts in under 5 words)  
- Brutally concise answers, max ~15 words  
- No greetings or filler  

---

## Demo Prompts

| Prompt | Expected behavior |
|--------|-------------------|
| `What is 2 + 2?` | `MATH_PYTHON`, instant, 0 tokens |
| `Hello, how are you today?` | `TEXT_LOCAL` (short) |
| A paragraph > threshold chars | `TEXT_REMOTE` after distillation |
| `capital of Japan and who wrote Hamlet` | Agent swarm, 2 parallel sub-agents |
| `tell me the capital of france, london, paris, cambodia, 2+12` | Swarm + math local + terse/denial answers |
| Image + `"What is in this photo?"` | `VISION_REMOTE` |

---

## Repository Layout

```text
amd-hackathon-agent/
├── app.py                          # Streamlit demo (primary entry point)
├── README.md
├── my_routing_agent/
│   ├── main.py                     # CLI orchestrator
│   ├── config.py                   # Env-based configuration
│   ├── requirements.txt            # Python dependencies
│   ├── clients/
│   │   ├── local_client.py         # Ollama / OpenAI-compatible client
│   │   └── remote_client.py        # Fireworks client
│   ├── middleware/
│   │   └── compressor.py           # Input compression
│   ├── routers/
│   │   └── engine.py               # Tiered routing engine
│   └── utils/
│       └── tokenizer.py              # tiktoken wrapper
└── venv/                           # Local virtualenv (gitignored)
```

There is **no** root `requirements.txt`, `src/`, `tests/`, or `.env.example`—use `my_routing_agent/requirements.txt` and the sidebar / env vars above.

---

## Troubleshooting

### `Ollama is not running on localhost:11434`

```bash
ollama serve
ollama pull llama3.2
curl http://localhost:11434/api/tags
```

### `Fireworks API Key required`

Enter your key (`fw_...`) in the Streamlit sidebar, or export `FIREWORKS_API_KEY` for the CLI.

### `ModuleNotFoundError: streamlit` / `PIL` / `requests`

Activate the venv and reinstall:

```bash
source venv/bin/activate
pip install -r my_routing_agent/requirements.txt
```

### Task decomposition returns one task for a multi-question prompt

Ollama must be running. The planner uses `llama3.2` at `temperature: 0.0`. Retry with clearer comma-separated questions.

### Math not intercepted

The expression must include an operator (`+`, `-`, `*`, `/`) and digits. Bare numbers or years are not routed to `MATH_PYTHON`.

### Agent swarm feels slow

Parallel latency is dominated by the **slowest** sub-agent (often a remote call). Check per-sub-agent routes in **Middleware Telemetry**.

### `python` command not found

Use `python3`:

```bash
python3 -m venv venv
python3 -m my_routing_agent.main "test"
```

---

## License

Add license information here if applicable.
