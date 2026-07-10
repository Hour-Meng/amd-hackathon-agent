#!/usr/bin/env python3
"""Streamlit dashboard for the AMD test suite runner."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

os.environ.setdefault("SKIP_LOCAL", "true")

st.set_page_config(page_title="AMD Test Suite Dashboard", layout="wide")
st.title("🔬 AMD Test Suite — Live Runner")
st.caption("Load a test_cases.json file, run the suite, and inspect results interactively.")

# ── File upload ──

uploaded = st.sidebar.file_uploader("Upload test_cases.json", type=["json"])
sample_path = Path(__file__).parent / "test_cases.json"
if not sample_path.exists():
    sample_path = Path(__file__).parent / "test_cases_example.json"
if sample_path.exists() and st.sidebar.button("Use bundled test cases"):
    uploaded = sample_path.read_bytes()

# ── Settings ──

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Fireworks API Key", type="password",
                                 value=os.getenv("FIREWORKS_API_KEY", ""))
_default_models = os.getenv(
    "ALLOWED_MODELS",
    "accounts/fireworks/models/qwen3p7-plus,"
    "accounts/fireworks/models/minimax-m3",
)
model = st.sidebar.text_input(
    "Remote Models (comma-separated)",
    value=_default_models,
)
workers = st.sidebar.slider("Parallel Workers", 1, 8, 2)
auto_run = st.sidebar.checkbox("Run automatically on file load", value=True)

# ── Load cases ──

cases = []
if uploaded is not None:
    try:
        raw = json.loads(uploaded) if isinstance(uploaded, bytes) else json.load(uploaded)
        cases = list(raw)
        st.sidebar.success(f"Loaded {len(cases)} test case(s)")
    except Exception as e:
        st.sidebar.error(f"Failed to parse: {e}")

# ── Run ──

run_col, clear_col = st.columns([1, 5])
run_btn = run_col.button("▶ Run Suite", disabled=not cases or not api_key, type="primary")
clear_col.button("Clear Results")

if run_btn or (auto_run and cases and api_key and "results" not in st.session_state):
    if not api_key:
        st.error("Enter a Fireworks API key in the sidebar")
        st.stop()

    os.environ["FIREWORKS_API_KEY"] = api_key
    # Sidebar accepts a comma-separated allow-list; primary model is the first entry.
    model_parts = [part.strip() for part in (model or "").split(",") if part.strip()]
    if model_parts:
        os.environ["ALLOWED_MODELS"] = ",".join(model_parts)
    primary_model = model_parts[0] if model_parts else None

    from run_test_suite import (
        _configure_parallel_remote_limit,
        _ensure_multi_model_allowlist,
        run_suite,
        generate_report,
        print_terminal_report,
    )

    _ensure_multi_model_allowlist(primary_model)
    _configure_parallel_remote_limit(workers)

    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    results_container = st.container()

    status_text.info("Running test suite...")

    log_lines: list[str] = []

    def _on_progress(line: str) -> None:
        log_lines.append(line)
        pct = min(1.0, len(log_lines) / len(cases))
        progress_bar.progress(pct, text=f"{len(log_lines)}/{len(cases)} tasks")
        if len(log_lines) % 5 == 0 or pct >= 1.0:
            with results_container:
                st.code("\n".join(log_lines[-20:]), language="")

    results = run_suite(
        [type("", (), {"task_id": c.get("task_id", ""), "prompt": c.get("prompt", ""),
                        "category": c.get("category", ""), "difficulty": c.get("difficulty", ""),
                        "tags": c.get("tags", []), "expected_answer": c.get("expected_answer", ""),
                        "answer_type": c.get("answer_type", "")})() for c in cases],
        api_key=api_key,
        remote_model=primary_model,
        max_workers=workers,
        progress_callback=_on_progress,
    )

    progress_bar.progress(1.0, text="Done!")
    status_text.success(f"Completed {len(results)} tasks")

    report = generate_report(results)
    st.session_state["results"] = results
    st.session_state["report"] = report

# ── Display results ──

if "report" in st.session_state:
    report = st.session_state["report"]
    results = st.session_state["results"]

    # Summary metrics
    s = report.summary
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Success Rate", f"{s['success_rate']:.1f}%", f"{s['passed']}/{s['total']}")
    m2.metric("False Positives", s.get("false_positives", 0), "router ok, bad output")
    m3.metric("Total Tokens", f"{s['total_tokens']:,}")
    m4.metric("Avg Latency", f"{s['avg_latency_ms']:.0f} ms")
    m5.metric("Total Time", f"{s['total_latency_ms']/1000:.1f}s")
    m6.metric("Estimated Cost", f"${report.cost_estimate['estimated_cost_usd']:.4f}")

    # Route distribution
    st.subheader("Route Distribution")
    route_dist = s.get("route_distribution", {})
    if route_dist:
        total = sum(route_dist.values())
        cols = st.columns(len(route_dist))
        for i, (route, count) in enumerate(sorted(route_dist.items(), key=lambda x: -x[1])):
            cols[i].metric(route, count, f"{count/total*100:.1f}%")

    # Per-category table
    st.subheader("By Category")
    cat_data = []
    for cat, cs in sorted(report.by_category.items()):
        cat_data.append({
            "Category": cat.capitalize(),
            "Total": cs["total"],
            "Passed": cs["passed"],
            "Rate": f"{cs['success_rate']:.1f}%",
            "Avg Tokens": f"{cs['avg_tokens']:.0f}",
            "Total Tokens": cs["total_tokens"],
            "False Positives": cs.get("false_positives", 0),
            "Empty Answers": cs.get("empty_answers", 0),
            "Avg Latency": f"{cs['avg_latency_ms']:.0f} ms",
        })
    if cat_data:
        st.dataframe(cat_data, use_container_width=True, hide_index=True)

    # Per-difficulty
    st.subheader("By Difficulty")
    diff_data = []
    for diff, ds in sorted(report.by_difficulty.items()):
        diff_data.append({
            "Difficulty": diff.capitalize(),
            "Total": ds["total"],
            "Passed": ds["passed"],
            "Rate": f"{ds['success_rate']:.1f}%",
            "Avg Tokens": f"{ds['avg_tokens']:.0f}",
            "Avg Latency": f"{ds['avg_latency_ms']:.0f} ms",
        })
    if diff_data:
        st.dataframe(diff_data, use_container_width=True, hide_index=True)

    # Latency details
    st.subheader("Latency Summary")
    ls = report.latency_summary
    if ls:
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Fastest", f"{ls['fastest']['latency_ms']:.1f}ms", ls['fastest']['task_id'])
        lc2.metric("Median", f"{ls['median_ms']:.1f}ms")
        lc3.metric("P95", f"{ls['p95_ms']:.1f}ms")
        lc4.metric("Slowest", f"{ls['slowest']['latency_ms']:.1f}ms", ls['slowest']['task_id'])

    # Full results table
    st.subheader("All Results")
    if results:
        table = []
        for r in results:
            table.append({
                "ID": r.task_id,
                "Category": r.category,
                "Difficulty": r.difficulty,
                "Route": r.route,
                "Tokens": r.tokens,
                "Latency": f"{r.latency_ms:.0f}ms",
                "Success": "✓" if r.success else "✗",
                "Router OK": "✓" if r.router_success else "✗",
                "Failure": r.failure_reason[:40] if r.failure_reason else "",
                "Model": r.model_used[:20] if r.model_used else "",
            })
        st.dataframe(table, use_container_width=True, hide_index=True)

    # Failures
    if report.failures:
        st.subheader(f"Failures ({len(report.failures)})")
        fail_data = []
        for f in report.failures:
            fail_data.append({
                "ID": f["task_id"],
                "Category": f.get("category", ""),
                "Prompt": f["prompt"][:60],
                "Route": f["route"],
                "Model": f.get("model_used", "")[:24],
                "Tokens": f.get("tokens", 0),
                "Reason": f.get("failure_reason", ""),
                "Answer": (f.get("answer_preview") or "")[:80],
            })
        st.dataframe(fail_data, use_container_width=True, hide_index=True)

    # Download
    col1, col2 = st.columns(2)
    report_dict = {
        "summary": report.summary,
        "by_category": report.by_category,
        "by_difficulty": report.by_difficulty,
        "by_route": report.by_route,
        "failures": report.failures,
        "cost_estimate": report.cost_estimate,
        "latency_summary": report.latency_summary,
        "task_results": report.task_results,
    }
    col1.download_button("📥 Download JSON Report", json.dumps(report_dict, indent=2),
                          file_name="test_report.json", mime="application/json")

    from run_test_suite import task_result_to_dict
    raw_results = [task_result_to_dict(r) for r in results]
    col2.download_button("📥 Download Raw Results", json.dumps(raw_results, indent=2),
                          file_name="test_results_raw.json", mime="application/json")

elif not cases:
    st.info("Upload a test_cases.json file to get started, or run `python run_test_suite.py --generate 200`")
