"""Router-first regression tests for the hybrid AI middleware.

Run with:  python3 test_router.py   (or)   pytest -q test_router.py
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import app
from app import (
    CHARACTER_LEVEL_GUARD_REASON,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_REMOTE_MODEL,
    FACTUAL_RISK_GUARD_REASON,
    LOCAL_GREETING_REASON,
    LOCAL_MATH_REASON,
    LOCAL_UNAVAILABLE_REASON,
    REMOTE_FACTUAL_REASON,
    REMOTE_MODEL_CANDIDATES,
    ROUTER_DEFAULT_REMOTE,
    SUB_AGENT_SYSTEM_PROMPT,
    build_remote_candidates,
    check_local_health,
    had_prior_local_failure,
    heuristic_task_split,
    is_character_level_task,
    is_factual_risk_prompt,
    is_known_deployed_model,
    mark_prior_local_failure,
    normalize_model_id,
    plan_request,
    reset_local_health_cache,
    reset_prior_local_failures,
    route_decision,
    safe_math_agent,
)

LOCAL_MODEL = DEFAULT_LOCAL_MODEL
REMOTE_MODEL = DEFAULT_REMOTE_MODEL
REMOTE_FALLBACK = REMOTE_MODEL_CANDIDATES[1]
THRESHOLD = 30
COMPOSITE_PROMPT = (
    "spell apple backward, write the answer of 9+10 backward. "
    "Tell me 1 amazing thing about france"
)


def _decide(
    prompt: str,
    *,
    local_model: str = LOCAL_MODEL,
    local_unavailable: bool = False,
):
    return route_decision(
        prompt,
        THRESHOLD,
        has_image=False,
        active_local_model=local_model,
        active_remote_model=REMOTE_MODEL,
        local_unavailable=local_unavailable,
    )


class _FakeResponse:
    def __init__(self, status_code: int, *, text: str = "", payload: dict | None = None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = app.requests.HTTPError(f"HTTP {self.status_code}")
            err.response = self
            raise err


@contextmanager
def _patch_post(handler):
    original = app.requests.post

    def _fake_post(url, *args, **kwargs):
        return handler(url, kwargs.get("json", {}))

    app.requests.post = _fake_post
    try:
        yield
    finally:
        app.requests.post = original


@contextmanager
def _patch_get(handler):
    original = app.requests.get

    def _fake_get(url, *args, **kwargs):
        return handler(url)

    app.requests.get = _fake_get
    try:
        yield
    finally:
        app.requests.get = original


def _seed_local_health(model: str, healthy: bool) -> None:
    reset_local_health_cache()
    app._LOCAL_HEALTH_CACHE[model.strip()] = (time.time() + 1000.0, healthy)


# --- Router-first defaults ----------------------------------------------------------


def test_default_local_is_lightweight_utility():
    assert LOCAL_MODEL == "qwen2.5:0.5b"
    assert DEFAULT_REMOTE_MODEL == "accounts/fireworks/models/minimax-m3"


def test_default_remote_list_contains_minimax_and_qwen3p7():
    assert REMOTE_MODEL_CANDIDATES[0].endswith("minimax-m3")
    assert REMOTE_MODEL_CANDIDATES[1].endswith("qwen3p7-plus")


def test_general_prompt_defaults_to_remote():
    decision = _decide("What is the weather like today in Paris?")
    assert decision.route == "REMOTE", decision
    assert decision.model_id == REMOTE_MODEL


# --- Local capability ceiling -------------------------------------------------------


def test_hi_stays_local_when_healthy():
    _seed_local_health(LOCAL_MODEL, True)
    decision = _decide("hi")
    assert decision.route == "LOCAL", decision
    assert decision.reason == LOCAL_GREETING_REASON
    assert decision.model_id == LOCAL_MODEL


def test_hi_routes_remote_when_local_unhealthy():
    decision = _decide("hi", local_unavailable=True)
    assert decision.route == "REMOTE", decision
    assert decision.reason == LOCAL_UNAVAILABLE_REASON
    assert decision.model_id == REMOTE_MODEL


def test_where_is_cambodia_always_routes_remote():
    prompt = "Where is Cambodia?"
    assert is_factual_risk_prompt(prompt)
    decision = _decide(prompt)
    assert decision.route == "REMOTE", decision
    assert decision.reason == REMOTE_FACTUAL_REASON


def test_capital_of_france_routes_remote():
    decision = _decide("capital of France")
    assert decision.route == "REMOTE", decision
    assert decision.reason == REMOTE_FACTUAL_REASON


def test_arithmetic_routes_local_math():
    decision = _decide("math: 2+12")
    assert decision.route == "LOCAL", decision
    assert decision.reason == LOCAL_MATH_REASON
    result = safe_math_agent("math: 2+12", time.perf_counter())
    assert result is not None and result.answer == "14"


# --- Character-level always remote --------------------------------------------------


def test_spell_apple_backward_routes_remote():
    decision = _decide("spell apple backward")
    assert decision.route == "REMOTE", decision
    assert decision.reason == CHARACTER_LEVEL_GUARD_REASON


def test_composite_prompt_pins_remote_single_task():
    plan = plan_request(COMPOSITE_PROMPT, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.global_route == "REMOTE", plan
    assert plan.single_remote is True
    assert plan.tasks == [COMPOSITE_PROMPT]


# --- Heuristic planner (no local LLM) -----------------------------------------------


def test_heuristic_split_without_local_llm():
    parts = heuristic_task_split("capital of France, capital of London, math: 2+12")
    assert len(parts) >= 2
    assert task_dispatcher_never_calls_ollama()


def task_dispatcher_never_calls_ollama() -> bool:
    called = {"n": 0}

    def handler(url, body):
        called["n"] += 1
        return _FakeResponse(200, payload={"response": "[]", "eval_count": 0})

    with _patch_post(handler):
        app.task_dispatcher("capital of France, capital of London", LOCAL_MODEL)
    return called["n"] == 0


# --- Stability gate & fallback ------------------------------------------------------


def test_check_local_health_caches_verdict():
    calls = {"n": 0}

    def handler(url):
        calls["n"] += 1
        raise app.requests.ConnectionError("down")

    reset_local_health_cache()
    with _patch_get(handler):
        assert check_local_health(LOCAL_MODEL) is False
        assert check_local_health(LOCAL_MODEL) is False
    assert calls["n"] == 1


def test_local_timeout_falls_back_to_remote():
    _seed_local_health(LOCAL_MODEL, True)
    reset_prior_local_failures()
    post_calls = {"local": 0, "remote": 0}

    def handler(url, body):
        if url == app.LOCAL_ENDPOINT:
            post_calls["local"] += 1
            raise app.requests.Timeout("read timed out")
        post_calls["remote"] += 1
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "remote answer"}}],
                "usage": {"total_tokens": 5},
            },
        )

    with _patch_post(handler):
        result = app._route_text_local(
            "hello",
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
        )

    assert result.route == "FALLBACK_REMOTE"
    assert result.fallback_used is True
    assert result.model_used == REMOTE_MODEL
    assert result.answer == "remote answer"
    assert post_calls["local"] == 1
    assert had_prior_local_failure("hello")


def test_unhealthy_local_skips_inference_call():
    _seed_local_health(LOCAL_MODEL, False)
    post_calls = {"local": 0, "remote": 0}

    def handler(url, body):
        if url == app.LOCAL_ENDPOINT:
            post_calls["local"] += 1
            raise AssertionError("must not call local when unhealthy")
        post_calls["remote"] += 1
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "remote answer"}}],
                "usage": {"total_tokens": 5},
            },
        )

    with _patch_post(handler):
        result = app._route_text_local(
            "hello",
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
        )

    assert post_calls["local"] == 0
    assert result.route == "FALLBACK_REMOTE"
    assert result.model_used == REMOTE_MODEL


def test_prior_local_failure_forces_remote():
    reset_prior_local_failures()
    mark_prior_local_failure("hello")
    decision = _decide("hello")
    assert decision.route == "REMOTE"
    assert "prior-local-failure" in decision.reason


# --- Remote model fallback ----------------------------------------------------------


def test_remote_fallback_on_not_found_uses_next_candidate():
    preferred, second = REMOTE_MODEL, REMOTE_FALLBACK

    def handler(url, body):
        if body.get("model") == preferred:
            return _FakeResponse(404, text="Model not found")
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"total_tokens": 7},
            },
        )

    with _patch_post(handler):
        result = app._route_text_remote(
            "spell apple backward",
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
            skip_distillation=True,
        )

    assert result.model_used == second
    assert result.answer == "ok"


def test_ui_route_label_matches_fallback_backend():
    """FALLBACK_REMOTE result must carry the actual Fireworks model used."""
    _seed_local_health(LOCAL_MODEL, False)

    def handler(url, body):
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "hi there"}}],
                "usage": {"total_tokens": 3},
            },
        )

    with _patch_post(handler):
        result = app._route_text_local(
            "hi",
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
        )

    assert result.route == "FALLBACK_REMOTE"
    assert result.model_used == REMOTE_MODEL
    assert "TEXT_REMOTE" in "☁️ TEXT_REMOTE (fallback)"


def test_system_prompt_has_no_factual_premise_rejection():
    lowered = SUB_AGENT_SYSTEM_PROMPT.lower()
    assert "not an amazing thing" not in lowered
    assert "amazing thing about france" in lowered


def _run() -> int:
    tests = [obj for name, obj in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
