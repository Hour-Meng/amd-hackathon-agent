"""Routing-authority regression tests for the hybrid LLM router.

Run with:  python3 test_router.py   (or)   pytest -q test_router.py
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import app
from app import (
    CHARACTER_LEVEL_GUARD_REASON,
    DEFAULT_REMOTE_MODEL,
    FACTUAL_RISK_GUARD_REASON,
    REMOTE_MODEL_CANDIDATES,
    SUB_AGENT_SYSTEM_PROMPT,
    build_remote_candidates,
    is_character_level_task,
    is_factual_risk_prompt,
    is_known_deployed_model,
    normalize_model_id,
    plan_request,
    route_decision,
    safe_math_agent,
)

LOCAL_MODEL = "qwen2.5:0.5b"
REMOTE_MODEL = DEFAULT_REMOTE_MODEL
REMOTE_FALLBACK = REMOTE_MODEL_CANDIDATES[1]
THRESHOLD = 30
COMPOSITE_PROMPT = (
    "spell apple backward, write the answer of 9+10 backward. "
    "Tell me 1 amazing thing about france"
)


def _decide(prompt: str):
    return route_decision(
        prompt,
        THRESHOLD,
        has_image=False,
        active_local_model=LOCAL_MODEL,
        active_remote_model=REMOTE_MODEL,
    )


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

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


# --- Default remote model list ------------------------------------------------------


def test_default_remote_list_contains_minimax_and_qwen3p7():
    assert REMOTE_MODEL_CANDIDATES[0] == "accounts/fireworks/models/minimax-m3"
    assert REMOTE_MODEL_CANDIDATES[1] == "accounts/fireworks/models/qwen3p7-plus"
    assert DEFAULT_REMOTE_MODEL == REMOTE_MODEL_CANDIDATES[0]
    assert is_known_deployed_model("minimax-m3")
    assert is_known_deployed_model("qwen3p7-plus")


def test_normalize_and_registry():
    assert normalize_model_id("minimax-m3") == REMOTE_MODEL
    assert normalize_model_id("//accounts/fireworks/models/qwen3p7-plus/") == (
        "accounts/fireworks/models/qwen3p7-plus"
    )
    assert not is_known_deployed_model("accounts/fireworks/models/qwen2p5-72b-instruct")


# --- Character-level routing --------------------------------------------------------


def test_spell_apple_backward_routes_remote():
    decision = _decide("spell apple backward")
    assert decision.route == "REMOTE", decision
    assert decision.reason == CHARACTER_LEVEL_GUARD_REASON, decision
    assert decision.model_id == REMOTE_MODEL
    assert decision.model_id.endswith("minimax-m3")


def test_write_123_backward_remote_and_not_rejected_as_number():
    decision = _decide("write 123 backward")
    assert decision.route == "REMOTE", decision
    assert decision.reason == CHARACTER_LEVEL_GUARD_REASON
    assert safe_math_agent("write 123 backward", time.perf_counter()) is None


def test_count_letters_in_strawberry_routes_remote():
    decision = _decide("count letters in strawberry")
    assert decision.route == "REMOTE", decision
    assert decision.reason == CHARACTER_LEVEL_GUARD_REASON
    assert decision.model_id == REMOTE_MODEL


def test_composite_prompt_pins_remote_single_task():
    plan = plan_request(COMPOSITE_PROMPT, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.global_route == "REMOTE", plan
    assert plan.reason == CHARACTER_LEVEL_GUARD_REASON, plan
    assert plan.single_remote is True, plan
    assert plan.tasks == [COMPOSITE_PROMPT], plan


def test_character_level_subtask_never_local():
    for sub in ("spell apple backward", "write 9+10 backward", "reverse the word cat"):
        decision = _decide(sub)
        assert decision.route == "REMOTE", (sub, decision)
        assert decision.reason == CHARACTER_LEVEL_GUARD_REASON, (sub, decision)
        assert decision.model_id != LOCAL_MODEL, (sub, decision)


# --- Factual-risk routing -----------------------------------------------------------


def test_where_is_cambodia_routes_remote():
    prompt = "Where is Cambodia?"
    assert is_factual_risk_prompt(prompt)
    decision = _decide(prompt)
    assert decision.route == "REMOTE", decision
    assert decision.reason == FACTUAL_RISK_GUARD_REASON, decision
    assert decision.model_id != LOCAL_MODEL


def test_capital_of_france_routes_remote_on_weak_local():
    decision = _decide("capital of France")
    assert decision.route == "REMOTE", decision
    assert decision.reason == FACTUAL_RISK_GUARD_REASON, decision
    assert decision.model_id == REMOTE_MODEL


def test_country_or_city_question_routes_remote():
    decision = _decide("Is London a country or a city?")
    assert decision.route == "REMOTE", decision
    assert decision.reason == FACTUAL_RISK_GUARD_REASON, decision


def test_casual_greeting_stays_local():
    decision = _decide("hello")
    assert decision.route == "LOCAL", decision
    assert decision.model_id == LOCAL_MODEL


def test_amazing_fact_about_france_routes_remote():
    prompt = "Tell me 1 amazing thing about France"
    assert is_factual_risk_prompt(prompt)
    decision = _decide(prompt)
    assert decision.route == "REMOTE", decision
    assert decision.reason == FACTUAL_RISK_GUARD_REASON, decision


def test_arithmetic_still_intercepted_locally():
    assert not is_character_level_task("2+12")
    result = safe_math_agent("math: 2+12", time.perf_counter())
    assert result is not None and result.answer == "14"


# --- Remote model fallback ----------------------------------------------------------


def test_unknown_model_not_tried_first():
    bogus = "accounts/fireworks/models/totally-bogus-model"
    candidates = build_remote_candidates(bogus)
    assert candidates[0] != bogus, candidates
    assert is_known_deployed_model(candidates[0]), candidates
    assert bogus in candidates, candidates


def test_known_model_tried_first():
    candidates = build_remote_candidates(REMOTE_MODEL)
    assert candidates[0] == REMOTE_MODEL, candidates
    assert LOCAL_MODEL not in candidates


def test_remote_fallback_on_not_found_uses_next_candidate():
    """First default model returns NOT_FOUND -> second default is used automatically."""
    preferred, second = REMOTE_MODEL, REMOTE_FALLBACK

    def handler(url, body):
        model = body.get("model")
        if model == preferred:
            return _FakeResponse(404, text="Model not found / not deployed")
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "olppa"}}],
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

    assert result.model_used == second, result.model_used
    assert result.model_used != LOCAL_MODEL
    assert result.answer == "olppa", result.answer
    attempts = result.diagnostics["remote_attempts"]
    assert [a["model_id"] for a in attempts] == [preferred, second], attempts
    assert attempts[0]["status"] == "unavailable"
    assert attempts[1]["status"] == "ok"


def test_remote_all_candidates_fail_returns_structured_error():
    def handler(url, body):
        return _FakeResponse(404, text="not found")

    original = "spell apple backward"
    with _patch_post(handler):
        result = app._route_text_remote(
            original,
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
            skip_distillation=True,
        )

    assert "All remote models failed" in result.answer, result.answer
    assert result.original_prompt == original
    attempts = result.diagnostics["remote_attempts"]
    assert len(attempts) >= 2
    assert all(a["status"] == "unavailable" for a in attempts), attempts


# --- Sub-agent factual behavior -----------------------------------------------------


def test_text_agent_returns_factual_answer_without_judgment():
    """Remote path must answer factual prompts without premise rejection."""
    fact = "The Eiffel Tower was the tallest structure in the world until 1930."

    def handler(url, body):
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": fact}}],
                "usage": {"total_tokens": 12},
            },
        )

    with _patch_post(handler):
        result = app._route_text_remote(
            "Tell me 1 amazing thing about France",
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
            skip_distillation=True,
        )

    assert result.route == "TEXT_REMOTE", result.route
    assert result.answer == fact, result.answer
    assert "not an amazing thing" not in result.answer.lower()
    assert not result.answer.lower().startswith("error")


def test_system_prompt_has_no_factual_premise_rejection():
    lowered = SUB_AGENT_SYSTEM_PROMPT.lower()
    assert "not an amazing thing" not in lowered
    assert "amazing thing about france" in lowered
    assert "never argue" in lowered or "do not argue" in lowered


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
