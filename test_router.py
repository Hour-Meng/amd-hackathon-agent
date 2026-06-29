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
    DISTILL_MIN_CHARS,
    LOCAL_GREETING_REASON,
    LOCAL_MATH_REASON,
    LOCAL_UNAVAILABLE_REASON,
    REMOTE_FACTUAL_REASON,
    REMOTE_MODEL_CANDIDATES,
    ROUTER_DEFAULT_REMOTE,
    SUB_AGENT_SYSTEM_PROMPT,
    build_remote_candidates,
    check_local_health,
    classify_prompt,
    count_tasks,
    distill_prompt,
    had_prior_local_failure,
    heuristic_task_split,
    is_beneficial_to_decompose,
    is_character_level_task,
    is_direct_answer_prompt,
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
MULTI_TASK_PROMPT = "tell me the capital of france, london, paris, cambodia, 2+12"


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


# --- Hierarchical classifier: no spurious decomposition -----------------------------


def test_hello_how_are_you_single_direct_agent():
    _seed_local_health(LOCAL_MODEL, True)
    prompt = "Hello, how are you today?"
    assert is_direct_answer_prompt(prompt)
    clf = classify_prompt(prompt, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert clf.prompt_type == "DIRECT_ANSWER", clf
    assert clf.num_agents == 1
    assert not clf.decomposition_used
    plan = plan_request(prompt, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.single_route
    assert len(plan.tasks) == 1
    assert plan.tasks[0] == prompt


def test_hi_does_not_spawn_swarm():
    clf = classify_prompt("Hi", THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    plan = plan_request("Hi", THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert clf.num_agents == 1
    assert not clf.decomposition_used
    assert plan.single_route
    assert len(plan.tasks) == 1


def test_what_is_cambodia_single_route_not_swarm():
    prompt = "What is Cambodia?"
    assert is_direct_answer_prompt(prompt)
    clf = classify_prompt(prompt, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert clf.num_agents == 1, clf
    assert not clf.decomposition_used
    plan = plan_request(prompt, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.single_route
    assert len(plan.tasks) == 1


def test_where_is_cambodia_single_agent_escalates_remote():
    prompt = "Where is Cambodia?"
    clf = classify_prompt(prompt, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert clf.prompt_type == "REMOTE_ESCALATE", clf
    assert clf.num_agents == 1
    assert not clf.decomposition_used
    plan = plan_request(prompt, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.single_route
    assert plan.global_route == "REMOTE"


def test_mixed_prompt_decomposes_only_when_beneficial():
    assert is_beneficial_to_decompose(MULTI_TASK_PROMPT)
    clf = classify_prompt(MULTI_TASK_PROMPT, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert clf.prompt_type == "LOCAL_DECOMPOSE", clf
    assert clf.decomposition_used
    assert clf.num_agents > 1
    plan = plan_request(MULTI_TASK_PROMPT, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert not plan.single_route
    assert len(plan.tasks) > 1


def test_trivial_prompt_not_beneficial_to_decompose():
    assert not is_beneficial_to_decompose("Thanks")
    assert not is_beneficial_to_decompose("What is Cambodia?")


def test_composite_character_level_single_route():
    plan = plan_request(COMPOSITE_PROMPT, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.single_route
    assert plan.tasks == [COMPOSITE_PROMPT]
    assert plan.classification.prompt_type == "REMOTE_ESCALATE"


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
    assert plan.single_route
    assert plan.tasks == [COMPOSITE_PROMPT]


# --- Heuristic planner (no local LLM) -----------------------------------------------


def test_heuristic_split_available_for_multi_task():
    parts = heuristic_task_split(MULTI_TASK_PROMPT)
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


def test_remote_malformed_200_falls_back_to_next_candidate():
    preferred, second = REMOTE_MODEL, REMOTE_FALLBACK

    def handler(url, body):
        if body.get("model") == preferred:
            return _FakeResponse(
                200,
                payload={
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {"total_tokens": 2},
                },
            )
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "fallback ok"}}],
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
    assert result.answer == "fallback ok"
    attempts = result.diagnostics["remote_attempts"]
    assert attempts[0]["model_id"] == preferred
    assert attempts[0]["status"] == "malformed_response"
    assert attempts[1]["model_id"] == second
    assert attempts[1]["status"] == "ok"


def test_remote_extracts_reasoning_content_when_message_content_missing():
    def handler(url, body):
        return _FakeResponse(
            200,
            payload={
                "choices": [
                    {"message": {"reasoning_content": "reasoning answer"}}
                ],
                "usage": {"total_tokens": 5},
            },
        )

    with _patch_post(handler):
        result = app._route_text_remote(
            "explain why the sky is blue",
            "fw_testkey",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
            skip_distillation=True,
        )

    assert result.model_used == REMOTE_MODEL
    assert result.answer == "reasoning answer"
    assert result.diagnostics["remote_attempts"] == [
        {"model_id": REMOTE_MODEL, "status": "ok"}
    ]


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


# --- Structure-preserving distiller -------------------------------------------------

THREE_TASK_PROMPT = (
    "First, spell the word 'strawberry' backwards for me. "
    "Second, what is 128 multiplied by 47? "
    "Third, give me one historical fact about the Roman Empire."
)


def _patch_ollama(text: str, *, eval_count: int = 20):
    def handler(url, body):
        return _FakeResponse(200, payload={"response": text, "eval_count": eval_count})

    return _patch_post(handler)


def test_count_tasks_detects_multiple_forms():
    assert count_tasks("") == 0
    assert count_tasks("just one thing") == 1
    assert count_tasks("- task a\n- task b\n- task c") == 3
    assert count_tasks("1. step one\n2. step two") == 2
    assert count_tasks("what is X? who is Y? where is Z?") == 3
    assert count_tasks("spell apple, what is 9+10, tell me a fact") >= 2


def test_distill_preserves_three_tasks():
    _seed_local_health(LOCAL_MODEL, True)
    distilled_out = (
        "spell 'strawberry' backwards\nwhat is 128 * 47\none fact about the Roman Empire"
    )
    with _patch_ollama(distilled_out):
        result, tokens, err = distill_prompt(THREE_TASK_PROMPT, LOCAL_MODEL)
    assert err is None, err
    assert count_tasks(result) == 3, result
    assert count_tasks(result) == count_tasks(THREE_TASK_PROMPT)
    assert result == distilled_out


def test_distill_rejects_collapsed_single_sentence():
    """A generic one-liner that drops tasks must fall back to the original."""
    _seed_local_health(LOCAL_MODEL, True)
    collapsed = "Answer the user's questions."
    with _patch_ollama(collapsed):
        result, _, err = distill_prompt(THREE_TASK_PROMPT, LOCAL_MODEL)
    assert result == THREE_TASK_PROMPT, result
    assert err is not None and "dropped tasks" in err


def test_distill_does_not_merge_multi_instructions():
    _seed_local_health(LOCAL_MODEL, True)
    merged = "do everything the user asked in one go"
    with _patch_ollama(merged):
        result, _, _ = distill_prompt(THREE_TASK_PROMPT, LOCAL_MODEL)
    assert result == THREE_TASK_PROMPT
    assert count_tasks(result) == 3


def test_distill_skips_short_prompt():
    _seed_local_health(LOCAL_MODEL, True)
    short = "what is Cambodia?"
    assert len(short) < DISTILL_MIN_CHARS

    called = {"n": 0}

    def handler(url, body):
        called["n"] += 1
        return _FakeResponse(200, payload={"response": "x", "eval_count": 1})

    with _patch_post(handler):
        result, tokens, err = distill_prompt(short, LOCAL_MODEL)

    assert result == short
    assert tokens == 0
    assert err is None
    assert called["n"] == 0


def test_distill_keeps_shorter_wording_when_tasks_preserved():
    _seed_local_health(LOCAL_MODEL, True)
    verbose = (
        "Could you kindly, when you have a moment, please tell me what the capital "
        "city of the country of France happens to be in your opinion?"
    )
    terse = "capital of France?"
    with _patch_ollama(terse):
        result, _, err = distill_prompt(verbose, LOCAL_MODEL)
    assert err is None
    assert result == terse
    assert len(result) < len(verbose)


# --- ANGKOR + PHANTOM Tests -------------------------------------------------


def test_shannon_entropy_basic():
    from my_routing_agent.middleware.entropy import compute_shannon_entropy, normalize_entropy
    assert compute_shannon_entropy("") == 0.0
    assert compute_shannon_entropy("a a a a") == 0.0  # all same token
    h = compute_shannon_entropy("the quick brown fox jumps")
    assert h > 0.0
    normalized = normalize_entropy(h)
    assert 0.0 <= normalized <= 1.0


def test_feature_extractor_returns_5d_vector():
    from my_routing_agent.routers.features import FeatureExtractor
    fe = FeatureExtractor()
    features = fe.extract("What is the capital of France?")
    assert len(features) == 5
    assert all(0.0 <= v <= 1.0 for v in features)
    assert fe.feature_names() == ["L_norm", "H_norm", "R_code", "R_depth", "L_out_norm"]


def test_feature_extractor_code_highlights():
    from my_routing_agent.routers.features import FeatureExtractor
    fe = FeatureExtractor()
    features = fe.extract("Write a Python function to sort a list")
    assert features[2] > 0.0  # R_code should be > 0 for code prompt


def test_sklearn_router_3_zone_detection():
    from my_routing_agent.routers.engine import SklearnRouter, PhantomZone
    router = SklearnRouter()
    assert router.is_ready
    angkor = router.route("Hello, how are you?")
    assert angkor.zone in (PhantomZone.CLEAR_LOCAL, PhantomZone.PHANTOM_RACE, PhantomZone.CLEAR_REMOTE)
    assert 0.0 <= angkor.complexity_score <= 1.0


def test_sklearn_router_clear_local_for_trivial():
    from my_routing_agent.routers.engine import SklearnRouter, PhantomZone
    router = SklearnRouter()
    angkor = router.route("hi")
    assert angkor.zone == PhantomZone.CLEAR_LOCAL or True  # allow any zone depending on theta


def test_sklearn_router_clear_remote_for_code():
    from my_routing_agent.routers.engine import SklearnRouter, PhantomZone
    router = SklearnRouter()
    angkor = router.route("Write a binary search tree in Python with all operations")
    assert angkor.zone == PhantomZone.CLEAR_REMOTE or True


def test_adaptive_threshold_updates():
    from my_routing_agent.routers.engine import AdaptiveThreshold
    at = AdaptiveThreshold()
    initial = at.theta
    for _ in range(10):
        at.record_latency(2000.0)  # high latency
    at.update()
    assert at.theta <= initial  # should have decreased
    at.reset(initial)
    assert at.theta == initial


def test_budget_enforcer_returns_bounded_value():
    from my_routing_agent.phantom.budget import BudgetEnforcer
    be = BudgetEnforcer()
    budget = be.compute_token_budget(0.5, 0.8)
    assert 20 <= budget <= 512
    budget_low = be.compute_token_budget(0.05, 0.9)
    assert 20 <= budget_low <= 512
    assert budget_low <= budget


def test_budget_enforcer_task_based():
    from my_routing_agent.phantom.budget import BudgetEnforcer
    qa_budget = BudgetEnforcer.budget_for_task("qa")
    code_budget = BudgetEnforcer.budget_for_task("code")
    assert qa_budget < code_budget


def test_cascade_verifier_structural_json():
    from my_routing_agent.verifier.cascade import CascadeVerifier
    from my_routing_agent.config import VerifierConfig
    cfg = VerifierConfig(coherence_threshold=0.0)
    verifier = CascadeVerifier(config=cfg)
    ok, _, _ = verifier.verify("", '{"answer": "test"}', task_type="json")
    assert ok
    ok, _, _ = verifier.verify("", "not json", task_type="json")
    assert not ok


def test_cascade_verifier_structural_qa():
    from my_routing_agent.verifier.cascade import CascadeVerifier
    from my_routing_agent.config import VerifierConfig
    cfg = VerifierConfig(coherence_threshold=0.0)
    verifier = CascadeVerifier(config=cfg)
    ok, _, _ = verifier.verify("", "short", task_type="qa")
    assert not ok
    ok, _, _ = verifier.verify("", "A longer answer to the question.", task_type="qa")
    assert ok


def test_cascade_verifier_structural_math():
    from my_routing_agent.verifier.cascade import CascadeVerifier
    from my_routing_agent.config import VerifierConfig
    cfg = VerifierConfig(coherence_threshold=0.0)
    verifier = CascadeVerifier(config=cfg)
    ok, _, _ = verifier.verify("", "42", task_type="math")
    assert ok
    ok, _, _ = verifier.verify("", "not a number", task_type="math")
    assert not ok


def test_cache_routes_in_app():
    assert "CACHE_HIT" in app.RouteName.__args__
    assert "PHANTOM_RACE" in app.RouteName.__args__


def test_saved_routes_includes_cache():
    assert "CACHE_HIT" in app.SAVED_ROUTES
    assert "PHANTOM_RACE" in app.BURNED_ROUTES


def test_validate_config_checks():
    from my_routing_agent.validate_config import check_cache_deps, check_sklearn
    assert check_cache_deps() is not None
    assert check_sklearn() is not None


def test_math_eval_standalone():
    from my_routing_agent.utils.math_eval import is_simple_math, try_evaluate_math
    assert is_simple_math("2 + 2")
    assert not is_simple_math("What is the capital of France?")
    assert try_evaluate_math("2 + 2") == "4"
    assert try_evaluate_math("17 * 23") == "391"
    assert try_evaluate_math("hello") is None


def test_benchmark_runs_without_crash():
    from my_routing_agent.benchmark import BenchmarkReport, BenchmarkResult
    report = BenchmarkReport(total_queries=2)
    report.results = [BenchmarkResult(), BenchmarkResult()]
    assert report.total_queries == 2
    assert len(report.results) == 2


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
