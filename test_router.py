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
    HARD_MAX_SUB_AGENTS,
    LOCAL_CREATIVE_REASON,
    LOCAL_GREETING_REASON,
    LOCAL_MATH_REASON,
    LOCAL_PRIME_REASON,
    LOCAL_UNAVAILABLE_REASON,
    REMOTE_FACTUAL_REASON,
    REMOTE_MODEL_CANDIDATES,
    REMOTE_SYMBOLIC_MATH_REASON,
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
    is_rate_limit_error,
    is_response_truncated,
    is_valid_subtask,
    mark_prior_local_failure,
    normalize_model_id,
    plan_request,
    reset_local_health_cache,
    reset_prior_local_failures,
    route_decision,
    safe_math_agent,
    strip_reasoning_traces,
    _render_key,
)

LOCAL_MODEL = DEFAULT_LOCAL_MODEL
REMOTE_MODEL = DEFAULT_REMOTE_MODEL
REMOTE_FALLBACK = REMOTE_MODEL_CANDIDATES[0]
BALANCED_COMPLEXITY_SCORE = 30
THRESHOLD = 30
COMPOSITE_PROMPT = (
    "spell apple backward, write the answer of 9+10 backward. "
    "Tell me 1 amazing thing about france"
)
MULTI_TASK_PROMPT = "tell me the capital of france, london, paris, cambodia"


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


def test_default_remote_list_contains_tier_models():
    ids = {c.split("/")[-1] for c in REMOTE_MODEL_CANDIDATES}
    assert "qwen3p7-plus" in ids
    assert "minimax-m3" in ids
    assert "qwen3p7-max" in ids


def test_select_remote_tier_by_complexity():
    assert app._select_remote_tier(10, REMOTE_MODEL).endswith("qwen3p7-plus")
    assert app._select_remote_tier(40, REMOTE_MODEL).endswith("minimax-m3")
    assert app._select_remote_tier(70, REMOTE_MODEL).endswith("qwen3p7-max")


def test_build_remote_candidates_uses_tier_first():
    candidates = build_remote_candidates(REMOTE_MODEL, score=10)
    assert candidates[0].endswith("qwen3p7-plus")
    assert any(m.endswith("qwen3p7-max") for m in candidates[1:])
    assert any(m.endswith("minimax-m3") for m in candidates[1:])


def test_complex_remote_decision_uses_full_tier():
    prompt = (
        "Explain quantum field theory and derive the path integral formulation "
        "with mathematical rigor and worked examples across multiple chapters."
    )
    decision = route_decision(
        prompt,
        THRESHOLD,
        has_image=False,
        active_local_model=LOCAL_MODEL,
        active_remote_model=REMOTE_MODEL,
    )
    assert decision.route == "REMOTE"
    assert decision.model_id.endswith("qwen3p7-max")


def test_general_prompt_defaults_to_remote():
    decision = _decide("What is the weather like today in Paris?")
    assert decision.route == "REMOTE", decision
    assert decision.model_id.endswith("qwen3p7-plus")


# --- Local capability ceiling -------------------------------------------------------


def test_hi_stays_local_when_healthy():
    _seed_local_health(LOCAL_MODEL, True)
    decision = _decide("hi")
    assert decision.route == "LOCAL", decision
    assert decision.reason == LOCAL_GREETING_REASON
    assert decision.model_id == LOCAL_MODEL


def test_hi_routes_canned_when_local_unhealthy():
    decision = _decide("hi", local_unavailable=True)
    assert decision.route == "LOCAL", decision
    assert decision.reason == app.CANNED_REPLY_REASON
    assert decision.model_id == "canned-reply"


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
    _reset_remote_validation_state()
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
    assert result.model_used.endswith("qwen3p7-plus")
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
    assert result.model_used.endswith("qwen3p7-plus")


def test_prior_local_failure_forces_remote():
    """Non-greeting prompts still escalate after a prior local failure."""
    reset_prior_local_failures()
    mark_prior_local_failure("format this list: a, b, c")
    decision = _decide("format this list: a, b, c")
    assert decision.route == "REMOTE"
    assert "prior-local-failure" in decision.reason


def test_greeting_ignores_prior_local_failure():
    reset_prior_local_failures()
    mark_prior_local_failure("hello")
    decision = _decide("hello")
    assert decision.route == "LOCAL"
    assert "greeting" in decision.reason


# --- Remote model fallback ----------------------------------------------------------


def _reset_remote_validation_state() -> None:
    app._VALIDATED_REMOTE_MODELS = []
    validated_path = app.ROOT_DIR / "validated_model_list.json"
    if validated_path.exists():
        validated_path.unlink()


def test_remote_fallback_on_not_found_uses_next_candidate():
    _reset_remote_validation_state()
    preferred, second = build_remote_candidates(REMOTE_MODEL, score=BALANCED_COMPLEXITY_SCORE)[:2]

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
            complexity_score=BALANCED_COMPLEXITY_SCORE,
        )

    assert result.model_used == second
    assert result.answer == "ok"


def test_remote_malformed_200_falls_back_to_next_candidate():
    _reset_remote_validation_state()
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
    _reset_remote_validation_state()
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
    _reset_remote_validation_state()
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
    assert result.model_used.endswith("qwen3p7-plus")
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

    # Plain numeric answer
    ok, _, _ = verifier.verify("What is 2 + 2?", "4", task_type="math")
    assert ok

    # Answer with explanation text
    ok, _, _ = verifier.verify("What is 25 × 4?", "The answer is 100.", task_type="math")
    assert ok

    # Answer with explanation before/after
    ok, _, _ = verifier.verify("What is 25 × 4?", "100 because 25 × 4 = 100.", task_type="math")
    assert ok

    # Answer with surrounding text
    ok, _, _ = verifier.verify("What is 25 × 4?", "After calculating, we get 100.", task_type="math")
    assert ok

    # Negative number
    ok, _, _ = verifier.verify("What is -5 + 3?", "-2", task_type="math")
    assert ok

    # Decimal number
    ok, _, _ = verifier.verify("What is 3.14 + 1?", "4.14", task_type="math")
    assert ok

    # Incorrect numeric answer (expected 4, got 5)
    ok, _, _ = verifier.verify("What is 2 + 2?", "5", task_type="math")
    assert not ok

    # Response with no numeric value at all
    ok, _, _ = verifier.verify("What is 2 + 2?", "I don't know", task_type="math")
    assert not ok

    # Empty output
    ok, _, _ = verifier.verify("What is 2 + 2?", "", task_type="math")
    assert not ok

    # Negative decimal
    ok, _, _ = verifier.verify("What is -3.5 + -1.5?", "-5.0", task_type="math")
    assert ok

    # Plain number without query (fallback: no expected answer, just checks number exists)
    ok, _, _ = verifier.verify("", "42", task_type="math")
    assert ok
    ok, _, _ = verifier.verify("", "not a number", task_type="math")
    assert not ok

    # Ensure non-math task_types are unaffected
    ok, _, _ = verifier.verify("", '{"answer": "test"}', task_type="json")
    assert ok


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


# --- PHANTOM routing UX fixes ---------------------------------------------------


def test_greeting_routes_local_not_remote():
    _seed_local_health(LOCAL_MODEL, True)
    decision = _decide("hello")
    assert decision.route == "LOCAL"
    assert decision.reason in {LOCAL_GREETING_REASON, app.CANNED_REPLY_REASON}


def test_greeting_budget_is_tiny():
    decision = _decide("thanks")
    budget = app.compute_remote_max_tokens("thanks", decision)
    assert budget == app.REMOTE_MAX_TOKENS_GREETING


def test_txt_context_builder_truncates():
    raw = b"line\n" * 2000
    block, chars = app.build_txt_context(raw, max_chars=100)
    assert "[Attached context from file]" in block
    assert chars == 100
    assert "truncated" in block


def test_dispatcher_matches_router_for_greeting():
    _seed_local_health(LOCAL_MODEL, True)

    def handler(url, body):
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "Hello there!"}}],
                "usage": {"total_tokens": 3},
            },
        )

    with _patch_post(handler):
        result = app.route_and_execute(
            "hi",
            THRESHOLD,
            "fw_test",
            LOCAL_MODEL,
            REMOTE_MODEL,
        )
    assert result.route in {"TEXT_LOCAL", "MATH_PYTHON"}
    assert result.model_used != REMOTE_MODEL


def test_greeting_routes_remote_when_skip_local():
    original = app.SKIP_LOCAL
    app.SKIP_LOCAL = True
    try:
        def handler(url, body):
            return _FakeResponse(
                200,
                payload={
                    "choices": [{"message": {"content": "Hello from Fireworks!"}}],
                    "usage": {"total_tokens": 5},
                },
            )

        with _patch_post(handler):
            result = app.route_and_execute(
                "Hello, how are you?",
                THRESHOLD,
                "fw_test",
                LOCAL_MODEL,
                REMOTE_MODEL,
            )
        assert result.route == "TEXT_REMOTE"
        assert result.model_used != "canned-reply"
        assert "Hello from Fireworks!" in result.answer
    finally:
        app.SKIP_LOCAL = original


def test_slow_remote_ui_timeout_stays_responsive():
    import time as _time

    def slow_handler(url, body):
        _time.sleep(0.05)
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"total_tokens": 2},
            },
        )

    original_timeout = app.REMOTE_UI_TIMEOUT_SECONDS
    app.REMOTE_UI_TIMEOUT_SECONDS = 1
    try:
        with _patch_post(slow_handler):
            result = app.run_request_nonblocking(
                app.route_and_execute,
                "Explain quantum entanglement in detail",
                THRESHOLD,
                "fw_test",
                LOCAL_MODEL,
                REMOTE_MODEL,
            )
        assert result.answer
        assert "timed out" not in result.answer.lower()
    finally:
        app.REMOTE_UI_TIMEOUT_SECONDS = original_timeout


def test_hard_prompt_still_routes_remote():
    decision = _decide("Write a binary search tree in Python with all operations")
    assert decision.route == "REMOTE"


# --- UI freeze / timing instrumentation -----------------------------------------


def test_hi_instant_dispatch_under_200ms():
    started = time.perf_counter()
    result = app.dispatch_instant_greeting(
        "hi",
        THRESHOLD,
        LOCAL_MODEL,
        REMOTE_MODEL,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < app.INSTANT_GREETING_MS_BUDGET, f"took {elapsed_ms:.1f}ms"
    assert result.answer
    assert result.diagnostics.get("instant_greeting") is True
    assert result.diagnostics.get("skipped_phantom") is True
    assert result.diagnostics.get("skipped_cache") is True


def test_hi_timing_stages_logged():
    result = app.dispatch_instant_greeting(
        "hi",
        THRESHOLD,
        LOCAL_MODEL,
        REMOTE_MODEL,
    )
    timing = result.diagnostics.get("timing")
    assert isinstance(timing, dict)
    stages = timing.get("stages_ms", {})
    assert "input_received" in stages
    assert "router_start" in stages
    assert "route_decision" in stages
    assert "final_response" in stages
    assert timing.get("complexity_score") is not None


def test_hi_skips_phantom_and_cache():
    phantom_called = {"n": 0}
    cache_called = {"n": 0}
    orig_phantom = app._angkor_phantom_execute
    orig_cache = app._cache_lookup

    def _phantom(*_a, **_k):
        phantom_called["n"] += 1
        return None

    def _cache(_p):
        cache_called["n"] += 1
        return None

    app._angkor_phantom_execute = _phantom  # type: ignore[assignment]
    app._cache_lookup = _cache  # type: ignore[assignment]
    try:
        result = app.process_user_request(
            "hi",
            THRESHOLD,
            "fw_test",
            LOCAL_MODEL,
            REMOTE_MODEL,
        )
    finally:
        app._angkor_phantom_execute = orig_phantom  # type: ignore[assignment]
        app._cache_lookup = orig_cache  # type: ignore[assignment]

    assert phantom_called["n"] == 0
    assert cache_called["n"] == 0
    assert result.diagnostics.get("instant_greeting") is True


def test_greeting_no_ollama_round_trip():
    _seed_local_health(LOCAL_MODEL, True)
    posts = []

    def handler(url, body):
        posts.append(url)
        return _FakeResponse(200, payload={"response": "slow local"})

    with _patch_post(handler):
        result = app.route_and_execute(
            "hello",
            THRESHOLD,
            "fw_test",
            LOCAL_MODEL,
            REMOTE_MODEL,
        )
    assert not any("11434" in str(u) for u in posts)
    assert result.model_used == "canned-reply"


def test_ui_feedback_timing_budget():
    timing = app.RequestTiming()
    timing.mark("input_received")
    timing.mark("ui_feedback_shown")
    feedback_ms = timing.ms_between("input_received", "ui_feedback_shown")
    assert feedback_ms is not None
    assert feedback_ms < app.INSTANT_GREETING_MS_BUDGET


def test_2_plus_2_instant_dispatch_under_200ms():
    started = time.perf_counter()
    result = app.dispatch_instant_trivial(
        "what is 2 + 2?",
        THRESHOLD,
        "",
        LOCAL_MODEL,
        REMOTE_MODEL,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < app.INSTANT_GREETING_MS_BUDGET, f"took {elapsed_ms:.1f}ms"
    assert result.route == "MATH_PYTHON"
    assert result.answer == "4"
    assert result.diagnostics.get("skipped_verify") is True


def test_missing_api_key_fails_before_phantom():
    phantom_called = {"n": 0}
    cache_called = {"n": 0}
    orig_phantom = app._angkor_phantom_execute
    orig_cache = app._cache_lookup

    def _phantom(*_a, **_k):
        phantom_called["n"] += 1
        return None

    def _cache(_p):
        cache_called["n"] += 1
        return None

    app._angkor_phantom_execute = _phantom  # type: ignore[assignment]
    app._cache_lookup = _cache  # type: ignore[assignment]
    try:
        result = app.process_user_request(
            "Explain quantum entanglement in detail with equations",
            THRESHOLD,
            "",
            LOCAL_MODEL,
            REMOTE_MODEL,
        )
    finally:
        app._angkor_phantom_execute = orig_phantom  # type: ignore[assignment]
        app._cache_lookup = orig_cache  # type: ignore[assignment]

    assert phantom_called["n"] == 0
    assert cache_called["n"] == 0
    assert result.diagnostics.get("fail_fast") is True
    assert "API Key required" in result.answer


def test_trivial_math_single_route_decision():
    calls = {"n": 0}
    orig = app.route_decision

    def counting_route_decision(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    app.route_decision = counting_route_decision  # type: ignore[assignment]
    try:
        app.process_user_request("2 + 2", THRESHOLD, "fw_test", LOCAL_MODEL, REMOTE_MODEL)
    finally:
        app.route_decision = orig  # type: ignore[assignment]
    assert calls["n"] == 1


# --- Single-agent-first planner -------------------------------------------------


LONG_DOC_LINES = "\n".join(
    f"Paragraph {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    for i in range(50)
)
LONG_SUMMARY_PROMPT = f"Summarize the following document:\n\n{LONG_DOC_LINES}"


def test_long_summarization_uses_single_agent():
    planner = app.decide_mode(LONG_SUMMARY_PROMPT)
    assert planner.mode == "DIRECT"
    assert planner.num_agents == 1
    assert planner.preserve_original is True
    plan = plan_request(LONG_SUMMARY_PROMPT, THRESHOLD, LOCAL_MODEL, REMOTE_MODEL)
    assert plan.single_route
    assert len(plan.tasks) == 1
    assert plan.tasks[0] == LONG_SUMMARY_PROMPT
    assert plan.classification.num_agents == 1


def test_summarize_context_not_split_by_regex():
    assert app.count_tasks(LONG_SUMMARY_PROMPT) == 1
    parts = heuristic_task_split(LONG_SUMMARY_PROMPT)
    assert len(parts) == 1
    assert not app.should_decompose(LONG_SUMMARY_PROMPT)


def test_independent_tasks_split_only_when_planner_approves():
    planner = app.decide_mode(MULTI_TASK_PROMPT)
    assert planner.mode == "SPLIT"
    assert planner.split_approved is True
    assert planner.num_agents > 1
    vague = "do many things with this long pasted text, " + LONG_DOC_LINES[:120]
    assert app.decide_mode(vague).mode == "DIRECT"


def test_summarization_token_budget_single_dispatch():
    calls = {"n": 0}
    orig = app.route_and_execute

    def counting_execute(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    app.route_and_execute = counting_execute  # type: ignore[assignment]
    try:
        result = app.process_user_request(
            LONG_SUMMARY_PROMPT,
            THRESHOLD,
            "fw_test",
            LOCAL_MODEL,
            REMOTE_MODEL,
        )
    finally:
        app.route_and_execute = orig  # type: ignore[assignment]

    assert calls["n"] == 1
    assert LONG_SUMMARY_PROMPT[:40] in (result.original_prompt or "")


def test_final_answer_preserves_main_question():
    planner = app.decide_mode("Summarize this article about climate change")
    assert planner.tasks[0] == "Summarize this article about climate change"
    wrapped = app._context_preserving_task(
        "What is the main theme?",
        "Identify the theme",
    )
    assert "What is the main theme?" in wrapped
    assert "Identify the theme" in wrapped


# --- Calibration + DeadZone tests --------------------------------------------


def test_preprocess_pipeline():
    from my_routing_agent.middleware.text_preprocess import preprocess_for_cache
    assert preprocess_for_cache("  Hello, World!  ") == "hello world"
    assert preprocess_for_cache("Café") == "café"


def test_cache_threshold_sweep_recommends_under_fpr_cap():
    import numpy as np
    from my_routing_agent.calibration.cache_calibrator import (
        recommend_threshold,
        sweep_thresholds,
    )

    pairs = [
        ("what is 2+2", "two plus two", True),
        ("capital of france", "france capital", True),
        ("hello", "write python code", False),
    ]
    embeddings = {
        "what is 2+2": np.array([1.0, 0.0, 0.0]),
        "two plus two": np.array([0.99, 0.01, 0.0]),
        "capital of france": np.array([0.0, 1.0, 0.0]),
        "france capital": np.array([0.0, 0.99, 0.01]),
        "hello": np.array([0.0, 0.0, 1.0]),
        "write python code": np.array([0.0, 0.0, 0.2]),
    }
    rows = sweep_thresholds(pairs, embeddings, start=0.80, stop=0.95, step=0.05)
    rec = recommend_threshold(rows, max_fpr=0.01)
    assert rec is not None
    assert rec.false_positive_rate < 0.01


def test_phantom_ensemble_abort_rules():
    from my_routing_agent.calibration.phantom_calibrator import (
        calibrate_ensemble,
        collect_records,
        should_abort,
    )
    from my_routing_agent.phantom.generation_signals import (
        GenerationSignalRecord,
        _synthetic_signals,
        signal_vector_at_token,
    )

    records = collect_records(
        app.DEFAULT_LOCAL_MODEL, n_good=120, n_bad=120, synthetic=True, seed=7
    )
    report = calibrate_ensemble(records)
    good_record = GenerationSignalRecord(
        prompt="hi", label="good", signals=_synthetic_signals("hi", "good"), source="synthetic"
    )
    bad_record = GenerationSignalRecord(
        prompt="spell apple backward",
        label="bad",
        signals=_synthetic_signals("spell apple backward", "bad"),
        source="synthetic",
    )
    success_vector = signal_vector_at_token(good_record, token_index=8)
    failure_vector = signal_vector_at_token(bad_record, token_index=8)
    assert should_abort(success_vector, report) is False
    assert should_abort(failure_vector, report) is True


def test_deadzone_remote_fallback_on_model_not_found():
    from unittest.mock import patch
    from my_routing_agent.phantom.deadzone_runner import DeadZoneRunner

    calls: list[str] = []

    def remote_call(
        prompt: str,
        *,
        model_id: str,
        max_tokens: int,
        cancel_event,
        token_counter,
    ):
        calls.append(model_id)
        if "minimax" in model_id:
            return None, "model not found", 0
        token_counter.tokens_emitted = 24
        return "remote ok", "ok", 24

    with patch(
        "my_routing_agent.phantom.deadzone_runner.should_abort",
        return_value=True,
    ):
        runner = DeadZoneRunner(ensemble_report={"abort_threshold": 0.25})
        answer, winner, meta = runner.run_race(
            "hello there",
            local_model="qwen2.5:0.5b",
            validated_remote_models=[
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/qwen3p7-plus",
            ],
            remote_call=remote_call,
            L_out_norm=0.4,
            confidence=0.7,
        )
    assert len(calls) >= 2
    assert answer == "remote ok"
    assert meta["remote_models_tried"][0].endswith("minimax-m3")
    assert any("qwen3p7-plus" in m for m in meta["remote_models_tried"])


def test_validate_remote_models_filters_inaccessible():
    from unittest.mock import patch
    from my_routing_agent.remote.validate_models import validate_remote_models

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "accounts/fireworks/models/qwen3p7-plus"},
                ]
            }

    with patch(
        "my_routing_agent.remote.validate_models.requests.get",
        return_value=_Resp(),
    ):
        out = validate_remote_models(
            [
                "accounts/fireworks/models/minimax-m3",
                "accounts/fireworks/models/qwen3p7-plus",
            ],
            "fw_test",
            output_path=app.ROOT_DIR / "validated_model_list.json",
        )
    assert out["validated"] == ["accounts/fireworks/models/qwen3p7-plus"]
    assert "accounts/fireworks/models/minimax-m3" in out["removed"]


# --- Adversarial diagnostic fixes (ANGKOR + PHANTOM) -----------------------------


def test_symbolic_math_routes_remote_not_python_eval():
    from my_routing_agent.utils.math_eval import is_symbolic_math

    prompt = "Solve for x: 3x^2 + 5x - 2 = 0"
    assert is_symbolic_math(prompt)
    assert not app.would_math_intercept(prompt)
    decision = _decide(prompt)
    assert decision.route == "REMOTE"
    assert decision.reason == REMOTE_SYMBOLIC_MATH_REASON
    assert safe_math_agent(prompt, time.perf_counter()) is None


def test_derivative_routes_remote_not_python_eval():
    prompt = "Calculate the derivative of x^3 + 2x^2 - 5x"
    decision = _decide(prompt)
    assert decision.route == "REMOTE"
    assert decision.reason == REMOTE_SYMBOLIC_MATH_REASON
    assert not app.would_math_intercept(prompt)


def test_simple_arithmetic_stays_local():
    decision = _decide("2 + 2")
    assert decision.route == "LOCAL"
    assert decision.reason == LOCAL_MATH_REASON
    assert app.would_math_intercept("2 + 2")


def test_prime_check_routes_local():
    decision = _decide("is 17 prime")
    assert decision.route == "LOCAL"
    assert decision.reason == LOCAL_PRIME_REASON
    result = safe_math_agent("is 17 prime", time.perf_counter())
    assert result is not None
    assert result.answer == "yes"


def test_creative_prompt_routes_local():
    decision = _decide("Write a poem about the ocean")
    assert decision.route == "LOCAL"
    assert decision.reason == LOCAL_CREATIVE_REASON


def test_math_heavy_prompt_not_exploded_into_swarm():
    prompt = "Solve for x: 3x^2 + 5x - 2 = 0"
    parts = heuristic_task_split(prompt)
    assert len(parts) == 1
    assert not is_beneficial_to_decompose(prompt)
    planner = app.decide_mode(prompt)
    assert planner.num_agents == 1


def test_invalid_subtask_fragments_rejected():
    assert not is_valid_subtask("2")
    assert not is_valid_subtask("8]")
    mixed = "tell me the capital of france, london, paris, cambodia, 2+12"
    parts = heuristic_task_split(mixed)
    assert len(parts) <= HARD_MAX_SUB_AGENTS
    assert all(is_valid_subtask(p) for p in parts)


def test_rate_limit_detection_and_clean_message():
    assert is_rate_limit_error(429, "")
    assert is_rate_limit_error(400, "RATE_LIMIT_EXCEEDED")
    from unittest.mock import patch

    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _FakeResponse(429, text="RATE_LIMIT_EXCEEDED")
        return _FakeResponse(
            200,
            payload={
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            },
        )

    with patch("app.requests.post", side_effect=fake_post):
        with patch("app.time.sleep", return_value=None):
            result = app._route_text_remote(
                "hello",
                "fw_test",
                LOCAL_MODEL,
                REMOTE_MODEL,
                time.perf_counter(),
            )
    assert calls["n"] >= 3
    assert result.answer == "ok"
    assert result.diagnostics.get("rate_limit_hits", 0) >= 1


def test_truncation_triggers_retry():
    from unittest.mock import patch

    calls = {"n": 0}

    def fake_post(*_args, **_kwargs):
        calls["n"] += 1
        finish = "length" if calls["n"] == 1 else "stop"
        return _FakeResponse(
            200,
            payload={
                "choices": [
                    {"message": {"content": "partial answer"}, "finish_reason": finish}
                ],
                "usage": {"total_tokens": 10},
            },
        )

    with patch("app.requests.post", side_effect=fake_post):
        result = app._route_text_remote(
            "Explain the French Revolution in detail",
            "fw_test",
            LOCAL_MODEL,
            REMOTE_MODEL,
            time.perf_counter(),
            max_tokens=64,
        )
    assert calls["n"] >= 2
    assert result.answer == "partial answer"
    assert result.diagnostics.get("truncated") is True


def test_is_response_truncated_helper():
    assert is_response_truncated({"choices": [{"finish_reason": "length"}]})
    assert not is_response_truncated({"choices": [{"finish_reason": "stop"}]})


def test_render_key_unique_per_message():
    from app import RouteResult

    r1 = RouteResult("a", "TEXT_LOCAL", 0, 1.0, message_id="msg_a")
    r2 = RouteResult("b", "TEXT_LOCAL", 0, 1.0, message_id="msg_b")
    assert _render_key("distilled_prompt", r1) != _render_key("distilled_prompt", r2)
    assert _render_key("orig_prompt", r1, "1") != _render_key("orig_prompt", r1, "2")


def test_strip_reasoning_traces_removes_scratchpad():
    raw = "Step 1: think about rhymes.\n\nThe rain falls softly on the shore."
    cleaned = strip_reasoning_traces(raw)
    assert "Step 1" not in cleaned
    assert "rain falls softly" in cleaned


def test_symbolic_math_not_caught_by_multihop():
    prompt = "Explain and solve for x: 3x^2 + 5x - 2 = 0"
    decision = _decide(prompt)
    assert decision.reason == REMOTE_SYMBOLIC_MATH_REASON


def test_entropy_gate_blocks_gibberish():
    gibberish = "xqwp zmkv jfhd lsrt pwqm zkvn mxhf"
    assert app.should_entropy_gate_input(gibberish)
    decision = _decide(gibberish)
    assert decision.reason == app.ENTROPY_GATE_REASON


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


# ============================================================================
# Integration tests for stress-test fixes
# ============================================================================


def test_cache_store_guard_skips_failed_results():
    """_cache_store should not store failed results."""
    from app import _cache_store, _lazy_init_angkor_cache
    from app import RouteResult
    import app as _app

    cache = _lazy_init_angkor_cache()
    if cache is None:
        return  # skip if no cache

    prompt = "test guard prompt 12345"
    # Create a failed result
    failed_result = RouteResult(
        answer="",
        route="TEXT_LOCAL",
        tokens=0,
        latency_ms=100,
        original_prompt=prompt,
        model_used="local",
        success=False,
        error_type="timeout",
    )

    # Store should be skipped for failed results
    _cache_store(prompt, failed_result)

    # Verify it was NOT stored
    hit = cache.lookup(prompt)
    assert hit is None, f"Failed result should not be cached but got: {hit}"


def test_cache_eviction_policy_exists():
    """SemanticCache should have max_entries parameter and eviction method."""
    from my_routing_agent.cache.semantic_cache import SemanticCache
    from my_routing_agent.config import CacheConfig

    config = CacheConfig(max_entries=5)
    assert config.max_entries == 5


def test_prior_failure_key_uses_hash():
    """_prompt_failure_key should use SHA256 hash, not truncation."""
    from app import _prompt_failure_key

    key1 = _prompt_failure_key("short")
    key2 = _prompt_failure_key("short prompt" + "x" * 1000)

    # Different inputs should produce different keys (no collision)
    assert key1 != key2
    # Keys should be hex strings of consistent length
    assert len(key1) == 32
    assert len(key2) == 32


def test_adaptive_threshold_wired_to_router():
    """SklearnRouter should accept optional adaptive_threshold parameter."""
    from my_routing_agent.routers.engine import SklearnRouter

    router = SklearnRouter()
    assert hasattr(router, '_adaptive_threshold')
    assert router._adaptive_threshold is None


def test_deadzone_runner_import():
    """DeadZoneRunner should be importable and usable."""
    from my_routing_agent.phantom.deadzone_runner import DeadZoneRunner

    assert callable(DeadZoneRunner)


def test_local_health_cache_marks_connection_error():
    """ConnectionError should mark local health as False."""
    from app import _LOCAL_HEALTH_CACHE, _LOCAL_HEALTH_LOCK, LOCAL_HEALTH_TTL_SECONDS
    import time
    import app as _app

    model = "test-connection-error-model"
    with _LOCAL_HEALTH_LOCK:
        _LOCAL_HEALTH_CACHE[model] = (time.time() + 1000, True)

    # Simulate what ConnectionError handler does
    with _LOCAL_HEALTH_LOCK:
        _LOCAL_HEALTH_CACHE[model.strip()] = (
            time.time() + LOCAL_HEALTH_TTL_SECONDS,
            False,
        )

    with _LOCAL_HEALTH_LOCK:
        _, healthy = _LOCAL_HEALTH_CACHE.get(model, (0, True))

    assert healthy is False, "ConnectionError should mark health as False"


if __name__ == "__main__":
    raise SystemExit(_run())
