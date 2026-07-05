# Graph Report - .  (2026-07-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 621 nodes · 1570 edges · 36 communities (35 shown, 1 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 176 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `dbd7c265`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_RouteResult|RouteResult]]
- [[_COMMUNITY_config.py|config.py]]
- [[_COMMUNITY_BudgetEnforcer|BudgetEnforcer]]
- [[_COMMUNITY_SemanticCache|SemanticCache]]
- [[_COMMUNITY_app.py|app.py]]
- [[_COMMUNITY_test_router.py|test_router.py]]
- [[_COMMUNITY_FeatureExtractor|FeatureExtractor]]
- [[_COMMUNITY_phantom_calibrator.py|phantom_calibrator.py]]
- [[_COMMUNITY_CascadeVerifier|CascadeVerifier]]
- [[_COMMUNITY__decide|_decide]]
- [[_COMMUNITY_engine.py|engine.py]]
- [[_COMMUNITY_InputCompressor|InputCompressor]]
- [[_COMMUNITY_should_decompose|should_decompose]]
- [[_COMMUNITY_RoutingEngine|RoutingEngine]]
- [[_COMMUNITY_is_symbolic_math|is_symbolic_math]]
- [[_COMMUNITY_TokenCounter|TokenCounter]]
- [[_COMMUNITY_route_decision|route_decision]]
- [[_COMMUNITY_SklearnRouter|SklearnRouter]]
- [[_COMMUNITY_classify_prompt|classify_prompt]]
- [[_COMMUNITY_main.py|main.py]]
- [[_COMMUNITY__seed_local_health|_seed_local_health]]
- [[_COMMUNITY__patch_post|_patch_post]]
- [[_COMMUNITY_had_prior_local_failure|had_prior_local_failure]]
- [[_COMMUNITY_validate_remote_models|validate_remote_models]]
- [[_COMMUNITY_heuristic_task_split|heuristic_task_split]]
- [[_COMMUNITY_would_math_intercept|would_math_intercept]]
- [[_COMMUNITY_is_heavy_local_model|is_heavy_local_model]]
- [[_COMMUNITY_is_valid_subtask|is_valid_subtask]]
- [[_COMMUNITY_should_entropy_gate_input|should_entropy_gate_input]]
- [[_COMMUNITY_reset_local_health_cache|reset_local_health_cache]]
- [[_COMMUNITY___init__.py|__init__.py]]

## God Nodes (most connected - your core abstractions)
1. `FeatureExtractor` - 37 edges
2. `SklearnRouter` - 36 edges
3. `RouteResult` - 35 edges
4. `TokenCounter` - 33 edges
5. `SemanticCache` - 29 edges
6. `BudgetEnforcer` - 28 edges
7. `route_decision()` - 27 edges
8. `CascadeVerifier` - 27 edges
9. `route_and_execute()` - 26 edges
10. `HybridRoutingAgent` - 26 edges

## Surprising Connections (you probably didn't know these)
- `RouteResult` --uses--> `SemanticCache`  [INFERRED]
  app.py → my_routing_agent/cache/semantic_cache.py
- `RouteResult` --uses--> `BudgetEnforcer`  [INFERRED]
  app.py → my_routing_agent/phantom/budget.py
- `RouteResult` --uses--> `ConfidencePredictor`  [INFERRED]
  app.py → my_routing_agent/phantom/confidence.py
- `RouteResult` --uses--> `SpeculativeRunner`  [INFERRED]
  app.py → my_routing_agent/phantom/speculative.py
- `RouteResult` --uses--> `AdaptiveThreshold`  [INFERRED]
  app.py → my_routing_agent/routers/engine.py

## Import Cycles
- None detected.

## Communities (36 total, 1 thin omitted)

### Community 0 - "RouteResult"
Cohesion: 0.05
Nodes (63): _aggregate_swarm_metrics(), _angkor_phantom_execute(), _cache_lookup(), _cache_store(), calculate_complexity(), compress_image_to_base64(), dispatch_instant_greeting(), dispatch_instant_trivial() (+55 more)

### Community 1 - "config.py"
Cohesion: 0.06
Nodes (33): Path, run_phantom_calibration(), Inference clients for local and remote models., InferenceResponse, LocalClient, Any, OpenAI-compatible client for local Ollama / CPU inference with logprobs streamin, Stream chat completions with logprobs. Yields dicts with keys:           - token (+25 more)

### Community 2 - "BudgetEnforcer"
Cohesion: 0.06
Nodes (35): _attach_orchestration(), ClassificationResult, _log_orchestration(), PlannerDecision, Single-agent-first planner output — one read of the full prompt., Level-1 cheap classifier output — runs on the full prompt before any split., Pre-swarm plan from the hierarchical classifier + planner., Backward-compatible alias for single-route plans. (+27 more)

### Community 3 - "SemanticCache"
Cohesion: 0.09
Nodes (27): CacheEntry, Any, ndarray, Tier 0 — FAISS semantic cache gate for zero-token repeat queries., FAISS + MiniLM semantic cache for intercepting semantically identical queries., SemanticCache, cosine_similarity(), embed_queries() (+19 more)

### Community 4 - "app.py"
Cohesion: 0.10
Nodes (29): _bootstrap_angkor_session(), build_remote_candidates(), build_txt_context(), check_local_health(), get_local_health_for_ui(), init_session_state(), is_known_deployed_model(), is_weak_local_model() (+21 more)

### Community 5 - "test_router.py"
Cohesion: 0.08
Nodes (9): is_rate_limit_error(), is_response_truncated(), Remove chain-of-thought / scratchpad leakage from model output., strip_reasoning_traces(), Router-first regression tests for the hybrid AI middleware.  Run with:  python3, test_count_tasks_detects_multiple_forms(), test_is_response_truncated_helper(), test_rate_limit_detection_and_clean_message() (+1 more)

### Community 6 - "FeatureExtractor"
Cohesion: 0.13
Nodes (19): BenchmarkReport, BenchmarkResult, main(), Any, Benchmark suite for ANGKOR Router + PHANTOM Layer token savings & accuracy measu, run_benchmark(), compute_shannon_entropy(), normalize_entropy() (+11 more)

### Community 7 - "phantom_calibrator.py"
Cohesion: 0.18
Nodes (20): calibrate_ensemble(), collect_records(), _expand_prompts(), Any, PHANTOM ensemble calibration: ROC-AUC + logistic regression abort rule., should_abort(), Dead-zone speculative race with cancellable remote + telemetry., _entropy_from_logprobs() (+12 more)

### Community 8 - "CascadeVerifier"
Cohesion: 0.14
Nodes (12): Tier 3 — Cascade verify settings., VerifierConfig, CascadeVerifier, Any, Tier 3 — Cascade Verify: structural validation → semantic coherence → binary esc, Three-step output quality verifier with binary escalation., Run the cascade. Returns (accepted, final_output, escalated).          Step 1: S, Extract all integers and decimal numbers (including negative) from text. (+4 more)

### Community 9 - "_decide"
Cohesion: 0.10
Nodes (22): is_factual_risk_prompt(), Extract embedded mathematical expressions from natural-language prompts.     Pla, True for direct geographic, civic, identity, or encyclopedia-style facts., safe_math_agent(), _decide(), test_arithmetic_routes_local_math(), test_capital_of_france_routes_remote(), test_creative_prompt_routes_local() (+14 more)

### Community 10 - "engine.py"
Cohesion: 0.18
Nodes (15): Enum, AdaptiveThresholdConfig, Adaptive routing threshold θ settings., Strict routing thresholds for Tier-1 heuristics and Tier-2 classification., RoutingThresholds, AdaptiveThreshold, AngkorRoutingResult, PhantomZone (+7 more)

### Community 11 - "InputCompressor"
Cohesion: 0.17
Nodes (10): Image, CompressorConfig, Image down-sampling and text pruning settings., InputCompressor, Any, Path, Down-sample images, prune text, and aggressively compress prompts before inferen, Second-pass aggressive compression for maximum token savings. (+2 more)

### Community 12 - "should_decompose"
Cohesion: 0.20
Nodes (19): _context_preserving_task(), count_tasks(), decide_mode(), is_character_level_task(), is_direct_answer_prompt(), is_iterative_long_task(), is_long_context_prompt(), is_simple_format_task() (+11 more)

### Community 13 - "RoutingEngine"
Cohesion: 0.21
Nodes (5): ProcessedInput, Normalized payload ready for routing and inference., Original rule-based router — kept as fallback when sklearn unavailable., RoutingEngine, RoutingResult

### Community 14 - "is_symbolic_math"
Cohesion: 0.19
Nodes (17): is_multi_hop_prompt(), Multi-part or analytical prompts that need remote reasoning., _eval_arithmetic(), extract_arithmetic_expression(), is_local_arithmetic(), _is_prime(), is_prime_check_prompt(), is_simple_math() (+9 more)

### Community 15 - "TokenCounter"
Cohesion: 0.18
Nodes (10): Encoding, Local token estimation settings., TokenizerConfig, Utility modules for the hybrid routing agent., estimate_tokens(), _get_encoding(), Instant local token estimation without API round-trips., CPU-only token counter backed by tiktoken encodings. (+2 more)

### Community 16 - "route_decision"
Cohesion: 0.16
Nodes (16): compute_remote_max_tokens(), has_complex_attachment(), is_code_generation_prompt(), is_creative_prompt(), is_greeting_or_tiny_chat(), is_local_capable_prompt(), is_local_trivial_whitelisted(), is_pure_greeting_request() (+8 more)

### Community 17 - "SklearnRouter"
Cohesion: 0.16
Nodes (8): _bootstrap_classifier(), Any, Create a LogisticRegression with hand-tuned weights for hackathon., Tier 2 ANGKOR router: 5-feature sklearn classifier + 3-zone routing + adaptive θ, SklearnRouter, test_sklearn_router_3_zone_detection(), test_sklearn_router_clear_local_for_trivial(), test_sklearn_router_clear_remote_for_code()

### Community 18 - "classify_prompt"
Cohesion: 0.18
Nodes (15): classify_prompt(), is_beneficial_to_decompose(), plan_request(), Backward-compatible alias for the conservative decomposition gate., Pre-classifier reader: reads the full prompt once via decide_mode(), then maps, Hierarchical orchestration planner.      Level-1: classify the WHOLE prompt (no, test_composite_character_level_single_route(), test_composite_prompt_pins_remote_single_task() (+7 more)

### Community 19 - "main.py"
Cohesion: 0.25
Nodes (8): HybridRoutingAgent, main(), _parse_args(), Any, Core orchestrator: middleware → router → ANGKOR+PHANTOM → inference → telemetry., End-to-end pipeline for token-efficient hybrid inference routing., TelemetryMetrics, Namespace

### Community 20 - "_seed_local_health"
Cohesion: 0.23
Nodes (14): adjust_prompt_for_remote(), distill_prompt(), is_local_unavailable(), Middle-layer prompt adjustment before remote inference.     Normalizes whitespac, Structure-preserving compression via local Ollama before remote inference., Convenience inverse of check_local_health (cached)., _patch_ollama(), A generic one-liner that drops tasks must fall back to the original. (+6 more)

### Community 21 - "_patch_post"
Cohesion: 0.24
Nodes (11): _patch_post(), FALLBACK_REMOTE result must carry the actual Fireworks model used., _reset_remote_validation_state(), test_dispatcher_matches_router_for_greeting(), test_greeting_no_ollama_round_trip(), test_remote_extracts_reasoning_content_when_message_content_missing(), test_remote_fallback_on_not_found_uses_next_candidate(), test_remote_malformed_200_falls_back_to_next_candidate() (+3 more)

### Community 22 - "had_prior_local_failure"
Cohesion: 0.32
Nodes (8): had_prior_local_failure(), mark_prior_local_failure(), _prompt_failure_key(), reset_prior_local_failures(), Non-greeting prompts still escalate after a prior local failure., test_greeting_ignores_prior_local_failure(), test_local_timeout_falls_back_to_remote(), test_prior_local_failure_forces_remote()

### Community 23 - "validate_remote_models"
Cohesion: 0.36
Nodes (7): fetch_accessible_model_ids(), load_validated_models(), _normalize(), Path, Startup validation for Fireworks remote model candidates., validate_remote_models(), test_validate_remote_models_filters_inaccessible()

### Community 24 - "heuristic_task_split"
Cohesion: 0.29
Nodes (7): heuristic_task_split(), Backward-compatible alias — delegates to the conservative splitter., task_dispatcher_never_calls_ollama(), test_heuristic_split_available_for_multi_task(), test_invalid_subtask_fragments_rejected(), test_math_heavy_prompt_not_exploded_into_swarm(), test_summarize_context_not_split_by_regex()

### Community 25 - "would_math_intercept"
Cohesion: 0.33
Nodes (6): is_trivial_fast_path(), Skip FAISS cache, PHANTOM race, planner, and cascade verify for trivial prompts., UI instant path: greetings and deterministic math., True when deterministic math eval would handle the prompt (no LLM)., should_skip_expensive_preprocess(), would_math_intercept()

### Community 26 - "is_heavy_local_model"
Cohesion: 0.50
Nodes (4): is_heavy_local_model(), _local_inference_timeout(), Strict read-timeout (seconds) for a local model, tighter when heavy., True for large local models (>= HEAVY_LOCAL_PARAM_BILLIONS params, e.g.     qwen

### Community 27 - "is_valid_subtask"
Cohesion: 0.50
Nodes (4): is_valid_subtask(), merge_short_fragments(), Reject garbage fragments from regex splitting., Merge fragments that fail is_valid_subtask into neighbors.

### Community 28 - "should_entropy_gate_input"
Cohesion: 0.50
Nodes (4): Gate gibberish / keyboard-mash prompts before any model call., should_entropy_gate_input(), compute_char_entropy(), Shannon entropy over characters — better for detecting random gibberish.

### Community 29 - "reset_local_health_cache"
Cohesion: 0.50
Nodes (4): Clear the cached health verdicts (used by tests and the UI refresh)., reset_local_health_cache(), _patch_get(), test_check_local_health_caches_verdict()

## Knowledge Gaps
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FeatureExtractor` connect `FeatureExtractor` to `RouteResult`, `config.py`, `BudgetEnforcer`, `app.py`, `test_router.py`, `CascadeVerifier`, `engine.py`, `RoutingEngine`, `TokenCounter`, `SklearnRouter`, `main.py`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `BudgetEnforcer` connect `BudgetEnforcer` to `RouteResult`, `config.py`, `app.py`, `test_router.py`, `phantom_calibrator.py`, `CascadeVerifier`, `main.py`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `SklearnRouter` connect `SklearnRouter` to `RouteResult`, `config.py`, `BudgetEnforcer`, `app.py`, `test_router.py`, `FeatureExtractor`, `CascadeVerifier`, `engine.py`, `RoutingEngine`, `TokenCounter`, `main.py`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Are the 20 inferred relationships involving `FeatureExtractor` (e.g. with `ClassificationResult` and `PlannerDecision`) actually correct?**
  _`FeatureExtractor` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `SklearnRouter` (e.g. with `ClassificationResult` and `PlannerDecision`) actually correct?**
  _`SklearnRouter` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `RouteResult` (e.g. with `SemanticCache` and `BudgetEnforcer`) actually correct?**
  _`RouteResult` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `TokenCounter` (e.g. with `BenchmarkReport` and `BenchmarkResult`) actually correct?**
  _`TokenCounter` has 16 INFERRED edges - model-reasoned connections that need verification._