"""PHANTOM ensemble calibration: ROC-AUC + logistic regression abort rule."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from my_routing_agent.phantom.generation_signals import (
    GenerationSignalRecord,
    record_generation_signals,
    signal_vector_at_token,
)

GOOD_PROMPTS = [
    "hi", "hello there", "thanks", "what is 2+2", "capital of france",
    "who wrote hamlet", "good morning", "how are you", "bye", "see you",
]
BAD_PROMPTS = [
    "spell strawberry backward", "count r's in strawberry", "write 9+10 answer backward",
    "what is the detailed history of the roman empire from founding to fall",
    "explain quantum field theory with full mathematical derivation",
    "analyze this image and extract every object label",  # image-mixed proxy
    "reverse the string supercalifragilisticexpialidocious",
    "how many letters in Mississippi", "anagram of listen", "nth character of alphabet soup",
]


def _expand_prompts(seed_list: list[str], target: int, rng: random.Random) -> list[str]:
    out = list(seed_list)
    while len(out) < target:
        base = rng.choice(seed_list)
        suffix = rng.choice([" please", " quickly", " now", " for me", ""])
        out.append(f"{base}{suffix}")
    return out[:target]


def collect_records(
    model: str,
    *,
    n_good: int = 1000,
    n_bad: int = 1000,
    seed: int = 42,
    synthetic: bool = True,
) -> list[GenerationSignalRecord]:
    from my_routing_agent.phantom.generation_signals import (
        GenerationSignalRecord,
        _synthetic_signals,
        record_generation_signals,
    )

    rng = random.Random(seed)
    goods = _expand_prompts(GOOD_PROMPTS, n_good, rng)
    bads = _expand_prompts(BAD_PROMPTS, n_bad, rng)
    records: list[GenerationSignalRecord] = []
    for prompt in goods:
        if synthetic:
            records.append(
                GenerationSignalRecord(
                    prompt=prompt,
                    label="good",
                    signals=_synthetic_signals(prompt, "good"),
                    source="synthetic",
                )
            )
        else:
            records.append(record_generation_signals(prompt, model, label="good"))
    for prompt in bads:
        if synthetic:
            records.append(
                GenerationSignalRecord(
                    prompt=prompt,
                    label="bad",
                    signals=_synthetic_signals(prompt, "bad"),
                    source="synthetic",
                )
            )
        else:
            records.append(record_generation_signals(prompt, model, label="bad"))
    return records


def calibrate_ensemble(
    records: list[GenerationSignalRecord],
    *,
    token_index: int = 8,
    abort_threshold: float = 0.25,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X = np.array([signal_vector_at_token(r, token_index) for r in records], dtype=np.float64)
    y = np.array([1 if r.label == "good" else 0 for r in records], dtype=np.int32)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)[:, 1]

    # Per-signal ROC-AUC (one-vs-rest style on each feature)
    signal_names = ["shannon_entropy", "top1_probability", "top3_probability_sum", "max_repeat_streak"]
    per_signal_auc: dict[str, float] = {}
    for i, name in enumerate(signal_names):
        try:
            per_signal_auc[name] = float(roc_auc_score(y, X[:, i]))
        except ValueError:
            per_signal_auc[name] = 0.5
    ensemble_auc = float(roc_auc_score(y, proba))

    weights = {
        name: float(coef)
        for name, coef in zip(signal_names, clf.coef_[0])
    }
    weights["intercept"] = float(clf.intercept_[0])

    def ensemble_score(vector: list[float]) -> float:
        arr = np.array(vector, dtype=np.float64).reshape(1, -1)
        return float(clf.predict_proba(arr)[0, 1])

    return {
        "token_index": token_index,
        "abort_threshold": abort_threshold,
        "production_rule": (
            f"abort if ensemble_score < {abort_threshold} at token {token_index}, else continue"
        ),
        "weights": weights,
        "per_signal_roc_auc": per_signal_auc,
        "ensemble_roc_auc": ensemble_auc,
        "samples": len(records),
        "ensemble_score_fn": ensemble_score,
    }


def should_abort(vector: list[float], report: dict[str, Any]) -> bool:
    score_fn = report.get("ensemble_score_fn")
    if callable(score_fn):
        return score_fn(vector) < float(report["abort_threshold"])
    # Stateless fallback using saved weights
    w = report["weights"]
    names = ["shannon_entropy", "top1_probability", "top3_probability_sum", "max_repeat_streak"]
    z = w["intercept"] + sum(w[n] * v for n, v in zip(names, vector))
    prob = 1.0 / (1.0 + np.exp(-z))
    return prob < float(report["abort_threshold"])


def run_phantom_calibration(
    model: str,
    out_path: Path,
    *,
    n_good: int = 1000,
    n_bad: int = 1000,
    synthetic: bool = True,
) -> dict[str, Any]:
    records = collect_records(model, n_good=n_good, n_bad=n_bad, synthetic=synthetic)
    report = calibrate_ensemble(records)
    serializable = {k: v for k, v in report.items() if k != "ensemble_score_fn"}
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return report
