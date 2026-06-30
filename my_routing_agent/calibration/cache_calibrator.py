"""Cache threshold calibration from labeled query pairs."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from my_routing_agent.config import CacheConfig
from my_routing_agent.middleware.text_preprocess import preprocess_for_cache

logger = logging.getLogger("cache_calibrator")

try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class ThresholdMetrics:
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


def load_label_pairs(labels_path: Path) -> list[tuple[str, str, bool]]:
    pairs: list[tuple[str, str, bool]] = []
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = (row.get("query_a") or row.get("query1") or "").strip()
            b = (row.get("query_b") or row.get("query2") or "").strip()
            label_raw = (row.get("identical_intent") or row.get("label") or "0").strip()
            if not a or not b:
                continue
            pairs.append((a, b, label_raw in {"1", "true", "True", "yes"}))
    return pairs


def load_cached_queries(cache_store_path: Path, calibration_path: Path | None) -> list[str]:
    queries: list[str] = []
    if cache_store_path.exists():
        try:
            raw = json.loads(cache_store_path.read_text(encoding="utf-8"))
            for entry in raw.values():
                q = (entry.get("metadata") or {}).get("query") or entry.get("query")
                if q:
                    queries.append(str(q))
        except Exception as exc:
            logger.warning("Could not read cache store: %s", exc)
    if calibration_path and calibration_path.exists():
        try:
            items = json.loads(calibration_path.read_text(encoding="utf-8"))
            for item in items:
                q = item.get("query")
                if q:
                    queries.append(str(q))
        except Exception as exc:
            logger.warning("Could not read calibration data: %s", exc)
    return list(dict.fromkeys(queries))


def embed_queries(
    queries: list[str],
    model_name: str,
) -> dict[str, np.ndarray]:
    if not ST_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for cache calibration")
    encoder = SentenceTransformer(model_name)
    cleaned = [preprocess_for_cache(q) for q in queries]
    vectors = encoder.encode(cleaned, normalize_embeddings=True)
    return {q: np.asarray(v, dtype=np.float32) for q, v in zip(queries, vectors)}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def sweep_thresholds(
    pairs: list[tuple[str, str, bool]],
    embeddings: dict[str, np.ndarray],
    *,
    start: float = 0.80,
    stop: float = 0.95,
    step: float = 0.01,
) -> list[ThresholdMetrics]:
    rows: list[ThresholdMetrics] = []
    threshold = start
    while threshold <= stop + 1e-9:
        tp = fp = fn = tn = 0
        for qa, qb, same_intent in pairs:
            if qa not in embeddings or qb not in embeddings:
                continue
            sim = cosine_similarity(embeddings[qa], embeddings[qb])
            predicted_same = sim >= threshold
            if same_intent and predicted_same:
                tp += 1
            elif same_intent and not predicted_same:
                fn += 1
            elif not same_intent and predicted_same:
                fp += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append(
            ThresholdMetrics(
                threshold=round(threshold, 2),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                false_positive_rate=round(fpr, 4),
            )
        )
        threshold = round(threshold + step, 2)
    return rows


def recommend_threshold(rows: list[ThresholdMetrics], max_fpr: float = 0.01) -> ThresholdMetrics | None:
    eligible = [r for r in rows if r.false_positive_rate < max_fpr]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r.f1)


def write_threshold_csv(rows: list[ThresholdMetrics], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threshold",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.threshold,
                    row.true_positives,
                    row.false_positives,
                    row.false_negatives,
                    row.precision,
                    row.recall,
                    row.f1,
                ]
            )


def run_cache_calibration(
    *,
    labels_path: Path,
    cache_store_path: Path,
    calibration_path: Path | None,
    sweep_csv_path: Path,
    report_json_path: Path,
    config: CacheConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CacheConfig()
    pairs = load_label_pairs(labels_path)
    if not pairs:
        raise ValueError(f"No labeled pairs found in {labels_path}")

    cached_queries = load_cached_queries(cache_store_path, calibration_path)
    unique_queries = list(dict.fromkeys([q for q, _, _ in pairs for q in (q,)] + cached_queries))
    embeddings = embed_queries(unique_queries, cfg.model_name)
    rows = sweep_thresholds(pairs, embeddings)
    recommended = recommend_threshold(rows)

    write_threshold_csv(rows, sweep_csv_path)

    report: dict[str, Any] = {
        "model": cfg.model_name,
        "labeled_pairs": len(pairs),
        "cached_queries": len(cached_queries),
        "recommended_threshold": recommended.threshold if recommended else cfg.threshold,
        "recommended_f1": recommended.f1 if recommended else None,
        "recommended_fpr": recommended.false_positive_rate if recommended else None,
        "candidate_range": [],
        "config_suggestion": {},
        "micro_report": "",
    }
    if recommended:
        lo = round(max(0.0, recommended.threshold - 0.02), 2)
        hi = round(min(1.0, recommended.threshold + 0.02), 2)
        report["candidate_range"] = [lo, hi]
        report["config_suggestion"] = {
            "CACHE_THRESHOLD": recommended.threshold,
            "CACHE_CANDIDATE_RANGE": [lo, hi],
        }
        report["micro_report"] = (
            f"Recommended CACHE_THRESHOLD={recommended.threshold:.2f} "
            f"(F1={recommended.f1:.3f}, FPR={recommended.false_positive_rate:.3%}). "
            f"Runtime soft-check candidate-range=[{lo:.2f}, {hi:.2f}]."
        )
    else:
        report["micro_report"] = (
            "No threshold met FPR < 1%; keeping default CACHE_THRESHOLD="
            f"{cfg.threshold:.2f}."
        )

    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
