#!/usr/bin/env python3
"""Generate test_cases.json — 200 routing agent test cases across 5 categories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_test_suite import TestCase, generate_sample_cases  # noqa: E402


def case_to_dict(case: TestCase) -> dict[str, object]:
    item: dict[str, object] = {
        "task_id": case.task_id,
        "prompt": case.prompt,
        "category": case.category,
        "difficulty": case.difficulty,
    }
    if case.tags:
        item["tags"] = case.tags
    if case.answer_type:
        item["answer_type"] = case.answer_type
    if case.expected_answer:
        item["expected_answer"] = case.expected_answer
    return item


def generate_test_cases(count: int = 200) -> list[dict[str, object]]:
    """Return the full test suite as JSON-serializable dicts."""
    return [case_to_dict(c) for c in generate_sample_cases(count)]


def validate_cases(cases: list[dict[str, object]]) -> None:
    """Basic sanity checks against testcase-instruction.md."""
    expected_counts = {
        "math": 50,
        "logic": 35,
        "trivia": 40,
        "facts": 40,
        "coding": 35,
    }
    expected_difficulty = {
        "easy": 40,
        "medium": 80,
        "hard": 60,
        "expert": 20,
    }
    from collections import Counter

    categories = Counter(str(c.get("category", "")) for c in cases)
    difficulties = Counter(str(c.get("difficulty", "")) for c in cases)
    task_ids = [str(c.get("task_id", "")) for c in cases]

    if len(cases) != 200:
        raise ValueError(f"Expected 200 cases, got {len(cases)}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task_id values found")
    for cat, n in expected_counts.items():
        if categories[cat] != n:
            raise ValueError(f"Category {cat}: expected {n}, got {categories[cat]}")
    for diff, n in expected_difficulty.items():
        if difficulties[diff] != n:
            raise ValueError(f"Difficulty {diff}: expected {n}, got {difficulties[diff]}")

    prefix_ranges = {
        "M": ("M001", "M050"),
        "L": ("L001", "L035"),
        "T": ("T001", "T040"),
        "F": ("F001", "F040"),
        "C": ("C001", "C035"),
    }
    for prefix, (start, end) in prefix_ranges.items():
        ids = sorted(tid for tid in task_ids if tid.startswith(prefix))
        if not ids:
            raise ValueError(f"No cases with prefix {prefix}")
        if ids[0] != start or ids[-1] != end:
            raise ValueError(f"Prefix {prefix}: expected {start}-{end}, got {ids[0]}-{ids[-1]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AMD routing test_cases.json")
    parser.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "tests" / "test_cases.json"),
        help="Output JSON path (default: tests/test_cases.json)",
    )
    parser.add_argument("--count", type=int, default=200, help="Number of cases to generate")
    args = parser.parse_args()

    cases = generate_test_cases(args.count)
    validate_cases(cases)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} test cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
