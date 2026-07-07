#!/usr/bin/env python3
"""AMD Hackathon Track 1 batch entrypoint — read tasks, route via Fireworks, write results."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

# Batch defaults before importing the Streamlit-oriented router module.
os.environ.setdefault("SKIP_LOCAL", "true")

from app import (  # noqa: E402
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_COMPLEXITY_THRESHOLD,
    DEFAULT_LOCAL_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    RouteResult,
    configure_allowed_models,
    get_fireworks_api_key,
    process_user_request,
)

logger = logging.getLogger("run_batch")

INPUT_PATH = Path(os.getenv("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "/output/results.json"))
THRESHOLD = int(os.getenv("COMPLEXITY_THRESHOLD", str(DEFAULT_COMPLEXITY_THRESHOLD)))


class BatchTimeoutError(Exception):
    """Raised when the global batch alarm fires."""


def _batch_timeout_handler(signum: int, frame: object) -> None:
    raise BatchTimeoutError(f"Batch exceeded {BATCH_TIMEOUT_SECONDS:.0f}s limit")


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("tasks.json must be a JSON array")
    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Task at index {index} must be an object")
        task_id = item.get("task_id")
        prompt = item.get("prompt")
        if task_id is None or prompt is None:
            raise ValueError(f"Task at index {index} must include task_id and prompt")
        tasks.append({"task_id": task_id, "prompt": str(prompt)})
    return tasks


def _process_task(
    task_id: object,
    prompt: str,
    *,
    api_key: str,
    remote_model: str,
) -> dict[str, object]:
    started_msg = f"task_id={task_id!r} chars={len(prompt)}"
    logger.info("Processing %s", started_msg)

    def _run() -> RouteResult:
        return process_user_request(
            prompt,
            THRESHOLD,
            api_key,
            DEFAULT_LOCAL_MODEL,
            remote_model,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        try:
            result = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            logger.warning("Task timed out after %ss: %s", REQUEST_TIMEOUT_SECONDS, started_msg)
            return {
                "task_id": task_id,
                "answer": (
                    f"Error: request timed out after {REQUEST_TIMEOUT_SECONDS:.0f} seconds"
                ),
            }

    answer = result.answer.strip() if result.answer else ""
    if not result.success and answer.startswith("❌"):
        pass
    elif not result.success and not answer:
        answer = "Error: inference failed"

    logger.info(
        "Completed %s route=%s success=%s latency_ms=%.1f",
        started_msg,
        result.route,
        result.success,
        result.latency_ms,
    )
    return {"task_id": task_id, "answer": answer}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    api_key = get_fireworks_api_key()
    if not api_key:
        logger.error("FIREWORKS_API_KEY is required")
        return 1

    try:
        remote_model = configure_allowed_models(strict=True)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    try:
        tasks = _load_tasks(INPUT_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to read %s: %s", INPUT_PATH, exc)
        return 1

    if not tasks:
        logger.error("No tasks found in %s", INPUT_PATH)
        return 1

    logger.info(
        "Starting batch: tasks=%d remote_model=%s timeout_per_request=%ss batch_limit=%ss",
        len(tasks),
        remote_model,
        REQUEST_TIMEOUT_SECONDS,
        BATCH_TIMEOUT_SECONDS,
    )

    signal.signal(signal.SIGALRM, _batch_timeout_handler)
    signal.alarm(int(BATCH_TIMEOUT_SECONDS))

    results: list[dict[str, object]] = []
    try:
        for task in tasks:
            results.append(
                _process_task(
                    task["task_id"],
                    task["prompt"],
                    api_key=api_key,
                    remote_model=remote_model,
                )
            )
    except BatchTimeoutError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        signal.alarm(0)

    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        logger.error("Failed to write %s: %s", OUTPUT_PATH, exc)
        return 1

    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
