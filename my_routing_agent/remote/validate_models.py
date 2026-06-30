"""Startup validation for Fireworks remote model candidates."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import requests

logger = logging.getLogger("remote_validate")

FIREWORKS_MODELS_URL = "https://api.fireworks.ai/inference/v1/models"
DEFAULT_OUTPUT = Path("validated_model_list.json")


def _normalize(model_id: str) -> str:
    model_id = (model_id or "").strip()
    if not model_id:
        return ""
    if not model_id.startswith("accounts/"):
        return f"accounts/fireworks/models/{model_id}"
    return model_id


def fetch_accessible_model_ids(api_key: str, timeout: float = 8.0) -> set[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(FIREWORKS_MODELS_URL, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return {str(item.get("id", "")).strip() for item in data if item.get("id")}


def validate_remote_models(
    candidates: Iterable[str],
    api_key: str,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    output_path = output_path or DEFAULT_OUTPUT
    normalized = [_normalize(c) for c in candidates if c]
    unique = list(dict.fromkeys(normalized))

    if not api_key:
        logger.warning("No Fireworks API key; skipping remote model validation")
        payload = {
            "validated": unique,
            "removed": [],
            "reason": "missing_api_key",
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    try:
        accessible = fetch_accessible_model_ids(api_key)
    except requests.RequestException as exc:
        logger.warning("Fireworks metadata check failed: %s", exc)
        payload = {
            "validated": unique,
            "removed": [],
            "reason": f"metadata_error:{exc}",
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    validated = [m for m in unique if m in accessible]
    removed = [m for m in unique if m not in accessible]
    payload = {
        "validated": validated,
        "removed": removed,
        "accessible_count": len(accessible),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Validated remote models: kept=%d removed=%d",
        len(validated),
        len(removed),
    )
    return payload


def load_validated_models(path: Path | None = None) -> list[str]:
    path = path or DEFAULT_OUTPUT
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("validated", []))
    except Exception:
        return []
