"""
GitHub push webhook → Telegram notifier.

Receives GitHub "push" events and posts a formatted summary to a Telegram channel.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("github-telegram-webhook")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
MAX_COMMITS = 5

app = FastAPI(
    title="GitHub → Telegram Webhook",
    description="Forwards GitHub push events to a Telegram channel.",
)


def _telegram_send_url() -> str | None:
    if not TELEGRAM_TOKEN:
        return None
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"


def _branch_from_ref(ref: str) -> str:
    """refs/heads/main -> main"""
    if ref.startswith("refs/heads/"):
        return ref.removeprefix("refs/heads/")
    return ref or "unknown"


def _first_line(message: str) -> str:
    line = (message or "").strip().splitlines()[0] if message else ""
    return line[:200] if line else "(no message)"


def _escape_markdown(text: str) -> str:
    """Escape characters that break Telegram legacy Markdown."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def _format_push_message(payload: dict[str, Any]) -> str:
    repo_name = payload.get("repository", {}).get("name", "unknown-repo")
    pusher = payload.get("pusher", {}).get("name") or payload.get("pusher", {}).get(
        "username", "unknown"
    )
    branch = _branch_from_ref(payload.get("ref", ""))
    commits: list[dict[str, Any]] = payload.get("commits") or []

    lines = [
        f"🚀 *Push to* `{_escape_markdown(repo_name)}`",
        f"👤 *Pusher:* {_escape_markdown(pusher)}",
        f"🌿 *Branch:* `{_escape_markdown(branch)}`",
        "",
        f"*Commits* ({min(len(commits), MAX_COMMITS)} shown):",
    ]

    for commit in commits[-MAX_COMMITS:]:
        author = (
            commit.get("author", {}).get("name")
            or commit.get("committer", {}).get("name")
            or "unknown"
        )
        message = _escape_markdown(_first_line(commit.get("message", "")))
        lines.append(f"• {_escape_markdown(author)}: {message}")

    if len(commits) > MAX_COMMITS:
        lines.append(f"_+{len(commits) - MAX_COMMITS} more commit(s) omitted_")

    return "\n".join(lines)


def _send_telegram_message(text: str) -> None:
    url = _telegram_send_url()
    if not url:
        raise RuntimeError("TELEGRAM_TOKEN is not set")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set")

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
        timeout=15,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body}")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def github_webhook(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        logger.error("Failed to parse webhook JSON: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "Invalid JSON payload"},
        )

    if not isinstance(payload, dict):
        logger.warning("Webhook payload is not a JSON object; ignoring.")
        return JSONResponse(content={"ok": True, "ignored": "invalid payload type"})

    event = request.headers.get("X-GitHub-Event", "unknown")
    commits = payload.get("commits")

    if event != "push" or not commits:
        logger.info("Ignoring non-push event (event=%s, has_commits=%s)", event, bool(commits))
        return JSONResponse(content={"ok": True, "ignored": f"event={event}"})

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variable")
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Telegram credentials not configured"},
        )

    try:
        message = _format_push_message(payload)
        await asyncio.to_thread(_send_telegram_message, message)
        repo = payload.get("repository", {}).get("name", "unknown")
        logger.info(
            "Telegram notification sent for push to %s (%d commit(s))",
            repo,
            len(commits),
        )
        return JSONResponse(content={"ok": True, "sent": True})
    except requests.RequestException as exc:
        logger.error("Telegram request failed: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"ok": False, "error": "Failed to reach Telegram API"},
        )
    except RuntimeError as exc:
        logger.error("Telegram API rejected message: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"ok": False, "error": str(exc)},
        )
    except Exception as exc:
        logger.error("Unexpected error handling webhook: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Internal server error"},
        )
