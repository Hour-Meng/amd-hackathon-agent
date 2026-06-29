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
from fastapi import FastAPI, Request, Response, status
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
    return (ref or "refs/heads/main").split("/")[-1]


def _first_line(message: str) -> str:
    line = (message or "").strip().splitlines()[0] if message else ""
    return line[:200] if line else "No message"


def _escape_markdown(text: str) -> str:
    """Escape characters that break Telegram legacy Markdown."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text


def _format_push_message(payload: dict[str, Any]) -> str:
    repo_name = payload.get("repository", {}).get("name", "Unknown Repo")
    pusher = payload.get("pusher", {}).get("name", "Someone")
    branch = _branch_from_ref(payload.get("ref", "refs/heads/main"))
    commits: list[dict[str, Any]] = payload.get("commits") or []

    message = f"🚀 *New Push to {_escape_markdown(repo_name)}* \\[{_escape_markdown(branch)}\\]\n"
    message += f"👤 *By:* {_escape_markdown(pusher)}\n\n"
    message += "📝 *Commits:*\n"

    for commit in commits[:MAX_COMMITS]:
        msg = _escape_markdown(_first_line(commit.get("message", "No message")))
        author = _escape_markdown(
            commit.get("author", {}).get("name", "Unknown")
        )
        message += f"• {msg} \\(- {author}\\)\n"

    if len(commits) > MAX_COMMITS:
        message += f"_+{len(commits) - MAX_COMMITS} more commit(s) omitted_\n"

    return message.rstrip()


def _send_telegram_message(text: str) -> requests.Response:
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
    logger.info("Telegram response: %s - %s", response.status_code, response.text)
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API error: {body}")
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/webhook")
async def github_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        logger.error("Failed to parse webhook JSON: %s", exc)
        return Response(
            content="Invalid JSON",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    if not isinstance(payload, dict):
        logger.warning("Webhook payload is not a JSON object; ignoring.")
        return {"status": "processed"}

    # Handle GitHub's initial setup ping event safely
    if "zen" in payload:
        logger.info("Received GitHub Ping Event!")
        return {"status": "ping received successfully"}

    # Check if it's an actual push event
    commits = payload.get("commits")
    if commits:
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variable")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "error": "Telegram credentials not configured"},
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
        except requests.RequestException as exc:
            logger.error("Failed to reach Telegram API: %s", exc)
        except RuntimeError as exc:
            logger.error("Telegram API rejected message: %s", exc)
        except Exception as exc:
            logger.error("Unexpected error handling webhook: %s", exc)

    return {"status": "processed"}
