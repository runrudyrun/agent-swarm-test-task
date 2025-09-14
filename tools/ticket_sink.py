"""Optional ticket sink to post structured tickets to an external webhook.

If SUPPORT_WEBHOOK_URL is not configured, post_ticket() becomes a no-op and
returns an empty dict.
"""
from __future__ import annotations

import os
import logging
from typing import Dict, Any

import httpx

logger = logging.getLogger(__name__)


def post_ticket(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Post ticket payload to an external webhook if configured.

    Reads SUPPORT_WEBHOOK_URL and SUPPORT_WEBHOOK_TOKEN from environment.
    Returns a dict, possibly containing {"remote_id": str, "status": str}.

    In absence of configuration or on failure, returns {} and logs a warning.
    """
    url = os.getenv("SUPPORT_WEBHOOK_URL")
    token = os.getenv("SUPPORT_WEBHOOK_TOKEN")

    if not url:
        # No sink configured; noop
        return {}

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=6.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            # Normalize a couple of common fields
            remote_id = (
                data.get("id")
                or data.get("ticket_id")
                or data.get("remote_id")
            )
            status = data.get("status") or "posted"
            result = {}
            if remote_id:
                result["remote_id"] = str(remote_id)
            result["status"] = status
            return result
    except Exception as e:
        logger.warning(f"post_ticket() failed: {e}")
        return {}
