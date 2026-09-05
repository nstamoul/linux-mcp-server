"""Shared helpers for gatekeeper HTTP clients."""

import json

from typing import Any

import httpx


DEFAULT_TIMEOUT_SECONDS = 120

HTTP_CLIENT = httpx.AsyncClient()


class GatekeeperHTTPError(RuntimeError):
    """Raised when an LLM provider returns an error response."""

    def __init__(self, provider: str, status_code: int, body: str):
        snippet = body[:500] + ("..." if len(body) > 500 else "")
        super().__init__(f"{provider} API error ({status_code}): {snippet}")
        self.provider = provider
        self.status_code = status_code
        self.body = body


async def post_json(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    response = await HTTP_CLIENT.post(url, headers=headers, json=body, timeout=timeout)
    if not response.is_success:
        raise GatekeeperHTTPError(provider, response.status_code, response.text)
    return response.json()


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Decode an SSE response body into its `data:` JSON payloads, in order.

    Skips the terminal `data: [DONE]` sentinel and any non-JSON data lines.
    """
    events = []
    for raw_event in text.split("\n\n"):
        data_lines = [line[len("data:") :].strip() for line in raw_event.splitlines() if line.startswith("data:")]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except ValueError:
            continue
    return events


async def post_maybe_sse(
    *,
    provider: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    """Like `post_json`, but tolerates a `text/event-stream` response.

    Some Responses-API-compatible backends (e.g. a self-hosted codex-lb
    proxy) stream via SSE unconditionally, ignoring `stream: false` in the
    request body. Returns `(json_body, None)` for a normal JSON response, or
    `(None, events)` for an SSE response - callers that know how to
    reassemble the specific event schema they expect can do so from `events`.
    """
    response = await HTTP_CLIENT.post(url, headers=headers, json=body, timeout=timeout)
    if not response.is_success:
        raise GatekeeperHTTPError(provider, response.status_code, response.text)
    if "text/event-stream" in response.headers.get("content-type", ""):
        return None, parse_sse_events(response.text)
    return response.json(), None
