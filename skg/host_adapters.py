"""Real adapter implementations for SKG custom host imports.

Phase 3e of the WASM import-level enforcement work. The Phase 3b host
wrappers in `skg/host_imports.py` validate the per-run handle table on
every call. This module replaces selected stub bodies with real
operations against `urllib`. The wrapper still owns the security
checks; the adapter only runs after scope validation passes.

What is implemented here:
  - skg.http_get: real HTTP GET via urllib.request
  - skg.http_post: real HTTP POST via urllib.request

Everything else stays in `skg/host_imports.py` as a stub, returning
`{"stub": true, "effect": "..."}`. Real adapters for git, browser,
external-channel, secret, and production write are downstream work
because each requires either subprocess integration, MCP, vault
access, or approval-flow infrastructure.

Test infrastructure: tests use `http.server.HTTPServer` on
`127.0.0.1` to avoid real network. The wrappers call urllib against
that loopback host. URL pattern matching is what `HandleTable.validate`
already does; the adapter only fetches.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

from wasmtime import Caller


HTTP_TIMEOUT_S = 5


# Errno values shared with skg.host_imports
ERRNO_SUCCESS  = 0
ERRNO_DENIED   = 13
ERRNO_INVAL    = 28
ERRNO_IO       = 29


def http_get_body(url: str) -> tuple[int, bytes]:
    """Fetch a URL via HTTP GET. Returns (errno, body_bytes)."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return ERRNO_SUCCESS, resp.read()
    except (urllib.error.URLError, ValueError, TimeoutError):
        return ERRNO_IO, b""


def http_post_body(url: str, body: bytes, content_type: str) -> tuple[int, bytes]:
    """Send an HTTP POST. Returns (errno, response_body_bytes)."""
    try:
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            return ERRNO_SUCCESS, resp.read()
    except (urllib.error.URLError, ValueError, TimeoutError):
        return ERRNO_IO, b""


def real_http_get(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.http_get. Returns (errno, response body)."""
    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return ERRNO_INVAL, b""
    return http_get_body(url)


def real_http_post(payload: dict) -> tuple[int, bytes]:
    """Adapter for skg.http_post. Returns (errno, response body)."""
    url = payload.get("url")
    if not isinstance(url, str) or not url:
        return ERRNO_INVAL, b""
    body_value = payload.get("body", "")
    if isinstance(body_value, str):
        body_bytes = body_value.encode("utf-8")
    elif isinstance(body_value, dict) or isinstance(body_value, list):
        body_bytes = json.dumps(body_value).encode("utf-8")
    else:
        return ERRNO_INVAL, b""
    content_type = payload.get("content_type", "application/json")
    if not isinstance(content_type, str):
        content_type = "application/json"
    return http_post_body(url, body_bytes, content_type)


# Registry mapping qualified host import name to a function that takes
# the decoded JSON payload and returns (errno, response_body).
ADAPTERS: dict[str, Callable[[dict], tuple[int, bytes]]] = {
    "skg.http_get":  real_http_get,
    "skg.http_post": real_http_post,
}


def has_adapter(qualified_name: str) -> bool:
    """Return True if a real adapter exists for this host import."""
    return qualified_name in ADAPTERS
