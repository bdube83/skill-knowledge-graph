"""Vendor adapters for the skg run subcommand.

Each adapter exposes a single callable, ``chat(task: str, *, model: str | None
= None) -> VendorResponse``. The CLI looks adapters up by name in
``VENDORS``; adding a vendor is one registry entry. Adapters that depend on
external binaries return ``VendorResponse(text="", error=ERRNO_NOAVAIL)``
rather than raising, so the CLI can print a clean error and exit.

Three adapters ship today:
  * ``openai``: Uses the OpenAI Python SDK. Reads the key from
    ``~/.agent-proxy/openai-key`` (see ``eval/baseline_a_runner.py`` for the
    same pattern). Default model is ``gpt-4o-mini``.
  * ``claude``: Shells out to ``claude -p "<task>"``.
  * ``codex``: Shells out to ``codex exec --json "<task>"``.

The shell adapters look the binary up on PATH with ``shutil.which``; a
missing binary returns ERRNO_NOAVAIL and the CLI surfaces a clear message.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ERRNO_NOAVAIL = "vendor-not-available"
ERRNO_FAILED  = "vendor-call-failed"


@dataclass
class VendorResponse:
    """Result of one vendor call.

    ``text`` holds the model output. ``tokens_used`` is the total token
    count when the vendor reports it; otherwise None. ``error`` is empty
    on success and one of ``ERRNO_NOAVAIL`` / ``ERRNO_FAILED`` on failure.
    """

    text:        str
    tokens_used: int | None = None
    error:       str        = ""
    model:       str        = ""


# ---- OpenAI ----------------------------------------------------------------

_OPENAI_KEY_PATH = Path.home() / ".agent-proxy" / "openai-key"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def _read_openai_key() -> str | None:
    """Read the OpenAI key from the agent-proxy file. Return None if absent."""
    try:
        return _OPENAI_KEY_PATH.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None
    except OSError:
        return None


def call_openai(task: str, *, model: str | None = None) -> VendorResponse:
    """Send the task to OpenAI's Chat Completions API."""
    key = _read_openai_key()
    if not key:
        return VendorResponse(
            text="",
            error=ERRNO_NOAVAIL,
            model=model or _OPENAI_DEFAULT_MODEL,
        )
    try:
        from openai import OpenAI
    except ImportError:
        return VendorResponse(text="", error=ERRNO_NOAVAIL, model="")

    os.environ["OPENAI_API_KEY"] = key
    client = OpenAI()
    used_model = model or _OPENAI_DEFAULT_MODEL
    try:
        resp = client.chat.completions.create(
            model=used_model,
            messages=[
                {"role": "system", "content":
                    "You are a software engineering assistant. "
                    "Answer the user's task directly. No preamble."},
                {"role": "user", "content": task},
            ],
            temperature=0.0,
        )
    except Exception as exc:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {exc}", model=used_model)

    body  = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    total = None
    if usage is not None:
        total = int(getattr(usage, "total_tokens", 0)) or None
    return VendorResponse(text=body, tokens_used=total, model=used_model)


# ---- Claude (shell) --------------------------------------------------------

def call_claude(task: str, *, model: str | None = None) -> VendorResponse:  # noqa: ARG001
    """Invoke the ``claude`` CLI binary with ``-p``."""
    bin_path = shutil.which("claude")
    if not bin_path:
        return VendorResponse(text="", error=ERRNO_NOAVAIL, model="claude")
    try:
        proc = subprocess.run(
            [bin_path, "-p", task],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: timeout", model="claude")
    except OSError as exc:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {exc}", model="claude")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {msg[:200]}", model="claude")

    return VendorResponse(text=(proc.stdout or "").rstrip("\n"), model="claude")


# ---- Codex (shell) ---------------------------------------------------------

def call_codex(task: str, *, model: str | None = None) -> VendorResponse:  # noqa: ARG001
    """Invoke the ``codex exec --json`` CLI."""
    bin_path = shutil.which("codex")
    if not bin_path:
        return VendorResponse(text="", error=ERRNO_NOAVAIL, model="codex")
    try:
        proc = subprocess.run(
            [bin_path, "exec", "--json", task],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: timeout", model="codex")
    except OSError as exc:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {exc}", model="codex")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {msg[:200]}", model="codex")

    raw = (proc.stdout or "").strip()
    text = raw
    tokens = None
    # Try to extract a string field from the JSON envelope; fall back to raw.
    try:
        last_line = raw.splitlines()[-1] if raw else ""
        parsed = json.loads(last_line) if last_line else {}
        if isinstance(parsed, dict):
            for key in ("output", "text", "response", "content"):
                if isinstance(parsed.get(key), str):
                    text = parsed[key]
                    break
            usage = parsed.get("usage") or {}
            if isinstance(usage, dict) and "total_tokens" in usage:
                tokens = int(usage["total_tokens"])
    except (json.JSONDecodeError, ValueError, IndexError):
        pass

    return VendorResponse(text=text, tokens_used=tokens, model="codex")


# ---- Copilot (shell, gh extension) -----------------------------------------

def call_copilot(task: str, *, model: str | None = None) -> VendorResponse:  # noqa: ARG001
    """Invoke GitHub Copilot in non-interactive programmatic mode.

    Uses the standalone ``copilot`` binary with the ``-p / --prompt``
    flag (programmatic mode). The legacy ``gh copilot suggest`` flow
    is interactive only and would block a subprocess call, so we do
    not use it here. ``--allow-all-tools`` skips the per-tool approval
    prompt that the binary would otherwise issue.

    Authentication: relies on the Copilot CLI's existing OAuth login,
    or a fine-grained PAT in ``GH_TOKEN`` / ``GITHUB_TOKEN`` with the
    "Copilot Requests" permission.

    Returns ERRNO_NOAVAIL when the ``copilot`` binary is absent.
    """
    bin_path = shutil.which("copilot")
    if not bin_path:
        return VendorResponse(text="", error=ERRNO_NOAVAIL, model="copilot")
    try:
        proc = subprocess.run(
            [bin_path, "-p", task, "--allow-all-tools"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: timeout", model="copilot")
    except OSError as exc:
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {exc}", model="copilot")

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        return VendorResponse(text="", error=f"{ERRNO_FAILED}: {msg[:200]}", model="copilot")

    return VendorResponse(text=(proc.stdout or "").rstrip("\n"), model="copilot")


# ---- Registry --------------------------------------------------------------

VendorFn = Callable[..., VendorResponse]

VENDORS: dict[str, VendorFn] = {
    "openai":  call_openai,
    "claude":  call_claude,
    "codex":   call_codex,
    "copilot": call_copilot,
}


def get_vendor(name: str) -> VendorFn:
    """Look up a vendor adapter by name. Raise KeyError if unknown."""
    try:
        return VENDORS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown vendor '{name}'. Known: {sorted(VENDORS)}"
        ) from exc
