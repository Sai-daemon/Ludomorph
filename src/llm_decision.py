"""
LLM Decision Call — Phase 2.7

Sends the assembled prompt (from Phase 2.6's ``build_llm_prompt``) to
Ollama's native ``/api/chat`` endpoint, enforces macro name constraints via
a dynamic Pydantic JSON schema, and handles timeouts/errors with a safe
fallback chain.

Key behaviours per the spec:

* Uses ``get_macro_model()`` to create a ``Literal``-constrained Pydantic
  model whose ``model_json_schema()`` is passed as the ``format`` parameter.
* httpx timeout at 200 ms via ``asyncio.wait_for``.
* Falls back to ``last_action`` or ``"WAIT"`` on any failure (timeout,
  non‑JSON, invalid macro, connection error).
* URL resolution: strips ``/v1`` suffix from ``ollama_url`` (the native
  ``/api/chat`` endpoint lives at the base URL).

Usage::

    from src.llm_decision import call_llm_decision

    action = await call_llm_decision(
        messages=prompt_messages,
        profile_macros=available_macros,
        config=config,
        last_action="ATTACK",
    )
    # → "ATTACK", "USE_POTION", "WAIT", etc.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import httpx

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default fallback action when nothing else is available.
_DEFAULT_FALLBACK: str = "WAIT"

# Regex to extract a JSON object from garbled LLM output (fallback parsing).
_JSON_EXTRACT_RE: re.Pattern[str] = re.compile(
    r'\{\s*"action"\s*:\s*"([^"]+)"\s*\}'
)

# ---------------------------------------------------------------------------
# Public: get_macro_model
# ---------------------------------------------------------------------------


def get_macro_model(profile_macros: list[dict[str, Any]]) -> Any:
    """Dynamically create a Pydantic model whose ``action`` field is restricted
    to the macro names listed in *profile_macros*.

    The returned model class can be used with ``.model_json_schema()`` to
    generate the ``format`` parameter for Ollama's ``/api/chat`` endpoint,
    guaranteeing that the LLM output is a valid macro name.

    Args:
        profile_macros: List of macro dicts, each with at least a ``"name"``
            key.  Same structure as the ``"macros"`` value in ``macros.json``.

    Returns:
        A Pydantic ``BaseModel`` subclass with a single ``action`` field of
        type ``Literal[macro_name, ...]``.

    Raises:
        ImportError: If ``pydantic`` is not installed.
    """
    # Deferred import — pydantic is only needed at runtime for this function.
    from pydantic import BaseModel, create_model  # type: ignore[import-untyped]
    from typing import Literal  # noqa: F811  (re‑import for create_model runtime)

    macro_names: list[str] = [m["name"] for m in profile_macros]
    if not macro_names:
        macro_names = [_DEFAULT_FALLBACK]

    # The Literal tuple must be a flat tuple of string values.
    return create_model(
        "MacroAction",
        action=(Literal[tuple(macro_names)], ...),  # type: ignore[valid-type]
        __base__=BaseModel,
    )


# ---------------------------------------------------------------------------
# Public: parse_llm_response
# ---------------------------------------------------------------------------


def parse_llm_response(
    response_text: str,
    valid_macros: set[str],
) -> str | None:
    """Parse a raw LLM response string into a validated macro name.

    Tries strict JSON parsing first, then falls back to regex extraction.
    Returns ``None`` if the response cannot be mapped to a valid macro.

    Args:
        response_text: The raw ``content`` string from the LLM response
            (expected to be a JSON string like ``'{"action":"ATTACK"}'``).
        valid_macros: Set of allowed macro names (case‑sensitive).

    Returns:
        The validated macro name, or ``None``.
    """
    if not response_text or not response_text.strip():
        return None

    text = response_text.strip()

    # 1) Strict JSON parse
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict) and "action" in parsed:
        action = str(parsed["action"]).strip()
        if action in valid_macros:
            return action
        else:
            logger.debug(
                "LLM returned unknown action %r (allowed: %s)",
                action,
                sorted(valid_macros),
            )

    # 2) Fallback regex extraction
    match = _JSON_EXTRACT_RE.search(text)
    if match:
        action = match.group(1).strip()
        if action in valid_macros:
            logger.debug("Regex-extracted action %r from garbled LLM response", action)
            return action

    # 3) If the entire string is a single word that matches a macro
    single_word = text.split()[0].strip().upper()
    if single_word in valid_macros:
        logger.debug("Single-word fallback action %r", single_word)
        return single_word

    return None


# ---------------------------------------------------------------------------
# Public: call_llm_decision
# ---------------------------------------------------------------------------


async def call_llm_decision(
    messages: list[dict[str, str]],
    profile_macros: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    last_action: str | None = None,
    timeout: float = 0.200,
) -> str:
    """Send messages to Ollama and return the chosen macro name.

    This is the primary entry point for Phase 2.7.  It builds a JSON-schema
    constraint from *profile_macros*, calls Ollama's native ``/api/chat`` with
    a hard 200 ms timeout, parses the response, and falls back on failure.

    Args:
        messages: Message list from ``build_llm_prompt()`` (system + user).
        profile_macros: Available macro definitions (same format as
            ``macros.json["macros"]``).  Must not be empty.
        config: Global config dict (must contain ``ollama_url`` and
            ``ollama_model`` keys).
        last_action: The action chosen in the previous decision cycle.
            Used as the first fallback when the LLM call fails.
        timeout: Maximum time in seconds to wait for Ollama's response.
            Default 200 ms per the spec.

    Returns:
        A validated macro name from the LLM response, *last_action*, or
        ``"WAIT"``.
    """
    # Resolve Ollama endpoint
    ollama_url: str = config.get("ollama_url", "http://localhost:11434/v1")
    model: str = config.get("ollama_model", "")
    base_url = _strip_openai_suffix(ollama_url)

    # Collect valid macro names for validation
    valid_names: set[str] = {m["name"] for m in profile_macros} if profile_macros else set()
    if not valid_names:
        valid_names = {_DEFAULT_FALLBACK}

    # Build JSON schema constraint
    try:
        MacroAction = get_macro_model(profile_macros)
        schema = MacroAction.model_json_schema()
    except ImportError:
        logger.error("pydantic is required for Phase 2.7 LLM decision calls")
        return last_action or _DEFAULT_FALLBACK
    except Exception as exc:
        logger.error("Failed to build macro schema: %s", exc)
        return last_action or _DEFAULT_FALLBACK

    # Build request payload
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_predict": 20,
            "num_ctx": 4096,
        },
        "stream": False,
    }

    logger.debug(
        "Calling Ollama /api/chat — model=%r, macros=%s, timeout=%.0fms",
        model,
        sorted(valid_names),
        timeout * 1000,
    )

    # Connection timeout from config (generous, only applies to connect+read).
    # The decision-timeout is enforced by asyncio.wait_for below.
    httpx_timeout = httpx.Timeout(2.0, connect=1.0)
    async with httpx.AsyncClient(timeout=httpx_timeout) as client:
        try:
            response_data = await asyncio.wait_for(
                _post_chat(client, base_url, payload),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LLM decision timed out after %.0fms; falling back to %r",
                timeout * 1000,
                last_action or _DEFAULT_FALLBACK,
            )
            return last_action or _DEFAULT_FALLBACK
        except httpx.ConnectError:
            logger.error(
                "Ollama unreachable at %s; falling back to %r",
                base_url,
                last_action or _DEFAULT_FALLBACK,
            )
            return last_action or _DEFAULT_FALLBACK
        except Exception as exc:
            logger.error(
                "LLM decision call failed: %s; falling back to %r",
                exc,
                last_action or _DEFAULT_FALLBACK,
            )
            return last_action or _DEFAULT_FALLBACK

    if response_data is None:
        logger.warning("Empty Ollama response; falling back")
        return last_action or _DEFAULT_FALLBACK

    # Extract the content string from Ollama's response
    message = response_data.get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""

    # Parse and validate
    action = parse_llm_response(content, valid_names)

    if action is not None:
        logger.info("LLM chose action: %r", action)
        return action

    # Fallback to last_action, then WAIT
    fallback = last_action or _DEFAULT_FALLBACK
    logger.warning("Could not parse LLM response %r; falling back to %r", content, fallback)
    return fallback


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _post_chat(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    """POST to ``{base_url}/api/chat`` and return the JSON response dict.

    Returns ``None`` on connection errors (caller handles fallback).
    """
    resp = await client.post(f"{base_url}/api/chat", json=payload)
    resp.raise_for_status()
    return resp.json()


def _strip_openai_suffix(url: str) -> str:
    """Remove trailing ``/v1`` (or ``/v1/``) from an Ollama URL to obtain
    the base URL where native API endpoints like ``/api/chat`` live.

    Examples
    --------
    >>> _strip_openai_suffix('http://localhost:11434/v1')
    'http://localhost:11434'
    >>> _strip_openai_suffix('http://localhost:11434')
    'http://localhost:11434'
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path == "/v1":
        parsed = parsed._replace(path="")
    return urlunparse(parsed)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "call_llm_decision",
    "get_macro_model",
    "parse_llm_response",
]