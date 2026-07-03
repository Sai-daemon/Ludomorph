"""
Ollama Health Check Module — Step 1.6

Async health check against a local Ollama server:
- GET /api/version  → verify version ≥0.5.0 (required for JSON schema support)
- GET /api/tags     → confirm the configured model is available

All checks are async (httpx), return structured results, and provide
actionable error messages per the spec.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Structured result & error types
# ---------------------------------------------------------------------------


@dataclass
class OllamaHealthResult:
    """Immutable snapshot of an Ollama health probe."""

    healthy: bool = False
    """True when the server is reachable, model is present, and version ≥0.5.0."""

    server_reachable: bool = False
    """True if /api/tags returned a valid response (server is running)."""

    model_found: bool = False
    """True if the configured model name appears in the server's model list."""

    version: str = ""
    """Raw version string reported by /api/version (e.g. '0.5.4')."""

    version_ok: bool = False
    """True when version ≥0.5.0 (required for JSON schema constraints)."""

    configured_model: str = ""
    """The model name requested in config (e.g. 'phi3.5:3.8b-mini-instruct-q4_K_M')."""

    base_url: str = ""
    """The Ollama base URL used for health probes (no /v1 suffix)."""

    error: str = ""
    """Human-readable, actionable error message when healthy is False."""

    raw_tags: list[dict[str, Any]] = field(default_factory=list)
    """Raw model list from /api/tags (for debugging)."""


class OllamaHealthError(Exception):
    """Raised when a health check fails and the caller wants an exception.

    The ``result`` attribute carries the full structured result so error
    handlers can inspect individual fields.
    """

    def __init__(self, result: OllamaHealthResult) -> None:
        self.result = result
        super().__init__(result.error or "Ollama health check failed")


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _strip_openai_suffix(url: str) -> str:
    """Remove trailing /v1 (or /v1/) from an Ollama URL to obtain the
    base URL where native API endpoints like /api/tags live.

    Examples
    --------
    >>> _strip_openai_suffix('http://localhost:11434/v1')
    'http://localhost:11434'
    >>> _strip_openai_suffix('http://localhost:11434')
    'http://localhost:11434'
    """
    parsed = urlparse(url)
    # Only strip if the path component is exactly /v1 or /v1/
    path = parsed.path.rstrip("/")
    if path == "/v1":
        parsed = parsed._replace(path="")
    return urlunparse(parsed)


def _semver_ge(version: str, minimum: tuple[int, int, int]) -> bool:
    """Compare a version string against a minimum (major, minor, patch).

    Non-numeric segments are ignored; e.g. '0.5.0-rc1' → (0, 5, 0).
    Returns True if *version* ≥ *minimum*.
    """
    parts: list[int] = []
    for segment in version.split("-")[0].split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3]) >= minimum


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


async def _get_version(base_url: str, client: httpx.AsyncClient) -> tuple[str, bool]:
    """Fetch the Ollama server version from GET /api/version.

    Returns
    -------
    (version_string, meets_minimum)
        *version_string* is the raw version field or '' on failure.
    """
    try:
        resp = await client.get(f"{base_url}/api/version")
        resp.raise_for_status()
        data = resp.json()
        version = data.get("version", "")
        ok = _semver_ge(version, (0, 5, 0)) if version else False
        logger.debug(f"Ollama version: {version} (≥0.5.0? {ok})")
        return version, ok
    except httpx.HTTPError as exc:
        logger.debug(f"/api/version failed: {exc}")
        return "", False
    except Exception as exc:
        logger.debug(f"/api/version unexpected error: {exc}")
        return "", False


async def _get_tags(
    base_url: str, client: httpx.AsyncClient
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch the model list from GET /api/tags.

    Returns
    -------
    (models, reachable)
        *models* is the raw list from ``resp['models']`` (or []).
        *reachable* is True when the server responded successfully.
    """
    try:
        resp = await client.get(f"{base_url}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models: list[dict[str, Any]] = data.get("models", [])
        logger.debug(f"/api/tags returned {len(models)} model(s)")
        return models, True
    except httpx.HTTPError as exc:
        logger.debug(f"/api/tags failed: {exc}")
        return [], False
    except Exception as exc:
        logger.debug(f"/api/tags unexpected error: {exc}")
        return [], False


def _model_in_tags(model: str, tags: list[dict[str, Any]]) -> bool:
    """Check whether *model* is listed in the /api/tags response.

    Matches by exact name, then falls back to base-name (strip tag suffix).
    For example, 'phi3.5:3.8b' will match 'phi3.5:3.8b-mini-instruct-q4_K_M'
    when the base name ('phi3.5:3.8b') is a prefix of the listed name.
    """
    if not tags or not model:
        return False

    listed_names = {entry.get("name", "") for entry in tags}

    # 1) Exact match
    if model in listed_names:
        return True

    # 2) Case-insensitive exact match
    model_lower = model.lower()
    for name in listed_names:
        if name.lower() == model_lower:
            return True

    # 3) Base-name prefix match (e.g. 'phi3.5' matches 'phi3.5:latest')
    base = model.split(":")[0].lower()
    if base:
        for name in listed_names:
            if name.lower().split(":")[0] == base:
                logger.debug(
                    f"Model '{model}' not found exactly, but base name "
                    f"'{base}' matches listed model '{name}'"
                )
                return True

    return False


# ---------------------------------------------------------------------------
# Top-level health check
# ---------------------------------------------------------------------------


async def ollama_health_check(
    config: dict[str, Any],
    *,
    timeout: float = 5.0,
) -> OllamaHealthResult:
    """Run the full Ollama health probe asynchronously.

    Parameters
    ----------
    config : dict
        The global config dictionary (from config_manager).  Must contain
        ``ollama_url`` and ``ollama_model`` keys.
    timeout : float
        Total timeout in seconds for the health check (default 5 s).
        Covers connect + first-read for both endpoints.

    Returns
    -------
    OllamaHealthResult
        A dataclass with all probe results and a consolidated ``healthy``
        flag.  The ``error`` field contains an actionable message when
        ``healthy`` is False.

    Raises
    ------
    OllamaHealthError
        *Only* if the caller explicitly asks for exceptions.  This function
        is designed to return a result object so callers can inspect
        individual fields without try/except.
    """
    ollama_url: str = config.get("ollama_url", "http://localhost:11434")
    model: str = config.get("ollama_model", "")
    base_url = _strip_openai_suffix(ollama_url)

    logger.info(f"Running Ollama health check against {base_url} (model: {model})")

    result = OllamaHealthResult(
        configured_model=model,
        base_url=base_url,
    )

    timeout_cfg = httpx.Timeout(timeout, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout_cfg) as client:
        # Run probes concurrently
        version_str, version_ok = await _get_version(base_url, client)
        tags, reachable = await _get_tags(base_url, client)

    result.server_reachable = reachable
    result.version = version_str
    result.version_ok = version_ok
    result.raw_tags = tags

    # --- Build actionable error messages ---

    if not reachable:
        result.error = (
            f"Ollama not reachable at {base_url}. "
            f"Make sure 'ollama serve' is running."
        )
        return result

    # Server is reachable — check model and version
    result.model_found = _model_in_tags(model, tags)

    errors: list[str] = []

    if not result.model_found:
        if model:
            errors.append(
                f"Model '{model}' not found. "
                f"Pull it with: ollama pull {model}"
            )
        else:
            errors.append(
                "No Ollama model configured. Set 'ollama_model' in config.json."
            )

    if not version_ok:
        min_ver = "0.5.0"
        current = version_str or "unknown"
        errors.append(
            f"Ollama version {current} is too old. "
            f"Version ≥{min_ver} required for JSON schema support. "
            f"Please upgrade Ollama."
        )

    if errors:
        result.error = " | ".join(errors)
    else:
        result.healthy = True
        logger.info(f"Ollama health OK: version={version_str}, model={model} found")

    return result


# ---------------------------------------------------------------------------
# Convenience async wrapper for callers that want exceptions
# ---------------------------------------------------------------------------


async def ollama_health_check_or_raise(
    config: dict[str, Any],
    *,
    timeout: float = 5.0,
) -> OllamaHealthResult:
    """Like :func:`ollama_health_check`, but raises :exc:`OllamaHealthError`
    when the health probe fails.

    Use this at startup when you want to abort early.
    """
    result = await ollama_health_check(config, timeout=timeout)
    if not result.healthy:
        raise OllamaHealthError(result)
    return result