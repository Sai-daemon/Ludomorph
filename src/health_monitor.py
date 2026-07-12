"""
Phase 5.6 — HealthMonitor Backend.

Provides a central async ``HealthMonitor`` that polls all dependent services
(Ollama, Tesseract/OCR, MCP memory server, game window) and returns a
consolidated status suitable for both the UI status bar and the decision
loop's graceful-degradation logic (Phase 6.4).

Spec references
---------------
* ``Calibration_UI_research.md`` §Problem 1 — HealthMonitor & UI hook
* ``Extra_research06.md`` §5 — Unified Health Monitor Architecture
* ``Extra_research06.md`` §4.3 — Circuit breaker pattern
* ``Implementation_Phases.md`` §5.6 — Phase definition
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from src.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHECK_INTERVAL: float = 5.0  # seconds between automatic health polls
_CIRCUIT_BREAKER_THRESHOLD: int = 3   # consecutive failures before marking unhealthy
_CIRCUIT_BREAKER_RECOVERY: float = 30.0  # seconds before half-open probe
_HEALTH_CHECK_TIMEOUT: float = 2.0    # per-check HTTP timeout


# ---------------------------------------------------------------------------
# Status level enumeration
# ---------------------------------------------------------------------------


class HealthLevel(str, Enum):
    """Consolidated health level for the overall system."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"


# ---------------------------------------------------------------------------
# ServiceHealth — per-service health record
# ---------------------------------------------------------------------------


@dataclass
class ServiceHealth:
    """Immutable snapshot of a single service's health.

    Attributes:
        healthy: ``True`` when the last check succeeded.
        reason: Human-readable status message (or error description).
        last_check: ``time.monotonic()`` timestamp of the most recent probe.
        consecutive_failures: Number of sequential failed checks (reset on success).
        open_circuit: ``True`` when the circuit breaker has tripped.
    """

    healthy: bool = False
    reason: str = ""
    last_check: float = 0.0
    consecutive_failures: int = 0
    open_circuit: bool = False


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Central health authority for all dependent services.

    Constructor accepts references to the live modules so each check can
    probe the real component rather than relying on stubs.  Every reference
    is optional — when a reference is ``None`` the corresponding check
    returns ``(healthy=True, reason="Not configured")``.

    Usage::

        monitor = HealthMonitor(
            ocr_module=ocr,
            capture_obj=capture,
            mcp_client=mcp,
            config=cfg,
        )
        status = await monitor.get_overall_status()
        # {"level": "OK", "reason": "All systems nominal", "services": {...}}
    """

    def __init__(
        self,
        ocr_module: Any = None,
        capture_obj: Any = None,
        mcp_client: Any = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._ocr = ocr_module
        self._capture = capture_obj
        self._mcp = mcp_client
        self._config = config or {}

        # Per-service state
        self.services: dict[str, ServiceHealth] = {
            "ollama": ServiceHealth(reason="Not yet checked"),
            "tesseract": ServiceHealth(reason="Not yet checked"),
            "mcp": ServiceHealth(reason="Not yet checked"),
            "game_window": ServiceHealth(reason="Not yet checked"),
        }

    # ------------------------------------------------------------------
    # Per-service probes
    # ------------------------------------------------------------------

    async def check_ollama(self) -> tuple[bool, str]:
        """Ping the Ollama server and validate model presence.

        Delegates to :func:`src.ollama_health.ollama_health_check` for the
        full version + model check.  Falls back to a simple HTTP reachability
        test when the config is unavailable.
        """
        try:
            from src.ollama_health import ollama_health_check

            result = await ollama_health_check(
                self._config, timeout=_HEALTH_CHECK_TIMEOUT
            )
            if result.healthy:
                return True, f"Ollama {result.version} — {result.configured_model}"
            return False, result.error or "Ollama health check failed"
        except ImportError:
            pass
        except Exception as exc:
            logger.debug(f"ollama_health_check raised: {exc}")

        # Fallback: simple HTTP reachability
        base_url = self._config.get("ollama_url", "http://localhost:11434")
        from src.ollama_health import _strip_openai_suffix

        base_url = _strip_openai_suffix(base_url)
        try:
            async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    return True, "Ollama reachable"
                return False, f"Ollama returned HTTP {resp.status_code}"
        except httpx.ConnectError:
            return False, f"Ollama not reachable at {base_url}. Make sure 'ollama serve' is running."
        except httpx.ReadTimeout:
            return False, f"Ollama timed out at {base_url}."
        except Exception as exc:
            return False, f"Ollama check error: {exc}"

    async def check_tesseract(self) -> tuple[bool, str]:
        """Verify the OCR module is initialised and ready."""
        if self._ocr is None:
            return True, "OCR not configured"

        # Check if the module has a healthy attribute or can be probed
        try:
            # If the OCR module exposes a health probe, use it
            if hasattr(self._ocr, "is_healthy") and callable(self._ocr.is_healthy):
                ok = self._ocr.is_healthy()
                if ok:
                    return True, "OCR ready"
                return False, "OCR module reports unhealthy"

            # Fallback: check if it's initialised (has a recognisable attr)
            if hasattr(self._ocr, "_tesseract_cmd") or hasattr(self._ocr, "get_text"):
                return True, "OCR initialised"
            return True, "OCR available"
        except Exception as exc:
            return False, f"OCR check failed: {exc}"

    async def check_mcp(self) -> tuple[bool, str]:
        """Probe the MCP memory server's health endpoint."""
        if self._mcp is None:
            return True, "MCP not configured"

        mcp_port = self._config.get("mcp_port", 8000)
        mcp_url = f"http://localhost:{mcp_port}"

        try:
            async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
                # Try /api/health first (standard MCP endpoint), then /health
                for path in ("/api/health", "/health", "/"):
                    try:
                        resp = await client.get(f"{mcp_url}{path}")
                        if resp.status_code == 200:
                            return True, f"MCP server reachable (port {mcp_port})"
                    except Exception:
                        continue
                return False, f"MCP server not responding on port {mcp_port}"
        except httpx.ConnectError:
            return False, f"MCP server not found on port {mcp_port}. Memory persistence unavailable."
        except Exception as exc:
            return False, f"MCP check error: {exc}"

    async def check_game_window(self) -> tuple[bool, str]:
        """Verify the target game window is still present."""
        if self._capture is None:
            return True, "Window capture not configured"

        try:
            # Prefer a dedicated window-focus check if available
            if hasattr(self._capture, "is_window_alive") and callable(
                self._capture.is_window_alive
            ):
                alive = self._capture.is_window_alive()
                if alive:
                    return True, "Game window present"
                return False, "Game window closed or lost"

            # Fallback: attempt a capture and check for None / black frame
            if hasattr(self._capture, "capture") and callable(self._capture.capture):
                import asyncio as _asyncio

                try:
                    frame = await _asyncio.wait_for(
                        self._capture.capture(), timeout=2.0
                    )
                except _asyncio.TimeoutError:
                    return False, "Capture timed out — window may be frozen"
                if frame is None:
                    return False, "Capture returned None — window may be closed"
                return True, "Game window present"
            return True, "Window capture available"
        except Exception as exc:
            return False, f"Window check failed: {exc}"

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    async def check_all(self) -> dict[str, tuple[bool, str]]:
        """Run all four health probes concurrently.

        Returns a dict mapping service name → ``(healthy, reason)``.
        """
        results = await asyncio.gather(
            self.check_ollama(),
            self.check_tesseract(),
            self.check_mcp(),
            self.check_game_window(),
            return_exceptions=True,
        )

        names = ("ollama", "tesseract", "mcp", "game_window")
        parsed: dict[str, tuple[bool, str]] = {}
        for name, result in zip(names, results):
            if isinstance(result, BaseException):
                parsed[name] = (False, f"Check raised: {result}")
            else:
                parsed[name] = result  # type: ignore[assignment]

        return parsed

    async def get_overall_status(self) -> dict[str, Any]:
        """Return a consolidated status dict suitable for UI consumption.

        Returns a dict with keys:

        * ``level`` — ``"OK"``, ``"DEGRADED"``, or ``"STOPPING"``
        * ``reason`` — human-readable summary
        * ``services`` — nested dict of per-service ``{"healthy": bool, "reason": str}``
        """
        results = await self.check_all()

        # Update stored service records + circuit breaker
        now = time.monotonic()
        for name, (healthy, reason) in results.items():
            svc = self.services[name]
            svc.last_check = now

            if healthy:
                svc.healthy = True
                svc.reason = reason
                svc.consecutive_failures = 0
                svc.open_circuit = False
            else:
                svc.consecutive_failures += 1
                svc.reason = reason
                if svc.consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    svc.open_circuit = True
                    svc.healthy = False
                elif svc.open_circuit:
                    # Still open — only reopen after recovery timeout
                    if now - svc.last_check > _CIRCUIT_BREAKER_RECOVERY:
                        svc.open_circuit = False
                        svc.healthy = False  # will retry next cycle
                    # else: stay unhealthy & open
                else:
                    svc.healthy = False

        # --- Determine overall level ---
        services_dict: dict[str, dict[str, Any]] = {
            name: {"healthy": svc.healthy, "reason": svc.reason}
            for name, svc in self.services.items()
        }

        # STOPPING: game window lost
        if not services_dict["game_window"]["healthy"]:
            return {
                "level": HealthLevel.STOPPING.value,
                "reason": "Game window closed or lost",
                "services": services_dict,
            }

        # DEGRADED: any other critical service down
        degraded_reasons: list[str] = []
        for name in ("ollama", "tesseract", "mcp"):
            if not services_dict[name]["healthy"]:
                label = {
                    "ollama": "LLM unreachable",
                    "tesseract": "OCR unavailable",
                    "mcp": "Memory server down",
                }.get(name, f"{name} unhealthy")
                degraded_reasons.append(label)

        if degraded_reasons:
            return {
                "level": HealthLevel.DEGRADED.value,
                "reason": "; ".join(degraded_reasons),
                "services": services_dict,
            }

        return {
            "level": HealthLevel.OK.value,
            "reason": "All systems nominal",
            "services": services_dict,
        }

    # ------------------------------------------------------------------
    # Convenience: periodic background poller
    # ------------------------------------------------------------------

    async def run_periodic_checks(self, interval: float = _DEFAULT_CHECK_INTERVAL) -> None:
        """Continuously poll all services every *interval* seconds.

        Intended to be launched as a background ``asyncio.Task``.  The
        :meth:`get_overall_status` method updates internal state that the
        UI hook reads.
        """
        while True:
            try:
                await self.get_overall_status()
            except Exception as exc:
                logger.warning(f"Periodic health check failed: {exc}")
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "HealthMonitor",
    "ServiceHealth",
    "HealthLevel",
]