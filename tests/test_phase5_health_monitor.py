"""
Phase 5.6 — HealthMonitor Tests.

Validates:
- ``ServiceHealth`` dataclass fields and defaults
- ``HealthLevel`` enum values
- ``HealthMonitor`` class is importable and instantiatable
- ``HealthMonitor.check_all()`` and ``get_overall_status()`` with mocked services
- Overall status returns correct levels for different failure combos
  (all OK → "OK", Ollama down → "DEGRADED", game window lost → "STOPPING")
- Circuit breaker: 3 consecutive failures marks service unhealthy
- ``HealthPanel`` class is importable and instantiatable (without display)
- ``HealthPanel.update_health()`` correctly colour‑codes status levels
- ``gui`` package exports ``HealthPanel``
- ``MainWindow`` integration: health panel is accessible via ``health_panel`` property
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_path() -> None:
    import sys

    root = str(_PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_path()


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.health_monitor import HealthMonitor, HealthLevel, ServiceHealth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """Minimal config for health monitor tests."""
    return {
        "ollama_url": "http://localhost:11434",
        "ollama_model": "test-model",
        "mcp_port": 8000,
    }


@pytest.fixture
def mock_ocr() -> MagicMock:
    """Mock OCR module that is healthy."""
    ocr = MagicMock()
    ocr.is_healthy.return_value = True
    return ocr


@pytest.fixture
def mock_capture() -> MagicMock:
    """Mock screen capture that returns a dummy frame."""
    import numpy as np

    capture = MagicMock()
    capture.is_window_alive.return_value = True

    async def _capture() -> "np.ndarray":
        return np.zeros((100, 100, 3), dtype=np.uint8)

    capture.capture = _capture
    return capture


@pytest.fixture
def mock_mcp() -> MagicMock:
    """Mock MCP client (presence of it signals MCP is configured)."""
    return MagicMock()


@pytest.fixture
def health_monitor(
    mock_ocr: MagicMock,
    mock_capture: MagicMock,
    mock_mcp: MagicMock,
    mock_config: dict[str, Any],
) -> HealthMonitor:
    """Fully wired HealthMonitor with mocked dependencies."""
    return HealthMonitor(
        ocr_module=mock_ocr,
        capture_obj=mock_capture,
        mcp_client=mock_mcp,
        config=mock_config,
    )


# ---------------------------------------------------------------------------
# Tests: ServiceHealth dataclass
# ---------------------------------------------------------------------------


class TestServiceHealth:
    """Unit tests for the ServiceHealth dataclass."""

    def test_defaults(self) -> None:
        """ServiceHealth has sensible defaults."""
        sh = ServiceHealth()
        assert sh.healthy is False
        assert sh.reason == ""
        assert sh.last_check == 0.0
        assert sh.consecutive_failures == 0
        assert sh.open_circuit is False

    def test_custom_values(self) -> None:
        """ServiceHealth accepts custom values."""
        sh = ServiceHealth(
            healthy=True,
            reason="All good",
            last_check=12345.0,
            consecutive_failures=2,
            open_circuit=True,
        )
        assert sh.healthy is True
        assert sh.reason == "All good"

    def test_equality(self) -> None:
        """ServiceHealth instances compare by value."""
        a = ServiceHealth(healthy=True, reason="ok")
        b = ServiceHealth(healthy=True, reason="ok")
        assert a == b


# ---------------------------------------------------------------------------
# Tests: HealthLevel enum
# ---------------------------------------------------------------------------


class TestHealthLevel:
    """Unit tests for HealthLevel enum."""

    def test_values(self) -> None:
        assert HealthLevel.OK.value == "OK"
        assert HealthLevel.DEGRADED.value == "DEGRADED"
        assert HealthLevel.STOPPING.value == "STOPPING"

    def test_membership(self) -> None:
        assert HealthLevel("OK") == HealthLevel.OK
        assert HealthLevel("DEGRADED") == HealthLevel.DEGRADED


# ---------------------------------------------------------------------------
# Tests: HealthMonitor — importability & construction
# ---------------------------------------------------------------------------


class TestHealthMonitorConstruction:
    """Verify HealthMonitor is importable and constructable."""

    def test_import(self) -> None:
        """HealthMonitor is importable from src.health_monitor."""
        from src.health_monitor import HealthMonitor, ServiceHealth, HealthLevel

        assert HealthMonitor is not None
        assert ServiceHealth is not None
        assert HealthLevel is not None

    def test_instantiate_no_args(self) -> None:
        """HealthMonitor can be created without any arguments."""
        hm = HealthMonitor()
        assert hm is not None
        assert "ollama" in hm.services
        assert "tesseract" in hm.services
        assert "mcp" in hm.services
        assert "game_window" in hm.services

    def test_instantiate_with_deps(
        self,
        mock_ocr: MagicMock,
        mock_capture: MagicMock,
        mock_mcp: MagicMock,
        mock_config: dict[str, Any],
    ) -> None:
        """HealthMonitor accepts all constructor dependencies."""
        hm = HealthMonitor(
            ocr_module=mock_ocr,
            capture_obj=mock_capture,
            mcp_client=mock_mcp,
            config=mock_config,
        )
        assert hm._ocr is mock_ocr
        assert hm._capture is mock_capture
        assert hm._mcp is mock_mcp
        assert hm._config is mock_config


# ---------------------------------------------------------------------------
# Tests: HealthMonitor — individual checks
# ---------------------------------------------------------------------------


class TestHealthMonitorChecks:
    """Test individual probe methods."""

    @pytest.mark.asyncio
    async def test_check_tesseract_healthy(self, mock_ocr: MagicMock, mock_config: dict[str, Any]) -> None:
        """OCR check returns healthy when module is ready."""
        hm = HealthMonitor(ocr_module=mock_ocr, config=mock_config)
        mock_ocr.is_healthy.return_value = True
        healthy, reason = await hm.check_tesseract()
        assert healthy is True
        assert "ready" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_tesseract_none(self, mock_config: dict[str, Any]) -> None:
        """OCR check returns healthy when not configured (no module)."""
        hm = HealthMonitor(config=mock_config)
        healthy, reason = await hm.check_tesseract()
        assert healthy is True
        assert "not configured" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_mcp_none(self, mock_config: dict[str, Any]) -> None:
        """MCP check returns healthy when not configured."""
        hm = HealthMonitor(config=mock_config)
        healthy, reason = await hm.check_mcp()
        assert healthy is True
        assert "not configured" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_game_window_healthy(
        self, mock_capture: MagicMock, mock_config: dict[str, Any]
    ) -> None:
        """Game window check returns healthy when window is alive."""
        mock_capture.is_window_alive.return_value = True
        hm = HealthMonitor(capture_obj=mock_capture, config=mock_config)
        healthy, reason = await hm.check_game_window()
        assert healthy is True
        assert "present" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_game_window_closed(
        self, mock_capture: MagicMock, mock_config: dict[str, Any]
    ) -> None:
        """Game window check returns unhealthy when window is gone."""
        mock_capture.is_window_alive.return_value = False
        hm = HealthMonitor(capture_obj=mock_capture, config=mock_config)
        healthy, reason = await hm.check_game_window()
        assert healthy is False
        assert "closed" in reason.lower() or "lost" in reason.lower()

    @pytest.mark.asyncio
    async def test_check_all_concurrent(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """check_all() runs all 4 probes and returns results dict."""
        results = await health_monitor.check_all()
        assert set(results.keys()) == {"ollama", "tesseract", "mcp", "game_window"}
        for name, (healthy, reason) in results.items():
            assert isinstance(healthy, bool)
            assert isinstance(reason, str)


# ---------------------------------------------------------------------------
# Tests: HealthMonitor — get_overall_status
# ---------------------------------------------------------------------------


class TestHealthMonitorOverallStatus:
    """Test the consolidated get_overall_status() method."""

    @pytest.mark.asyncio
    async def test_all_healthy(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """When all services pass, overall level is OK."""
        # Patch the HTTP-based checks that would fail without real servers
        with (
            patch.object(health_monitor, "check_ollama", return_value=(True, "Ollama OK")),
            patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")),
        ):
            status = await health_monitor.get_overall_status()
        assert status["level"] == HealthLevel.OK.value
        assert "nominal" in status["reason"].lower()

    @pytest.mark.asyncio
    async def test_ollama_down_degraded(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """When Ollama is down, overall level is DEGRADED."""
        with (
            patch.object(health_monitor, "check_ollama", return_value=(False, "Ollama unreachable")),
            patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")),
        ):
            status = await health_monitor.get_overall_status()
        assert status["level"] == HealthLevel.DEGRADED.value
        assert "LLM unreachable" in status["reason"]

    @pytest.mark.asyncio
    async def test_mcp_down_degraded(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """When MCP is down, overall level is DEGRADED."""
        with (
            patch.object(health_monitor, "check_ollama", return_value=(True, "Ollama OK")),
            patch.object(health_monitor, "check_mcp", return_value=(False, "MCP unreachable")),
        ):
            status = await health_monitor.get_overall_status()
        assert status["level"] == HealthLevel.DEGRADED.value
        assert "Memory server down" in status["reason"]

    @pytest.mark.asyncio
    async def test_game_window_lost_stopping(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """When game window is lost, overall level is STOPPING."""
        with (
            patch.object(health_monitor, "check_ollama", return_value=(True, "Ollama OK")),
            patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")),
            patch.object(health_monitor, "check_game_window", return_value=(False, "Window closed")),
        ):
            status = await health_monitor.get_overall_status()
        assert status["level"] == HealthLevel.STOPPING.value

    @pytest.mark.asyncio
    async def test_multiple_degraded_reasons(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """When multiple services fail, all reasons appear in the reason string."""
        with (
            patch.object(health_monitor, "check_ollama", return_value=(False, "Ollama unreachable")),
            patch.object(health_monitor, "check_mcp", return_value=(False, "MCP unreachable")),
            patch.object(health_monitor, "check_tesseract", return_value=(False, "OCR unavailable")),
        ):
            status = await health_monitor.get_overall_status()
        assert status["level"] == HealthLevel.DEGRADED.value
        assert "LLM unreachable" in status["reason"]
        assert "Memory server down" in status["reason"]
        assert "OCR unavailable" in status["reason"]

    @pytest.mark.asyncio
    async def test_services_dict_structure(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """The 'services' key contains per-service healthy + reason."""
        with (
            patch.object(health_monitor, "check_ollama", return_value=(True, "Ollama OK")),
            patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")),
        ):
            status = await health_monitor.get_overall_status()
        services = status["services"]
        assert services["ollama"]["healthy"] is True
        assert services["game_window"]["healthy"] is True
        assert "reason" in services["ollama"]


# ---------------------------------------------------------------------------
# Tests: HealthMonitor — circuit breaker
# ---------------------------------------------------------------------------


class TestHealthMonitorCircuitBreaker:
    """Verify the circuit breaker pattern in service health records."""

    @pytest.mark.asyncio
    async def test_consecutive_failures_increment(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """Consecutive failed checks increment the counter."""
        with patch.object(health_monitor, "check_ollama", return_value=(False, "Ollama unreachable")):
            with patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")):
                # Three consecutive failures
                for _ in range(3):
                    await health_monitor.get_overall_status()

        svc = health_monitor.services["ollama"]
        assert svc.consecutive_failures >= 3
        assert svc.open_circuit is True
        assert svc.healthy is False

    @pytest.mark.asyncio
    async def test_failure_then_recovery_resets_counter(
        self, health_monitor: HealthMonitor,
    ) -> None:
        """A successful check resets the failure counter."""
        # Fail once
        with patch.object(health_monitor, "check_ollama", return_value=(False, "Ollama unreachable")):
            with patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")):
                await health_monitor.get_overall_status()
        assert health_monitor.services["ollama"].consecutive_failures == 1

        # Then succeed
        with patch.object(health_monitor, "check_ollama", return_value=(True, "Ollama OK")):
            with patch.object(health_monitor, "check_mcp", return_value=(True, "MCP OK")):
                await health_monitor.get_overall_status()
        assert health_monitor.services["ollama"].consecutive_failures == 0
        assert health_monitor.services["ollama"].open_circuit is False
        assert health_monitor.services["ollama"].healthy is True


# ---------------------------------------------------------------------------
# Tests: HealthPanel — importability
# ---------------------------------------------------------------------------


class TestHealthPanelImport:
    """Verify the HealthPanel widget is importable and constructable."""

    def test_import(self) -> None:
        """HealthPanel is importable from src.gui.health_panel."""
        from src.gui.health_panel import HealthPanel

        assert HealthPanel is not None

    def test_import_from_gui_package(self) -> None:
        """HealthPanel is exported from src.gui."""
        from src.gui import HealthPanel

        assert HealthPanel is not None

    def test_instantiate_without_display(self) -> None:
        """HealthPanel can be instantiated without a display (widget creation test)."""
        import tkinter as tk

        from src.gui.health_panel import HealthPanel

        root = tk.Tk()
        root.withdraw()  # hide window
        try:
            panel = HealthPanel(root)
            assert panel is not None
            assert panel._cards is not None
            assert len(panel._cards) == 4  # one card per service
        finally:
            root.destroy()

    def test_update_health_ok(self) -> None:
        """update_health() with OK status colour-codes correctly."""
        import tkinter as tk

        from src.gui.health_panel import HealthPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = HealthPanel(root)
            status = {
                "level": "OK",
                "reason": "All systems nominal",
                "services": {
                    "ollama": {"healthy": True, "reason": "Ollama OK"},
                    "tesseract": {"healthy": True, "reason": "OCR ready"},
                    "mcp": {"healthy": True, "reason": "MCP reachable"},
                    "game_window": {"healthy": True, "reason": "Window present"},
                },
            }
            panel.update_health(status)
            # Verify the header was updated (text contains OK message)
            assert "All systems nominal" in panel._lbl_overall.cget("text")
        finally:
            root.destroy()

    def test_update_health_degraded(self) -> None:
        """update_health() with DEGRADED status uses warning colour."""
        import tkinter as tk

        from src.gui.health_panel import HealthPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = HealthPanel(root)
            status = {
                "level": "DEGRADED",
                "reason": "LLM unreachable",
                "services": {
                    "ollama": {"healthy": False, "reason": "Ollama unreachable"},
                    "tesseract": {"healthy": True, "reason": "OCR ready"},
                    "mcp": {"healthy": True, "reason": "MCP reachable"},
                    "game_window": {"healthy": True, "reason": "Window present"},
                },
            }
            panel.update_health(status)
            # Ollama card should show unhealthy
            ollama_card = panel._cards["ollama"]
            assert "Unhealthy" in ollama_card["status"].cget("text")
        finally:
            root.destroy()

    def test_push_and_poll_queue(self) -> None:
        """push_status() puts items into the async queue for polling."""
        import tkinter as tk

        from src.gui.health_panel import HealthPanel

        root = tk.Tk()
        root.withdraw()
        try:
            panel = HealthPanel(root)
            test_status = {"level": "OK", "reason": "Test", "services": {}}
            panel.push_status(test_status)
            # The queue should have one item
            assert panel._status_queue.qsize() == 1
            # Polling should drain it
            panel._poll_queue()
            assert panel._status_queue.qsize() == 0
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Tests: MainWindow integration
# ---------------------------------------------------------------------------


class TestMainWindowIntegration:
    """Verify the MainWindow includes HealthPanel integration points."""

    def test_main_window_has_health_panel_property(self) -> None:
        """MainWindow.health_panel property returns None initially."""
        import tkinter as tk

        from src.gui.main_window import MainWindow

        root = MainWindow()
        root.withdraw()
        try:
            assert root.health_panel is None
            # Toggle the health monitor on — creates the panel
            root._on_health_monitor()
            assert root.health_panel is not None
            assert root._health_panel_visible is True
            # Toggle off
            root._on_health_monitor()
            assert root._health_panel_visible is False
        finally:
            root.destroy()

    def test_main_window_has_set_health_monitor(self) -> None:
        """MainWindow has set_health_monitor() injection method."""
        import tkinter as tk

        from src.gui.main_window import MainWindow

        root = MainWindow()
        root.withdraw()
        try:
            mock_hm = MagicMock()
            root.set_health_monitor(mock_hm)
            # After injection, health panel should exist
            assert root._health_panel is not None
        finally:
            root.destroy()

    def test_view_menu_has_health_monitor(self) -> None:
        """The View menu and _on_health_monitor callback exist on MainWindow."""
        import tkinter as tk

        from src.gui.main_window import MainWindow

        root = MainWindow()
        root.withdraw()
        try:
            # The _on_health_monitor method is the View → Health Monitor callback
            assert hasattr(root, "_on_health_monitor")
            assert callable(root._on_health_monitor)

            # Verify a menu is configured on the window
            menu_str = root.cget("menu")
            assert menu_str is not None and len(menu_str) > 0
        finally:
            root.destroy()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "TestServiceHealth",
    "TestHealthLevel",
    "TestHealthMonitorConstruction",
    "TestHealthMonitorChecks",
    "TestHealthMonitorOverallStatus",
    "TestHealthMonitorCircuitBreaker",
    "TestHealthPanelImport",
    "TestMainWindowIntegration",
]