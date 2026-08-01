"""
Phase 6.1 — GUIEngineBridge: connects the Tkinter GUI to the async engine.

Runs the asyncio event loop in a background daemon thread and manages
the full lifecycle of all Phase 1‑4 engine components.  Communication
between the GUI (main) thread and the asyncio thread uses:

* ``asyncio.run_coroutine_threadsafe()``  — GUI → engine
* ``AsyncTk.call_async()``                — engine → GUI

Spec references
---------------
* ``Implementation_Phases.md`` §6.1 — GUI-to-Engine Wiring & Async Bridge
"""

from __future__ import annotations

import asyncio
import json
import threading
from enum import Enum
from pathlib import Path
from typing import Any

from src.logging_config import get_logger
from src.gui.theme import ThemeManager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project root (for resolving bundled config/ and models/)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# EngineState
# ---------------------------------------------------------------------------


class EngineState(str, Enum):
    """Top‑level engine lifecycle state."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


# ---------------------------------------------------------------------------
# GUIEngineBridge
# ---------------------------------------------------------------------------


class GUIEngineBridge:
    """Manages the async engine lifecycle on a background asyncio thread.

    Constructor spawns the thread and schedules :meth:`_async_init` which
    creates every engine component.  After construction the bridge is in
    ``EngineState.IDLE`` — call :meth:`start` to begin the decision loop.

    Parameters
    ----------
    config:
        Global configuration dict (already loaded & merged with defaults).
    input_ctrl:
        ``InputController`` instance.
    focus_mgr:
        ``WindowFocusManager`` instance.
    main_window:
        ``MainWindow`` instance (the ``AsyncTk`` root).
    profile_path:
        Optional path to a per‑game profile directory.  If ``None`` the
        bundled ``config/`` files are used.
    """

    def __init__(
        self,
        config: dict[str, Any],
        input_ctrl: Any,
        focus_mgr: Any,
        main_window: Any,
        profile_path: Path | None = None,
    ) -> None:
        # --- Saved references -----------------------------------------------
        self._config = config
        self._input_ctrl = input_ctrl
        self._focus_mgr = focus_mgr
        self._window = main_window
        self._profile_path = profile_path

        # --- Engine component slots (populated during _async_init) ----------
        self._mcp_manager: Any = None
        self._capture: Any = None
        self._ocr: Any = None
        self._state_processor: Any = None
        self._macro_executor: Any = None
        self._vision_processor: Any = None
        self._mcp_client: Any = None
        self._summariser: Any = None
        self._health_monitor: Any = None

        # --- Cached profile refs for re-init on config reload ---
        self._schema: Any = None
        self._profile: Any = None
        self._regions_for_preview: list[dict[str, Any]] = []
        self._macros_data: list[dict[str, Any]] = []

        # --- Task references -------------------------------------------------
        self._loop_task: asyncio.Task[Any] | None = None
        self._health_task: asyncio.Task[Any] | None = None
        self._preview_task: asyncio.Task[Any] | None = None

        # --- State -----------------------------------------------------------
        self._state: EngineState = EngineState.IDLE

        # --- Background asyncio thread ---------------------------------------
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready_event = threading.Event()
        self._shutdown_complete = threading.Event()

        self._thread = threading.Thread(
            target=self._run_async_loop, name="gameai-engine", daemon=True
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            logger.error("Engine thread failed to start within 10 s.")
            raise RuntimeError("Engine asyncio loop did not become ready.")

        # Schedule component initialisation on the fresh loop
        asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)

        logger.info("GUIEngineBridge created — engine thread running.")

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run_async_loop(self) -> None:
        """Target for the background daemon thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready_event.set()
        try:
            self._loop.run_forever()
        except BaseException as exc:
            logger.exception(f"Engine event loop crashed: {exc}")
        finally:
            # Clean up any remaining tasks
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
            self._loop.close()
            logger.debug("Engine event loop closed.")

    # ------------------------------------------------------------------
    # Async initialisation
    # ------------------------------------------------------------------

    async def _async_init(self) -> None:
        """Create every engine component (called once on the asyncio loop)."""
        try:
            # -- MCP server ---------------------------------------------------
            mcp_enabled: bool = self._config.get("mcp_enabled", True)
            if mcp_enabled:
                from src.mcp_server import MCPServerManager

                self._mcp_manager = MCPServerManager(
                    port=8000,
                    ready_timeout=10.0,
                    grace_period=5.0,
                )
                logger.info("MCP server manager created (not yet started).")
            else:
                logger.info("MCP memory server disabled in config.")

            # -- Screen capture -----------------------------------------------
            from src.screen_capture import CaptureConfig, ScreenCapture, WindowTracker

            # Read window_title and capture_region from config (Phase 6.8)
            window_title: str | None = self._config.get("window_title") or None
            capture_region_raw = self._config.get("capture_region")
            capture_region: tuple[int, int, int, int] | None = None
            if isinstance(capture_region_raw, list) and len(capture_region_raw) == 4:
                capture_region = tuple(int(v) for v in capture_region_raw)  # type: ignore[assignment]

            capture_config = CaptureConfig(
                window_title=window_title,
                capture_region=capture_region,
            )
            tracker = WindowTracker(capture_config)
            self._capture = ScreenCapture(capture_config, tracker=tracker)
            test_frame = await self._capture.capture()
            if test_frame is not None:
                h, w = test_frame.shape[:2]
                logger.info(f"ScreenCapture ready — {w}×{h}")
            else:
                logger.warning("Initial capture returned None — is a game window visible?")
            # Inject capture into MainWindow for calibration
            try:
                self._window.call_async(
                    lambda: self._window.set_screen_capture(self._capture)
                )
            except (RuntimeError, Exception):
                pass

            # -- Config manager injection -------------------------------------
            try:
                self._window.call_async(
                    lambda: self._window.set_config_manager(
                        _create_config_proxy(self._config, self._profile_path)
                    )
                )
            except (RuntimeError, Exception):
                pass

            # -- Profile files ------------------------------------------------
            schema_data, regions_data, macros_data = self._resolve_profile_files()
            if schema_data is None or regions_data is None:
                logger.error("Cannot start without state schema and regions.")
                self._set_state(EngineState.ERROR)
                return

            # -- StateSchema & RegionProfile ----------------------------------
            from src.game_state import StateSchema
            from src.region_profile import RegionProfile

            # Inject source_resolution from the active window rect so that
            # region coordinates (stored in source resolution) can be
            # correctly scaled to downsampled capture frames.
            if "source_resolution" not in regions_data and tracker is not None:
                rect = tracker.get_active_window_rect()
                if rect is not None:
                    _r_left, _r_top, r_width, r_height = rect
                    if r_width > 0 and r_height > 0:
                        regions_data["source_resolution"] = {
                            "width": r_width,
                            "height": r_height,
                        }
                        logger.info(
                            f"Injected source_resolution={r_width}×{r_height} "
                            f"from active window tracker."
                        )

            self._schema = StateSchema.from_dict(schema_data)
            self._profile = RegionProfile.from_dict(regions_data)
            # Store regions for live preview overlay
            self._regions_for_preview = regions_data.get("regions", [])
            logger.info(
                "Profile loaded: %d slot(s), %d region(s) — src_res=%s",
                len(self._schema.slots),
                len(self._profile.regions),
                self._profile.source_resolution,
            )

            # -- OCR module ---------------------------------------------------
            from src.ocr_module import OCRConfig, OCRModule

            ocr_config = OCRConfig(
                cache_ttl_seconds=self._config.get("ocr_cache_ttl_seconds", 2.0),
            )
            self._ocr = OCRModule(ocr_config)
            # Inject OCR into MainWindow for calibration preview
            try:
                self._window.call_async(
                    lambda: self._window.set_ocr_module(self._ocr)
                )
            except (RuntimeError, Exception):
                pass
            logger.info("OCR module initialised.")

            # -- Vision processor (optional) ----------------------------------
            await self._refresh_vision(test_frame)

            # -- StateProcessor -----------------------------------------------
            self._state_processor = await self._build_state_processor()
            logger.info("StateProcessor initialised.")

            # -- MacroExecutor ------------------------------------------------
            from src.macro_executor import MacroExecutor

            self._macro_executor = MacroExecutor(self._input_ctrl, config=self._config)
            logger.info("MacroExecutor created (not yet started).")

            # -- MCP client (created after server starts) --------------------
            # We create the client during _start_engine after MCP is running.

            # -- HealthMonitor ------------------------------------------------
            self._health_monitor = self._build_health_monitor()
            logger.info("HealthMonitor created.")

            # Store profile data for later use by start()
            self._macros_data = macros_data or []

            self._set_state(EngineState.IDLE)
            logger.info("Engine components initialised — ready to start.")

        except Exception as exc:
            logger.exception(f"Engine async init failed: {exc}")
            self._set_state(EngineState.ERROR)
            self._window.call_async(
                lambda: self._window._set_engine_state(
                    "Error: init failed", ThemeManager().palette.danger)
            )

    # ------------------------------------------------------------------
    # Component rebuild helpers (used on config reload)
    # ------------------------------------------------------------------

    async def _refresh_vision(self, test_frame: Any = None) -> None:
        """Re‑evaluate vision.enabled from the current config and initialise
        or tear down the vision processor as appropriate."""
        vision_cfg = self._config.get("vision", {})
        vision_enabled = vision_cfg.get("enabled", False)
        logger.debug(
            "Vision config refresh: enabled=%s, has_frame=%s, "
            "current_processor=%s",
            vision_enabled,
            test_frame is not None,
            self._vision_processor is not None,
        )
        if vision_enabled:
            self._vision_processor = await self._init_vision(vision_cfg, test_frame)
            logger.info(
                "Vision refresh result: processor_created=%s",
                self._vision_processor is not None,
            )
        else:
            if self._vision_processor is not None:
                logger.info("Vision disabled by config — tearing down processor.")
            self._vision_processor = None

    async def _build_state_processor(self) -> Any:
        """Create a fresh StateProcessor from current self._profile, _ocr,
        _schema, _vision_processor, and config values."""
        from src.state_processor import StateProcessor

        return StateProcessor(
            profile=self._profile,
            ocr_module=self._ocr,
            schema=self._schema,
            vision_processor=self._vision_processor,
            cache_ttl=self._config.get("state_cache_ttl_seconds", 0.3),
        )

    def _build_health_monitor(self) -> Any:
        """Create a fresh HealthMonitor wired to current components."""
        from src.health_monitor import HealthMonitor

        return HealthMonitor(
            ocr_module=self._ocr,
            capture_obj=self._capture,
            mcp_client=self._mcp_client,  # may be None; wired after MCP is created
            config=self._config,
            vision_processor=self._vision_processor,
        )

    # ------------------------------------------------------------------
    # Vision initialisation helper
    # ------------------------------------------------------------------

    async def _init_vision(
        self,
        vision_cfg: dict[str, Any],
        test_frame: Any,
    ) -> Any:
        """Attempt to initialise the vision processor.  Returns None on failure."""
        from src.vision_detector import VisionConfig, VisionProcessor

        model_path_str = vision_cfg.get("model_path", "models/yolo11n.onnx")
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            model_path = _PROJECT_ROOT / model_path_str

        try:
            vconfig = VisionConfig.from_dict(vision_cfg)
            vconfig.model_path = str(model_path)

            screen_w, screen_h = 1920, 1080
            if test_frame is not None:
                h, w = test_frame.shape[:2]
                screen_w, screen_h = w, h

            vp = VisionProcessor(vconfig, screen_size=(screen_w, screen_h))
            logger.info(
                "Vision processor enabled (model=%s, input=%d, detect_every=%d frames)",
                model_path,
                vconfig.input_size,
                vconfig.detection_interval,
            )
            return vp
        except Exception as exc:
            logger.warning(
                "Vision processor failed to initialise: %s — vision disabled.",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Profile file resolution
    # ------------------------------------------------------------------

    def _resolve_profile_files(
        self,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
        """Load state_schema.json, regions.json, and macros.json.

        Uses *self._profile_path* when set; falls back to ``config/``.
        """
        profile = self._profile_path

        # -- state_schema.json -----------------------------------------------
        schema_data: dict[str, Any] | None = None
        if profile is not None:
            schema_path = profile / "state_schema.json"
            if schema_path.exists():
                schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
        if not schema_data or not schema_data.get("slots"):
            fallback = _CONFIG_DIR / "state_schema.json"
            if fallback.exists():
                schema_data = json.loads(fallback.read_text(encoding="utf-8"))
            else:
                logger.error("No state_schema.json found.")

        # -- regions.json -----------------------------------------------------
        regions_data: dict[str, Any] | None = None
        if profile is not None:
            regions_path = profile / "regions.json"
            if regions_path.exists():
                regions_data = json.loads(regions_path.read_text(encoding="utf-8"))
        if not regions_data or not regions_data.get("regions"):
            fallback = _CONFIG_DIR / "regions.json"
            if fallback.exists():
                regions_data = json.loads(fallback.read_text(encoding="utf-8"))
            else:
                logger.error("No regions.json found.")

        # -- macros.json ------------------------------------------------------
        macros_data: list[dict[str, Any]] = []
        if profile is not None:
            macros_path = profile / "macros.json"
            if macros_path.exists():
                raw = json.loads(macros_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    macros_data = raw
                elif isinstance(raw, dict):
                    macros_data = raw.get("macros", [])
        if not macros_data:
            fallback = _CONFIG_DIR / "macros.json"
            if fallback.exists():
                raw = json.loads(fallback.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    macros_data = raw
                elif isinstance(raw, dict):
                    macros_data = raw.get("macros", [])

        if not macros_data:
            logger.warning("No macros found — LLM will have no actions to choose.")

        return schema_data, regions_data, macros_data

    # ------------------------------------------------------------------
    # Public API  (called from GUI main thread)
    # ------------------------------------------------------------------

    def start(self, profile_path: Path | None = None) -> None:
        """Start the AI engine.

        May be called multiple times — ignores when not IDLE.
        *profile_path* can be updated between runs to switch profiles.

        Reloads config from disk so any settings changes made via the
        Settings Panel take effect immediately.  Vision enabled/disabled
        changes are applied by re‑building the vision processor and
        state processor before each run.
        """
        if self._state not in (EngineState.IDLE, EngineState.ERROR):
            logger.warning(f"Ignoring start request — engine is {self._state.value}.")
            return

        # Reload config from disk so Settings Panel changes take effect
        try:
            from src.config_manager import load_global_config

            fresh_config = load_global_config()
            self._config = fresh_config
            logger.info("Config reloaded from disk for this run.")
        except Exception as exc:
            logger.warning("Could not reload config: %s — using previous config.", exc)

        if profile_path is not None:
            self._profile_path = profile_path
            # Re-resolve profile files
            _, __, macros = self._resolve_profile_files()
            if macros:
                self._macros_data = macros

        self._set_state(EngineState.STARTING)
        self._window.call_async(
            lambda: self._window._set_engine_state(
                "Starting …", ThemeManager().palette.warning)
        )
        # Defer to async so vision/config re‑eval happens on the asyncio loop
        asyncio.run_coroutine_threadsafe(self._start_engine(), self._loop)

    def stop(self) -> None:
        """Gracefully stop the decision loop (keeps components alive)."""
        if self._state not in (EngineState.RUNNING, EngineState.PAUSED):
            logger.warning(f"Ignoring stop request — engine is {self._state.value}.")
            return

        self._set_state(EngineState.STOPPING)
        self._window.call_async(
            lambda: self._window._set_engine_state(
                "Stopping …", ThemeManager().palette.warning)
        )
        asyncio.run_coroutine_threadsafe(self._stop_engine(), self._loop)

    def pause(self) -> None:
        """Toggle pause / resume on the decision loop."""
        if self._state == EngineState.RUNNING:
            self._set_state(EngineState.PAUSED)
            self._window.call_async(
                lambda: self._window._set_engine_state(
                    "Paused", ThemeManager().palette.warning)
            )
            asyncio.run_coroutine_threadsafe(self._pause_engine(), self._loop)
        elif self._state == EngineState.PAUSED:
            self._set_state(EngineState.RUNNING)
            self._window.call_async(
                lambda: self._window._on_engine_started()
            )
            asyncio.run_coroutine_threadsafe(self._resume_engine(), self._loop)

    def shutdown(self) -> None:
        """Full tear‑down: stop loop, executor, capture, MCP, summariser, loop.

        Blocks the calling thread for up to 15 s waiting for async cleanup.
        Safe to call from any state.
        """
        logger.info("Shutdown requested …")

        # Schedule async shutdown ONLY if the event loop is still running.
        # If the loop is already closed or stopped (e.g., the engine was
        # never fully started, or the thread crashed), skip the async
        # cleanup to avoid "Event loop is closed" / coroutine-never-awaited.
        loop_ok = (
            self._loop is not None
            and not self._loop.is_closed()
            and self._loop.is_running()
        )
        if loop_ok:
            try:
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
                future.result(timeout=15.0)
            except (TimeoutError, Exception) as exc:
                logger.warning(f"Shutdown timed out or raised: {exc}")

            # Stop the event loop so run_forever() returns
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass  # loop already shutting down
        else:
            logger.debug("Event loop not running — skipping async shutdown phase.")

        # Join the engine thread
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Engine thread did not join within 5 s.")

        self._set_state(EngineState.IDLE)
        logger.info("Shutdown complete.")

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def health_monitor(self) -> Any:
        return self._health_monitor

    @property
    def macro_executor(self) -> Any:
        """Expose the MacroExecutor for debug overlay / external triggers."""
        return self._macro_executor

    @property
    def macros_data(self) -> list[dict[str, Any]]:
        """Expose the current macros list for the debug overlay."""
        return self._macros_data

    # ------------------------------------------------------------------
    # State management (thread‑safe via self._loop)
    # ------------------------------------------------------------------

    def _set_state(self, new_state: EngineState) -> None:
        old = self._state
        self._state = new_state
        logger.debug("Engine state: %s → %s", old.value, new_state.value)

    # ------------------------------------------------------------------
    # Async engine lifecycle (runs on the asyncio loop)
    # ------------------------------------------------------------------

    async def _start_engine(self) -> None:
        """Launch MCP server, macro executor, health checker, and decision loop.

        Before launching, re‑evaluates vision config and rebuilds the state
        processor + health monitor so that settings changes (especially
        vision.enabled) take effect without a full app restart.
        """
        try:
            # 0. Capture a frame for vision resolution (screen dimensions).
            #    Fall back to get_last_frame() if a live capture fails.
            test_frame = None
            if self._capture is not None:
                test_frame = self._capture.get_last_frame()
                if test_frame is None:
                    try:
                        test_frame = await self._capture.capture()
                    except Exception:
                        pass

            # 1. Re-evaluate vision config → rebuild state processor & health
            await self._refresh_vision(test_frame)
            self._state_processor = await self._build_state_processor()
            self._health_monitor = self._build_health_monitor()
            logger.info(
                "Vision & state processor rebuilt for this run. "
                "vision_enabled=%s",
                self._vision_processor is not None,
            )

            # 2. Start MCP memory server
            if self._mcp_manager is not None:
                try:
                    await self._mcp_manager.start()
                    # Create MCP client now that the server is running
                    from src.mcp_client import MCPMemoryClient

                    self._mcp_client = MCPMemoryClient(
                        base_url=self._config.get("mcp_url", "http://localhost:8000"),
                    )
                    logger.info("MCPMemoryClient connected.")
                    # Wire into health monitor
                    if self._health_monitor is not None:
                        self._health_monitor._mcp = self._mcp_client
                except RuntimeError as exc:
                    logger.warning(
                        "MCP memory server failed to start: %s — memory tier disabled.",
                        exc,
                    )
                    self._mcp_manager = None

            # 3. Start MacroExecutor
            if self._macro_executor is not None:
                await self._macro_executor.start()
                logger.info("MacroExecutor started.")

            # 4. Start MemorySummariser
            enable_summarization: bool = self._config.get("enable_summarization", True)
            if self._mcp_client is not None and enable_summarization:
                from src.memory_summariser import MemorySummariser

                summarisation_model = self._config.get(
                    "summarization_model", self._config.get("ollama_model", "")
                )
                self._summariser = MemorySummariser(
                    client=self._mcp_client,
                    ollama_url=self._config.get("ollama_url", "http://localhost:11434/v1"),
                    model=summarisation_model,
                )
                await self._summariser.start()
                logger.info("MemorySummariser started.")
            elif not enable_summarization:
                logger.info("Automatic summarisation disabled in config.")

            # 5. Start HealthMonitor background polling
            if self._health_monitor is not None:
                self._health_task = asyncio.create_task(
                    self._run_health_polling()
                )
                logger.info("HealthMonitor polling started.")

            # 6. Start Preview polling (Live Overlay feature)
            self._preview_task = asyncio.create_task(
                self._run_preview_polling()
            )
            logger.info("Preview polling started.")

            # 7. Launch decision loop
            from src.decision_loop import decision_loop

            self._loop_task = asyncio.create_task(
                decision_loop(
                    state_processor=self._state_processor,
                    macro_executor=self._macro_executor,
                    profile_macros=self._macros_data,
                    config=self._config,
                    capture_obj=self._capture,
                    mcp=self._mcp_client,
                    summariser=self._summariser,
                    health_monitor=self._health_monitor,
                ),
                name="decision-loop",
            )

            self._set_state(EngineState.RUNNING)
            self._window.call_async(
                lambda: self._window._set_engine_state(
                    "Running", ThemeManager().palette.success)
            )
            self._window.call_async(
                lambda: self._window._on_engine_started()
            )
            logger.info("Engine started — decision loop running.")

        except Exception as exc:
            logger.exception(f"Engine start failed: {exc}")
            self._set_state(EngineState.ERROR)
            self._window.call_async(
                lambda exc=exc: self._window._set_engine_state(
                    f"Error: {exc}", ThemeManager().palette.danger)
            )

    async def _stop_engine(self) -> None:
        """Cancel the decision loop and clean up runtime components."""
        await self._cancel_loop_task()

        # Stop summariser
        if self._summariser is not None:
            try:
                await self._summariser.stop()
            except Exception as exc:
                logger.warning(f"Summariser stop error: {exc}")
            self._summariser = None

        # Stop macro executor
        if self._macro_executor is not None:
            try:
                await self._macro_executor.stop()
            except Exception as exc:
                logger.warning(f"MacroExecutor stop error: {exc}")

        # Cancel preview polling
        if self._preview_task is not None:
            self._preview_task.cancel()
            try:
                await self._preview_task
            except asyncio.CancelledError:
                pass
            self._preview_task = None

        # Cancel health polling
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        # Close MCP client
        if self._mcp_client is not None:
            try:
                await self._mcp_client.close()
            except Exception as exc:
                logger.warning(f"MCP client close error: {exc}")
            self._mcp_client = None

        # Stop MCP server
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.stop()
            except Exception as exc:
                logger.warning(f"MCP server stop error: {exc}")
            self._mcp_manager = None

        self._set_state(EngineState.IDLE)
        self._window.call_async(
            lambda: self._window._on_engine_stopped()
        )
        logger.info("Engine stopped.")

    async def _pause_engine(self) -> None:
        """Pause the decision loop task without tearing down."""
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
            logger.info("Decision loop paused.")

    async def _resume_engine(self) -> None:
        """Resume the decision loop."""
        from src.decision_loop import decision_loop

        self._loop_task = asyncio.create_task(
            decision_loop(
                state_processor=self._state_processor,
                macro_executor=self._macro_executor,
                profile_macros=self._macros_data,
                config=self._config,
                capture_obj=self._capture,
                mcp=self._mcp_client,
                summariser=self._summariser,
                health_monitor=self._health_monitor,
            ),
            name="decision-loop",
        )
        logger.info("Decision loop resumed.")

    async def _shutdown(self) -> None:
        """Full async teardown — stop everything, close capture, kill subprocesses."""
        # Cancel decision loop
        await self._cancel_loop_task()

        # Stop summariser
        if self._summariser is not None:
            try:
                await self._summariser.stop()
            except Exception as exc:
                logger.warning(f"Summariser stop error during shutdown: {exc}")
            self._summariser = None

        # Stop macro executor
        if self._macro_executor is not None:
            try:
                await self._macro_executor.stop()
            except Exception as exc:
                logger.warning(f"MacroExecutor stop error during shutdown: {exc}")

        # Close screen capture
        if self._capture is not None:
            try:
                await self._capture.close()
            except Exception as exc:
                logger.warning(f"Capture close error during shutdown: {exc}")

        # Cancel preview polling
        if self._preview_task is not None:
            self._preview_task.cancel()
            try:
                await self._preview_task
            except asyncio.CancelledError:
                pass
            self._preview_task = None

        # Cancel health polling
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        # Close MCP client
        if self._mcp_client is not None:
            try:
                await self._mcp_client.close()
            except Exception as exc:
                logger.warning(f"MCP client close error during shutdown: {exc}")
            self._mcp_client = None

        # Stop MCP server (SIGTERM → wait → SIGKILL)
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.stop()
            except Exception as exc:
                logger.warning(f"MCP server stop error during shutdown: {exc}")
            self._mcp_manager = None

        # Cancel any remaining tasks (except this one)
        if self._loop is not None:
            current = asyncio.current_task()
            pending = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not current and not t.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        logger.info("Async shutdown complete.")

    async def _cancel_loop_task(self) -> None:
        """Cancel the decision loop task if running."""
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    # ------------------------------------------------------------------
    # Preview polling (Live Overlay feature — background asyncio task)
    # ------------------------------------------------------------------

    async def _run_preview_polling(self) -> None:
        """Poll the latest captured frame and state data every ~200 ms
        (5 FPS) and push to the Live Preview Overlay via call_async.

        This runs as a background asyncio task alongside the decision
        loop.  When the overlay is not active, ``push_preview_frame``
        is a fast no‑op on the GUI thread.

        Pushes engine state updates even when no frame is available so
        the overlay status bar stays accurate.
        """
        # Shared mutable container for the decision loop to write its
        # latest action and cycle stats into.
        _latest_action: list[Any] = [None]  # [action_name]
        _latest_cycle_ms: list[float] = [0.0]  # [cycle_time_ms]
        _latest_fps: list[float] = [0.0]  # [fps]

        # Inject into self so the decision loop can update them.
        # (The decision loop has no direct reference to the bridge,
        # but we can patch these onto the state_processor.)
        if self._state_processor is not None:
            self._state_processor._preview_action = _latest_action
            self._state_processor._preview_cycle_ms = _latest_cycle_ms
            self._state_processor._preview_fps = _latest_fps

        _consecutive_failures: int = 0
        _max_consecutive_failures: int = 10

        while True:
            try:
                engine_state_str = self._state.value

                # Only push if we have a capture object
                if self._capture is None:
                    # Still push engine state so overlay updates
                    self._window.call_async(
                        lambda es=engine_state_str: self._window.push_preview_frame(
                            engine_state=es,
                            regions=self._regions_for_preview,
                        )
                    )
                    await asyncio.sleep(0.5)
                    continue

                # Get latest frame from the capture
                last_frame = self._capture.get_last_frame()
                if last_frame is None:
                    try:
                        last_frame = await self._capture.capture()
                    except Exception:
                        last_frame = None

                if last_frame is not None:
                    _consecutive_failures = 0
                else:
                    _consecutive_failures += 1

                # Get latest state data from the state processor
                state_data: dict[str, Any] = {}
                detections: list[Any] = []
                if self._state_processor is not None:
                    try:
                        # Access the most recent GameState via internal ref
                        _last_state = getattr(self._state_processor, "_last_game_state", None)
                        if _last_state is not None:
                            state_data = _last_state.to_dict()
                    except Exception:
                        pass

                    # Get vision detections if available
                    try:
                        _vision = getattr(self._state_processor, "_vision", None)
                        if _vision is not None and getattr(_vision, "is_enabled", False):
                            _last_spatial = getattr(self._state_processor, "_last_spatial_ctx", None)
                            if _last_spatial is not None:
                                detections = getattr(_last_spatial, "detections", [])
                    except Exception:
                        pass

                # Push to GUI — always push engine_state and regions,
                # even if frame is None (so the overlay shows status).
                self._window.call_async(
                    lambda f=(last_frame.copy() if last_frame is not None else None),
                    r=self._regions_for_preview,
                    sd=state_data, d=detections,
                    la=_latest_action[0], cms=_latest_cycle_ms[0],
                    fps=_latest_fps[0], es=engine_state_str:
                        self._window.push_preview_frame(
                            frame=f,
                            regions=r,
                            state_data=sd,
                            detections=d,
                            last_action=la,
                            cycle_time_ms=cms,
                            fps=fps,
                            engine_state=es,
                        )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug(f"Preview poll error (non-fatal): {exc}")

            # Adaptive sleep: if repeatedly failing, slow down to 1 FPS
            if _consecutive_failures > _max_consecutive_failures:
                await asyncio.sleep(1.0)
            else:
                await asyncio.sleep(0.2)  # 5 FPS

    # ------------------------------------------------------------------
    # Health polling (background asyncio task)
    # ------------------------------------------------------------------

    async def _run_health_polling(self) -> None:
        """Poll HealthMonitor every 5 s and push to GUI."""
        while True:
            try:
                status = await self._health_monitor.get_overall_status()
                # Inject vision throttle override (when auto‑disabled by latency)
                self._inject_vision_throttle_override(status)
                # Push to MainWindow status bar
                self._window.call_async(
                    lambda s=status: self._apply_health_status(s)
                )
                # Push to HealthPanel if visible
                try:
                    hp = self._window.health_panel
                    if hp is not None:
                        hp.push_status(status)
                except (RuntimeError, Exception):
                    pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Health poll error: {exc}")
            await asyncio.sleep(5.0)

    def _inject_vision_throttle_override(self, status: dict[str, Any]) -> None:
        """Override the Vision health status when adaptive throttling has
        auto‑disabled the vision processor (latency spikes).

        The native ``HealthMonitor.check_vision()`` reports the processor's
        own health — but when the *decision loop* disables it due to high
        pipeline latency (§7.7D), we need to surface that in the UI.
        """
        services = status.setdefault("services", {})
        vision = services.get("vision", {})
        if self._state_processor is not None:
            metrics = getattr(self._state_processor, "_metrics", None)
            if metrics is not None and getattr(metrics, "vision_disabled_by_throttle", False):
                vision["healthy"] = False
                vision["reason"] = "Auto‑disabled — consecutive high latency cycles"

    def _apply_health_status(self, status: dict[str, Any]) -> None:
        """Update the MainWindow status bar from a health dict.

        MUST be called on the tkinter main thread (via call_async).
        """
        services = status.get("services", {})
        level = status.get("level", "UNKNOWN")

        # Engine state colour
        tm = ThemeManager()
        p = tm.palette
        if level == "OK":
            engine_colour = p.success
        elif level == "DEGRADED":
            engine_colour = p.warning
        elif level == "STOPPING":
            engine_colour = p.danger
        else:
            engine_colour = p.disabled_fg

        if self._state == EngineState.RUNNING:
            self._window._set_engine_state(status.get("reason", level), engine_colour)

        # Ollama
        ollama = services.get("ollama", {})
        ollama_healthy = ollama.get("healthy", False)
        self._window.set_ollama_status(
            ollama.get("reason", "Unknown"), ollama_healthy
        )

        # MCP
        mcp = services.get("mcp", {})
        mcp_healthy = mcp.get("healthy", False)
        self._window.set_mcp_status(
            mcp.get("reason", "Unknown"), mcp_healthy
        )


# ---------------------------------------------------------------------------
# Config proxy for calibration (lightweight callable shim)
# ---------------------------------------------------------------------------


def _create_config_proxy(
    config: dict[str, Any],
    profile_path: Path | None,
) -> Any:
    """Return a duck‑typed object that MainWindow can use for
    ``load_state_schema``, ``load_regions``, and ``save_regions`` calls
    during calibration.
    """
    class _Proxy:
        def load_state_schema(self) -> dict[str, Any]:
            from src.game_state import StateSchema
            from src.config_manager import DEFAULT_STATE_SCHEMA

            path = (
                profile_path / "state_schema.json"
                if profile_path
                else _CONFIG_DIR / "state_schema.json"
            )
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return dict(DEFAULT_STATE_SCHEMA)

        def load_regions(self) -> dict[str, Any]:
            path = (
                profile_path / "regions.json"
                if profile_path
                else _CONFIG_DIR / "regions.json"
            )
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return {"version": "1.0.0", "regions": []}

        def save_regions(self, data: dict[str, Any]) -> None:
            path = (
                profile_path / "regions.json"
                if profile_path
                else _CONFIG_DIR / "regions.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(path)

    return _Proxy()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "GUIEngineBridge",
    "EngineState",
]