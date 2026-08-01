#!/usr/bin/env python3
"""
Ludomorph - Main entry point.

A universal, external application that injects an autonomous LLM agent
into any PC game by capturing the screen and simulating keyboard/mouse input.

Phase 1 (Foundation): asyncio engine, screen capture, input injection,
                      window management, static macro playback, Ollama check.
Phase 2 (Perception & Decision): Full async perception → LLM → action loop.
"""

import argparse
import asyncio
import json
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so that 'src' is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logging_config import setup_logging, get_logger
from src import __version__
from src.config_manager import (
    load_global_config,
    list_profiles,
    load_macros as load_profile_macros,
    load_regions,
    load_state_schema,
)
from src.input_controller import InputController, InputError
from src.window_focus import WindowFocusManager
from src.screen_capture import ScreenCapture, CaptureConfig, WindowTracker
from src.macro_executor import MacroExecutor, MacroRequest, MacroPriority
from src.ollama_health import ollama_health_check
from src.mcp_server import MCPServerManager
from src.vision_detector import VisionProcessor, VisionConfig

logger = get_logger(__name__)


# =============================================================================
# Unhandled exception hook — logs crash forensics before exit
# =============================================================================

def _install_excepthook() -> None:
    """Install a sys.excepthook that logs unhandled exceptions."""
    _original_hook = sys.excepthook

    def _log_unhandled(exc_type, exc_value, exc_tb):
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        logger.critical(
            "UNHANDLED EXCEPTION — the process will now exit.\n{}",
            "".join(tb_lines).rstrip(),
        )
        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _log_unhandled


# =============================================================================
# Macro loader — reads the first named macro from config/macros.json
# =============================================================================

_MACROS_PATH = Path(__file__).resolve().parent / "config" / "macros.json"


def load_named_macro(name: str) -> list[dict[str, Any]] | None:
    """Load a specific named macro from config/macros.json."""
    try:
        with open(_MACROS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error(f"Cannot load macros file ({_MACROS_PATH}): {exc}")
        return None

    macros: list[dict[str, Any]] = data.get("macros", [])
    for entry in macros:
        if entry.get("name") == name:
            return entry.get("actions", [])
    return None


# =============================================================================
# Phase 1 entry point
# =============================================================================

def main() -> None:
    """Application entry point."""

    # ------------------------------------------------------------------
    # 0. Parse CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Ludomorph — universal LLM agent for any PC game.",
    )
    parser.add_argument(
        "--check",
        "--dry-run",
        dest="check_only",
        action="store_true",
        help="Run health probes (config, capture, Ollama) without executing macros.",
    )
    parser.add_argument(
        "--macro",
        dest="macro_name",
        default=None,
        metavar="NAME",
        help="Name of a macro from config/macros.json to execute (default: 'type_hello').",
    )
    parser.add_argument(
        "--ready-delay",
        dest="ready_delay",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Seconds to wait before executing the macro, giving you time to "
        "switch focus to the target window (default: 3).",
    )
    parser.add_argument(
        "--loop",
        dest="decision_loop",
        action="store_true",
        help="Run the Phase 2 decision loop (perception → LLM → action) "
        "instead of the Phase 1 single-macro execution.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        default=None,
        metavar="NAME",
        help="Name of the game profile to use for the decision loop "
        "(Phase 2). Must exist under ~/.gameai/profiles/<name>/.",
    )
    parser.add_argument(
        "--gui",
        dest="launch_gui",
        action="store_true",
        help="Launch the Tkinter GUI instead of the CLI (Phase 5).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Set up logging (use default level; config loads afterwards
    #    and can re-configure if needed).
    # ------------------------------------------------------------------
    setup_logging(log_level="INFO")

    # Install crash-forensics hook as early as possible
    _install_excepthook()

    logger.info(f"Ludomorph v{__version__} starting up ...")

    if args.check_only:
        logger.info("Running in CHECK mode — no macros will be executed.")

    # ------------------------------------------------------------------
    # 2. Load global configuration (or write defaults on first run).
    # ------------------------------------------------------------------
    try:
        config = load_global_config()
    except Exception as exc:
        logger.error(f"Failed to load config: {exc}")
        sys.exit(1)

    logger.info(
        f"Config loaded. "
        f"Ollama URL: {config.get('ollama_url')}, "
        f"Model: {config.get('ollama_model')}"
    )

    # Re-apply log_level from config if different
    configured_level = config.get("log_level", "INFO")
    if configured_level != "INFO":
        setup_logging(log_level=configured_level)
        logger.debug(f"Log level updated to {configured_level}")

    # ------------------------------------------------------------------
    # 3. List available profiles (informational, nothing to do yet).
    # ------------------------------------------------------------------
    profiles = list_profiles()
    if profiles:
        logger.info(f"Found {len(profiles)} profile(s): {', '.join(profiles)}")
    else:
        logger.info("No game profiles found. Create one via the GUI (Phase 5).")

    # ------------------------------------------------------------------
    # 4. Initialise the input controller (Step 1.3).
    # ------------------------------------------------------------------
    try:
        input_ctrl = InputController(config)
        logger.info(f"Input backend active: {input_ctrl.backend_name}")
    except InputError as exc:
        logger.error(f"Input initialisation failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Initialise the window focus manager (Step 1.4).
    # ------------------------------------------------------------------
    focus_mgr = WindowFocusManager(config, input_controller=input_ctrl)
    logger.info(f"Window focus manager ready. Compositor: {focus_mgr.compositor.value}")

    # ------------------------------------------------------------------
    # 6. Dispatch to Phase 1 or Phase 2 based on CLI flags.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 6a. Start MCP memory server (Phase 3.1) if enabled
    # ------------------------------------------------------------------
    mcp_manager: MCPServerManager | None = None
    mcp_enabled: bool = config.get("mcp_enabled", True)

    if mcp_enabled:
        mcp_manager = MCPServerManager(
            port=8000,  # fixed per spec
            ready_timeout=10.0,
            grace_period=5.0,
        )
        try:
            # Start is blocking until ready — run inside a temp event loop
            # or schedule it within the main async entry point.
            # We defer the actual start to the async entry points below.
            logger.info("MCP memory server will be started in async context.")
        except Exception as exc:
            logger.warning("MCP server setup failed: {} — memory tier disabled.", exc)
            mcp_manager = None
    else:
        logger.info("MCP memory server disabled in config (mcp_enabled=false).")

    # ------------------------------------------------------------------
    # 7. Dispatch: GUI, Phase 2, or Phase 1 based on CLI flags.
    # ------------------------------------------------------------------
    if args.launch_gui:
        logger.info("--- Launching GUI (Phase 5) ---")
        from src.gui import launch_gui
        launch_gui()
    elif args.decision_loop:
        logger.info("--- Phase 2 Decision Loop ---")
        asyncio.run(_phase2_main(config, input_ctrl, args, mcp_manager))
    else:
        asyncio.run(_phase1_main(config, input_ctrl, args, mcp_manager))


async def _phase1_main(
    config: dict[str, Any],
    input_ctrl: Any,
    args: Any,
    mcp_manager: MCPServerManager | None = None,
) -> None:
    """
    Phase 1 main coroutine.

    Runs screen capture probe, optionally executes a macro from
    config/macros.json, and performs the Ollama health check.
    """

    # -- Capture probe ----------------------------------------------------
    logger.info("--- Screen Capture Probe ---")
    capture_config = CaptureConfig()
    tracker = WindowTracker(capture_config)
    capture = ScreenCapture(capture_config, tracker=tracker)

    frame = await capture.capture()
    if frame is not None:
        h, w = frame.shape[:2]
        channels = frame.shape[2] if frame.ndim > 2 else 1
        logger.info(f"Capture OK — frame {w}×{h} (channels: {channels})")
    else:
        logger.warning("Capture returned None (no window / region available).")

    await capture.close()

    # -- Macro execution --------------------------------------------------
    if args.check_only:
        logger.info("Macro execution SKIPPED (--check).")
    else:
        macro_name = args.macro_name or "type_hello"
        actions = load_named_macro(macro_name)
        if actions is None:
            logger.warning(
                f"Macro '{macro_name}' not found in config/macros.json. "
                f"Skipping macro execution."
            )
        else:
            # -- Ready countdown --
            # Let the user switch focus to the target window before
            # we start injecting keystrokes.
            delay = max(args.ready_delay, 0.0)
            if delay > 0:
                logger.info(
                    f"Starting macro in {delay:.0f} second(s) — "
                    f"switch focus to the target window now!"
                )
                for remaining in range(int(delay), 0, -1):
                    logger.info(f"  {remaining}...")
                    await asyncio.sleep(1)

            logger.info(f"Executing macro '{macro_name}' ({len(actions)} step(s)) ...")
            executor = MacroExecutor(input_ctrl)
            await executor.start()
            try:
                request = MacroRequest(
                    name=macro_name,
                    actions=actions,
                    priority=MacroPriority.NORMAL,
                )
                future = await executor.submit(request)
                await future
                logger.info(f"Macro '{macro_name}' completed successfully.")
            except Exception as exc:
                logger.error(f"Macro '{macro_name}' failed: {exc}")
            await executor.stop()

    # -- Ollama health check ----------------------------------------------
    logger.info("--- Ollama Health Check ---")
    health = await ollama_health_check(config)
    if health.healthy:
        logger.info(
            f"Ollama healthy — version={health.version}, model={health.configured_model}"
        )
    else:
        logger.warning(f"Ollama health issue: {health.error}")

    # -- MCP memory server cleanup (Phase 3.1) --
    if mcp_manager is not None:
        await mcp_manager.stop()

    logger.info("Phase 1 done.")


# =============================================================================
# Phase 2 entry point  (perception → LLM → action loop)
# =============================================================================


async def _phase2_main(
    config: dict[str, Any],
    input_ctrl: InputController,
    args: Any,
    mcp_manager: MCPServerManager | None = None,
) -> None:
    """Run the Phase 2 decision loop.

    Requires a profile name (``--profile``) whose ``regions.json``,
    ``state_schema.json``, and ``macros.json`` are loaded from
    ``~/.gameai/profiles/<name>/``.

    Falls back to the bundled ``config/`` directory files if the
    profile directory is not found.

    Phase 3.1: If *mcp_manager* is provided, the MCP memory server
    is started before the decision loop and stopped on exit.
    """
    # ------------------------------------------------------------------
    # 0. Start MCP memory server (Phase 3.1)
    # ------------------------------------------------------------------
    if mcp_manager is not None:
        try:
            await mcp_manager.start()
        except RuntimeError as exc:
            logger.warning(
                "MCP memory server failed to start: {} — "
                "memory tier disabled for this session.",
                exc,
            )
            mcp_manager = None

    profile_name: str = args.profile_name or ""

    # ------------------------------------------------------------------
    # 1. Resolve profile files
    # ------------------------------------------------------------------
    _CONFIG_DIR = _PROJECT_ROOT / "config"

    # State schema
    schema_data: dict[str, Any] | None = None
    if profile_name:
        schema_data = load_state_schema(profile_name)
    if not schema_data or not schema_data.get("slots"):
        # Fall back to the bundled schema in config/state_schema.json
        fallback = _CONFIG_DIR / "state_schema.json"
        if fallback.exists():
            schema_data = json.loads(fallback.read_text(encoding="utf-8"))
        else:
            logger.error(
                "No state_schema.json found for profile %r or in %s",
                profile_name,
                _CONFIG_DIR,
            )
            sys.exit(1)

    # Regions
    regions_data: dict[str, Any] | None = None
    if profile_name:
        regions_data = load_regions(profile_name)
    if not regions_data or not regions_data.get("regions"):
        fallback = _CONFIG_DIR / "regions.json"
        if fallback.exists():
            regions_data = json.loads(fallback.read_text(encoding="utf-8"))
        else:
            logger.error(
                "No regions.json found for profile %r or in %s",
                profile_name,
                _CONFIG_DIR,
            )
            sys.exit(1)

    # Macros
    macros_data: list[dict[str, Any]] = []
    if profile_name:
        macros_data = load_profile_macros(profile_name)
    if not macros_data:
        fallback = _CONFIG_DIR / "macros.json"
        if fallback.exists():
            raw = json.loads(fallback.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                macros_data = raw.get("macros", [])
            elif isinstance(raw, list):
                macros_data = raw
    if not macros_data:
        logger.warning(
            "No macros found for profile %r or in %s — "
            "LLM will have no actions to choose.",
            profile_name,
            _CONFIG_DIR,
        )

    # ------------------------------------------------------------------
    # 2. Build StateSchema and RegionProfile
    # ------------------------------------------------------------------
    from src.game_state import StateSchema
    from src.region_profile import RegionProfile

    try:
        schema = StateSchema.from_dict(schema_data)
        logger.info(
            "State schema loaded: %d slot(s)",
            len(schema.slots),
        )
    except Exception as exc:
        logger.error(f"Failed to load state schema: {exc}")
        sys.exit(1)

    try:
        profile = RegionProfile.from_dict(regions_data)
        logger.info(
            "Region profile loaded: %d region(s)",
            len(profile.regions),
        )
    except Exception as exc:
        logger.error(f"Failed to load region profile: {exc}")
        sys.exit(1)

    # Validate region roles against schema (non‑fatal warnings)
    region_roles = {
        r.role
        for r in profile.regions
        if r.role
    }
    warnings = schema.validate_region_roles(region_roles)
    for w in warnings:
        logger.warning(w)

    # ------------------------------------------------------------------
    # 3. Set up OCR module
    # ------------------------------------------------------------------
    from src.ocr_module import OCRModule, OCRConfig

    ocr_config = OCRConfig(
        cache_ttl_seconds=config.get("ocr_cache_ttl_seconds", 2.0),
    )
    ocr = OCRModule(ocr_config)
    logger.info("OCR module initialised.")

    # ------------------------------------------------------------------
    # 4. Build StateProcessor
    # ------------------------------------------------------------------
    from src.state_processor import StateProcessor

    # --- Phase 4.1: Optionally wire up the vision processor ---
    vision_processor: Any = None
    vision_cfg = config.get("vision", {})
    if vision_cfg.get("enabled", False):
        # Resolve model path relative to project root if not absolute
        model_path_str = vision_cfg.get("model_path", "models/yolo11n.onnx")
        model_path = Path(model_path_str)
        if not model_path.is_absolute():
            model_path = _PROJECT_ROOT / model_path_str

        try:
            vconfig = VisionConfig.from_dict(vision_cfg)
            vconfig.model_path = str(model_path)

            # Determine screen size from capture (or use a default)
            screen_w, screen_h = 1920, 1080
            if test_frame is not None:
                h, w = test_frame.shape[:2]
                screen_w, screen_h = w, h

            vision_processor = VisionProcessor(vconfig, screen_size=(screen_w, screen_h))
            logger.info(
                "Vision processor enabled (model=%s, input=%d, detect_every=%d frames)",
                model_path,
                vconfig.input_size,
                vconfig.detection_interval,
            )
        except Exception as exc:
            logger.warning(
                "Vision processor failed to initialise: %s — "
                "vision disabled for this session.",
                exc,
            )
            vision_processor = None

    state_processor = StateProcessor(
        profile=profile,
        ocr_module=ocr,
        schema=schema,
        vision_processor=vision_processor,
        cache_ttl=config.get("state_cache_ttl_seconds", 0.3),
    )
    logger.info("StateProcessor initialised.")

    # ------------------------------------------------------------------
    # 5. Set up screen capture
    # ------------------------------------------------------------------
    capture_config = CaptureConfig()
    tracker = WindowTracker(capture_config)
    capture = ScreenCapture(capture_config, tracker=tracker)

    test_frame = await capture.capture()
    if test_frame is not None:
        h, w = test_frame.shape[:2]
        logger.info(f"Capture OK — frame {w}×{h}")
    else:
        logger.warning(
            "Initial capture returned None — "
            "is a game window visible?"
        )

    # ------------------------------------------------------------------
    # 6. Start MacroExecutor
    # ------------------------------------------------------------------
    macro_executor = MacroExecutor(input_ctrl)
    await macro_executor.start()

    # ------------------------------------------------------------------
    # 7. Run the decision loop
    # ------------------------------------------------------------------
    from src.decision_loop import decision_loop

    # Set up graceful shutdown on SIGINT / SIGTERM
    loop = asyncio.get_running_loop()
    main_task = asyncio.current_task()

    def _shutdown_callback() -> None:
        logger.info("Shutdown signal received.")
        if main_task is not None:
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_callback)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # --- Instantiate real MCP client if server is healthy ---
    mcp_client: Any = None
    if mcp_manager is not None and mcp_manager.is_healthy:
        from src.mcp_client import MCPMemoryClient
        mcp_client = MCPMemoryClient(
            base_url=config.get("mcp_url", "http://localhost:8000"),
        )
        logger.info("MCPMemoryClient connected to {}", mcp_client.base_url)

    # --- Phase 3.4: Automatic summarisation background task ---
    summariser: Any = None
    enable_summarization: bool = config.get("enable_summarization", True)
    if mcp_client is not None and enable_summarization:
        from src.memory_summariser import MemorySummariser

        summarisation_model = config.get(
            "summarization_model", config.get("ollama_model", "")
        )
        summariser = MemorySummariser(
            client=mcp_client,
            ollama_url=config.get("ollama_url", "http://localhost:11434/v1"),
            model=summarisation_model,
        )
        await summariser.start()
    elif not enable_summarization:
        logger.info("Automatic summarisation disabled in config.")
    elif mcp_client is None:
        logger.info("MCP client unavailable — summarisation disabled.")

    try:
        await decision_loop(
            state_processor=state_processor,
            macro_executor=macro_executor,
            profile_macros=macros_data,
            config=config,
            capture_obj=capture,
            mcp=mcp_client,
            summariser=summariser,  # Phase 3.4
        )
    except asyncio.CancelledError:
        logger.info("Phase 2 main cancelled — shutting down.")
    finally:
        # Cleanup
        # Stop summariser first (Phase 3.4)
        if summariser is not None:
            await summariser.stop()
        await macro_executor.stop()
        await capture.close()
        # Close MCP client (Phase 3.2)
        if mcp_client is not None:
            await mcp_client.close()
        # Stop MCP memory server (Phase 3.1)
        if mcp_manager is not None:
            await mcp_manager.stop()
        logger.info("Phase 2 shutdown complete.")


if __name__ == "__main__":
    main()
