# Ludomorph

Have a local LLM play your game, in realish time.

Currently a work in progress.

Version 0.6.0

## Phase 6 Complete

### What This Version Does

Phase 6 delivers a fully-integrated, usable standalone desktop application where every component works together. The GUI now actually drives the decision loop, health monitoring enables graceful degradation, macro execution is audited for timing accuracy, window capture is hardened across platforms, and extensive user-driven refinements have been incorporated.

**New in 0.6.0:**

- **GUI-to-Engine Wiring** - Start/Stop buttons now spawn and cancel the async decision loop via `asyncio.ensure_future`. Loguru output streams into the GUI log dashboard in real time. HealthMonitor status (OK / DEGRADED / STOPPING) displays in the colour-coded status bar. Settings panel changes write to `config.json` and take effect on next run. Clean shutdown sequence: stop loop -> stop executor -> close capture -> stop MCP -> cancel summariser -> exit, with no orphaned subprocesses.

- **Window Capture & Focus Hardening** - Tested against diverse targets: native games, emulators, browser-based games, windowed/borderless/fullscreen modes. Handles edge cases: minimized windows, windows on other virtual desktops, multi-monitor setups with mixed DPI scaling. ScreenCapture returns correct frames immediately after focus switches. Fallback to user-defined fixed-region capture when window detection fails, with a clear warning in the UI.

- **Macro Execution Audit & Accuracy** - Sub-10ms drift for hold durations, correct key-up sequencing, no dropped keys. Cancelled macros fully release all held keys. High-priority interrupt macros preempt running low-priority ones cleanly with full key release. MacroResolver correctly translates dynamic vision-based coordinates into actual screen coordinates for the active window, accounting for window offsets and DPI scaling.

- **HealthMonitor & Graceful Degradation** - Central system-health authority polls all services (Ollama, Tesseract, MCP, game window) on a configurable interval. Degradation logic wired into the decision loop: LLM unreachable -> rule-based fallback decisions; OCR unavailable -> colour-bars-only mode; MCP down -> memory-less mode; Game window lost -> graceful stop with user notification. Circuit-breaker patterns prevent tight restart loops for persistently failing components.

- **MCP Memory Server Hardening** - Auto-spawn on port 8000 works from both CLI and GUI launch paths. Health-check with exponential backoff retry on startup failure (up to 3 restarts). Memory persistence verified: store events during a session, stop the app, restart, queryable memories. Summarisation background task correctly compresses short-term events into medium/long-term summaries on the configured threshold. Graceful shutdown: SIGTERM -> wait -> SIGKILL pattern on the subprocess.

- **Vision Pipeline Verification** - With vision disabled (default), there is zero performance impact and no YOLO model loaded into memory. With vision enabled, spatial context is injected into the LLM prompt and MacroResolver correctly translates detections to coordinates. Vision-OCR scheduling rules verified (OCR priority, staggered vision, timeout & cancel). Auto-disable safety net: if the vision pipeline exceeds 500ms for 5 consecutive cycles, it disables itself and notifies the user.

- **Specification Compliance** - All functional requirements verified: FR-01 (screen capture with region extraction), FR-02 (LLM decisions within 200ms with rule-based fallback), FR-03 (accurate macro playback on all three platform backends), FR-04 (persistent memory with summarisation), FR-05 (complete GUI - profiles, calibration, settings), FR-06 (zero external installs - Tesseract and MCP bundled), FR-07 (toggleable vision with zero impact when off). Non-functional requirements: NFR-01 (<300ms p95 end-to-end latency), NFR-02 (no data leaves the machine), NFR-03 (<5 minute first-time setup).

- **Visual Macro Builder (Phase 6.8)** - Card-based visual editor with move-up/move-down/delete controls, per-type edit widgets, COCO class filterable dropdown for dynamic steps. Five selection strategies: nearest-to-center, nearest-to-point, highest-confidence, largest, random. JSON/Visual mode toggle with two-way sync. Existing macros without `selection_strategy` default to `nearest_to_center` (backward compatible).

- **Live Preview Overlay (Phase 6.8)** - Resizeable non-modal window rendering the AI's live "view" of the game. Colour-coded region overlays: green for OCR regions (with live OCR text), blue for colour bar regions (with fill percentages), orange-red for YOLO detection bounding boxes with class labels. Dedicated background polling task at 5 FPS. Zero performance impact when hidden.

- **Centralised Theme System (Phase 6.8)** - Single `ThemePalette` frozen dataclass with 30+ colour/font fields. Dark and light built-in palettes. System-theme detection (GNOME/KDE/Windows/macOS). Custom background colour override with live preview in Settings > Theme/UI. Platform-aware font stacks (Segoe UI / DejaVu Sans / Noto Sans). Runtime theme switching via `<<ThemeChanged>>` event across all widgets.

- **Dynamic OCR Region Locator (Phase 6.8)** - Three anchoring modes for moving text positions: `vision_anchor` (object-relative), `motion` (change-triggered), and `text_detection` (EAST model-based search). Lazy EAST model loading (30MB download on first use of `text_detection` mode only). Sub-region aggregation via highest-confidence. Backward compatible - existing static regions unchanged.

- **Branding Update (Phase 6.8)** - Project renamed to "Ludomorph" (from Latin *ludus* = game/play + Greek *morphē* = form/shape) for trademark safety and distinctiveness. Internal identifiers (`~/.gameai/`, `.gameai_profile` extension) retained for backwards compatibility.

- **In-Game Debug Overlay (Phase 6.8)** - Semi-transparent tabbed debug menu triggered by pressing `~` (grave/tilde). Floats above the game without stealing focus so the AI continues running uninterrupted. Macros tab: select any macro from the active profile and click `▶ Run` to test it in-game at high priority (preempts the AI's current macro). Draggable window with session-position memory. Placeholder tabs (Toggles, State, Perf) ready for future debug tools. Global hotkey listener runs as a daemon thread via pynput — zero overhead when overlay is hidden.

## Phase History

| Phase | Version | Key Features |
|-------|---------|-------------|
| 1 | 0.1.0 | Screen capture, input injection, static macro playback, window auto-focus, Ollama health check |
| 2 | 0.2.0 | Colour-bar detection, OCR text reading, game-state model, LLM decision loop, adaptive frame skipping |
| 3 | 0.3.0 | Bundled MCP memory server, persistent searchable memory, automatic summarisation |
| 4 | 0.4.0 | YOLO object detection (ONNX), optical-flow tracking, dynamic macro resolution, spatial-context LLM prompts |
| 5 | 0.5.0 | Tkinter GUI, transparent calibration overlay, colour-bar capture, macro editor, settings panel, health monitor, profile manager |
| 6 | 0.6.0 | GUI-to-engine wiring, window capture hardening, macro audit, health degradation, MCP hardening, vision verification, spec compliance, macro builder, live preview overlay, theme system, dynamic OCR locator, branding |

All features from previous phases remain available and work as before.

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR: `sudo apt install tesseract-ocr`
- On X11: `python-xlib` is installed automatically via pip
- tkinter: `sudo apt install python3-tk` (usually pre-installed;
  required for the GUI)

### Setup

```bash
cd AI-Game-Master
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Wayland (optional)

The `python-ydotool` package requires `evdev`, which needs the Python
development headers to compile:

```bash
sudo apt install python3-dev
pip install python-ydotool
```

If you are on X11 or Windows, `pynput` is sufficient and you can skip
this step.

### Ollama

The decision loop requires a running Ollama instance with a model pulled.
The configured model is `phi3.5:3.8b-mini-instruct-q4_K_M` by default
(change `ollama_model` in `config.json` if you prefer a different model).

```bash
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

## Quick Start

### Desktop GUI (new in 0.5.0)

```bash
source venv/bin/activate
python main.py --gui
```

Launches the full Tkinter application with log dashboard, calibration
overlay, macro editor, settings, health monitor, and profile manager.

### Command-Line Modes

```bash
# Single macro execution
python main.py --macro test --ready-delay 3

# Continuous decision loop
python main.py --loop

# Use a per-game profile from ~/.gameai/profiles/<name>/
python main.py --loop --profile <name>
```

## Tests

Tests Removed from repo as of v0.6.0. They will return in experimental branches after the 1.0 release.

## Configuration Files

All JSON files live under `config/` (bundled defaults) or
`~/.gameai/profiles/<name>/` (per-game overrides).  The GUI Settings
panel writes directly to `config.json` at runtime.

| File | Purpose |
|------|---------|
| `config.json` | Global settings (Ollama URL, model, vision, diff, cache TTLs, theme) |
| `regions.json` | Screen region definitions with `role` -> state slot mapping |
| `state_schema.json` | Declares known state slots and their types/priorities |
| `macros.json` | Named macro definitions the LLM can choose from |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Packages not installed or venv not activated | `source venv/bin/activate && pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'pynput'` | evdev failed to build | If on X11: `pip install --no-deps pynput` |
| Capture returns `None` | No window found or Wayland without `capture_region` | On Wayland, set `capture_region` in config. On X11, ensure a game window is active. |
| Macro does nothing or wrong keys | Input backend mismatch or evdev missing | Check the log for `Input backend active: ...`. On Wayland, ensure ydotoold is running. |
| Window auto-focus fails | "Game Window" is a placeholder | Real window titles come from game profiles created with the calibration UI (`--gui`). |
| Ollama warning or LLM fallback to WAIT | Ollama not running or model not pulled | Start `ollama serve`, then `ollama pull <model>`. The agent gracefully falls back to the previous action or WAIT. |
| Tesseract error / OCR unavailable | `tesseract-ocr` not installed | `sudo apt install tesseract-ocr` |
| Live demo window doesn't appear | No X11 display or headless environment | The live demo requires a GUI display (uses OpenCV `cv2.imshow`). |
| GUI fails to start (`_tkinter` not found) | `python3-tk` system package missing | `sudo apt install python3-tk` |
| `ModuleNotFoundError: No module named 'tkinter'` | venv can't see system tkinter | Set `include-system-site-packages = true` in `venv/pyvenv.cfg` |

## AI Usage Disclosure

This project was almost entirely coded by AI models (Deepseek V4 Pro,
with Embedding Models assistance), with extensive human guidance,
project management, and testing.