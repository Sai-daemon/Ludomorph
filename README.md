# AI-Game-Master

Have a local LLM play your game, in realish time.

Currently a work in progress.

Version 0.5.0

## Phase 5 Complete

### What This Version Does

Phase 5 delivers a standalone desktop application with a full Tkinter GUI,
region calibration overlay, colour‑bar capture, macro editor, settings
panel, health monitor, profile manager, and end‑to‑end testing.  The app
launches via `python main.py --gui` and provides everything needed to
create game profiles without editing JSON by hand.

**New in 0.5.0:**

- **Tkinter shell & async bridge** — `AsyncTk` event‑loop bridge,
  real‑time `LogDashboard` sink for loguru, Start/Stop/Pause controls,
  and colour‑coded status bar.
- **Region calibration overlay** — transparent full‑screen window with
  click‑and‑drag bounding boxes, role/type assignment, and live OCR
  preview.
- **Colour‑bar calibration capture** — "Capture Empty/Full" buttons
  grab live frames through the overlay, auto‑compute HSV thresholds,
  preview mask visualisation, and bar‑type/orientation dropdowns.
- **Macro JSON editor** — syntax‑highlighted editor with add/delete/
  refresh actions.
- **Settings panel** — tabbed configuration for Ollama URL, model, MCP
  toggle, vision toggle, input backend, and more.
- **Health Monitor panel** — per‑component status display (Ollama,
  Tesseract, MCP, game window) with actionable error messages and
  colour‑coded indicators.
- **Profile manager** — import/export `.gameai_profile` zip archives
  with full 8‑point validation (manifest, required files, valid JSON,
  action types, region bounds, icon, paths, encoding).
- **End‑to‑end tests** — 208 passing tests across 6 suites covering
  calibration, macros, settings, health, profiles, and full agent
  pipeline integration.

## Phase History

| Phase | Version | Key Features |
|-------|---------|-------------|
| 1 | 0.1.0 | Screen capture, input injection, static macro playback, window auto‑focus, Ollama health check |
| 2 | 0.2.0 | Colour‑bar detection, OCR text reading, game‑state model, LLM decision loop, adaptive frame skipping |
| 3 | 0.3.0 | Bundled MCP memory server, persistent searchable memory, automatic summarisation |
| 4 | 0.4.0 | YOLO object detection (ONNX), optical‑flow tracking, dynamic macro resolution, spatial‑context LLM prompts |

All features from previous phases remain available and work as before.

## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR: `sudo apt install tesseract-ocr`
- On X11: `python-xlib` is installed automatically via pip
- tkinter: `sudo apt install python3-tk` (usually pre‑installed;
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

### Command‑Line Modes

```bash
# Single macro execution
python main.py --macro test --ready-delay 3

# Continuous decision loop
python main.py --loop

# Use a per‑game profile from ~/.gameai/profiles/<name>/
python main.py --loop --profile <name>
```

### Live Demos (no game window needed)

| Demo | Command | What It Shows |
|------|---------|--------------|
| Phase 4 (vision) | `python tests/live_demo_phase4.py` | YOLO overlay, dynamic clicks, MCP, summariser |
| Phase 3 (memory) | `python tests/live_demo_phase3.py` | MCP memory, LLM decisions, summariser |
| Phase 2 (perception) | `python tests/live_demo.py` | Health bars, OCR, LLM pipeline |

Each demo opens an interactive OpenCV window with on‑screen controls;
press `q` or `Esc` to quit.

## Running Tests

```bash
source venv/bin/activate

# All tests
pytest tests/ -v

# Phase 5 tests only (208 tests)
python -m pytest tests/test_phase5_*.py -v
```

Tests marked `requires_ollama` will automatically skip if Ollama isn't
running.

## Configuration Files

All JSON files live under `config/` (bundled defaults) or
`~/.gameai/profiles/<name>/` (per‑game overrides).  The GUI Settings
panel writes directly to `config.json` at runtime.

| File | Purpose |
|------|---------|
| `config.json` | Global settings (Ollama URL, model, vision, diff, cache TTLs) |
| `regions.json` | Screen region definitions with `role` → state slot mapping |
| `state_schema.json` | Declares known state slots and their types/priorities |
| `macros.json` | Named macro definitions the LLM can choose from |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | Packages not installed or venv not activated | `source venv/bin/activate && pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'pynput'` | evdev failed to build | If on X11: `pip install --no-deps pynput` |
| Capture returns `None` | No window found or Wayland without `capture_region` | On Wayland, set `capture_region` in config. On X11, ensure a game window is active. |
| Macro does nothing or wrong keys | Input backend mismatch or evdev missing | Check the log for `Input backend active: ...`. On Wayland, ensure ydotoold is running. |
| Window auto‑focus fails | "Game Window" is a placeholder | Real window titles come from game profiles created with the calibration UI (`--gui`). |
| Ollama warning or LLM fallback to WAIT | Ollama not running or model not pulled | Start `ollama serve`, then `ollama pull <model>`. The agent gracefully falls back to the previous action or WAIT. |
| Tesseract error / OCR unavailable | `tesseract-ocr` not installed | `sudo apt install tesseract-ocr` |
| Live demo window doesn't appear | No X11 display or headless environment | The live demo requires a GUI display (uses OpenCV `cv2.imshow`). |
| GUI fails to start (`_tkinter` not found) | `python3-tk` system package missing | `sudo apt install python3-tk` |
| `ModuleNotFoundError: No module named 'tkinter'` | venv can't see system tkinter | Set `include-system-site-packages = true` in `venv/pyvenv.cfg` |

## AI Usage Disclosure

This project was almost entirely coded by AI models (Deepseek V4 Pro,
with Embedding Models assistance), with extensive human guidance,
project management, and testing.