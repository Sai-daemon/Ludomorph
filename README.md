# AI-Game-Master
 Have a local LLM play your game, in realish time. 

Currently a work in progress. 

Version 0.1

## Phase 1 Complete
 What This Version Does

Phase 1 provides the core plumbing: screen capture, keyboard/mouse input injection,
window management, static macro playback from a JSON file, and a health check against
the Ollama LLM server. There is no AI decision-making yet (that's Phase 2).


## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR (for Phase 2+): `sudo apt install tesseract-ocr`
- On X11: `python-xlib` is installed automatically via pip

### Setup

```bash
cd AI-Game-Master
python -m venv venv
source venv/bin/activate
pip install httpx mss pywinctl pynput pytesseract opencv-python loguru appdirs portend tenacity
```

### Wayland (optional — not needed for Phase 1 on X11)

The `python-ydotool` package requires `evdev`, which needs the Python development
headers to compile:

```bash
sudo apt install python3-dev
pip install python-ydotool
```

If you are on X11 or Windows, `pynput` is sufficient and you can skip this step.

### Ollama (for Phase 2)

If you don't have Ollama installed, the health check will log a warning — that's
acceptable for Phase 1. When you're ready for Phase 2:

```bash
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

Or change `ollama_model` in `config.json` to any other model you prefer.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'httpx'` | Packages not installed or venv not activated | Run `source venv/bin/activate && pip list` to verify |
| `ModuleNotFoundError: No module named 'pynput'` | Same as above, or evdev failed to build | If on X11: `pip install --no-deps pynput` (evdev isn't needed for X11) |
| Capture returns None | No window found or Wayland without capture_region | On Wayland, set `capture_region` in config. On X11, make sure a window is active. |
| Macro does nothing or wrong keys | Input backend mismatch or evdev missing | Check the log for `Input backend active: ...`. On Wayland, ensure ydotoold is running. |
| Window auto-focus fails | "Game Window" is a placeholder | Expected in Phase 1. Real window titles come from game profiles in Phase 2. |
| Ollama warning | Ollama not running or model not pulled | Start `ollama serve`, then `ollama pull <model>`. Non-fatal for Phase 1. |

## AI Usage Disclosure
This project was almost entirely coded by Deepseek V4 Pro, with extensive human guidance, project management, and testing. 