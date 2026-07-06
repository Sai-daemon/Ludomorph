# AI-Game-Master

Have a local LLM play your game, in realish time.

Currently a work in progress.

Version 0.2.0

## Phase 2 Complete

### What This Version Does

Phase 2 adds AI perception and decision-making on top of the Phase 1
plumbing. The agent can now "see" health bars and on-screen text, build a
game-state summary, ask a local LLM what to do next, and execute the chosen
macro — all in a continuous async loop.

**New in 0.2.0:**

- **Colour bar detection** — reads health, mana, and similar status bars
  from the screen using HSV thresholding and projection analysis.
- **OCR text reading** — extracts location names, objective text, and
  numeric readouts via Tesseract with preprocessing and caching.
- **Game state model** — typed `GameState` container with a user-extensible
  JSON schema (`state_schema.json`).
- **Region profiles** — screen regions mapped to state slots via
  `regions.json` with `color_bar` or `ocr` types and a `role` field.
- **State hashing & caching** — salted SHA-256 hash of game state with a
  0.3 s TTL, avoiding duplicate LLM calls on static frames.
- **LLM prompt builder & decision call** — assembles a token-budgeted
  prompt, sends it to Ollama's `/api/chat`, parses the JSON response,
  and falls back gracefully on timeouts.
- **Adaptive frame skipping** — pixel-diff detector that skips processing
  when the screen is static, saving CPU and LLM calls.
- **Decision loop** — wires everything together: capture → skip? →
  process state → cache? → LLM → execute macro, with latency monitoring
  and adaptive throttling.
- **Integration tests** — synthetic game frames with colour bars and OCR
  text validate the full pipeline end‑to‑end.
- **Live demo** — an interactive OpenCV window where you can change the
  simulated health bar and watch the agent respond in real time.

**Phase 1 features** (screen capture, input injection, window auto-focus,
static macro playback, Ollama health check) are all still available and
work as before.


## Installation

### Prerequisites

- Python 3.12+
- Tesseract OCR: `sudo apt install tesseract-ocr`
- On X11: `python-xlib` is installed automatically via pip

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

### Phase 1 — single macro execution

```bash
source venv/bin/activate
python main.py --macro test --ready-delay 3
```

### Phase 2 — continuous decision loop

```bash
source venv/bin/activate
python main.py --loop
```

This runs the full perception → LLM → action loop using the bundled
`config/` files (regions, state schema, macros).  Press **Ctrl+C** to stop.

Use `--profile <name>` to load profile files from
`~/.gameai/profiles/<name>/` instead.

### Live Demo (no game window needed)

The live demo creates a synthetic game frame with colour bars and OCR text
so you can see the entire Phase 2 pipeline in action without launching a game:

```bash
source venv/bin/activate
python tests/live_demo.py
```

Controls:

| Key | Action |
|-----|--------|
| `1` | Set health to 78 % (healthy) |
| `2` | Set health to 15 % (critical) |
| `3` | Set health to 5 % (near-death) |
| `h` | Toggle health bar visibility |
| `o` | Toggle OCR text rendering |
| `q` / `Esc` | Quit |

The window shows detected health/mana percentages, OCR text, the state
hash, the LLM-chosen action, and the LLM response time.  If Ollama is
unreachable a warning is shown and LLM calls are skipped.


## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

Tests marked `requires_ollama` will automatically skip if Ollama isn't
running.


## Configuration Files

All JSON files live under `config/` (bundled defaults) or
`~/.gameai/profiles/<name>/` (per-game overrides).

| File | Purpose |
|------|---------|
| `config.json` | Global settings (Ollama URL, model, diff, cache TTLs) |
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
| Window auto-focus fails | "Game Window" is a placeholder | Expected in Phase 1/2. Real window titles come from game profiles (Phase 5 calibration UI). |
| Ollama warning or LLM fallback to WAIT | Ollama not running or model not pulled | Start `ollama serve`, then `ollama pull <model>`. The agent gracefully falls back to the previous action or WAIT. |
| Tesseract error / OCR unavailable | `tesseract-ocr` not installed | `sudo apt install tesseract-ocr` |
| Live demo window doesn't appear | No X11 display or headless environment | The live demo requires a GUI display (uses OpenCV `cv2.imshow`). |


## AI Usage Disclosure

This project was almost entirely coded by AI models (Deepseek V4 Pro, with Embedding Models assistance), with extensive human guidance, project management, and testing.