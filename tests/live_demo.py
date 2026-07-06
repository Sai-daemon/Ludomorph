#!/usr/bin/env python3
"""
Phase 2 Live Demo — Visual proof that bar detection, OCR, state processing,
and LLM decision work end‑to‑end on a simulated game window.

Usage::

    cd AI-Game-Master
    source venv/bin/activate
    python tests/live_demo.py

Controls
--------
  ``1`` — Set health to 78 % (healthy)
  ``2`` — Set health to 15 % (critical)
  ``3`` — Set health to  5 % (near‑death)
  ``h`` — Toggle health bar visibility
  ``o`` — Toggle OCR rendering
  ``q`` / ``Esc`` — Quit

What you see on screen
----------------------
  - Red health bar + blue mana bar (HSV‑calibrated for regions.json)
  - OCR text regions with rendered text
  - **Top‑left overlay**: detected health %, mana %, OCR strings,
    state hash (first 12 chars)
  - **Bottom bar**: LLM‑chosen action (if Ollama is reachable), or
    "LLM unavailable"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logging_config import setup_logging

setup_logging(log_level="WARNING")  # keep stdout clean

# ---------------------------------------------------------------------------
# Configuration — pulled from config/ and tests/test_profile/
# ---------------------------------------------------------------------------

_PROFILE_DIR = _PROJECT_ROOT / "tests" / "test_profile"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


async def _try_load_model(config: dict) -> bool:
    """Return True if the configured Ollama model responds within 2 s."""
    import httpx

    url = config.get("ollama_url", "http://localhost:11434/v1")
    base = url.rstrip("/v1").rstrip("/")
    model = config.get("ollama_model", "")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.post(f"{base}/api/chat", json={
                "model": model,
                "messages": [{"role": "user", "content": "Say YES"}],
                "stream": False,
            })
            return r.status_code == 200 and "YES" in r.text.upper()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the live demo loop."""

    # ---- Load profile ----
    regions_data = _load_json(_PROFILE_DIR / "regions.json")
    schema_data = _load_json(_PROFILE_DIR / "state_schema.json")
    macros_data = _load_json(_PROFILE_DIR / "macros.json")
    global_config = _load_json(_PROJECT_ROOT / "config" / "config.json")

    from src.game_state import StateSchema
    from src.region_profile import RegionProfile
    from src.ocr_module import OCRConfig, OCRModule
    from src.state_processor import StateProcessor
    from src.llm_prompt_builder import build_llm_prompt
    from src.llm_decision import call_llm_decision

    schema = StateSchema.from_dict(schema_data)
    profile = RegionProfile.from_dict(regions_data)
    ocr = OCRModule(OCRConfig(cache_ttl_seconds=2.0))
    processor = StateProcessor(
        profile=profile,
        ocr_module=ocr,
        schema=schema,
        vision_processor=None,
        cache_ttl=0.3,
    )

    profile_macros = macros_data.get("macros", [])

    # ---- Try Ollama ----
    ollama_ok = await _try_load_model(global_config)
    if ollama_ok:
        print("✅ Ollama reachable — LLM decisions will be live.")
    else:
        print("⚠️  Ollama unreachable — LLM decisions will be skipped.")

    # ---- Frame generator ----
    from tests.frame_generator import create_simulated_frame

    health = 78.0
    mana = 92.0

    # ---- UI state ----
    show_health_bar = True
    show_ocr = True
    last_action = "N/A"
    last_llm_ms = 0.0

    cv2.namedWindow("Game AI Player — Live Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Game AI Player — Live Demo", 600, 480)

    print("\n🎮 Live demo started — Controls: 1=78%HP  2=15%HP  3=5%HP  h=toggle bar  o=toggle OCR  q=quit\n")

    while True:
        # ---- Generate synthetic frame ----
        if not show_health_bar:
            hp_text_render = ""
        else:
            hp_text_render = f"{int(health)}/100"

        if not show_ocr:
            loc_text = ""
            obj_text = ""
        else:
            loc_text = "Training Grounds"
            obj_text = "Defeat the dummy"

        frame = create_simulated_frame(
            health_pct=health if show_health_bar else 0.0,
            mana_pct=mana if show_health_bar else 0.0,
            health_text=hp_text_render,
            mana_text=f"{int(mana)}/100",
            location_text=loc_text,
            objective_text=obj_text,
        )

        # ---- Process through pipeline ----
        state = await processor.process(frame, skip_ocr=not show_ocr)

        detected_health = state.get("health")
        detected_mana = state.get("mana")
        state_hash = state.get("_state_hash", "???")[:12]

        # ---- Optional LLM call ----
        if ollama_ok:
            messages = build_llm_prompt(
                state=state,
                available_macros=profile_macros,
                memories=[],
            )
            import time

            t0 = time.monotonic()
            action = await call_llm_decision(
                messages=messages,
                profile_macros=profile_macros,
                config=global_config,
                last_action=last_action if last_action != "N/A" else None,
                timeout=2.0,
            )
            last_llm_ms = (time.monotonic() - t0) * 1000
            last_action = action

        # ---- Build overlay ----
        overlay_lines = [
            f"health: {detected_health:.1f}%   mana: {detected_mana:.1f}%",
            f"hash: {state_hash}",
        ]
        if show_ocr:
            hp_text = state.get("health_text", "")
            loc = state.get("location", "")
            overlay_lines.append(f'OCR HP: "{hp_text}"  LOC: "{loc}"')

        if ollama_ok:
            overlay_lines.append(
                f"LLM → {last_action}  ({last_llm_ms:.0f} ms)"
            )
        else:
            overlay_lines.append("LLM: skipped (unreachable)")

        overlay_lines.append("[1]=78%  [2]=15%  [3]=5%  [h]=bar  [o]=OCR  [q]=quit")

        # ---- Draw overlay on frame ----
        display = frame.copy()
        y0 = 15
        for i, line in enumerate(overlay_lines):
            cv2.putText(
                display,
                line,
                (5, y0 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Game AI Player — Live Demo", display)

        # ---- Keyboard handling ----
        key = cv2.waitKey(33) & 0xFF  # ~30 fps

        if key == ord("q") or key == 27:  # q or Esc
            break
        elif key == ord("1"):
            health = 78.0
            processor._cache.clear()
            print(f"[Key 1] Health set to {health}%")
        elif key == ord("2"):
            health = 15.0
            processor._cache.clear()
            print(f"[Key 2] Health set to {health}%")
        elif key == ord("3"):
            health = 5.0
            processor._cache.clear()
            print(f"[Key 3] Health set to {health}%")
        elif key == ord("h"):
            show_health_bar = not show_health_bar
            print(f"[Key h] Health bar: {'ON' if show_health_bar else 'OFF'}")
        elif key == ord("o"):
            show_ocr = not show_ocr
            print(f"[Key o] OCR rendering: {'ON' if show_ocr else 'OFF'}")

    cv2.destroyAllWindows()
    print("\n👋 Live demo ended.")


if __name__ == "__main__":
    asyncio.run(main())