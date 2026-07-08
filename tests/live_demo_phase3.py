#!/usr/bin/env python3
"""
Phase 3 Live Demo — Full memory pipeline with MCP server, Ollama, and
summarisation running on a simulated game window.

Usage::

    cd AI-Game-Master
    source venv/bin/activate
    python tests/live_demo_phase3.py

What's demonstrated
-------------------
* MCP memory server auto‑start (bundled mcp-memory-service, non‑blocking)
* Ollama model auto‑pull if not cached
* Memory storage & semantic retrieval on every significant state change
* MemorySummariser background task compresses short‑term → medium‑term
* LLM decisions influenced by past memory context
* Real‑time MCP operation log in the overlay
* Fast rendering — LLM only re‑evaluates on health change ≥10 pp
* Two‑panel layout with dark background rectangles for readability

Controls
--------
  ``1`` — Set health to 78 % (healthy)
  ``2`` — Set health to 15 % (critical)
  ``3`` — Set health to  5 % (near‑death)
  ``s`` — Force‑store current state+action as memory
  ``m`` — Toggle MemorySummariser on/off
  ``t`` — Trigger an immediate summarisation cycle
  ``r`` — Reset all demo memories (delete + re‑seed)
  ``q`` / ``Esc`` — Quit
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — ensure project root and mcp-memory-service are importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MCP_SERVICE_SRC = _PROJECT_ROOT.parent / "mcp-memory-service-subroutine" / "src"
if str(_MCP_SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVICE_SRC))
# Also set PYTHONPATH for the MCP subprocess spawned by MCPServerManager
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = (
        str(_MCP_SERVICE_SRC) + os.pathsep + os.environ["PYTHONPATH"]
    )
else:
    os.environ["PYTHONPATH"] = str(_MCP_SERVICE_SRC)

from src.logging_config import setup_logging, get_logger

setup_logging(log_level="DEBUG")
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROFILE_DIR = _PROJECT_ROOT / "tests" / "test_profile"
_CONFIG_DIR = _PROJECT_ROOT / "config"

# Pre‑seeded memories so the demo shows memory recall immediately
_SEED_MEMORIES: list[str] = [
    "Player explored Forest with 100% health and full mana.",
    "Goblins attacked near the bridge — health dropped to 28%. Player drank a potion.",
    "Player entered Dark Cave with 85% health.",
    "Ambushed by a cave troll — health dropped to 12%.",
    "Player drank another potion — health restored to 88%.",
    "Player found a treasure chest and returned to town with 95% health.",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


async def _get_ollama_models(base_url: str) -> list[str]:
    """Return list of available model names from Ollama /api/tags."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.status_code == 200:
                models = r.json().get("models", [])
                return [m.get("name", "") for m in models]
    except Exception:
        pass
    return []


async def _ollama_pull_model(base_url: str, model: str) -> tuple[bool, str]:
    """Pull an Ollama model; returns (success, message)."""
    import subprocess

    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "pull", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300.0)
        output = stdout.decode("utf-8", errors="replace")
        if proc.returncode == 0:
            return True, "pull complete"
        return False, output[-200:].strip()
    except asyncio.TimeoutError:
        return False, "pull timed out after 5min"
    except FileNotFoundError:
        return False, "ollama CLI not found"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the Phase 3 live demo."""

    print("\n╔══════════════════════════════════════════════╗")
    print("║   AI Game Player — Phase 3 Memory Demo      ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ---- Load profile --------------------------------------------------
    regions_data = _load_json(_PROFILE_DIR / "regions.json")
    schema_data = _load_json(_PROFILE_DIR / "state_schema.json")
    macros_data = _load_json(_PROFILE_DIR / "macros.json")
    global_config = _load_json(_CONFIG_DIR / "config.json")

    from src.game_state import StateSchema
    from src.region_profile import RegionProfile
    from src.ocr_module import OCRConfig, OCRModule
    from src.state_processor import StateProcessor
    from src.llm_prompt_builder import build_llm_prompt
    from src.llm_decision import call_llm_decision
    from src.decision_loop import build_memory_query

    schema = StateSchema.from_dict(schema_data)
    profile = RegionProfile.from_dict(regions_data)
    ocr = OCRModule(OCRConfig(cache_ttl_seconds=2.0))
    processor = StateProcessor(
        profile=profile, ocr_module=ocr, schema=schema,
        vision_processor=None, cache_ttl=0.3,
    )
    profile_macros = macros_data.get("macros", [])
    valid_names = {m["name"] for m in profile_macros}

    # ---- Shared state --------------------------------------------------
    mcp_manager = None
    mcp_client = None
    summariser = None
    mcp_online = False
    mcp_status_text = "starting"
    ollama_ok = False
    ollama_status_text = "checking..."
    mcp_log: list[str] = []

    def _log_mcp(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        mcp_log.append(f"[{ts}] {msg}")
        if len(mcp_log) > 8:
            mcp_log.pop(0)

    # ---- Pre‑warm ONNX model cache (downloads ~100 MB if first run) ------
    _log_mcp("Pre‑warming ONNX embedding model cache...")
    try:
        from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel
        # This downloads the model to ~/.cache/mcp_memory/onnx_models
        # synchronously.  The MCP server subprocess won't need to
        # download it again.
        _model = ONNXEmbeddingModel()
        _log_mcp("ONNX model cache ready")
    except Exception as exc:
        _log_mcp(f"ONNX pre‑warm skipped: {exc}")

    # ---- Start MCP server in background (non‑blocking) -----------------
    _log_mcp("MCP server starting in background...")
    mcp_status_text = "⏳ starting (model download may take ~20s on first run)"

    async def _mcp_startup() -> None:
        nonlocal mcp_online, mcp_status_text, mcp_manager, mcp_client, summariser
        try:
            from src.mcp_server import MCPServerManager

            mcp_manager = MCPServerManager(
                port=8000, ready_timeout=120.0, grace_period=3.0,
            )
            await mcp_manager.start()
            mcp_online = True
            mcp_status_text = "● ONLINE"
            _log_mcp("MCP server healthy ✓")
            print("✅ MCP memory server running on port 8000")

            # Create client
            from src.mcp_client import MCPMemoryClient

            mcp_client = MCPMemoryClient(base_url="http://localhost:8000")

            # Pre‑seed
            for mem_text in _SEED_MEMORIES:
                try:
                    await mcp_client.store_memory(
                        content=mem_text,
                        memory_type="short_term",
                        tags=["demo", "game_event", "short_term"],
                    )
                except Exception:
                    pass
            await asyncio.sleep(0.3)
            _log_mcp(f"Seeded {len(_SEED_MEMORIES)} demo memories")

            # Start summariser
            summarisation_model = global_config.get(
                "summarization_model", global_config.get("ollama_model", "phi3.5")
            )
            from src.memory_summariser import MemorySummariser

            summariser = MemorySummariser(
                client=mcp_client,
                ollama_url=global_config.get(
                    "ollama_url", "http://localhost:11434/v1"
                ),
                model=summarisation_model,
                event_threshold=3,
                time_interval=60.0,
            )
            await summariser.start()
            _log_mcp("Summariser started (threshold=3 events)")

        except Exception as exc:
            mcp_status_text = f"○ OFFLINE ({str(exc)[:60]})"
            _log_mcp(f"MCP failed: {exc}")
            print(f"⚠️  MCP server unavailable: {exc}")
            print("   Memory tier disabled — demo runs without persistence.")

    mcp_task = asyncio.create_task(_mcp_startup())

    # ---- Check / pull Ollama model -------------------------------------
    ollama_url = global_config.get("ollama_url", "http://localhost:11434/v1")
    ollama_base = ollama_url.rstrip("/v1").rstrip("/")
    ollama_model = global_config.get("ollama_model", "phi3.5:mini")

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ollama_base}/api/tags")
            if r.status_code == 200:
                available = await _get_ollama_models(ollama_base)

                # Try partial match (e.g. "phi3.5:mini" matches "phi3.5:3.8b-mini-instruct-q4_K_M")
                model_found = any(
                    ollama_model in name or name.startswith(ollama_model.split(":")[0])
                    for name in available
                )

                if model_found:
                    ollama_ok = True
                    # Show the actual detected model name, not the config
                    # abbreviation (e.g. "phi3.5:3.8b-mini-instruct-q4_K_M"
                    # instead of "phi3.5:mini").
                    matched = next(
                        (n for n in available
                         if ollama_model in n or n.startswith(ollama_model.split(":")[0])),
                        ollama_model,
                    )
                    ollama_status_text = f"● ONLINE ({matched})"
                    print(f"✅ Ollama model '{matched}' available")
                else:
                    ollama_status_text = f"⏳ pulling {ollama_model}..."
                    print(f"⏳ Model '{ollama_model}' not cached — pulling...")
                    _log_mcp(f"Pulling model: {ollama_model}")
                    ok, msg = await _ollama_pull_model(ollama_base, ollama_model)
                    if ok:
                        ollama_ok = True
                        ollama_status_text = f"● ONLINE ({ollama_model})"
                        print(f"✅ Model '{ollama_model}' pulled successfully")
                        _log_mcp("Model pull complete")
                    else:
                        ollama_status_text = f"pull failed: {msg[:50]}"
                        print(f"⚠️  Model pull failed: {msg}")
                        _log_mcp(f"Model pull failed: {msg[:80]}")
            else:
                ollama_status_text = f"○ offline ({r.status_code})"
                print(f"⚠️  Ollama unreachable (status {r.status_code})")
    except Exception as exc:
        ollama_status_text = f"○ offline ({str(exc)[:40]})"
        print(f"⚠️  Ollama unreachable: {exc}")

    # ---- State variables ------------------------------------------------
    health = 78.0
    mana = 92.0
    last_processed_health = -999.0
    latest_state: Any = None
    latest_action: str = "N/A"
    latest_llm_ms: float = 0.0
    latest_memories: list[str] = []
    memory_count: int = 0
    stored_count: int = 0
    processor_lock = asyncio.Lock()
    state_changed = asyncio.Event()
    state_changed.set()

    # ---- Processing pipeline task (runs when state changes) ------------
    async def process_pipeline() -> None:
        nonlocal latest_state, latest_action, latest_llm_ms
        nonlocal latest_memories, memory_count, stored_count

        while True:
            await state_changed.wait()
            state_changed.clear()

            async with processor_lock:
                try:
                    frame = create_simulated_frame(
                        health_pct=health,
                        mana_pct=mana,
                        health_text=f"{int(health)}/100",
                        mana_text=f"{int(mana)}/100",
                        location_text="Training Grounds",
                        objective_text="Defeat the dummy",
                    )
                    state = await processor.process(frame, skip_ocr=True)
                    latest_state = state

                    # MCP memory search
                    if mcp_online and mcp_client is not None:
                        try:
                            query = build_memory_query(state)
                            results = await mcp_client.search_memories(
                                query=query, tags=["demo"], limit=5,
                            )
                            latest_memories = [
                                r.get("content", "")[:150] for r in results
                            ]
                            memory_count = len(results)
                            _log_mcp(
                                f"Search: {len(results)} results for "
                                f"'{query[:50]}...'"
                            )
                        except Exception as exc:
                            _log_mcp(f"Search error: {exc}")
                            latest_memories = []
                            memory_count = 0
                    else:
                        if not mcp_online:
                            latest_memories = []

                    # LLM decision
                    if ollama_ok:
                        memories_for_prompt = (
                            latest_memories[:5] if latest_memories else []
                        )
                        messages = build_llm_prompt(
                            state=state,
                            available_macros=profile_macros,
                            memories=memories_for_prompt,
                            max_tokens=800,
                        )
                        t0 = time.monotonic()
                        action = await call_llm_decision(
                            messages=messages,
                            profile_macros=profile_macros,
                            config=global_config,
                            last_action=(
                                latest_action
                                if latest_action not in ("N/A", "WAIT (LLM offline)")
                                else None
                            ),
                            timeout=5.0,
                        )
                        latest_llm_ms = (time.monotonic() - t0) * 1000
                        if action not in valid_names:
                            action = "WAIT"
                        latest_action = action
                    else:
                        latest_action = "WAIT (LLM offline)"

                    # MCP memory store (fire-and-forget)
                    if mcp_online and mcp_client is not None:

                        async def _store() -> None:
                            nonlocal stored_count
                            try:
                                state_dict = (
                                    state.to_dict()
                                    if hasattr(state, "to_dict")
                                    else {}
                                )
                                await mcp_client.store_memory(
                                    content=json.dumps({
                                        "state": {
                                            k: v
                                            for k, v in state_dict.items()
                                            if not k.startswith("_")
                                        },
                                        "action": latest_action,
                                        "health_pct": health,
                                    }),
                                    memory_type="short_term",
                                    tags=["demo", "game_event", "short_term"],
                                )
                                stored_count += 1
                                if summariser is not None:
                                    await summariser.record_new_event()
                                _log_mcp(
                                    f"Store: hp={health:.0f}% "
                                    f"→ {latest_action} (total={stored_count})"
                                )
                            except Exception as exc:
                                _log_mcp(f"Store error: {exc}")

                        asyncio.create_task(_store())

                except Exception as exc:
                    # Log to both overlay AND terminal with full traceback
                    import traceback
                    tb = traceback.format_exc()
                    logger.error("Pipeline error:\n{}", tb)
                    _log_mcp(f"Pipeline error: {exc}")

    pipeline_task = asyncio.create_task(process_pipeline())

    # ---- Frame generator ------------------------------------------------
    from tests.frame_generator import create_simulated_frame

    # ---- OpenCV window --------------------------------------------------
    WINDOW_W, WINDOW_H = 750, 680
    cv2.namedWindow("Game AI Player — Phase 3 Memory Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Game AI Player — Phase 3 Memory Demo", WINDOW_W, WINDOW_H)

    print(
        "\n🎮 Live demo started — "
        "[1]=78%HP  [2]=15%HP  [3]=5%HP  "
        "[s]=store  [m]=summariser  [t]=trigger  [r]=reset  [q]=quit\n"
    )

    render_count = 0
    fps_update_time = time.monotonic()
    fps_value = 0.0

    try:
        while True:
            # ---- Detect health change → trigger pipeline ----------------
            if abs(health - last_processed_health) >= 10.0:
                last_processed_health = health
                state_changed.set()

            # ---- Generate frame with current visual state ----------------
            frame = create_simulated_frame(
                health_pct=health,
                mana_pct=mana,
                health_text=f"{int(health)}/100",
                mana_text=f"{int(mana)}/100",
                location_text="Training Grounds",
                objective_text="Defeat the dummy",
            )

            # Create a blank canvas and place frame + overlay side-by-side
            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)

            # Game frame goes on the left (resized to ~360x270)
            game_area_w = 370
            game_area_h = 280
            if frame is not None:
                resized = cv2.resize(frame, (game_area_w, game_area_h))
                canvas[10 : 10 + game_area_h, 10 : 10 + game_area_w] = resized

            # ---- Overlay panel (right side) -------------------------------
            panel_x = game_area_w + 25
            y = 12
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 0.7
            fg_color = (200, 230, 200)
            dim_color = (120, 140, 120)
            hl_color = (100, 255, 150)
            warn_color = (100, 180, 255)
            mem_color = (200, 200, 100)
            th = cv2.FONT_HERSHEY_PLAIN
            th_scale = 0.55

            # Helper: draw a dark background rect + text
            def _put_section(
                lines: list[tuple[str, tuple[int, int, int]]],
                bg: tuple[int, int, int] = (25, 30, 25),
            ) -> None:
                nonlocal y
                if not lines:
                    return
                line_h = 13
                pad = 4
                max_w = 0
                for txt, _col in lines:
                    (tw, _), _ = cv2.getTextSize(txt, font, font_scale, 1)
                    max_w = max(max_w, tw)
                rect_x1 = panel_x - 3
                rect_y1 = y - line_h + 2 - pad
                rect_x2 = panel_x + max_w + 8
                rect_y2 = y + pad
                cv2.rectangle(canvas, (rect_x1, rect_y1), (rect_x2, rect_y2), bg, -1)
                for txt, col in lines:
                    cv2.putText(
                        canvas, txt, (panel_x, y), font, font_scale, col, 1,
                        cv2.LINE_4,
                    )
                    y += line_h

            # --- Section 1: Health + MCP status ---
            detected_health = (
                latest_state.get("health") if latest_state is not None else None
            )
            state_hash = (
                latest_state.get("_state_hash", "???")[:12]
                if latest_state is not None
                else "???"
            )
            hp_str = (
                f"{detected_health:.0f}%"
                if detected_health is not None
                else "?"
            )
            _put_section(
                [
                    (f"health: {hp_str}   mana: {mana:.0f}%   hash: {state_hash}", hl_color),
                    (f"MCP: {mcp_status_text}", fg_color if mcp_online else warn_color),
                    (f"Ollama: {ollama_status_text}", fg_color if ollama_ok else warn_color),
                    (f"Memories: search={memory_count}  stored={stored_count}", fg_color),
                ],
                bg=(20, 28, 20),
            )
            y += 2

            # --- Section 2: LLM decision ---
            llm_color = fg_color if ollama_ok else dim_color
            llm_text = (
                f"LLM → {latest_action}  ({latest_llm_ms:.0f}ms)"
                if ollama_ok
                else "LLM: offline"
            )
            _put_section([(llm_text, llm_color)], bg=(20, 20, 28))
            y += 4

            # --- Section 3: Memory search results ---
            if latest_memories:
                mem_lines: list[tuple[str, tuple[int, int, int]]] = [
                    ("Recent memories:", mem_color)
                ]
                for mem in latest_memories[:3]:
                    truncated = mem[:75] + "..." if len(mem) > 75 else mem
                    mem_lines.append((f"  {truncated}", dim_color))
                _put_section(mem_lines, bg=(28, 26, 18))
                y += 2
            elif mcp_online:
                _put_section([("Recent memories: (none yet)", dim_color)], bg=(28, 26, 18))
                y += 2
            else:
                _put_section([("Memories: waiting for MCP...", dim_color)], bg=(28, 26, 18))
                y += 2

            # --- Section 4: Summariser status ---
            if summariser is not None:
                task_running = (
                    summariser._task is not None
                    and not summariser._task.done()
                )
                _put_section(
                    [
                        (
                            f"Summariser: {'running' if task_running else 'stopped'}  "
                            f"(events={summariser._event_count}/{summariser.threshold})",
                            fg_color,
                        )
                    ],
                    bg=(20, 28, 28),
                )
                y += 2

            # --- Section 5: MCP operation log ---
            log_lines: list[tuple[str, tuple[int, int, int]]] = [
                ("MCP Log:", (140, 140, 140))
            ]
            for log_line in mcp_log[-6:]:
                log_lines.append((f"  {log_line[:200]}", (100, 100, 100)))
            _put_section(log_lines, bg=(18, 18, 18))
            y += 2

            # --- Section 6: FPS ---
            _put_section(
                [(f"FPS: {fps_value:.0f}", dim_color)], bg=(20, 20, 20)
            )
            y += 4

            # --- Bottom controls bar ---
            ctrl_text = (
                "[1]=78%  [2]=15%  [3]=5%  [s]=store  "
                "[m]=summ  [t]=trig  [r]=reset  [q]=quit"
            )
            cv2.rectangle(
                canvas,
                (0, WINDOW_H - 22),
                (WINDOW_W, WINDOW_H),
                (15, 15, 20),
                -1,
            )
            cv2.putText(
                canvas,
                ctrl_text,
                (8, WINDOW_H - 6),
                font,
                0.62,
                (140, 140, 160),
                1,
                cv2.LINE_4,
            )

            # ---- Display -------------------------------------------------
            cv2.imshow("Game AI Player — Phase 3 Memory Demo", canvas)

            # ---- FPS counter ---------------------------------------------
            render_count += 1
            now = time.monotonic()
            if now - fps_update_time >= 1.0:
                fps_value = render_count / (now - fps_update_time)
                render_count = 0
                fps_update_time = now

            # ---- Keyboard ------------------------------------------------
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord("1"):
                health = 78.0
                processor._cache.clear()
                print(f"  Health → {health:.0f}%")
            elif key == ord("2"):
                health = 15.0
                processor._cache.clear()
                print(f"  Health → {health:.0f}%")
            elif key == ord("3"):
                health = 5.0
                processor._cache.clear()
                print(f"  Health → {health:.0f}%")
            elif key == ord("s"):
                if mcp_online and mcp_client is not None and latest_state is not None:
                    try:
                        state_dict = (
                            latest_state.to_dict()
                            if hasattr(latest_state, "to_dict")
                            else {}
                        )
                        await mcp_client.store_memory(
                            content=json.dumps({
                                "state": {
                                    k: v
                                    for k, v in state_dict.items()
                                    if not k.startswith("_")
                                },
                                "action": latest_action,
                                "health_pct": health,
                                "manual": True,
                            }),
                            memory_type="short_term",
                            tags=["demo", "game_event", "short_term", "manual"],
                        )
                        stored_count += 1
                        _log_mcp(
                            f"Manual store: {health:.0f}% → {latest_action}"
                        )
                        print(f"  📝 Manual memory stored (total: {stored_count})")
                    except Exception as exc:
                        print(f"  ❌ Store failed: {exc}")
                else:
                    print("  ⚠️  MCP offline — cannot store")
            elif key == ord("m"):
                if summariser is not None:
                    if (
                        summariser._task is not None
                        and not summariser._task.done()
                    ):
                        await summariser.stop()
                        _log_mcp("Summariser stopped")
                        print("  ⏸️  Summariser stopped")
                    else:
                        await summariser.start()
                        _log_mcp("Summariser started")
                        print("  ▶️  Summariser started")
            elif key == ord("t"):
                if summariser is not None and mcp_client is not None:
                    for _ in range(summariser.threshold + 1):
                        await summariser.record_new_event()
                    _log_mcp("Manual summarisation trigger")
                    print("  🔄 Summarisation triggered — check MCP log")
            elif key == ord("r"):
                if mcp_online and mcp_client is not None:
                    try:
                        results = await mcp_client.search_memories(
                            query="game", tags=["demo"], limit=100,
                        )
                        deleted = 0
                        for r in results:
                            ch = r.get("content_hash") or r.get(
                                "memory", {}
                            ).get("content_hash")
                            if ch:
                                try:
                                    await mcp_client.delete_memory(ch)
                                    deleted += 1
                                except Exception:
                                    pass
                        _log_mcp(f"Reset: deleted {deleted} memories")
                        for mem_text in _SEED_MEMORIES:
                            await mcp_client.store_memory(
                                content=mem_text,
                                memory_type="short_term",
                                tags=["demo", "game_event", "short_term"],
                            )
                        stored_count = len(_SEED_MEMORIES)
                        _log_mcp(
                            f"Re-seeded {len(_SEED_MEMORIES)} demo memories"
                        )
                        print(
                            f"  🔄 Reset: deleted {deleted}, "
                            f"re-seeded {len(_SEED_MEMORIES)}"
                        )
                    except Exception as exc:
                        print(f"  ❌ Reset failed: {exc}")
                else:
                    print("  ⚠️  MCP offline — cannot reset")

            await asyncio.sleep(0)

    finally:
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass

        mcp_task.cancel()
        try:
            await mcp_task
        except asyncio.CancelledError:
            pass

        if summariser is not None:
            await summariser.stop()
        if mcp_client is not None:
            await mcp_client.close()
        if mcp_manager is not None:
            await mcp_manager.stop()

        cv2.destroyAllWindows()
        print("\n👋 Phase 3 demo ended.")


if __name__ == "__main__":
    asyncio.run(main())