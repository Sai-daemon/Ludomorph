#!/usr/bin/env python3
"""
Phase 4 Live Demo — Full vision pipeline with YOLO object detection,
dynamic macro resolution, MCP memory, Ollama LLM, and summarisation
running on a simulated game window.

Usage::

    cd AI-Game-Master
    source venv/bin/activate
    python tests/live_demo_phase4.py

What's demonstrated
-------------------
- YOLO ONNX model auto‑loading (yolo11n.onnx)
- Synthetic Detection injection when YOLO can't detect drawn shapes
- Vision‑OCR scheduling with stagger intervals, latency gates, and contention tracking
- Dynamic macro resolution — LLM picks dynamic macros, MacroResolver
  converts them to concrete mouse coordinates
- Visual targeting reticles on the game frame
- MCP memory server auto‑start with ONNX embedding pre‑warming
- Ollama model auto‑pull / health check
- Memory storage & semantic retrieval on every state change
- MemorySummariser background task
- Vision‑aware LLM prompt extension with spatial context injection
- Vision status panel with real‑time detection and resolution feedback
- All errors visible in both the OpenCV overlay AND terminal

Controls
--------
  ``1`` — Set health to 78 % (healthy)
  ``2`` — Set health to 15 % (critical)
  ``3`` — Set health to  5 % (near‑death)
  ``e`` — Spawn enemy (red rect, simulated "person" class)
  ``p`` — Spawn potion (blue rect, simulated "bottle" class)
  ``c`` — Clear all spawned objects
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
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MCP_SERVICE_SRC = _PROJECT_ROOT.parent / "mcp-memory-service-subroutine" / "src"
if str(_MCP_SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVICE_SRC))
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
_MODEL_PATH = _PROJECT_ROOT / "models" / "yolo11n.onnx"

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


def _build_synthetic_detections(
    game_objects: list[dict[str, Any]],
) -> list[Any]:
    """Create synthetic Detection objects from game object definitions.

    YOLOv11n is trained on COCO photos — it cannot detect synthetic
    coloured rectangles.  This helper injects Detection objects that
    mirror what YOLO *would* output if our objects were real‑world
    photos, keeping all downstream components (SpatialContextBuilder,
    MacroResolver, LLM prompt builder) working correctly.
    """
    from src.vision_detector import Detection

    COCO_ID: dict[str, int] = {"person": 0, "bottle": 39}

    detections: list[Any] = []
    for obj in game_objects:
        label = obj.get("label", "")
        class_name = ""
        if label == "enemy":
            class_name = "person"
        elif label == "item":
            class_name = "bottle"
        else:
            continue
        x, y = obj.get("x", 0), obj.get("y", 0)
        w, h = obj.get("w", 40), obj.get("h", 40)
        cx, cy = x + w // 2, y + h // 2
        detections.append(
            Detection(
                class_id=COCO_ID.get(class_name, 0),
                class_name=class_name,
                confidence=0.92,
                bbox=(x, y, x + w, y + h),
                center=(cx, cy),
                area=w * h,
            )
        )
    return detections


# ---------------------------------------------------------------------------
# Main Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run the Phase 4 live demo."""

    print("\n╔══════════════════════════════════════════════╗")
    print("║   AI Game Player — Phase 4 Vision Demo      ║")
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

    schema = StateSchema.from_dict(schema_data)
    profile = RegionProfile.from_dict(regions_data)
    ocr = OCRModule(OCRConfig(cache_ttl_seconds=2.0))

    # ---- Load Vision module (Phase 4.1) ----------------------------------
    from src.vision_detector import VisionConfig, VisionProcessor

    vision_cfg_dict = dict(global_config.get("vision", {}))
    vision_cfg_dict["enabled"] = True
    vision_cfg_dict["model_path"] = str(_MODEL_PATH)
    vision_cfg_dict["detection_interval"] = 3
    vision_cfg_dict["input_size"] = 320  # faster inference (640px ~4x slower, exceeds 100ms timeout)
    vision_cfg = VisionConfig.from_dict(vision_cfg_dict)
    vision_enabled = _MODEL_PATH.exists()

    if vision_enabled:
        print(f"  Loading YOLO model from {_MODEL_PATH}...")
        vision_processor_obj = VisionProcessor(vision_cfg, screen_size=(500, 400))
        print(f"  YOLO ONNX loaded (input={vision_cfg.input_size}, interval={vision_cfg.detection_interval})")
    else:
        vision_processor_obj = None
        print(f"  YOLO model not found at {_MODEL_PATH} — vision disabled")

    processor = StateProcessor(
        profile=profile, ocr_module=ocr, schema=schema,
        vision_processor=vision_processor_obj, cache_ttl=0.3,
        vision_interval=vision_cfg.detection_interval if vision_enabled else 2,
    )

    profile_macros = macros_data.get("macros", [])
    valid_names = {m["name"] for m in profile_macros}

    dynamic_macro_names: set[str] = set()
    for m in profile_macros:
        for action in m.get("actions", []):
            if isinstance(action, dict) and action.get("type", "").startswith("dynamic_"):
                dynamic_macro_names.add(m["name"])
                break

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
        if len(mcp_log) > 10:
            mcp_log.pop(0)

    # ---- Pre‑warm ONNX embedding cache ----------------------------------
    _log_mcp("Pre‑warming ONNX embedding model cache...")
    try:
        from mcp_memory_service.embeddings.onnx_embeddings import ONNXEmbeddingModel
        _model = ONNXEmbeddingModel()
        _log_mcp("ONNX model cache ready")
    except Exception as exc:
        _log_mcp(f"ONNX pre‑warm skipped: {exc}")

    # ---- Start MCP server ----------------------------------------------
    _log_mcp("MCP server starting...")
    mcp_status_text = "starting (model download may take ~20s on first run)"

    async def _mcp_startup() -> None:
        nonlocal mcp_online, mcp_status_text, mcp_manager, mcp_client, summariser
        try:
            from src.mcp_server import MCPServerManager
            mcp_manager = MCPServerManager(port=8000, ready_timeout=120.0, grace_period=3.0)
            await mcp_manager.start()
            mcp_online = True
            mcp_status_text = "ONLINE"
            _log_mcp("MCP server healthy")
            print("  MCP memory server running on port 8000")

            from src.mcp_client import MCPMemoryClient
            mcp_client = MCPMemoryClient(base_url="http://localhost:8000")
            for mem_text in _SEED_MEMORIES:
                try:
                    await mcp_client.store_memory(
                        content=mem_text, memory_type="short_term",
                        tags=["demo", "game_event", "short_term"],
                    )
                except Exception:
                    pass
            await asyncio.sleep(0.3)
            _log_mcp(f"Seeded {len(_SEED_MEMORIES)} demo memories")

            summarisation_model = global_config.get(
                "summarization_model", global_config.get("ollama_model", "phi3.5")
            )
            from src.memory_summariser import MemorySummariser
            summariser = MemorySummariser(
                client=mcp_client,
                ollama_url=global_config.get("ollama_url", "http://localhost:11434/v1"),
                model=summarisation_model, event_threshold=3, time_interval=60.0,
            )
            await summariser.start()
            _log_mcp("Summariser started (threshold=3 events)")
        except Exception as exc:
            mcp_status_text = f"OFFLINE ({str(exc)[:60]})"
            _log_mcp(f"MCP failed: {exc}")
            print(f"  MCP server unavailable: {exc}")

    mcp_task = asyncio.create_task(_mcp_startup())

    # ---- Check / pull Ollama model -------------------------------------
    ollama_url = global_config.get("ollama_url", "http://localhost:11434/v1")
    ollama_base = ollama_url.rstrip("/v1").rstrip("/")
    ollama_model = global_config.get("ollama_model", "phi3.5:3.8b-mini-instruct-q4_K_M")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ollama_base}/api/tags")
            if r.status_code == 200:
                available = await _get_ollama_models(ollama_base)
                model_found = any(
                    ollama_model in name or name.startswith(ollama_model.split(":")[0])
                    for name in available
                )
                if model_found:
                    ollama_ok = True
                    matched = next(
                        (n for n in available
                         if ollama_model in n or n.startswith(ollama_model.split(":")[0])),
                        ollama_model,
                    )
                    ollama_status_text = f"ONLINE ({matched})"
                    print(f"  Ollama model '{matched}' available")
                else:
                    ollama_status_text = f"pulling {ollama_model}..."
                    print(f"  Model '{ollama_model}' not cached — pulling...")
                    _log_mcp(f"Pulling model: {ollama_model}")
                    ok, msg = await _ollama_pull_model(ollama_base, ollama_model)
                    if ok:
                        ollama_ok = True
                        ollama_status_text = f"ONLINE ({ollama_model})"
                        print(f"  Model pulled successfully")
                        _log_mcp("Model pull complete")
                    else:
                        ollama_status_text = f"pull failed: {msg[:50]}"
                        print(f"  Model pull failed: {msg}")
            else:
                ollama_status_text = f"offline ({r.status_code})"
    except Exception as exc:
        ollama_status_text = f"offline ({str(exc)[:40]})"

    # ---- State variables ------------------------------------------------
    health = 78.0
    mana = 92.0
    last_processed_health = -999.0
    last_llm_call_time: float = 0.0
    llm_cooldown_s: float = 2.0  # minimum seconds between LLM calls
    latest_state: Any = None
    latest_action: str = "N/A"
    latest_llm_ms: float = 0.0
    latest_memories: list[str] = []
    memory_count: int = 0
    stored_count: int = 0
    processor_lock = asyncio.Lock()
    state_changed = asyncio.Event()
    state_changed.set()
    llm_thinking: bool = False

    # Vision‑specific state
    spatial_text: str = ""
    detection_summary: str = "no detections"
    detection_count: int = 0
    vision_latency_ms: float = 0.0
    vision_skipped: bool = False
    resolved_coords: str = ""
    resolver_skipped: int = 0
    game_objects: list[dict[str, Any]] = []
    synth_detections: list[Any] = []  # synthetic Detection objects from game_objects
    resolved_target_x: int = -1
    resolved_target_y: int = -1
    targetable_count: int = 0

    # ---- Processing pipeline task ---------------------------------------
    async def process_pipeline() -> None:
        nonlocal latest_state, latest_action, latest_llm_ms, llm_thinking
        nonlocal latest_memories, memory_count, stored_count
        nonlocal spatial_text, detection_summary, detection_count
        nonlocal vision_latency_ms, vision_skipped
        nonlocal resolved_coords, resolver_skipped
        nonlocal synth_detections, resolved_target_x, resolved_target_y
        nonlocal targetable_count, last_llm_call_time

        while True:
            await state_changed.wait()
            state_changed.clear()

            async with processor_lock:
                try:
                    frame = create_simulated_frame(
                        health_pct=health, mana_pct=mana,
                        health_text=f"{int(health)}/100",
                        mana_text=f"{int(mana)}/100",
                        location_text="Training Grounds",
                        objective_text="Defeat the dummy",
                        objects=game_objects if game_objects else None,
                    )

                    t0_vision = time.monotonic()
                    state = await processor.process(frame, skip_ocr=True)
                    vision_latency_ms = (time.monotonic() - t0_vision) * 1000
                    latest_state = state

                    # Extract vision data from state
                    sp = state.get("spatial_context")
                    spatial_text = sp if isinstance(sp, str) else ""
                    dets = state.get("detections")

                    # Build synthetic detections from game objects.
                    # YOLO can't detect coloured rectangles, so we inject
                    # Detection objects that feed SpatialContextBuilder,
                    # MacroResolver, and the LLM prompt builder.
                    synth_detections = []
                    if game_objects:
                        synth_detections = _build_synthetic_detections(game_objects)
                        dets = synth_detections
                        spatial_text = _build_spatial_context_text(
                            synth_detections, 500, 400
                        )
                        # Inject into state so LLM prompt builder sees them
                        state.set("spatial_context", spatial_text)
                        state.set("detections", synth_detections)
                    elif dets and len(dets) > 0:
                        pass  # YOLO returned real detections (rare for demo frames)
                    else:
                        dets = None

                    if dets:
                        detection_count = len(dets)
                        names = [d.class_name for d in dets[:5]]
                        detection_summary = f"{detection_count}: {', '.join(names)}"
                        if synth_detections:
                            print(
                                f"  [Vision] Built {len(synth_detections)} synthetic "
                                f"detection(s): {detection_summary}"
                            )
                            print(f"  [Vision] Context: {spatial_text}")
                    else:
                        detection_count = 0
                        detection_summary = "no detections"

                    vision_skipped = (
                        spatial_text == ""
                        and vision_processor_obj is not None
                        and vision_processor_obj.is_enabled
                    )

                    # Count targetable objects (detections that match dynamic macro target classes)
                    target_classes: set[str] = set()
                    for m in profile_macros:
                        for action in m.get("actions", []):
                            if isinstance(action, dict) and action.get("type", "").startswith("dynamic_"):
                                target_classes.add(action.get("target_class", ""))
                    targetable_count = sum(
                        1 for d in (dets or [])
                        if d.class_name in target_classes
                    )

                    # MCP memory search
                    if mcp_online and mcp_client is not None:
                        try:
                            from src.decision_loop import build_memory_query
                            query = build_memory_query(state)
                            results = await mcp_client.search_memories(
                                query=query, tags=["demo"], limit=5,
                            )
                            latest_memories = [
                                r.get("content", "")[:150] for r in results
                            ]
                            memory_count = len(results)
                            _log_mcp(f"Search: {len(results)} results for '{query[:40]}...'")
                        except Exception as exc:
                            _log_mcp(f"Search error: {exc}")
                            latest_memories = []
                            memory_count = 0
                    else:
                        if not mcp_online:
                            latest_memories = []

                    # LLM decision (rate‑limited)
                    resolved_coords = ""
                    resolver_skipped = 0
                    resolved_target_x = -1
                    resolved_target_y = -1

                    if ollama_ok:
                        now_ts = time.monotonic()
                        if now_ts - last_llm_call_time >= llm_cooldown_s:
                            from src.llm_prompt_builder import build_llm_prompt
                            from src.llm_decision import call_llm_decision

                            llm_thinking = True
                            try:
                                memories_for_prompt = (
                                    latest_memories[:5] if latest_memories else []
                                )
                                # Vision is active when processor is enabled
                                # (even if YOLO returns 0 on synthetic frames,
                                # we still inject synthetics below).
                                vision_active = (
                                    vision_processor_obj is not None
                                    and vision_processor_obj.is_enabled
                                )

                                token_budget = 1000 if vision_active else 800
                                messages = build_llm_prompt(
                                    state=state,
                                    available_macros=profile_macros,
                                    memories=memories_for_prompt,
                                    max_tokens=token_budget,
                                    vision_enabled=vision_active,
                                    config=global_config if vision_active else None,
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
                                last_llm_call_time = time.monotonic()
                                if action not in valid_names:
                                    action = "WAIT"
                                latest_action = action

                                # Dynamic macro resolution
                                if action in dynamic_macro_names and dets:
                                    from src.macro_resolver import MacroResolver
                                    macro_def = next(
                                        (m for m in profile_macros if m["name"] == action),
                                        None,
                                    )
                                    if macro_def:
                                        resolver = MacroResolver(
                                            detections=dets, screen_size=(500, 400),
                                        )
                                        resolved = resolver.resolve_all(
                                            macro_def.get("actions", [])
                                        )
                                        resolver_skipped = resolver.skipped_count
                                        if resolved:
                                            parts: list[str] = []
                                            for step in resolved:
                                                if step.get("type") == "mouse_move":
                                                    parts.append(
                                                        f"mouse→({step['x']},{step['y']})"
                                                    )
                                                    # Store for visual reticle
                                                    resolved_target_x = step["x"]
                                                    resolved_target_y = step["y"]
                                                elif step.get("type") == "click":
                                                    parts.append(
                                                        f"click({step.get('button','left')})"
                                                    )
                                            resolved_coords = " → ".join(parts)
                                            print(
                                                f"  Resolution: {action} → {resolved_coords}"
                                            )
                                        else:
                                            resolved_coords = "(resolution failed)"
                                            print(f"  Resolution: {action} → FAILED (skipped)")
                            finally:
                                llm_thinking = False
                        # else: rate-limited, keep previous action
                    else:
                        latest_action = "WAIT (LLM offline)"

                    # MCP memory store
                    if mcp_online and mcp_client is not None:

                        async def _store() -> None:
                            nonlocal stored_count
                            try:
                                state_dict = (
                                    state.to_dict()
                                    if hasattr(state, "to_dict")
                                    else {}
                                )
                                payload: dict[str, Any] = {
                                    "state": {
                                        k: v
                                        for k, v in state_dict.items()
                                        if not k.startswith("_")
                                    },
                                    "action": latest_action,
                                    "health_pct": health,
                                }
                                if spatial_text:
                                    payload["spatial_context"] = spatial_text[:200]
                                if resolved_coords:
                                    payload["resolved_coords"] = resolved_coords
                                await mcp_client.store_memory(
                                    content=json.dumps(payload),
                                    memory_type="short_term",
                                    tags=["demo", "game_event", "short_term"],
                                )
                                stored_count += 1
                                if summariser is not None:
                                    await summariser.record_new_event()
                                _log_mcp(
                                    f"Store: hp={health:.0f}% → {latest_action} "
                                    f"(total={stored_count})"
                                )
                            except Exception as exc:
                                _log_mcp(f"Store error: {exc}")

                        asyncio.create_task(_store())

                except Exception as exc:
                    import traceback
                    tb = traceback.format_exc()
                    logger.error("Pipeline error:\n{}", tb)
                    _log_mcp(f"Pipeline error: {exc}")

    pipeline_task = asyncio.create_task(process_pipeline())

    # ---- Frame generator ------------------------------------------------
    from tests.frame_generator import create_simulated_frame

    # ---- OpenCV window --------------------------------------------------
    WINDOW_W, WINDOW_H = 780, 920
    cv2.namedWindow("Game AI Player — Phase 4 Vision Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Game AI Player — Phase 4 Vision Demo", WINDOW_W, WINDOW_H)

    print(
        "\n  Live demo started —\n"
        "  [1]=78%HP  [2]=15%HP  [3]=5%HP  "
        "[e]=enemy  [p]=potion  [c]=clear\n"
        "  [s]=store  [m]=summariser  [t]=trigger  [r]=reset  [q]=quit\n"
    )

    render_count = 0
    fps_update_time = time.monotonic()
    fps_value = 0.0

    try:
        while True:
            # ---- Detect health change → trigger pipeline ----------------
            # Use 15pp threshold to reduce LLM calls (phi3.5 is slow)
            if abs(health - last_processed_health) >= 15.0:
                last_processed_health = health
                state_changed.set()

            # ---- Generate frame ------------------------------------------
            frame = create_simulated_frame(
                health_pct=health, mana_pct=mana,
                health_text=f"{int(health)}/100",
                mana_text=f"{int(mana)}/100",
                location_text="Training Grounds",
                objective_text="Defeat the dummy",
                objects=game_objects if game_objects else None,
            )

            canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
            game_area_w = 370
            game_area_h = 296
            game_x, game_y = 10, 10

            if frame is not None:
                resized = cv2.resize(frame, (game_area_w, game_area_h))
                canvas[game_y : game_y + game_area_h, game_x : game_x + game_area_w] = resized

                # ---- Draw detection overlays on the game frame ----------
                scale_x = game_area_w / 500.0
                scale_y = game_area_h / 400.0
                for det in synth_detections:
                    bx1 = int(game_x + det.bbox[0] * scale_x)
                    by1 = int(game_y + det.bbox[1] * scale_y)
                    bx2 = int(game_x + det.bbox[2] * scale_x)
                    by2 = int(game_y + det.bbox[3] * scale_y)
                    # Cyan detection box
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 220, 80), 2)
                    # Label
                    cx = int(game_x + det.center[0] * scale_x)
                    cy = int(game_y + det.center[1] * scale_y)
                    cv2.putText(
                        canvas, det.class_name,
                        (bx1, by1 - 4), cv2.FONT_HERSHEY_PLAIN, 0.55,
                        (255, 220, 80), 1, cv2.LINE_4,
                    )
                    # Small cross at center
                    cv2.drawMarker(
                        canvas, (cx, cy), (80, 255, 220),
                        cv2.MARKER_CROSS, 8, 1,
                    )

                # ---- Draw targeting reticle if resolution happened ------
                if resolved_target_x >= 0 and resolved_target_y >= 0:
                    rx = int(game_x + resolved_target_x * scale_x)
                    ry = int(game_y + resolved_target_y * scale_y)
                    # Green crosshair — larger, more visible
                    cv2.drawMarker(
                        canvas, (rx, ry), (0, 255, 0),
                        cv2.MARKER_CROSS, 16, 2,
                    )
                    cv2.circle(canvas, (rx, ry), 18, (0, 255, 0), 1)
                    cv2.putText(
                        canvas, "TARGET",
                        (rx + 14, ry + 4), cv2.FONT_HERSHEY_PLAIN, 0.5,
                        (0, 255, 0), 1, cv2.LINE_4,
                    )

            # ---- Overlay panel (right side) -------------------------------
            panel_x = game_area_w + 25
            y = 12
            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 0.65
            fg_color = (200, 230, 200)
            dim_color = (120, 140, 120)
            hl_color = (100, 255, 150)
            warn_color = (100, 180, 255)
            mem_color = (200, 200, 100)
            vision_color = (180, 220, 255)
            vision_dim = (120, 160, 180)
            active_color = (100, 255, 150)

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
                        canvas, txt, (panel_x, y), font, font_scale, col, 1, cv2.LINE_4,
                    )
                    y += line_h

            # --- Section 1: Health + status ---
            detected_health = (
                latest_state.get("health") if latest_state is not None else None
            )
            hp_str = f"{detected_health:.0f}%" if detected_health is not None else "?"
            _put_section(
                [
                    (f"health: {hp_str}   mana: {mana:.0f}%", hl_color),
                    (f"MCP: {mcp_status_text}", fg_color if mcp_online else warn_color),
                    (f"Ollama: {ollama_status_text}", fg_color if ollama_ok else warn_color),
                    (f"Memories: search={memory_count}  stored={stored_count}", fg_color),
                ],
                bg=(20, 28, 20),
            )
            y += 2

            # --- Section 2: LLM decision ---
            if llm_thinking:
                _put_section(
                    [(f"LLM → thinking...  (cooldown={llm_cooldown_s}s)", warn_color)],
                    bg=(20, 20, 28),
                )
            else:
                llm_color = fg_color if ollama_ok else dim_color
                llm_text = (
                    f"LLM → {latest_action}  ({latest_llm_ms:.0f}ms)"
                    if ollama_ok
                    else "LLM: offline"
                )
                _put_section([(llm_text, llm_color)], bg=(20, 20, 28))
            y += 1

            # --- Section 2b: Resolution ---
            if resolved_coords:
                _put_section(
                    [(f"Resolved: {resolved_coords}", active_color)],
                    bg=(18, 22, 32),
                )
                y += 1
            elif resolver_skipped > 0:
                _put_section(
                    [(f"Resolution: {resolver_skipped} step(s) skipped", vision_dim)],
                    bg=(18, 22, 32),
                )
                y += 1
            y += 2

            # --- Section 3: Vision status ---
            vision_status_text = (
                "ONLINE (synth)" if synth_detections
                else "ONLINE (yolo)" if (vision_processor_obj is not None and vision_processor_obj.is_enabled)
                else "disabled"
            )
            _put_section(
                [
                    (f"Vision: {vision_status_text}  "
                     f"lat={vision_latency_ms:.0f}ms",
                     vision_color if vision_enabled else vision_dim),
                    (f"Detections: {detection_summary}",
                     vision_color if detection_count else vision_dim),
                    (f"Targetable objects: {targetable_count}",
                     active_color if targetable_count else dim_color),
                ],
                bg=(18, 22, 28),
            )
            y += 1

            # --- Section 4: Spatial context ---
            if spatial_text:
                ctx_short = spatial_text[:110] + "..." if len(spatial_text) > 110 else spatial_text
                _put_section(
                    [(f"Context: {ctx_short}", vision_dim)],
                    bg=(22, 22, 18),
                )
                y += 1

            if vision_skipped:
                _put_section([("  (vision skipped this frame)", dim_color)], bg=(22, 22, 18))
                y += 1
            y += 2

            # --- Section 5: Memory search ---
            if latest_memories:
                mem_lines: list[tuple[str, tuple[int, int, int]]] = [
                    ("Recent memories:", mem_color)
                ]
                for mem in latest_memories[:3]:
                    truncated = mem[:65] + "..." if len(mem) > 65 else mem
                    mem_lines.append((f"  {truncated}", dim_color))
                _put_section(mem_lines, bg=(28, 26, 18))
                y += 2
            elif mcp_online:
                _put_section([("Recent memories: (none yet)", dim_color)], bg=(28, 26, 18))
                y += 2
            else:
                _put_section([("Memories: waiting for MCP...", dim_color)], bg=(28, 26, 18))
                y += 2

            # --- Section 6: Summariser ---
            if summariser is not None:
                task_running = (
                    summariser._task is not None and not summariser._task.done()
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

            # --- Section 7: Game objects ---
            if game_objects:
                obj_lines: list[tuple[str, tuple[int, int, int]]] = [
                    (f"Objects on screen: {len(game_objects)}", vision_color)
                ]
                for obj in game_objects:
                    label = obj.get("label", "?")
                    obj_lines.append(
                        (f"  {label} at ({obj.get('x',0)},{obj.get('y',0)})", vision_dim)
                    )
                _put_section(obj_lines, bg=(18, 22, 28))
                y += 2

            # --- Section 8: MCP Log ---
            log_lines: list[tuple[str, tuple[int, int, int]]] = [
                ("MCP Log:", (140, 140, 140))
            ]
            for log_line in mcp_log[-7:]:
                log_lines.append((f"  {log_line[:200]}", (100, 100, 100)))
            _put_section(log_lines, bg=(18, 18, 18))
            y += 2

            # --- Section 9: Vision metrics ---
            if vision_enabled and hasattr(processor, "_metrics"):
                pm = processor._metrics
                _put_section(
                    [
                        (f"Skips: {pm.vision_frames_skipped_due_to_ocr}  "
                         f"timeouts: {pm.vision_timeouts}  "
                         f"contention: {pm.vision_contention_events}",
                         vision_dim),
                    ],
                    bg=(20, 20, 20),
                )
                y += 2

            # --- Section 10: FPS ---
            _put_section([(f"FPS: {fps_value:.0f}", dim_color)], bg=(20, 20, 20))
            y += 4

            # --- Bottom controls bar ---
            ctrl_text = (
                "[1]=78%  [2]=15%  [3]=5%  [e]=enemy  [p]=potion  [c]=clear  "
                "[s]=store  [m]=summ  [t]=trig  [r]=reset  [q]=quit"
            )
            cv2.rectangle(canvas, (0, WINDOW_H - 22), (WINDOW_W, WINDOW_H), (15, 15, 20), -1)
            cv2.putText(
                canvas, ctrl_text, (8, WINDOW_H - 6), font, 0.58,
                (140, 140, 160), 1, cv2.LINE_4,
            )

            # ---- Display -------------------------------------------------
            cv2.imshow("Game AI Player — Phase 4 Vision Demo", canvas)

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
            elif key == ord("e"):
                game_objects.append({
                    "x": 340, "y": 110, "w": 60, "h": 70,
                    "color": (0, 0, 255), "label": "enemy",
                })
                state_changed.set()
                print("  Enemy spawned at (340,110) — simulated 'person' detection")
            elif key == ord("p"):
                game_objects.append({
                    "x": 375, "y": 275, "w": 40, "h": 35,
                    "color": (255, 144, 30), "label": "item",
                })
                state_changed.set()
                print("  Potion spawned at (375,275) — simulated 'bottle' detection")
            elif key == ord("c"):
                count = len(game_objects)
                game_objects.clear()
                synth_detections.clear()
                resolved_target_x = -1
                resolved_target_y = -1
                state_changed.set()
                print(f"  Cleared {count} game object(s)")
            elif key == ord("s"):
                if mcp_online and mcp_client is not None and latest_state is not None:
                    try:
                        state_dict = (
                            latest_state.to_dict()
                            if hasattr(latest_state, "to_dict")
                            else {}
                        )
                        payload: dict[str, Any] = {
                            "state": {
                                k: v
                                for k, v in state_dict.items()
                                if not k.startswith("_")
                            },
                            "action": latest_action,
                            "health_pct": health,
                            "manual": True,
                        }
                        if spatial_text:
                            payload["spatial_context"] = spatial_text[:200]
                        await mcp_client.store_memory(
                            content=json.dumps(payload),
                            memory_type="short_term",
                            tags=["demo", "game_event", "short_term", "manual"],
                        )
                        stored_count += 1
                        _log_mcp(f"Manual store: {health:.0f}% → {latest_action}")
                        print(f"  Manual memory stored (total: {stored_count})")
                    except Exception as exc:
                        print(f"  Store failed: {exc}")
                else:
                    print("  MCP offline — cannot store")
            elif key == ord("m"):
                if summariser is not None:
                    if summariser._task is not None and not summariser._task.done():
                        await summariser.stop()
                        _log_mcp("Summariser stopped")
                        print("  Summariser stopped")
                    else:
                        await summariser.start()
                        _log_mcp("Summariser started")
                        print("  Summariser started")
            elif key == ord("t"):
                if summariser is not None and mcp_client is not None:
                    for _ in range(summariser.threshold + 1):
                        await summariser.record_new_event()
                    _log_mcp("Manual summarisation trigger")
                    print("  Summarisation triggered — check MCP log")
            elif key == ord("r"):
                if mcp_online and mcp_client is not None:
                    try:
                        results = await mcp_client.search_memories(
                            query="game", tags=["demo"], limit=100,
                        )
                        deleted = 0
                        for r in results:
                            ch = r.get("content_hash") or r.get("memory", {}).get("content_hash")
                            if ch:
                                try:
                                    await mcp_client.delete_memory(ch)
                                    deleted += 1
                                except Exception:
                                    pass
                        _log_mcp(f"Reset: deleted {deleted} memories")
                        for mem_text in _SEED_MEMORIES:
                            await mcp_client.store_memory(
                                content=mem_text, memory_type="short_term",
                                tags=["demo", "game_event", "short_term"],
                            )
                        stored_count = len(_SEED_MEMORIES)
                        _log_mcp(f"Re-seeded {len(_SEED_MEMORIES)} memories")
                        print(f"  Reset: deleted {deleted}, re-seeded {len(_SEED_MEMORIES)}")
                    except Exception as exc:
                        print(f"  Reset failed: {exc}")
                else:
                    print("  MCP offline — cannot reset")

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

        from src.state_processor import shutdown_vision_executor
        shutdown_vision_executor(wait=False)

        cv2.destroyAllWindows()
        print("\n  Phase 4 demo ended.")


# ---------------------------------------------------------------------------
# Helper: build spatial context text from synthetic detections
# ---------------------------------------------------------------------------

def _build_spatial_context_text(
    detections: list[Any],
    screen_w: int,
    screen_h: int,
) -> str:
    """Build a spatial context string identical to what
    SpatialContextBuilder would produce, from synthetic detections."""
    if not detections:
        return "No objects detected visually."
    grouped: dict[str, list[Any]] = {}
    for d in detections:
        grouped.setdefault(d.class_name, []).append(d)
    lines = ["Visual detections:"]
    h_thresh = (0.4, 0.6)
    v_thresh = (0.4, 0.6)
    for class_name, items in grouped.items():
        closest = min(
            items,
            key=lambda d: ((d.center[0] - screen_w / 2) ** 2 + (d.center[1] - screen_h / 2) ** 2),
        )
        rx = closest.center[0] / screen_w
        ry = closest.center[1] / screen_h
        h_pos = "left" if rx < h_thresh[0] else "right" if rx > h_thresh[1] else "center"
        v_pos = "top" if ry < v_thresh[0] else "bottom" if ry > v_thresh[1] else "middle"
        lines.append(f"- {class_name} detected at {v_pos}-{h_pos} (approx. screen position)")
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())