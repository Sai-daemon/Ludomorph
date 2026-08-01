"""
LLM Prompt Builder — Phase 2.6 / Phase 4.4

Builds a messages array for the Ollama ``/api/chat`` endpoint from game
state, available macros, and (stubbed) recent memories, respecting a
configurable token budget (default 800 tokens, raised to 1000 when vision
is active).

Phase 4.4 extends the builder with:
- Vision‑aware system prompt (dynamic macro instructions when objects detected)
- Detection‑to‑macro mapping section (maps detected classes → dynamic macros)
- Configurable token budgets per mode (``llm_max_tokens`` / ``llm_vision_max_tokens``)
- ``_DETECTION_GOOD_CLASSES`` filtering to suppress noisy COCO detections
  (e.g. ``"chair"``, ``"potted plant"``) from the prompt, reducing budget waste.

The prompt builder is a pure, stateless function — no async, no side effects.

Usage::

    from src.game_state import GameState, StateSchema
    from src.llm_prompt_builder import build_llm_prompt

    messages = build_llm_prompt(
        state=game_state,
        available_macros=profile_macros,
        memories=mcp_memories,          # list[dict] | None
        state_schema=schema,
        config=global_config,           # dict with optional llm_* keys
        vision_enabled=True,
    )
    # messages is a list of dicts suitable for httpx → Ollama /api/chat
    # [
    #     {"role": "system", "content": "You are a game AI agent. ..."},
    #     {"role": "user",   "content": "Current Game State:\\nhealth: 78.5\\n..."},
    # ]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.game_state import GameState, StateSchema

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN: float = 3.5
"""Rough estimate: ~3.5 characters per token for English text."""

# Suffixes to skip when iterating extra state keys
_SKIP_KEY_SUFFIXES: tuple[str, ...] = ("_raw_bar", "_raw_image")

# Max number of memories to include
_MAX_MEMORIES: int = 5

# Max characters per memory content
_MAX_MEMORY_CONTENT_CHARS: int = 200

# Default token budgets
_DEFAULT_MAX_TOKENS: int = 800
_DEFAULT_VISION_MAX_TOKENS: int = 1000

# Phase 4.4 — COCO classes that are useful for game targeting.
# Generic objects like "chair", "potted plant" produce too much noise
# in the prompt.  Only keep classes that are plausible as game entities.
_DETECTION_GOOD_CLASSES: frozenset[str] = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "boat", "airplane",
    "traffic light", "fire hydrant", "stop sign",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear",
    "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
    "bed", "dining table", "toilet",
    "hair drier", "toothbrush",
})
"""Subset of COCO classes that are worth mentioning to the LLM.

Excludes overly generic scenery classes (``"potted plant"``, ``"chair"``,
``"couch"``, ``"parking meter"``, ``"bench"``) that rarely represent
interactive game objects and waste token budget.
"""


# ---------------------------------------------------------------------------
# Public: token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Rough token count using character-length heuristic.

    Returns at least 1 to ensure even short strings count as ≥1 token.
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


# ---------------------------------------------------------------------------
# Public: build_llm_prompt
# ---------------------------------------------------------------------------


def build_llm_prompt(
    state: "GameState",
    available_macros: list[dict[str, Any]],
    memories: list[dict[str, Any]] | None = None,
    state_schema: "StateSchema | None" = None,
    max_tokens: int | None = None,
    *,
    config: dict[str, Any] | None = None,
    vision_enabled: bool = False,
) -> list[dict[str, str]]:
    """Create a message list for the Ollama ``/api/chat`` endpoint.

    The system prompt lists allowed macro names and instructs the LLM to
    return a JSON object with an ``"action"`` field.  The user message
    combines game state, spatial context, available macros, and recent
    memories — all within the token budget.

    Phase 4.4: When *vision_enabled* is ``True`` the system prompt is
    extended with dynamic‑macro targeting instructions, a
    detection‑to‑macro mapping section is injected (priority 0), and the
    token budget is raised to ``llm_vision_max_tokens`` from *config*
    (default 1000).  Detection results are filtered through
    ``_DETECTION_GOOD_CLASSES`` to suppress noisy COCO classes.

    Args:
        state: Populated ``GameState`` from ``StateProcessor.process()``.
        available_macros: List of macro dicts, each with at least
            ``"name"``; optionally ``"description"``.
        memories: List of memory dicts (each with ``"content"`` key).
            Pass ``None`` or ``[]`` for the Phase 2 stub.
        state_schema: Optional ``StateSchema``; if ``None``, derived from
            ``state.schema``.
        max_tokens: Override for the token budget.  When ``None`` (the
            default) the budget is determined from *config*:
            ``llm_vision_max_tokens`` if *vision_enabled*, else
            ``llm_max_tokens``, else :data:`_DEFAULT_MAX_TOKENS`.
        config: Global config dict (Phase 1.1) containing optional
            ``llm_max_tokens`` and ``llm_vision_max_tokens`` keys.
        vision_enabled: If ``True``, the system prompt gains dynamic
            macro targeting instructions and a detection‑mapping section
            is added.

    Returns:
        A list of message dicts suitable for Ollama ``/api/chat``:
        ``[{"role": "system", ...}, {"role": "user", ...}]``
    """
    if memories is None:
        memories = []

    # Resolve schema
    schema = state_schema if state_schema is not None else state.schema

    # Resolve token budget from config or defaults
    if max_tokens is None:
        if config is not None:
            if vision_enabled:
                max_tokens = config.get("llm_vision_max_tokens", _DEFAULT_VISION_MAX_TOKENS)
            else:
                max_tokens = config.get("llm_max_tokens", _DEFAULT_MAX_TOKENS)
        else:
            max_tokens = _DEFAULT_VISION_MAX_TOKENS if vision_enabled else _DEFAULT_MAX_TOKENS

    # Serialize state to a plain dict for section building
    state_dict = _build_state_dict(state, schema)

    # Extract vision detections (if any) for Phase 4.4 sections
    detections: list[Any] = _normalise_detections(state.get("detections"))

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    macro_names = [m["name"] for m in available_macros] if available_macros else ["WAIT"]

    if vision_enabled and detections:
        # Phase 4.4 — vision‑aware system prompt: tell the LLM it can
        # target detected objects via dynamic macros.
        system_prompt = (
            "You are a game-playing AI. Your ONLY output must be a single JSON object "
            "with exactly one key \"action\". The value must be one of: "
            f"{', '.join(macro_names)}.\n"
            "If a targetable object is detected, you may use a dynamic macro "
            "that targets that object. Prefer static macros when no target is needed.\n"
            "Do NOT output any text before or after the JSON. "
            "Do NOT add extra fields, reasoning, or explanation. "
            "Example: {\"action\": \"" + macro_names[0] + "\"}"
        )
    else:
        system_prompt = (
            "You are a game-playing AI. Your ONLY output must be a single JSON object "
            "with exactly one key \"action\". The value must be one of: "
            f"{', '.join(macro_names)}.\n"
            "Do NOT output any text before or after the JSON. "
            "Do NOT add extra fields, reasoning, or explanation. "
            "Example: {\"action\": \"" + macro_names[0] + "\"}"
        )

    system_tokens = count_tokens(system_prompt)
    remaining_tokens = max_tokens - system_tokens

    if remaining_tokens <= 0:
        logger.warning(
            f"System prompt alone ({system_tokens} tokens) exceeds budget "
            f"({max_tokens}); user content will be empty."
        )
        remaining_tokens = 0

    # ------------------------------------------------------------------
    # User content — sections (priority 0 = essential, 1 = trimmable)
    # ------------------------------------------------------------------
    sections: list[tuple[int, str]] = []

    # 1. Game state (priority 0)
    if state_dict:
        lines = [f"{key}: {value}" for key, value in state_dict.items()]
        state_section = "Current Game State:\n" + "\n".join(lines)
        sections.append((0, state_section))

    # 2. Spatial context from vision (priority 0)
    spatial = state.get("spatial_context")
    if spatial and isinstance(spatial, str) and spatial.strip():
        sections.append((0, spatial))

    # 3. Phase 4.4 — detection‑to‑macro mapping (priority 0)
    if vision_enabled and detections:
        mapping = _build_detection_macro_mapping(detections, available_macros)
        if mapping:
            sections.append((0, mapping))

    # 4. Available macros (priority 0)
    if available_macros:
        macro_lines = []
        for m in available_macros:
            name = m.get("name", "?")
            desc = m.get("description", "")
            if desc:
                macro_lines.append(f"- {name}: {desc}")
            else:
                macro_lines.append(f"- {name}")
        macro_section = "Available Macros:\n" + "\n".join(macro_lines)
        sections.append((0, macro_section))

    # 5. Recent memories (priority 1 — trimmable)
    if memories:
        memory_lines: list[str] = []
        for mem in memories[:_MAX_MEMORIES]:
            content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            memory_lines.append(f"- {content[:_MAX_MEMORY_CONTENT_CHARS]}")
        if memory_lines:
            memory_section = "Recent memory:\n" + "\n".join(memory_lines)
            sections.append((1, memory_section))

    # ------------------------------------------------------------------
    # Assemble, trimming low-priority sections to fit the budget
    # ------------------------------------------------------------------
    sections.sort(key=lambda x: x[0])  # essential (0) before trimmable (1)
    assembled: list[str] = []
    used = 0

    for priority, text in sections:
        tokens = count_tokens(text)
        if used + tokens <= remaining_tokens:
            assembled.append(text)
            used += tokens
        elif priority > 0:
            # Try to include just the first line of a lower-priority section
            first_line = text.split("\n")[0] + "\n(trimmed)"
            trunc_tokens = count_tokens(first_line)
            if used + trunc_tokens <= remaining_tokens:
                assembled.append(first_line)
                used += trunc_tokens
            break  # no room for anything lower
        else:
            # Essential section already too large — stop assembling
            break

    user_content = "\n\n".join(assembled) if assembled else "(no state data available)"

    logger.debug(
        f"Built LLM prompt: {len(system_prompt)} chars system, "
        f"{len(user_content)} chars user (~{count_tokens(user_content)} tokens)"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_state_dict(state: "GameState", schema: "StateSchema") -> dict[str, Any]:
    """Build an ordered dict of state values suitable for prompt inclusion.

    Iterates schema-defined slots first (in schema order), then appends
    extra keys not in the schema — excluding binary/image values and
    spatial context (handled separately by the caller).
    """
    result: dict[str, Any] = {}

    # 1. Schema-defined slots (in declaration order)
    for slot_name in schema.slots:
        value = state.get(slot_name)
        if value is not None:
            result[slot_name] = value

    # 2. Extra keys not in the schema
    for key in state.keys():
        if key in schema.slots:
            continue  # already handled
        if key == "spatial_context":
            continue  # handled separately in build_llm_prompt
        if key == "detections":
            continue  # handled separately via _build_detection_macro_mapping
        if key.startswith("_"):
            continue  # internal metadata (e.g. _state_hash)
        if any(key.endswith(suffix) for suffix in _SKIP_KEY_SUFFIXES):
            continue  # raw images can't go in the prompt
        value = state.get(key)
        if value is not None and _is_serialisable_for_prompt(value):
            result[key] = value

    return result


def _is_serialisable_for_prompt(value: Any) -> bool:
    """Return ``False`` for types that can't be meaningfully included in text."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return False
    # numpy arrays
    if hasattr(value, "dtype"):
        return False
    # Callable — probably a stray function reference
    if callable(value):
        return False
    return True


# ---------------------------------------------------------------------------
# Phase 4.4 — Detection helpers
# ---------------------------------------------------------------------------


def _normalise_detections(raw: Any) -> list[Any]:
    """Safely extract a flat list of detection objects from *raw*.

    Accepts ``None``, an empty list, a single ``Detection``, a
    ``SpatialContext``, or any iterable.  Returns a plain ``list`` of
    detection‑like objects (each expected to have ``class_name`` and
    ``confidence`` attributes).
    """
    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    # Single detection object
    if hasattr(raw, "class_name") and hasattr(raw, "confidence"):
        return [raw]

    # SpatialContext or similar aggregate
    if hasattr(raw, "detections"):
        return list(raw.detections) if raw.detections else []

    return []


def _build_detection_macro_mapping(
    detections: list[Any],
    available_macros: list[dict[str, Any]],
) -> str | None:
    """Build a detection‑to‑macro mapping section for the LLM prompt.

    For each unique detected class name that also appears as a
    ``target_class`` in a dynamic macro step, produce a line like::

        - person → can be targeted with: DYNAMIC_CLICK

    Noisy COCO classes that are not in :data:`_DETECTION_GOOD_CLASSES`
    are silently dropped.

    Returns ``None`` if no useful mappings exist (i.e. no detections
    match any dynamic macro target class).
    """
    # 1. Collect all target_class values from dynamic macros
    dynamic_targets: dict[str, list[str]] = {}  # target_class → [macro_name, ...]
    for macro in available_macros:
        steps = macro.get("actions", macro.get("steps", []))
        for step in steps:
            step_type = step.get("type", "") if isinstance(step, dict) else ""
            if isinstance(step_type, str) and step_type.startswith("dynamic_"):
                target = step.get("target_class", "")
                if target:
                    dynamic_targets.setdefault(target, []).append(macro["name"])

    if not dynamic_targets:
        return None

    # 2. Map detected classes → available dynamic macros
    seen: set[str] = set()
    mappings: list[str] = []
    for det in detections:
        cls_name = getattr(det, "class_name", None)
        if cls_name is None or cls_name in seen:
            continue
        # Filter noisy COCO classes
        if cls_name not in _DETECTION_GOOD_CLASSES:
            continue
        if cls_name in dynamic_targets:
            seen.add(cls_name)
            macro_list = ", ".join(sorted(set(dynamic_targets[cls_name])))
            mappings.append(f"- {cls_name} → can be targeted with: {macro_list}")

    if not mappings:
        return None

    return "Targetable objects:\n" + "\n".join(mappings)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "build_llm_prompt",
    "count_tokens",
    "_DETECTION_GOOD_CLASSES",
    "_normalise_detections",
    "_build_detection_macro_mapping",
]
