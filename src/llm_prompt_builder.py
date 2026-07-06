"""
LLM Prompt Builder — Phase 2.6

Builds a messages array for the Ollama ``/api/chat`` endpoint from game
state, available macros, and (stubbed) recent memories, respecting a
configurable token budget (default 800 tokens).

The prompt builder is a pure, stateless function — no async, no side effects.

Usage::

    from src.game_state import GameState, StateSchema
    from src.llm_prompt_builder import build_llm_prompt

    messages = build_llm_prompt(
        state=game_state,
        available_macros=profile_macros,
        memories=mcp_memories,          # list[dict] | None
        state_schema=schema,
        max_tokens=800,
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
    max_tokens: int = 800,
) -> list[dict[str, str]]:
    """Create a message list for the Ollama ``/api/chat`` endpoint.

    The system prompt lists allowed macro names and instructs the LLM to
    return a JSON object with an ``"action"`` field.  The user message
    combines game state, spatial context, available macros, and recent
    memories — all within the token budget.

    Args:
        state: Populated ``GameState`` from ``StateProcessor.process()``.
        available_macros: List of macro dicts, each with at least
            ``"name"``; optionally ``"description"``.
        memories: List of memory dicts (each with ``"content"`` key).
            Pass ``None`` or ``[]`` for the Phase 2 stub.
        state_schema: Optional ``StateSchema``; if ``None``, derived from
            ``state.schema``.
        max_tokens: Maximum total tokens for the assembled message
            (system + user combined).

    Returns:
        A list of message dicts suitable for Ollama ``/api/chat``:
        ``[{"role": "system", ...}, {"role": "user", ...}]``
    """
    if memories is None:
        memories = []

    # Resolve schema
    schema = state_schema if state_schema is not None else state.schema

    # Serialize state to a plain dict for section building
    state_dict = _build_state_dict(state, schema)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    macro_names = [m["name"] for m in available_macros] if available_macros else ["WAIT"]
    system_prompt = (
        "You are a game AI agent. Respond ONLY with a JSON object containing "
        f"an \"action\" field from: {', '.join(macro_names)}.\n"
        "No extra words, no explanations."
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

    # 3. Available macros (priority 0)
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

    # 4. Recent memories (priority 1 — trimmable)
    if memories:
        memory_lines: list[str] = []
        for mem in memories[:_MAX_MEMORIES]:
            content = mem.get("content", "")
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
# Module exports
# ---------------------------------------------------------------------------

__all__ = ["build_llm_prompt", "count_tokens"]