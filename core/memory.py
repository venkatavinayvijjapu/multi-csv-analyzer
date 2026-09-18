"""
core/memory.py
--------------
Per-session conversation memory for follow-up questions.

Memory is stored as an ordered list of exchanges inside the session_store
(session_store.py). This module provides a clean API over that storage.

Usage:
    from core.memory import get_history, add_exchange, format_history_for_prompt

    # Load last 5 turns
    history = get_history(session_id, max_turns=5)

    # After a query completes
    add_exchange(session_id, query="...", answer="...")

    # Inject into an LLM prompt
    history_text = format_history_for_prompt(history)
"""

from __future__ import annotations

from typing import List, Dict

from core.logging_config import get_logger

logger = get_logger(__name__)

# Imported lazily to avoid circular import at module load time
_session_store = None

def _ss():
    global _session_store
    if _session_store is None:
        import session_store as ss
        _session_store = ss
    return _session_store


# ── Types ─────────────────────────────────────────────────────────────────────

Exchange = Dict[str, str]   # {"role": "user"|"assistant", "content": "..."}


# ── Public API ────────────────────────────────────────────────────────────────

def get_history(session_id: str, max_turns: int = 5) -> List[Exchange]:
    """
    Return the last *max_turns* (user + assistant) exchange pairs for the session.
    Each pair is two entries: {role: "user", content: ...}, {role: "assistant", content: ...}.
    """
    all_messages: List[Exchange] = _ss().get_conversation_history(session_id)
    # Take the last max_turns * 2 messages (each turn = 1 user + 1 assistant)
    tail = all_messages[-(max_turns * 2):]
    return tail


def add_exchange(session_id: str, query: str, answer: str) -> None:
    """
    Append a completed user↔assistant exchange to the session's history.
    Truncates to avoid the history growing unbounded (keep last 20 turns).
    """
    ss = _ss()
    ss.append_message(session_id, role="user", content=query)
    # Truncate assistant answer to 1000 chars to keep prompts manageable
    ss.append_message(session_id, role="assistant", content=answer[:1000])
    logger.debug(f"Memory: saved exchange for session {session_id[:8]}")


def clear_history(session_id: str) -> None:
    """Wipe conversation history for a session."""
    _ss().clear_conversation_history(session_id)


def format_history_for_prompt(history: List[Exchange]) -> str:
    """
    Format conversation history as a compact block for LLM prompt injection.

    Example output:
        Previous conversation:
        User: what is the total revenue?
        Assistant: ### Total Revenue\n**Grand Total: 1,234,567.00**
        User: break it down by region
        (current query follows)
    """
    if not history:
        return ""

    lines = ["Previous conversation:"]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        # Trim long assistant answers in the history context
        content = msg["content"]
        if role == "Assistant" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{role}: {content}")
    lines.append("(Current query follows below)")
    return "\n".join(lines)
