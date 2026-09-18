"""
session_store.py
----------------
In-memory session store with:
  • TTL-based expiry (default 2 hours)
  • Conversation history per session
  • DataQualityReport per file
  • Session metadata (created_at, last_accessed, file_count)
  • Background cleanup thread (runs every 10 minutes)

Maps: session_id → SessionData
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from core.logging_config import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
SESSION_TTL_SECONDS: int = int(__import__("os").getenv("SESSION_TTL_SECONDS", "7200"))  # 2 hours
CLEANUP_INTERVAL_SECONDS: int = 600   # 10 minutes
MAX_CONVERSATION_TURNS: int = 40      # max messages (20 exchanges) per session


# ── Session data container ────────────────────────────────────────────────────

@dataclass
class SessionData:
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    quality_reports: Dict[str, dict] = field(default_factory=dict)   # filename → report dict
    conversation_history: List[dict] = field(default_factory=list)    # [{role, content}]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed = time.time()

    def is_expired(self, ttl: int = SESSION_TTL_SECONDS) -> bool:
        return (time.time() - self.last_accessed) > ttl

    def memory_estimate_mb(self) -> float:
        """Rough in-memory size estimate of stored DataFrames."""
        total = 0
        for df in self.dataframes.values():
            try:
                total += df.memory_usage(deep=True).sum()
            except Exception:
                pass
        return total / (1024 * 1024)


# ── Global registry ───────────────────────────────────────────────────────────

_store: Dict[str, SessionData] = {}
_store_lock = threading.Lock()


def _get_or_create(session_id: str) -> SessionData:
    with _store_lock:
        if session_id not in _store:
            _store[session_id] = SessionData()
            logger.info(f"Session created: {session_id[:8]}")
        sess = _store[session_id]
        sess.touch()
        return sess


# ── DataFrames ────────────────────────────────────────────────────────────────

def get_session(session_id: str) -> Dict[str, pd.DataFrame]:
    """Return the DataFrames dict for a session. Never raises."""
    with _store_lock:
        sess = _store.get(session_id)
        if sess:
            sess.touch()
            return dict(sess.dataframes)
        return {}


def set_file(
    session_id: str,
    filename: str,
    df: pd.DataFrame,
    quality_report: Optional[dict] = None,
) -> None:
    """Store a DataFrame (and optional quality report) under the given session."""
    sess = _get_or_create(session_id)
    with _store_lock:
        sess.dataframes[filename] = df
        if quality_report is not None:
            sess.quality_reports[filename] = quality_report
    logger.info(f"Session {session_id[:8]}: stored '{filename}' shape={df.shape}")


def remove_file(session_id: str, filename: str) -> None:
    """Remove a single file from a session."""
    with _store_lock:
        sess = _store.get(session_id)
        if sess:
            sess.dataframes.pop(filename, None)
            sess.quality_reports.pop(filename, None)


def clear_session(session_id: str) -> None:
    """Wipe all data for a session."""
    with _store_lock:
        _store.pop(session_id, None)
    logger.info(f"Session {session_id[:8]}: cleared")


def list_files(session_id: str) -> list:
    """Return file metadata list for a session."""
    sess = _get_or_create(session_id) if session_id in _store else None
    if sess is None:
        with _store_lock:
            sess = _store.get(session_id)
    if not sess:
        return []

    result = []
    for name, df in sess.dataframes.items():
        num_cols = df.select_dtypes("number").columns.tolist()
        entry = {
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "col_names": df.columns.tolist(),
            "numeric_cols": num_cols,
            "null_count": int(df.isnull().sum().sum()),
        }
        if name in sess.quality_reports:
            entry["quality"] = sess.quality_reports[name]
        result.append(entry)
    return result


# ── Quality reports ───────────────────────────────────────────────────────────

def get_quality_report(session_id: str, filename: str) -> Optional[dict]:
    """Return the quality report for a specific file."""
    with _store_lock:
        sess = _store.get(session_id)
        if sess:
            return sess.quality_reports.get(filename)
    return None


# ── Conversation history ──────────────────────────────────────────────────────

def get_conversation_history(session_id: str) -> List[dict]:
    """Return all conversation messages for a session."""
    with _store_lock:
        sess = _store.get(session_id)
        if sess:
            return list(sess.conversation_history)
        return []


def append_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session's conversation history."""
    sess = _get_or_create(session_id)
    with _store_lock:
        sess.conversation_history.append({"role": role, "content": content})
        # Rolling cap
        if len(sess.conversation_history) > MAX_CONVERSATION_TURNS:
            sess.conversation_history = sess.conversation_history[-MAX_CONVERSATION_TURNS:]


def clear_conversation_history(session_id: str) -> None:
    """Clear conversation history for a session."""
    with _store_lock:
        sess = _store.get(session_id)
        if sess:
            sess.conversation_history = []


# ── Session metadata ──────────────────────────────────────────────────────────

def session_info(session_id: str) -> dict:
    """Return metadata for a session."""
    with _store_lock:
        sess = _store.get(session_id)
    if not sess:
        return {"exists": False}
    now = time.time()
    return {
        "exists": True,
        "file_count": len(sess.dataframes),
        "message_count": len(sess.conversation_history),
        "age_seconds": int(now - sess.created_at),
        "idle_seconds": int(now - sess.last_accessed),
        "ttl_remaining_seconds": max(0, SESSION_TTL_SECONDS - int(now - sess.last_accessed)),
        "memory_estimate_mb": round(sess.memory_estimate_mb(), 2),
    }


def extend_session(session_id: str) -> None:
    """Reset the TTL for a session."""
    with _store_lock:
        sess = _store.get(session_id)
        if sess:
            sess.touch()
            logger.debug(f"Session {session_id[:8]}: TTL extended")


def active_session_count() -> int:
    """Return the number of live sessions."""
    with _store_lock:
        return len(_store)


# ── Background TTL cleanup ────────────────────────────────────────────────────

def _cleanup_expired_sessions() -> None:
    """Remove sessions that have exceeded their TTL."""
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        expired = []
        with _store_lock:
            for sid, sess in list(_store.items()):
                if sess.is_expired():
                    expired.append(sid)
            for sid in expired:
                del _store[sid]
        if expired:
            logger.info(f"TTL cleanup: removed {len(expired)} expired session(s)")


# Start cleanup daemon at import time
_cleanup_thread = threading.Thread(
    target=_cleanup_expired_sessions, daemon=True, name="session-cleanup"
)
_cleanup_thread.start()
