"""
core/provenance.py
------------------
Data provenance and source tracking.

Records, per session, the chain of queries and which files/columns they touched.

Public API:
    record_query(session_id, query, files_used, columns_used, operation, result)
    get_provenance(session_id) → List[ProvenanceRecord]
    clear_provenance(session_id) → None
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

# In-memory store: {session_id: [ProvenanceRecord, ...]}
_store: Dict[str, List["ProvenanceRecord"]] = {}
_MAX_RECORDS_PER_SESSION = 100


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ProvenanceRecord:
    session_id: str
    query: str
    files_used: List[str]
    columns_used: List[str]
    operation_type: str          # "analysis" | "visualization" | "structural" | "error"
    timestamp: str               # ISO 8601
    latency_ms: int
    result_hash: str             # SHA-256 of the result string (not the result itself)
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query[:200],
            "files_used": self.files_used,
            "columns_used": self.columns_used[:20],   # cap for readability
            "operation_type": self.operation_type,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "result_hash": self.result_hash[:12],     # prefix only
            "success": self.success,
            "error_message": self.error_message,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def record_query(
    session_id: str,
    query: str,
    operation_type: str,
    result: str,
    latency_ms: int,
    files_used: Optional[List[str]] = None,
    columns_used: Optional[List[str]] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Append a provenance record for a completed (or failed) query.
    """
    result_hash = hashlib.sha256(result.encode()).hexdigest()
    success = error_message is None

    record = ProvenanceRecord(
        session_id=session_id,
        query=query,
        files_used=files_used or [],
        columns_used=columns_used or [],
        operation_type=operation_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=latency_ms,
        result_hash=result_hash,
        success=success,
        error_message=error_message,
    )

    if session_id not in _store:
        _store[session_id] = []

    _store[session_id].append(record)

    # Rolling cap
    if len(_store[session_id]) > _MAX_RECORDS_PER_SESSION:
        _store[session_id] = _store[session_id][-_MAX_RECORDS_PER_SESSION:]

    status = "✓" if success else "✗"
    logger.debug(
        f"Provenance [{status}] session={session_id[:8]} "
        f"op={operation_type} files={files_used} latency={latency_ms}ms"
    )


def get_provenance(session_id: str) -> List[ProvenanceRecord]:
    """Return all provenance records for a session (oldest first)."""
    return list(_store.get(session_id, []))


def get_provenance_dicts(session_id: str) -> List[dict]:
    """Return provenance as JSON-serialisable dicts."""
    return [r.to_dict() for r in get_provenance(session_id)]


def clear_provenance(session_id: str) -> None:
    """Remove all provenance records for a session."""
    _store.pop(session_id, None)


def session_stats(session_id: str) -> dict:
    """
    Return a summary statistics dict for a session's query history.
    """
    records = get_provenance(session_id)
    if not records:
        return {"total_queries": 0}

    total = len(records)
    successes = sum(1 for r in records if r.success)
    avg_latency = sum(r.latency_ms for r in records) / total
    ops = {}
    for r in records:
        ops[r.operation_type] = ops.get(r.operation_type, 0) + 1

    all_files: set = set()
    for r in records:
        all_files.update(r.files_used)

    return {
        "total_queries": total,
        "successful_queries": successes,
        "failed_queries": total - successes,
        "success_rate_pct": round(successes / total * 100, 1),
        "avg_latency_ms": round(avg_latency),
        "operations_breakdown": ops,
        "files_queried": sorted(all_files),
    }
