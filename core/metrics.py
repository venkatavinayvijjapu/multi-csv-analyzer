"""
core/metrics.py
---------------
Accuracy and reliability metrics for the DataLens AI system.

Tracks per-session and global:
  - Code execution success rate
  - LLM retry rate
  - Average response latency
  - Timeout rate

Used by /api/health and /api/metrics endpoints, and by the eval framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

_lock = Lock()


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class QueryMetric:
    session_id: str
    operation: str         # "analysis" | "visualization" | "structural"
    success: bool
    retried: bool
    timed_out: bool
    latency_ms: int
    code_executed: bool


# ── Global rolling window (last 1000 queries) ─────────────────────────────────

_MAX_HISTORY = 1000
_history: List[QueryMetric] = []


# ── Public API ────────────────────────────────────────────────────────────────

def record_metric(
    session_id: str,
    operation: str,
    success: bool,
    latency_ms: int,
    retried: bool = False,
    timed_out: bool = False,
    code_executed: bool = True,
) -> None:
    """Record one query's outcome metrics."""
    with _lock:
        _history.append(QueryMetric(
            session_id=session_id,
            operation=operation,
            success=success,
            retried=retried,
            timed_out=timed_out,
            latency_ms=latency_ms,
            code_executed=code_executed,
        ))
        if len(_history) > _MAX_HISTORY:
            del _history[0]

    logger.debug(
        f"Metric: {operation} success={success} retried={retried} "
        f"timeout={timed_out} latency={latency_ms}ms"
    )


def get_global_stats() -> dict:
    """Return aggregated stats across the last _MAX_HISTORY queries."""
    with _lock:
        data = list(_history)

    if not data:
        return {
            "total_queries": 0,
            "message": "No queries recorded yet.",
        }

    total = len(data)
    successes = sum(1 for m in data if m.success)
    retried = sum(1 for m in data if m.retried)
    timeouts = sum(1 for m in data if m.timed_out)
    code_exec = sum(1 for m in data if m.code_executed)
    avg_lat = sum(m.latency_ms for m in data) / total

    ops: Dict[str, int] = {}
    for m in data:
        ops[m.operation] = ops.get(m.operation, 0) + 1

    p95_lat = sorted(m.latency_ms for m in data)[int(total * 0.95)] if total >= 20 else None

    return {
        "total_queries": total,
        "success_rate_pct": round(successes / total * 100, 1),
        "retry_rate_pct": round(retried / total * 100, 1),
        "timeout_rate_pct": round(timeouts / total * 100, 1),
        "code_execution_rate_pct": round(code_exec / total * 100, 1),
        "avg_latency_ms": round(avg_lat),
        "p95_latency_ms": p95_lat,
        "operations_breakdown": ops,
    }


class LatencyTimer:
    """Context manager that returns elapsed ms."""
    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "LatencyTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
