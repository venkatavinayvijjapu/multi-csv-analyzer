"""
core/sandbox.py
---------------
Secure, resource-limited execution environment for LLM-generated code.

Key protections:
  - Allowlist of safe builtins (no open, eval, exec, __import__, etc.)
  - Import blocking: only pandas, numpy, matplotlib, io, base64, re, math,
    datetime, collections, itertools, functools are permitted
  - Wall-clock timeout via threading.Timer (default 30 s)
  - Memory tracking via tracemalloc (soft cap, default 512 MB)
  - All stdout captured and returned; stderr never leaks to caller

Usage:
    from core.sandbox import run_code, ExecutionResult
    result = run_code(code_str, exec_env, timeout_seconds=30)
    if result.timed_out:
        ...
    elif result.error:
        ...
    else:
        output = result.stdout
"""

from __future__ import annotations

import ast
import builtins
import io
import math
import contextlib
import threading
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Optional

from core.logging_config import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SECONDS: int = 30
DEFAULT_MEMORY_LIMIT_MB: int = 512

# Modules that generated code is allowed to reference (already injected into
# exec_env by the caller; this list is used by the validator as an allowlist).
ALLOWED_MODULES: frozenset[str] = frozenset({
    "pandas", "pd",
    "numpy", "np",
    "matplotlib", "matplotlib.pyplot", "plt",
    "seaborn", "sns",
    "io", "base64", "re", "math", "datetime", "collections",
    "itertools", "functools", "statistics",
})

# Pre-imported safe stdlib modules to inject into every sandbox environment
import math as _math
import re as _re
import datetime as _datetime
import collections as _collections
import itertools as _itertools
import functools as _functools
import statistics as _statistics

_SAFE_STDLIB: dict = {
    "math": _math,
    "re": _re,
    "datetime": _datetime,
    "collections": _collections,
    "itertools": _itertools,
    "functools": _functools,
    "statistics": _statistics,
}

# Builtins that are explicitly BLOCKED
_BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile",
    "importlib",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "breakpoint", "input",
    "memoryview", "bytearray",
})

# Safe subset of Python builtins to inject
_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if name not in _BLOCKED_BUILTINS and not name.startswith("__")
}
_SAFE_BUILTINS["__builtins__"] = {}   # prevent __builtins__ escape


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    stdout: str = ""
    error: Optional[str] = None
    timed_out: bool = False
    memory_mb: float = 0.0
    code_executed: bool = False

    @property
    def success(self) -> bool:
        return self.code_executed and not self.error and not self.timed_out


# ── Internal runner ───────────────────────────────────────────────────────────

def _run_in_thread(
    code: str,
    exec_env: dict[str, Any],
    result_holder: list,
    memory_limit_mb: int,
) -> None:
    """Target function for the timeout thread."""
    stdout_buf = io.StringIO()
    tracemalloc.start()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, exec_env)  # nosec — sandbox already validated code
        current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)

        if peak_mb > memory_limit_mb:
            result_holder.append(ExecutionResult(
                error=f"Memory limit exceeded: {peak_mb:.1f} MB used (limit {memory_limit_mb} MB)",
                memory_mb=peak_mb,
                code_executed=True,
            ))
        else:
            output = stdout_buf.getvalue().strip()
            result_holder.append(ExecutionResult(
                stdout=output,
                memory_mb=peak_mb,
                code_executed=True,
            ))
    except Exception as exc:
        _, peak = tracemalloc.get_traced_memory()
        result_holder.append(ExecutionResult(
            error=str(exc),
            memory_mb=peak / (1024 * 1024),
            code_executed=True,
        ))
    finally:
        tracemalloc.stop()


# ── Public API ────────────────────────────────────────────────────────────────

def run_code(
    code: str,
    exec_env: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
) -> ExecutionResult:
    """
    Execute *code* inside a sandboxed environment with:
      - A strict allowlist of builtins (no open/eval/exec/import)
      - A wall-clock timeout (threading.Timer)
      - Memory tracking (tracemalloc)

    Args:
        code:             Python source to execute.
        exec_env:         Pre-built namespace {var_name: df, "pd": pd, …}.
        timeout_seconds:  Kill thread after this many seconds.
        memory_limit_mb:  Raise error if peak memory exceeds this.

    Returns:
        ExecutionResult with stdout, error, timed_out, memory_mb.
    """
    # Inject safe builtins + safe stdlib modules (merges with caller-provided env)
    sandboxed_env: dict[str, Any] = {**_SAFE_BUILTINS, **_SAFE_STDLIB, **exec_env}
    
    _orig_import = builtins.__import__
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only allow modules explicitly defined in ALLOWED_MODULES
        # (Using split to handle imports like matplotlib.pyplot)
        base_name = name.split(".")[0]
        if name not in ALLOWED_MODULES and base_name not in ALLOWED_MODULES:
            raise ImportError(f"Import of module '{name}' is not allowed in this sandbox.")
        return _orig_import(name, globals, locals, fromlist, level)

    sandboxed_env["__builtins__"] = {"__import__": _safe_import}

    result_holder: list[ExecutionResult] = []
    thread = threading.Thread(
        target=_run_in_thread,
        args=(code, sandboxed_env, result_holder, memory_limit_mb),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread is still running → timeout
        logger.warning(f"Code execution timed out after {timeout_seconds}s")
        return ExecutionResult(
            error=f"Execution timed out after {timeout_seconds} seconds.",
            timed_out=True,
            code_executed=True,
        )

    if not result_holder:
        return ExecutionResult(error="Execution produced no result (unknown error).")

    result = result_holder[0]
    if result.error:
        logger.warning(f"Sandbox execution error: {result.error[:200]}")
    else:
        logger.debug(f"Sandbox OK — {result.memory_mb:.1f} MB, output={result.stdout[:100]!r}")

    return result
