"""
core/code_validator.py
-----------------------
AST-based static analysis of LLM-generated code BEFORE execution.

Checks:
  1. Code is syntactically valid Python (ast.parse)
  2. No import of forbidden modules (os, sys, subprocess, socket, shutil, …)
  3. No use of forbidden built-in calls (open, eval, exec nested, compile, …)
  4. No __dunder__ attribute access that could escape the sandbox
  5. No infinite-loop constructs (while True without break — heuristic)

Usage:
    from core.code_validator import validate_code, CodeValidationError
    issues = validate_code(code_str)
    if issues:
        raise CodeValidationError(issues)
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import List

from core.logging_config import get_logger

logger = get_logger(__name__)


# ── Blocked modules ───────────────────────────────────────────────────────────

_BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "builtins", "ctypes", "cffi", "pickle", "shelve",
    "tempfile", "glob", "fnmatch", "signal", "multiprocessing",
    "threading", "concurrent", "asyncio", "inspect", "gc",
    "weakref", "copy_reg", "atexit", "site", "sysconfig",
    "platform", "distutils", "pkg_resources", "setuptools",
})

# Blocked built-in call names (as Name nodes inside Call nodes)
_BLOCKED_CALLS: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__",
    "breakpoint", "input", "memoryview",
    "getattr", "setattr", "delattr",
})

# Attribute names that are suspicious dunder accesses
_BLOCKED_ATTRS: frozenset[str] = frozenset({
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__builtins__", "__code__", "__func__", "__self__",
    "__dict__", "__mro__",
})


# ── Exception ─────────────────────────────────────────────────────────────────

class CodeValidationError(Exception):
    """Raised when LLM-generated code fails static analysis."""

    def __init__(self, issues: List[str]) -> None:
        self.issues = issues
        super().__init__("Code validation failed:\n" + "\n".join(f"  • {i}" for i in issues))


# ── AST visitor ───────────────────────────────────────────────────────────────

class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[str] = []

    # ── Import checks ─────────────────────────────────────────────────────────

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in _BLOCKED_MODULES:
                self.issues.append(
                    f"Line {node.lineno}: forbidden import '{alias.name}'"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        top = module.split(".")[0]
        if top in _BLOCKED_MODULES:
            self.issues.append(
                f"Line {node.lineno}: forbidden 'from {module} import …'"
            )
        self.generic_visit(node)

    # ── Call checks ───────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call) -> None:
        # Direct name call: open(...), eval(...), etc.
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            self.issues.append(
                f"Line {node.lineno}: forbidden call '{node.func.id}()'"
            )
        # Attribute call: something.__import__(...)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in _BLOCKED_CALLS:
                self.issues.append(
                    f"Line {node.lineno}: forbidden method call '.{node.func.attr}()'"
                )
        self.generic_visit(node)

    # ── Attribute access checks ───────────────────────────────────────────────

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _BLOCKED_ATTRS:
            self.issues.append(
                f"Line {node.lineno}: suspicious dunder attribute access '.{node.attr}'"
            )
        self.generic_visit(node)

    # ── Standalone dunder name references ──────────────────────────────────────

    def visit_Name(self, node: ast.Name) -> None:
        # Catch: `b = __builtins__`, `f = __import__`, etc.
        if node.id in _BLOCKED_ATTRS or node.id in {"__import__", "__builtins__", "__build_class__"}:
            self.issues.append(
                f"Line {node.lineno}: suspicious dunder name reference '{node.id}'"
            )
        self.generic_visit(node)

    # ── Heuristic: while True without break ──────────────────────────────────

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # Check if there's a break anywhere in the body
            has_break = any(
                isinstance(n, ast.Break)
                for n in ast.walk(node)
            )
            if not has_break:
                self.issues.append(
                    f"Line {node.lineno}: potential infinite loop (while True without break)"
                )
        self.generic_visit(node)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_code(code: str) -> List[str]:
    """
    Statically analyse *code* and return a list of security/safety issues.
    Returns an empty list if the code passes all checks.

    Does NOT execute the code.
    """
    # Step 1: syntax check
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return [f"SyntaxError at line {exc.lineno}: {exc.msg}"]

    # Step 2: security walk
    visitor = _SecurityVisitor()
    visitor.visit(tree)

    if visitor.issues:
        logger.warning(f"Code validation found {len(visitor.issues)} issue(s): {visitor.issues[:3]}")
    else:
        logger.debug("Code validation passed.")

    return visitor.issues


def assert_safe(code: str) -> None:
    """
    Validate *code* and raise `CodeValidationError` if any issues are found.
    """
    issues = validate_code(code)
    if issues:
        raise CodeValidationError(issues)
