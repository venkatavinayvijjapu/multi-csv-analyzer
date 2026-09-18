"""
tests/test_sandbox.py
---------------------
Unit tests for the secure code execution sandbox (core.sandbox).
"""

import pytest
from core.sandbox import run_code


# ── Helpers ───────────────────────────────────────────────────────────────────

def _env():
    import pandas as pd
    import numpy as np
    return {"pd": pd, "np": np}


# ── Success cases ─────────────────────────────────────────────────────────────

def test_basic_print():
    result = run_code('print("hello")', _env())
    assert result.success
    assert result.stdout == "hello"


def test_pandas_operation():
    import pandas as pd
    env = _env()
    env["df"] = pd.DataFrame({"a": [1, 2, 3]})
    result = run_code("print(df['a'].sum())", env)
    assert result.success
    assert "6" in result.stdout


def test_no_output_still_success():
    result = run_code("x = 1 + 1", _env())
    assert result.code_executed
    assert not result.error


# ── Blocked calls ─────────────────────────────────────────────────────────────

def test_open_blocked():
    result = run_code("open('/etc/passwd')", _env())
    assert result.error is not None


def test_import_os_blocked():
    result = run_code("import os; os.listdir('/')", _env())
    assert result.error is not None


def test_eval_blocked():
    result = run_code("eval('1+1')", _env())
    assert result.error is not None


def test_exec_nested_blocked():
    result = run_code("exec('x=1')", _env())
    assert result.error is not None


# ── Timeout ───────────────────────────────────────────────────────────────────

def test_timeout():
    code = "while True: pass"
    result = run_code(code, _env(), timeout_seconds=2)
    assert result.timed_out


# ── Safe operations ───────────────────────────────────────────────────────────

def test_math_allowed():
    result = run_code("import math; print(math.pi)", _env())
    assert result.success
    assert "3.14" in result.stdout


def test_list_comprehension():
    result = run_code("print([x*2 for x in range(5)])", _env())
    assert result.success
    assert "[0, 2, 4, 6, 8]" in result.stdout


def test_string_operations():
    result = run_code("print('hello'.upper())", _env())
    assert result.success
    assert "HELLO" in result.stdout
