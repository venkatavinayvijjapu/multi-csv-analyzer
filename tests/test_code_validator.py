"""
tests/test_code_validator.py
-----------------------------
Unit tests for core.code_validator AST-based static analysis.
"""

import pytest
from core.code_validator import validate_code, assert_safe, CodeValidationError


# ── Safe code ─────────────────────────────────────────────────────────────────

def test_safe_pandas_code():
    code = """
import pandas as pd
df = pd.DataFrame({'a': [1,2,3]})
print(df['a'].sum())
"""
    issues = validate_code(code)
    assert issues == []


def test_safe_math_code():
    code = "import math\nprint(math.sqrt(16))"
    issues = validate_code(code)
    assert issues == []


# ── Blocked imports ───────────────────────────────────────────────────────────

def test_blocks_os_import():
    code = "import os\nprint(os.listdir('/'))"
    issues = validate_code(code)
    assert any("os" in i for i in issues)


def test_blocks_subprocess():
    code = "import subprocess\nsubprocess.run(['ls'])"
    issues = validate_code(code)
    assert any("subprocess" in i for i in issues)


def test_blocks_sys():
    code = "import sys\nsys.exit()"
    issues = validate_code(code)
    assert any("sys" in i for i in issues)


def test_blocks_from_import():
    code = "from os import path\npath.exists('/')"
    issues = validate_code(code)
    assert any("os" in i for i in issues)


# ── Blocked calls ─────────────────────────────────────────────────────────────

def test_blocks_open():
    code = "open('/etc/passwd', 'r')"
    issues = validate_code(code)
    assert any("open" in i for i in issues)


def test_blocks_eval():
    code = "eval('1 + 1')"
    issues = validate_code(code)
    assert any("eval" in i for i in issues)


def test_blocks_exec_call():
    code = "exec('import os')"
    issues = validate_code(code)
    assert any("exec" in i for i in issues)


def test_blocks_compile():
    code = "compile('x=1', '', 'exec')"
    issues = validate_code(code)
    assert any("compile" in i for i in issues)


# ── Blocked dunder attributes ─────────────────────────────────────────────────

def test_blocks_dunder_subclasses():
    code = "x = ().__class__.__subclasses__()"
    issues = validate_code(code)
    assert any("__subclasses__" in i or "__class__" in i for i in issues)


def test_blocks_builtins_access():
    code = "b = __builtins__"
    issues = validate_code(code)
    assert any("__builtins__" in i for i in issues)


# ── Syntax error ──────────────────────────────────────────────────────────────

def test_syntax_error():
    code = "def foo(:"
    issues = validate_code(code)
    assert any("SyntaxError" in i for i in issues)


# ── assert_safe helper ────────────────────────────────────────────────────────

def test_assert_safe_raises():
    with pytest.raises(CodeValidationError):
        assert_safe("import os")


def test_assert_safe_passes():
    assert_safe("print('hello')")  # should not raise
