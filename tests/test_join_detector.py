"""
tests/test_join_detector.py
----------------------------
Unit tests for core.join_detector cross-file join detection.
"""

import pandas as pd
import pytest

from core.join_detector import detect_joins, format_join_hints


def _make_dfs():
    employees = pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "name":        ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "dept":        ["HR", "Eng", "HR", "Fin", "Eng"],
    })
    salaries = pd.DataFrame({
        "emp_id":    [1, 2, 3, 4, 5],
        "salary":    [50000, 80000, 55000, 90000, 75000],
        "dept":      ["HR", "Eng", "HR", "Fin", "Eng"],
    })
    return {"employees": employees, "salaries": salaries}


# ── Basic detection ───────────────────────────────────────────────────────────

def test_exact_column_match_detected():
    dfs = _make_dfs()
    suggestions = detect_joins(dfs)
    exact = [s for s in suggestions if s.match_type == "exact"]
    # 'dept' appears in both with the same name
    assert any(s.left_col == "dept" and s.right_col == "dept" for s in exact)


def test_fuzzy_match_detected():
    dfs = _make_dfs()
    suggestions = detect_joins(dfs)
    # employee_id ↔ emp_id should be detected as fuzzy match
    fuzzy = [s for s in suggestions if s.match_type in ("exact", "fuzzy")]
    cols = [(s.left_col, s.right_col) for s in fuzzy]
    assert any(
        ("employee_id" in pair and "emp_id" in pair)
        for pair in cols
    )


def test_no_join_for_single_df():
    dfs = {"only": pd.DataFrame({"a": [1, 2, 3]})}
    suggestions = detect_joins(dfs)
    assert suggestions == []


def test_confidence_sorted():
    dfs = _make_dfs()
    suggestions = detect_joins(dfs)
    if len(suggestions) > 1:
        for i in range(len(suggestions) - 1):
            assert suggestions[i].confidence >= suggestions[i + 1].confidence


def test_cardinality_label():
    dfs = {
        "a": pd.DataFrame({"id": [1, 2, 3, 4, 5]}),
        "b": pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]}),
    }
    suggestions = detect_joins(dfs)
    id_match = next((s for s in suggestions if s.left_col == "id"), None)
    assert id_match is not None
    assert id_match.cardinality in ("1:1", "1:N", "N:1", "N:M")


def test_merge_example_string():
    dfs = _make_dfs()
    suggestions = detect_joins(dfs)
    if suggestions:
        example = suggestions[0].merge_example()
        assert ".merge(" in example


def test_format_join_hints_nonempty():
    dfs = _make_dfs()
    suggestions = detect_joins(dfs)
    hint = format_join_hints(suggestions)
    assert "JOIN" in hint or "join" in hint


def test_format_join_hints_empty():
    hint = format_join_hints([])
    assert hint == ""


def test_max_suggestions_cap():
    # Create many columns to produce lots of suggestions
    cols_a = {f"col{i}": list(range(5)) for i in range(20)}
    cols_b = {f"col{i}": list(range(5)) for i in range(20)}
    dfs = {"a": pd.DataFrame(cols_a), "b": pd.DataFrame(cols_b)}
    suggestions = detect_joins(dfs, max_suggestions=5)
    assert len(suggestions) <= 5
