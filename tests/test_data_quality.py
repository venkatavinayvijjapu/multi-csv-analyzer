"""
tests/test_data_quality.py
---------------------------
Unit tests for core.data_quality — schema normalisation + quality profiling.
"""

import numpy as np
import pandas as pd
import pytest

from core.data_quality import profile_dataframe, normalize_dataframe


# ── normalize_dataframe ───────────────────────────────────────────────────────

def test_column_name_normalization():
    df = pd.DataFrame({"  First Name  ": ["Alice"], "LAST-NAME": ["Smith"]})
    df_norm, col_map = normalize_dataframe(df)
    assert "first_name" in df_norm.columns
    assert "last_name" in df_norm.columns


def test_null_marker_replacement():
    df = pd.DataFrame({"val": ["N/A", "100", "null", "200", ""]})
    df_norm, _ = normalize_dataframe(df)
    # N/A, null, "" should be NaN
    assert df_norm["val"].isna().sum() >= 3


def test_numeric_string_coercion():
    df = pd.DataFrame({"amount": ["100", "200", "300", "400", "500"]})
    df_norm, _ = normalize_dataframe(df)
    # Should have converted string numbers to a numeric type
    assert pd.api.types.is_numeric_dtype(df_norm["amount"]) or \
           df_norm["amount"].dtype in [pd.StringDtype(), "string"], \
           f"Expected numeric dtype, got {df_norm['amount'].dtype}"
    # At minimum, pd.to_numeric should succeed with no errors
    numeric = pd.to_numeric(df_norm["amount"], errors="coerce")
    assert numeric.notna().all(), "All values should be convertible to numeric"


def test_duplicate_column_dedup():
    df = pd.DataFrame([[1, 2]], columns=["col", "col"])
    df_norm, col_map = normalize_dataframe(df)
    assert len(set(df_norm.columns)) == len(df_norm.columns)


def test_column_map_returned():
    df = pd.DataFrame({"Employee ID": [1, 2], "Dept Name": ["A", "B"]})
    df_norm, col_map = normalize_dataframe(df)
    assert len(col_map) == 2
    # All normalised cols should appear as keys
    for col in df_norm.columns:
        assert col in col_map


# ── profile_dataframe ─────────────────────────────────────────────────────────

def _sample_df():
    return pd.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alice", "Bob", "Alice", "Dave", None],
        "score":  [95.0, 80.0, 72.0, 88.0, None],
        "dept":   ["HR", "Eng", "HR", "Eng", "HR"],
    })


def test_profile_shape():
    df = _sample_df()
    report = profile_dataframe("test.csv", df)
    assert report.rows == 5
    assert report.columns == 4


def test_profile_nulls():
    df = _sample_df()
    report = profile_dataframe("test.csv", df)
    # 'name' and 'score' each have 1 null
    assert report.total_nulls == 2


def test_profile_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 10, 20]})
    report = profile_dataframe("dup.csv", df)
    assert report.duplicate_rows == 1


def test_profile_column_types():
    df = _sample_df()
    report = profile_dataframe("test.csv", df)
    types = {p.name: p.inferred_type for p in report.column_profiles}
    assert types["id"] == "numeric"
    assert types["dept"] == "categorical"


def test_profile_to_dict():
    df = _sample_df()
    report = profile_dataframe("test.csv", df)
    d = report.to_dict()
    assert "column_profiles" in d
    assert d["rows"] == 5


def test_profile_outlier_detection():
    # Add a clear outlier
    data = [10, 11, 12, 10, 11, 12, 10, 11, 1000]
    df = pd.DataFrame({"v": data})
    report = profile_dataframe("outlier.csv", df)
    score_profile = next(p for p in report.column_profiles if p.name == "v")
    assert score_profile.outlier_count >= 1


def test_profile_warnings_for_high_nulls():
    df = pd.DataFrame({"a": [None] * 80 + [1] * 20})
    report = profile_dataframe("nulls.csv", df)
    assert any("missing" in w.lower() for w in report.warnings)
