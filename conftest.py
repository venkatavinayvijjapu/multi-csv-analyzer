"""
conftest.py
-----------
Shared pytest fixtures for DataLens AI test suite.

Fixtures:
    sample_df       — a basic 5-row DataFrame for reuse across tests
    two_dfs         — dict of two related DataFrames (for join tests)
    session_id      — a fresh UUID session_id (cleaned up after each test)
    seeded_session  — a session_id with sample_df already stored
    test_client     — FastAPI TestClient instance
"""

from __future__ import annotations

import io
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Data fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5],
        "name":        ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "salary":      [50000.0, 80000.0, 60000.0, 90000.0, 70000.0],
        "dept":        ["HR", "Eng", "HR", "Fin", "Eng"],
        "joined":      ["2020-01-15", "2019-06-01", "2021-03-22", "2018-11-05", "2022-07-30"],
    })


@pytest.fixture
def two_dfs(sample_df) -> dict:
    salaries = pd.DataFrame({
        "emp_id":    [1, 2, 3, 4, 5],
        "base_pay":  [50000, 80000, 60000, 90000, 70000],
        "bonus_pct": [10, 15, 8, 20, 12],
    })
    return {"employees": sample_df, "salaries": salaries}


# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def session_id() -> str:
    """Fresh UUID session ID, cleaned up after each test."""
    import session_store
    sid = str(uuid.uuid4())
    yield sid
    session_store.clear_session(sid)


@pytest.fixture
def seeded_session(session_id, sample_df) -> str:
    """Session with sample_df pre-loaded."""
    import session_store
    session_store.set_file(session_id, "employees.csv", sample_df)
    return session_id


# ── API client ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """FastAPI TestClient — shared across the test session."""
    from api import app
    with TestClient(app) as client:
        yield client


# ── CSV upload helper ─────────────────────────────────────────────────────────

@pytest.fixture
def csv_bytes(sample_df) -> bytes:
    """sample_df serialised as CSV bytes — for upload endpoint tests."""
    buf = io.BytesIO()
    sample_df.to_csv(buf, index=False)
    return buf.getvalue()
