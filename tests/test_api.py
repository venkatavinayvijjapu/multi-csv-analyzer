"""
tests/test_api.py
-----------------
Integration tests for all FastAPI endpoints using TestClient.

These tests do NOT make real LLM calls — the query endpoint is tested
with pre-seeded session data and lightweight mock patching where needed.
"""

import io
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Import the app
from api import app

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_session_id() -> str:
    return str(uuid.uuid4())


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "employee_id": [1, 2, 3],
        "name":        ["Alice", "Bob", "Carol"],
        "salary":      [50000, 80000, 60000],
        "dept":        ["HR", "Eng", "HR"],
    })


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "active_sessions" in data
    assert "limits" in data


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_csv():
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    resp = client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("employees.csv", csv_data, "text/csv"))],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["files"]) >= 1
    assert data["files"][0]["rows"] == 3


def test_upload_unsupported_format():
    sid = _make_session_id()
    resp = client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("data.txt", b"hello", "text/plain"))],
    )
    assert resp.status_code == 400


def test_upload_file_size_limit(monkeypatch):
    """Simulate a file that exceeds the size limit."""
    import api as api_module
    monkeypatch.setattr(api_module, "MAX_FILE_SIZE_MB", 0)  # set limit to 0 MB
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    resp = client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("data.csv", csv_data, "text/csv"))],
    )
    assert resp.status_code == 400


# ── List files ────────────────────────────────────────────────────────────────

def test_list_files_empty():
    sid = _make_session_id()
    resp = client.get(f"/api/files?session_id={sid}")
    assert resp.status_code == 200
    assert resp.json()["files"] == []


def test_list_files_after_upload():
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("emp.csv", csv_data, "text/csv"))],
    )
    resp = client.get(f"/api/files?session_id={sid}")
    assert resp.status_code == 200
    files = resp.json()["files"]
    assert len(files) == 1
    assert files[0]["name"] == "emp.csv"


# ── Clear / remove ────────────────────────────────────────────────────────────

def test_clear_session():
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("emp.csv", csv_data, "text/csv"))],
    )
    resp = client.delete(f"/api/clear?session_id={sid}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    # Verify cleared
    resp2 = client.get(f"/api/files?session_id={sid}")
    assert resp2.json()["files"] == []


def test_remove_single_file():
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("emp.csv", csv_data, "text/csv"))],
    )
    resp = client.delete(f"/api/file?session_id={sid}&filename=emp.csv")
    assert resp.status_code == 200
    assert resp.json()["files"] == []


# ── Query (no files) ──────────────────────────────────────────────────────────

def test_query_without_files():
    resp = client.post(
        "/api/query",
        json={"session_id": _make_session_id(), "query": "what is the total?"},
    )
    assert resp.status_code == 400


def test_query_empty_query():
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("emp.csv", csv_data, "text/csv"))],
    )
    resp = client.post("/api/query", json={"session_id": sid, "query": "   "})
    assert resp.status_code == 400


# ── Session info ──────────────────────────────────────────────────────────────

def test_session_info():
    sid = _make_session_id()
    csv_data = _csv_bytes(_sample_df())
    client.post(
        "/api/upload",
        data={"session_id": sid},
        files=[("files", ("emp.csv", csv_data, "text/csv"))],
    )
    resp = client.get(f"/api/session/info?session_id={sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["exists"] is True
    assert data["file_count"] == 1


def test_session_extend():
    sid = _make_session_id()
    resp = client.post(f"/api/session/extend?session_id={sid}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


# ── History ───────────────────────────────────────────────────────────────────

def test_clear_history():
    sid = _make_session_id()
    import session_store
    session_store.append_message(sid, "user", "hello")
    resp = client.delete(f"/api/history?session_id={sid}")
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    history = session_store.get_conversation_history(sid)
    assert history == []


def test_get_history():
    sid = _make_session_id()
    import session_store
    session_store.append_message(sid, "user", "test question")
    session_store.append_message(sid, "assistant", "test answer")
    resp = client.get(f"/api/history?session_id={sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) == 2


# ── Provenance ────────────────────────────────────────────────────────────────

def test_provenance_empty():
    sid = _make_session_id()
    resp = client.get(f"/api/provenance?session_id={sid}")
    assert resp.status_code == 200
    assert resp.json()["provenance"] == []


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_endpoint():
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_queries" in data


# ── Prompts ───────────────────────────────────────────────────────────────────

def test_prompts_endpoint():
    resp = client.get("/api/prompts")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompts" in data
    assert len(data["prompts"]) > 0


# ── Request ID header ─────────────────────────────────────────────────────────

def test_request_id_in_response():
    resp = client.get("/api/health")
    assert "x-request-id" in resp.headers


def test_custom_request_id_echoed():
    custom_rid = "test-rid-123"
    resp = client.get("/api/health", headers={"X-Request-ID": custom_rid})
    assert resp.headers.get("x-request-id") == custom_rid
