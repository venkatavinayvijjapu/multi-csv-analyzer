"""
tests/test_session_store.py
----------------------------
Unit tests for session_store — TTL, conversation history, quality reports.
"""

import time
import pandas as pd
import pytest

import session_store


def _df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


# ── Basic file ops ────────────────────────────────────────────────────────────

def test_set_and_get_file():
    sid = "test-set-get"
    session_store.set_file(sid, "data.csv", _df())
    dfs = session_store.get_session(sid)
    assert "data.csv" in dfs
    assert len(dfs["data.csv"]) == 3
    session_store.clear_session(sid)


def test_remove_file():
    sid = "test-remove"
    session_store.set_file(sid, "a.csv", _df())
    session_store.set_file(sid, "b.csv", _df())
    session_store.remove_file(sid, "a.csv")
    dfs = session_store.get_session(sid)
    assert "a.csv" not in dfs
    assert "b.csv" in dfs
    session_store.clear_session(sid)


def test_clear_session():
    sid = "test-clear"
    session_store.set_file(sid, "data.csv", _df())
    session_store.clear_session(sid)
    dfs = session_store.get_session(sid)
    assert dfs == {}


def test_list_files_metadata():
    sid = "test-list"
    session_store.set_file(sid, "data.csv", _df())
    files = session_store.list_files(sid)
    assert len(files) == 1
    assert files[0]["name"] == "data.csv"
    assert files[0]["rows"] == 3
    session_store.clear_session(sid)


# ── Quality report ────────────────────────────────────────────────────────────

def test_quality_report_stored():
    sid = "test-quality"
    report = {"rows": 3, "warnings": []}
    session_store.set_file(sid, "data.csv", _df(), quality_report=report)
    retrieved = session_store.get_quality_report(sid, "data.csv")
    assert retrieved == report
    session_store.clear_session(sid)


def test_quality_report_in_list_files():
    sid = "test-quality-list"
    report = {"rows": 3, "quality_score": 95}
    session_store.set_file(sid, "data.csv", _df(), quality_report=report)
    files = session_store.list_files(sid)
    assert "quality" in files[0]
    session_store.clear_session(sid)


# ── Conversation history ──────────────────────────────────────────────────────

def test_append_and_get_history():
    sid = "test-history"
    session_store.append_message(sid, "user", "hello")
    session_store.append_message(sid, "assistant", "hi there")
    history = session_store.get_conversation_history(sid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["content"] == "hi there"
    session_store.clear_session(sid)


def test_clear_conversation_history():
    sid = "test-hist-clear"
    session_store.append_message(sid, "user", "test")
    session_store.clear_conversation_history(sid)
    history = session_store.get_conversation_history(sid)
    assert history == []
    session_store.clear_session(sid)


def test_history_rolling_cap():
    sid = "test-hist-cap"
    # Write more than MAX_CONVERSATION_TURNS messages
    for i in range(60):
        session_store.append_message(sid, "user", f"msg {i}")
    history = session_store.get_conversation_history(sid)
    assert len(history) <= session_store.MAX_CONVERSATION_TURNS
    session_store.clear_session(sid)


# ── Session info ──────────────────────────────────────────────────────────────

def test_session_info_exists():
    sid = "test-info"
    session_store.set_file(sid, "data.csv", _df())
    info = session_store.session_info(sid)
    assert info["exists"] is True
    assert info["file_count"] == 1
    assert "ttl_remaining_seconds" in info
    session_store.clear_session(sid)


def test_session_info_nonexistent():
    info = session_store.session_info("nonexistent-session-xyz")
    assert info["exists"] is False


def test_extend_session():
    sid = "test-extend"
    session_store.set_file(sid, "data.csv", _df())
    time.sleep(0.05)
    before = session_store.session_info(sid)["idle_seconds"]
    session_store.extend_session(sid)
    after = session_store.session_info(sid)["idle_seconds"]
    # idle should have reset to ~0
    assert after <= 1
    session_store.clear_session(sid)


# ── Active session count ──────────────────────────────────────────────────────

def test_active_session_count():
    sid1, sid2 = "cnt-a", "cnt-b"
    session_store.set_file(sid1, "a.csv", _df())
    session_store.set_file(sid2, "b.csv", _df())
    count = session_store.active_session_count()
    assert count >= 2
    session_store.clear_session(sid1)
    session_store.clear_session(sid2)
