"""
session_store.py
----------------
Simple in-memory session store mapping session_id → {filename: DataFrame}.
Replaces Streamlit's st.session_state for the new SPA architecture.
"""

from typing import Dict
import pandas as pd

# Global session registry: {session_id: {filename: DataFrame}}
_store: Dict[str, Dict[str, pd.DataFrame]] = {}


def get_session(session_id: str) -> Dict[str, pd.DataFrame]:
    """Return the DataFrames dict for a session. Never raises."""
    return _store.get(session_id, {})


def set_file(session_id: str, filename: str, df: pd.DataFrame) -> None:
    """Store a DataFrame under the given session and filename."""
    if session_id not in _store:
        _store[session_id] = {}
    _store[session_id][filename] = df


def remove_file(session_id: str, filename: str) -> None:
    """Remove a single file from a session."""
    if session_id in _store and filename in _store[session_id]:
        del _store[session_id][filename]


def clear_session(session_id: str) -> None:
    """Wipe all files for a session."""
    _store.pop(session_id, None)


def list_files(session_id: str) -> list[dict]:
    """Return file metadata list for a session."""
    dfs = get_session(session_id)
    result = []
    for name, df in dfs.items():
        num_cols = df.select_dtypes("number").columns.tolist()
        result.append({
            "name": name,
            "rows": len(df),
            "columns": len(df.columns),
            "col_names": df.columns.tolist(),
            "numeric_cols": num_cols,
            "null_count": int(df.isnull().sum().sum()),
        })
    return result
