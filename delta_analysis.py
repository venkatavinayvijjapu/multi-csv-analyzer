"""
delta_analysis.py
-----------------
STRUCTURAL-ONLY fast path — handles queries about data SHAPE/SCHEMA
without needing the LLM (describe, nulls, duplicates, correlation).

All DATA queries (totals, averages, trends, comparisons, insights…)
are intentionally NOT handled here. They fall through to (None, None)
and are answered by the LLM via pandas code generation in analysis.py.

This design means the app can handle ANY arbitrary question through
LLM code generation, with no keyword-based column/intent matching.

Upgrade note (v4):
  - _handle_describe now includes data-quality warnings if a
    quality_report is available in the session store.
  - All handlers accept an optional session_id to look up reports.
"""

import re
import pandas as pd
from typing import Optional, Tuple, Dict, Any


# ── Only structural triggers (describe the data structure, not its values) ──
_DESCRIBE_WORDS    = {"describe", "summary", "statistics", "overview", "info",
                      "schema", "explore", "columns", "fields", "structure", "about"}
_NULL_WORDS        = {"null", "missing", "nan", "blank", "incomplete", "empty"}
_DUPLICATE_WORDS   = {"duplicate", "duplicates", "repeated", "repeat"}
_CORR_WORDS        = {"correlation", "correlate", "correlated", "relationship", "related"}


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


# ── Structural handlers ──────────────────────────────────────────────────────

def _handle_describe(dfs: Dict[str, pd.DataFrame], session_id: str = "") -> Tuple[str, None]:
    lines = []
    for name, df in dfs.items():
        lines.append(f"### 📄 {name}")
        lines.append(f"**Shape:** {len(df):,} rows × {len(df.columns)} columns")
        lines.append(f"**Columns:** {', '.join(f'`{c}`' for c in df.columns)}")

        # ── Numeric stats (clean — no NaN rows) ──
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            stats = num_df.describe().round(2)
            lines.append("\n**Numeric statistics:**")
            lines.append(f"```\n{stats.to_string()}\n```")

        # ── Categorical summary ──
        cat_df = df.select_dtypes(exclude="number")
        if not cat_df.empty:
            lines.append("\n**Categorical columns:**")
            for col in cat_df.columns:
                n_unique = df[col].nunique()
                vc = df[col].value_counts()
                top_val = vc.index[0] if len(vc) else "N/A"
                top_pct = f"{vc.iloc[0]/len(df)*100:.0f}%" if len(vc) else ""
                lines.append(
                    f"  - `{col}`: {n_unique} unique values, "
                    f"most common → **{top_val}** ({top_pct})"
                )

        # ── Missing values ──
        nulls = df.isnull().sum()
        missing = nulls[nulls > 0]
        if missing.empty:
            lines.append("\n✅ **No missing values**")
        else:
            lines.append("\n⚠️ **Missing values:**")
            for col, cnt in missing.items():
                pct = cnt / len(df) * 100
                lines.append(f"  - `{col}`: {cnt:,} missing ({pct:.1f}%)")

        # ── Quality warnings from stored report ──
        if session_id:
            try:
                import session_store
                report = session_store.get_quality_report(session_id, name)
                if report and report.get("warnings"):
                    lines.append("\n⚠️ **Data quality warnings:**")
                    for w in report["warnings"][:5]:
                        lines.append(f"  - {w}")
            except Exception:
                pass

        lines.append("")
    return "\n".join(lines), None


def _handle_null_check(dfs: Dict[str, pd.DataFrame]) -> Tuple[str, None]:
    lines = ["### 🔍 Missing Value Report"]
    for name, df in dfs.items():
        null_counts = df.isnull().sum()
        lines.append(f"\n**{name}** — {int(null_counts.sum()):,} total missing values")
        nulls = null_counts[null_counts > 0]
        if nulls.empty:
            lines.append("  ✅ No missing values!")
        else:
            for col, cnt in nulls.items():
                pct = cnt / len(df) * 100
                lines.append(f"  - `{col}`: {cnt:,} ({pct:.1f}%)")
    return "\n".join(lines), None


def _handle_duplicate_check(dfs: Dict[str, pd.DataFrame]) -> Tuple[str, None]:
    lines = ["### 🔁 Duplicate Rows Report"]
    for name, df in dfs.items():
        dup_count = int(df.duplicated().sum())
        icon = "✅" if dup_count == 0 else "⚠️"
        lines.append(f"- **{name}**: {icon} {dup_count:,} duplicate rows out of {len(df):,} total")
    return "\n".join(lines), None


def _handle_correlation(dfs: Dict[str, pd.DataFrame]) -> Tuple[str, None]:
    lines = ["### 🔗 Correlation Analysis"]
    for name, df in dfs.items():
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            lines.append(f"**{name}**: Need ≥ 2 numeric columns for correlation.")
            continue
        corr = num_df.corr().round(3)
        lines.append(f"\n**{name}** correlation matrix:\n```\n{corr.to_string()}\n```")
    return "\n".join(lines), None


# ── Public entry point ───────────────────────────────────────────────────────

def check_delta(
    query: str,
    dfs: Dict[str, pd.DataFrame],
    session_id: str = "",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Fast path for purely structural queries (data shape/schema).
    Returns (None, None) for ALL data-value queries so the LLM handles them.

    Structural queries handled here (no LLM needed):
      - describe / schema / overview
      - missing / null / blank values
      - duplicate rows
      - correlation matrix

    Everything else (totals, averages, max, min, trends, comparisons,
    insights, filters, ranked lists, etc.) → returns (None, None) → LLM.
    """
    if not dfs:
        return "No datasets uploaded yet. Please upload files first.", None

    toks = _tokens(query)

    if toks & _DESCRIBE_WORDS:
        return _handle_describe(dfs, session_id=session_id)

    if toks & _NULL_WORDS:
        return _handle_null_check(dfs)

    if toks & _DUPLICATE_WORDS:
        return _handle_duplicate_check(dfs)

    if toks & _CORR_WORDS:
        return _handle_correlation(dfs)

    # All data-value questions → LLM generates pandas code
    return None, None
