"""
core/join_detector.py
----------------------
Cross-file join key detection with fuzzy name matching and value-overlap scoring.

Replaces the simple `_find_join_keys` function in analysis.py with a richer
implementation that supports:
  • Exact column name matching
  • Fuzzy / alias matching  (emp_id ↔ employee_id, dept ↔ department)
  • Value-overlap scoring   (sample values from both columns to confirm semantics)
  • Cardinality analysis    (1:1, 1:N, N:M relationship detection)

Public API:
    detect_joins(dfs)  →  List[JoinSuggestion]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.logging_config import get_logger

logger = get_logger(__name__)


# ── Common alias pairs ────────────────────────────────────────────────────────
# Each tuple is a set of synonymous column name fragments.
_ALIAS_GROUPS: List[frozenset] = [
    frozenset({"emp_id", "employee_id", "empid", "staff_id", "worker_id", "person_id"}),
    frozenset({"dept", "department", "dept_id", "department_id", "division"}),
    frozenset({"id", "uid", "uuid", "user_id", "userid"}),
    frozenset({"name", "full_name", "fullname", "employee_name", "emp_name"}),
    frozenset({"email", "email_id", "mail", "email_address"}),
    frozenset({"date", "dt", "datetime", "timestamp", "created_at", "updated_at"}),
    frozenset({"product_id", "prod_id", "item_id", "sku", "product_code"}),
    frozenset({"customer_id", "cust_id", "client_id", "account_id"}),
    frozenset({"order_id", "transaction_id", "txn_id", "invoice_id"}),
    frozenset({"country", "country_code", "nation"}),
    frozenset({"region", "state", "province", "territory"}),
]


def _normalize_col(col: str) -> str:
    """Lowercase, strip, alphanumeric only."""
    return re.sub(r"[^a-z0-9]", "_", str(col).strip().lower()).strip("_")


def _fuzzy_match(a: str, b: str) -> bool:
    """
    Return True if two column names are likely aliases.
    Checks: exact match, substring containment, alias group membership.
    """
    na, nb = _normalize_col(a), _normalize_col(b)
    if na == nb:
        return True
    # Substring containment (one is prefix/suffix of the other)
    if na in nb or nb in na:
        return True
    # Alias group lookup
    for group in _ALIAS_GROUPS:
        if na in group and nb in group:
            return True
    return False


def _value_overlap_score(
    s1: pd.Series, s2: pd.Series, sample_size: int = 200
) -> float:
    """
    Fraction of s1's sample values that also appear in s2.
    Returns 0.0–1.0.
    """
    try:
        v1 = set(s1.dropna().astype(str).sample(min(sample_size, len(s1)), random_state=42))
        v2 = set(s2.dropna().astype(str).sample(min(sample_size, len(s2)), random_state=42))
        if not v1:
            return 0.0
        return len(v1 & v2) / len(v1)
    except Exception:
        return 0.0


def _cardinality(s1: pd.Series, s2: pd.Series) -> str:
    """
    Rough cardinality label: '1:1', '1:N', 'N:1', or 'N:M'.
    """
    u1, u2 = s1.nunique(), s2.nunique()
    total1, total2 = s1.count(), s2.count()
    if u1 == 0 or u2 == 0:
        return "N/A"
    r1 = u1 / total1 if total1 else 0
    r2 = u2 / total2 if total2 else 0
    if r1 > 0.95 and r2 > 0.95:
        return "1:1"
    if r1 > 0.95:
        return "1:N"
    if r2 > 0.95:
        return "N:1"
    return "N:M"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class JoinSuggestion:
    left_df: str
    left_col: str
    right_df: str
    right_col: str
    match_type: str          # "exact" | "fuzzy" | "value_overlap"
    overlap_score: float     # 0.0–1.0 value overlap
    cardinality: str         # "1:1" | "1:N" | "N:1" | "N:M"
    confidence: float        # combined score 0.0–1.0

    def to_dict(self) -> dict:
        return {
            "left_df": self.left_df,
            "left_col": self.left_col,
            "right_df": self.right_df,
            "right_col": self.right_col,
            "match_type": self.match_type,
            "overlap_score": round(self.overlap_score, 3),
            "cardinality": self.cardinality,
            "confidence": round(self.confidence, 3),
        }

    def merge_example(self) -> str:
        """Return a pd.merge() example string."""
        return (
            f"merged = {self.left_df}.merge({self.right_df}, "
            f"left_on='{self.left_col}', right_on='{self.right_col}', how='left')"
        )

    def to_schema_hint(self) -> str:
        """One-liner hint for LLM prompt injection."""
        return (
            f"  • '{self.left_col}' ({self.left_df}) ↔ '{self.right_col}' ({self.right_df})"
            f"  [confidence={self.confidence:.0%}, cardinality={self.cardinality}]"
        )


# ── Core detection ────────────────────────────────────────────────────────────

def detect_joins(
    dfs: Dict[str, pd.DataFrame],
    min_overlap: float = 0.10,
    max_suggestions: int = 10,
) -> List[JoinSuggestion]:
    """
    Detect potential join keys across all DataFrames in *dfs*.

    Args:
        dfs:             {name: DataFrame} dict.
        min_overlap:     Minimum value-overlap fraction to include a suggestion.
        max_suggestions: Cap on the number of suggestions returned.

    Returns:
        List of JoinSuggestion sorted by confidence descending.
    """
    if len(dfs) < 2:
        return []

    df_names = list(dfs.keys())
    suggestions: List[JoinSuggestion] = []

    for i in range(len(df_names)):
        for j in range(i + 1, len(df_names)):
            name_a, name_b = df_names[i], df_names[j]
            df_a, df_b = dfs[name_a], dfs[name_b]

            for col_a in df_a.columns:
                for col_b in df_b.columns:
                    na, nb = _normalize_col(col_a), _normalize_col(col_b)

                    # Exact match
                    if na == nb:
                        match_type = "exact"
                        name_conf = 1.0
                    elif _fuzzy_match(col_a, col_b):
                        match_type = "fuzzy"
                        name_conf = 0.7
                    else:
                        continue  # fast skip if no name similarity

                    # Value overlap (expensive — only if name matches)
                    overlap = _value_overlap_score(df_a[col_a], df_b[col_b])
                    if overlap < min_overlap and match_type != "exact":
                        continue

                    cardinality = _cardinality(df_a[col_a], df_b[col_b])

                    # Confidence: weighted average of name similarity and value overlap
                    confidence = name_conf * 0.5 + overlap * 0.5

                    suggestions.append(JoinSuggestion(
                        left_df=name_a,
                        left_col=col_a,
                        right_df=name_b,
                        right_col=col_b,
                        match_type=match_type,
                        overlap_score=overlap,
                        cardinality=cardinality,
                        confidence=confidence,
                    ))

    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    result = suggestions[:max_suggestions]

    logger.debug(f"detect_joins: found {len(result)} suggestions across {len(dfs)} DataFrames")
    return result


def format_join_hints(suggestions: List[JoinSuggestion]) -> str:
    """
    Format join suggestions as a compact string for LLM prompt injection.
    """
    if not suggestions:
        return ""
    lines = ["🔗 AUTO-DETECTED JOIN KEYS (columns shared across files):"]
    for s in suggestions:
        lines.append(s.to_schema_hint())
    lines.append("  Use pd.merge() or pd.concat() to combine these datasets.")
    return "\n".join(lines)
