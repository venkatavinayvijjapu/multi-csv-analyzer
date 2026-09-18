"""
core/data_quality.py
---------------------
Data-quality profiling, schema normalisation, and type inference.

Called at upload time (api.py) so every dataset has a quality report
available before any query is run.

Public API:
    profile_dataframe(name, df)  → DataQualityReport
    normalize_dataframe(df)      → (df_normalized, ColumnMap)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.logging_config import get_logger

logger = get_logger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    original_name: str          # before normalisation
    dtype: str
    inferred_type: str          # "numeric" | "categorical" | "datetime" | "text" | "boolean"
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    top_value: Optional[Any]
    top_value_pct: float
    min_value: Optional[Any]
    max_value: Optional[Any]
    mean_value: Optional[float]
    outlier_count: int          # IQR-based for numeric columns
    has_mixed_types: bool       # numeric strings mixed with text?


@dataclass
class DataQualityReport:
    filename: str
    rows: int
    columns: int
    duplicate_rows: int
    duplicate_pct: float
    total_nulls: int
    total_null_pct: float
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Map: normalised_name → original_name
    column_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _safe(v):
            """Convert any numpy scalar to a native Python type for JSON serialization."""
            if v is None:
                return None
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, (float, np.floating)):
                f = float(v)
                return None if (f != f) else f   # NaN → None
            return str(v)

        return {
            "filename": str(self.filename),
            "rows": int(self.rows),
            "columns": int(self.columns),
            "duplicate_rows": int(self.duplicate_rows),
            "duplicate_pct": round(float(self.duplicate_pct), 2),
            "total_nulls": int(self.total_nulls),
            "total_null_pct": round(float(self.total_null_pct), 2),
            "warnings": [str(w) for w in self.warnings],
            "column_map": {str(k): str(v) for k, v in self.column_map.items()},
            "column_profiles": [
                {
                    "name": str(c.name),
                    "dtype": str(c.dtype),
                    "inferred_type": str(c.inferred_type),
                    "null_count": int(c.null_count),
                    "null_pct": round(float(c.null_pct), 2),
                    "unique_count": int(c.unique_count),
                    "outlier_count": int(c.outlier_count),
                    "has_mixed_types": bool(c.has_mixed_types),
                    "top_value": _safe(c.top_value),
                    "top_value_pct": round(float(c.top_value_pct), 2),
                }
                for c in self.column_profiles
            ],
        }


# ── Normalisation helpers ─────────────────────────────────────────────────────

_NULL_MARKERS = frozenset({
    "na", "n/a", "nan", "null", "none", "nil", "missing",
    "-", "--", "?", "unknown", "undefined", "",
})


def _normalize_col_name(col: str) -> str:
    """Strip, lowercase, replace non-alphanumeric runs with underscore."""
    col = str(col).strip()
    col = re.sub(r"[^a-zA-Z0-9]+", "_", col)
    col = col.strip("_").lower()
    return col or "col"


def _infer_type(series: pd.Series) -> str:
    """Best-effort semantic type label for a Series."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Try to parse as datetime
    sample = series.dropna().head(200)
    if len(sample) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().mean() > 0.8:
                return "datetime"
        except Exception:
            pass
        # Try numeric cast
        try:
            numeric = pd.to_numeric(sample, errors="coerce")
            if numeric.notna().mean() > 0.9:
                return "numeric"
        except Exception:
            pass
    n_unique = series.nunique()
    n_total = len(series.dropna())
    # Use adaptive threshold: small DFs (< 50 rows) use 50% unique ratio
    threshold = 0.50 if n_total < 50 else 0.05
    if n_total > 0 and n_unique / n_total <= threshold:
        return "categorical"
    return "text"


def _has_mixed_types(series: pd.Series) -> bool:
    """Heuristic: object dtype series where some values look numeric, some don't."""
    if series.dtype != object:
        return False
    sample = series.dropna().head(500)
    if len(sample) == 0:
        return False
    numeric_like = pd.to_numeric(sample, errors="coerce").notna()
    ratio = numeric_like.mean()
    return 0.05 < ratio < 0.95   # mixed


def _outlier_count_iqr(series: pd.Series) -> int:
    """Count IQR-based outliers in a numeric Series."""
    clean = series.dropna()
    if len(clean) < 4:
        return 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((clean < lower) | (clean > upper)).sum())


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalise a DataFrame for consistent downstream use.

    Returns:
        (df_normalised, column_map) where column_map is
        {normalised_name: original_name}.

    Steps:
      1. Strip whitespace from column names
      2. Normalize column names (lowercase, alphanumeric)
      3. Deduplicate column names (append _2, _3, …)
      4. Replace string null markers with np.nan
      5. Infer and cast datetime columns
      6. Infer and cast numeric-string columns
    """
    df = df.copy()

    # 1 & 2. Strip + normalize column names
    original_cols = list(df.columns)
    new_cols = [_normalize_col_name(c) for c in original_cols]

    # 3. Deduplicate
    seen: dict[str, int] = {}
    deduped = []
    for name in new_cols:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    column_map = {norm: orig for norm, orig in zip(deduped, original_cols)}
    df.columns = deduped

    # 4. Replace null markers in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: np.nan if (str(v).strip().lower() in _NULL_MARKERS) else v
        )

    # 5 & 6. Type coercion
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'str':
            # Try datetime
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.85:
                    df[col] = parsed
                    continue
            except Exception:
                pass
            # Try numeric — use coerce mode and check success rate
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().mean() > 0.90:
                    df[col] = numeric
            except Exception:
                pass

    return df, column_map


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_dataframe(filename: str, df: pd.DataFrame) -> DataQualityReport:
    """
    Generate a DataQualityReport for a (possibly already normalised) DataFrame.
    """
    n_rows = len(df)
    n_cols = len(df.columns)
    dup_count = int(df.duplicated().sum())
    total_nulls = int(df.isnull().sum().sum())
    dup_pct = (dup_count / n_rows * 100) if n_rows > 0 else 0.0
    total_null_pct = (total_nulls / (n_rows * n_cols) * 100) if (n_rows * n_cols) > 0 else 0.0

    qual_warnings: List[str] = []
    if dup_pct > 5:
        qual_warnings.append(f"{dup_pct:.1f}% duplicate rows detected")
    if total_null_pct > 20:
        qual_warnings.append(f"{total_null_pct:.1f}% of all values are missing")

    col_profiles: List[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isnull().sum())
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0.0
        unique_count = int(series.nunique())
        unique_pct = (unique_count / n_rows * 100) if n_rows > 0 else 0.0
        inferred = _infer_type(series)
        mixed = _has_mixed_types(series)

        top_value, top_pct = None, 0.0
        vc = series.value_counts()
        if len(vc) > 0:
            top_value = vc.index[0]
            top_pct = vc.iloc[0] / n_rows * 100

        min_val = max_val = mean_val = None
        outliers = 0
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                min_val = float(clean.min())
                max_val = float(clean.max())
                mean_val = float(clean.mean())
                outliers = _outlier_count_iqr(series)

        if null_pct > 30:
            qual_warnings.append(f"Column '{col}' has {null_pct:.0f}% missing values")
        if mixed:
            qual_warnings.append(f"Column '{col}' has mixed numeric/text values")
        if outliers > 0 and n_rows > 0 and outliers / n_rows > 0.05:
            qual_warnings.append(f"Column '{col}' has {outliers} outliers ({outliers/n_rows*100:.0f}%)")

        col_profiles.append(ColumnProfile(
            name=col,
            original_name=col,
            dtype=str(series.dtype),
            inferred_type=inferred,
            null_count=null_count,
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            top_value=top_value,
            top_value_pct=top_pct,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            outlier_count=outliers,
            has_mixed_types=mixed,
        ))

    logger.info(
        f"Profiled '{filename}': {n_rows} rows, {n_cols} cols, "
        f"{dup_count} dups, {total_nulls} nulls, {len(qual_warnings)} warnings"
    )

    return DataQualityReport(
        filename=filename,
        rows=n_rows,
        columns=n_cols,
        duplicate_rows=dup_count,
        duplicate_pct=dup_pct,
        total_nulls=total_nulls,
        total_null_pct=total_null_pct,
        column_profiles=col_profiles,
        warnings=qual_warnings,
        column_map={c.name: c.original_name for c in col_profiles},
    )
