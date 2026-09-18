"""
tests/eval_framework.py
------------------------
Evaluation framework for DataLens AI.

Runs a suite of predefined analytical questions against a session
and reports accuracy, latency, and failure modes.

Usage:
    from tests.eval_framework import run_eval, EvalQuestion
    results = run_eval(session_id, questions=STANDARD_QUESTIONS)
    print(results.summary())

Or run as a script (requires a live server):
    python -m tests.eval_framework --session <session_id>
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict

import pandas as pd


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class EvalQuestion:
    question: str
    expected_contains: List[str]         # substrings that must appear in the answer
    expected_not_contains: List[str] = field(default_factory=list)
    operation: str = "analysis"          # "analysis" | "visualization" | "structural"
    description: str = ""


@dataclass
class EvalResult:
    question: str
    operation: str
    answer: str
    latency_ms: int
    passed: bool
    failure_reasons: List[str]
    retried: bool = False
    timed_out: bool = False


@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    avg_latency_ms: float
    results: List[EvalResult]

    @property
    def accuracy_pct(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"=== DataLens AI Eval Report ===",
            f"Total:    {self.total}",
            f"Passed:   {self.passed}  ({self.accuracy_pct:.1f}%)",
            f"Failed:   {self.failed}",
            f"Avg Lat:  {self.avg_latency_ms:.0f}ms",
            "",
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"{icon} [{r.operation}] {r.question[:70]}")
            if not r.passed:
                for reason in r.failure_reasons:
                    lines.append(f"    ↳ {reason}")
        return "\n".join(lines)


# ── Predefined standard questions ─────────────────────────────────────────────

def make_standard_questions(df_name: str, numeric_col: str, cat_col: str) -> List[EvalQuestion]:
    """
    Return a standard set of questions for any dataset.
    Fill in df_name, numeric_col (e.g. 'salary'), cat_col (e.g. 'dept').
    """
    return [
        EvalQuestion(
            question=f"Describe the data in {df_name}",
            expected_contains=["rows", "columns"],
            operation="structural",
            description="Schema description",
        ),
        EvalQuestion(
            question=f"How many missing values are in {df_name}?",
            expected_contains=["missing"],
            operation="structural",
            description="Null check",
        ),
        EvalQuestion(
            question=f"Are there any duplicate rows in {df_name}?",
            expected_contains=["duplicate"],
            operation="structural",
            description="Duplicate check",
        ),
        EvalQuestion(
            question=f"What is the total {numeric_col}?",
            expected_contains=[numeric_col],
            operation="analysis",
            description="Sum aggregation",
        ),
        EvalQuestion(
            question=f"What is the average {numeric_col}?",
            expected_contains=[numeric_col],
            expected_not_contains=["error", "❌"],
            operation="analysis",
            description="Mean aggregation",
        ),
        EvalQuestion(
            question=f"What is the maximum {numeric_col}?",
            expected_contains=[numeric_col],
            expected_not_contains=["❌"],
            operation="analysis",
            description="Max aggregation",
        ),
        EvalQuestion(
            question=f"How many unique values are in {cat_col}?",
            expected_contains=[cat_col],
            expected_not_contains=["❌"],
            operation="analysis",
            description="Unique count",
        ),
        EvalQuestion(
            question=f"Group by {cat_col} and show the total {numeric_col}",
            expected_contains=[],
            expected_not_contains=["❌", "error"],
            operation="analysis",
            description="GroupBy aggregation",
        ),
    ]


# ── Evaluator ─────────────────────────────────────────────────────────────────

def run_eval(
    session_id: str,
    questions: List[EvalQuestion],
    run_query_fn: Optional[Callable] = None,
) -> EvalReport:
    """
    Run the evaluation suite against a session.

    Args:
        session_id:    Active session with uploaded data.
        questions:     List of EvalQuestion to test.
        run_query_fn:  Optional callable(session_id, query) → dict.
                       Defaults to importing agent.run_query.

    Returns:
        EvalReport with per-question results and summary statistics.
    """
    if run_query_fn is None:
        from agent import run_query
        run_query_fn = run_query

    results: List[EvalResult] = []

    for eq in questions:
        start = time.perf_counter()
        failure_reasons: List[str] = []

        try:
            resp = run_query_fn(session_id, eq.question)
            answer = resp.get("result", "") or ""
            latency_ms = int((time.perf_counter() - start) * 1000)

            # Check expected_contains
            answer_lower = answer.lower()
            for substr in eq.expected_contains:
                if substr.lower() not in answer_lower:
                    failure_reasons.append(f"Expected '{substr}' in answer, not found")

            # Check expected_not_contains
            for substr in eq.expected_not_contains:
                if substr.lower() in answer_lower:
                    failure_reasons.append(f"Found unexpected '{substr}' in answer")

            passed = len(failure_reasons) == 0

        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            answer = f"EXCEPTION: {exc}"
            failure_reasons.append(f"Exception: {exc}")
            passed = False

        results.append(EvalResult(
            question=eq.question,
            operation=eq.operation,
            answer=answer[:500],
            latency_ms=latency_ms,
            passed=passed,
            failure_reasons=failure_reasons,
        ))

    total = len(results)
    passed_count = sum(1 for r in results if r.passed)
    avg_lat = sum(r.latency_ms for r in results) / total if total > 0 else 0

    return EvalReport(
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        avg_latency_ms=avg_lat,
        results=results,
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DataLens AI eval suite")
    parser.add_argument("--session",      required=True, help="Session ID with uploaded data")
    parser.add_argument("--numeric-col",  default="salary", help="A numeric column name")
    parser.add_argument("--cat-col",      default="dept",   help="A categorical column name")
    parser.add_argument("--df-name",      default="data",   help="DataFrame/file name")
    args = parser.parse_args()

    questions = make_standard_questions(args.df_name, args.numeric_col, args.cat_col)
    report = run_eval(args.session, questions)
    print(report.summary())
