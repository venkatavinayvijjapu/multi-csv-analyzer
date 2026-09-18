"""
api_endpoints/analysis.py
--------------------------
LLM-powered analysis endpoint — production-grade version.

Improvements over the original:
  • Secure sandboxed execution (core.sandbox)  — no bare exec()
  • AST-based code validation before execution (core.code_validator)
  • Auto-retry on exec failure (MAX_RETRIES = 2) with error feedback
  • Conversation memory injected into prompts (core.memory)
  • Cross-file join detection via core.join_detector
  • Data provenance tracking (core.provenance)
  • Reliability metrics (core.metrics)
  • Versioned prompt templates (prompt.PromptRegistry)
  • Configurable query timeout (QUERY_TIMEOUT_SECONDS env var)

Public function:
    run_analysis(dfs, query, session_id="") → str

HTTP endpoint:
    POST /analyze   (kept for backward compatibility)
"""

from __future__ import annotations

import io
import os
import re
import time
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import base64

# ── Core imports ──────────────────────────────────────────────────────────────
from core.sandbox import run_code, ExecutionResult
from core.code_validator import validate_code, CodeValidationError
from core.join_detector import detect_joins, format_join_hints
from core.memory import get_history, format_history_for_prompt
from core.provenance import record_query
from core.metrics import record_metric, LatencyTimer
from core.logging_config import get_logger
from prompt import registry as prompt_registry

logger = get_logger(__name__)

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
EXPLABS_API_KEY  = os.getenv("EXPLABS_API_KEY") or os.getenv("GROQ_API_KEY", "")
EXPLABS_BASE_URL = os.getenv("EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1")
EXPLABS_MODEL    = os.getenv("EXPLABS_MODEL") or os.getenv("GROQ_MODEL", "gpt-5.6-luna")
MAX_RETRIES      = int(os.getenv("MAX_RETRIES", "2"))
QUERY_TIMEOUT    = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))

if not EXPLABS_API_KEY:
    raise ValueError("EXPLABS_API_KEY (or GROQ_API_KEY) not found in environment.")

client = OpenAI(api_key=EXPLABS_API_KEY, base_url=EXPLABS_BASE_URL)
router = APIRouter()


# ── Pydantic models (HTTP endpoint) ───────────────────────────────────────────

class FilePayload(BaseModel):
    name: str
    data_base64: str


class AnalysisRequest(BaseModel):
    files: List[FilePayload]
    query: str
    session_id: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_dataframes(files: List[FilePayload]) -> dict:
    dfs = {}
    for f in files:
        raw = base64.b64decode(f.data_base64)
        if f.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        dfs[f.name] = df
    return dfs


def _safe_var(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    return ("df_" + safe) if (safe and safe[0].isdigit()) else (safe or "df")


def _build_full_schema(dfs: dict) -> tuple[str, dict]:
    """
    Returns (schema_text, var_map).
    Schema includes: column names, dtypes, null/unique counts, sample rows.
    """
    schema_parts = []
    var_map: dict = {}

    for name, df in dfs.items():
        var = _safe_var(name)
        base_var = var
        suffix = 1
        while var in var_map:
            var = f"{base_var}_{suffix}"
            suffix += 1
        var_map[var] = df

        col_details = "\n".join(
            f"    {col!r}: {df[col].dtype}  "
            f"(unique={df[col].nunique()}, nulls={df[col].isnull().sum()})"
            for col in df.columns
        )
        try:
            sample = df.head(5).to_string(index=False, max_cols=20)
        except Exception:
            sample = "(could not render sample)"

        schema_parts.append(
            f"Dataset: {name!r}  →  variable: `{var}`\n"
            f"  Rows: {len(df):,}   Columns: {len(df.columns)}\n"
            f"  Column details:\n{col_details}\n"
            f"  Sample (first 5 rows):\n{sample}"
        )

    return "\n\n" + "─" * 60 + "\n\n".join(schema_parts), var_map


def _call_llm(system: str, user: str) -> str:
    """Make an LLM API call and return the response text."""
    response = client.chat.completions.create(
        model=EXPLABS_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def _extract_code(raw: str) -> Optional[str]:
    """Extract code from a ```python ... ``` block."""
    match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ── Core analysis function ─────────────────────────────────────────────────────

def run_analysis(dfs: dict, query: str, session_id: str = "") -> str:
    """
    Execute an analytical query against the provided DataFrames.

    Pipeline:
      1. Build full schema + var_map
      2. Detect cross-file join keys
      3. Load conversation history
      4. Call LLM → get code
      5. Validate code (AST)
      6. Execute in sandbox
      7. On failure → retry with error feedback (up to MAX_RETRIES)
      8. Fallback to plain-text if all retries exhausted
      9. Record provenance + metrics
    """
    if not dfs:
        return "❌ No datasets available."

    with LatencyTimer() as timer:
        schema_text, var_map = _build_full_schema(dfs)

        # Prepare execution environment
        exec_env: dict = {"pd": pd, "np": np}
        for var_name, df in var_map.items():
            exec_env[var_name] = df.copy()

        var_names_str = ", ".join(f"`{v}`" for v in var_map)

        # Join key hints
        join_suggestions = detect_joins(dfs)
        join_hints = format_join_hints(join_suggestions)

        # Conversation history
        history = get_history(session_id, max_turns=5) if session_id else []
        history_text = format_history_for_prompt(history)

        # Prompts
        system_msg = prompt_registry.get("analysis_system")
        user_msg = prompt_registry.render(
            "analysis_user",
            history=history_text,
            schema=schema_text,
            join_hints=join_hints,
            var_names=var_names_str,
            query=query,
        )

        result = ""
        retried = False
        error_for_retry: Optional[str] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt == 0:
                    logger.info(f"Analysis LLM call: {query[:80]}")
                    raw = _call_llm(system_msg, user_msg)
                else:
                    # Retry with error context
                    logger.warning(f"Analysis retry #{attempt}: {error_for_retry[:100]}")
                    retried = True
                    retry_msg = prompt_registry.render(
                        "analysis_retry",
                        error=error_for_retry or "",
                        query=query,
                        schema=schema_text,
                        var_names=var_names_str,
                    )
                    raw = _call_llm(system_msg, retry_msg)

                code = _extract_code(raw)
                if not code:
                    # Plain-text answer — return directly
                    result = raw.strip()
                    break

                logger.debug(f"Analysis code (attempt {attempt}):\n{code[:400]}")

                # Validate before execution
                issues = validate_code(code)
                if issues:
                    error_for_retry = "Code validation failed: " + "; ".join(issues)
                    if attempt >= MAX_RETRIES:
                        result = f"⚠️ Generated code failed security validation: {'; '.join(issues)}"
                        break
                    continue

                # Execute in sandbox
                exec_result: ExecutionResult = run_code(
                    code, exec_env, timeout_seconds=QUERY_TIMEOUT
                )

                if exec_result.timed_out:
                    result = f"⏱️ Query timed out after {QUERY_TIMEOUT}s. Try a simpler question or smaller data."
                    break

                if exec_result.success and exec_result.stdout:
                    result = exec_result.stdout
                    break

                # Execution error → retry
                error_for_retry = exec_result.error or "No output produced."
                if attempt >= MAX_RETRIES:
                    # All retries exhausted → plain-text fallback
                    logger.warning("All retries exhausted — falling back to plain-text LLM")
                    fallback_msg = prompt_registry.render(
                        "analysis_fallback",
                        schema=schema_text,
                        query=query,
                    )
                    fb_raw = _call_llm(
                        "You are a helpful data analyst. Answer in plain English or markdown.",
                        fallback_msg,
                    )
                    result = fb_raw or "Could not analyze the data."
                    break

            except Exception as exc:
                logger.error(f"Analysis attempt {attempt} exception: {exc}")
                error_for_retry = str(exc)
                if attempt >= MAX_RETRIES:
                    result = f"❌ Analysis failed after {MAX_RETRIES + 1} attempts: {exc}"
                    break

    # ── Post-execution tracking ────────────────────────────────────────────────
    files_used = list(dfs.keys())
    success = not result.startswith("❌")

    if session_id:
        record_query(
            session_id=session_id,
            query=query,
            operation_type="analysis",
            result=result,
            latency_ms=timer.elapsed_ms,
            files_used=files_used,
            error_message=None if success else result,
        )

    record_metric(
        session_id=session_id,
        operation="analysis",
        success=success,
        latency_ms=timer.elapsed_ms,
        retried=retried,
        timed_out="⏱️" in result,
        code_executed=True,
    )

    logger.info(
        f"Analysis done: success={success} retried={retried} "
        f"latency={timer.elapsed_ms}ms result_len={len(result)}"
    )
    return result


# ── HTTP Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_endpoint(payload: AnalysisRequest):
    try:
        dfs = _decode_dataframes(payload.files)
        result = run_analysis(dfs, payload.query, session_id=payload.session_id)
        return {"result": result}
    except Exception as exc:
        logger.error(f"Analyze endpoint error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))