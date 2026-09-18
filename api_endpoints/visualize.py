"""
api_endpoints/visualize.py
---------------------------
LLM-powered visualization endpoint — production-grade version.

Improvements over the original:
  • Secure sandboxed execution (core.sandbox)
  • AST-based code validation (core.code_validator)
  • Auto-retry on exec failure (MAX_RETRIES = 2)
  • Conversation memory injected into prompts
  • matplotlib rcParams reset between executions (prevents state leakage)
  • Provenance tracking + reliability metrics
  • Versioned prompt templates (prompt.PromptRegistry)

Public function:
    run_visualization(dfs, query, session_id="") → (image_base64, error)

HTTP endpoint:
    POST /visualize  (kept for backward compatibility)
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from core.sandbox import run_code, ExecutionResult
from core.code_validator import validate_code
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


# ── Pydantic models ────────────────────────────────────────────────────────────

class FilePayload(BaseModel):
    name: str
    data_base64: str


class VisualizeRequest(BaseModel):
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


def _safe_var_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    if safe and safe[0].isdigit():
        safe = "df_" + safe
    return safe or "df"


def _build_schema(dfs: dict) -> Tuple[str, str, Dict[str, str]]:
    """Returns (schema_text, var_map_display_str, var_map_dict)."""
    parts = []
    var_map: Dict[str, str] = {}  # variable_name → df_name
    for name, df in dfs.items():
        var = _safe_var_name(name)
        var_map[var] = name
        col_info = "\n".join(f"  - {col}: {df[col].dtype}" for col in df.columns)
        sample = df.head(3).to_string(index=False, max_cols=10)
        parts.append(
            f"DataFrame '{name}':\n"
            f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"  Columns:\n{col_info}\n"
            f"  Sample:\n{sample}"
        )
    schema = "\n\n".join(parts)
    var_map_str = "\n".join(f"  - '{name}' → variable `{var}`" for var, name in var_map.items())
    return schema, var_map_str, var_map


def _call_llm(system: str, user: str) -> str:
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
    match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else None


def _reset_matplotlib() -> None:
    """Close all figures and reset rcParams to prevent state bleed."""
    plt.close("all")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.use("Agg")


# ── Core visualization function ────────────────────────────────────────────────

def run_visualization(
    dfs: dict,
    query: str,
    session_id: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a matplotlib chart for the given query.

    Returns: (image_base64, error_message) — one is always None.
    """
    if not dfs:
        return None, "No datasets available."

    _reset_matplotlib()

    with LatencyTimer() as timer:
        schema, var_map_str, var_map_dict = _build_schema(dfs)
        n_datasets = len(dfs)

        exec_env: dict = {
            "pd": pd, "plt": plt, "io": io,
            "base64": base64, "np": np,
        }
        for var, name in var_map_dict.items():
            exec_env[var] = dfs[name].copy()

        # Conversation history
        history = get_history(session_id, max_turns=3) if session_id else []
        history_text = format_history_for_prompt(history)

        # Join hints
        join_suggestions = detect_joins(dfs)
        join_hints = format_join_hints(join_suggestions)
        if join_hints:
            schema = schema + "\n\n" + join_hints

        system_msg = prompt_registry.get("visualization_system")
        user_msg = prompt_registry.render(
            "visualization_user",
            n_datasets=n_datasets,
            history=history_text,
            schema=schema,
            var_map=var_map_str,
            query=query,
        )

        img_b64: Optional[str] = None
        error: Optional[str] = None
        retried = False

        for attempt in range(MAX_RETRIES + 1):
            _reset_matplotlib()
            try:
                if attempt == 0:
                    logger.info(f"Visualization LLM call: {query[:80]}")
                    raw = _call_llm(system_msg, user_msg)
                else:
                    logger.warning(f"Visualization retry #{attempt}: {error}")
                    retried = True
                    retry_msg = prompt_registry.render(
                        "visualization_retry",
                        error=error or "",
                        query=query,
                    )
                    raw = _call_llm(system_msg, retry_msg)

                code = _extract_code(raw)
                if not code:
                    error = "No valid Python code returned by LLM."
                    if attempt >= MAX_RETRIES:
                        break
                    continue

                logger.debug(f"Viz code (attempt {attempt}):\n{code[:400]}")

                # AST validation
                issues = validate_code(code)
                if issues:
                    error = "Code validation failed: " + "; ".join(issues)
                    if attempt >= MAX_RETRIES:
                        break
                    continue

                # Sandbox execution
                exec_result: ExecutionResult = run_code(
                    code, exec_env, timeout_seconds=QUERY_TIMEOUT
                )

                if exec_result.timed_out:
                    return None, f"⏱️ Chart generation timed out after {QUERY_TIMEOUT}s."

                stdout = exec_result.stdout
                if exec_result.success and stdout and not stdout.startswith("Error"):
                    img_b64 = stdout
                    error = None
                    break

                error = exec_result.error or stdout or "No image generated."
                if attempt >= MAX_RETRIES:
                    break

            except Exception as exc:
                logger.error(f"Visualization attempt {attempt} exception: {exc}")
                error = str(exc)
                if attempt >= MAX_RETRIES:
                    break

    # ── Tracking ───────────────────────────────────────────────────────────────
    success = img_b64 is not None
    files_used = list(dfs.keys())

    if session_id:
        record_query(
            session_id=session_id,
            query=query,
            operation_type="visualization",
            result=img_b64 or "",
            latency_ms=timer.elapsed_ms,
            files_used=files_used,
            error_message=error,
        )

    record_metric(
        session_id=session_id,
        operation="visualization",
        success=success,
        latency_ms=timer.elapsed_ms,
        retried=retried,
    )

    logger.info(
        f"Visualization done: success={success} retried={retried} latency={timer.elapsed_ms}ms"
    )
    return img_b64, error


# ── HTTP Endpoint ──────────────────────────────────────────────────────────────

@router.post("/visualize")
async def visualize_endpoint(payload: VisualizeRequest):
    try:
        dfs = _decode_dataframes(payload.files)
        img_b64, error = run_visualization(dfs, payload.query, session_id=payload.session_id)
        if img_b64:
            return {"image_base64": img_b64}
        raise HTTPException(status_code=400, detail=error or "Visualization failed.")
    except HTTPException:
        raise
    except Exception as ex:
        logger.error(f"Visualize endpoint error: {ex}")
        raise HTTPException(status_code=400, detail=str(ex))