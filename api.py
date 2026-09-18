"""
api.py
------
Main FastAPI application — production-grade version.

New features over original:
  • Centralised logging with request IDs via core.logging_config
  • Request ID middleware (X-Request-ID header injected on all responses)
  • Upload validation: file size, file count, row count limits
  • Data quality profiling at upload time (core.data_quality)
  • Schema normalisation at upload (core.data_quality.normalize_dataframe)
  • GET  /api/health           — uptime, session count, global metrics
  • GET  /api/provenance       — per-session query lineage
  • GET  /api/session/info     — session metadata + TTL remaining
  • POST /api/session/extend   — reset session TTL
  • GET  /api/metrics          — global reliability stats
  • GET  /api/prompts          — list all registered prompt templates
  • DELETE /api/history        — clear conversation history for a session
  • Existing endpoints preserved: /analyze, /visualize, /smart_chart
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Optional

import pandas as pd
import io

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

import session_store
from api_endpoints import analysis, visualize, smart_chart
from core.logging_config import setup_logging, get_logger, set_request_id
from core.data_quality import profile_dataframe, normalize_dataframe
from core.provenance import get_provenance_dicts, session_stats
from core.metrics import get_global_stats

# ── Logging (must be first) ──────────────────────────────────────────────────
setup_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

# ── Upload limits (configurable via env) ──────────────────────────────────────
MAX_FILE_SIZE_MB      = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILES_PER_SESSION = int(os.getenv("MAX_FILES_PER_SESSION", "20"))
MAX_ROWS_PER_FILE     = int(os.getenv("MAX_ROWS_PER_FILE", "500000"))

_START_TIME = time.time()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataLens AI — Multi-File Analytics",
    description=(
        "AI-powered analysis and visualization for multiple CSV/Excel datasets. "
        "Powered by ExperientialLabs LLM + LangGraph. "
        "Features: secure sandboxed execution, conversation memory, data quality profiling, "
        "cross-file join detection, and full provenance tracking."
    ),
    version="4.0.0",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject a unique request ID into every request context and response header."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:8]
    set_request_id(rid)
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-Ms"] = str(latency_ms)
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} latency={latency_ms}ms"
    )
    return response


from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Always return JSON for HTTP errors (overrides FastAPI HTML default)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail or str(exc)},
        headers={"Content-Type": "application/json"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return JSON for request validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors())},
        headers={"Content-Type": "application/json"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: always return JSON for any unhandled 500 error."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
        headers={"Content-Type": "application/json"},
    )


# ── Routers (legacy HTTP endpoints kept for compatibility) ────────────────────
app.include_router(analysis.router,   tags=["Analysis"])
app.include_router(visualize.router,  tags=["Visualize"])
app.include_router(smart_chart.router, tags=["Smart Chart"])

# ── SPA Serving ───────────────────────────────────────────────────────────────
SPA_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_spa():
    try:
        with open(SPA_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h1>SPA not found — ensure static/index.html exists.</h1>",
            status_code=500,
        )


# ── API Models ────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    query: str


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["SPA API"])
async def upload_files(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """
    Upload one or more CSV/Excel files into a session.

    Limits:
      - Max file size: MAX_FILE_SIZE_MB (default 50 MB)
      - Max files per session: MAX_FILES_PER_SESSION (default 20)
      - Max rows per file: MAX_ROWS_PER_FILE (default 500,000)
    """
    # Check session file count
    current_files = session_store.list_files(session_id)
    if len(current_files) + len(files) > MAX_FILES_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Session already has {len(current_files)} file(s). "
                f"Maximum is {MAX_FILES_PER_SESSION} files per session."
            ),
        )

    uploaded = []
    errors = []

    for uf in files:
        try:
            raw = await uf.read()
            name = uf.filename or "file.csv"

            # File size check
            size_mb = len(raw) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                errors.append(
                    f"{name}: file too large ({size_mb:.1f} MB, limit {MAX_FILE_SIZE_MB} MB)"
                )
                continue

            name_lower = name.lower()

            if name_lower.endswith((".xlsx", ".xls")):
                all_sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
                for sheet_key, df_raw in all_sheets.items():
                    if df_raw.empty:
                        continue
                    store_key = name if len(all_sheets) == 1 else (
                        f"{os.path.splitext(name)[0]}[{sheet_key}]"
                    )
                    _store_file(session_id, store_key, df_raw, uploaded, errors, name)
                continue

            elif name_lower.endswith(".csv"):
                try:
                    df_raw = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
                except UnicodeDecodeError:
                    df_raw = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
            else:
                errors.append(f"{name}: unsupported format (use CSV, XLSX, or XLS)")
                continue

            _store_file(session_id, name, df_raw, uploaded, errors, name)

        except Exception as e:
            errors.append(f"{uf.filename}: {e}")
            logger.error(f"Upload error for {uf.filename}: {e}")

    if errors and not uploaded:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return {
        "success": True,
        "files":   session_store.list_files(session_id),
        "errors":  errors,
    }


def _store_file(
    session_id: str,
    store_key: str,
    df_raw: pd.DataFrame,
    uploaded: list,
    errors: list,
    original_name: str,
) -> None:
    """Normalise, validate row count, profile, and store a DataFrame."""
    # Row count check
    if len(df_raw) > MAX_ROWS_PER_FILE:
        errors.append(
            f"{original_name}: too many rows ({len(df_raw):,}, limit {MAX_ROWS_PER_FILE:,})"
        )
        return

    # Schema normalisation + type inference
    try:
        df, col_map = normalize_dataframe(df_raw)
    except Exception as exc:
        logger.warning(f"Normalisation failed for '{store_key}': {exc}, using raw df")
        df = df_raw
        col_map = {}

    # Quality profiling
    try:
        report = profile_dataframe(store_key, df)
        quality_dict = report.to_dict()
    except Exception as exc:
        logger.warning(f"Quality profiling failed for '{store_key}': {exc}")
        quality_dict = {}

    session_store.set_file(session_id, store_key, df, quality_report=quality_dict)
    logger.info(f"Session {session_id[:8]}: stored '{store_key}' {df.shape}")

    uploaded.append({
        "name":      store_key,
        "rows":      len(df),
        "columns":   len(df.columns),
        "col_names": df.columns.tolist()[:15],
        "null_count": int(df.isnull().sum().sum()),
        "quality":   quality_dict,
    })


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/api/query", tags=["SPA API"])
async def query_data(request: QueryRequest):
    """Run the LangGraph agent on the given query for the session's files."""
    from agent import run_query

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    dfs = session_store.get_session(request.session_id)
    if not dfs:
        raise HTTPException(
            status_code=400,
            detail="No files uploaded for this session. Please upload files first.",
        )

    logger.info(f"Session {request.session_id[:8]}: query='{request.query[:80]}'")

    try:
        result = run_query(request.session_id, request.query)
        return result
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── File management ───────────────────────────────────────────────────────────

@app.get("/api/files", tags=["SPA API"])
async def list_files(session_id: str):
    """List uploaded files and their metadata for a session."""
    return {"files": session_store.list_files(session_id)}


@app.delete("/api/clear", tags=["SPA API"])
async def clear_session(session_id: str):
    """Remove all files and history for a session."""
    session_store.clear_session(session_id)
    logger.info(f"Session {session_id[:8]}: cleared")
    return {"success": True}


@app.delete("/api/file", tags=["SPA API"])
async def remove_file(session_id: str, filename: str):
    """Remove a single file from a session."""
    session_store.remove_file(session_id, filename)
    logger.info(f"Session {session_id[:8]}: removed '{filename}'")
    return {"success": True, "files": session_store.list_files(session_id)}


# ── Conversation history ──────────────────────────────────────────────────────

@app.delete("/api/history", tags=["SPA API"])
async def clear_history(session_id: str):
    """Clear conversation history for a session without clearing file data."""
    session_store.clear_conversation_history(session_id)
    logger.info(f"Session {session_id[:8]}: history cleared")
    return {"success": True}


@app.get("/api/history", tags=["SPA API"])
async def get_history(session_id: str, max_turns: int = 10):
    """Return conversation history for a session."""
    history = session_store.get_conversation_history(session_id)
    # Return last max_turns * 2 messages
    return {"history": history[-(max_turns * 2):]}


# ── Session management ────────────────────────────────────────────────────────

@app.get("/api/session/info", tags=["Session"])
async def get_session_info(session_id: str):
    """Return session metadata: age, file count, TTL remaining, memory estimate."""
    return session_store.session_info(session_id)


@app.post("/api/session/extend", tags=["Session"])
async def extend_session(session_id: str):
    """Reset the TTL for a session."""
    session_store.extend_session(session_id)
    return {"success": True, "message": "Session TTL extended."}


# ── Provenance ────────────────────────────────────────────────────────────────

@app.get("/api/provenance", tags=["Observability"])
async def get_provenance(session_id: str):
    """Return the query provenance trail for a session."""
    return {
        "provenance": get_provenance_dicts(session_id),
        "stats":      session_stats(session_id),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

@app.get("/api/metrics", tags=["Observability"])
async def get_metrics():
    """Return global reliability and performance metrics."""
    return get_global_stats()


# ── Prompts registry ──────────────────────────────────────────────────────────

@app.get("/api/prompts", tags=["Observability"])
async def list_prompts():
    """List all registered prompt templates and their versions."""
    from prompt import registry
    return {"prompts": registry.list_prompts()}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["Observability"])
async def health():
    """System health check with uptime, session count, and reliability metrics."""
    uptime_s = int(time.time() - _START_TIME)
    return {
        "status":          "ok",
        "version":         "4.0.0",
        "uptime_seconds":  uptime_s,
        "active_sessions": session_store.active_session_count(),
        "metrics":         get_global_stats(),
        "limits": {
            "max_file_size_mb":      MAX_FILE_SIZE_MB,
            "max_files_per_session": MAX_FILES_PER_SESSION,
            "max_rows_per_file":     MAX_ROWS_PER_FILE,
        },
    }
