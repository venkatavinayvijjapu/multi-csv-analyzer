"""
api.py
------
Main FastAPI application.
- Serves the SPA at GET /
- /api/upload, /api/query, /api/files, /api/clear — for the SPA frontend
- /analyze, /visualize, /smart_chart — existing endpoints (kept for compatibility)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import pandas as pd
import io
import os
import logging

import session_store
from api_endpoints import analysis, visualize, smart_chart

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="api.log", filemode="a", level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataLens AI — Multi-File Analytics",
    description="AI-powered analysis and visualization for multiple CSV/Excel datasets. "
                "Powered by ExperientialLabs gpt-6-astra + LangGraph.",
    version="3.0.0",
)

# ─── Middleware ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)

# ─── Routers (legacy HTTP endpoints kept for compatibility) ───────────────────
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(visualize.router, tags=["Visualize"])
app.include_router(smart_chart.router, tags=["Smart Chart"])

# ─── SPA Serving ─────────────────────────────────────────────────────────────
SPA_PATH = os.path.join(os.path.dirname(__file__), "static", "index.html")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_spa():
    try:
        with open(SPA_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>SPA not found — ensure static/index.html exists.</h1>", status_code=500)


# ─── API Models ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    session_id: str
    query: str


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.post("/api/upload", tags=["SPA API"])
async def upload_files(
    session_id: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Upload one or more CSV/Excel files. Associates them with session_id."""
    uploaded = []
    errors = []

    for uf in files:
        try:
            raw = await uf.read()
            name = uf.filename or "file.csv"
            name_lower = name.lower()

            if name_lower.endswith((".xlsx", ".xls")):
                # Read ALL sheets — each sheet becomes its own dataset
                all_sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
                for sheet_name_key, df in all_sheets.items():
                    if df.empty:
                        continue
                    # Key: "filename[SheetName]" for multi-sheet, "filename" for single sheet
                    if len(all_sheets) == 1:
                        store_key = name
                    else:
                        base = os.path.splitext(name)[0]
                        store_key = f"{base}[{sheet_name_key}]"
                    session_store.set_file(session_id, store_key, df)
                    logger.info(f"Session {session_id[:8]}: uploaded '{store_key}' {df.shape}")
                    uploaded.append({
                        "name": store_key,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "col_names": df.columns.tolist()[:15],
                        "null_count": int(df.isnull().sum().sum()),
                    })
                continue   # already appended above

            elif name_lower.endswith(".csv"):
                # Try UTF-8 first, then latin-1 as fallback for messy CSVs
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
            else:
                errors.append(f"{name}: unsupported format (use CSV, XLSX, or XLS)")
                continue

            session_store.set_file(session_id, name, df)
            logger.info(f"Session {session_id[:8]}: uploaded '{name}' {df.shape}")
            uploaded.append({
                "name": name,
                "rows": len(df),
                "columns": len(df.columns),
                "col_names": df.columns.tolist()[:15],
                "null_count": int(df.isnull().sum().sum()),
            })
        except Exception as e:
            errors.append(f"{uf.filename}: {e}")
            logger.error(f"Upload error for {uf.filename}: {e}")


    if errors and not uploaded:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return {
        "success": True,
        "files": session_store.list_files(session_id),
        "errors": errors,
    }


@app.post("/api/query", tags=["SPA API"])
async def query_data(request: QueryRequest):
    """Run the LangGraph agent on the given query for the session's files."""
    from agent import run_query   # lazy import to avoid circular import at startup

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


@app.get("/api/files", tags=["SPA API"])
async def list_files(session_id: str):
    """List uploaded files and their metadata for a session."""
    return {"files": session_store.list_files(session_id)}


@app.delete("/api/clear", tags=["SPA API"])
async def clear_session(session_id: str):
    """Remove all files for a session."""
    session_store.clear_session(session_id)
    logger.info(f"Session {session_id[:8]}: cleared")
    return {"success": True}


@app.delete("/api/file", tags=["SPA API"])
async def remove_file(session_id: str, filename: str):
    """Remove a single file from a session."""
    session_store.remove_file(session_id, filename)
    logger.info(f"Session {session_id[:8]}: removed '{filename}'")
    return {"success": True, "files": session_store.list_files(session_id)}
