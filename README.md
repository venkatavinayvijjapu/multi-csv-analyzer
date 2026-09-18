# DataLens AI

**Production-grade multi-file CSV/Excel analytics powered by ExperientialLabs LLM + LangGraph.**

---

## Quick Start

```bash
# One-time setup
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configure (copy and fill in your API key)
copy .env.example .env

# Run
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — upload any CSV or Excel, ask anything in plain English.

---

## Architecture

```
Browser (SPA)
    │   GET /           → static/index.html
    │   POST /api/upload
    │   POST /api/query
    │   GET  /api/files
    │   GET  /api/health
    │   GET  /api/provenance
    └──────────────────────────────────────────────────
                          FastAPI (api.py)
                               │
                        LangGraph Agent (agent.py)
                    ┌──────────┴──────────────┐
              structural_check           intent_classify
              (pandas, no LLM)          (1 LLM call)
                                    ┌────┴────┐
                                  analyze  visualize
                                     │        │
                               api_endpoints/
                               analysis.py   visualize.py
                                     │        │
                              core/ ─────────────────────
                              ├── sandbox.py         (secure exec)
                              ├── code_validator.py  (AST check)
                              ├── data_quality.py    (profiling)
                              ├── join_detector.py   (cross-file)
                              ├── memory.py          (history)
                              ├── provenance.py      (lineage)
                              ├── metrics.py         (reliability)
                              └── logging_config.py  (observability)
                                     │
                              session_store.py        (TTL sessions)
                              prompt.py               (versioned prompts)
```

---

## API Reference

### File Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload CSV/Excel files to a session |
| `GET` | `/api/files` | List files and metadata for a session |
| `DELETE` | `/api/file` | Remove a single file |
| `DELETE` | `/api/clear` | Remove all files for a session |

### Query

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/query` | Run an analytical or visualization query |

**Request:**
```json
{ "session_id": "abc-123", "query": "What is the total salary by department?" }
```

**Response:**
```json
{
  "result": "### Total Salary by Department\n...",
  "image_base64": null,
  "chart_data": null,
  "delta_used": false,
  "intent": "analysis"
}
```

### Conversation

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/history` | Get conversation history (last N turns) |
| `DELETE` | `/api/history` | Clear conversation history |

### Session Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/session/info` | Session metadata (age, TTL, memory) |
| `POST` | `/api/session/extend` | Reset session TTL |

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Uptime, session count, reliability metrics |
| `GET` | `/api/metrics` | Global query success/retry/latency stats |
| `GET` | `/api/provenance` | Per-session query lineage trail |
| `GET` | `/api/prompts` | List all registered prompt templates |
| `GET` | `/metrics` | Prometheus metrics (via instrumentator) |

### Legacy (Backward Compatible)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Direct analysis with base64-encoded file payloads |
| `POST` | `/visualize` | Direct visualization |
| `POST` | `/smart_chart` | Convert chart_data dict to PNG |

---

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXPLABS_API_KEY` | *(required)* | ExperientialLabs API key |
| `EXPLABS_BASE_URL` | `https://api.experientiallabs.ai/v1` | API base URL |
| `EXPLABS_MODEL` | `gpt-5.6-luna` | Model name |
| `GROQ_API_KEY` | — | Fallback: Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Fallback model |
| `SESSION_TTL_SECONDS` | `7200` | Session expiry (2 hours) |
| `MAX_FILE_SIZE_MB` | `50` | Max upload file size |
| `MAX_FILES_PER_SESSION` | `20` | Max files per session |
| `MAX_ROWS_PER_FILE` | `500000` | Max rows per file |
| `MAX_RETRIES` | `2` | LLM code retry attempts on failure |
| `QUERY_TIMEOUT_SECONDS` | `30` | Sandbox execution timeout |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## Security

- **Code execution is sandboxed**: `exec()` runs in an isolated namespace with a strict allowlist of builtins. `open`, `eval`, `exec`, `__import__`, `os`, `sys`, `subprocess`, `socket` are all blocked.
- **Static analysis before execution**: All LLM-generated code is AST-parsed to detect forbidden imports and calls before any code runs.
- **Timeout enforcement**: Every code execution is limited to `QUERY_TIMEOUT_SECONDS` (default 30s) via a threading timeout.
- **Memory tracking**: `tracemalloc` monitors peak memory; execution fails if it exceeds `512 MB`.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v --tb=short

# Run specific test file
python -m pytest tests/test_sandbox.py -v

# Run evaluation suite (requires live session with uploaded data)
python -m tests.eval_framework --session <session_id> --numeric-col salary --cat-col dept
```

---

## Docker

```bash
# Build
docker build -t datalens-ai .

# Run
docker run -p 8080:8080 --env-file .env datalens-ai

# Docker Compose (with env vars)
docker compose up
```

---

## Known Limitations

1. **In-memory session storage** — sessions are lost on server restart. Use Redis for persistence (swap `session_store.py`).
2. **Single-worker required** — the in-memory store does not work across multiple uvicorn workers. Use `--workers 1` or add a shared backend.
3. **LLM hallucinations** — the sandbox catches runtime errors (wrong column names, type mismatches) and retries, but deeply incorrect logic may still produce a plausible-looking but wrong answer. Always sanity-check critical results.
4. **File format support** — CSV and XLSX/XLS only. Parquet, JSON, and database connectors are not yet supported.
5. **No authentication** — all sessions are open by session_id UUID. Add auth middleware if deploying publicly.
6. **Visualization complexity** — very complex multi-axis or animated charts may exceed LLM context or timeout; simplify the query if this occurs.

---

## Data Flow

```
1. Upload:   raw bytes → pandas → normalize_dataframe() → profile_dataframe() → session_store
2. Query:    session_id + query → LangGraph agent
3. Agent:    schema_summary + conversation_history + join_hints → LLM prompt
4. LLM:      generates Python pandas/matplotlib code
5. Validate: code_validator.assert_safe(code) — AST check
6. Execute:  sandbox.run_code(code, env) — timeout + memory limits
7. Retry:    on failure, append error to prompt, re-ask LLM (up to MAX_RETRIES)
8. Track:    provenance.record_query() + metrics.record_metric()
9. Memory:   memory.add_exchange() → stored in session for next query
```
