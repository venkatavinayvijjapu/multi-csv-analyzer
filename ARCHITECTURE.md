# DataLens AI — Architecture Documentation

## System Overview

DataLens AI is a production-grade, multi-file data analytics API. Users upload CSV or Excel files and ask questions in plain English. Every answer is computed by executing LLM-generated Python code against the actual data — no hallucinated estimates.

---

## Component Map

```
api.py                     Main FastAPI application
├── Middleware
│   ├── CORSMiddleware
│   └── RequestIDMiddleware        injects X-Request-ID into every log line
├── Routers
│   ├── /api/upload                parse + normalise + profile → session_store
│   ├── /api/query                 → LangGraph agent
│   ├── /api/files, /clear, /file  file management
│   ├── /api/history               conversation history
│   ├── /api/session/*             lifecycle management
│   ├── /api/health, /metrics      observability
│   ├── /api/provenance, /prompts  audit + debug
│   └── /analyze, /visualize, /smart_chart  (legacy compatibility)
│
agent.py                   LangGraph StateGraph orchestrator
├── structural_check_node  pandas-only, zero LLM latency
├── intent_classify_node   one LLM call → ANALYZE or VISUALIZE
├── analysis_node          → api_endpoints/analysis.py
└── visualization_node     → api_endpoints/visualize.py
│
api_endpoints/
├── analysis.py            LLM → pandas code → sandbox → result
├── visualize.py           LLM → matplotlib code → sandbox → base64 PNG
└── smart_chart.py         chart_data dict → matplotlib PNG (no LLM)
│
core/                      Shared infrastructure
├── sandbox.py             Secure exec with timeout + blocked builtins
├── code_validator.py      AST static analysis before any exec()
├── data_quality.py        Schema normalisation + quality profiling
├── join_detector.py       Cross-file key detection (fuzzy + value overlap)
├── memory.py              Conversation history API
├── provenance.py          Query → file/column lineage tracking
├── metrics.py             Reliability metrics (success/retry/latency)
└── logging_config.py      Centralised rotating logs + request IDs
│
session_store.py           Thread-safe in-memory session store + TTL cleanup
prompt.py                  Versioned prompt registry (PromptRegistry)
delta_analysis.py          Structural fast-path (describe, nulls, dups, corr)
```

---

## Data Flow: Query Lifecycle

```
POST /api/query
  │
  ├── 1. Validate: session has files, query non-empty
  │
  ├── 2. agent.run_query(session_id, query)
  │       │
  │       ├── Load conversation history (core.memory)
  │       │
  │       ├── Node: structural_check
  │       │     → if schema/describe/null/dup/corr query → pandas answer instantly
  │       │     → else fall through
  │       │
  │       ├── Node: intent_classify (1 LLM call)
  │       │     → prompt: schema + query → ANALYZE or VISUALIZE
  │       │
  │       └── Node: analysis_node  OR  visualization_node
  │             │
  │             ├── Build full schema (all DFs: col names, dtypes, null counts, sample rows)
  │             ├── detect_joins() → join key hints
  │             ├── format_history_for_prompt() → last 5 turns
  │             ├── LLM call (system + user prompt from PromptRegistry)
  │             │
  │             ├── Extract ```python...``` code block
  │             ├── code_validator.assert_safe(code)  [AST check]
  │             ├── sandbox.run_code(code, env, timeout=30s)
  │             │
  │             ├── [on failure] retry up to MAX_RETRIES:
  │             │     append error → re-prompt → re-validate → re-execute
  │             │
  │             ├── [all retries exhausted] plain-text LLM fallback
  │             │
  │             ├── memory.add_exchange(session_id, query, result)
  │             ├── provenance.record_query(...)
  │             └── metrics.record_metric(...)
  │
  └── Return JSON: {result, image_base64, chart_data, delta_used, intent}
```

---

## Data Flow: Upload Lifecycle

```
POST /api/upload
  │
  ├── Check file count limit (MAX_FILES_PER_SESSION)
  ├── For each file:
  │     ├── Read bytes
  │     ├── Check size limit (MAX_FILE_SIZE_MB)
  │     ├── Parse: pd.read_csv / pd.read_excel (all sheets)
  │     ├── Check row count (MAX_ROWS_PER_FILE)
  │     ├── normalize_dataframe():
  │     │     ├── Strip whitespace from column names
  │     │     ├── Lowercase + alphanumeric normalisation
  │     │     ├── Deduplicate column names
  │     │     ├── Replace null markers (N/A, null, "", -, etc.)
  │     │     ├── Infer + cast datetime strings
  │     │     └── Infer + cast numeric strings
  │     ├── profile_dataframe():
  │     │     ├── Null counts per column + total
  │     │     ├── Duplicate row detection
  │     │     ├── Type inference per column
  │     │     ├── IQR-based outlier detection
  │     │     ├── Mixed-type detection
  │     │     └── Quality warnings
  │     └── session_store.set_file(session_id, name, df, quality_report)
  │
  └── Return: {success, files (with quality reports), errors}
```

---

## Security Model

### Sandboxed Execution

All LLM-generated code goes through two layers before execution:

**Layer 1: Static Analysis (code_validator.py)**
- AST parse — syntax check
- Blocked imports: `os`, `sys`, `subprocess`, `socket`, `shutil`, `pathlib`, `ctypes`, `pickle`, `threading`, `multiprocessing`, `inspect`, `gc`, and 20+ others
- Blocked calls: `open`, `eval`, `exec`, `compile`, `__import__`, `breakpoint`, `input`
- Blocked dunder attributes: `__class__`, `__subclasses__`, `__globals__`, `__builtins__`, `__code__`

**Layer 2: Runtime Sandbox (sandbox.py)**
- Restricted builtins: `__builtins__` is set to `{}` — no escape path
- Thread-based timeout: code runs in a daemon thread; killed after `QUERY_TIMEOUT_SECONDS`
- Memory tracking: `tracemalloc` measures peak usage; errors if > 512 MB
- Namespace isolation: exec_env contains only explicitly provided variables (`pd`, `np`, dataframes)

### What's Not Protected

- The LLM itself may produce incorrect logic (wrong aggregation, wrong join key) — this is a semantic problem, not a security problem. The retry mechanism catches many such errors.
- No rate limiting on API endpoints (add with `slowapi` for production).
- No authentication on session IDs (add JWT/API-key middleware for public deployment).

---

## Session Store

Sessions are in-memory Python dicts, keyed by a UUID generated by the browser. Each session contains:

```python
SessionData:
    dataframes:           {filename: pd.DataFrame}
    quality_reports:      {filename: dict}
    conversation_history: [{role, content}, ...]  # rolling cap: 40 messages
    created_at:           float (Unix timestamp)
    last_accessed:        float (Unix timestamp)
```

A background daemon thread runs every 10 minutes and removes sessions where `last_accessed` is more than `SESSION_TTL_SECONDS` (default 2 hours) ago.

**For production with multiple workers or server restarts:** Replace the in-memory store with Redis using `redis-py`. The `session_store.py` interface is stable — only the storage backend needs to change.

---

## Prompt Management

All LLM prompts live in `prompt.py` (`PromptRegistry`). Each prompt has:
- A **name** (e.g., `"analysis_system"`, `"visualization_user"`)
- A **version** (e.g., `"v1"`, `"v2"`)
- A **current version pointer** (default used unless pinned)
- **`$variable` placeholders** substituted via `registry.render(name, **kwargs)`

To add a new prompt version:
1. Add entry to `_templates[name][new_version]`
2. Update `_current_versions[name]` to the new version
3. The old version is preserved for rollback

---

## Cross-File Join Detection

`core/join_detector.py` detects potential join keys between DataFrames:

1. **Exact match**: column names are identical (after normalisation)
2. **Fuzzy/alias match**: column names are known synonyms (`emp_id` ↔ `employee_id`, `dept` ↔ `department`)
3. **Value overlap**: samples values from both columns; if ≥10% overlap → likely join key
4. **Cardinality**: labels the join as `1:1`, `1:N`, `N:1`, or `N:M`
5. **Confidence**: weighted score combining name similarity (50%) and value overlap (50%)

Join suggestions are injected into every LLM prompt as structured hints, with merge examples.

---

## Conversation Memory

Each query is stored as a `{role, content}` pair in the session's `conversation_history`. The last 5 exchanges (10 messages) are injected at the top of every subsequent LLM prompt.

This allows follow-up queries like:
- *"What is the total salary?"* → answer
- *"Break that down by department"* → LLM sees the previous exchange and can reference it
- *"Now filter to just the Eng department"* → further refinement

History is capped at 40 messages per session to keep prompts manageable.

---

## Evaluation

Run `tests/eval_framework.py` against any live session:

```bash
# Upload a file first, note the session_id from the upload response
python -m tests.eval_framework \
    --session <session_id> \
    --numeric-col salary \
    --cat-col dept \
    --df-name employees.csv
```

Output:
```
=== DataLens AI Eval Report ===
Total:    8
Passed:   7  (87.5%)
Failed:   1
Avg Lat:  2340ms

✅ [structural] Describe the data in employees.csv
✅ [structural] How many missing values are in employees.csv?
✅ [analysis]   What is the total salary?
❌ [analysis]   Group by dept and show the total salary
    ↳ Expected 'dept' in answer, not found
```

---

## Extension Points

| Feature | File to Modify | Notes |
|---------|---------------|-------|
| Add a new prompt | `prompt.py` | Add to `_templates`, bump version |
| Add a new API endpoint | `api.py` | Use existing patterns |
| Add a new LangGraph node | `agent.py` | Add node + conditional edge |
| Persistent sessions | `session_store.py` | Swap `_store` dict for Redis client |
| Authentication | `api.py` middleware | Add `fastapi-users` or JWT middleware |
| New file format (Parquet, JSON) | `api.py` `_store_file()` | Add branch to format detection |
| Stricter sandbox | `core/sandbox.py` | Add to `_BLOCKED_BUILTINS` |
| New join alias | `core/join_detector.py` | Add to `_ALIAS_GROUPS` |
