# DataLens AI

**Multi-file CSV/Excel analytics powered by ExperientialLabs `gpt-6-astra` + LangGraph**

---

## Approach

The app lets users upload any number of CSV or Excel files and ask analytical questions in plain English. Every answer is produced by **LLM-generated pandas code that is actually executed** — so the result is always a precise, computed value, never a hallucinated estimate.

```
Upload files → ask anything → LLM reads schema → writes pandas/matplotlib code → exec() → exact answer
```

Multi-sheet Excel files are automatically split into one dataset per sheet. CSVs fall back to latin-1 encoding if UTF-8 fails. Files persist in a server-side in-memory session store keyed by a UUID the browser generates on first visit.

---

## Key Decisions

**1. LLM writes code, not answers.**  
Instead of asking the LLM "what is the total revenue?" and trusting its text reply, we ask it to write `df['revenue'].sum()` and execute that. Precision is guaranteed by the Python runtime, not by the model's memory.

**2. LangGraph StateGraph for orchestration.**  
A two-node graph: `classify → analyze | visualize`. The intent classification is itself an LLM call that receives the real column names and sample data — so it routes correctly for any phrasing without brittle keyword lists.

**3. Full schema in every prompt.**  
Every LLM call receives column names, dtypes, null counts, unique counts, and 5 sample rows for every uploaded file. The model generates code using the actual column names — no assumptions, no hardcoded mappings.

**4. FastAPI serves the SPA directly.**  
No Streamlit, no separate frontend process. `GET /` returns a single HTML file; `POST /api/query` runs the agent. One `uvicorn` command starts everything.

**5. Multi-sheet Excel awareness.**  
`pd.read_excel(sheet_name=None)` reads all sheets. A 3-sheet workbook becomes three queryable datasets (`report[Sales]`, `report[Expenses]`, `report[Summary]`), all visible in the sidebar.

---

## Stack

| Layer | Choice |
|---|---|
| LLM | ExperientialLabs `gpt-6-astra` via OpenAI-compatible client |
| Agent | LangGraph `StateGraph` (Python 3.14) |
| Backend | FastAPI + uvicorn |
| Frontend | Vanilla HTML/CSS/JS SPA (Chart.js, marked.js) |
| Data | pandas, openpyxl, xlrd |

---

## What I'd Build Next

**Conversation memory** — the current session is stateless within a query. Feeding the last 3–5 exchanges into each LLM call would let users ask follow-ups like "now filter that to Q4 only."

**Auto-chart on analysis results** — when the LLM analysis produces a numeric table, automatically offer an interactive Chart.js chart without the user having to re-ask "show this as a chart."

**SQL export** — let users download the LLM-generated pandas code as a SQL query so they can run it against their production database directly.

**Streaming responses** — stream the LLM output token-by-token to the browser so long analysis tasks feel instant rather than making the user wait for the full response.

**Persistent sessions** — swap the in-memory store for Redis or SQLite so file uploads survive server restarts and users can return to a previous session.

---

## Run

```bash
# one-time setup
uv python install 3.14
uv venv --python 3.14 --seed --clear
uv pip install -r requirements.txt

# start (single command)
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** — upload any CSV or Excel, ask anything.
