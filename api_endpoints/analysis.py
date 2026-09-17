"""
api_endpoints/analysis.py
--------------------------
LLM-powered analysis endpoint.

• run_analysis(dfs, query) → str   — called directly by the LangGraph agent
• POST /analyze              — HTTP endpoint (kept for compatibility)

The LLM receives:
  - Full schema: column names, dtypes, row count for every uploaded file
  - Sample rows (first 5) so it understands data shape
  - Exact variable names to use in generated code

This means it can answer ANY question about ANY dataset with ANY column names.
No keyword matching. No column-name assumptions. Pure LLM code generation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import base64
import io
import re
import contextlib
import os
import logging
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG, filename="analysis.log", filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()
EXPLABS_API_KEY  = os.getenv("EXPLABS_API_KEY") or os.getenv("GROQ_API_KEY", "")
EXPLABS_BASE_URL = os.getenv("EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1")
EXPLABS_MODEL    = os.getenv("EXPLABS_MODEL") or os.getenv("GROQ_MODEL", "gpt-5.6-luna")

if not EXPLABS_API_KEY:
    raise ValueError("EXPLABS_API_KEY not found in .env")

client = OpenAI(api_key=EXPLABS_API_KEY, base_url=EXPLABS_BASE_URL)
router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────────

class FilePayload(BaseModel):
    name: str
    data_base64: str

class AnalysisRequest(BaseModel):
    files: List[FilePayload]
    query: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _decode_dataframes(files: List[FilePayload]) -> dict:
    dfs = {}
    for f in files:
        raw = base64.b64decode(f.data_base64)
        if f.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
        dfs[f.name] = df
    return dfs


def _safe_var(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    return ("df_" + safe) if (safe and safe[0].isdigit()) else (safe or "df")


def _find_join_keys(var_map: dict) -> list:
    """
    Detect columns that appear in more than one DataFrame — these are
    candidate join keys. Returns list of (col_name, [var_names]) tuples.
    """
    from collections import defaultdict
    col_to_vars = defaultdict(list)
    for var_name, df in var_map.items():
        for col in df.columns:
            col_to_vars[col.lower()].append((var_name, col))
    # Only return cols found in ≥ 2 DataFrames
    return [
        (col_lower, occurrences)
        for col_lower, occurrences in col_to_vars.items()
        if len(occurrences) >= 2
    ]


def _build_full_schema(dfs: dict) -> tuple[str, dict]:
    """
    Returns (schema_text, var_map) where var_map is {variable_name: df}.
    Schema includes: column names, dtypes, null/unique counts, sample rows,
    and auto-detected cross-file join keys.
    """
    schema_parts = []
    var_map = {}

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

    # Append cross-file join key hints
    join_hints = ""
    if len(var_map) > 1:
        join_keys = _find_join_keys(var_map)
        if join_keys:
            key_lines = []
            for col_lower, occurrences in join_keys:
                vars_with_col = ", ".join(
                    f"`{vn}`['{col}']" for vn, col in occurrences
                )
                key_lines.append(f"  • '{col_lower}' appears in: {vars_with_col}")
            join_hints = (
                "\n\n" + "─" * 60 +
                "\n🔗 AUTO-DETECTED POTENTIAL JOIN KEYS (columns shared across files):\n"
                + "\n".join(key_lines) +
                "\n  Use pd.merge() or pd.concat() to combine these datasets."
            )

    return "\n\n" + "─" * 60 + "\n\n".join(schema_parts) + join_hints, var_map


# ── Core analysis function ─────────────────────────────────────────────────────

def run_analysis(dfs: dict, query: str) -> str:
    """
    Sends the full dataset schema + user query to ExperientialLabs.
    The LLM generates pandas code that is then executed to produce an exact answer.
    Handles ANY question about ANY dataset — no column assumptions, no keyword matching.
    """
    if not dfs:
        return "❌ No datasets available."

    schema_text, var_map = _build_full_schema(dfs)

    # Prepare exec environment
    exec_env: dict = {"pd": pd, "np": __import__("numpy")}
    for var_name, df in var_map.items():
        exec_env[var_name] = df.copy()

    var_list = "\n".join(f"  • `{v}` → {dfs.get(v, list(dfs.keys())[0])}" for v in var_map)

    system_message = (
        "You are an expert Python data analyst who produces clear, human-readable reports. "
        "When given a user question, write pandas code that computes the answer AND presents it in "
        "a well-formatted, self-explanatory way. "
        "IMPORTANT OUTPUT RULES: "
        "  - Never print a bare number alone. Always add a label/heading explaining what it is. "
        "  - Use a descriptive heading (e.g. '### Total Revenue') before each result. "
        "  - Show a per-file breakdown when data spans multiple files. "
        "  - End with a bold summary line (e.g. '**Grand Total: 1,541,000**'). "
        "  - Format numbers with commas and 2 decimal places where appropriate. "
        "  - Use markdown: **bold** for key values, bullet lists for breakdowns, tables for comparisons. "
        "  - Use the EXACT column and variable names from the schema — never invent names. "
        "  - For cross-file work: pd.merge() for joins, pd.concat() for unions/stacking. "
        "  - Handle edge cases gracefully. Use .to_string() for DataFrame output."
    )

    n_files = len(var_map)
    var_names_str = ", ".join(f"`{v}`" for v in var_map)

    multi_file_hint = ""
    if n_files > 1:
        join_keys = _find_join_keys(var_map)
        if join_keys:
            example_key = join_keys[0][0]
            example_vars = join_keys[0][1]
            va, ca = example_vars[0]
            vb, cb = example_vars[1]
            multi_file_hint = f"""
⚠️  MULTI-FILE CONTEXT: You have {n_files} datasets. For questions that span files:
  - JOIN example:  merged = {va}.merge({vb}, left_on='{ca}', right_on='{cb}', how='left')
  - UNION example: all_data = pd.concat([{', '.join(v for v, _ in example_vars)}], ignore_index=True)
  - Always use the detected join keys shown in the schema above.
"""
        else:
            multi_file_hint = f"""
⚠️  MULTI-FILE CONTEXT: You have {n_files} datasets. If columns have compatible types,
  use pd.concat() for unions or pd.merge() for joins.
"""

    user_message = f"""Available datasets (each is a pandas DataFrame):
{schema_text}
{multi_file_hint}
Variable names to use in code: {var_names_str}
pandas = `pd` | numpy = `np` | Do NOT import anything else.

User question: {query}

Write Python pandas code that answers this question AND presents the result clearly.

Output rules (MUST follow ALL of them):
  1. Use ONLY the variable names and column names from the schema above.
  2. NEVER print a bare number — always include a label or heading before every value.
  3. Start each result section with a descriptive heading, e.g.:
       print('### Total Revenue from Sales')
  4. Show a per-file breakdown when the question spans multiple files:
       print(f'  • sales: {{sales["revenue"].sum():,.2f}}')
       print(f'  • business_data_Sales: {{business_data_Sales["revenue"].sum():,.2f}}')
  5. End with a bold summary line:
       print(f'**Grand Total: {{total:,.2f}}**')
  6. Wrap ALL code in a single ```python ... ``` block.
  7. For cross-file questions use pd.merge() or pd.concat() as appropriate.
  8. Use markdown — **bold** for key values, bullet points for lists, tables for comparisons.
  9. If the question mentions 'all' files, compute and show results for every relevant file."""


    try:
        logger.debug(f"Calling LLM for analysis: {query[:100]}")
        response = client.chat.completions.create(
            model=EXPLABS_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        logger.debug(f"LLM response (first 400 chars): {raw[:400]}")

        # Extract code block
        match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        if not match:
            # LLM gave a plain text answer — return it directly
            return raw.strip()

        code = match.group(1).strip()
        logger.debug(f"Executing:\n{code}")

        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(code, exec_env)  # nosec
            output = stdout_buf.getvalue().strip()
            if output:
                return output
            return "✅ Analysis complete — no printed output was produced."

        except Exception as exec_err:
            logger.warning(f"Code execution failed: {exec_err}. Attempting plain-text fallback.")
            # Fallback: ask the LLM for a plain-text answer
            fb = client.chat.completions.create(
                model=EXPLABS_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst. Answer in clear plain English or markdown tables."},
                    {"role": "user",   "content": f"Given these datasets:\n{schema_text}\n\nPlease answer in plain text (no code):\n{query}"},
                ],
                temperature=0,
            )
            return fb.choices[0].message.content or "Could not analyze the data."

    except Exception as exc:
        logger.error(f"Analysis error: {exc}")
        return f"❌ Analysis failed: {exc}"


# ── HTTP Endpoint ──────────────────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_endpoint(payload: AnalysisRequest):
    try:
        dfs = _decode_dataframes(payload.files)
        result = run_analysis(dfs, payload.query)
        return {"result": result}
    except Exception as exc:
        logger.error(f"Endpoint error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))