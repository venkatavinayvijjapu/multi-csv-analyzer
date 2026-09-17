"""
api_endpoints/visualize.py
---------------------------
Visualization endpoint using ExperientialLabs (OpenAI-compatible) API.
Exposes both:
  - run_visualization(dfs, query) -> (image_base64, error)  — called by LangGraph agent
  - POST /visualize                                          — HTTP endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import base64
import io
import re
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# ─── Setup ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG, filename="visualize.log", filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()
EXPLABS_API_KEY = os.getenv("EXPLABS_API_KEY") or os.getenv("GROQ_API_KEY", "")
EXPLABS_BASE_URL = os.getenv("EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1")
EXPLABS_MODEL = os.getenv("EXPLABS_MODEL") or os.getenv("GROQ_MODEL", "gpt-5.6-luna")

if not EXPLABS_API_KEY:
    raise ValueError("EXPLABS_API_KEY not found in .env file.")

client = OpenAI(api_key=EXPLABS_API_KEY, base_url=EXPLABS_BASE_URL)

router = APIRouter()


# ─── Helpers ─────────────────────────────────────────────────────────────────

class FilePayload(BaseModel):
    name: str
    data_base64: str


class VisualizeRequest(BaseModel):
    files: List[FilePayload]
    query: str


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


def _safe_var_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    if safe and safe[0].isdigit():
        safe = "df_" + safe
    return safe or "df"


def _build_schema(dfs: dict) -> str:
    parts = []
    for name, df in dfs.items():
        col_info = "\n".join(f"  - {col}: {df[col].dtype}" for col in df.columns)
        sample = df.head(3).to_string(index=False, max_cols=10)
        parts.append(
            f"DataFrame '{name}':\n"
            f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            f"  Columns:\n{col_info}\n"
            f"  Sample:\n{sample}"
        )
    return "\n\n".join(parts)


# ─── Core Visualization Function ─────────────────────────────────────────────

def run_visualization(dfs: dict, query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Core visualization logic callable directly by the LangGraph agent.
    Returns (image_base64, error_message). One will always be None.
    """
    if not dfs:
        return None, "No datasets available."

    schema = _build_schema(dfs)
    exec_env: dict = {
        "pd": pd, "plt": plt, "io": io,
        "base64": base64, "np": __import__("numpy"),
    }
    var_map_lines = []
    for name, df in dfs.items():
        var = _safe_var_name(name)
        exec_env[var] = df.copy()
        var_map_lines.append(f"  - '{name}' → variable `{var}`")

    var_map_str = "\n".join(var_map_lines)

    prompt = f"""You are a data visualization expert. Create a beautiful matplotlib chart.
You have {len(dfs)} dataset(s).

SCHEMAS AND SAMPLE DATA:
{schema}

VARIABLE NAMES TO USE IN CODE:
{var_map_str}

USER REQUEST:
{query}

Generate Python matplotlib code:
- Use variable names listed above EXACTLY.
- pandas=pd, matplotlib.pyplot=plt, io=io, base64=base64, numpy=np are pre-imported.
- Dark theme: set figure face color to '#0D1117', axes face color to '#161B22'
- Use this color palette: ['#8B5CF6','#06B6D4','#F59E0B','#10B981','#EF4444','#A78BFA','#38BDF8']
- Add title, axis labels, and legend.
- KEEP IT SIMPLE. Do NOT add data labels or text annotations (avoid plt.text, ax.annotate, xytext, padding).
- Use plt.tight_layout()

End code with EXACTLY:
```
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
buf.seek(0)
print(base64.b64encode(buf.getvalue()).decode('utf-8'))
```

Wrap ALL code in triple backticks.
If the query cannot be answered, print "Error: <reason>".
"""

    try:
        logger.debug(f"Calling ExperientialLabs for visualization: {query[:100]}")
        response = client.chat.completions.create(
            model=EXPLABS_MODEL,
            messages=[
                {"role": "system", "content": "You generate clean, dark-themed matplotlib Python plots."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content or ""

        match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
        if not match:
            return None, "No valid Python code returned."

        code = match.group(1).strip()
        logger.debug(f"Executing visualization code:\n{code[:500]}")

        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, exec_env)  # nosec
            img_b64 = stdout.getvalue().strip()
            if not img_b64 or img_b64.startswith("Error"):
                return None, img_b64 or "No image generated."
            return img_b64, None
        except Exception as exec_err:
            logger.error(f"Viz exec failed: {exec_err}")
            return None, str(exec_err)

    except Exception as e:
        logger.error(f"Visualization error: {e}")
        return None, str(e)


# ─── HTTP Endpoint ────────────────────────────────────────────────────────────

@router.post("/visualize")
async def visualize_endpoint(payload: VisualizeRequest):
    try:
        dfs = _decode_dataframes(payload.files)
        img_b64, error = run_visualization(dfs, payload.query)
        if img_b64:
            return {"image_base64": img_b64}
        raise HTTPException(status_code=400, detail=error or "Visualization failed.")
    except HTTPException:
        raise
    except Exception as ex:
        logger.error(f"Visualize endpoint error: {ex}")
        raise HTTPException(status_code=400, detail=str(ex))