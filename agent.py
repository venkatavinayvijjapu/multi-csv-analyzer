"""
agent.py
--------
LangGraph agent — ZERO static keyword matching.

All queries, regardless of phrasing, column names, or data structure,
are handled entirely by the ExperientialLabs LLM which:
  1. Reads the actual schema (column names, dtypes, sample rows)
  2. Generates precise pandas / matplotlib code
  3. Executes the code for an exact answer

Flow:
  START → intent_classify_node   (1 fast LLM call — ANALYZE or VISUALIZE?)
            ├── VISUALIZE → visualization_node  (LLM writes matplotlib code)
            └── ANALYZE  → analysis_node        (LLM writes pandas code)
"""

from __future__ import annotations

import logging
import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import session_store
from delta_analysis import check_delta

load_dotenv()
logger = logging.getLogger(__name__)

EXPLABS_API_KEY  = os.getenv("EXPLABS_API_KEY", "")
EXPLABS_BASE_URL = os.getenv("EXPLABS_BASE_URL", "https://api.experientiallabs.ai/v1")
EXPLABS_MODEL    = os.getenv("EXPLABS_MODEL", "gpt-5.6-luna")


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    session_id: str
    result: Optional[str]
    image_base64: Optional[str]
    chart_data: Optional[dict]
    delta_used: bool   # always False; kept for API response compat
    intent: str        # "analysis" | "visualization" | "error"


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm: Optional[ChatOpenAI] = None

def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=EXPLABS_MODEL,
            api_key=EXPLABS_API_KEY,
            base_url=EXPLABS_BASE_URL,
            temperature=0,
        )
    return _llm


# ── Schema builder ────────────────────────────────────────────────────────────

def _schema_summary(session_id: str) -> str:
    dfs = session_store.get_session(session_id)
    lines = []
    for name, df in dfs.items():
        col_info = ", ".join(
            f"{c} ({df[c].dtype})" for c in df.columns
        )
        lines.append(f"• {name}: {len(df)} rows — columns: {col_info}")
    return "\n".join(lines) if lines else "(no datasets)"



# ── Node 0: Structural check (no LLM — instant pandas for schema queries) ─────

def structural_check_node(state: AgentState) -> dict:
    """
    Intercepts schema/structural queries (describe, nulls, duplicates, correlation)
    and answers them instantly using pandas — no LLM call, zero latency.
    For all other queries returns delta_used=False to fall through to LLM.
    """
    dfs = session_store.get_session(state["session_id"])
    if not dfs:
        return {
            "result":       "❌ No datasets found. Please upload files first.",
            "delta_used":   True,
            "intent":       "error",
            "image_base64": None,
            "chart_data":   None,
        }

    result, chart_data = check_delta(state["query"], dfs)
    if result is not None:
        logger.info(f"Structural fast-path: {state['query'][:60]}")
        return {
            "result":       result,
            "chart_data":   chart_data,
            "image_base64": None,
            "delta_used":   True,
            "intent":       "structural",
        }

    return {"delta_used": False, "result": None}


def _route_after_structural(state: AgentState) -> str:
    return "end" if state.get("delta_used") else "classify"


# ── Node 1: Intent classification (LLM-based, no keywords) ───────────────────

def intent_classify_node(state: AgentState) -> dict:
    """
    One fast LLM call.
    The model sees the real column names + query and decides:
      VISUALIZE → user wants a chart / graph / plot
      ANALYZE   → user wants a number, table, text, list, or any other answer
    """
    dfs = session_store.get_session(state["session_id"])
    if not dfs:
        return {
            "result":       "❌ No datasets found. Please upload files first.",
            "intent":       "error",
            "image_base64": None,
            "chart_data":   None,
            "delta_used":   False,
        }

    schema = _schema_summary(state["session_id"])

    prompt = f"""You are an intent classifier for a data analytics tool.

Uploaded datasets:
{schema}

User query: "{state['query']}"

Should this query be answered as:
  VISUALIZE — the user explicitly wants a chart, graph, plot, or visual output
  ANALYZE   — the user wants any kind of text, number, table, list, summary, or insight

Reply with exactly one word: VISUALIZE or ANALYZE"""

    try:
        response = _get_llm().invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().upper()
        intent = "visualization" if "VISUAL" in decision else "analysis"
        logger.info(f"Intent [{intent}]: {state['query'][:70]}")
    except Exception as exc:
        logger.warning(f"Intent classification failed ({exc}), defaulting to analysis")
        intent = "analysis"

    return {"intent": intent}


def _route_from_intent(state: AgentState) -> str:
    if state.get("intent") == "error":
        return "end"
    return "visualize" if state["intent"] == "visualization" else "analyze"


# ── Node 2a: Analysis (LLM writes pandas code → execute) ─────────────────────

def analysis_node(state: AgentState) -> dict:
    from api_endpoints.analysis import run_analysis

    dfs = session_store.get_session(state["session_id"])
    if not dfs:
        return {"result": "❌ No datasets in session.",
                "intent": "analysis", "image_base64": None, "chart_data": None}

    logger.info(f"Analysis node: {state['query'][:70]}")
    result = run_analysis(dfs, state["query"])
    return {"result": result, "intent": "analysis",
            "image_base64": None, "chart_data": None, "delta_used": False}


# ── Node 2b: Visualization (LLM writes matplotlib code → execute) ─────────────

def visualization_node(state: AgentState) -> dict:
    from api_endpoints.visualize import run_visualization

    dfs = session_store.get_session(state["session_id"])
    if not dfs:
        return {"result": "❌ No datasets in session.",
                "intent": "visualization", "image_base64": None, "chart_data": None}

    logger.info(f"Visualization node: {state['query'][:70]}")
    img_b64, error = run_visualization(dfs, state["query"])

    if img_b64:
        return {"image_base64": img_b64, "result": "",
                "intent": "visualization", "chart_data": None, "delta_used": False}
    return {"result": f"⚠️ Could not generate chart: {error}",
            "intent": "visualization", "image_base64": None,
            "chart_data": None, "delta_used": False}


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph():
    wf = StateGraph(AgentState)

    wf.add_node("structural", structural_check_node)
    wf.add_node("classify",   intent_classify_node)
    wf.add_node("analyze",    analysis_node)
    wf.add_node("visualize",  visualization_node)

    wf.set_entry_point("structural")

    wf.add_conditional_edges(
        "structural",
        _route_after_structural,
        {"end": END, "classify": "classify"},
    )
    wf.add_conditional_edges(
        "classify",
        _route_from_intent,
        {"end": END, "analyze": "analyze", "visualize": "visualize"},
    )
    wf.add_edge("analyze",   END)
    wf.add_edge("visualize", END)

    return wf.compile()


_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = _build_graph()
    return _agent


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(session_id: str, query: str) -> dict:
    """Called by POST /api/query."""
    initial: AgentState = {
        "query":        query,
        "session_id":   session_id,
        "result":       None,
        "image_base64": None,
        "chart_data":   None,
        "delta_used":   False,
        "intent":       "unknown",
    }
    try:
        final = get_agent().invoke(initial)
    except Exception as exc:
        logger.error(f"Agent error: {exc}")
        return {"result": f"❌ Agent error: {exc}",
                "image_base64": None, "chart_data": None,
                "delta_used": False, "intent": "error"}

    return {
        "result":       final.get("result") or "No answer produced.",
        "image_base64": final.get("image_base64"),
        "chart_data":   final.get("chart_data"),
        "delta_used":   final.get("delta_used", False),
        "intent":       final.get("intent", "unknown"),
    }
