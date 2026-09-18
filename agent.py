"""
agent.py
--------
LangGraph agent — production-grade version.

Improvements over original:
  • Conversation history stored in AgentState and persisted via core.memory
  • session_id threaded through all nodes so memory / provenance tracking works
  • All queries, regardless of phrasing, are handled by the LLM which:
      1. Reads the actual schema (column names, dtypes, sample rows)
      2. Generates precise pandas / matplotlib code
      3. Executes the code in a secure sandbox for an exact answer

Flow:
  START → structural_check_node   (fast pandas — schema queries)
              ├── DONE   → END
              └── PASS   → intent_classify_node  (1 fast LLM call)
                              ├── VISUALIZE → visualization_node
                              └── ANALYZE  → analysis_node
"""

from __future__ import annotations

import os
from typing import Optional, TypedDict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import session_store
from core.logging_config import get_logger
from core.memory import add_exchange, get_history
from core.metrics import record_metric, LatencyTimer
from delta_analysis import check_delta
from prompt import registry as prompt_registry

load_dotenv()
logger = get_logger(__name__)

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
    delta_used: bool
    intent: str                  # "analysis" | "visualization" | "structural" | "error"
    conversation_history: List[dict]   # injected for prompt context


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
        col_info = ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
        lines.append(f"• {name}: {len(df)} rows — columns: {col_info}")
    return "\n".join(lines) if lines else "(no datasets)"


# ── Node 0: Structural check ──────────────────────────────────────────────────

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

    # Load conversation history for the session
    history = get_history(state["session_id"], max_turns=5)

    result, chart_data = check_delta(state["query"], dfs, session_id=state["session_id"])
    if result is not None:
        logger.info(f"Structural fast-path: {state['query'][:60]}")

        # Persist this exchange
        if state["session_id"]:
            add_exchange(state["session_id"], state["query"], result)

        record_metric(
            session_id=state["session_id"],
            operation="structural",
            success=True,
            latency_ms=0,
        )
        return {
            "result":                result,
            "chart_data":            chart_data,
            "image_base64":          None,
            "delta_used":            True,
            "intent":                "structural",
            "conversation_history":  history,
        }

    return {
        "delta_used":           False,
        "result":               None,
        "conversation_history": history,
    }


def _route_after_structural(state: AgentState) -> str:
    return "end" if state.get("delta_used") else "classify"


# ── Node 1: Intent classification ────────────────────────────────────────────

def intent_classify_node(state: AgentState) -> dict:
    """
    One fast LLM call.  Decides VISUALIZE or ANALYZE.
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
    prompt = prompt_registry.render(
        "intent_classifier",
        schema=schema,
        query=state["query"],
    )

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


# ── Node 2a: Analysis ─────────────────────────────────────────────────────────

def analysis_node(state: AgentState) -> dict:
    from api_endpoints.analysis import run_analysis

    dfs = session_store.get_session(state["session_id"])
    if not dfs:
        return {
            "result": "❌ No datasets in session.",
            "intent": "analysis", "image_base64": None, "chart_data": None,
        }

    logger.info(f"Analysis node: {state['query'][:70]}")

    with LatencyTimer() as timer:
        result = run_analysis(dfs, state["query"], session_id=state["session_id"])

    # Persist exchange to memory
    if state["session_id"]:
        add_exchange(state["session_id"], state["query"], result)

    return {
        "result": result, "intent": "analysis",
        "image_base64": None, "chart_data": None, "delta_used": False,
    }


# ── Node 2b: Visualization ────────────────────────────────────────────────────

def visualization_node(state: AgentState) -> dict:
    from api_endpoints.visualize import run_visualization

    dfs = session_store.get_session(state["session_id"])
    if not dfs:
        return {
            "result": "❌ No datasets in session.",
            "intent": "visualization", "image_base64": None, "chart_data": None,
        }

    logger.info(f"Visualization node: {state['query'][:70]}")
    img_b64, error = run_visualization(dfs, state["query"], session_id=state["session_id"])

    answer = "✅ Chart generated." if img_b64 else f"⚠️ Could not generate chart: {error}"

    # Persist exchange to memory
    if state["session_id"]:
        add_exchange(state["session_id"], state["query"], answer)

    if img_b64:
        return {
            "image_base64": img_b64, "result": answer,
            "intent": "visualization", "chart_data": None, "delta_used": False,
        }
    return {
        "result": answer, "intent": "visualization",
        "image_base64": None, "chart_data": None, "delta_used": False,
    }


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
    # Pre-load history to put in initial state
    history = get_history(session_id, max_turns=5)

    initial: AgentState = {
        "query":                query,
        "session_id":           session_id,
        "result":               None,
        "image_base64":         None,
        "chart_data":           None,
        "delta_used":           False,
        "intent":               "unknown",
        "conversation_history": history,
    }
    try:
        final = get_agent().invoke(initial)
    except Exception as exc:
        logger.error(f"Agent error: {exc}")
        return {
            "result": f"❌ Agent error: {exc}",
            "image_base64": None, "chart_data": None,
            "delta_used": False, "intent": "error",
        }

    return {
        "result":       final.get("result") or "No answer produced.",
        "image_base64": final.get("image_base64"),
        "chart_data":   final.get("chart_data"),
        "delta_used":   final.get("delta_used", False),
        "intent":       final.get("intent", "unknown"),
    }
