"""
prompt.py
---------
Versioned prompt registry for DataLens AI.

Instead of scattered f-strings, all prompt templates live here,
versioned and named so they can be audited, pinned, and A/B tested.

Usage:
    from prompt import PromptRegistry
    pr = PromptRegistry()
    system_msg = pr.get("analysis_system", version="v2")
    user_msg   = pr.render("analysis_user",  schema=..., query=..., history=...)
"""

from __future__ import annotations

from string import Template
from typing import Any, Dict, Optional


# ── Registry ───────────────────────────────────────────────────────────────────

class PromptRegistry:
    """
    Central store of named, versioned prompt templates.

    Templates use Python's string.Template syntax ($variable).
    Call render(name, **kwargs) for substitution.
    """

    _templates: Dict[str, Dict[str, str]] = {

        # ── Intent classification ──────────────────────────────────────────
        "intent_classifier": {
            "v1": (
                "You are an intent classifier for a data analytics tool.\n\n"
                "Uploaded datasets:\n$schema\n\n"
                "User query: \"$query\"\n\n"
                "Should this query be answered as:\n"
                "  VISUALIZE — the user explicitly wants a chart, graph, plot, or visual output\n"
                "  ANALYZE   — the user wants any kind of text, number, table, list, summary, or insight\n\n"
                "Reply with exactly one word: VISUALIZE or ANALYZE"
            ),
        },

        # ── Analysis system message ────────────────────────────────────────
        "analysis_system": {
            "v2": (
                "You are an expert Python data analyst who produces clear, human-readable reports.\n"
                "When given a user question, write pandas code that computes the answer AND presents it in\n"
                "a well-formatted, self-explanatory way.\n\n"
                "IMPORTANT OUTPUT RULES:\n"
                "  - Never print a bare number alone. Always add a label/heading explaining what it is.\n"
                "  - Use a descriptive heading (e.g. '### Total Revenue') before each result.\n"
                "  - Show a per-file breakdown when data spans multiple files.\n"
                "  - End with a bold summary line (e.g. '**Grand Total: 1,541,000**').\n"
                "  - Format numbers with commas and 2 decimal places where appropriate.\n"
                "  - Use markdown: **bold** for key values, bullet lists for breakdowns.\n"
                "  - Use the EXACT column and variable names from the schema — never invent names.\n"
                "  - For cross-file work: pd.merge() for joins, pd.concat() for unions.\n"
                "  - Handle edge cases gracefully. Use .to_string() for DataFrame output.\n"
                "SAFETY RULES (STRICT):\n"
                "  - Do NOT import os, sys, subprocess, socket, or any module not in the provided env.\n"
                "  - Do NOT use open(), eval(), exec(), compile(), or __import__().\n"
                "  - Do NOT access __builtins__, __class__, __subclasses__, or similar dunder attributes.\n"
                "  - Only use: pd, np, io, base64, re, math, datetime, collections, itertools, functools."
            ),
        },

        # ── Analysis user message ──────────────────────────────────────────
        "analysis_user": {
            "v2": (
                "$history\n\n"
                "Available datasets (each is a pandas DataFrame):\n"
                "$schema\n"
                "$join_hints\n\n"
                "Variable names to use in code: $var_names\n"
                "pandas = `pd` | numpy = `np` | Do NOT import anything else.\n\n"
                "User question: $query\n\n"
                "Write Python pandas code that answers this question AND presents the result clearly.\n\n"
                "Output rules (MUST follow ALL of them):\n"
                "  1. Use ONLY the variable names and column names from the schema above.\n"
                "  2. NEVER print a bare number — always include a label or heading.\n"
                "  3. Start each result section with a descriptive heading:\n"
                "       print('### Total Revenue from Sales')\n"
                "  4. Show a per-file breakdown when the question spans multiple files.\n"
                "  5. End with a bold summary line:\n"
                "       print(f'**Grand Total: {total:,.2f}**')\n"
                "  6. Wrap ALL code in a single ```python ... ``` block.\n"
                "  7. For cross-file questions use pd.merge() or pd.concat() as appropriate.\n"
                "  8. Use markdown — **bold** for key values, bullet points for lists.\n"
                "  9. If the question mentions 'all' files, compute results for every relevant file."
            ),
        },

        # ── Analysis retry user message (after code exec failure) ─────────
        "analysis_retry": {
            "v1": (
                "The code you previously generated produced this error:\n"
                "```\n$error\n```\n\n"
                "Please fix the code. Common causes:\n"
                "  - Wrong column name (check the schema — use exact names)\n"
                "  - Wrong variable name (use only those listed)\n"
                "  - Type mismatch (cast with .astype() if needed)\n"
                "  - Missing join key (use pd.merge() with the keys shown)\n\n"
                "Original question: $query\n\n"
                "Schema:\n$schema\n\n"
                "Variable names: $var_names\n\n"
                "Provide the corrected code in a single ```python ... ``` block."
            ),
        },

        # ── Visualization system message ───────────────────────────────────
        "visualization_system": {
            "v1": (
                "You generate clean, dark-themed matplotlib Python plots.\n"
                "SAFETY RULES (STRICT):\n"
                "  - Do NOT import os, sys, subprocess, or any module not provided.\n"
                "  - Do NOT use open(), eval(), exec(), compile(), or __import__().\n"
                "  - Only use: pd, np, plt, io, base64, seaborn (if provided)."
            ),
        },

        # ── Visualization user message ─────────────────────────────────────
        "visualization_user": {
            "v1": (
                "You are a data visualization expert. Create a beautiful matplotlib chart.\n"
                "You have $n_datasets dataset(s).\n\n"
                "$history\n\n"
                "SCHEMAS AND SAMPLE DATA:\n$schema\n\n"
                "VARIABLE NAMES TO USE IN CODE:\n$var_map\n\n"
                "USER REQUEST:\n$query\n\n"
                "Generate Python matplotlib code:\n"
                "- Use variable names listed above EXACTLY.\n"
                "- pandas=pd, matplotlib.pyplot=plt, io=io, base64=base64, numpy=np are pre-imported.\n"
                "- Dark theme: set figure face color to '#0D1117', axes face color to '#161B22'\n"
                "- Use this color palette: ['#8B5CF6','#06B6D4','#F59E0B','#10B981','#EF4444','#A78BFA','#38BDF8']\n"
                "- Add title, axis labels, and legend.\n"
                "- KEEP IT SIMPLE. Do NOT add data labels or text annotations.\n"
                "- Use plt.tight_layout()\n\n"
                "End code with EXACTLY:\n"
                "```\n"
                "buf = io.BytesIO()\n"
                "plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0D1117')\n"
                "plt.close()\n"
                "buf.seek(0)\n"
                "print(base64.b64encode(buf.getvalue()).decode('utf-8'))\n"
                "```\n\n"
                "Wrap ALL code in triple backticks.\n"
                "If the query cannot be answered, print \"Error: <reason>\"."
            ),
        },

        # ── Visualization retry ────────────────────────────────────────────
        "visualization_retry": {
            "v1": (
                "The visualization code you generated produced this error:\n"
                "```\n$error\n```\n\n"
                "Fix the code. Ensure:\n"
                "  - All variable names match exactly those listed in the schema\n"
                "  - matplotlib state is clean (use fig, ax = plt.subplots())\n"
                "  - The last 4 lines save and print the base64 image\n\n"
                "Original request: $query\n\n"
                "Provide only the corrected code in a single ```python ... ``` block."
            ),
        },

        # ── Plain-text fallback ────────────────────────────────────────────
        "analysis_fallback": {
            "v1": (
                "You are a helpful data analyst. Answer in clear plain English or markdown tables.\n"
                "Given these datasets:\n$schema\n\n"
                "Please answer in plain text (no code):\n$query"
            ),
        },
    }

    _current_versions: Dict[str, str] = {
        "intent_classifier": "v1",
        "analysis_system": "v2",
        "analysis_user": "v2",
        "analysis_retry": "v1",
        "visualization_system": "v1",
        "visualization_user": "v1",
        "visualization_retry": "v1",
        "analysis_fallback": "v1",
    }

    def get(self, name: str, version: Optional[str] = None) -> str:
        """Return the raw template string for *name* at *version* (default: current)."""
        v = version or self._current_versions.get(name, "v1")
        try:
            return self._templates[name][v]
        except KeyError:
            raise ValueError(f"Unknown prompt '{name}' version '{v}'")

    def render(self, name: str, version: Optional[str] = None, **kwargs: Any) -> str:
        """Return the template with $variables substituted from kwargs."""
        template_str = self.get(name, version)
        try:
            return Template(template_str).safe_substitute(**kwargs)
        except Exception as exc:
            raise ValueError(f"Prompt render error for '{name}': {exc}") from exc

    def list_prompts(self) -> list:
        """Return all registered prompt names and their available versions."""
        return [
            {"name": name, "versions": list(versions.keys()), "current": self._current_versions.get(name)}
            for name, versions in self._templates.items()
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────

registry = PromptRegistry()