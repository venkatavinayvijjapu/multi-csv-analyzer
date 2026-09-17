"""
smart_chart.py
--------------
FastAPI router that converts chart_data dicts (from the delta layer or
AI analysis) into base64-encoded matplotlib PNG images.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import base64
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

PALETTE = [
    "#6C63FF", "#FF6584", "#43B89C", "#F7B731", "#A29BFE",
    "#FD7272", "#55EFC4", "#FDCB6E", "#74B9FF", "#E17055",
]


class ChartDataset(BaseModel):
    label: str
    values: List[float]


class ChartRequest(BaseModel):
    chart_type: str  # "bar" | "line" | "pie" | "scatter" | "histogram"
    labels: List[str]
    datasets: List[ChartDataset]
    title: Optional[str] = ""
    x_label: Optional[str] = ""
    y_label: Optional[str] = ""


def _apply_style(ax, title: str, x_label: str, y_label: str):
    ax.set_facecolor("#1a1a2e")
    ax.figure.set_facecolor("#16213e")
    ax.tick_params(colors="#e0e0e0", labelsize=9)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#333", linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title, color="#ffffff", fontsize=13, pad=12, fontweight="bold")
    if x_label:
        ax.set_xlabel(x_label, color="#aaa", fontsize=10)
    if y_label:
        ax.set_ylabel(y_label, color="#aaa", fontsize=10)


@router.post("/smart_chart")
async def generate_smart_chart(req: ChartRequest):
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        n_datasets = len(req.datasets)
        n_labels = len(req.labels)

        if req.chart_type == "bar":
            x = np.arange(n_labels)
            width = 0.8 / max(n_datasets, 1)
            for i, ds in enumerate(req.datasets):
                vals = ds.values[:n_labels]
                offset = (i - n_datasets / 2 + 0.5) * width
                bars = ax.bar(
                    x + offset, vals, width=width * 0.9,
                    label=ds.label, color=PALETTE[i % len(PALETTE)],
                    alpha=0.88, zorder=3
                )
                for bar in bars:
                    h = bar.get_height()
                    ax.annotate(
                        f"{h:,.2g}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color="#ddd", fontsize=7
                    )
            ax.set_xticks(x)
            ax.set_xticklabels(req.labels, rotation=30, ha="right", color="#ccc")

        elif req.chart_type == "line":
            for i, ds in enumerate(req.datasets):
                ax.plot(
                    req.labels[:len(ds.values)], ds.values,
                    label=ds.label, color=PALETTE[i % len(PALETTE)],
                    linewidth=2, marker="o", markersize=4
                )
            ax.set_xticklabels(req.labels, rotation=30, ha="right", color="#ccc")

        elif req.chart_type == "pie":
            ds = req.datasets[0] if req.datasets else None
            if ds:
                wedges, texts, autotexts = ax.pie(
                    ds.values, labels=req.labels,
                    colors=PALETTE[:len(ds.values)],
                    autopct="%1.1f%%", startangle=140,
                    wedgeprops={"edgecolor": "#16213e", "linewidth": 1.5}
                )
                for t in texts:
                    t.set_color("#ddd")
                for at in autotexts:
                    at.set_color("#fff")
                    at.set_fontsize(8)

        elif req.chart_type == "histogram":
            for i, ds in enumerate(req.datasets):
                ax.hist(
                    ds.values, bins=20, label=ds.label,
                    color=PALETTE[i % len(PALETTE)], alpha=0.7, edgecolor="#222"
                )

        elif req.chart_type == "scatter":
            if len(req.datasets) >= 2:
                ax.scatter(
                    req.datasets[0].values, req.datasets[1].values,
                    color=PALETTE[0], alpha=0.7, edgecolors="#222", linewidths=0.5
                )

        _apply_style(ax, req.title or "", req.x_label or "", req.y_label or "")

        if n_datasets > 1 or req.chart_type == "line":
            legend = ax.legend(
                facecolor="#1a1a2e", edgecolor="#444",
                labelcolor="#ddd", fontsize=9
            )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {"image_base64": img_b64}

    except Exception as ex:
        logger.error(f"Smart chart error: {ex}")
        raise HTTPException(status_code=400, detail=str(ex))
