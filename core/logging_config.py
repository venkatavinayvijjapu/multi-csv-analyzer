"""
core/logging_config.py
-----------------------
Centralised logging configuration for DataLens AI.

Replaces repeated `logging.basicConfig(...)` calls scattered across modules.

Usage:
    from core.logging_config import get_logger, setup_logging
    setup_logging()            # call once at app startup (api.py)
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

# ── Context variable for per-request correlation ID ───────────────────────────
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return request_id_var.get() or ""


def set_request_id(rid: Optional[str] = None) -> str:
    rid = rid or str(uuid.uuid4())[:8]
    request_id_var.set(rid)
    return rid


# ── Custom Formatter ──────────────────────────────────────────────────────────

class ContextFormatter(logging.Formatter):
    """Injects the current request_id into every log record."""

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = get_request_id() or "-"
        return super().format(record)


# ── Log format strings ────────────────────────────────────────────────────────

_PLAIN_FMT = (
    "%(asctime)s [%(request_id)s] %(levelname)-8s %(name)s — %(message)s"
)
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# ── Public setup ──────────────────────────────────────────────────────────────

def setup_logging(
    log_level: str = "INFO",
    log_dir: str = ".",
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Initialise root logger.  Call this ONCE at application startup.

    • Writes INFO+ to  app.log     (rotating, 10 MB × 5)
    • Writes ERROR+ to errors.log  (rotating, 10 MB × 5)
    • Writes DEBUG+  to stdout     (only when DEBUG env var is set)

    All handlers use ContextFormatter to include the request_id.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = ContextFormatter(fmt=_PLAIN_FMT, datefmt=_DATE_FMT)

    handlers: list[logging.Handler] = []

    # ── App log (INFO+, rotating) ─────────────────────────────────────────────
    app_log_path = os.path.join(log_dir, "app.log")
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(formatter)
    handlers.append(app_handler)

    # ── Error log (ERROR+, rotating) ─────────────────────────────────────────
    err_log_path = os.path.join(log_dir, "errors.log")
    err_handler = logging.handlers.RotatingFileHandler(
        err_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)
    handlers.append(err_handler)

    # ── Stdout (DEBUG, only when LOG_LEVEL=DEBUG or DEBUG env var) ────────────
    if os.getenv("DEBUG") or log_level.upper() == "DEBUG":
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)
        handlers.append(stdout_handler)
    else:
        # Always show WARNING+ to stdout so Docker/k8s log scrapers see issues
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.WARNING)
        stdout_handler.setFormatter(formatter)
        handlers.append(stdout_handler)

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers added by earlier basicConfig calls
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (call after setup_logging())."""
    return logging.getLogger(name)
