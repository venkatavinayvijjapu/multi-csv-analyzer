# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build

# Copy only dependency files first (layer cache optimization)
COPY requirements.txt ./

# Install all Python packages into /build/packages
RUN uv pip install --system --target /build/packages -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="DataLens AI"
LABEL description="AI-powered multi-file CSV/Excel analytics — LangGraph + FastAPI"
LABEL version="4.0.0"

# Only runtime system deps (matplotlib needs these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy installed packages from builder
COPY --from=builder /build/packages /usr/local/lib/python3.12/site-packages

WORKDIR /app

# Copy application code (owned by non-root user)
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose port (AWS App Runner default 8080)
EXPOSE 8080

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

# ── Startup ───────────────────────────────────────────────────────────────────
# Single worker required: in-memory session_store is not shared across workers.
# For multi-worker deployments, replace session_store with a Redis backend first.
CMD ["python", "-m", "uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "warning", \
     "--access-log"]
