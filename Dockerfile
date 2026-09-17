FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files first (for layer caching)
COPY requirements.txt ./

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Expose port (AWS App Runner defaults to 8080)
EXPOSE 8080

# Run with uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
