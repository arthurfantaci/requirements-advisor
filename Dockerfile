# Multi-stage Dockerfile for Requirements Advisor MCP Server
# Uses UV for fast, reliable Python dependency management

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install production dependencies
RUN uv pip install --no-cache .

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY src/ ./src/

# Create directories for data and content
RUN mkdir -p /app/data/chroma /app/content \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables (can be overridden)
ENV VOYAGE_API_KEY=""
ENV VOYAGE_MODEL="voyage-3"
ENV VECTOR_STORE_TYPE="chroma"
ENV VECTOR_STORE_PATH="/app/data/chroma"
ENV CONTENT_DIR="/app/content"
ENV HOST="0.0.0.0"
ENV PORT="8000"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

# Default command: start the MCP server
CMD ["python", "-m", "requirements_advisor.cli", "serve"]
