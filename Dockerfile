# Multi-stage Dockerfile for Requirements Advisor MCP Server
# Uses UV for fast, reliable Python dependency management
# Includes pre-built vector database for Railway deployment

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code (needed for package build)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install the package (includes dependencies)
RUN uv pip install --no-cache .

# ============================================
# Stage 2: Ingestion - Build vector database
# ============================================
FROM builder AS ingestion

WORKDIR /app

# Copy content files for ingestion
COPY content/ ./content/

# Set environment for ingestion
ENV VECTOR_STORE_TYPE="chroma"
ENV VECTOR_STORE_PATH="/app/data/chroma"
ENV CONTENT_DIR="/app/content"

# Build argument for Voyage API key (required for ingestion)
ARG VOYAGE_API_KEY
ENV VOYAGE_API_KEY=${VOYAGE_API_KEY}

# Run ingestion to build the vector database
RUN python -m requirements_advisor.cli ingest --clear

# ============================================
# Stage 3: Runtime - Minimal production image
# ============================================
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source and content
COPY src/ ./src/
COPY content/ ./content/

# Copy pre-built vector database from ingestion stage
COPY --from=ingestion /app/data/chroma /app/data/chroma

# Set ownership
RUN chown -R appuser:appuser /app

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

# Health check - verify server process is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD pgrep -f "requirements_advisor" || exit 1

# Default command: start the MCP server
CMD ["python", "-m", "requirements_advisor.cli", "serve"]
