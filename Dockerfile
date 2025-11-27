# Multi-stage Dockerfile for Requirements Advisor MCP Server
# Uses UV for fast, reliable Python dependency management
# Includes pre-built vector database for Railway deployment

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install build dependencies (including libjpeg for Pillow image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and source code (needed for package build)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install the package with ingestion extras (includes Pillow for image processing)
RUN uv pip install --no-cache ".[ingestion]"

# ============================================
# Stage 2: Ingestion - Build vector database and cache images
# ============================================
FROM builder AS ingestion

WORKDIR /app

# Copy content files for ingestion
COPY content/ ./content/

# Set environment for ingestion
ENV VECTOR_STORE_TYPE="chroma"
ENV VECTOR_STORE_PATH="/app/data/chroma"
ENV IMAGE_CACHE_PATH="/app/data/images"
ENV CONTENT_DIR="/app/content"

# Build arguments for ingestion
ARG VOYAGE_API_KEY
ARG VOYAGE_MODEL="voyage-context-3"
ARG VOYAGE_BATCH_SIZE="20"
ENV VOYAGE_API_KEY=${VOYAGE_API_KEY}
ENV VOYAGE_MODEL=${VOYAGE_MODEL}
ENV VOYAGE_BATCH_SIZE=${VOYAGE_BATCH_SIZE}

# Run ingestion to build the vector database and cache images
RUN python -m requirements_advisor.cli ingest --clear --fetch-images

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

# Copy pre-built vector database and image cache from ingestion stage
COPY --from=ingestion /app/data/chroma /app/data/chroma
COPY --from=ingestion /app/data/images /app/data/images

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables (can be overridden)
ENV VOYAGE_API_KEY=""
ENV VOYAGE_MODEL="voyage-context-3"
ENV VECTOR_STORE_TYPE="chroma"
ENV VECTOR_STORE_PATH="/app/data/chroma"
ENV IMAGE_CACHE_PATH="/app/data/images"
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
