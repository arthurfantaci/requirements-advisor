# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP (Model Context Protocol) server providing expert guidance on requirements management best practices. Uses RAG (Retrieval-Augmented Generation) with Voyage AI embeddings and ChromaDB vector storage to serve authoritative content from sources like Jama Software guides, INCOSE, and EARS notation.

## Development Commands

```bash
# Install dependencies (using UV)
uv sync

# Install with development dependencies
uv sync --dev

# Install with ingestion dependencies (for image processing)
uv sync --extra ingestion

# Start the MCP server
uv run requirements-advisor serve

# Ingest content into vector store
uv run requirements-advisor ingest

# Clear and re-ingest with image caching
uv run requirements-advisor ingest --clear --fetch-images

# Test a search query
uv run requirements-advisor test-search "how to write requirements"

# Show configuration status
uv run requirements-advisor info

# Run linting
uv run ruff check src/
uv run ruff format src/

# Run tests
uv run pytest
```

### Docker Commands

```bash
docker compose build
docker compose run --rm ingestion    # Ingest content
docker compose up -d                 # Start server
docker compose logs -f mcp-server    # View logs
```

## Architecture

### Core Abstraction Pattern

The codebase uses abstract base classes for swappable providers:

- **EmbeddingProvider** (`embeddings/base.py`): Interface for text embeddings
  - `embed_texts()` for document storage, `embed_query()` for retrieval
  - Voyage AI implementation in `embeddings/voyage.py`

- **VectorStore** (`vectorstore/base.py`): Interface for vector databases
  - Uses `Document` and `SearchResult` Pydantic models
  - ChromaDB implementation in `vectorstore/chroma.py`

Factory functions `create_embedding_provider()` and `create_vector_store()` instantiate the appropriate implementation based on config.

### Server Components

- **server.py**: FastMCP server with Streamable HTTP transport and lazy-initialized providers. Defines MCP tools:
  - `search_requirements_guidance`: Semantic search over guidance content
  - `get_definition`: Look up requirements management terms
  - `list_available_topics`: Show available sources and topics
  - `get_best_practices`: Get best practices for specific topics
- **cli.py**: Typer CLI with commands for serve, ingest, info, test-search
- **config.py**: Pydantic-settings configuration loaded from environment/.env
- **logging.py**: Centralized loguru logging configuration

### Image Caching

- **images/base.py**: Pydantic models (`CachedImage`, `ImageIndex`)
- **images/cache.py**: `ImageCache` class for fetching, processing, and caching images from content

### Data Flow

1. JSONL content files in `content/` are ingested via `ingestion/pipeline.py`
2. Documents are embedded using Voyage AI and stored in ChromaDB (`data/chroma/`)
3. Images are fetched, resized, and cached in `data/images/`
4. MCP server exposes tools that query the vector store with semantic search
5. Search results can include cached images when available

## Key Configuration

Environment variables (via `.env`):
- `VOYAGE_API_KEY` (required): Voyage AI API key
- `VOYAGE_MODEL`: Embedding model (default: `voyage-context-3`). Options include:
  - `voyage-context-3`: Contextualized embeddings (recommended for RAG)
  - `voyage-3-large`: Best general-purpose quality
  - `voyage-3.5`, `voyage-3.5-lite`: Latest standard models
- `VOYAGE_BATCH_SIZE`: Texts per API call (default: `20`)
- `VECTOR_STORE_TYPE`: `chroma` (default) or `qdrant`
- `VECTOR_STORE_PATH`: Local storage path (default: `./data/chroma`)
- `COLLECTION_NAME`: Vector collection name (default: `requirements_guidance`)
- `IMAGE_CACHE_PATH`: Image cache directory (default: `./data/images`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Content Format

JSONL files with fields:
- `article_id` or `term`: unique identifier
- `markdown_content` or `definition`: text content
- `title`: document title
- Optional: `chapter_title`, `chapter_number`, `url`, `type`, `key_concepts`, `images`

## Testing

The test suite uses pytest with pytest-asyncio for async tests:
- `tests/conftest.py`: Shared fixtures (mock providers, temp directories, sample data)
- Tests cover: config, embeddings, vectorstore, images, ingestion, CLI, server tools

## Code Style

- Google-style docstrings with Args, Returns, Raises, Example sections
- Ruff linting with rules: E, W, F, I, B, C4, UP, SIM, D
- Type hints throughout using Python 3.11+ syntax
