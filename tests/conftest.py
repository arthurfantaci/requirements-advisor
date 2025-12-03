"""Pytest fixtures and configuration for requirements-advisor tests.

This module provides shared fixtures for testing the MCP server, embedding providers,
vector stores, and ingestion pipeline.
"""

import asyncio
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from requirements_advisor.embeddings.base import EmbeddingProvider
from requirements_advisor.vectorstore.base import Document, SearchResult, VectorStore


# Configure pytest-asyncio
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# --- Temporary Directory Fixtures ---


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_content_dir(temp_dir: Path) -> Path:
    """Create a temporary content directory with sample JSONL files."""
    content_dir = temp_dir / "content"
    content_dir.mkdir()
    return content_dir


@pytest.fixture
def temp_vector_store_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for vector store persistence."""
    store_dir = temp_dir / "chroma"
    store_dir.mkdir()
    return store_dir


@pytest.fixture
def temp_image_cache_dir(temp_dir: Path) -> Path:
    """Create a temporary directory for image cache."""
    image_dir = temp_dir / "images"
    image_dir.mkdir()
    return image_dir


# --- Sample Data Fixtures ---


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Requirements traceability is essential for compliance.",
            metadata={"source": "jama_guide", "title": "Traceability Basics"},
        ),
        Document(
            id="doc2",
            content="EARS notation provides structured requirements syntax.",
            metadata={"source": "ears", "title": "EARS Overview"},
        ),
        Document(
            id="doc3",
            content="System requirements should be testable and measurable.",
            metadata={"source": "incose", "title": "Good Requirements"},
        ),
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Create sample embeddings matching sample_documents."""
    # 1024-dimensional embeddings (Voyage-3 dimension)
    return [
        [0.1] * 1024,
        [0.2] * 1024,
        [0.3] * 1024,
    ]


@pytest.fixture
def sample_query_embedding() -> list[float]:
    """Create a sample query embedding."""
    return [0.15] * 1024


@pytest.fixture
def sample_jsonl_content() -> str:
    """Create sample JSONL content for ingestion testing."""
    import json

    records = [
        {
            "article_id": "1",
            "title": "Introduction to Requirements",
            "markdown_content": "Requirements management is critical for project success.",
            "chapter_title": "Basics",
            "url": "https://example.com/article/1",
        },
        {
            "article_id": "2",
            "title": "Traceability Matrices",
            "markdown_content": "A traceability matrix links requirements to tests.",
            "chapter_title": "Traceability",
            "url": "https://example.com/article/2",
        },
    ]
    return "\n".join(json.dumps(r) for r in records)


# --- Mock Provider Fixtures ---


@pytest.fixture
def mock_embedding_provider() -> EmbeddingProvider:
    """Create a mock embedding provider."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.dimension = 1024
    provider.model_name = "mock-model"

    async def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        """Return mock embeddings for a list of texts."""
        return [[0.1] * 1024 for _ in texts]

    async def mock_embed_query(query: str) -> list[float]:
        """Return a mock embedding for a query."""
        return [0.1] * 1024

    provider.embed_texts = AsyncMock(side_effect=mock_embed_texts)
    provider.embed_query = AsyncMock(side_effect=mock_embed_query)

    return provider


@pytest.fixture
def mock_vector_store() -> VectorStore:
    """Create a mock vector store."""
    store = MagicMock(spec=VectorStore)

    async def mock_add_documents(
        documents: list[Document], embeddings: list[list[float]]
    ) -> None:
        """Mock adding documents to the store (no-op)."""
        pass

    async def mock_search(
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Return a mock search result."""
        return [
            SearchResult(
                document=Document(
                    id="test-doc",
                    content="Test content about requirements.",
                    metadata={"source": "test", "title": "Test Doc"},
                ),
                score=0.95,
            )
        ]

    async def mock_count() -> int:
        """Return a mock document count."""
        return 10

    async def mock_get_metadata_values(field: str) -> list[str]:
        """Return mock metadata values."""
        return ["jama_guide", "incose", "ears"]

    async def mock_delete_collection() -> None:
        """Mock deleting a collection (no-op)."""
        pass

    store.add_documents = AsyncMock(side_effect=mock_add_documents)
    store.search = AsyncMock(side_effect=mock_search)
    store.count = AsyncMock(side_effect=mock_count)
    store.get_metadata_values = AsyncMock(side_effect=mock_get_metadata_values)
    store.delete_collection = AsyncMock(side_effect=mock_delete_collection)

    return store


# --- Settings Override Fixtures ---


@pytest.fixture
def mock_settings(temp_dir: Path, monkeypatch):
    """Override settings with test values."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-api-key")
    monkeypatch.setenv("VECTOR_STORE_PATH", str(temp_dir / "chroma"))
    monkeypatch.setenv("CONTENT_DIR", str(temp_dir / "content"))
    monkeypatch.setenv("IMAGE_CACHE_PATH", str(temp_dir / "images"))
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    # Reload settings to pick up new env vars
    from requirements_advisor.config import Settings

    return Settings()


# --- HTTP Mock Fixtures ---


@pytest.fixture
def mock_voyage_response():
    """Create a mock Voyage API response."""
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1] * 1024, "index": 0}],
        "model": "voyage-3",
        "usage": {"total_tokens": 10},
    }


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create sample image bytes for testing."""
    from io import BytesIO

    from PIL import Image

    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()
