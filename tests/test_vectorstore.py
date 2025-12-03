"""Tests for vector store implementations.

Tests ChromaVectorStore implementation and factory function.
"""

import pytest

from requirements_advisor.vectorstore import create_vector_store
from requirements_advisor.vectorstore.base import Document, SearchResult, VectorStore
from requirements_advisor.vectorstore.chroma import ChromaVectorStore


class TestChromaVectorStore:
    """Test ChromaVectorStore class."""

    @pytest.fixture
    def chroma_store(self, temp_vector_store_dir):
        """Create a ChromaVectorStore instance for testing."""
        return ChromaVectorStore(
            collection_name="test_collection", persist_dir=temp_vector_store_dir
        )

    def test_init_creates_collection(self, chroma_store):
        """Test that initialization creates a collection."""
        assert chroma_store.collection is not None
        assert chroma_store.collection.name == "test_collection"

    def test_init_creates_directory(self, temp_dir):
        """Test that initialization creates the persist directory."""
        store_dir = temp_dir / "new_store"
        ChromaVectorStore(collection_name="test", persist_dir=store_dir)

        assert store_dir.exists()

    @pytest.mark.asyncio
    async def test_add_documents(self, chroma_store, sample_documents, sample_embeddings):
        """Test adding documents to the store."""
        await chroma_store.add_documents(sample_documents, sample_embeddings)

        count = await chroma_store.count()
        assert count == len(sample_documents)

    @pytest.mark.asyncio
    async def test_add_empty_documents(self, chroma_store):
        """Test adding empty documents list does nothing."""
        await chroma_store.add_documents([], [])

        count = await chroma_store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_search_returns_results(
        self, chroma_store, sample_documents, sample_embeddings, sample_query_embedding
    ):
        """Test searching returns relevant results."""
        await chroma_store.add_documents(sample_documents, sample_embeddings)

        results = await chroma_store.search(sample_query_embedding, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(isinstance(r.document, Document) for r in results)
        # Allow small floating point errors (cosine similarity can slightly exceed 1)
        assert all(-0.01 <= r.score <= 1.01 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_filter(
        self, chroma_store, sample_documents, sample_embeddings, sample_query_embedding
    ):
        """Test searching with metadata filter."""
        await chroma_store.add_documents(sample_documents, sample_embeddings)

        results = await chroma_store.search(
            sample_query_embedding, top_k=5, filter_metadata={"source": "jama_guide"}
        )

        # Should only return jama_guide documents
        for r in results:
            assert r.document.metadata.get("source") == "jama_guide"

    @pytest.mark.asyncio
    async def test_search_empty_store(self, chroma_store, sample_query_embedding):
        """Test searching empty store returns empty list."""
        results = await chroma_store.search(sample_query_embedding, top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_count(self, chroma_store, sample_documents, sample_embeddings):
        """Test document count."""
        assert await chroma_store.count() == 0

        await chroma_store.add_documents(sample_documents, sample_embeddings)

        assert await chroma_store.count() == len(sample_documents)

    @pytest.mark.asyncio
    async def test_get_metadata_values(
        self, chroma_store, sample_documents, sample_embeddings
    ):
        """Test getting distinct metadata values."""
        await chroma_store.add_documents(sample_documents, sample_embeddings)

        sources = await chroma_store.get_metadata_values("source")

        assert "jama_guide" in sources
        assert "ears" in sources
        assert "incose" in sources
        assert len(sources) == 3

    @pytest.mark.asyncio
    async def test_get_metadata_values_empty_store(self, chroma_store):
        """Test getting metadata values from empty store."""
        values = await chroma_store.get_metadata_values("source")

        assert values == []

    @pytest.mark.asyncio
    async def test_delete_collection(
        self, chroma_store, sample_documents, sample_embeddings
    ):
        """Test deleting collection."""
        await chroma_store.add_documents(sample_documents, sample_embeddings)
        assert await chroma_store.count() > 0

        await chroma_store.delete_collection()

        # Collection is deleted, so accessing it would raise an error
        # We just verify the method completes without error


class TestVectorStoreFactory:
    """Test create_vector_store factory function."""

    def test_create_chroma_store(self, temp_vector_store_dir):
        """Test creating ChromaDB store."""
        store = create_vector_store(
            store_type="chroma",
            collection_name="test",
            persist_dir=temp_vector_store_dir,
        )

        assert isinstance(store, ChromaVectorStore)

    def test_create_unknown_store_raises(self, temp_vector_store_dir):
        """Test that unknown store type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown vector store type"):
            create_vector_store(
                store_type="unknown",
                collection_name="test",
                persist_dir=temp_vector_store_dir,
            )


class TestVectorStoreInterface:
    """Test VectorStore interface compliance."""

    def test_chroma_implements_interface(self, temp_vector_store_dir):
        """Test ChromaVectorStore implements VectorStore interface."""
        store = ChromaVectorStore(
            collection_name="test", persist_dir=temp_vector_store_dir
        )

        assert isinstance(store, VectorStore)
        assert hasattr(store, "add_documents")
        assert hasattr(store, "search")
        assert hasattr(store, "count")
        assert hasattr(store, "delete_collection")
        assert hasattr(store, "get_metadata_values")


class TestDocument:
    """Test Document model."""

    def test_create_document(self):
        """Test creating a document."""
        doc = Document(
            id="test-1", content="Test content", metadata={"source": "test"}
        )

        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.metadata == {"source": "test"}

    def test_document_default_metadata(self):
        """Test document with default empty metadata."""
        doc = Document(id="test-1", content="Test content")

        assert doc.metadata == {}


class TestSearchResult:
    """Test SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        doc = Document(id="test-1", content="Test content")
        result = SearchResult(document=doc, score=0.95)

        assert result.document == doc
        assert result.score == 0.95

    def test_search_result_score_bounds(self):
        """Test search result score can be any float."""
        doc = Document(id="test-1", content="Test content")

        # Scores are typically 0-1 but model doesn't enforce
        result_low = SearchResult(document=doc, score=0.0)
        result_high = SearchResult(document=doc, score=1.0)

        assert result_low.score == 0.0
        assert result_high.score == 1.0
