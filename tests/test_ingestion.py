"""Tests for content ingestion pipeline.

Tests JSONL parsing, document creation, and batch ingestion.
"""

import json

import pytest

from requirements_advisor.ingestion.pipeline import ingest_all_sources, ingest_jsonl


class TestIngestJsonl:
    """Test ingest_jsonl function."""

    @pytest.fixture
    def jsonl_file(self, temp_content_dir, sample_jsonl_content):
        """Create a sample JSONL file for testing."""
        file_path = temp_content_dir / "test_content.jsonl"
        file_path.write_text(sample_jsonl_content)
        return file_path

    @pytest.mark.asyncio
    async def test_ingest_jsonl_success(
        self, jsonl_file, mock_embedding_provider, mock_vector_store
    ):
        """Test successful JSONL ingestion."""
        count = await ingest_jsonl(
            jsonl_path=jsonl_file,
            source_name="test_source",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            batch_size=10,
        )

        assert count == 2
        assert mock_embedding_provider.embed_texts.called
        assert mock_vector_store.add_documents.called

    @pytest.mark.asyncio
    async def test_ingest_jsonl_file_not_found(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test ingestion with non-existent file."""
        nonexistent = temp_content_dir / "nonexistent.jsonl"

        count = await ingest_jsonl(
            jsonl_path=nonexistent,
            source_name="test",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_ingest_jsonl_empty_file(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test ingestion with empty file."""
        empty_file = temp_content_dir / "empty.jsonl"
        empty_file.write_text("")

        count = await ingest_jsonl(
            jsonl_path=empty_file,
            source_name="test",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        assert count == 0

    @pytest.mark.asyncio
    async def test_ingest_jsonl_skips_invalid_json(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test that invalid JSON lines are skipped."""
        content = '{"article_id": "1", "title": "Valid", "markdown_content": "Content"}\n'
        content += "invalid json line\n"
        content += '{"article_id": "2", "title": "Also Valid", "markdown_content": "More"}\n'

        mixed_file = temp_content_dir / "mixed.jsonl"
        mixed_file.write_text(content)

        count = await ingest_jsonl(
            jsonl_path=mixed_file,
            source_name="test",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        # Should process 2 valid documents, skipping the invalid one
        assert count == 2

    @pytest.mark.asyncio
    async def test_ingest_jsonl_skips_empty_content(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test that documents with empty content are skipped."""
        records = [
            {"article_id": "1", "title": "Valid", "markdown_content": "Content"},
            {"article_id": "2", "title": "Empty", "markdown_content": ""},
            {"article_id": "3", "title": "Whitespace", "markdown_content": "   "},
        ]
        content = "\n".join(json.dumps(r) for r in records)

        file_path = temp_content_dir / "with_empty.jsonl"
        file_path.write_text(content)

        count = await ingest_jsonl(
            jsonl_path=file_path,
            source_name="test",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        # Only the first document has content
        assert count == 1

    @pytest.mark.asyncio
    async def test_ingest_jsonl_batching(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test that documents are processed in batches."""
        # Create 15 documents
        records = [
            {"article_id": str(i), "title": f"Doc {i}", "markdown_content": f"Content {i}"}
            for i in range(15)
        ]
        content = "\n".join(json.dumps(r) for r in records)

        file_path = temp_content_dir / "many.jsonl"
        file_path.write_text(content)

        count = await ingest_jsonl(
            jsonl_path=file_path,
            source_name="test",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
            batch_size=5,
        )

        assert count == 15
        # With batch_size=5, should have 3 batches
        assert mock_embedding_provider.embed_texts.call_count == 3

    @pytest.mark.asyncio
    async def test_ingest_jsonl_extracts_metadata(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test that metadata is correctly extracted."""
        record = {
            "article_id": "1",
            "title": "Test Title",
            "markdown_content": "Test content",
            "chapter_title": "Chapter 1",
            "chapter_number": 1,
            "url": "https://example.com",
            "type": "guide",
            "key_concepts": ["concept1", "concept2"],
        }

        file_path = temp_content_dir / "metadata.jsonl"
        file_path.write_text(json.dumps(record))

        await ingest_jsonl(
            jsonl_path=file_path,
            source_name="test_source",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        # Check the documents passed to add_documents
        call_args = mock_vector_store.add_documents.call_args
        documents = call_args.args[0]

        assert len(documents) == 1
        doc = documents[0]
        assert doc.metadata["source"] == "test_source"
        assert doc.metadata["title"] == "Test Title"
        assert doc.metadata["chapter_title"] == "Chapter 1"
        assert doc.metadata["url"] == "https://example.com"


class TestIngestAllSources:
    """Test ingest_all_sources function."""

    @pytest.fixture
    def content_dir_with_files(self, temp_content_dir, sample_jsonl_content):
        """Create content directory with expected files."""
        # Create jama_guide.jsonl
        jama_file = temp_content_dir / "jama_guide.jsonl"
        jama_file.write_text(sample_jsonl_content)

        return temp_content_dir

    @pytest.mark.asyncio
    async def test_ingest_all_sources_success(
        self, content_dir_with_files, mock_embedding_provider, mock_vector_store
    ):
        """Test successful ingestion of all sources."""
        results = await ingest_all_sources(
            content_dir=content_dir_with_files,
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        assert "jama_guide" in results
        assert results["jama_guide"] == 2

    @pytest.mark.asyncio
    async def test_ingest_all_sources_empty_dir(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test ingestion with empty content directory."""
        results = await ingest_all_sources(
            content_dir=temp_content_dir,
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        assert results == {}

    @pytest.mark.asyncio
    async def test_ingest_all_sources_multiple_files(
        self, temp_content_dir, sample_jsonl_content, mock_embedding_provider, mock_vector_store
    ):
        """Test ingestion with multiple source files."""
        # Create multiple source files
        (temp_content_dir / "jama_guide.jsonl").write_text(sample_jsonl_content)
        (temp_content_dir / "ears_notation.jsonl").write_text(sample_jsonl_content)

        results = await ingest_all_sources(
            content_dir=temp_content_dir,
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        assert "jama_guide" in results
        assert "ears" in results
        assert results["jama_guide"] == 2
        assert results["ears"] == 2


class TestDocumentIdGeneration:
    """Test document ID generation."""

    @pytest.mark.asyncio
    async def test_document_id_uses_article_id(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test that document ID uses article_id field."""
        record = {"article_id": "my-article", "title": "Test", "markdown_content": "Content"}
        file_path = temp_content_dir / "test.jsonl"
        file_path.write_text(json.dumps(record))

        await ingest_jsonl(
            jsonl_path=file_path,
            source_name="source",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        call_args = mock_vector_store.add_documents.call_args
        documents = call_args.args[0]
        assert documents[0].id == "source:my-article"

    @pytest.mark.asyncio
    async def test_document_id_uses_term_for_glossary(
        self, temp_content_dir, mock_embedding_provider, mock_vector_store
    ):
        """Test that document ID uses term field for glossary entries."""
        record = {"term": "traceability", "definition": "The ability to trace."}
        file_path = temp_content_dir / "test.jsonl"
        file_path.write_text(json.dumps(record))

        await ingest_jsonl(
            jsonl_path=file_path,
            source_name="glossary",
            embedding_provider=mock_embedding_provider,
            vector_store=mock_vector_store,
        )

        call_args = mock_vector_store.add_documents.call_args
        documents = call_args.args[0]
        assert documents[0].id == "glossary:traceability"
