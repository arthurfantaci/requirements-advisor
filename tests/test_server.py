"""
Tests for MCP server and tools.

Tests the FastMCP server tools: search_requirements_guidance, get_definition,
list_available_topics, and get_best_practices.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from requirements_advisor.vectorstore.base import Document, SearchResult


class TestSearchRequirementsGuidance:
    """Test search_requirements_guidance tool."""

    @pytest.fixture
    def mock_providers(self, mock_embedding_provider, mock_vector_store):
        """Set up mock providers for server tests."""
        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store",
                return_value=mock_vector_store,
            ),
            patch("requirements_advisor.server.get_image_cache", return_value=None),
        ):
            yield

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_providers):
        """Test search returns formatted results."""
        from requirements_advisor.server import search_requirements_guidance

        # Access underlying function via .fn attribute
        result = await search_requirements_guidance.fn(
            query="requirements traceability", top_k=3
        )

        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], TextContent)

    @pytest.mark.asyncio
    async def test_search_clamps_top_k(self, mock_providers):
        """Test search clamps top_k to 1-10 range."""
        from requirements_advisor.server import search_requirements_guidance

        # top_k > 10 should be clamped to 10
        result = await search_requirements_guidance.fn(query="test", top_k=100)
        assert isinstance(result, list)

        # top_k < 1 should be clamped to 1
        result = await search_requirements_guidance.fn(query="test", top_k=0)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_search_with_source_filter(self, mock_providers, mock_vector_store):
        """Test search with source filter passes filter to vector store."""
        from requirements_advisor.server import search_requirements_guidance

        await search_requirements_guidance.fn(
            query="test", top_k=3, source="jama_guide"
        )

        # Verify filter was passed to search
        mock_vector_store.search.assert_called()
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs.get("filter_metadata") == {"source": "jama_guide"}

    @pytest.mark.asyncio
    async def test_search_no_results(self, mock_embedding_provider):
        """Test search with no results returns appropriate message."""
        # Create a mock vector store that returns no results
        empty_store = MagicMock()
        empty_store.search = AsyncMock(return_value=[])

        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store", return_value=empty_store
            ),
            patch("requirements_advisor.server.get_image_cache", return_value=None),
        ):
            from requirements_advisor.server import search_requirements_guidance

            result = await search_requirements_guidance.fn(query="nonexistent topic")

            assert isinstance(result, list)
            assert len(result) == 1
            assert "No relevant guidance found" in result[0].text


class TestGetDefinition:
    """Test get_definition tool."""

    @pytest.fixture
    def mock_providers(self, mock_embedding_provider, mock_vector_store):
        """Set up mock providers for definition tests."""
        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store",
                return_value=mock_vector_store,
            ),
        ):
            yield

    @pytest.mark.asyncio
    async def test_get_definition_returns_string(self, mock_providers):
        """Test get_definition returns formatted string."""
        from requirements_advisor.server import get_definition

        result = await get_definition.fn(term="traceability")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_definition_no_results(self, mock_embedding_provider):
        """Test get_definition with no results."""
        empty_store = MagicMock()
        empty_store.search = AsyncMock(return_value=[])

        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store", return_value=empty_store
            ),
        ):
            from requirements_advisor.server import get_definition

            result = await get_definition.fn(term="unknownterm")

            assert "No definition found" in result

    @pytest.mark.asyncio
    async def test_get_definition_prefers_glossary(self, mock_embedding_provider):
        """Test get_definition prefers glossary entries."""
        glossary_doc = Document(
            id="glossary-1",
            content="Traceability is the ability to track requirements.",
            metadata={"source": "jama_guide", "type": "glossary_term"},
        )
        regular_doc = Document(
            id="doc-1",
            content="Traceability helps with compliance.",
            metadata={"source": "jama_guide", "type": "article"},
        )

        mock_store = MagicMock()
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(document=regular_doc, score=0.9),
                SearchResult(document=glossary_doc, score=0.85),
            ]
        )

        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store", return_value=mock_store
            ),
        ):
            from requirements_advisor.server import get_definition

            result = await get_definition.fn(term="traceability")

            # Should use glossary content
            assert "ability to track requirements" in result


class TestListAvailableTopics:
    """Test list_available_topics tool."""

    @pytest.fixture
    def mock_providers(self, mock_vector_store):
        """Set up mock vector store for topics tests."""
        with patch(
            "requirements_advisor.server.get_vector_store", return_value=mock_vector_store
        ):
            yield

    @pytest.mark.asyncio
    async def test_list_topics_returns_string(self, mock_providers):
        """Test list_available_topics returns formatted string."""
        from requirements_advisor.server import list_available_topics

        result = await list_available_topics.fn()

        assert isinstance(result, str)
        assert "Knowledge Base Summary" in result

    @pytest.mark.asyncio
    async def test_list_topics_includes_sources(self, mock_providers):
        """Test list_available_topics includes source names."""
        from requirements_advisor.server import list_available_topics

        result = await list_available_topics.fn()

        # Mock returns jama_guide, incose, ears
        assert "jama_guide" in result or "Jama" in result

    @pytest.mark.asyncio
    async def test_list_topics_includes_count(self, mock_providers):
        """Test list_available_topics includes document count."""
        from requirements_advisor.server import list_available_topics

        result = await list_available_topics.fn()

        # Mock returns count of 10
        assert "10" in result


class TestGetBestPractices:
    """Test get_best_practices tool."""

    @pytest.fixture
    def mock_providers(self, mock_embedding_provider, mock_vector_store):
        """Set up mock providers for best practices tests."""
        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store",
                return_value=mock_vector_store,
            ),
            patch("requirements_advisor.server.get_image_cache", return_value=None),
        ):
            yield

    @pytest.mark.asyncio
    async def test_get_best_practices_returns_list(self, mock_providers):
        """Test get_best_practices returns list of content."""
        from requirements_advisor.server import get_best_practices

        result = await get_best_practices.fn(topic="writing requirements")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], TextContent)

    @pytest.mark.asyncio
    async def test_get_best_practices_includes_topic(self, mock_providers):
        """Test get_best_practices includes topic in response."""
        from requirements_advisor.server import get_best_practices

        result = await get_best_practices.fn(topic="traceability")

        # First item should be text content with topic
        assert "Traceability" in result[0].text

    @pytest.mark.asyncio
    async def test_get_best_practices_no_results(self, mock_embedding_provider):
        """Test get_best_practices with no results."""
        empty_store = MagicMock()
        empty_store.search = AsyncMock(return_value=[])

        with (
            patch(
                "requirements_advisor.server.get_embedding_provider",
                return_value=mock_embedding_provider,
            ),
            patch(
                "requirements_advisor.server.get_vector_store", return_value=empty_store
            ),
            patch("requirements_advisor.server.get_image_cache", return_value=None),
        ):
            from requirements_advisor.server import get_best_practices

            result = await get_best_practices.fn(topic="unknown topic")

            assert "No best practices found" in result[0].text


class TestServerInitialization:
    """Test server initialization and lazy loading."""

    def test_mcp_server_created(self):
        """Test MCP server is created."""
        from requirements_advisor.server import mcp

        assert mcp is not None
        assert mcp.name == "requirements-advisor"

    def test_create_app_returns_mcp(self):
        """Test create_app returns the MCP instance."""
        from requirements_advisor.server import create_app, mcp

        app = create_app()
        assert app is mcp

    def test_create_sse_app_returns_app(self):
        """Test create_sse_app returns SSE application."""
        from requirements_advisor.server import create_sse_app

        sse_app = create_sse_app()
        assert sse_app is not None
