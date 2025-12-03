"""Tests for embedding providers.

Tests VoyageEmbedding implementation and factory function.
"""

import pytest

from requirements_advisor.embeddings import create_embedding_provider
from requirements_advisor.embeddings.base import EmbeddingProvider
from requirements_advisor.embeddings.voyage import (
    CONTEXTUALIZED_MODELS,
    MODEL_DIMENSIONS,
    VoyageEmbedding,
)


class TestVoyageEmbedding:
    """Test VoyageEmbedding class."""

    def test_init_with_valid_api_key(self):
        """Test initialization with valid API key."""
        provider = VoyageEmbedding(api_key="test-key", model="voyage-3")

        assert provider.model == "voyage-3"
        assert provider.dimension == 1024
        assert provider.model_name == "voyage-3"

    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        with pytest.raises(ValueError, match="Voyage API key is required"):
            VoyageEmbedding(api_key="", model="voyage-3")

    def test_dimension_for_different_models(self):
        """Test correct dimensions for different models."""
        for model, expected_dim in MODEL_DIMENSIONS.items():
            provider = VoyageEmbedding(api_key="test-key", model=model)
            assert provider.dimension == expected_dim

    def test_unknown_model_defaults_to_1024(self):
        """Test that unknown model defaults to 1024 dimensions."""
        provider = VoyageEmbedding(api_key="test-key", model="unknown-model")
        assert provider.dimension == 1024

    def test_default_model_is_voyage_context_3(self):
        """Test default model is voyage-context-3."""
        provider = VoyageEmbedding(api_key="test-key")
        assert provider.model == "voyage-context-3"

    def test_contextualized_model_detection(self):
        """Test that contextualized models are correctly detected."""
        # voyage-context-3 should be contextualized
        provider = VoyageEmbedding(api_key="test-key", model="voyage-context-3")
        assert provider.is_contextualized is True

        # Standard models should not be contextualized
        provider = VoyageEmbedding(api_key="test-key", model="voyage-3-large")
        assert provider.is_contextualized is False

        provider = VoyageEmbedding(api_key="test-key", model="voyage-3")
        assert provider.is_contextualized is False

    @pytest.mark.asyncio
    async def test_embed_texts_empty_list(self):
        """Test embedding empty list returns empty list."""
        provider = VoyageEmbedding(api_key="test-key", model="voyage-3")
        result = await provider.embed_texts([])

        assert result == []


class TestEmbeddingFactory:
    """Test create_embedding_provider factory function."""

    def test_create_voyage_provider(self):
        """Test creating Voyage provider."""
        provider = create_embedding_provider(
            provider_type="voyage", api_key="test-key", model="voyage-3"
        )

        assert isinstance(provider, VoyageEmbedding)
        assert provider.model == "voyage-3"

    def test_create_unknown_provider_raises(self):
        """Test that unknown provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(
                provider_type="unknown", api_key="test-key", model="model"
            )

    def test_default_model_is_voyage_context_3(self):
        """Test default model is voyage-context-3."""
        provider = create_embedding_provider(provider_type="voyage", api_key="test-key")

        assert provider.model_name == "voyage-context-3"


class TestEmbeddingProviderInterface:
    """Test EmbeddingProvider interface compliance."""

    def test_voyage_implements_interface(self):
        """Test VoyageEmbedding implements EmbeddingProvider interface."""
        provider = VoyageEmbedding(api_key="test-key", model="voyage-3")

        assert isinstance(provider, EmbeddingProvider)
        assert hasattr(provider, "embed_texts")
        assert hasattr(provider, "embed_query")
        assert hasattr(provider, "dimension")
        assert hasattr(provider, "model_name")

    def test_voyage_has_contextualized_property(self):
        """Test VoyageEmbedding has is_contextualized property."""
        provider = VoyageEmbedding(api_key="test-key", model="voyage-context-3")
        assert hasattr(provider, "is_contextualized")
        assert provider.is_contextualized is True


class TestModelDimensions:
    """Test MODEL_DIMENSIONS configuration."""

    def test_all_current_models_defined(self):
        """Test that all current recommended models are defined."""
        expected_models = [
            "voyage-context-3",
            "voyage-3-large",
            "voyage-3.5",
            "voyage-3.5-lite",
            "voyage-code-3",
            "voyage-finance-2",
            "voyage-law-2",
        ]
        for model in expected_models:
            assert model in MODEL_DIMENSIONS, f"Missing model: {model}"

    def test_contextualized_models_set(self):
        """Test that CONTEXTUALIZED_MODELS is correctly defined."""
        assert "voyage-context-3" in CONTEXTUALIZED_MODELS
        assert "voyage-3-large" not in CONTEXTUALIZED_MODELS
