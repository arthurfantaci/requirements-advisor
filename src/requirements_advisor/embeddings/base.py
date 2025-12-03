"""Abstract base class for embedding providers.

Enables swapping between Voyage AI, OpenAI, Cohere, or local models.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts for document storage.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        """
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query for retrieval.

        Some providers use different models/settings for queries vs documents.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector

        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass
