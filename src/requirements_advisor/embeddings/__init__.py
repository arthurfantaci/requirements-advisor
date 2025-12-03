"""Embedding providers package.

Provides a factory function to create the configured embedding provider.
"""

from .base import EmbeddingProvider
from .voyage import VoyageEmbedding


def create_embedding_provider(
    provider_type: str = "voyage",
    api_key: str = "",
    model: str | None = None,
) -> EmbeddingProvider:
    """Create an embedding provider instance.

    Args:
        provider_type: Type of provider ("voyage", future: "openai", "cohere")
        api_key: API key for the provider
        model: Optional model override

    Returns:
        Configured EmbeddingProvider instance

    Raises:
        ValueError: If provider_type is not recognized or api_key is empty

    Example:
        >>> provider = create_embedding_provider(
        ...     provider_type="voyage",
        ...     api_key="your-api-key",
        ...     model="voyage-context-3",
        ... )
        >>> embedding = await provider.embed_query("What is traceability?")

    """
    if provider_type == "voyage":
        return VoyageEmbedding(
            api_key=api_key,
            model=model or "voyage-context-3",
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")


__all__ = [
    "EmbeddingProvider",
    "VoyageEmbedding",
    "create_embedding_provider",
]
