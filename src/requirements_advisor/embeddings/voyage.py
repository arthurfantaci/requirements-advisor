"""Voyage AI embedding provider implementation.

Voyage AI offers high-quality embeddings optimized for retrieval,
particularly good for technical and domain-specific content.
"""

from typing import cast

import voyageai  # type: ignore[import-untyped]
from loguru import logger  # type: ignore[import-untyped]
from rich.console import Console  # type: ignore[import-untyped]

from .base import EmbeddingProvider

console = Console()

# Model dimensions (as of 2025)
MODEL_DIMENSIONS = {
    # Current recommended models
    "voyage-context-3": 1024,
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-multimodal-3": 1024,
    # Legacy models
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}

# Models that use the contextualized embed API
CONTEXTUALIZED_MODELS = {"voyage-context-3"}


class VoyageEmbedding(EmbeddingProvider):
    """Voyage AI embedding provider.

    Supports both standard embedding models (voyage-3-large, voyage-3.5, etc.)
    and contextualized embedding models (voyage-context-3) which preserve
    document context in chunk embeddings.
    """

    def __init__(self, api_key: str, model: str = "voyage-context-3"):
        """Initialize Voyage AI client.

        Args:
            api_key: Voyage AI API key
            model: Model name. Recommended: voyage-context-3 (contextualized),
                   voyage-3-large (standard), voyage-3.5 (standard)

        Raises:
            ValueError: If api_key is empty or None

        """
        if not api_key:
            raise ValueError("Voyage API key is required")

        self.client = voyageai.AsyncClient(api_key=api_key)
        self.model = model
        self._dimension = MODEL_DIMENSIONS.get(model, 1024)
        self._is_contextualized = model in CONTEXTUALIZED_MODELS

        model_type = "contextualized" if self._is_contextualized else "standard"
        logger.info(
            "Initialized Voyage AI: model={}, dim={}, type={}",
            model,
            self._dimension,
            model_type,
        )
        console.print(f"[green]Initialized Voyage AI: {model} ({model_type})[/]")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for storage.

        For contextualized models (voyage-context-3), each text is embedded
        with awareness of surrounding context. For standard models, texts
        are embedded independently.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        """
        if not texts:
            return []

        logger.debug("Embedding {} texts for document storage", len(texts))

        if self._is_contextualized:
            # Contextualized API expects list of lists
            # Each inner list contains chunks from the same document
            # For batch processing, we treat each text as independent
            inputs = [[text] for text in texts]
            result = await self.client.contextualized_embed(
                inputs=inputs,
                model=self.model,
                input_type="document",
            )
            # ContextualizedEmbeddingsObject stores results differently:
            # result.results[i].embeddings contains embeddings for document i
            embeddings = cast(
                list[list[float]], [r.embeddings[0] for r in result.results]
            )
        else:
            # Standard embed API
            result = await self.client.embed(
                texts,
                model=self.model,
                input_type="document",
            )
            embeddings = cast(list[list[float]], result.embeddings)

        logger.debug("Generated {} embeddings", len(embeddings))
        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed query for retrieval.

        Args:
            query: Query string to embed

        Returns:
            Embedding vector for the query

        """
        logger.debug("Embedding query: '{}'", query[:50])

        if self._is_contextualized:
            # Contextualized API: query as single-element list
            result = await self.client.contextualized_embed(
                inputs=[[query]],
                model=self.model,
                input_type="query",
            )
            # ContextualizedEmbeddingsObject: result.results[0].embeddings[0]
            return cast(list[float], result.results[0].embeddings[0])
        else:
            # Standard embed API
            result = await self.client.embed(
                [query],
                model=self.model,
                input_type="query",
            )
            return cast(list[float], result.embeddings[0])

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for this model.

        Returns:
            Number of dimensions in the embedding vector (e.g., 1024 for voyage-3).

        """
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the model name.

        Returns:
            Model identifier string (e.g., "voyage-context-3").

        """
        return self.model

    @property
    def is_contextualized(self) -> bool:
        """Return whether this model uses contextualized embeddings.

        Returns:
            True if model uses contextualized embed API, False for standard API.

        """
        return self._is_contextualized
