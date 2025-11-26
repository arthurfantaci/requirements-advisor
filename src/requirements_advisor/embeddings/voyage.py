"""
Voyage AI embedding provider implementation.

Voyage AI offers high-quality embeddings optimized for retrieval,
particularly good for technical and domain-specific content.
"""

import voyageai
from rich.console import Console

from .base import EmbeddingProvider

console = Console()

# Model dimensions (as of 2024)
MODEL_DIMENSIONS = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
}


class VoyageEmbedding(EmbeddingProvider):
    """Voyage AI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "voyage-3"):
        """
        Initialize Voyage AI client.
        
        Args:
            api_key: Voyage AI API key
            model: Model name (voyage-3, voyage-3-lite, etc.)
        """
        if not api_key:
            raise ValueError("Voyage API key is required")
        
        self.client = voyageai.AsyncClient(api_key=api_key)
        self.model = model
        self._dimension = MODEL_DIMENSIONS.get(model, 1024)
        
        console.print(f"[green]Initialized Voyage AI with model: {model}[/]")
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for storage."""
        if not texts:
            return []
        
        result = await self.client.embed(
            texts,
            model=self.model,
            input_type="document",
        )
        return result.embeddings
    
    async def embed_query(self, query: str) -> list[float]:
        """Embed query for retrieval."""
        result = await self.client.embed(
            [query],
            model=self.model,
            input_type="query",
        )
        return result.embeddings[0]
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self.model
