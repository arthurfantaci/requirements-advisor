"""
Abstract base class for vector stores.

Enables swapping between ChromaDB (local), Qdrant, Pinecone, etc.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A document to store in the vector database."""

    id: str = Field(description="Unique document identifier")
    content: str = Field(description="Text content of the document")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata for filtering and display (source, title, url, etc.)",
    )


class SearchResult(BaseModel):
    """A search result with relevance score."""

    document: Document
    score: float = Field(description="Similarity score (0-1, higher is better)")


class VectorStore(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """
        Add documents with their embeddings to the store.

        Args:
            documents: List of documents to add
            embeddings: Corresponding embedding vectors
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results ordered by relevance
        """
        pass

    @abstractmethod
    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Return the number of documents in the collection."""
        pass

    @abstractmethod
    async def get_metadata_values(self, field: str) -> list[str]:
        """Get distinct values for a metadata field."""
        pass
