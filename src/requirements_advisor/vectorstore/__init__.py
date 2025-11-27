"""
Vector store package.

Provides a factory function to create the configured vector store.
"""

from pathlib import Path

from .base import Document, SearchResult, VectorStore
from .chroma import ChromaVectorStore


def create_vector_store(
    store_type: str = "chroma",
    collection_name: str = "requirements_guidance",
    persist_dir: str | Path = "./data/chroma",
    **kwargs,
) -> VectorStore:
    """
    Factory function to create a vector store.

    Args:
        store_type: Type of store ("chroma", future: "qdrant", "pinecone")
        collection_name: Name of the collection
        persist_dir: Directory for local persistence (ChromaDB)
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured VectorStore instance

    Raises:
        ValueError: If store_type is not recognized
        NotImplementedError: If store_type is planned but not yet implemented
    """
    if store_type == "chroma":
        return ChromaVectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir,
        )
    elif store_type == "qdrant":
        raise NotImplementedError("Qdrant support coming soon")
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")


__all__ = [
    "Document",
    "SearchResult",
    "VectorStore",
    "ChromaVectorStore",
    "create_vector_store",
]
