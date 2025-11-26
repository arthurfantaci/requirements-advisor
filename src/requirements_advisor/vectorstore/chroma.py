"""
ChromaDB vector store implementation.

ChromaDB provides a simple, local vector database that persists to disk.
Ideal for development and single-instance deployments.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from rich.console import Console

from .base import Document, SearchResult, VectorStore

console = Console()


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore."""
    
    def __init__(
        self,
        collection_name: str = "requirements_guidance",
        persist_dir: str | Path = "./data/chroma",
    ):
        """
        Initialize ChromaDB client with persistent storage.
        
        Args:
            collection_name: Name of the collection
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        
        console.print(f"[green]ChromaDB initialized at {self.persist_dir}[/]")
        console.print(f"[green]Collection '{collection_name}' has {self.collection.count()} documents[/]")
    
    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with embeddings to ChromaDB."""
        if not documents:
            return
        
        self.collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings,
        )
    
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar documents."""
        where_filter = None
        if filter_metadata:
            # ChromaDB uses specific filter syntax
            if len(filter_metadata) == 1:
                key, value = next(iter(filter_metadata.items()))
                where_filter = {key: {"$eq": value}}
            else:
                where_filter = {
                    "$and": [{k: {"$eq": v}} for k, v in filter_metadata.items()]
                }
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for id_, doc, meta, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB returns distance; convert to similarity score
                # For cosine distance: similarity = 1 - distance
                score = 1 - distance
                
                search_results.append(
                    SearchResult(
                        document=Document(id=id_, content=doc, metadata=meta),
                        score=score,
                    )
                )
        
        return search_results
    
    async def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection.name)
        console.print(f"[yellow]Deleted collection {self.collection.name}[/]")
    
    async def count(self) -> int:
        """Return document count."""
        return self.collection.count()
    
    async def get_metadata_values(self, field: str) -> list[str]:
        """Get distinct values for a metadata field."""
        # ChromaDB doesn't have a direct distinct query, so we fetch all and dedupe
        # This is fine for small collections; for large ones, consider caching
        results = self.collection.get(include=["metadatas"])
        
        values = set()
        for meta in results["metadatas"]:
            if field in meta and meta[field]:
                values.add(str(meta[field]))
        
        return sorted(list(values))
