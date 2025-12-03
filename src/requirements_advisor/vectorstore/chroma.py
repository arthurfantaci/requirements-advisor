"""ChromaDB vector store implementation.

ChromaDB provides a simple, local vector database that persists to disk.
Ideal for development and single-instance deployments.
"""

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger
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
        """Initialize ChromaDB client with persistent storage.

        Args:
            collection_name: Name of the collection
            persist_dir: Directory for persistent storage

        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Initializing ChromaDB at {}", self.persist_dir)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        doc_count = self.collection.count()
        logger.info(
            "ChromaDB initialized: collection='{}', documents={}", collection_name, doc_count
        )
        console.print(f"[green]ChromaDB initialized at {self.persist_dir}[/]")
        console.print(f"[green]Collection '{collection_name}' has {doc_count} documents[/]")

    async def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        """Add documents with embeddings to ChromaDB.

        Args:
            documents: List of Document objects to store
            embeddings: Corresponding embedding vectors (must match documents length)

        """
        if not documents:
            return

        logger.debug("Adding {} documents to collection", len(documents))
        self.collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=cast(Any, embeddings),
        )
        logger.debug("Documents added successfully")

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar documents.

        Performs a cosine similarity search and returns the most relevant documents.

        Args:
            query_embedding: Vector representation of the search query
            top_k: Maximum number of results to return
            filter_metadata: Optional metadata filters (e.g., {"source": "jama_guide"})

        Returns:
            List of SearchResult objects ordered by descending similarity score.

        """
        logger.debug("Searching collection: top_k={}, filter={}", top_k, filter_metadata)
        where_filter: Any = None
        if filter_metadata:
            # ChromaDB uses specific filter syntax
            if len(filter_metadata) == 1:
                key, value = next(iter(filter_metadata.items()))
                where_filter = {key: {"$eq": value}}
            else:
                where_filter = {"$and": [{k: {"$eq": v}} for k, v in filter_metadata.items()]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        # These are guaranteed non-None because we include them in the query
        documents = results["documents"]
        metadatas = results["metadatas"]
        distances = results["distances"]

        if results["ids"] and results["ids"][0] and documents and metadatas and distances:
            for id_, doc, meta, distance in zip(
                results["ids"][0],
                documents[0],
                metadatas[0],
                distances[0],
                strict=False,
            ):
                # ChromaDB returns distance; convert to similarity score
                # For cosine distance: similarity = 1 - distance
                score = 1 - distance

                search_results.append(
                    SearchResult(
                        document=Document(id=id_, content=doc, metadata=cast(dict, meta)),
                        score=score,
                    )
                )

        logger.debug("Search returned {} results", len(search_results))
        return search_results

    async def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB.

        Warning:
            This permanently removes all documents and cannot be undone.

        """
        collection_name = self.collection.name
        logger.info("Deleting collection: {}", collection_name)
        self.client.delete_collection(collection_name)
        console.print(f"[yellow]Deleted collection {collection_name}[/]")

    async def count(self) -> int:
        """Return the total number of documents in the collection.

        Returns:
            Number of documents stored in the collection.

        """
        count = self.collection.count()
        logger.debug("Collection count: {}", count)
        return count

    async def get_metadata_values(self, field: str) -> list[str]:
        """Get distinct values for a metadata field.

        Args:
            field: Metadata field name to query (e.g., "source", "chapter_title")

        Returns:
            Sorted list of unique values for the specified field.

        """
        logger.debug("Getting distinct values for field: {}", field)
        # ChromaDB doesn't have a direct distinct query, so we fetch all and dedupe
        # This is fine for small collections; for large ones, consider caching
        results = self.collection.get(include=["metadatas"])

        values: set[str] = set()
        metadatas = results["metadatas"]
        if metadatas:
            for meta in metadatas:
                if field in meta and meta[field]:
                    values.add(str(meta[field]))

        logger.debug("Found {} distinct values for field '{}'", len(values), field)
        return sorted(values)
