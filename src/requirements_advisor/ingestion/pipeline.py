"""
Content ingestion pipeline.

Loads JSONL content files, creates embeddings, and stores in vector database.
Optionally fetches and caches images from content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from ..embeddings.base import EmbeddingProvider
from ..vectorstore.base import Document, VectorStore

if TYPE_CHECKING:
    from ..images import ImageCache

console = Console()


async def ingest_jsonl(
    jsonl_path: Path,
    source_name: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    image_cache: ImageCache | None = None,
    batch_size: int = 50,
) -> int:
    """
    Ingest a JSONL file into the vector store.

    Each line in the JSONL should have:
    - article_id or term: unique identifier
    - markdown_content or definition: text content
    - title: document title
    - Optional: chapter_title, chapter_number, url, type, key_concepts, images

    Args:
        jsonl_path: Path to JSONL file
        source_name: Source identifier (e.g., "jama_guide", "incose", "ears")
        embedding_provider: Provider for creating embeddings
        vector_store: Store for saving documents
        image_cache: Optional image cache for fetching/storing images
        batch_size: Number of documents to embed at once

    Returns:
        Number of documents ingested
    """
    if not jsonl_path.exists():
        logger.warning("File not found: {}", jsonl_path)
        console.print(f"[red]File not found: {jsonl_path}[/]")
        return 0

    logger.info("Loading content from {} (source: {})", jsonl_path, source_name)
    console.print(f"[blue]Loading content from {jsonl_path}[/]")

    # Track images per document for later processing
    doc_images: dict[str, list[dict]] = {}

    # Load all documents
    documents = []
    with open(jsonl_path) as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)

                # Extract content (handle different field names)
                content = record.get("markdown_content") or record.get("definition") or ""
                if not content.strip():
                    continue

                # Build document ID
                doc_id = f"{source_name}:{record.get('article_id', record.get('term', f'doc_{line_num}'))}"

                # Build metadata
                metadata = {
                    "source": source_name,
                    "title": record.get("title") or record.get("term") or "Untitled",
                    "type": record.get("type", "article"),
                }

                # Add optional metadata
                if record.get("chapter_title"):
                    metadata["chapter_title"] = record["chapter_title"]
                if record.get("chapter_number"):
                    metadata["chapter_number"] = record["chapter_number"]
                if record.get("url"):
                    metadata["url"] = record["url"]
                if record.get("key_concepts"):
                    metadata["key_concepts"] = ",".join(record["key_concepts"][:10])

                # Store images for later processing
                if record.get("images") and image_cache:
                    doc_images[doc_id] = record["images"]

                documents.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata=metadata,
                    )
                )

            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at line {}: {}", line_num, e)
                console.print(f"[yellow]Skipping invalid JSON at line {line_num}: {e}[/]")

    if not documents:
        logger.warning("No valid documents found in {}", jsonl_path)
        console.print("[yellow]No valid documents found[/]")
        return 0

    logger.info("Found {} documents to ingest from {}", len(documents), source_name)
    console.print(f"[green]Found {len(documents)} documents to ingest[/]")

    # Batch embed and store
    total_ingested = 0
    logger.debug("Starting batch embedding and storage (batch_size={})", batch_size)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding and storing...", total=len(documents))

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug(
                "Processing batch {}/{}", batch_num, (len(documents) + batch_size - 1) // batch_size
            )

            # Create embeddings
            texts = [doc.content for doc in batch]
            embeddings = await embedding_provider.embed_texts(texts)

            # Store in vector database
            await vector_store.add_documents(batch, embeddings)

            total_ingested += len(batch)
            progress.advance(task, len(batch))

    logger.info("Ingested {} documents from {}", total_ingested, source_name)
    console.print(f"[green]✓ Ingested {total_ingested} documents from {source_name}[/]")

    # Process images if cache is provided
    if image_cache and doc_images:
        total_images = sum(len(imgs) for imgs in doc_images.values())
        logger.info("Fetching {} images from {} documents", total_images, len(doc_images))
        console.print(
            f"[blue]Fetching {total_images} images from {len(doc_images)} documents...[/]"
        )

        images_cached = 0
        images_failed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Caching images...", total=total_images)

            for doc_id, images in doc_images.items():
                for img_data in images:
                    try:
                        cached_img = await image_cache.fetch_and_cache(
                            url=img_data.get("url", ""),
                            doc_id=doc_id,
                            alt_text=img_data.get("alt_text"),
                            title=img_data.get("title"),
                            caption=img_data.get("caption"),
                            context=img_data.get("context"),
                        )
                        image_cache.index.add_image(doc_id, cached_img)

                        if cached_img.fetch_error:
                            logger.debug(
                                "Failed to fetch image for doc {}: {}",
                                doc_id,
                                cached_img.fetch_error,
                            )
                            images_failed += 1
                        else:
                            images_cached += 1

                    except Exception as e:
                        logger.warning("Exception caching image for doc {}: {}", doc_id, e)
                        console.print(f"[yellow]Failed to cache image: {e}[/]")
                        images_failed += 1

                    progress.advance(task)

            # Save the index
            image_cache.save_index()
            logger.debug("Image index saved")

        logger.info("Image caching complete: {} cached, {} failed", images_cached, images_failed)
        console.print(
            f"[green]✓ Cached {images_cached} images[/]"
            + (f" [yellow]({images_failed} failed)[/]" if images_failed else "")
        )

    return total_ingested


async def ingest_all_sources(
    content_dir: Path,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    image_cache: ImageCache | None = None,
    batch_size: int = 50,
) -> dict[str, int]:
    """
    Ingest all JSONL files from the content directory.

    Expected files:
    - jama_guide.jsonl: Jama Requirements Management Guide
    - incose_gwr.jsonl: INCOSE Guide for Writing Requirements (future)
    - ears_notation.jsonl: EARS Notation standards (future)

    Args:
        content_dir: Directory containing JSONL content files
        embedding_provider: Provider for creating embeddings
        vector_store: Store for saving documents
        image_cache: Optional image cache for fetching/storing images
        batch_size: Number of documents to embed at once

    Returns:
        Dict mapping source name to document count
    """
    logger.info("Starting ingestion of all sources from {}", content_dir)
    results = {}

    # Map filenames to source names
    source_map = {
        "requirements_management_guide.jsonl": "jama_guide",
        "jama_guide.jsonl": "jama_guide",
        "incose_gwr.jsonl": "incose",
        "ears_notation.jsonl": "ears",
    }

    content_dir = Path(content_dir)

    for filename, source_name in source_map.items():
        filepath = content_dir / filename
        if filepath.exists():
            logger.debug("Found source file: {}", filepath)
            count = await ingest_jsonl(
                filepath,
                source_name,
                embedding_provider,
                vector_store,
                image_cache,
                batch_size,
            )
            results[source_name] = count

    if not results:
        logger.warning("No content files found in {}", content_dir)
        console.print(f"[yellow]No content files found in {content_dir}[/]")
        console.print("Expected files: " + ", ".join(source_map.keys()))
    else:
        total = sum(results.values())
        logger.info("Ingestion complete: {} total documents from {} sources", total, len(results))

    return results
