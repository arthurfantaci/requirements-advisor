"""
Content ingestion pipeline.

Loads JSONL content files, creates embeddings, and stores in vector database.
"""

import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..embeddings.base import EmbeddingProvider
from ..vectorstore.base import Document, VectorStore

console = Console()


async def ingest_jsonl(
    jsonl_path: Path,
    source_name: str,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    batch_size: int = 50,
) -> int:
    """
    Ingest a JSONL file into the vector store.
    
    Each line in the JSONL should have:
    - article_id or term: unique identifier
    - markdown_content or definition: text content
    - title: document title
    - Optional: chapter_title, chapter_number, url, type, key_concepts
    
    Args:
        jsonl_path: Path to JSONL file
        source_name: Source identifier (e.g., "jama_guide", "incose", "ears")
        embedding_provider: Provider for creating embeddings
        vector_store: Store for saving documents
        batch_size: Number of documents to embed at once
        
    Returns:
        Number of documents ingested
    """
    if not jsonl_path.exists():
        console.print(f"[red]File not found: {jsonl_path}[/]")
        return 0
    
    console.print(f"[blue]Loading content from {jsonl_path}[/]")
    
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
                
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                ))
                
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Skipping invalid JSON at line {line_num}: {e}[/]")
    
    if not documents:
        console.print("[yellow]No valid documents found[/]")
        return 0
    
    console.print(f"[green]Found {len(documents)} documents to ingest[/]")
    
    # Batch embed and store
    total_ingested = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding and storing...", total=len(documents))
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Create embeddings
            texts = [doc.content for doc in batch]
            embeddings = await embedding_provider.embed_texts(texts)
            
            # Store in vector database
            await vector_store.add_documents(batch, embeddings)
            
            total_ingested += len(batch)
            progress.advance(task, len(batch))
    
    console.print(f"[green]✓ Ingested {total_ingested} documents from {source_name}[/]")
    return total_ingested


async def ingest_all_sources(
    content_dir: Path,
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    batch_size: int = 50,
) -> dict[str, int]:
    """
    Ingest all JSONL files from the content directory.
    
    Expected files:
    - jama_guide.jsonl: Jama Requirements Management Guide
    - incose_gwr.jsonl: INCOSE Guide for Writing Requirements (future)
    - ears_notation.jsonl: EARS Notation standards (future)
    
    Returns:
        Dict mapping source name to document count
    """
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
            count = await ingest_jsonl(
                filepath,
                source_name,
                embedding_provider,
                vector_store,
                batch_size,
            )
            results[source_name] = count
    
    if not results:
        console.print(f"[yellow]No content files found in {content_dir}[/]")
        console.print("Expected files: " + ", ".join(source_map.keys()))
    
    return results
