"""
CLI for the Requirements Advisor MCP server.

Commands:
- serve: Start the MCP server
- ingest: Load content into vector store
- info: Show configuration and status
- test-search: Test search queries
"""

import asyncio
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from .config import settings
from .logging import setup_logging

app = typer.Typer(
    name="requirements-advisor",
    help="MCP server for requirements management best practices",
)
console = Console()


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
):
    """Requirements Advisor - MCP server for requirements management guidance."""
    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(level=log_level, json_output=settings.log_json)
    logger.debug("CLI initialized with log level: {}", log_level)


@app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Port to bind to"),
):
    """Start the MCP server with Streamable HTTP transport."""
    from .server import mcp

    logger.info("Starting MCP server on {}:{}", host, port)
    console.print("[bold blue]Starting Requirements Advisor MCP Server[/]")
    console.print(f"Host: {host}:{port}")
    console.print(f"MCP endpoint: http://{host}:{port}/mcp")
    console.print()

    # Check for API key
    if not settings.voyage_api_key:
        logger.warning("VOYAGE_API_KEY not set - tools will fail")
        console.print("[red]Warning: VOYAGE_API_KEY not set. Tools will fail.[/]")

    mcp.run(transport="http", host=host, port=port, path="/mcp")


@app.command()
def ingest(
    content_dir: Path = typer.Option(
        settings.content_dir,
        "--content-dir",
        "-c",
        help="Directory containing JSONL content files",
    ),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Embedding batch size"),
    clear: bool = typer.Option(False, "--clear", help="Clear existing data before ingesting"),
    fetch_images: bool = typer.Option(
        True,
        "--fetch-images/--no-images",
        help="Fetch and cache images from content",
    ),
):
    """Ingest content into the vector store."""
    logger.info("Starting content ingestion from {}", content_dir)
    logger.debug(
        "Ingestion options: batch_size={}, clear={}, fetch_images={}",
        batch_size,
        clear,
        fetch_images,
    )
    console.print("[bold blue]Content Ingestion[/]")

    # Check for API key
    if not settings.voyage_api_key:
        logger.error("VOYAGE_API_KEY not set - cannot proceed with ingestion")
        console.print("[red]Error: VOYAGE_API_KEY not set[/]")
        raise typer.Exit(1)

    async def run_ingestion():
        from .embeddings import create_embedding_provider
        from .ingestion import ingest_all_sources
        from .vectorstore import create_vector_store

        logger.debug("Creating embedding provider: {}", settings.voyage_model)
        embedding_provider = create_embedding_provider(
            provider_type="voyage",
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
        )

        logger.debug(
            "Creating vector store: {} at {}",
            settings.vector_store_type,
            settings.vector_store_path,
        )
        vector_store = create_vector_store(
            store_type=settings.vector_store_type,
            collection_name=settings.collection_name,
            persist_dir=settings.vector_store_path,
        )

        # Initialize image cache if requested
        image_cache = None
        if fetch_images:
            import shutil

            from .images import ImageCache

            if clear and settings.image_path.exists():
                logger.info("Clearing image cache at {}", settings.image_path)
                console.print("[yellow]Clearing image cache...[/]")
                shutil.rmtree(settings.image_path)

            logger.debug("Initializing image cache at {}", settings.image_path)
            image_cache = ImageCache(
                cache_dir=settings.image_path,
                max_dimension=settings.image_max_dimension,
                quality=settings.image_quality,
                timeout=settings.image_fetch_timeout,
            )
            console.print(f"[blue]Image caching enabled (max {settings.image_max_dimension}px)[/]")

        if clear:
            logger.info("Clearing existing vector store data")
            console.print("[yellow]Clearing existing data...[/]")
            await vector_store.delete_collection()
            # Recreate collection
            logger.debug("Recreating vector store collection")
            vector_store = create_vector_store(
                store_type=settings.vector_store_type,
                collection_name=settings.collection_name,
                persist_dir=settings.vector_store_path,
            )

        logger.info("Starting ingestion pipeline")
        results = await ingest_all_sources(
            content_dir=content_dir,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            image_cache=image_cache,
            batch_size=batch_size,
        )

        if results:
            total_docs = sum(results.values())
            logger.info(
                "Ingestion complete: {} documents from {} sources", total_docs, len(results)
            )
            console.print("\n[bold green]Ingestion Complete![/]")
            table = Table(title="Ingestion Summary")
            table.add_column("Source", style="cyan")
            table.add_column("Documents", style="green")
            for source, count in results.items():
                logger.debug("Source '{}': {} documents", source, count)
                table.add_row(source, str(count))
            table.add_row("[bold]Total[/]", f"[bold]{total_docs}[/]")
            console.print(table)
        else:
            logger.warning("No content was ingested")
            console.print("[yellow]No content ingested[/]")

    asyncio.run(run_ingestion())


@app.command()
def info():
    """Show configuration and vector store status."""
    logger.debug("Displaying configuration and status")
    console.print("[bold blue]Requirements Advisor Configuration[/]")

    table = Table(title="Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Voyage Model", settings.voyage_model)
    table.add_row("Voyage API Key", "***" if settings.voyage_api_key else "[red]NOT SET[/]")
    table.add_row("Vector Store Type", settings.vector_store_type)
    table.add_row("Vector Store Path", settings.vector_store_path)
    table.add_row("Collection Name", settings.collection_name)
    table.add_row("Content Directory", settings.content_dir)
    table.add_row("Image Cache Path", settings.image_cache_path)
    table.add_row("Server Host", settings.host)
    table.add_row("Server Port", str(settings.port))

    console.print(table)

    # Check vector store status
    console.print("\n[bold]Vector Store Status[/]")
    try:
        from .vectorstore import create_vector_store

        vector_store = create_vector_store(
            store_type=settings.vector_store_type,
            collection_name=settings.collection_name,
            persist_dir=settings.vector_store_path,
        )

        async def get_status():
            count = await vector_store.count()
            sources = await vector_store.get_metadata_values("source")
            return count, sources

        count, sources = asyncio.run(get_status())
        logger.debug("Vector store status: {} documents, sources: {}", count, sources)
        console.print(f"Documents: {count}")
        console.print(f"Sources: {', '.join(sources) if sources else 'none'}")

    except Exception as e:
        logger.error("Error accessing vector store: {}", e)
        console.print(f"[red]Error accessing vector store: {e}[/]")

    # Check image cache status
    console.print("\n[bold]Image Cache Status[/]")
    try:
        from .images import ImageCache

        if settings.image_path.exists():
            image_cache = ImageCache(cache_dir=settings.image_path)
            total_images = sum(len(imgs) for imgs in image_cache.index.images_by_doc.values())
            successful = sum(
                1
                for imgs in image_cache.index.images_by_doc.values()
                for img in imgs
                if not img.fetch_error
            )
            logger.debug(
                "Image cache status: {} successful, {} failed, {} docs",
                successful,
                total_images - successful,
                len(image_cache.index.images_by_doc),
            )
            console.print(f"Cached images: {successful}")
            if total_images > successful:
                console.print(f"Failed fetches: {total_images - successful}")
            console.print(f"Documents with images: {len(image_cache.index.images_by_doc)}")
        else:
            logger.debug("Image cache path does not exist: {}", settings.image_path)
            console.print("Image cache not initialized (run ingest with --fetch-images)")
    except Exception as e:
        logger.error("Error accessing image cache: {}", e)
        console.print(f"[red]Error accessing image cache: {e}[/]")


@app.command()
def test_search(
    query: str = typer.Argument(..., help="Query to test"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results"),
):
    """Test a search query against the vector store."""
    logger.info("Testing search query: '{}' (top_k={})", query[:50], top_k)

    if not settings.voyage_api_key:
        logger.error("VOYAGE_API_KEY not set - cannot perform search")
        console.print("[red]Error: VOYAGE_API_KEY not set[/]")
        raise typer.Exit(1)

    async def run_search():
        from .embeddings import create_embedding_provider
        from .vectorstore import create_vector_store

        logger.debug("Creating embedding provider and vector store")
        embedding_provider = create_embedding_provider(
            provider_type="voyage",
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
        )

        vector_store = create_vector_store(
            store_type=settings.vector_store_type,
            collection_name=settings.collection_name,
            persist_dir=settings.vector_store_path,
        )

        logger.debug("Generating query embedding")
        query_embedding = await embedding_provider.embed_query(query)

        logger.debug("Searching vector store")
        results = await vector_store.search(query_embedding, top_k=top_k)

        logger.info("Search returned {} results", len(results))
        console.print(f"\n[bold]Results for: '{query}'[/]\n")

        for i, result in enumerate(results, 1):
            logger.debug(
                "Result {}: '{}' (score: {:.2%})",
                i,
                result.document.metadata.get("title", "Untitled"),
                result.score,
            )
            console.print(f"[cyan]{i}. {result.document.metadata.get('title', 'Untitled')}[/]")
            console.print(f"   Source: {result.document.metadata.get('source', 'unknown')}")
            console.print(f"   Score: {result.score:.2%}")
            console.print(f"   Preview: {result.document.content[:200]}...")
            console.print()

    asyncio.run(run_search())


if __name__ == "__main__":
    app()
