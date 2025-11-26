"""
CLI for the Requirements Advisor MCP server.

Commands:
- serve: Start the MCP server
- ingest: Load content into vector store
- info: Show configuration and status
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import settings

app = typer.Typer(
    name="requirements-advisor",
    help="MCP server for requirements management best practices",
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(settings.port, "--port", "-p", help="Port to bind to"),
):
    """Start the MCP server with SSE transport."""
    from .server import mcp

    console.print(f"[bold blue]Starting Requirements Advisor MCP Server[/]")
    console.print(f"Host: {host}:{port}")
    console.print(f"SSE endpoint: http://{host}:{port}/sse")
    console.print()

    # Check for API key
    if not settings.voyage_api_key:
        console.print("[red]Warning: VOYAGE_API_KEY not set. Tools will fail.[/]")

    mcp.run(transport="sse", host=host, port=port)


@app.command()
def ingest(
    content_dir: Path = typer.Option(
        settings.content_dir,
        "--content-dir", "-c",
        help="Directory containing JSONL content files",
    ),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Embedding batch size"),
    clear: bool = typer.Option(False, "--clear", help="Clear existing data before ingesting"),
):
    """Ingest content into the vector store."""
    console.print("[bold blue]Content Ingestion[/]")
    
    # Check for API key
    if not settings.voyage_api_key:
        console.print("[red]Error: VOYAGE_API_KEY not set[/]")
        raise typer.Exit(1)
    
    async def run_ingestion():
        from .embeddings import create_embedding_provider
        from .vectorstore import create_vector_store
        from .ingestion import ingest_all_sources
        
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
        
        if clear:
            console.print("[yellow]Clearing existing data...[/]")
            await vector_store.delete_collection()
            # Recreate collection
            vector_store = create_vector_store(
                store_type=settings.vector_store_type,
                collection_name=settings.collection_name,
                persist_dir=settings.vector_store_path,
            )
        
        results = await ingest_all_sources(
            content_dir=content_dir,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            batch_size=batch_size,
        )
        
        if results:
            console.print("\n[bold green]Ingestion Complete![/]")
            table = Table(title="Ingestion Summary")
            table.add_column("Source", style="cyan")
            table.add_column("Documents", style="green")
            for source, count in results.items():
                table.add_row(source, str(count))
            table.add_row("[bold]Total[/]", f"[bold]{sum(results.values())}[/]")
            console.print(table)
        else:
            console.print("[yellow]No content ingested[/]")
    
    asyncio.run(run_ingestion())


@app.command()
def info():
    """Show configuration and vector store status."""
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
        console.print(f"Documents: {count}")
        console.print(f"Sources: {', '.join(sources) if sources else 'none'}")
        
    except Exception as e:
        console.print(f"[red]Error accessing vector store: {e}[/]")


@app.command()
def test_search(
    query: str = typer.Argument(..., help="Query to test"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Number of results"),
):
    """Test a search query against the vector store."""
    if not settings.voyage_api_key:
        console.print("[red]Error: VOYAGE_API_KEY not set[/]")
        raise typer.Exit(1)
    
    async def run_search():
        from .embeddings import create_embedding_provider
        from .vectorstore import create_vector_store
        
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
        
        query_embedding = await embedding_provider.embed_query(query)
        results = await vector_store.search(query_embedding, top_k=top_k)
        
        console.print(f"\n[bold]Results for: '{query}'[/]\n")
        
        for i, result in enumerate(results, 1):
            console.print(f"[cyan]{i}. {result.document.metadata.get('title', 'Untitled')}[/]")
            console.print(f"   Source: {result.document.metadata.get('source', 'unknown')}")
            console.print(f"   Score: {result.score:.2%}")
            console.print(f"   Preview: {result.document.content[:200]}...")
            console.print()
    
    asyncio.run(run_search())


if __name__ == "__main__":
    app()
