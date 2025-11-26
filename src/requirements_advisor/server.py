"""
FastMCP server for requirements management guidance.

Provides tools for searching and retrieving best practices
from authoritative sources like Jama Software, INCOSE, and EARS.
"""

from fastmcp import FastMCP

from .config import settings
from .embeddings import create_embedding_provider
from .vectorstore import create_vector_store

# Initialize components lazily (on first tool call)
_embedding_provider = None
_vector_store = None


def get_embedding_provider():
    """Get or create the embedding provider."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = create_embedding_provider(
            provider_type="voyage",
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
        )
    return _embedding_provider


def get_vector_store():
    """Get or create the vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store(
            store_type=settings.vector_store_type,
            collection_name=settings.collection_name,
            persist_dir=settings.vector_store_path,
        )
    return _vector_store


# Create MCP server
mcp = FastMCP(
    "requirements-advisor",
    description=(
        "Expert guidance on requirements management best practices. "
        "Provides answers from authoritative sources including Jama Software's "
        "Essential Guide to Requirements Management, INCOSE guidelines, and EARS notation."
    ),
)


@mcp.tool()
async def search_requirements_guidance(
    query: str,
    top_k: int = 5,
    source: str | None = None,
) -> str:
    """
    Search requirements management best practices and guidance.
    
    Use this tool to find expert guidance on topics like:
    - Writing effective requirements
    - Requirements traceability
    - Validation and verification
    - Regulatory compliance
    - Systems engineering
    - Industry-specific practices (medical, automotive, aerospace)
    
    Args:
        query: Natural language question about requirements management
        top_k: Number of results to return (1-10, default: 5)
        source: Optional filter by source ("jama_guide", "incose", "ears")
    
    Returns:
        Relevant guidance excerpts with source citations
    """
    top_k = max(1, min(10, top_k))  # Clamp to 1-10
    
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    
    # Create query embedding
    query_embedding = await embedding_provider.embed_query(query)
    
    # Build filter if source specified
    filter_metadata = {"source": source} if source else None
    
    # Search
    results = await vector_store.search(
        query_embedding,
        top_k=top_k,
        filter_metadata=filter_metadata,
    )
    
    if not results:
        return "No relevant guidance found. Try rephrasing your query or removing source filters."
    
    # Format response with citations
    response_parts = []
    for i, result in enumerate(results, 1):
        meta = result.document.metadata
        
        # Build citation header
        header = f"[{i}] **{meta.get('title', 'Untitled')}**"
        if meta.get("chapter_title"):
            header += f" — {meta['chapter_title']}"
        
        source_info = f"Source: {meta.get('source', 'unknown')}"
        if meta.get("url"):
            source_info += f" | URL: {meta['url']}"
        
        # Truncate content if very long
        content = result.document.content
        if len(content) > 1500:
            content = content[:1500] + "..."
        
        response_parts.append(
            f"{header}\n"
            f"({source_info})\n"
            f"Relevance: {result.score:.2%}\n\n"
            f"{content}"
        )
    
    return "\n\n---\n\n".join(response_parts)


@mcp.tool()
async def get_definition(term: str) -> str:
    """
    Get the definition of a requirements management term or acronym.
    
    Use this for terms like:
    - SRS (System Requirements Specification)
    - EARS (Easy Approach to Requirements Syntax)
    - Traceability
    - V&V (Verification and Validation)
    - RTM (Requirements Traceability Matrix)
    
    Args:
        term: The term or acronym to define
    
    Returns:
        Definition with source attribution
    """
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    
    # Search for definition
    query = f"definition of {term} in requirements management"
    query_embedding = await embedding_provider.embed_query(query)
    
    results = await vector_store.search(query_embedding, top_k=3)
    
    if not results:
        return f"No definition found for '{term}'. Try using the search_requirements_guidance tool for more context."
    
    # Prefer glossary entries
    glossary_results = [
        r for r in results 
        if r.document.metadata.get("type") == "glossary_term"
    ]
    
    if glossary_results:
        r = glossary_results[0]
        return (
            f"**{term}**\n\n"
            f"{r.document.content}\n\n"
            f"— Source: {r.document.metadata.get('source', 'unknown')}"
        )
    
    # Fall back to most relevant result
    r = results[0]
    content = r.document.content
    if len(content) > 800:
        content = content[:800] + "..."
    
    return (
        f"**{term}** (from {r.document.metadata.get('title', 'unknown source')})\n\n"
        f"{content}\n\n"
        f"— Source: {r.document.metadata.get('source', 'unknown')}"
    )


@mcp.tool()
async def list_available_topics() -> str:
    """
    List the topics and sources available in the knowledge base.
    
    Use this to understand what guidance is available before searching.
    
    Returns:
        Summary of available topics and content sources
    """
    vector_store = get_vector_store()
    
    # Get document count
    count = await vector_store.count()
    
    # Get available sources
    sources = await vector_store.get_metadata_values("source")
    
    # Get chapter titles if available
    chapters = await vector_store.get_metadata_values("chapter_title")
    
    response = f"**Knowledge Base Summary**\n\n"
    response += f"Total documents: {count}\n\n"
    
    response += "**Available Sources:**\n"
    source_descriptions = {
        "jama_guide": "Jama Software's Essential Guide to Requirements Management and Traceability",
        "incose": "INCOSE Guide for Writing Requirements",
        "ears": "EARS (Easy Approach to Requirements Syntax) Notation",
    }
    for source in sources:
        desc = source_descriptions.get(source, source)
        response += f"- `{source}`: {desc}\n"
    
    if chapters:
        response += "\n**Topics Covered:**\n"
        for chapter in chapters[:15]:  # Limit to avoid overwhelming
            response += f"- {chapter}\n"
        if len(chapters) > 15:
            response += f"- ... and {len(chapters) - 15} more topics\n"
    
    response += "\n**Example Queries:**\n"
    response += "- 'How do I write good functional requirements?'\n"
    response += "- 'What is requirements traceability and why does it matter?'\n"
    response += "- 'Best practices for medical device requirements'\n"
    response += "- 'How to use EARS notation for requirements'\n"
    
    return response


@mcp.tool()
async def get_best_practices(topic: str) -> str:
    """
    Get best practices for a specific requirements management topic.
    
    Good topics include:
    - writing requirements
    - requirements traceability
    - requirements validation
    - change management
    - regulatory compliance
    - agile requirements
    
    Args:
        topic: The topic to get best practices for
    
    Returns:
        Best practices with explanations and source citations
    """
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    
    query = f"best practices for {topic} in requirements management"
    query_embedding = await embedding_provider.embed_query(query)
    
    results = await vector_store.search(query_embedding, top_k=5)
    
    if not results:
        return f"No best practices found for '{topic}'. Try a different topic or use search_requirements_guidance for general queries."
    
    response = f"**Best Practices: {topic.title()}**\n\n"
    
    for i, result in enumerate(results, 1):
        meta = result.document.metadata
        content = result.document.content
        
        # Truncate if needed
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        response += f"### {i}. {meta.get('title', 'Guidance')}\n"
        response += f"*Source: {meta.get('source', 'unknown')}*\n\n"
        response += f"{content}\n\n"
    
    return response


# Export for uvicorn
def create_app():
    """Create the MCP application for deployment."""
    return mcp


# For SSE transport
def create_sse_app():
    """Create the SSE transport app."""
    return mcp.sse_app()
