"""
FastMCP server for requirements management guidance.

Provides tools for searching and retrieving best practices
from authoritative sources like Jama Software, INCOSE, and EARS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from loguru import logger
from mcp.types import TextContent

from .config import settings
from .embeddings import create_embedding_provider
from .vectorstore import create_vector_store

if TYPE_CHECKING:
    from .images import ImageCache

# Initialize components lazily (on first tool call)
_embedding_provider = None
_vector_store = None
_image_cache = None


def get_embedding_provider():
    """Get or create the embedding provider."""
    global _embedding_provider
    if _embedding_provider is None:
        logger.debug("Initializing embedding provider: {}", settings.voyage_model)
        _embedding_provider = create_embedding_provider(
            provider_type="voyage",
            api_key=settings.voyage_api_key,
            model=settings.voyage_model,
        )
        logger.info("Embedding provider initialized successfully")
    return _embedding_provider


def get_vector_store():
    """Get or create the vector store."""
    global _vector_store
    if _vector_store is None:
        logger.debug(
            "Initializing vector store: {} at {}",
            settings.vector_store_type,
            settings.vector_store_path,
        )
        _vector_store = create_vector_store(
            store_type=settings.vector_store_type,
            collection_name=settings.collection_name,
            persist_dir=settings.vector_store_path,
        )
        logger.info("Vector store initialized successfully")
    return _vector_store


def get_image_cache() -> ImageCache | None:
    """Get or create the image cache (if available)."""
    global _image_cache
    if _image_cache is None:
        if settings.image_path.exists():
            logger.debug("Initializing image cache at {}", settings.image_path)
            from .images import ImageCache

            _image_cache = ImageCache(cache_dir=settings.image_path)
            logger.info("Image cache initialized successfully")
        else:
            logger.debug("Image cache path does not exist: {}", settings.image_path)
    return _image_cache


# Create MCP server
mcp = FastMCP(
    name="requirements-advisor",
    instructions=(
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
    include_images: bool = True,
) -> list:
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
        include_images: Include related images in response (default: True)

    Returns:
        Relevant guidance excerpts with source citations, optionally with images
    """
    logger.info("Search request: query='{}', top_k={}, source={}", query[:50], top_k, source)
    top_k = max(1, min(10, top_k))  # Clamp to 1-10

    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    # Create query embedding
    logger.debug("Creating query embedding")
    query_embedding = await embedding_provider.embed_query(query)

    # Build filter if source specified
    filter_metadata = {"source": source} if source else None

    # Search
    logger.debug("Searching vector store")
    results = await vector_store.search(
        query_embedding,
        top_k=top_k,
        filter_metadata=filter_metadata,
    )

    if not results:
        logger.info("No results found for query: {}", query[:50])
        return [
            TextContent(
                type="text",
                text="No relevant guidance found. Try rephrasing your query or removing source filters.",
            )
        ]

    # Format response with citations
    response_parts = []
    doc_ids = []
    for i, result in enumerate(results, 1):
        meta = result.document.metadata
        doc_ids.append(result.document.id)

        # Build citation header
        header = f"[{i}] **{meta.get('title', 'Untitled')}**"
        if meta.get("chapter_title"):
            header += f" - {meta['chapter_title']}"

        source_info = f"Source: {meta.get('source', 'unknown')}"
        if meta.get("url"):
            source_info += f" | URL: {meta['url']}"

        # Truncate content if very long
        content = result.document.content
        if len(content) > 1500:
            content = content[:1500] + "..."

        response_parts.append(
            f"{header}\n({source_info})\nRelevance: {result.score:.2%}\n\n{content}"
        )

    text_response = "\n\n---\n\n".join(response_parts)

    # Build response content
    response_content = [TextContent(type="text", text=text_response)]

    # Add images if requested and available
    if include_images:
        image_cache = get_image_cache()
        if image_cache:
            cached_images = image_cache.get_images_for_documents(doc_ids)
            for img in cached_images:
                base64_data = image_cache.load_image_as_base64(img)
                if base64_data:
                    response_content.append(
                        Image(data=base64_data, media_type=img.media_type).to_image_content()
                    )
            logger.debug("Attached {} images to response", len(cached_images))

    logger.info("Search completed: {} results returned", len(results))
    return response_content


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
    logger.info("Definition request: term='{}'", term)
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    # Search for definition
    query = f"definition of {term} in requirements management"
    query_embedding = await embedding_provider.embed_query(query)

    results = await vector_store.search(query_embedding, top_k=3)

    if not results:
        logger.info("No definition found for term: {}", term)
        return f"No definition found for '{term}'. Try using the search_requirements_guidance tool for more context."

    # Prefer glossary entries
    glossary_results = [r for r in results if r.document.metadata.get("type") == "glossary_term"]

    if glossary_results:
        r = glossary_results[0]
        logger.info("Definition found for '{}' from glossary", term)
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

    logger.info("Definition found for '{}' from general search", term)
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
    logger.info("List topics request")
    vector_store = get_vector_store()

    # Get document count
    count = await vector_store.count()
    logger.debug("Vector store contains {} documents", count)

    # Get available sources
    sources = await vector_store.get_metadata_values("source")

    # Get chapter titles if available
    chapters = await vector_store.get_metadata_values("chapter_title")

    response = "**Knowledge Base Summary**\n\n"
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
async def get_best_practices(topic: str, include_images: bool = True) -> list:
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
        include_images: Include related images in response (default: True)

    Returns:
        Best practices with explanations and source citations
    """
    logger.info("Best practices request: topic='{}'", topic)
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()

    query = f"best practices for {topic} in requirements management"
    query_embedding = await embedding_provider.embed_query(query)

    results = await vector_store.search(query_embedding, top_k=5)

    if not results:
        logger.info("No best practices found for topic: {}", topic)
        return [
            TextContent(
                type="text",
                text=f"No best practices found for '{topic}'. Try a different topic or use search_requirements_guidance for general queries.",
            )
        ]

    response = f"**Best Practices: {topic.title()}**\n\n"
    doc_ids = []

    for i, result in enumerate(results, 1):
        meta = result.document.metadata
        content = result.document.content
        doc_ids.append(result.document.id)

        # Truncate if needed
        if len(content) > 1000:
            content = content[:1000] + "..."

        response += f"### {i}. {meta.get('title', 'Guidance')}\n"
        response += f"*Source: {meta.get('source', 'unknown')}*\n\n"
        response += f"{content}\n\n"

    # Build response content
    response_content = [TextContent(type="text", text=response)]

    # Add images if requested and available
    if include_images:
        image_cache = get_image_cache()
        if image_cache:
            cached_images = image_cache.get_images_for_documents(doc_ids)
            for img in cached_images:
                base64_data = image_cache.load_image_as_base64(img)
                if base64_data:
                    response_content.append(
                        Image(data=base64_data, media_type=img.media_type).to_image_content()
                    )
            logger.debug("Attached {} images to best practices response", len(cached_images))

    logger.info("Best practices completed: {} results for topic '{}'", len(results), topic)
    return response_content


# Export for uvicorn
def create_app():
    """Create the MCP application for deployment."""
    return mcp


# For SSE transport
def create_sse_app():
    """Create the SSE transport app."""
    return mcp.sse_app()
