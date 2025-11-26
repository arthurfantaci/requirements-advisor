"""
Requirements Advisor MCP Server.

An MCP server providing expert guidance on requirements management
best practices from authoritative sources.

Usage:
    # Start server
    requirements-advisor serve
    
    # Ingest content
    requirements-advisor ingest
    
    # Check status
    requirements-advisor info
"""

__version__ = "0.1.0"

from .server import mcp, create_app, create_sse_app

__all__ = [
    "mcp",
    "create_app",
    "create_sse_app",
]
