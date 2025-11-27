"""
Image caching package.

Provides image fetching, processing, and caching for MCP tool responses.
"""

from .base import CachedImage, ImageIndex
from .cache import ImageCache

__all__ = [
    "CachedImage",
    "ImageIndex",
    "ImageCache",
]
