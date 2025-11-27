"""
Data models for image caching.

Provides Pydantic models for cached images and the image index.
"""

from pydantic import BaseModel, Field


class CachedImage(BaseModel):
    """A cached image with metadata."""

    id: str = Field(description="Hash-based unique identifier")
    source_doc_id: str = Field(description="Reference to parent document ID")
    original_url: str = Field(description="Original source URL")
    alt_text: str | None = Field(default=None, description="Alternative text")
    title: str | None = Field(default=None, description="Image title")
    caption: str | None = Field(default=None, description="Image caption")
    context: str | None = Field(default=None, description="Section context where image appears")
    media_type: str = Field(description="MIME type (e.g., 'image/jpeg', 'image/png')")
    file_path: str = Field(description="Relative path to image file in cache directory")
    width: int | None = Field(default=None, description="Image width in pixels")
    height: int | None = Field(default=None, description="Image height in pixels")
    fetch_error: str | None = Field(default=None, description="Error message if fetch failed")


class ImageIndex(BaseModel):
    """Index mapping document IDs to their cached images."""

    images_by_doc: dict[str, list[CachedImage]] = Field(
        default_factory=dict,
        description="Mapping from document ID to list of cached images",
    )

    def add_image(self, doc_id: str, image: CachedImage) -> None:
        """Add an image to the index for a document."""
        if doc_id not in self.images_by_doc:
            self.images_by_doc[doc_id] = []
        self.images_by_doc[doc_id].append(image)

    def get_images(self, doc_id: str) -> list[CachedImage]:
        """Get all images for a document."""
        return self.images_by_doc.get(doc_id, [])

    def get_images_for_docs(self, doc_ids: list[str]) -> list[CachedImage]:
        """Get all images for multiple documents."""
        images = []
        for doc_id in doc_ids:
            images.extend(self.get_images(doc_id))
        return images
