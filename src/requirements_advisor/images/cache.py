"""
Image caching utilities.

Handles fetching, processing, and caching images from URLs.
"""

import asyncio
import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path

import httpx
from loguru import logger
from PIL import Image as PILImage
from rich.console import Console

from .base import CachedImage, ImageIndex

console = Console()


class ImageCache:
    """Manages image fetching, processing, and caching."""

    def __init__(
        self,
        cache_dir: Path | str,
        max_dimension: int = 1024,
        quality: int = 85,
        timeout: int = 30,
    ):
        """
        Initialize the image cache.

        Args:
            cache_dir: Directory to store cached images
            max_dimension: Maximum width or height in pixels (preserves aspect ratio)
            quality: JPEG quality (1-100)
            timeout: HTTP request timeout in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_dimension = max_dimension
        self.quality = quality
        self.timeout = timeout
        self.index_path = self.cache_dir / "index.json"
        self._index: ImageIndex | None = None

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "ImageCache initialized: dir={}, max_dim={}, quality={}",
            self.cache_dir,
            max_dimension,
            quality,
        )

    @property
    def index(self) -> ImageIndex:
        """Load or create the image index."""
        if self._index is None:
            self._index = self._load_index()
        return self._index

    def _load_index(self) -> ImageIndex:
        """Load the image index from disk."""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text())
                index = ImageIndex.model_validate(data)
                logger.debug("Loaded image index with {} documents", len(index.images_by_doc))
                return index
            except Exception as e:
                logger.warning("Could not load image index: {}", e)
                console.print(f"[yellow]Warning: Could not load image index: {e}[/]")
        return ImageIndex()

    def save_index(self) -> None:
        """Save the image index to disk."""
        self.index_path.write_text(self.index.model_dump_json(indent=2))
        logger.debug("Saved image index to {}", self.index_path)

    async def fetch_and_cache(
        self,
        url: str,
        doc_id: str,
        alt_text: str | None = None,
        title: str | None = None,
        caption: str | None = None,
        context: str | None = None,
    ) -> CachedImage:
        """
        Fetch an image from URL, process it, and cache locally.

        Args:
            url: Image URL to fetch
            doc_id: Parent document ID
            alt_text: Alternative text for the image
            title: Image title
            caption: Image caption
            context: Section context where image appears

        Returns:
            CachedImage with metadata and file path. If fetch fails,
            the returned CachedImage will have fetch_error set.
        """
        image_id = self._hash_url(url)
        logger.debug("Fetching image: {} (id={})", url[:60], image_id)

        # Check if already cached
        existing = self._find_cached(image_id)
        if existing:
            logger.debug("Image already cached: {}", image_id)
            # Return a copy linked to this document
            return CachedImage(
                id=image_id,
                source_doc_id=doc_id,
                original_url=url,
                alt_text=alt_text,
                title=title,
                caption=caption,
                context=context,
                media_type=existing.media_type,
                file_path=existing.file_path,
                width=existing.width,
                height=existing.height,
                fetch_error=None,
            )

        # Fetch with retries
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    image_data = response.content
                    content_type = response.headers.get("content-type", "image/jpeg")
                    logger.debug("Fetched image: {} bytes, type={}", len(image_data), content_type)
                    break
            except Exception as e:
                if attempt == 2:
                    logger.warning("Failed to fetch image after 3 attempts: {} - {}", url[:60], e)
                    # Return error entry
                    return CachedImage(
                        id=image_id,
                        source_doc_id=doc_id,
                        original_url=url,
                        alt_text=alt_text,
                        title=title,
                        caption=caption,
                        context=context,
                        media_type="image/jpeg",
                        file_path="",
                        width=None,
                        height=None,
                        fetch_error=str(e),
                    )
                logger.debug("Fetch attempt {} failed, retrying: {}", attempt + 1, e)
                await asyncio.sleep(2**attempt)  # Exponential backoff

        # Process and save
        try:
            processed_data, media_type, width, height = self._process_image(
                image_data, content_type
            )
            logger.debug("Processed image: {}x{}, {} bytes", width, height, len(processed_data))
        except Exception as e:
            logger.warning("Failed to process image {}: {}", image_id, e)
            return CachedImage(
                id=image_id,
                source_doc_id=doc_id,
                original_url=url,
                alt_text=alt_text,
                title=title,
                caption=caption,
                context=context,
                media_type="image/jpeg",
                file_path="",
                width=None,
                height=None,
                fetch_error=f"Processing error: {e}",
            )

        # Determine file extension
        ext = "jpg" if media_type == "image/jpeg" else "png"
        file_path = f"{image_id}.{ext}"
        full_path = self.cache_dir / file_path

        # Save to disk
        full_path.write_bytes(processed_data)
        logger.debug("Saved image to {}", full_path)

        return CachedImage(
            id=image_id,
            source_doc_id=doc_id,
            original_url=url,
            alt_text=alt_text,
            title=title,
            caption=caption,
            context=context,
            media_type=media_type,
            file_path=file_path,
            width=width,
            height=height,
            fetch_error=None,
        )

    def _process_image(self, image_data: bytes, content_type: str) -> tuple[bytes, str, int, int]:
        """
        Process an image: resize if needed and optimize.

        Args:
            image_data: Raw image bytes
            content_type: Original content type

        Returns:
            Tuple of (processed_bytes, media_type, width, height)

        Raises:
            PIL.UnidentifiedImageError: If image format is not recognized
            IOError: If image data is corrupted
        """
        img = PILImage.open(BytesIO(image_data))

        # Convert RGBA to RGB if no transparency (allows JPEG conversion)
        has_transparency = img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        )

        if img.mode == "RGBA" and not has_transparency:
            img = img.convert("RGB")
        elif img.mode == "P":
            img = img.convert("RGBA") if has_transparency else img.convert("RGB")
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        # Resize if needed (preserve aspect ratio)
        width, height = img.size
        if width > self.max_dimension or height > self.max_dimension:
            ratio = min(self.max_dimension / width, self.max_dimension / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            width, height = new_width, new_height

        # Save to bytes
        output = BytesIO()
        if img.mode == "RGBA" or has_transparency:
            img.save(output, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            img.save(output, format="JPEG", quality=self.quality, optimize=True)
            media_type = "image/jpeg"

        return output.getvalue(), media_type, width, height

    def _hash_url(self, url: str) -> str:
        """Generate a hash-based ID from a URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def _find_cached(self, image_id: str) -> CachedImage | None:
        """Find an existing cached image by ID."""
        for images in self.index.images_by_doc.values():
            for img in images:
                if img.id == image_id and img.file_path and not img.fetch_error:
                    return img
        return None

    def get_images_for_documents(self, doc_ids: list[str]) -> list[CachedImage]:
        """
        Get all cached images for a list of document IDs.

        Args:
            doc_ids: List of document IDs

        Returns:
            List of CachedImage objects (excludes failed fetches)
        """
        return [
            img
            for img in self.index.get_images_for_docs(doc_ids)
            if not img.fetch_error and img.file_path
        ]

    def load_image_as_base64(self, image: CachedImage) -> str | None:
        """
        Load an image file and return as base64 string.

        Args:
            image: CachedImage with file_path

        Returns:
            Base64-encoded image data, or None if file not found
        """
        if not image.file_path or image.fetch_error:
            return None

        file_path = self.cache_dir / image.file_path
        if not file_path.exists():
            return None

        return base64.b64encode(file_path.read_bytes()).decode("utf-8")
