"""
Tests for image caching module.

Tests ImageCache class, image processing, and index management.
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import respx
from httpx import Response
from PIL import Image

from requirements_advisor.images.base import CachedImage, ImageIndex
from requirements_advisor.images.cache import ImageCache


class TestCachedImage:
    """Test CachedImage model."""

    def test_create_cached_image(self):
        """Test creating a cached image."""
        img = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )

        assert img.id == "abc123"
        assert img.source_doc_id == "doc-1"
        assert img.original_url == "https://example.com/image.jpg"
        assert img.fetch_error is None

    def test_cached_image_with_error(self):
        """Test cached image with fetch error."""
        img = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image.jpg",
            media_type="image/jpeg",
            file_path="",
            fetch_error="Connection timeout",
        )

        assert img.fetch_error == "Connection timeout"
        assert img.file_path == ""

    def test_cached_image_optional_fields(self):
        """Test cached image with optional fields."""
        img = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
            alt_text="A diagram",
            title="Figure 1",
            caption="This shows the process",
            context="Chapter 3",
            width=800,
            height=600,
        )

        assert img.alt_text == "A diagram"
        assert img.title == "Figure 1"
        assert img.caption == "This shows the process"
        assert img.context == "Chapter 3"
        assert img.width == 800
        assert img.height == 600


class TestImageIndex:
    """Test ImageIndex model."""

    def test_create_empty_index(self):
        """Test creating an empty index."""
        index = ImageIndex()

        assert index.images_by_doc == {}

    def test_add_image(self):
        """Test adding an image to the index."""
        index = ImageIndex()
        img = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )

        index.add_image("doc-1", img)

        assert "doc-1" in index.images_by_doc
        assert len(index.images_by_doc["doc-1"]) == 1
        assert index.images_by_doc["doc-1"][0] == img

    def test_add_multiple_images_to_doc(self):
        """Test adding multiple images to same document."""
        index = ImageIndex()
        img1 = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image1.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )
        img2 = CachedImage(
            id="def456",
            source_doc_id="doc-1",
            original_url="https://example.com/image2.jpg",
            media_type="image/png",
            file_path="def456.png",
        )

        index.add_image("doc-1", img1)
        index.add_image("doc-1", img2)

        assert len(index.images_by_doc["doc-1"]) == 2

    def test_get_images(self):
        """Test getting images for a document."""
        index = ImageIndex()
        img = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )
        index.add_image("doc-1", img)

        result = index.get_images("doc-1")

        assert result == [img]

    def test_get_images_empty(self):
        """Test getting images for document with no images."""
        index = ImageIndex()

        result = index.get_images("nonexistent")

        assert result == []

    def test_get_images_for_docs(self):
        """Test getting images for multiple documents."""
        index = ImageIndex()
        img1 = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image1.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )
        img2 = CachedImage(
            id="def456",
            source_doc_id="doc-2",
            original_url="https://example.com/image2.jpg",
            media_type="image/png",
            file_path="def456.png",
        )
        index.add_image("doc-1", img1)
        index.add_image("doc-2", img2)

        result = index.get_images_for_docs(["doc-1", "doc-2", "doc-3"])

        assert len(result) == 2


class TestImageCache:
    """Test ImageCache class."""

    @pytest.fixture
    def image_cache(self, temp_image_cache_dir):
        """Create an ImageCache instance for testing."""
        return ImageCache(
            cache_dir=temp_image_cache_dir,
            max_dimension=512,
            quality=80,
            timeout=10,
        )

    def test_init_creates_directory(self, temp_dir):
        """Test that init creates cache directory."""
        cache_dir = temp_dir / "new_cache"
        ImageCache(cache_dir=cache_dir)

        assert cache_dir.exists()

    def test_init_loads_existing_index(self, temp_image_cache_dir):
        """Test that init loads existing index."""
        # Create an index file
        index_data = {
            "images_by_doc": {
                "doc-1": [
                    {
                        "id": "abc123",
                        "source_doc_id": "doc-1",
                        "original_url": "https://example.com/image.jpg",
                        "media_type": "image/jpeg",
                        "file_path": "abc123.jpg",
                    }
                ]
            }
        }
        index_path = temp_image_cache_dir / "index.json"
        index_path.write_text(json.dumps(index_data))

        cache = ImageCache(cache_dir=temp_image_cache_dir)

        assert "doc-1" in cache.index.images_by_doc

    def test_save_index(self, image_cache, temp_image_cache_dir):
        """Test saving index to disk."""
        img = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )
        image_cache.index.add_image("doc-1", img)

        image_cache.save_index()

        index_path = temp_image_cache_dir / "index.json"
        assert index_path.exists()
        data = json.loads(index_path.read_text())
        assert "doc-1" in data["images_by_doc"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_and_cache_success(self, image_cache, sample_image_bytes):
        """Test successful image fetch and cache."""
        respx.get("https://example.com/image.jpg").mock(
            return_value=Response(200, content=sample_image_bytes)
        )

        result = await image_cache.fetch_and_cache(
            url="https://example.com/image.jpg",
            doc_id="doc-1",
            alt_text="Test image",
        )

        assert result.id is not None
        assert result.source_doc_id == "doc-1"
        assert result.alt_text == "Test image"
        assert result.fetch_error is None
        assert result.file_path != ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_and_cache_already_cached(self, image_cache, sample_image_bytes):
        """Test that already cached images are returned from cache."""
        respx.get("https://example.com/image.jpg").mock(
            return_value=Response(200, content=sample_image_bytes)
        )

        # First fetch
        result1 = await image_cache.fetch_and_cache(
            url="https://example.com/image.jpg", doc_id="doc-1"
        )
        image_cache.index.add_image("doc-1", result1)

        # Second fetch should return cached version
        result2 = await image_cache.fetch_and_cache(
            url="https://example.com/image.jpg", doc_id="doc-2"
        )

        assert result2.id == result1.id
        assert result2.source_doc_id == "doc-2"
        # Should only have made one HTTP request
        assert len(respx.calls) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_and_cache_failure(self, image_cache):
        """Test handling of fetch failure."""
        respx.get("https://example.com/image.jpg").mock(
            return_value=Response(404)
        )

        result = await image_cache.fetch_and_cache(
            url="https://example.com/image.jpg", doc_id="doc-1"
        )

        assert result.fetch_error is not None
        assert result.file_path == ""

    def test_hash_url(self, image_cache):
        """Test URL hashing produces consistent IDs."""
        url = "https://example.com/image.jpg"

        hash1 = image_cache._hash_url(url)
        hash2 = image_cache._hash_url(url)

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_hash_url_different_urls(self, image_cache):
        """Test different URLs produce different hashes."""
        hash1 = image_cache._hash_url("https://example.com/image1.jpg")
        hash2 = image_cache._hash_url("https://example.com/image2.jpg")

        assert hash1 != hash2

    def test_get_images_for_documents(self, image_cache):
        """Test getting images for multiple documents."""
        img1 = CachedImage(
            id="abc123",
            source_doc_id="doc-1",
            original_url="https://example.com/image1.jpg",
            media_type="image/jpeg",
            file_path="abc123.jpg",
        )
        img2 = CachedImage(
            id="def456",
            source_doc_id="doc-2",
            original_url="https://example.com/image2.jpg",
            media_type="image/png",
            file_path="def456.png",
            fetch_error=None,
        )
        img_error = CachedImage(
            id="err789",
            source_doc_id="doc-3",
            original_url="https://example.com/error.jpg",
            media_type="image/jpeg",
            file_path="",
            fetch_error="Failed",
        )

        image_cache.index.add_image("doc-1", img1)
        image_cache.index.add_image("doc-2", img2)
        image_cache.index.add_image("doc-3", img_error)

        # Should exclude error images
        result = image_cache.get_images_for_documents(["doc-1", "doc-2", "doc-3"])

        assert len(result) == 2
        assert all(img.fetch_error is None for img in result)

    def test_load_image_as_base64(self, image_cache, temp_image_cache_dir, sample_image_bytes):
        """Test loading image as base64."""
        # Write a sample image file
        image_path = temp_image_cache_dir / "test.jpg"
        image_path.write_bytes(sample_image_bytes)

        img = CachedImage(
            id="test",
            source_doc_id="doc-1",
            original_url="https://example.com/test.jpg",
            media_type="image/jpeg",
            file_path="test.jpg",
        )

        result = image_cache.load_image_as_base64(img)

        assert result is not None
        # Should be valid base64
        import base64

        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_load_image_as_base64_not_found(self, image_cache):
        """Test loading non-existent image returns None."""
        img = CachedImage(
            id="nonexistent",
            source_doc_id="doc-1",
            original_url="https://example.com/missing.jpg",
            media_type="image/jpeg",
            file_path="nonexistent.jpg",
        )

        result = image_cache.load_image_as_base64(img)

        assert result is None

    def test_load_image_as_base64_with_error(self, image_cache):
        """Test loading image with fetch error returns None."""
        img = CachedImage(
            id="error",
            source_doc_id="doc-1",
            original_url="https://example.com/error.jpg",
            media_type="image/jpeg",
            file_path="",
            fetch_error="Failed to fetch",
        )

        result = image_cache.load_image_as_base64(img)

        assert result is None


class TestImageProcessing:
    """Test image processing functionality."""

    @pytest.fixture
    def image_cache(self, temp_image_cache_dir):
        """Create ImageCache with specific dimensions for testing."""
        return ImageCache(
            cache_dir=temp_image_cache_dir, max_dimension=100, quality=85
        )

    def test_process_image_resize_large(self, image_cache):
        """Test that large images are resized."""
        # Create a 200x200 image
        img = Image.new("RGB", (200, 200), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        processed, media_type, width, height = image_cache._process_image(
            image_data, "image/jpeg"
        )

        # Should be resized to max 100px
        assert width <= 100
        assert height <= 100

    def test_process_image_preserve_aspect_ratio(self, image_cache):
        """Test that aspect ratio is preserved during resize."""
        # Create a 200x100 image (2:1 ratio)
        img = Image.new("RGB", (200, 100), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        _, _, width, height = image_cache._process_image(image_data, "image/jpeg")

        # Should maintain 2:1 ratio
        assert width == 100
        assert height == 50

    def test_process_image_small_not_resized(self, image_cache):
        """Test that small images are not resized."""
        # Create a 50x50 image (smaller than max)
        img = Image.new("RGB", (50, 50), color="green")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        _, _, width, height = image_cache._process_image(image_data, "image/jpeg")

        assert width == 50
        assert height == 50

    def test_process_image_preserves_transparency(self, image_cache):
        """Test that images with transparency are saved as PNG."""
        # Create an RGBA image with transparency
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        _, media_type, _, _ = image_cache._process_image(image_data, "image/png")

        assert media_type == "image/png"

    def test_process_image_rgb_becomes_jpeg(self, image_cache):
        """Test that RGB images are saved as JPEG."""
        img = Image.new("RGB", (50, 50), color="red")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()

        _, media_type, _, _ = image_cache._process_image(image_data, "image/jpeg")

        assert media_type == "image/jpeg"
