"""Configuration management using pydantic-settings.

Loads from environment variables and .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        voyage_api_key: Voyage AI API key for generating embeddings.
        voyage_model: Embedding model name (default: voyage-context-3).
        voyage_batch_size: Number of texts to embed per API call.
        vector_store_type: Backend type, either "chroma" or "qdrant".
        vector_store_path: Local directory for ChromaDB persistence.
        collection_name: Name of the vector collection.
        qdrant_url: Qdrant server URL (for remote deployments).
        qdrant_api_key: Qdrant API key (for authenticated connections).
        content_dir: Directory containing JSONL content files.
        image_cache_path: Directory for cached images.
        image_max_dimension: Maximum image dimension in pixels.
        image_quality: JPEG compression quality (1-100).
        image_fetch_timeout: HTTP timeout for image fetching in seconds.
        host: Server bind address.
        port: Server bind port.
        debug: Enable debug mode.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_json: Output logs in JSON format for production.

    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Voyage AI
    voyage_api_key: str = ""
    voyage_model: str = "voyage-context-3"  # Contextualized embeddings for better RAG
    voyage_batch_size: int = 20  # Reduced for voyage-context-3 token limits

    # Vector Store
    vector_store_type: str = "chroma"  # chroma | qdrant
    vector_store_path: str = "./data/chroma"
    collection_name: str = "requirements_guidance"

    # Qdrant (for future remote deployment)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # Content paths
    content_dir: str = "./content"

    # Image caching
    image_cache_path: str = "./data/images"
    image_max_dimension: int = 1024
    image_quality: int = 85
    image_fetch_timeout: int = 30

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def content_path(self) -> Path:
        """Return the content directory as a Path object.

        Returns:
            Path: Resolved path to the content directory.

        """
        return Path(self.content_dir)

    @property
    def vector_path(self) -> Path:
        """Return the vector store directory as a Path object.

        Returns:
            Path: Resolved path to the vector store directory.

        """
        return Path(self.vector_store_path)

    @property
    def image_path(self) -> Path:
        """Return the image cache directory as a Path object.

        Returns:
            Path: Resolved path to the image cache directory.

        """
        return Path(self.image_cache_path)


# Global settings instance
settings = Settings()
