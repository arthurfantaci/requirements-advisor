"""
Configuration management using pydantic-settings.

Loads from environment variables and .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Voyage AI
    voyage_api_key: str = ""
    voyage_model: str = "voyage-context-3"  # Contextualized embeddings for better RAG
    voyage_batch_size: int = 50

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
