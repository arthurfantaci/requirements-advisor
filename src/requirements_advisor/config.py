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
    voyage_model: str = "voyage-3"
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
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    @property
    def content_path(self) -> Path:
        return Path(self.content_dir)
    
    @property
    def vector_path(self) -> Path:
        return Path(self.vector_store_path)


# Global settings instance
settings = Settings()
