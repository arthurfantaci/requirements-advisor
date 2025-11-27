"""
Tests for configuration module.

Tests Settings class, environment variable loading, and property methods.
"""

import os
from pathlib import Path

import pytest

from requirements_advisor.config import Settings


class TestSettings:
    """Test Settings class."""

    def test_default_values(self, monkeypatch):
        """Test that default values are set correctly."""
        # Clear env vars that might override defaults from .env file
        monkeypatch.delenv("VOYAGE_MODEL", raising=False)
        monkeypatch.delenv("VOYAGE_BATCH_SIZE", raising=False)

        settings = Settings(voyage_api_key="test-key")

        assert settings.voyage_model == "voyage-context-3"
        assert settings.voyage_batch_size == 20
        assert settings.vector_store_type == "chroma"
        assert settings.collection_name == "requirements_guidance"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.log_level == "INFO"
        assert settings.log_json is False

    def test_content_path_property(self):
        """Test content_path property returns Path object."""
        settings = Settings(voyage_api_key="test-key", content_dir="./my/content")

        assert isinstance(settings.content_path, Path)
        assert str(settings.content_path) == "my/content"

    def test_vector_path_property(self):
        """Test vector_path property returns Path object."""
        settings = Settings(voyage_api_key="test-key", vector_store_path="./data/vectors")

        assert isinstance(settings.vector_path, Path)
        assert str(settings.vector_path) == "data/vectors"

    def test_image_path_property(self):
        """Test image_path property returns Path object."""
        settings = Settings(voyage_api_key="test-key", image_cache_path="./cache/images")

        assert isinstance(settings.image_path, Path)
        assert str(settings.image_path) == "cache/images"

    def test_image_settings_defaults(self):
        """Test image-related settings have correct defaults."""
        settings = Settings(voyage_api_key="test-key")

        assert settings.image_max_dimension == 1024
        assert settings.image_quality == 85
        assert settings.image_fetch_timeout == 30

    def test_logging_settings(self):
        """Test logging settings."""
        settings = Settings(voyage_api_key="test-key", log_level="DEBUG", log_json=True)

        assert settings.log_level == "DEBUG"
        assert settings.log_json is True

    def test_empty_api_key_allowed(self):
        """Test that empty API key is allowed (will fail at runtime)."""
        # Explicitly pass empty API key to override any env/file defaults
        settings = Settings(voyage_api_key="")

        assert settings.voyage_api_key == ""

    def test_qdrant_settings_optional(self):
        """Test that Qdrant settings are optional (None by default)."""
        settings = Settings(voyage_api_key="test-key")

        assert settings.qdrant_url is None
        assert settings.qdrant_api_key is None


class TestSettingsFromEnv:
    """Test Settings loading from environment variables."""

    def test_load_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("VOYAGE_API_KEY", "env-api-key")
        monkeypatch.setenv("VOYAGE_MODEL", "voyage-3-lite")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        settings = Settings()

        assert settings.voyage_api_key == "env-api-key"
        assert settings.voyage_model == "voyage-3-lite"
        assert settings.port == 9000
        assert settings.log_level == "WARNING"

    def test_env_overrides_defaults(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("VOYAGE_API_KEY", "test")
        monkeypatch.setenv("VECTOR_STORE_TYPE", "qdrant")
        monkeypatch.setenv("DEBUG", "true")

        settings = Settings()

        assert settings.vector_store_type == "qdrant"
        assert settings.debug is True
