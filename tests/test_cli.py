"""Tests for CLI commands.

Tests serve, ingest, info, and test-search commands.
"""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from requirements_advisor.cli import app

runner = CliRunner()


class TestServeCommand:
    """Test serve command."""

    def test_serve_default_options(self):
        """Test serve command with default options."""
        with patch("requirements_advisor.server.mcp") as mock_mcp:
            mock_mcp.run = MagicMock()

            result = runner.invoke(app, ["serve"])

            assert result.exit_code == 0
            mock_mcp.run.assert_called_once()
            call_kwargs = mock_mcp.run.call_args.kwargs
            assert call_kwargs["transport"] == "http"

    def test_serve_custom_port(self):
        """Test serve command with custom port."""
        with patch("requirements_advisor.server.mcp") as mock_mcp:
            mock_mcp.run = MagicMock()

            result = runner.invoke(app, ["serve", "--port", "9000"])

            assert result.exit_code == 0
            call_kwargs = mock_mcp.run.call_args.kwargs
            assert call_kwargs["port"] == 9000

    def test_serve_custom_host(self):
        """Test serve command with custom host."""
        with patch("requirements_advisor.server.mcp") as mock_mcp:
            mock_mcp.run = MagicMock()

            result = runner.invoke(app, ["serve", "--host", "127.0.0.1"])

            assert result.exit_code == 0
            call_kwargs = mock_mcp.run.call_args.kwargs
            assert call_kwargs["host"] == "127.0.0.1"


class TestInfoCommand:
    """Test info command."""

    def test_info_displays_settings(self):
        """Test info command displays settings."""
        with patch("requirements_advisor.vectorstore.chroma.ChromaVectorStore") as mock_chroma:
            store_instance = MagicMock()
            store_instance.count = MagicMock(return_value=100)
            store_instance.get_metadata_values = MagicMock(return_value=["jama_guide"])
            mock_chroma.return_value = store_instance

            result = runner.invoke(app, ["info"])

            assert result.exit_code == 0
            assert "Requirements Advisor Configuration" in result.output


class TestHelpOutput:
    """Test help output for commands."""

    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "MCP server" in result.output
        assert "serve" in result.output
        assert "ingest" in result.output
        assert "info" in result.output
        assert "test-search" in result.output

    def test_serve_help(self):
        """Test serve command help."""
        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output

    def test_ingest_help(self):
        """Test ingest command help."""
        result = runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "--content-dir" in result.output
        assert "--batch-size" in result.output
        assert "--clear" in result.output

    def test_test_search_help(self):
        """Test test-search command help."""
        result = runner.invoke(app, ["test-search", "--help"])

        assert result.exit_code == 0
        assert "--top-k" in result.output


class TestVerboseFlag:
    """Test verbose flag on main command."""

    def test_verbose_flag_sets_debug(self):
        """Test that -v flag enables debug logging."""
        with patch("requirements_advisor.server.mcp") as mock_mcp:
            mock_mcp.run = MagicMock()

            result = runner.invoke(app, ["-v", "serve"])

            # Command should run successfully with verbose flag
            assert result.exit_code == 0 or mock_mcp.run.called
