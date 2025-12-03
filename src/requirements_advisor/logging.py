"""Centralized logging configuration using loguru.

This module provides a unified logging setup for the requirements-advisor application.
It configures loguru with appropriate formatters for both development and production use.

Example:
    from requirements_advisor.logging import setup_logging

    # Initialize logging at application startup
    setup_logging(level="DEBUG")

    # Then use loguru's logger in any module
    from loguru import logger
    logger.info("Application started")

"""

import sys
from typing import Any

from loguru import logger


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> Any:
    """Configure loguru for the application.

    Sets up loguru with appropriate handlers based on the environment.
    Should be called once at application startup.

    Args:
        level: Minimum log level to capture. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        json_output: If True, output logs in JSON format for production/monitoring systems.
        log_file: Optional file path to write logs to. If None, logs only to stderr.

    Returns:
        The configured loguru logger instance.

    Example:
        # Development setup with debug logging
        setup_logging(level="DEBUG")

        # Production setup with JSON output
        setup_logging(level="INFO", json_output=True)

        # Log to file
        setup_logging(level="INFO", log_file="/var/log/requirements-advisor.log")

    """
    # Remove default handler
    logger.remove()

    # Define format for human-readable output
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    if json_output:
        # JSON format for production/monitoring
        logger.add(
            sys.stderr,
            format="{message}",
            serialize=True,
            level=level,
        )
    else:
        # Human-readable format for development
        logger.add(
            sys.stderr,
            format=console_format,
            level=level,
            colorize=True,
        )

    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            format=console_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    return logger


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance, optionally bound to a specific name.

    This is a convenience function for getting a contextualized logger.
    The name is typically the module name.

    Args:
        name: Optional name to bind to the logger context.

    Returns:
        A loguru logger instance, optionally with name context.

    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")

    """
    if name:
        return logger.bind(name=name)
    return logger
