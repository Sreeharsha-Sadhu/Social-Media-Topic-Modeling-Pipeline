"""
logging_config.py
-------------------------------------------------
Centralized logging configuration for the MMDS Project.

Provides:
- Unified console + rotating file logging
- Logger factory function (get_logger)
- Safe, idempotent initialization (runs once)
- Integration with config.get_log_file_path()

All modules should import loggers using:

    from src.core.logging_config import get_logger
    logger = get_logger(__name__)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.core.config import get_log_file_path


# Internal flag to ensure logging is only configured once
_LOGGING_INITIALIZED = False


def init_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
    backup_count: int = 3
) -> None:
    """
    Initialize global logging configuration (idempotent).

    Parameters
    ----------
    log_level : int
        Logging level (default: INFO)
    log_file : Path or None
        Path to log file. If None, falls back to config.get_log_file_path()
    max_bytes : int
        Max size of a single rotating log file
    backup_count : int
        Number of backup log files to keep
    """

    global _LOGGING_INITIALIZED

    if _LOGGING_INITIALIZED:
        return  # Avoid double initializations

    if log_file is None:
        log_file = get_log_file_path()

    log_file.parent.mkdir(exist_ok=True, parents=True)

    # Formatters
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Root Logger Setup
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger instance tied to the global configuration.
    Ensures logging is initialized before creating the logger.

    Parameters
    ----------
    name : str
        Logger name (usually `__name__`)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    init_logging()  # Safe to call multiple times
    return logging.getLogger(name)
