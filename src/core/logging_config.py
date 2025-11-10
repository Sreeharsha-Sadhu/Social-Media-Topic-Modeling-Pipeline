"""
logging_config.py
----------------------------------
Sets up unified logging for the MMDS project.
Use get_logger(__name__) to get a logger instance per module.
"""

import logging
import sys
from src.core import config


def setup_logging(log_file: str = "mmds.log") -> None:
    """Configure root logger for the project."""
    log_path = config.get_log_file_path(log_file)
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Returns a module-specific logger."""
    return logging.getLogger(name)
