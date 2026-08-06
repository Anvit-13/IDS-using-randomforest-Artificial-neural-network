"""
Logging Module for IDS GAN Feature Extraction Pipeline.

Provides thread-safe logger configuration with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "IDS_GAN_Pipeline",
    log_dir: Path = Path("logs"),
    log_file: str = "pipeline.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logger instance.

    Args:
        name: Name of the logger.
        log_dir: Directory where log files will be saved.
        log_file: Log filename.
        level: Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / log_file

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
