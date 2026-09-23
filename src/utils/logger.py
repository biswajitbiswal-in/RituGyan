"""Structured logging utilities for RituGyan."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "ritugyan", log_level: Optional[str] = None, log_file: Optional[Path | str] = None) -> logging.Logger:
    """Configure and return a structured logger for RituGyan modules.
    
    Args:
        name: Logger name, usually __name__ or module identifier.
        log_level: Optional log level override (e.g. 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional file path to append log output.
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        level_str = log_level or "INFO"
        level = getattr(logging, level_str.upper(), logging.INFO)
        logger.setLevel(level)

        # Standard console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formatting pattern
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger
