"""
Centralized logging configuration for the data creation package.
"""
import logging
from pathlib import Path
from typing import Optional
from .config import config


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: Optional[str] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.
    
    Args:
        name: The name of the logger
        log_file: Optional specific log file path
        level: Optional log level override
        format_str: Optional format string override
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Use config values as defaults
    level = level or config.logging.level
    format_str = format_str or config.logging.format
    
    # Set basic configuration
    logger.setLevel(level)
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_path = config.logging.log_dir / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
default_logger = setup_logger(
    "data_creation",
    log_file=config.logging.log_file
)
