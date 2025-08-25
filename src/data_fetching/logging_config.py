"""
Logging configuration for the data fetching module.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    module_name: str = "data_fetching"
) -> logging.Logger:
    """
    Configure logging for the data fetching module.

    Args:
        log_dir: Directory to store log files. If None, logs only to console
        log_level: The logging level to use
        module_name: Name of the module for the logger

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File handlers if log_dir is provided
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Regular log file
        log_file = log_dir_path / f"{module_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

        # Rotating file handler for detailed debug logs
        debug_log = log_dir_path / f"{module_name}_debug.log"
        rotating_handler = logging.handlers.RotatingFileHandler(
            debug_log,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        rotating_handler.setFormatter(detailed_formatter)
        rotating_handler.setLevel(logging.DEBUG)
        logger.addHandler(rotating_handler)

        # Error log file
        error_log = log_dir_path / f"{module_name}_error.log"
        error_handler = logging.FileHandler(error_log)
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter that adds context information to log messages.
    """
    def __init__(self, logger: logging.Logger, extra: Optional[dict] = None):
        """
        Initialize the adapter with a logger and optional extra context.

        Args:
            logger: The logger instance to adapt
            extra: Optional dictionary of extra contextual information
        """
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: dict) -> tuple:
        """
        Process the log message and keywords.

        Args:
            msg: The message to log
            kwargs: The keywords to pass to the logger

        Returns:
            Tuple of (modified message, modified keywords)
        """
        extra = kwargs.get('extra', {})
        if self.extra:
            extra.update(self.extra)
        
        # Add request_id if available
        if 'request_id' in extra:
            msg = f"[{extra['request_id']}] {msg}"
            
        # Add operation info if available
        if 'operation' in extra:
            msg = f"[{extra['operation']}] {msg}"
            
        kwargs['extra'] = extra
        return msg, kwargs


def get_logger(
    name: str,
    extra: Optional[dict] = None,
    log_dir: Optional[str] = None
) -> LoggerAdapter:
    """
    Get a configured logger with optional context information.

    Args:
        name: Name for the logger
        extra: Optional dictionary of extra contextual information
        log_dir: Optional directory for log files

    Returns:
        Configured LoggerAdapter instance
    """
    logger = setup_logging(log_dir=log_dir, module_name=name)
    return LoggerAdapter(logger, extra)
