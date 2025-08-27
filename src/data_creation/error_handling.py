"""
Unified error handling for the data creation package.
"""
from typing import Optional, Type, TypeVar
from functools import wraps
import time
from logging_utils import default_logger
from config import config

# Generic type for the exception
E = TypeVar('E', bound=Exception)


class LLMError(Exception):
    """Base exception for all LLM-related errors."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class APIError(LLMError):
    """Exception for API-related errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, raw_error: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.raw_error = raw_error


class RateLimitError(APIError):
    """Exception for rate limit errors."""
    def __init__(self, retry_after: int):
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Exception for authentication errors."""
    pass


class ModelNotFoundError(APIError):
    """Exception for model not found errors."""
    pass


class InvalidRequestError(APIError):
    """Exception for invalid request errors."""
    pass


class APIConnectionError(APIError):
    """Exception for API connection errors."""
    pass


def retry_on_error(
    max_retries: Optional[int] = None,
    delay: Optional[int] = None,
    exceptions: tuple[Type[E], ...] = (Exception,),
    logger=default_logger
):
    """
    Decorator for retrying operations that may fail.
    
    Args:
        max_retries: Maximum number of retry attempts (default from config)
        delay: Delay between retries in seconds (default from config)
        exceptions: Tuple of exceptions to catch
        logger: Logger instance to use
    """
    max_retries = max_retries or config.rate_limit.retry_attempts
    delay = delay or config.rate_limit.retry_delay
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                    )
                    
                    # Don't retry on authentication or invalid request errors
                    if isinstance(e, (AuthenticationError, InvalidRequestError)):
                        raise
                    
                    # Handle rate limit errors specially
                    if isinstance(e, RateLimitError):
                        time.sleep(e.retry_after)
                    elif attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
            
            # If we get here, all retries failed
            raise LLMError(
                f"All {max_retries} retry attempts failed",
                original_error=last_error
            )
        
        return wrapper
    
    return decorator
