"""
Retry decorators and error handling utilities.
"""
import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Union
from .exceptions import DataFetchingError, RateLimitError

logger = logging.getLogger(__name__)

def with_retry(
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (DataFetchingError,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Decorator that implements retry logic with exponential backoff.

    Args:
        retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff: Multiplicative factor for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback function to execute on each retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        raise

                    if isinstance(e, RateLimitError):
                        # For rate limit errors, use a longer delay
                        current_delay = max(current_delay, 5.0)

                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == retries:
                        raise

                    if isinstance(e, RateLimitError):
                        # For rate limit errors, use a longer delay
                        current_delay = max(current_delay, 5.0)

                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def handle_provider_errors(error_map: Optional[dict] = None) -> Callable:
    """
    Decorator that maps provider-specific exceptions to our custom exceptions.

    Args:
        error_map: Optional dictionary mapping provider exceptions to custom exceptions

    Returns:
        Decorated function with error mapping
    """
    if error_map is None:
        error_map = {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_type in error_map:
                    raise error_map[error_type](str(e))
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_type in error_map:
                    raise error_map[error_type](str(e))
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
