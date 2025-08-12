"""
Module for rate limiting and error handling utilities.
"""
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limit information for an API."""
    calls: int = 0
    reset_time: datetime = datetime.now()
    
class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_hour: int = 3000):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute (int): Maximum calls allowed per minute
            calls_per_hour (int): Maximum calls allowed per hour
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.minute_usage: Dict[str, RateLimitInfo] = {}
        self.hour_usage: Dict[str, RateLimitInfo] = {}
        self.lock = Lock()
    
    def _cleanup_old_entries(self, usage_dict: Dict[str, RateLimitInfo], window: timedelta):
        """Remove expired rate limit entries."""
        current_time = datetime.now()
        expired_keys = [
            key for key, info in usage_dict.items()
            if current_time - info.reset_time > window
        ]
        for key in expired_keys:
            del usage_dict[key]
    
    def check_rate_limit(self, api_key: str) -> Optional[float]:
        """
        Check if the rate limit has been exceeded.
        
        Args:
            api_key (str): The API key being used
            
        Returns:
            Optional[float]: Seconds to wait if rate limited, None if not limited
        """
        with self.lock:
            current_time = datetime.now()
            
            # Clean up old entries
            self._cleanup_old_entries(self.minute_usage, timedelta(minutes=1))
            self._cleanup_old_entries(self.hour_usage, timedelta(hours=1))
            
            # Check minute limit
            if api_key not in self.minute_usage:
                self.minute_usage[api_key] = RateLimitInfo(calls=0, reset_time=current_time)
            minute_info = self.minute_usage[api_key]
            
            # Check hour limit
            if api_key not in self.hour_usage:
                self.hour_usage[api_key] = RateLimitInfo(calls=0, reset_time=current_time)
            hour_info = self.hour_usage[api_key]
            
            # Reset counters if time window has passed
            if current_time - minute_info.reset_time > timedelta(minutes=1):
                minute_info.calls = 0
                minute_info.reset_time = current_time
            
            if current_time - hour_info.reset_time > timedelta(hours=1):
                hour_info.calls = 0
                hour_info.reset_time = current_time
            
            # Check if limits are exceeded
            if minute_info.calls >= self.calls_per_minute:
                wait_time = 60 - (current_time - minute_info.reset_time).total_seconds()
                return max(0, wait_time)
            
            if hour_info.calls >= self.calls_per_hour:
                wait_time = 3600 - (current_time - hour_info.reset_time).total_seconds()
                return max(0, wait_time)
            
            # Increment counters
            minute_info.calls += 1
            hour_info.calls += 1
            return None
    
    def log_rate_limit_headers(self, headers: Dict[str, str], api_key: str):
        """
        Update rate limit information from API response headers.
        
        Args:
            headers (Dict[str, str]): Response headers
            api_key (str): The API key being used
        """
        try:
            remaining = headers.get('x-ratelimit-remaining-requests')
            reset = headers.get('x-ratelimit-reset-requests')
            
            if remaining and reset:
                logger.info(
                    f"Rate limit status - Remaining: {remaining}, "
                    f"Reset in: {reset} seconds"
                )
        except Exception as e:
            logger.warning(f"Failed to parse rate limit headers: {str(e)}")


class APIError(Exception):
    """Base class for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, raw_error: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.raw_error = raw_error


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    def __init__(self, wait_time: float):
        super().__init__(f"Rate limit exceeded. Please wait {wait_time:.1f} seconds.")
        self.wait_time = wait_time


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class ModelNotFoundError(APIError):
    """Raised when the requested model is not found."""
    pass


class InvalidRequestError(APIError):
    """Raised when the request is invalid."""
    pass


class APIConnectionError(APIError):
    """Raised when connection to the API fails."""
    pass
