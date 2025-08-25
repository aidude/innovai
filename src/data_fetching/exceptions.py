"""
Custom exceptions for the data fetching module.
"""

class DataFetchingError(Exception):
    """Base exception for data fetching errors."""
    pass


class ProviderError(DataFetchingError):
    """Exception raised when a provider fails to fetch data."""
    pass


class ValidationError(DataFetchingError):
    """Exception raised when data validation fails."""
    pass


class RateLimitError(ProviderError):
    """Exception raised when hitting rate limits."""
    pass


class AuthenticationError(ProviderError):
    """Exception raised when authentication fails."""
    pass


class ConnectionError(ProviderError):
    """Exception raised when connection to provider fails."""
    pass
