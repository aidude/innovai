"""
Provider module for handling different data sources.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class DataProvider(ABC):
    """
    Abstract base class for data providers.
    All specific providers should inherit from this class.
    """

    @abstractmethod
    async def fetch(self, query: str, **kwargs) -> List[Dict]:
        """
        Fetch data from the provider.

        Args:
            query: The query or prompt to use for fetching data
            **kwargs: Additional provider-specific arguments

        Returns:
            List of dictionaries containing the fetched data
        """
        pass

    @abstractmethod
    def validate_response(self, response: Dict) -> bool:
        """
        Validate the response from the provider.

        Args:
            response: The response to validate

        Returns:
            True if valid, False otherwise
        """
        pass
