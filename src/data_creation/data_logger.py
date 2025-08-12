"""
Module for handling data storage and logging of LLM interactions.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid
from .config import config
from .logging_utils import setup_logger
from .data_maintenance import MetadataTracker, DataMaintenance
from .error_handling import LLMError


class DataLogger:
    """Class for handling data storage and logging of LLM interactions."""
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize DataLogger.
        
        Args:
            base_dir: Optional custom base directory for storing data and logs
        """
        self.base_dir = Path(base_dir) if base_dir else config.storage.base_dir
        self.synthetic_data_dir = self.base_dir / config.storage.synthetic_data_dir
        self.logs_dir = self.base_dir / config.storage.logs_dir
        
        # Initialize metadata and maintenance handlers
        self.metadata_tracker = MetadataTracker(self.base_dir)
        self.maintenance = DataMaintenance(self.base_dir)
        
        # Create directories if they don't exist
        self.synthetic_data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logger
        self.logger = setup_logger(
            "LLMDataLogger",
            log_file="llm_interactions.log"
        )
    
    def store_response(
        self,
        response: str,
        provider: str,
        prompt: str,
        tokens: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Store LLM response and log the interaction.
        
        Args:
            response (str): The response from the LLM
            provider (str): The LLM provider name
            prompt (str): The original prompt
            tokens (Dict[str, int], optional): Token usage information
            cost (float, optional): Cost of the API call
            metadata (Dict[str, Any], optional): Additional metadata about the interaction
            model (str, optional): The specific model used for generation
            
        Returns:
            str: The unique ID assigned to this data
        """
        # Generate unique ID for the data
        data_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Extract token usage and cost from metadata if available
        metadata = metadata or {}
        token_usage = metadata.pop("token_usage", None)
        cost = metadata.pop("cost", None)
        
        # Ensure metadata is a dictionary
        metadata = metadata or {}
        
        # Add model to metadata if provided
        if model:
            metadata["model"] = model
        
        # Prepare data structure
        data = {
            "id": data_id,
            "timestamp": timestamp,
            "provider": provider,
            "prompt": prompt,
            "response": response,
            "token_usage": token_usage,
            "cost": cost,
            "metadata": metadata
        }
        
        # Update metadata tracking
        self.metadata_tracker.update_metadata(
            data_id=data_id,
            provider=provider,
            tokens=token_usage,
            cost=cost,
            metadata=metadata
        )
        
        # Create provider-specific directory
        provider_dir = self.synthetic_data_dir / provider
        provider_dir.mkdir(exist_ok=True)
        
        # Save data to JSON file
        file_path = provider_dir / f"{data_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Log the interaction
        self.logger.info(
            f"Stored response from {provider} | ID: {data_id} | "
            f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
        )
        
        return data_id
    
    def get_response(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored response by ID.
        
        Args:
            data_id (str): The unique ID of the stored data
            
        Returns:
            Optional[Dict[str, Any]]: The stored data if found, None otherwise
        """
        # Search in all provider directories
        for provider_dir in self.synthetic_data_dir.iterdir():
            if provider_dir.is_dir():
                file_path = provider_dir / f"{data_id}.json"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        
        self.logger.warning(f"Data with ID {data_id} not found")
        return None
    
    def get_responses_by_provider(self, provider: str) -> list[Dict[str, Any]]:
        """
        Retrieve all stored responses from a specific provider.
        
        Args:
            provider (str): The provider name
            
        Returns:
            list[Dict[str, Any]]: List of stored responses
        """
        provider_dir = self.synthetic_data_dir / provider
        if not provider_dir.exists():
            self.logger.warning(f"No data found for provider {provider}")
            return []
        
        responses = []
        for file_path in provider_dir.glob("*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                responses.append(json.load(f))
        
        return responses
