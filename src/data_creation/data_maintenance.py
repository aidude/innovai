"""
Module for metadata tracking and data maintenance.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import shutil
from pathlib import Path
import logging

class MetadataTracker:
    """Class for tracking and managing LLM interaction metadata."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize MetadataTracker.
        
        Args:
            base_dir (Path): Base directory for the project
        """
        self.base_dir = base_dir
        self.metadata_file = base_dir / "synthetic_data" / "metadata.json"
        self.logger = logging.getLogger("MetadataTracker")
        
        # Initialize metadata storage
        if not self.metadata_file.exists():
            self._init_metadata_file()
    
    def _init_metadata_file(self):
        """Initialize the metadata file with default structure."""
        initial_metadata = {
            "summary": {
                "total_requests": 0,
                "total_tokens": 0,
                "providers": {}
            },
            "entries": {}
        }
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(initial_metadata, f, indent=2)
    
    def update_metadata(
        self,
        data_id: str,
        provider: str,
        tokens: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update metadata for a specific interaction.
        
        Args:
            data_id (str): The unique ID of the stored data
            provider (str): The LLM provider name
            tokens (Dict[str, int], optional): Token usage information
            cost (float, optional): Cost of the API call
            metadata (Dict[str, Any], optional): Additional metadata
        """
        try:
            with open(self.metadata_file, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                
                # Update provider statistics
                if provider not in data["summary"]["providers"]:
                    data["summary"]["providers"][provider] = {
                        "total_requests": 0,
                        "total_tokens": 0,
                        "total_cost": 0.0
                    }
                
                provider_stats = data["summary"]["providers"][provider]
                provider_stats["total_requests"] += 1
                if tokens:
                    provider_stats["total_tokens"] += sum(tokens.values())
                if cost:
                    provider_stats["total_cost"] += cost
                
                # Update total statistics
                data["summary"]["total_requests"] += 1
                if tokens:
                    data["summary"]["total_tokens"] += sum(tokens.values())
                
                # Add entry
                data["entries"][data_id] = {
                    "provider": provider,
                    "timestamp": datetime.now().isoformat(),
                    "tokens": tokens,
                    "cost": cost,
                    "metadata": metadata or {}
                }
                
                # Write updated data
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                
                self.logger.info(f"Updated metadata for interaction {data_id}")
        except Exception as e:
            self.logger.error(f"Failed to update metadata: {str(e)}")

    def get_provider_stats(self, provider: str) -> Dict[str, Any]:
        """
        Get statistics for a specific provider.
        
        Args:
            provider (str): The provider name
            
        Returns:
            Dict[str, Any]: Provider statistics
        """
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data["summary"]["providers"].get(provider, {})
        except Exception as e:
            self.logger.error(f"Failed to get provider stats: {str(e)}")
            return {}

class DataMaintenance:
    """Class for managing synthetic data maintenance and cleanup."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize DataMaintenance.
        
        Args:
            base_dir (Path): Base directory for the project
        """
        self.base_dir = base_dir
        self.synthetic_data_dir = base_dir / "synthetic_data"
        self.archive_dir = base_dir / "synthetic_data" / "archive"
        self.logger = logging.getLogger("DataMaintenance")
    
    def archive_old_data(self, days_threshold: int = 30):
        """
        Archive data older than the specified threshold.
        
        Args:
            days_threshold (int): Age threshold in days
        """
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        for provider_dir in self.synthetic_data_dir.glob("*"):
            if not provider_dir.is_dir() or provider_dir.name == "archive":
                continue
            
            archive_provider_dir = self.archive_dir / provider_dir.name
            archive_provider_dir.mkdir(exist_ok=True)
            
            for file_path in provider_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        
                        if timestamp < threshold_date:
                            archive_path = archive_provider_dir / file_path.name
                            shutil.move(str(file_path), str(archive_path))
                            self.logger.info(f"Archived {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {str(e)}")
    
    def cleanup_archived_data(self, days_threshold: int = 90):
        """
        Permanently delete archived data older than the specified threshold.
        
        Args:
            days_threshold (int): Age threshold in days
        """
        if not self.archive_dir.exists():
            return
        
        threshold_date = datetime.now() - timedelta(days=days_threshold)
        
        for provider_dir in self.archive_dir.glob("*"):
            if not provider_dir.is_dir():
                continue
            
            for file_path in provider_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        
                        if timestamp < threshold_date:
                            file_path.unlink()
                            self.logger.info(f"Deleted archived file {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to process archived file {file_path}: {str(e)}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for synthetic data.
        
        Returns:
            Dict[str, Any]: Storage statistics
        """
        stats = {
            "active_data": {},
            "archived_data": {},
            "total_files": 0,
            "total_size_mb": 0
        }
        
        # Active data stats
        for provider_dir in self.synthetic_data_dir.glob("*"):
            if not provider_dir.is_dir() or provider_dir.name == "archive":
                continue
            
            files = list(provider_dir.glob("*.json"))
            size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # Convert to MB
            stats["active_data"][provider_dir.name] = {
                "files": len(files),
                "size_mb": round(size, 2)
            }
            stats["total_files"] += len(files)
            stats["total_size_mb"] += size
        
        # Archived data stats
        if self.archive_dir.exists():
            for provider_dir in self.archive_dir.glob("*"):
                if not provider_dir.is_dir():
                    continue
                
                files = list(provider_dir.glob("*.json"))
                size = sum(f.stat().st_size for f in files) / (1024 * 1024)
                stats["archived_data"][provider_dir.name] = {
                    "files": len(files),
                    "size_mb": round(size, 2)
                }
                stats["total_files"] += len(files)
                stats["total_size_mb"] += size
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        return stats
