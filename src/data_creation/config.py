"""
Configuration management for the data creation package.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os


@dataclass
class APIConfig:
    """Configuration for API providers."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    calls_per_minute: int = 60
    calls_per_hour: int = 3000
    retry_attempts: int = 3
    retry_delay: int = 2


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: Path = Path("logs")
    log_file: str = "llm_operations.log"


@dataclass
class StorageConfig:
    """Data storage configuration."""
    base_dir: Path = Path(__file__).parent.parent.parent
    synthetic_data_dir: Path = Path("synthetic_data")
    logs_dir: Path = Path("logs")
    metadata_file: str = "metadata.json"
    

@dataclass
class Config:
    """Global configuration."""
    api: APIConfig = None
    rate_limit: RateLimitConfig = None
    logging: LoggingConfig = None
    storage: StorageConfig = None
    
    def __post_init__(self):
        # Initialize default configurations
        self.api = self.api or APIConfig()
        self.rate_limit = self.rate_limit or RateLimitConfig()
        self.logging = self.logging or LoggingConfig()
        self.storage = self.storage or StorageConfig()
        
        # Load API keys from environment
        self.api.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.api.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")


# Global configuration instance
config = Config()
