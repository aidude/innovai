"""
Module for handling interactions with various LLM APIs (OpenAI, Anthropic Claude, OpenRouter).
"""
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
import requests

# Set up logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Create a formatter that includes timestamp and log level
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create API debug logger
api_logger = logging.getLogger('api_debug')
api_logger.setLevel(logging.DEBUG)

# Create file handler for API debug logs with date in filename
debug_log_file = log_dir / f'api_debug_{datetime.now().strftime("%Y%m%d")}.log'
debug_handler = logging.FileHandler(debug_log_file)
debug_handler.setFormatter(formatter)
debug_handler.setLevel(logging.DEBUG)
api_logger.addHandler(debug_handler)
from .rate_limiter import (
    RateLimiter, APIError, RateLimitError, AuthenticationError,
    ModelNotFoundError, InvalidRequestError, APIConnectionError
)

# Load environment variables
load_dotenv()

class LLMClient(ABC):
    """
    Abstract base class for LLM API clients.
    """
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM API.
        
        Args:
            prompt (str): The input prompt for text generation
            **kwargs: Additional arguments specific to each API
            
        Returns:
            str: Generated text response
        """
        pass


class OpenAIClient(LLMClient):
    """
    Client for interacting with OpenAI's API.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize OpenAI client.
        
        Args:
            model (str): The model to use for generation
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1"  # Explicitly set the base URL
        )
        self.model = model
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI's API.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments for the API call
            
        Returns:
            str: Generated text response
        """
        try:
            # Log request details
            request_data = {
                "provider": "openai",
                "model": self.model,
                "base_url": str(self.client.base_url),  # Convert URL to string
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "kwargs": {k: v for k, v in kwargs.items() if k not in ['api_key']},
                "timestamp": datetime.now().isoformat()
            }
            api_logger.debug(f"OpenAI Request: {json.dumps(request_data, indent=2)}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            # Get token usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Calculate cost (approximate)
            cost = self._calculate_cost(usage)
            
            # Add usage and cost to kwargs for logging
            kwargs["token_usage"] = usage
            kwargs["cost"] = cost
            
            # Log response details
            response_data = {
                "provider": "openai",
                "model": self.model,
                "status": "success",
                "usage": usage,
                "cost": cost,
                "timestamp": datetime.now().isoformat()
            }
            api_logger.debug(f"OpenAI Response: {json.dumps(response_data, indent=2)}")
            
            return response.choices[0].message.content
        except Exception as e:
            # Log error details
            error_data = {
                "provider": "openai",
                "model": self.model,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            api_logger.error(f"OpenAI Error: {json.dumps(error_data, indent=2)}")
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """
        Calculate the cost of the API call based on token usage.
        
        Args:
            usage (Dict[str, int]): Token usage information
            
        Returns:
            float: Estimated cost in USD
        """
        # Cost per 1K tokens (approximate, may need updating)
        costs = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
            "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
        }
        
        model_costs = costs.get(self.model, costs["gpt-4"])
        
        prompt_cost = (usage["prompt_tokens"] / 1000) * model_costs["prompt"]
        completion_cost = (usage["completion_tokens"] / 1000) * model_costs["completion"]
        
        return prompt_cost + completion_cost


class ClaudeClient(LLMClient):
    """
    Client for interacting with Anthropic's Claude API.
    """
    
    def __init__(self, model: str = "claude-3-opus-20240229"):
        """
        Initialize Claude client.
        
        Args:
            model (str): The model to use for generation
        """
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Claude's API.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments for the API call
            
        Returns:
            str: Generated text response
        """
        try:
            # Log request details
            request_data = {
                "provider": "claude",
                "model": self.model,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "kwargs": {k: v for k, v in kwargs.items() if k not in ['api_key']},
                "timestamp": datetime.now().isoformat()
            }
            api_logger.debug(f"Claude Request: {json.dumps(request_data, indent=2)}")
            
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            # Log response details
            response_data = {
                "provider": "claude",
                "model": self.model,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            api_logger.debug(f"Claude Response: {json.dumps(response_data, indent=2)}")
            
            return response.content[0].text
        except Exception as e:
            # Log error details
            error_data = {
                "provider": "claude",
                "model": self.model,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            api_logger.error(f"Claude Error: {json.dumps(error_data, indent=2)}")
            raise Exception(f"Claude API error: {str(e)}")


class OpenRouterClient(LLMClient):
    """
    Client for interacting with OpenRouter's API.
    """
    
    # Class-level rate limiter shared across all instances
    _rate_limiter = RateLimiter(calls_per_minute=50, calls_per_hour=3000)
    
    def __init__(self, model: str = "liquid/lfm-7b"):
        """
        Initialize OpenRouter client.
        
        Args:
            model (str): The model to use for generation
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise AuthenticationError("OPENROUTER_API_KEY not found in environment variables")
        
        self.api_base = "https://openrouter.ai/api/v1"
        self.model = model
        
        # Headers that identify your application
        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/innovai",
            "X-Title": "InnovAI Data Generation"
        }
        
        # Maximum retries for recoverable errors
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenRouter's API.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments for the API call
            
        Returns:
            str: Generated text response
            
        Raises:
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If API key is invalid
            ModelNotFoundError: If model is not found
            InvalidRequestError: If request is malformed
            APIConnectionError: If connection fails
            APIError: For other API-related errors
        """
        # Check rate limit before making request
        wait_time = self._rate_limiter.check_rate_limit(self.api_key)
        if wait_time:
            raise RateLimitError(wait_time)
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    **kwargs
                }
                
                # Log request details
                request_data = {
                    "provider": "openrouter",
                    "model": self.model,
                    "base_url": self.api_base,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "kwargs": {k: v for k, v in kwargs.items() if k not in ['api_key']},
                    "retry_count": retries,
                    "timestamp": datetime.now().isoformat()
                }
                api_logger.debug(f"OpenRouter Request: {json.dumps(request_data, indent=2)}")
                
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    headers=self.default_headers,
                    json=data,
                    timeout=30
                )
                
                # Update rate limit info from headers
                self._rate_limiter.log_rate_limit_headers(response.headers, self.api_key)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('retry-after', 60))
                    raise RateLimitError(retry_after)
                
                response.raise_for_status()
                response_data = response.json()
                
                # Extract token usage if available
                usage = response_data.get("usage", {})
                if usage:
                    kwargs["token_usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                    
                    # Calculate cost
                    cost = self._calculate_cost(kwargs["token_usage"])
                    kwargs["cost"] = cost
                    
                    # Add model information to kwargs
                    if "metadata" not in kwargs:
                        kwargs["metadata"] = {}
                    kwargs["metadata"]["model"] = self.model
                
                # Log success response details
                response_data_log = {
                    "provider": "openrouter",
                    "model": self.model,
                    "status": "success",
                    "usage": kwargs.get("token_usage"),
                    "cost": kwargs.get("cost"),
                    "retry_count": retries,
                    "timestamp": datetime.now().isoformat()
                }
                api_logger.debug(f"OpenRouter Response: {json.dumps(response_data_log, indent=2)}")
                
                return response_data["choices"][0]["message"]["content"]
                
            except RateLimitError as e:
                # Log rate limit error
                error_data = {
                    "provider": "openrouter",
                    "model": self.model,
                    "status": "error",
                    "error_type": "RateLimitError",
                    "error": str(e),
                    "retry_count": retries,
                    "timestamp": datetime.now().isoformat()
                }
                api_logger.error(f"OpenRouter Error: {json.dumps(error_data, indent=2)}")
                # Don't retry rate limit errors
                raise e
            
            except requests.exceptions.Timeout:
                error_data = {
                    "provider": "openrouter",
                    "model": self.model,
                    "status": "error",
                    "error_type": "Timeout",
                    "error": "Request timed out after 30 seconds",
                    "retry_count": retries,
                    "timestamp": datetime.now().isoformat()
                }
                api_logger.error(f"OpenRouter Error: {json.dumps(error_data, indent=2)}")
                last_error = APIConnectionError(
                    "Request timed out",
                    raw_error="Timeout after 30 seconds"
                )
            
            except requests.exceptions.ConnectionError as e:
                error_data = {
                    "provider": "openrouter",
                    "model": self.model,
                    "status": "error",
                    "error_type": "ConnectionError",
                    "error": str(e),
                    "retry_count": retries,
                    "timestamp": datetime.now().isoformat()
                }
                api_logger.error(f"OpenRouter Error: {json.dumps(error_data, indent=2)}")
                last_error = APIConnectionError(
                    "Failed to connect to API",
                    raw_error=str(e)
                )
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                error_msg = None
                
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get('error', {}).get('message')
                except:
                    error_msg = e.response.text
                
                if status_code == 401:
                    raise AuthenticationError("Invalid API key", status_code, error_msg)
                elif (status_code == 404 or 
                      (status_code == 400 and "model" in str(error_msg).lower())):
                    raise ModelNotFoundError(
                        f"Model '{self.model}' not found",
                        status_code,
                        error_msg
                    )
                elif status_code == 400:
                    raise InvalidRequestError(
                        "Invalid request",
                        status_code,
                        error_msg
                    )
                else:
                    last_error = APIError(
                        f"HTTP error {status_code}",
                        status_code,
                        error_msg
                    )
            
            except Exception as e:
                last_error = APIError(f"Unexpected error: {str(e)}")
            
            # If we get here, the request failed but we can retry
            retries += 1
            if retries < self.max_retries:
                # Exponential backoff
                sleep_time = self.retry_delay * (2 ** (retries - 1))
                time.sleep(sleep_time)
        
        # If we've exhausted retries, raise the last error
        raise last_error or APIError("Maximum retries exceeded")
    
    def set_rate_limiter_for_testing(
        self, 
        rate_limiter: Optional[RateLimiter] = None
    ) -> RateLimiter:
        """
        Set a custom rate limiter for testing purposes.
        
        Args:
            rate_limiter (Optional[RateLimiter]): The rate limiter to use
            
        Returns:
            RateLimiter: The previous rate limiter
        """
        previous = self._rate_limiter
        if rate_limiter is not None:
            self._rate_limiter = rate_limiter
        return previous

    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """
        Calculate the cost of the API call based on token usage.
        Uses the model's base prices from OpenRouter.
        
        Args:
            usage (Dict[str, int]): Token usage information
            
        Returns:
            float: Estimated cost in USD
        """
        # Base costs per 1M tokens (from OpenRouter pricing)
        base_costs = {
            "openai/gpt-4-turbo-preview": {"prompt": 10.0, "completion": 30.0},
            "openai/gpt-4": {"prompt": 30.0, "completion": 60.0},
            "anthropic/claude-2": {"prompt": 8.0, "completion": 24.0},
            "meta-llama/llama-2-70b-chat": {"prompt": 1.0, "completion": 1.0},
            "google/palm-2-chat-bison": {"prompt": 0.5, "completion": 0.5},
            "liquid/lfm-7b": {"prompt": 0.1, "completion": 0.1}  # Added LFM model
        }
        
        # Get the base cost for the model, default to highest tier if unknown
        model_costs = base_costs.get(self.model, base_costs["openai/gpt-4"])
        
        # Calculate costs per token (converting from per 1M tokens to per token)
        prompt_cost = (usage["prompt_tokens"] * model_costs["prompt"]) / 1_000_000
        completion_cost = (usage["completion_tokens"] * model_costs["completion"]) / 1_000_000
        
        return prompt_cost + completion_cost


def create_llm_client(provider: str, **kwargs) -> LLMClient:
    """
    Factory function to create an LLM client based on the provider.
    
    Args:
        provider (str): The provider name ('openai', 'claude', or 'openrouter')
        **kwargs: Additional arguments for the client initialization
        
    Returns:
        LLMClient: An instance of the appropriate LLM client
        
    Raises:
        ValueError: If the provider is not supported
    """
    providers = {
        'openai': OpenAIClient,
        'claude': ClaudeClient,
        'openrouter': OpenRouterClient
    }
    
    if provider not in providers:
        raise ValueError(f"Unsupported provider. Choose from: {', '.join(providers.keys())}")
    
    return providers[provider](**kwargs)
