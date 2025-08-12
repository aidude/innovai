"""
Example usage scripts for the LLM clients.
"""
import time
from typing import List, Dict, Any
import json
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm_clients import create_llm_client, LLMClient
from .data_logger import DataLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LLMOperationError(Exception):
    """Custom exception for LLM operations."""
    pass


def retry_on_error(max_retries: int = 3, delay: int = 2):
    """
    Decorator for retrying failed API calls.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (int): Delay between retries in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(delay)
            raise LLMOperationError(f"All retry attempts failed: {str(last_error)}")
        return wrapper
    return decorator


class DataGenerator:
    """Class for handling data generation tasks using LLM clients."""
    
    def __init__(self, providers: List[str]):
        """
        Initialize with multiple LLM providers.
        
        Args:
            providers (List[str]): List of provider names ('openai', 'claude', 'openrouter')
        """
        self.clients = {}
        self.data_logger = DataLogger()
        
        for provider in providers:
            try:
                self.clients[provider] = create_llm_client(provider)
            except Exception as e:
                logger.error(f"Failed to initialize {provider} client: {str(e)}")
    
    @retry_on_error()
    def generate_single(self, prompt: str, provider: str, **kwargs) -> Dict[str, Any]:
        """
        Generate text using a single provider with retry logic.
        
        Args:
            prompt (str): Input prompt
            provider (str): Provider to use
            **kwargs: Additional arguments for the API call
            
        Returns:
            Dict[str, Any]: Result dictionary containing response and metadata
        """
        if provider not in self.clients:
            raise LLMOperationError(f"Provider {provider} not initialized")
        
        try:
            start_time = time.time()
            response = self.clients[provider].generate_text(prompt, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            # Extract model from kwargs metadata if available
            metadata = kwargs.get('metadata', {})
            metadata.update({
                'duration': duration,
                'kwargs': {k: v for k, v in kwargs.items() if k != 'metadata'}
            })
            
            # Get model from kwargs metadata or from client
            model = metadata.get('model') or getattr(self.clients[provider], 'model', None)
            
            # Store the response and get the data ID
            data_id = self.data_logger.store_response(
                response=response,
                provider=provider,
                prompt=prompt,
                metadata=metadata,
                model=model
            )
            
            return {
                'provider': provider,
                'response': response,
                'duration': duration,
                'status': 'success',
                'data_id': data_id
            }
        except Exception as e:
            logger.error(f"Error with {provider}: {str(e)}")
            return {
                'provider': provider,
                'error': str(e),
                'status': 'failed'
            }
    
    def generate_parallel(self, prompts: List[str], provider: str, max_workers: int = 3) -> List[Dict[str, Any]]:
        """
        Generate multiple responses in parallel using one provider.
        
        Args:
            prompts (List[str]): List of prompts
            provider (str): Provider to use
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            List[Dict[str, Any]]: List of result dictionaries
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_prompt = {
                executor.submit(self.generate_single, prompt, provider): prompt 
                for prompt in prompts
            }
            
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    result = future.result()
                    result['prompt'] = prompt
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel generation failed for prompt: {prompt[:50]}... Error: {str(e)}")
                    results.append({
                        'prompt': prompt,
                        'provider': provider,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        return results
    
    def compare_providers(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Compare responses from all initialized providers for the same prompt.
        
        Args:
            prompt (str): Input prompt
            **kwargs: Additional arguments for the API calls
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        results = {}
        for provider in self.clients:
            try:
                result = self.generate_single(prompt, provider, **kwargs)
                results[provider] = result
            except Exception as e:
                logger.error(f"Comparison failed for {provider}: {str(e)}")
                results[provider] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save generation results to a JSON file.
        
        Args:
            results (Dict[str, Any]): Results to save
            output_file (str): Output file path
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {str(e)}")
            raise LLMOperationError(f"Failed to save results: {str(e)}")
