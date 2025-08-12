"""
Unified interface for interacting with different LLM providers.
"""
from typing import Dict, List, Optional, Union
import json
import random
from pathlib import Path
from difflib import SequenceMatcher

from .config import config
from .logging_utils import setup_logger
from .error_handling import LLMError, ModelNotFoundError
from .operations import DataGenerator
from .llm_clients import create_llm_client


# Set up logger
logger = setup_logger("LLMInterface")


class LLMInterface:
    """A unified interface for interacting with different LLM providers."""
    
    # Available providers and their default models - consider moving to config
    PROVIDERS = {
        'openai': [
            "gpt-4o",                 # GPT-4 Omni (fastest, multimodal, cost-efficient)
            "gpt-4-turbo",            # Optimized GPT-4 (cheaper & faster than GPT-4)
            "gpt-4",                  # Original GPT-4 (no longer updated)
            "gpt-3.5-turbo",          # General-purpose model (great balance of speed & cost)
            "gpt-3.5-turbo-16k"       # Same as above, but supports 16K tokens
        ],
        'claude': ['claude-3-opus-20240229'],
        'openrouter': [
            'openai/gpt-4-turbo-preview',
            'openai/gpt-4',
            'anthropic/claude-2',
            'meta-llama/llama-2-70b-chat',
            'google/palm-2-chat-bison',
            'liquid/lfm-7b'
        ]
    }
    
    def __init__(self):
        """Initialize the interface with all available providers."""
        self.generator = DataGenerator(list(self.PROVIDERS.keys()))
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """Returns a list of all available providers."""
        return list(cls.PROVIDERS.keys())
    
    @classmethod
    def list_models(cls, provider: str) -> List[str]:
        """Returns list of models available for the given provider."""
        if provider not in cls.PROVIDERS:
            providers = ", ".join(cls.PROVIDERS.keys())
            raise ModelNotFoundError(
                f"Unsupported provider. Choose from: {providers}"
            )
        return cls.PROVIDERS[provider]
    
    def _calculate_uniqueness(self, current_response: str,
                         previous_responses: List[str]) -> float:
        """Calculate uniqueness score against previous responses."""
        from difflib import SequenceMatcher
        
        similarity_scores = [
            SequenceMatcher(None, current_response, prev).ratio()
            for prev in previous_responses
        ]
        
        if not similarity_scores:
            return 1.0
            
        # Higher score means more unique (less similar)
        return 1 - (sum(similarity_scores) / len(similarity_scores))
    
    def enhance_prompt(self, prompt: str, creativity_mode: bool = False) -> str:
        """Enhances the prompt to encourage more creative responses."""
        if not creativity_mode:
            return prompt
        
        creativity_prompts = [
            "Think outside the box and provide a unique perspective on this: ",
            "Explore innovative and unconventional approaches to address: ",
            "Consider multiple angles and generate a creative solution for: ",
            "Challenge traditional assumptions while answering: ",
            "Imagine novel possibilities when responding to: "
        ]
        
        selected_prompt = random.choice(creativity_prompts)
        return selected_prompt + prompt
    
    def generate_batch(self, prompt_file: Union[str, Path], provider: str, model: str,
                      creativity_mode: bool = False, **kwargs) -> List[Dict]:
        """Process multiple prompts from a file, one per line."""
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        
        with open(prompt_path, "r", encoding="utf-8") as file_handle:
            prompts = [line.strip() for line in file_handle if line.strip()]
        
        self.generator.clients[provider].model = model
        
        if creativity_mode:
            creativity_params = {
                "temperature": 0.9,
                "frequency_penalty": 0.7,
                "presence_penalty": 0.7,
                "top_p": 0.95
            }
            kwargs.update(creativity_params)
        
        results = []
        previous_responses = []
        
        for prompt in prompts:
            enhanced_prompt = self.enhance_prompt(prompt, creativity_mode)
            
            result = self.generator.generate_single(
                enhanced_prompt,
                provider,
                **kwargs
            )
            
            if result.get("response"):
                previous_responses.append(result["response"])
                
                if creativity_mode and len(previous_responses) > 1:
                    # Calculate uniqueness metrics
                    uniqueness_score = self._calculate_uniqueness(
                        result["response"],
                        previous_responses[:-1]
                    )
                    
                    # Add creativity metrics
                    result["metadata"]["creativity_metrics"] = {
                        "uniqueness_score": uniqueness_score,
                        "response_length": len(result["response"]),
                        "enhanced_prompt_used": enhanced_prompt != prompt
                    }
            
            results.append(result)
        
        return results
    
    def generate(self, prompt_file: Union[str, Path], provider: str, model: str,
                creativity_mode: bool = False, **kwargs) -> Dict:
        """Generate text for a single prompt."""
        kwargs["creativity_mode"] = creativity_mode
        results = self.generate_batch(prompt_file, provider, model, **kwargs)
        return results[0]


def run_from_file(
    prompt_file: Union[str, Path],
    provider: str,
    model: str,
    output_file: Optional[Union[str, Path]] = None,
    batch_mode: bool = False,
    **kwargs
) -> Union[Dict, List[Dict]]:
    """
    Run generation from a file.
    
    Args:
        prompt_file: Path to the file containing prompts
        provider: Name of the LLM provider to use
        model: Name of the model to use
        output_file: Optional path to save results
        batch_mode: Whether to process each line as a separate prompt
        **kwargs: Additional arguments for generation
        
    Returns:
        Union[Dict, List[Dict]]: Generation results
    """
    interface = LLMInterface()
    
    try:
        if batch_mode:
            # Run batch generation
            results = interface.generate_batch(
                prompt_file,
                provider,
                model,
                **kwargs
            )
            
            # Save results if output file specified
            if output_file:
                output_path = Path(output_file)
                output_dir = output_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for i, result in enumerate(results, 1):
                    result_file = (
                        output_dir / f"{output_path.stem}_{i}{output_path.suffix}"
                    )
                    with open(result_file, "w", encoding="utf-8") as file_handle:
                        json.dump(result, file_handle, indent=2)
            
            return results
        
        else:
            # Run single generation
            result = interface.generate(
                prompt_file,
                provider,
                model,
                **kwargs
            )
            
            # Save result if output file specified
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as file_handle:
                    json.dump(result, file_handle, indent=2)
            
            return result
            
    except Exception as error:
        logger.error(f"Error in run_from_file: {str(error)}")
        raise LLMError(
            f"Failed to run generation from file: {str(error)}",
            original_error=error
        )
