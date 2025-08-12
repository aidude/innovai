#!/usr/bin/env python
"""
Command-line script for interacting with LLM providers.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.data_creation import LLMInterface, run_from_file

def list_providers_and_models():
    """List all available providers and their models."""
    interface = LLMInterface()
    
    print("\nAvailable Providers and Models:")
    print("-" * 30)
    
    for provider in interface.list_providers():
        print(f"\n{provider.upper()}:")
        for model in interface.list_models(provider):
            print(f"  - {model}")

def process_prompts(
    prompt_file: str,
    provider: str,
    model: str,
    output_file: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    batch_mode: bool = False,
    creativity_mode: bool = False
):
    """
    Process prompts using the specified provider and model.
    
    Args:
        prompt_file (str): Path to the prompt file
        provider (str): Provider to use
        model (str): Model to use
        output_file (Optional[str]): Path to save the output
        temperature (float): Temperature for generation
        max_tokens (Optional[int]): Maximum tokens to generate
        batch_mode (bool): If True, process each line as a separate prompt
        creativity_mode (bool): Whether to enable creativity enhancements
    """
    try:
        # Validate prompt file
        prompt_path = Path(prompt_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        # Set up kwargs
        kwargs = {
            "temperature": temperature,
            "creativity_mode": creativity_mode
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        # Run generation
        result = run_from_file(
            prompt_file,
            provider,
            model,
            output_file,
            batch_mode=batch_mode,
            **kwargs
        )
        
        if batch_mode:
            # Print batch results
            print(f"\nProcessed {len(result)} prompts:")
            for i, res in enumerate(result, 1):
                print(f"\nPrompt {i}:")
                print("-" * 30)
                print(f"Status: {res['status']}")
                
                if res['status'] == 'success':
                    # Truncate long responses in console output
                    response_preview = res['response'][:100] + "..." if len(res['response']) > 100 else res['response']
                    print(f"Response: {response_preview}")
                    print(f"Generation time: {res['duration']:.2f} seconds")
                    print(f"Data ID: {res['data_id']}")
                else:
                    print(f"Error: {res.get('error', 'Unknown error')}")
            
            if output_file:
                print(f"\nResults saved in directory: {Path(output_file).parent}")
        else:
            # Print single result
            print("\nGeneration Result:")
            print("-" * 30)
            print(f"Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"\nResponse:\n{result['response']}")
                print(f"\nGeneration time: {result['duration']:.2f} seconds")
                print(f"Data ID: {result['data_id']}")
                
                if output_file:
                    print(f"\nResult saved to: {output_file}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified LLM Interface")
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available providers and models"
    )
    
    parser.add_argument(
        "--prompt-file",
        type=str,
        help="Path to the prompt file"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=LLMInterface.PROVIDERS.keys(),
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save the output"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process each line in the prompt file as a separate prompt"
    )
    
    parser.add_argument(
        "--creativity",
        action="store_true",
        help="Enable creativity enhancements"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_providers_and_models()
    elif all([args.prompt_file, args.provider, args.model]):
        process_prompts(
            args.prompt_file,
            args.provider,
            args.model,
            args.output_file,
            args.temperature,
            args.max_tokens,
            batch_mode=args.batch,
            creativity_mode=args.creativity
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
