"""
Example script demonstrating usage of LLM operations.
"""
import logging
from pathlib import Path
from typing import List

from src.data_creation.operations import DataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating various LLM operations."""
    # Initialize generator with multiple providers
    generator = DataGenerator(['openai', 'claude', 'openrouter'])
    
    # Example 1: Single generation
    prompt = "Write a short poem about artificial intelligence."
    try:
        result = generator.generate_single(prompt, 'openai')
        logger.info("Single generation result:")
        logger.info(result)
    except Exception as e:
        logger.error(f"Single generation failed: {str(e)}")
    
    # Example 2: Parallel generation
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "Describe deep learning.",
        "What is natural language processing?"
    ]
    try:
        results = generator.generate_parallel(prompts, 'claude')
        logger.info("\nParallel generation results:")
        for result in results:
            logger.info(f"Prompt: {result['prompt'][:30]}...")
            logger.info(f"Status: {result['status']}")
            if result['status'] == 'success':
                logger.info(f"Response: {result['response'][:100]}...")
            logger.info("---")
    except Exception as e:
        logger.error(f"Parallel generation failed: {str(e)}")
    
    # Example 3: Compare providers
    comparison_prompt = "Explain the concept of consciousness."
    try:
        comparison = generator.compare_providers(comparison_prompt)
        logger.info("\nProvider comparison results:")
        for provider, result in comparison.items():
            logger.info(f"\nProvider: {provider}")
            logger.info(f"Status: {result['status']}")
            if result['status'] == 'success':
                logger.info(f"Response: {result['response'][:100]}...")
    except Exception as e:
        logger.error(f"Provider comparison failed: {str(e)}")
    
    # Save results
    try:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        generator.save_results(
            comparison,
            output_dir / "provider_comparison.json"
        )
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

if __name__ == "__main__":
    main()
