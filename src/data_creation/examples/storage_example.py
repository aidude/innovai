"""
Example script demonstrating data storage and logging functionality.
"""
from src.data_creation.operations import DataGenerator
from src.data_creation.data_logger import DataLogger

def main():
    """Run example data generation with storage and logging."""
    # Initialize generator with providers
    generator = DataGenerator(['openai', 'claude'])
    data_logger = DataLogger()
    
    # Example prompts
    prompts = [
        "Write a short story about artificial intelligence.",
        "Create a poem about technology.",
        "Explain quantum computing to a 5-year-old."
    ]
    
    # Generate responses from different providers
    for prompt in prompts:
        # Try with OpenAI
        result_openai = generator.generate_single(prompt, 'openai')
        if result_openai['status'] == 'success':
            print(f"OpenAI Response ID: {result_openai['data_id']}")
            
            # Retrieve the stored data
            stored_data = data_logger.get_response(result_openai['data_id'])
            print(f"Retrieved prompt: {stored_data['prompt'][:50]}...")
            print(f"Retrieved response: {stored_data['response'][:50]}...")
            print(f"Generation time: {stored_data['metadata']['duration']:.2f} seconds")
            print("---")
        
        # Try with Claude
        result_claude = generator.generate_single(prompt, 'claude')
        if result_claude['status'] == 'success':
            print(f"Claude Response ID: {result_claude['data_id']}")
            
            # Retrieve the stored data
            stored_data = data_logger.get_response(result_claude['data_id'])
            print(f"Retrieved prompt: {stored_data['prompt'][:50]}...")
            print(f"Retrieved response: {stored_data['response'][:50]}...")
            print(f"Generation time: {stored_data['metadata']['duration']:.2f} seconds")
            print("---")
    
    # Show all responses from a specific provider
    openai_responses = data_logger.get_responses_by_provider('openai')
    print(f"\nFound {len(openai_responses)} responses from OpenAI")

if __name__ == "__main__":
    main()
