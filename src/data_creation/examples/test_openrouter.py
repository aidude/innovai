"""
Test script for OpenRouter API integration.
"""
from pathlib import Path
from dotenv import load_dotenv
import json
from src.data_creation.operations import DataGenerator
from src.data_creation.data_maintenance import MetadataTracker

def test_openrouter_generation():
    """Test data generation using OpenRouter."""
    # Load environment variables
    load_dotenv()
    
    # Initialize generator with OpenRouter
    generator = DataGenerator(['openrouter'])
    base_dir = Path(__file__).parent.parent.parent
    metadata_tracker = MetadataTracker(base_dir)
    
    # Test prompts with different models
    test_cases = [
        {
            "model": "openai/gpt-4-turbo-preview",
            "prompt": "Write a short poem about artificial intelligence.",
            "temperature": 0.7
        },
        {
            "model": "openai/gpt-4",
            "prompt": "Explain quantum computing to a 5-year-old.",
            "temperature": 0.8
        },
        {
            "model": "openai/gpt-4-turbo-preview",
            "prompt": "Write a one-sentence story about the future.",
            "temperature": 0.9
        }
    ]
    
    print("Starting OpenRouter API test...")
    print("-" * 50)
    
    for case in test_cases:
        print(f"\nTesting with model: {case['model']}")
        print(f"Prompt: {case['prompt']}")
        
        try:
            # Configure the client for this model
            generator.clients['openrouter'].model = case['model']
            
            # Generate response
            result = generator.generate_single(
                case['prompt'],
                'openrouter',
                temperature=case['temperature']
            )
            
            if result['status'] == 'success':
                print(f"\nResponse received (ID: {result['data_id']}):")
                print("-" * 30)
                print(result['response'][:200] + "..." if len(result['response']) > 200 else result['response'])
                print("-" * 30)
                print(f"Generation time: {result['duration']:.2f} seconds")
                
                # Get metadata for this generation
                provider_stats = metadata_tracker.get_provider_stats('openrouter')
                if provider_stats:
                    print("\nProvider Statistics:")
                    print(f"Total requests: {provider_stats.get('total_requests', 0)}")
                    print(f"Total tokens: {provider_stats.get('total_tokens', 0)}")
                    print(f"Total cost: ${provider_stats.get('total_cost', 0):.4f}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error testing {case['model']}: {str(e)}")
    
    # Save final statistics
    print("\nFinal Statistics:")
    provider_stats = metadata_tracker.get_provider_stats('openrouter')
    
    stats_file = base_dir / "synthetic_data" / "openrouter_test_results.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_cases": test_cases,
            "provider_stats": provider_stats
        }, f, indent=2)
    
    print(f"Test results saved to: {stats_file}")

if __name__ == "__main__":
    test_openrouter_generation()
