"""
Test script to verify data generation, storage, and maintenance functionality.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from src.data_creation.operations import DataGenerator
from src.data_creation.data_logger import DataLogger
from src.data_creation.data_maintenance import DataMaintenance, MetadataTracker

def main():
    """Run a complete test of the data generation and storage system."""
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    generator = DataGenerator(['openai'])  # Using only OpenAI for this test
    data_logger = generator.data_logger
    base_dir = Path(__file__).parent.parent
    maintenance = DataMaintenance(base_dir)
    metadata_tracker = MetadataTracker(base_dir)
    
    # Test prompts
    prompts = [
        "Write a one-sentence story about a robot.",
        "Create a haiku about programming.",
        "Define artificial intelligence in simple terms."
    ]
    
    print("Starting test sequence...")
    print("-" * 50)
    
    # Generate and store responses
    for prompt in prompts:
        print(f"\nProcessing prompt: {prompt}")
        result = generator.generate_single(prompt, 'openai')
        
        if result['status'] == 'success':
            print(f"Generated response with ID: {result['data_id']}")
            print(f"Response duration: {result['duration']:.2f} seconds")
            
            # Update metadata with mock token usage
            metadata_tracker.update_metadata(
                data_id=result['data_id'],
                provider='openai',
                tokens={
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_tokens": 150
                },
                cost=0.002 * 150,  # Mock cost calculation
                metadata={"model": "gpt-4", "temperature": 0.7}
            )
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\nChecking storage statistics...")
    print("-" * 50)
    stats = maintenance.get_storage_stats()
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    
    print("\nProvider Statistics:")
    print("-" * 50)
    openai_stats = metadata_tracker.get_provider_stats('openai')
    print(f"OpenAI total requests: {openai_stats.get('total_requests', 0)}")
    print(f"OpenAI total tokens: {openai_stats.get('total_tokens', 0)}")
    print(f"OpenAI total cost: ${openai_stats.get('total_cost', 0):.4f}")
    
    print("\nTesting data maintenance...")
    print("-" * 50)
    maintenance.archive_old_data(days_threshold=0)  # Archive everything for testing
    maintenance.cleanup_archived_data(days_threshold=90)
    
    print("\nFinal storage state:")
    print("-" * 50)
    final_stats = maintenance.get_storage_stats()
    print("Active data:")
    for provider, stats in final_stats['active_data'].items():
        print(f"- {provider}: {stats['files']} files, {stats['size_mb']:.2f} MB")
    print("\nArchived data:")
    for provider, stats in final_stats['archived_data'].items():
        print(f"- {provider}: {stats['files']} files, {stats['size_mb']:.2f} MB")

if __name__ == "__main__":
    main()
