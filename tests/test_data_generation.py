"""
Tests for the data generation functionality.
"""
from unittest.mock import Mock, patch
import pytest
from src.data_creation.operations import DataGenerator, LLMOperationError
from src.data_creation.llm_clients import LLMClient

class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Mock implementation of text generation.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments (unused in mock)
            
        Returns:
            str: A mock response containing the input prompt
        """
        return f"Mock response for: {prompt}"

def test_data_generator_initialization():
    """Test DataGenerator initialization."""
    with patch('src.data_creation.operations.create_llm_client') as mock_create:
        mock_create.return_value = MockLLMClient()
        generator = DataGenerator(['openai', 'claude'])
        assert len(generator.clients) == 2
        assert 'openai' in generator.clients
        assert 'claude' in generator.clients

def test_generate_single():
    """Test single text generation."""
    with patch('src.data_creation.operations.create_llm_client') as mock_create:
        mock_create.return_value = MockLLMClient()
        generator = DataGenerator(['openai'])
        
        result = generator.generate_single("Test prompt", 'openai')
        assert result['status'] == 'success'
        assert "Mock response for: Test prompt" in result['response']
        assert 'duration' in result

def test_generate_parallel():
    """Test parallel text generation."""
    with patch('src.data_creation.operations.create_llm_client') as mock_create:
        mock_create.return_value = MockLLMClient()
        generator = DataGenerator(['openai'])
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = generator.generate_parallel(prompts, 'openai')
        
        assert len(results) == len(prompts)
        for result in results:
            assert result['status'] == 'success'
            assert "Mock response for:" in result['response']

def test_compare_providers():
    """Test provider comparison."""
    with patch('src.data_creation.operations.create_llm_client') as mock_create:
        mock_create.return_value = MockLLMClient()
        generator = DataGenerator(['openai', 'claude'])
        
        results = generator.compare_providers("Test prompt")
        assert len(results) == 2
        for _, result in results.items():
            assert result['status'] == 'success'
            assert "Mock response for:" in result['response']

def test_error_handling():
    """Test error handling in DataGenerator."""
    with patch('src.data_creation.operations.create_llm_client') as mock_create:
        mock_client = MockLLMClient()
        mock_client.generate_text = Mock(side_effect=Exception("API Error"))
        mock_create.return_value = mock_client
        
        generator = DataGenerator(['openai'])
        result = generator.generate_single("Test prompt", 'openai')
        
        assert result['status'] == 'failed'
        assert 'error' in result
        assert "API Error" in result['error']

def test_invalid_provider():
    """Test handling of invalid provider."""
    generator = DataGenerator(['openai'])
    with pytest.raises(LLMOperationError):
        generator.generate_single("Test prompt", 'invalid_provider')
