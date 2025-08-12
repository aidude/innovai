"""
Tests for LLM API clients.
"""
import os
import pytest
from src.data_creation.llm_clients import create_llm_client
from src.data_creation.rate_limiter import (
    RateLimiter, RateLimitError, AuthenticationError, ModelNotFoundError
)

def test_openrouter_successful_call():
    """Test successful API call to OpenRouter."""
    client = create_llm_client('openrouter', model='liquid/lfm-7b')
    
    # Test a simple prompt
    response = client.generate_text(
        "What is 2+2? Answer with just the number.",
        temperature=0.1,
        max_tokens=10
    )
    
    # Basic validation
    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0

def test_openrouter_invalid_api_key():
    """Test behavior with invalid API key."""
    # Temporarily change API key
    original_key = os.environ.get('OPENROUTER_API_KEY')
    os.environ['OPENROUTER_API_KEY'] = 'invalid_key'
    
    try:
        client = create_llm_client('openrouter')
        with pytest.raises(AuthenticationError) as exc_info:
            client.generate_text("Test prompt")
        assert "Invalid API key" in str(exc_info.value)
    finally:
        # Restore original key
        if original_key:
            os.environ['OPENROUTER_API_KEY'] = original_key
        else:
            del os.environ['OPENROUTER_API_KEY']

def test_openrouter_invalid_model():
    """Test behavior with non-existent model."""
    client = create_llm_client('openrouter', model='nonexistent-model')
    
    with pytest.raises(ModelNotFoundError) as exc_info:
        client.generate_text("Test prompt")
    assert "Model 'nonexistent-model' not found" in str(exc_info.value)

@pytest.mark.rate_limit
def test_openrouter_rate_limit():
    """Test rate limiting functionality."""
    client = create_llm_client('openrouter')
    
    # Create a new rate limiter instance for testing
    test_limiter = RateLimiter(calls_per_minute=1, calls_per_hour=10)
    original_limiter = client.set_rate_limiter_for_testing(test_limiter)
    
    try:
        # First call should succeed
        response = client.generate_text(
            "Test prompt 1",
            temperature=0.1,
            max_tokens=10
        )
        assert response is not None
        
        # Second call should trigger rate limit
        with pytest.raises(RateLimitError) as exc_info:
            client.generate_text("Test prompt 2")
        assert "Rate limit exceeded" in str(exc_info.value)
    finally:
        # Restore original rate limiter
        client.set_rate_limiter_for_testing(original_limiter)

@pytest.mark.retry_test
def test_openrouter_retry_mechanism():
    """Test retry mechanism with a long prompt."""
    client = create_llm_client('openrouter')
    
    # Create a new rate limiter instance for testing with higher limits
    test_limiter = RateLimiter(calls_per_minute=10, calls_per_hour=100)
    original_limiter = client.set_rate_limiter_for_testing(test_limiter)
    
    try:
        # Test with a longer interaction to potentially trigger retries
        prompt = "Write a short story about a robot who learns to paint. Max 50 words."
        
        response = client.generate_text(
            prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0
    finally:
        # Restore original rate limiter
        client.set_rate_limiter_for_testing(original_limiter)
import os
import pytest
from src.data_creation.llm_clients import create_llm_client
from src.data_creation.rate_limiter import (
    RateLimiter, RateLimitError, AuthenticationError,
    ModelNotFoundError, InvalidRequestError, APIConnectionError
)

def test_openrouter_successful_call():
    """Test successful API call to OpenRouter."""
    client = create_llm_client('openrouter', model='liquid/lfm-7b')
    
    # Test a simple prompt
    response = client.generate_text(
        "What is 2+2? Answer with just the number.",
        temperature=0.1,
        max_tokens=10
    )
    
    # Basic validation
    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0

def test_openrouter_invalid_api_key():
    """Test behavior with invalid API key."""
    # Temporarily change API key
    original_key = os.environ.get('OPENROUTER_API_KEY')
    os.environ['OPENROUTER_API_KEY'] = 'invalid_key'
    
    try:
        client = create_llm_client('openrouter')
        with pytest.raises(AuthenticationError) as exc_info:
            client.generate_text("Test prompt")
        assert "Invalid API key" in str(exc_info.value)
    finally:
        # Restore original key
        if original_key:
            os.environ['OPENROUTER_API_KEY'] = original_key
        else:
            del os.environ['OPENROUTER_API_KEY']

def test_openrouter_invalid_model():
    """Test behavior with non-existent model."""
    client = create_llm_client('openrouter', model='nonexistent-model')
    
    with pytest.raises(ModelNotFoundError) as exc_info:
        client.generate_text("Test prompt")
    assert "Model 'nonexistent-model' not found" in str(exc_info.value)

@pytest.mark.rate_limit
def test_openrouter_rate_limit():
    """Test rate limiting functionality."""
    client = create_llm_client('openrouter')
    
    # Create a new rate limiter instance for testing
    test_limiter = RateLimiter(calls_per_minute=1, calls_per_hour=10)
    original_limiter = client.set_rate_limiter_for_testing(test_limiter)
    
    try:
        # First call should succeed
        response = client.generate_text(
            "Test prompt 1",
            temperature=0.1,
            max_tokens=10
        )
        assert response is not None
        
        # Second call should trigger rate limit
        with pytest.raises(RateLimitError) as exc_info:
            client.generate_text("Test prompt 2")
        assert "Rate limit exceeded" in str(exc_info.value)
    finally:
        # Restore original rate limiter
        client._rate_limiter = original_limiter

@pytest.mark.retry_test
def test_openrouter_retry_mechanism():
    """Test retry mechanism with a long prompt."""
    client = create_llm_client('openrouter')
    
    # Create a new rate limiter instance for testing with higher limits
    test_limiter = RateLimiter(calls_per_minute=10, calls_per_hour=100)
    original_limiter = client._rate_limiter
    client._rate_limiter = test_limiter
    
    try:
        # Test with a longer interaction to potentially trigger retries
        prompt = "Write a short story about a robot who learns to paint. Max 50 words."
        
        response = client.generate_text(
            prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0
    finally:
        # Restore original rate limiter
        client._rate_limiter = original_limiter
