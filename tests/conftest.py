"""
PyTest configuration and fixtures.
"""
import os
import pytest

@pytest.fixture(autouse=True)
def check_env_vars():
    """Check that required environment variables are set."""
    required_vars = ['OPENROUTER_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")
