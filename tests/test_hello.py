"""
Test module for the InnovAI project.
"""

from src import hello_world

def test_hello_world():
    """Test the hello_world function."""
    assert hello_world() == "Hello, InnovAI!"
