# InnovAI

A Python library for interacting with various LLM providers (OpenAI, Anthropic, OpenRouter) with robust error handling, rate limiting, and data logging capabilities.

## Features

- Multi-provider support (OpenAI, Anthropic, OpenRouter)
- Robust error handling and retry mechanisms
- Rate limiting to prevent API quota exhaustion
- Comprehensive data logging and tracking
- Configurable settings via environment variables
- Batch processing capabilities

## Project Structure

```
innovai/
├── src/
│   └── data_creation/         # Core package
│       ├── config.py          # Configuration management
│       ├── data_logger.py     # Data storage and logging
│       ├── error_handling.py  # Error handling utilities
│       ├── interface.py       # Main LLM interface
│       ├── llm_clients.py     # Provider-specific clients
│       ├── logging_utils.py   # Logging utilities
│       ├── operations.py      # Core operations
│       └── rate_limiter.py    # Rate limiting
├── tests/                     # Test suite
├── docs/                      # Documentation
├── prompts/                   # Sample prompts
└── research_logs/            # Research findings
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   venv\Scripts\activate     # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # OPENAI_API_KEY=your_key
   # ANTHROPIC_API_KEY=your_key
   # OPENROUTER_API_KEY=your_key
   ```

## Usage

### Basic Usage

```python
from src.data_creation import LLMInterface

# Initialize the interface
llm = LLMInterface()

# Generate text with OpenAI
response = llm.generate(
    prompt="Your prompt here",
    provider="openai",
    model="gpt-4"
)

# Process batch prompts
responses = llm.generate_batch(
    prompt_file="prompts/your_prompts.txt",
    provider="openai",
    model="gpt-4",
    temperature=0.7
)
```

### Command Line Interface

The package includes a command-line script for running prompts:

```bash
python src/data_creation/run_llm.py --prompt-file prompts/your_prompts.txt --provider openai --model gpt-4 --output-file outputs/results.json --batch --temperature 0.9
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
