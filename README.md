# StructAgent

A small, simple, and structured AI agent with reasoning capabilities and tool integration. Built with modern Python patterns and designed for extensibility.

## Features

- **Reasoning Agent**: Think-act-validate loop with structured reasoning steps
- **Tool Integration**: Modular tool system with pluggable capabilities
- **Multiple LLM Providers**: Support for OpenRouter, LM Studio, and other providers
- **Mathematical Operations**: Symbolic math and expression evaluation
- **Web Search**: SearXNG integration for meta-search capabilities
- **Vector Indexing**: LEANN integration for semantic search and RAG
- **Type Safety**: Full Pydantic validation and type hints

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- OpenRouter API key

### Installation

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/esshka/simple-react-agent.git
   cd simple_react_agent
   poetry install
   ```

2. **Configure environment**
   ```bash
   # Edit .env
   cp .env.example .env

   # And add your information
   OPENROUTER_MODEL_ID=qwen/qwen3-next-80b-a3b-thinking
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   OPENROUTER_API_KEY="your-api-key"

   LMSTUDIO_MODEL_ID=qwen/qwen3-next-80b-a3b-thinking
   LMSTUDIO_BASE_URL=http://123.123.1.123:1234/v1
   LMSTUDIO_API_KEY="your-api-key"

   LEANN_CHAT_MODEL=qwen/qwen3-next-80b-a3b-thinking
   SEARXNG_BASE_URL=http://localhost:8080
   ```

3. **Run the agent example**
   ```bash
   # Run as a module with PYTHONPATH
   export PYTHONPATH=~/projects/simple_react_agent/src
   poetry run python -m struct_agent.instructor_based.reasoning_agent
   ```

## Module Overview

### 1. Instructor Based (`src/struct_agent/instructor_based/`)
- **ReasoningAgent**: Main agent implementation with think-act-validate loop
- **Client Manager**: LLM client management and model resolution
- **Prompt Manager**: System prompt templates and management
- **Tool Manager**: Tool registration and execution framework

### 2. Tools (`src/struct_agent/tools/`)
- **Math Tools**: Arithmetic, symbolic math, equation solving
- **SearXNG Tools**: Web search and meta-search capabilities
- **LEANN Tools**: Vector indexing and semantic search
- **Toolkits**: Pre-configured tool bundles

## Basic Usage

### Using the Reasoning Agent

```python
from struct_agent.instructor_based import ReasoningAgent
from struct_agent.tools import MathsToolkit

# Initialize the agent
agent = ReasoningAgent()

# Add tools from toolkits
toolkits = [MathsToolkit()]
for toolkit in toolkits:
    for tool in toolkit:
        agent.add_tool(tool)

# Run the agent
response = agent.run("What is the square root of 144?")
print(f"Answer: {response.reasoning_steps[-1].result}")
print(f"Reasoning steps: {len(response.reasoning_steps)}")
```

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Required for LLM access
- `OPENROUTER_BASE_URL`: OpenRouter API base URL
- `SEARXNG_BASE_URL`: SearXNG instance URL (default: http://localhost:8080)
- `LEANN_CHAT_MODEL`: Model for LEANN chat functionality

### Supported Models

The agent supports various models through OpenRouter:
- `qwen/qwen3-next-80b-a3b-thinking`
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o`
- And many more

## Advanced Usage

### Custom Tool Integration

```python
from struct_agent.instructor_based import ReasoningAgent, ToolSpec
from pydantic import BaseModel, Field
from typing import Dict, Any

# Create structured arguments model
class WeatherArgs(BaseModel):
    location: str = Field(..., description="Location to get weather for")

# Create handler function
def weather_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    parsed_args = WeatherArgs(**args)
    # Your weather API logic here
    return {
        "temperature": "25°C",
        "condition": "sunny",
        "location": parsed_args["location"]
    }

# Create tool specification
weather_tool = ToolSpec(
    name="get_weather",
    description="Get current weather for a location",
    args_model=WeatherArgs,
    handler=weather_handler,
    parameters={
        "location": "The location to get weather for"
    }
)

# Add to agent
agent = ReasoningAgent()
agent.add_tool(weather_tool)

# Use the tool
response = agent.run("What's the weather in New York?")
```

### Running as Modules

```bash
# With PYTHONPATH set
export PYTHONPATH=~/projects/simple_react_agent/src

# Run individual module examples
poetry run python -m struct_agent.instructor_based.reasoning_agent
poetry run python -m struct_agent.tools.searxng_tools
```

## Dependencies

### Core Dependencies
- `pydantic>=2.7`: Data validation and serialization
- `instructor>=1.3`: Structured outputs for LLM
- `openai>=1.30`: OpenAI API client
- `requests>=2.31`: HTTP requests
- `tenacity>=8.2`: Retry mechanisms
- `python-dotenv>=1.0`: Environment variable management

### Tool Dependencies
- `sympy`: Symbolic mathematics
- `beautifulsoup4`: HTML parsing
- `ddgs`: DuckDuckGo search
- `leann>=0.3.4`: Vector indexing and search
- `aiohttp`: Async HTTP client

## Project Structure

```
src/struct_agent/
├── __init__.py                 # Main package exports
├── instructor_based/           # LLM client and agent logic
│   ├── __init__.py
│   ├── reasoning_agent.py      # Main ReasoningAgent class
│   ├── client_manager.py       # LLM client management
│   ├── prompt_manager.py       # System prompts
│   └── tool_manager.py         # Tool registry and execution
├── tools/                      # Modular tool system
│   ├── __init__.py
│   ├── maths_tools.py         # Mathematical operations
│   ├── searxng_tools.py       # Web search
│   ├── leann_tools.py         # Vector indexing
│   └── toolkits.py            # Pre-configured tool bundles
└── README.md                  # This file
```

## Development

### Adding New Tools

1. Create tool file in `src/struct_agent/tools/`
2. Implement handler function and Pydantic models
3. Add tool spec to module `__all__`

### Code Style

- Follow PEP 8
- Use type hints consistently
- Keep modules under 300 lines
- Use Pydantic for all data validation
- Include docstrings for all public methods

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure PYTHONPATH includes the src directory
   ```bash
   export PYTHONPATH=~/projects/simple_react_agent/src
   ```

2. **Import Issues**: Use absolute imports for python -m compatibility
   ```python
   from struct_agent.instructor_based import ReasoningAgent  # Good
   from .instructor_based import ReasoningAgent              # Bad for python -m
   ```

3. **Missing Dependencies**: Run poetry install to update
   ```bash
   poetry install
   ```

4. **Environment Variables**: Verify .env file exists and has required keys
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request