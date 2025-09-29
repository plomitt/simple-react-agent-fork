# Instructor Based Module

Provides the core LLM client infrastructure and reasoning agent implementation for StructAgent. This module handles model communication, structured reasoning, and tool orchestration.

## Components

### 1. ReasoningAgent (`reasoning_agent.py`)

The main agent implementation that implements a think-act-validate reasoning loop.

**Features:**
- Structured reasoning with think-act-validate steps
- Confidence scoring and step-by-step reasoning
- Tool integration and execution
- State management and conversation history
- Configurable reasoning parameters

**Key Classes:**
- `ReasoningStep`: Individual reasoning step with action, reasoning, and confidence
- `ReasoningSteps`: Collection of reasoning steps
- `ReasoningAgent`: Main agent class

### 2. Client Manager (`client_manager.py`)

Handles LLM client creation and model resolution.

**Features:**
- Support for multiple LLM providers (OpenRouter, OpenAI, etc.)
- Model resolution from names to specific model IDs
- Client configuration and authentication
- Retry mechanisms and error handling

**Key Functions:**
- `build_client()`: Create configured LLM client
- `resolve_model()`: Resolve model names to specific IDs

### 3. Prompt Manager (`prompt_manager.py`)

Manages system prompts and prompt templates.

**Features:**
- System prompt templates
- Dynamic prompt generation
- Prompt customization for different agent types
- Context-aware prompt management

**Key Functions:**
- `get_system_prompt()`: Get system prompt for agent type
- `format_prompt()`: Format prompts with context

### 4. Tool Manager (`tool_manager.py`)

Provides tool registry and execution framework.

**Features:**
- Tool registration and discovery
- Tool execution with proper error handling
- Tool specification validation
- Async tool execution support

**Key Classes:**
- `ToolSpec`: Tool specification interface
- `ToolRegistry`: Tool registration and management

## Installation

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- OpenRouter API key

### Setup

1. **Install dependencies**
   ```bash
   # From project root
   poetry install
   ```

2. **Configure environment**
   ```bash
   # Edit .env
   cp .env.example .env
   # And add your OpenRouter API key
   OPENROUTER_API_KEY="your-api-key-here"
   ```

3. **Set PYTHONPATH**
   ```bash
   export PYTHONPATH=~/projects/simple_react_agent/src
   ```

## Basic Usage

### Using the ReasoningAgent

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

### Running as a Module

```bash
# With PYTHONPATH set
export PYTHONPATH=~/projects/simple_react_agent/src

# Run the reasoning agent directly
poetry run python -m struct_agent.instructor_based.reasoning_agent
```

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

## Advanced Configuration

### Agent Parameters

```python
agent = ReasoningAgent(
    thinking_model="qwen/qwen3-next-80b-a3b-thinking",  # Model to use for thinking
    action_model="qwen/qwen3-next-80b-a3b-instruct",    # Model to use for tool calling (defaults to thinking model)
    temperature=0.1,                                    # Response randomness (0-1)
    min_steps=1,                                        # Minimum reasoning steps
    max_steps=10,                                       # Maximum reasoning steps
    base_url="https://openrouter.ai/api/v1",            # Custom provider API base URL
    api_key="your-api-key",                             # Custom provider API key
    use_lmstudio=False,                                 # Switch to a localy hosted model if True
)
```

### Custom Models

```python
from struct_agent.instructor_based.client_manager import build_client

# OpenRouter model configuration
agent = ReasoningAgent(
    model="anthropic/claude-3.5-sonnet",
    api_key="your-api-key",
    base_url="https://openrouter.ai/api/v1"
)

# Localy hosted model
agent = ReasoningAgent(
    model="openai/gpt-oss-20b",
    api_key="your-api-key",
    base_url="https://123.123.1.123:1234/v1",
    use_lmstudio=True
)
```

## API Reference

### ReasoningAgent

#### Constructor
```python
ReasoningAgent(
    thinking_model="qwen/qwen3-next-80b-a3b-thinking",
    action_model="qwen/qwen3-next-80b-a3b-instruct",
    temperature=0.1,
    min_steps=1,
    max_steps=10,
    base_url="https://openrouter.ai/api/v1",
    api_key="your-api-key",
    use_lmstudio=False,  
)
```

#### Methods
- `run(query: str) -> Response`: Run reasoning on a query
- `add_tool(tool: ToolSpec)`: Add a tool to the agent

### ToolSpec

#### Constructor
```python
ToolSpec(
    name: str,
    description: str,
    args_model: Type[BaseModel],
    handler: Callable,
    parameters: Optional[Dict[str, str]] = None
)
```

### Response Objects

- `Response`: Main response with `content` and `reasoning_steps`
- `ReasoningStep`: Individual step with `title`, `action`, `reasoning`, `confidence`
- `MaybeToolCall`: Tool execution result with `result` or `error`

## Dependencies

### Core Dependencies
- `pydantic>=2.7`: Data validation and serialization
- `instructor>=1.3`: Structured outputs for LLM
- `openai>=1.30`: OpenAI API client
- `python-dotenv>=1.0`: Environment variable management
- `tenacity>=8.2`: Retry mechanisms
- `aiohttp>=3.9`: Async HTTP client (for tool execution)

## Error Handling

The module includes comprehensive error handling:

### Common Errors
- `ValueError`: Invalid configuration parameters
- `ConnectionError`: LLM API connection issues
- `RuntimeError`: Tool execution failures
- `ValidationError`: Pydantic validation errors

### Retry Logic
- Automatic retries for transient failures
- Exponential backoff
- Configurable retry limits

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure PYTHONPATH is set correctly
   ```bash
   export PYTHONPATH=~/projects/simple_react_agent/src
   ```

2. **Authentication Errors**: Verify OpenRouter API key and base URL
   ```bash
   # Check .env file contains OPENROUTER_API_KEY and OPENROUTER_BASE_URL
   cat .env
   ```

3. **Tool Execution Failures**: Check tool handler implementation
   ```python
   # Test tool handler independently
   result = your_tool_handler({"test": "value"})
   print(result)
   ```

## Examples

See the main README for complete examples or run:
```bash
# With PYTHONPATH set
poetry run python -m struct_agent.instructor_based.reasoning_agent
```