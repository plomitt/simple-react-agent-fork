# Tools Module

A comprehensive collection of modular tools for the StructAgent system. This module provides mathematical operations, web search capabilities, vector indexing, and pre-configured toolkits.

## Available Tools

### 1. Mathematical Tools (`maths_tools.py`)

Advanced mathematical operations including arithmetic, symbolic algebra, and equation solving.

**Features:**
- Basic arithmetic expression evaluation
- Symbolic mathematics with SymPy
- Equation solving (linear, polynomial, systems)
- Expression simplification, expansion, and factorization
- Calculus operations (differentiation, integration)
- Matrix operations

**Available Functions:**
- `make_calc_tool()`: Basic arithmetic calculator
- `make_sympy_solve_equation_tool()`: Solve equations symbolically
- `make_sympy_simplify_expression_tool()`: Simplify mathematical expressions
- `make_sympy_expand_expression_tool()`: Expand algebraic expressions
- `make_sympy_factor_expression_tool()`: Factor polynomials
- `make_sympy_differentiate_tool()`: Compute derivatives
- `make_sympy_integrate_tool()`: Compute integrals
- `make_sympy_matrix_operation_tool()`: Matrix operations

### 2. SearXNG Tools (`searxng_tools.py`)

Web search and meta-search capabilities using SearXNG.

**Features:**
- Multi-query search execution
- Category-based filtering (news, images, videos)
- Asynchronous search operations
- Result deduplication and ranking
- Configurable result limits

**Available Functions:**
- `make_searxng_search_tool()`: Web search with SearXNG

### 3. LEANN Tools (`leann_tools.py`)

Vector indexing and semantic search using LEANN (The smallest vector index in the world).

**Features:**
- Text indexing and storage
- Semantic search capabilities
- Chat-based query interface
- Persistent index storage
- Configurable backend (DiskANN, HNSW)

**Available Functions:**
- `make_leann_add_text_tool()`: Add text to vector index
- `make_leann_search_tool()`: Search vector index
- `make_leann_chat_tool()`: Chat with indexed content

### 4. Toolkits (`toolkits.py`)

Pre-configured tool bundles for common use cases.

**Available Toolkits:**
- `MathsToolkit`: Complete mathematical operations bundle
- `MetaSearchToolkit`: Web search capabilities
- `VectorIndexToolkit`: Vector indexing and search bundle

## Installation

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- OpenRouter API key (for some features)

### Setup

1. **Install dependencies**
   ```bash
   # From project root
   poetry install
   ```

2. **Configure environment**
   ```bash
   # Edit .env with required API keys
   cp .env.example .env
   OPENROUTER_API_KEY="your-api-key"
   SEARXNG_BASE_URL="http://localhost:8080"
   ```

3. **Set PYTHONPATH**
   ```bash
   export PYTHONPATH=~/projects/simple_react_agent/src
   ```

## Basic Usage

### Using Individual Tools

```python
from struct_agent.tools.maths_tools import make_calc_tool
from struct_agent.tools.searxng_tools import make_searxng_search_tool
from struct_agent.tools.leann_tools import make_leann_add_text_tool

# Create tools
calc_tool = make_calc_tool()
search_tool = make_searxng_search_tool()
add_text_tool = make_leann_add_text_tool()

# Use tools
result = calc_tool.handler({"expression": "2 + 2 * 3"})
print(f"Calculation result: {result}")
```

### Using Toolkits

```python
from struct_agent.tools import MathsToolkit, MetaSearchToolkit, VectorIndexToolkit

# Create toolkits
math_kit = MathsToolkit()
search_kit = MetaSearchToolkit()
vector_kit = VectorIndexToolkit()

# Access tools from toolkits
for tool in math_kit:
    print(f"Available math tool: {tool.name}")

# Use tools from toolkits
calc_result = math_kit.tools[0].handler({"expression": "sin(π/2)"})
print(f"Sine result: {calc_result}")
```

### Running as Modules

```bash
# With PYTHONPATH set
export PYTHONPATH=~/projects/simple_react_agent/src

# Run mathematical tools
poetry run python -m struct_agent.tools.maths_tools

# Run search tools
poetry run python -m struct_agent.tools.searxng_tools

# Run LEANN tools
poetry run python -m struct_agent.tools.leann_tools

# Run toolkits
poetry run python -m struct_agent.tools.toolkits
```

## Tool-Specific Usage

### Mathematical Tools

```python
from struct_agent.tools.maths_tools import (
    make_calc_tool, make_sympy_solve_equation_tool,
    make_sympy_differentiate_tool
)

# Basic calculation
calc_tool = make_calc_tool()
result = calc_tool.handler({"expression": "(2 + 3) * 4"})
print(f"Result: {result['value']}")

# Solve equations
equation_tool = make_sympy_solve_equation_tool()
result = equation_tool.handler({
    "equations": ["x + y = 10", "x - y = 2"],
    "variables": ["x", "y"]
})
print(f"Solution: {result['solution']}")

# Calculus operations
diff_tool = make_sympy_differentiate_tool()
result = diff_tool.handler({
    "expression": "x**2 + 3*x + 1",
    "variable": "x"
})
print(f"Derivative: {result['result']}")
```

### SearXNG Search Tools
```python
from struct_agent.tools.searxng_tools import make_searxng_search_tool

# Create search tool
search_tool = make_searxng_search_tool()

# Perform search
result = search_tool.handler({
    "queries": ["Python programming", "machine learning"],
    "max_results": 5
})

print(f"Found {len(result['results'])} results")
for item in result['results'][:3]:
    print(f"- {item['title']}: {item['url']}")
```

### LEANN Vector Tools

```python
from struct_agent.tools.leann_tools import (
    make_leann_add_text_tool, make_leann_search_tool
)

# Add text to index
add_tool = make_leann_add_text_tool()
result = add_tool.handler({
    "text_content": "Python is a popular programming language for AI and machine learning."
})
print(f"Added to index: {result['status']}")

# Search the index
search_tool = make_leann_search_tool()
result = search_tool.handler({
    "query": "programming languages for AI",
    "top_k": 3
})

print(f"Search results: {len(result['results'])}")
for item in result['results']:
    print(f"- Score: {item['score']:.2f}: {item['text'][:100]}...")
```

## Advanced Configuration

### Custom Tool Integration

Check ```src/struct_agent/instructor_based/README.md``` section ```Instructor Based Module/Basic Usage/Custom Tool Integration```

## API Reference

### Mathematical Tools

#### CalcArgs / CalcResponse
```python
class CalcArgs(BaseModel):
    expression: str = Field(..., description="Arithmetic expression to evaluate")

class CalcResponse(BaseModel):
    expression: str = Field(..., description="The evaluated expression")
    value: float = Field(..., description="The result of the calculation")
```

#### Equation Solving
```python
class SolveEquationArgs(BaseModel):
    equations: List[str] = Field(..., description="Equations to solve")
    variables: List[str] = Field(..., description="Variables to solve for")

class SolveEquationResponse(BaseModel):
    solution: str = Field(..., description="The solution to the equations")
```

### SearXNG Tools

#### SearXNGSearchArgs
```python
class SearXNGSearchArgs(BaseModel):
    queries: List[str] = Field(..., description="Search queries to execute")
    category: Optional[str] = Field(None, description="Search category filter")
    max_results: int = Field(10, description="Maximum results per query")
```

### LEANN Tools

#### AddTextArgs / AddTextResponse
```python
class AddTextArgs(BaseModel):
    text_content: str = Field(..., description="Text to add to index")

class AddTextResponse(BaseModel):
    status: str = Field(..., description="Indexing status")
    text_id: Optional[str] = Field(None, description="Unique text identifier")
```

## Running Standalone Examples

### Search Tool Examples
```bash
export PYTHONPATH=~/projects/simple_react_agent/src
poetry run python -m struct_agent.tools.searxng_tools
```

## Dependencies

### Core Dependencies
- `pydantic>=2.7`: Data validation and serialization
- `python-dotenv>=1.0`: Environment variable management

### Tool-Specific Dependencies
- `sympy`: Symbolic mathematics (maths_tools)
- `beautifulsoup4`: HTML parsing (searxng_tools)
- `ddgs`: DuckDuckGo search (searxng_tools)
- `leann>=0.3.4`: Vector indexing and search (leann_tools)
- `aiohttp`: Async HTTP client (searxng_tools)
- `requests`: HTTP requests (various tools)

## Error Handling

All tools include comprehensive error handling:

### Common Errors
- `ValueError`: Invalid input parameters
- `SyntaxError`: Malformed expressions (math tools)
- `ConnectionError`: API connection failures
- `RuntimeError`: Tool execution failures

### Error Responses
```python
# Tools return error information
result = tool.handler({"invalid": "input"})
if "error" in result:
    print(f"Tool error: {result['error']}")
    print(f"Message: {result['message']}")
```

## Performance Considerations

### Mathematical Tools
- Complex symbolic operations can be CPU intensive
- Large equations may take significant time to solve
- Matrix operations scale with matrix size

### Search Tools
- Network latency affects search performance
- Multiple queries are executed in parallel where possible
- Result limits help manage response size

### LEANN Tools
- Index size affects search performance
- First-time index creation may take time
- Memory usage scales with index size

## Development

### Adding New Tools

1. Create new tool file in `src/struct_agent/tools/`
2. Implement handler function and Pydantic models
3. Create tool specification using `ToolSpec`
4. Add to appropriate toolkit or create new one
5. Update `__init__.py` exports
6. Add tests

### Code Style

- Follow PEP 8
- Use type hints consistently
- Keep handlers focused and single-purpose
- Use Pydantic for all input validation
- Include comprehensive docstrings
- Handle errors gracefully
- Keep module size under 300 lines

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure PYTHONPATH is set correctly
   ```bash
   export PYTHONPATH=~/projects/simple_react_agent/src
   ```

2. **Missing Dependencies**: Install required packages
   ```bash
   poetry install
   ```

3. **SearXNG Connection**: Verify SearXNG instance is running
   ```bash
   # Test SearXNG connection
   curl http://localhost:8080/
   ```

4. **LEANN Index Issues**: Check index directory permissions
   ```bash
   # Check index directory
   ls -la /path/to/leann_index/
   ```

5. **Math Expression Errors**: Validate expression syntax
   ```python
   # Test expression parsing
   import ast
   ast.parse("2 + 2", mode="eval")
   ```

## Examples

See the main project README for complete examples of using tools with the ReasoningAgent, or run individual tools as modules:
```bash
export PYTHONPATH=~/projects/simple_react_agent/src
poetry run python -m struct_agent.tools.searxng_tools
```