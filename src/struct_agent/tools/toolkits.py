from __future__ import annotations

from struct_agent.tools.searxng_tools import *
from struct_agent.tools.maths_tools import *
from struct_agent.tools.leann_tools import *

class BaseToolkit:
    """A base class for toolkits to avoid repetitive code."""
    TOOL_FACTORIES = []

    def __init__(self):
        """Initializes the toolkit by creating all the tool instances."""
        self.tools = [factory() for factory in self.TOOL_FACTORIES]
        # The f-string provides a nice confirmation message when a toolkit is created
        print(f"✅ {self.__class__.__name__} initialized with {len(self.tools)} tools.")

    def __iter__(self):
        """Makes the toolkit instance iterable."""
        return iter(self.tools)
    
class MathsToolkit(BaseToolkit):
    """Toolkit for mathematical operations and symbolic algebra."""
    TOOL_FACTORIES = [
        make_calc_tool,
        make_sympy_solve_equation_tool,
        make_sympy_simplify_expression_tool,
        make_sympy_expand_expression_tool,
        make_sympy_factor_expression_tool,
        make_sympy_differentiate_tool,
        make_sympy_integrate_tool,
        make_sympy_matrix_operation_tool,
    ]

class MetaSearchToolkit(BaseToolkit):
    """Toolkit for performing meta-searches with SearXNG."""
    TOOL_FACTORIES = [
        make_searxng_search_tool,
    ]

class VectorIndexToolkit(BaseToolkit):
    """Toolkit for vector indexing and semantic search operations using LEANN."""
    TOOL_FACTORIES = [
        make_leann_add_text_tool,
        make_leann_search_tool,
        make_leann_chat_tool,
    ]

__all__ = [
    'BaseToolkit',
    'MathsToolkit',
    'MetaSearchToolkit',
    'VectorIndexToolkit',
]

if __name__ == "__main__":
    for toolkit in [MathsToolkit(), MetaSearchToolkit(), VectorIndexToolkit()]:
        print(f"{toolkit.__class__.__name__}")
        for tool in toolkit:
            print(f"    {tool.name}")