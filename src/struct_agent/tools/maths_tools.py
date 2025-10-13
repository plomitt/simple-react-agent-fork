from __future__ import annotations

from sympy import sympify, solve, simplify, diff, integrate, Matrix, expand, factor
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import operator as op
import ast

from struct_agent.instructor_based import ToolSpec

# --- BaseModel Classes for Mathematical Tools ---

class CalcArgs(BaseModel):
    """Inputs for the calculator tool."""
    expression: str = Field(..., description="Arithmetic expression to evaluate")

class CalcResponse(BaseModel):
    """Response from the calculator tool."""
    expression: str = Field(..., description="The evaluated expression")
    value: float = Field(..., description="The result of the calculation")

class SolveEquationArgs(BaseModel):
    """Inputs for solving equations."""
    equations: List[str] = Field(..., description="A list of equations (as strings) to be solved")
    variables: List[str] = Field(..., description="A list of variables (as strings) to solve for")

class SolveEquationResponse(BaseModel):
    """Response from solving equations."""
    solution: str = Field(..., description="The solution to the equations")

class ExpressionArgs(BaseModel):
    """Inputs for expression operations (simplify, expand, factor)."""
    expression: str = Field(..., description="The mathematical expression to process")

class ExpressionResponse(BaseModel):
    """Response from expression operations."""
    result: str = Field(..., description="The processed expression result")

class DifferentiateArgs(BaseModel):
    """Inputs for differentiation."""
    expression: str = Field(..., description="The expression to differentiate")
    variable: str = Field(..., description="The variable to differentiate with respect to")

class DifferentiateResponse(BaseModel):
    """Response from differentiation."""
    derivative: str = Field(..., description="The derivative of the expression")

class IntegrateArgs(BaseModel):
    """Inputs for integration."""
    expression: str = Field(..., description="The expression to integrate")
    variable: str = Field(..., description="The variable of integration")
    bounds: Optional[List[str]] = Field(None, description="Optional bounds for definite integration [lower, upper]")

class IntegrateResponse(BaseModel):
    """Response from integration."""
    integral: str = Field(..., description="The integral of the expression")

class MatrixOperationArgs(BaseModel):
    """Inputs for matrix operations."""
    matrix: List[List[float]] = Field(..., description="The matrix as a list of lists")
    operation: str = Field(..., description="The operation to perform: det, inv, eigenvals, rref")

class MatrixOperationResponse(BaseModel):
    """Response from matrix operations."""
    result: Optional[str] = Field(None, description="The result of the operation")
    rref_form: Optional[str] = Field(None, description="Reduced row echelon form (for rref operation)")
    pivots: Optional[str] = Field(None, description="Pivots (for rref operation)")

# --- Basic Calculator Tool ---

def make_calc_tool() -> ToolSpec:
    """Create a safe calculator tool using a tiny AST evaluator."""
    allowed = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            return allowed[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("unsupported expression")

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = CalcArgs(**args)
            expr = parsed_args.expression.strip()
            if not expr:
                return {"error": "empty_expression"}

            tree = ast.parse(expr, mode="eval")
            val = _eval(tree.body)  # type: ignore[arg-type]
            return {"expression": expr, "value": val}
        except Exception as e:
            return {"error": str(e)}

    return ToolSpec(
        name="calc",
        description="Evaluate basic arithmetic expression and return a JSON result",
        args_model=CalcArgs,
        response_model=CalcResponse,
        handler=handler,
        parameters={"expression": "arithmetic expression to evaluate"}
    )

# --- Core Algebra and Expression Tools ---

def make_sympy_solve_equation_tool() -> ToolSpec:
    """Solves algebraic equations."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = SolveEquationArgs(**args)
            equations = [sympify(eq) for eq in parsed_args.equations]
            variables = [sympify(var) for var in parsed_args.variables]
            solution = solve(equations, variables)
            return {"solution": str(solution)}
        except Exception as e:
            return {"error": f"SymPy solve failed: {e}"}

    return ToolSpec(
        name="sympy_solve_equation",
        description="Solve a single or a system of algebraic equations for a set of variables.",
        args_model=SolveEquationArgs,
        response_model=SolveEquationResponse,
        handler=handler,
        parameters={
            "equations": "list of equations to solve",
            "variables": "list of variables to solve for"
        },
    )

def make_sympy_simplify_expression_tool() -> ToolSpec:
    """Simplifies a mathematical expression."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = ExpressionArgs(**args)
            expression = sympify(parsed_args.expression)
            simplified_expr = simplify(expression)
            return {"result": str(simplified_expr)}
        except Exception as e:
            return {"error": f"SymPy simplify failed: {e}"}

    return ToolSpec(
        name="sympy_simplify_expression",
        description="Simplify a mathematical expression into its most readable and compact form.",
        args_model=ExpressionArgs,
        response_model=ExpressionResponse,
        handler=handler,
        parameters={"expression": "mathematical expression to simplify"}
    )

def make_sympy_expand_expression_tool() -> ToolSpec:
    """Expands a mathematical expression."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = ExpressionArgs(**args)
            expression = sympify(parsed_args.expression)
            expanded_expr = expand(expression)
            return {"result": str(expanded_expr)}
        except Exception as e:
            return {"error": f"SymPy expand failed: {e}"}

    return ToolSpec(
        name="sympy_expand_expression",
        description="Expand a mathematical expression by carrying out products and powers.",
        args_model=ExpressionArgs,
        response_model=ExpressionResponse,
        handler=handler,
        parameters={"expression": "mathematical expression to expand"}
    )

def make_sympy_factor_expression_tool() -> ToolSpec:
    """Factors a mathematical expression."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = ExpressionArgs(**args)
            expression = sympify(parsed_args.expression)
            factored_expr = factor(expression)
            return {"result": str(factored_expr)}
        except Exception as e:
            return {"error": f"SymPy factor failed: {e}"}

    return ToolSpec(
        name="sympy_factor_expression",
        description="Factor a polynomial into irreducible factors.",
        args_model=ExpressionArgs,
        response_model=ExpressionResponse,
        handler=handler,
        parameters={"expression": "mathematical expression to factor"}
    )

# --- Calculus Tools ---

def make_sympy_differentiate_tool() -> ToolSpec:
    """Differentiates an expression."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = DifferentiateArgs(**args)
            expression = sympify(parsed_args.expression)
            variable = sympify(parsed_args.variable)
            derivative = diff(expression, variable)
            return {"derivative": str(derivative)}
        except Exception as e:
            return {"error": f"SymPy differentiate failed: {e}"}

    return ToolSpec(
        name="sympy_differentiate",
        description="Compute the derivative of an expression with respect to a variable.",
        args_model=DifferentiateArgs,
        response_model=DifferentiateResponse,
        handler=handler,
        parameters={"expression": "expression to differentiate", "variable": "variable to differentiate with respect to"}
    )

def make_sympy_integrate_tool() -> ToolSpec:
    """Integrates an expression."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = IntegrateArgs(**args)
            expression = sympify(parsed_args.expression)
            variable = sympify(parsed_args.variable)

            if parsed_args.bounds:
                lower_bound, upper_bound = parsed_args.bounds
                integral = integrate(expression, (variable, sympify(lower_bound), sympify(upper_bound)))
            else:
                integral = integrate(expression, variable)

            return {"integral": str(integral)}
        except Exception as e:
            return {"error": f"SymPy integrate failed: {e}"}

    return ToolSpec(
        name="sympy_integrate",
        description="Compute the indefinite or definite integral of an expression.",
        args_model=IntegrateArgs,
        response_model=IntegrateResponse,
        handler=handler,
        parameters={"expression": "expression to integrate", "variable": "variable of integration", "bounds": "optional bounds for definite integration"}
    )

# --- Linear Algebra Tool ---

def make_sympy_matrix_operation_tool() -> ToolSpec:
    """Performs various matrix operations."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            parsed_args = MatrixOperationArgs(**args)
            M = Matrix(parsed_args.matrix)

            if parsed_args.operation == "det":
                result = M.det()
                return {"result": str(result)}
            elif parsed_args.operation == "inv":
                result = M.inv()
                return {"result": str(result)}
            elif parsed_args.operation == "eigenvals":
                result = M.eigenvals()
                return {"result": str(result)}
            elif parsed_args.operation == "rref":
                result, pivots = M.rref()
                return {"rref_form": str(result), "pivots": str(pivots)}
            else:
                return {"error": f"Unknown matrix operation: {parsed_args.operation}"}

        except Exception as e:
            return {"error": f"SymPy matrix operation failed: {e}"}

    return ToolSpec(
        name="sympy_matrix_operation",
        description="Perform a linear algebra operation on a matrix.",
        args_model=MatrixOperationArgs,
        response_model=MatrixOperationResponse,
        handler=handler,
        parameters={"matrix": "matrix as list of lists", "operation": "operation to perform (det, inv, eigenvals, rref)"}
    )

__all__ = [
    "make_calc_tool",
    "make_sympy_solve_equation_tool",
    "make_sympy_simplify_expression_tool",
    "make_sympy_expand_expression_tool",
    "make_sympy_factor_expression_tool",
    "make_sympy_differentiate_tool",
    "make_sympy_integrate_tool",
    "make_sympy_matrix_operation_tool",
]