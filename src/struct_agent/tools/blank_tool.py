from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Dict, Any

from struct_agent.instructor_based import ToolSpec

class BlankTool(BaseModel):
    """Inputs for the blank tool - no actual parameters needed."""
    message: str = Field(default="", description="Optional message (ignored but kept for consistency)")

def make_blank_tool() -> ToolSpec:
    """Create a blank tool that does nothing but returns success."""

    def handler(args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the blank tool call - simply returns success."""
        return {"message": "No action taken."}

    return ToolSpec(
        name="blank_tool",
        description="A blank tool that does nothing and returns success. Used when no actual tool functionality is needed.",
        args_model=BlankTool,
        handler=handler,
        parameters={
            "message": "Optional message parameter (ignored but kept for API consistency)"
        }
    )


__all__ = ["make_blank_tool"]