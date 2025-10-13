from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type
from pydantic import BaseModel

class ToolSpec(BaseModel):
    """Describe a single tool the agent may call."""

    name: str
    description: str
    response_model: Optional[Type[BaseModel]] = None
    handler: Callable[[Dict[str, Any]], Any]
    args_model: Optional[Type[BaseModel]] = None
    parameters: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def model_class(self) -> Type[BaseModel]:
        """Return whichever model definition the tool exposes."""
        model = self.response_model or self.args_model
        if model is None:
            raise ValueError("ToolSpec requires response_model or args_model")
        return model

__all__ = ["ToolSpec"]