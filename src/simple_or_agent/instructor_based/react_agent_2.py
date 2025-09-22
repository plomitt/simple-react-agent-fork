from __future__ import annotations

import ast
import operator as op
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Literal, Union
import os

import instructor
from pydantic import BaseModel

from simple_or_agent.instructor_based import build_instructor_client
from simple_or_agent.instructor_based.prompt_manager import (
    DEFAULT_REACT_SYSTEM_PROMPT,
    DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class FinalAnswer(BaseModel):
    """Final answer response"""
    type: Literal["final"] = "final"
    content: str


class ThinkResponse(BaseModel):
    """Thought response"""
    type: Literal["think"] = "think"
    content: str


class ObservationResponse(BaseModel):
    """Observation response"""
    type: Literal["observation"] = "observation"
    content: str

class ToolSpec(BaseModel):
    name: str
    description: str
    response_model: Type[BaseModel]
    handler: Callable[[Dict[str, Any]], Any]


class ReActAgent:
    """Minimal ReAct loop that works."""
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT_TEMPLATE,
        temperature: float = 0.1,
        max_steps: int = 6,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        self.client = build_instructor_client(model_id=model, url=base_url, key=api_key)
        self.model_id = model or "qwen/qwen3-next-80b-a3b-instruct"
        self.temperature = temperature
        self.max_steps = max(1, int(max_steps))
        self._tools: Dict[str, ToolSpec] = {}
        self._system_prompt_template = system_prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._render_system_prompt()}
        ]

    def add_tool(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool
        self._refresh_system_prompt()

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)
        self._refresh_system_prompt()

    def _refresh_system_prompt(self) -> None:
        """Update the stored system prompt so the model sees the latest tool list."""
        prompt = self._render_system_prompt()
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = prompt
            return
        # Insert a fresh system message if the log somehow lost the original.
        self.messages.insert(0, {"role": "system", "content": prompt})

    def _render_system_prompt(self) -> str:
        """Render the prompt template with the current tool block."""
        return render_system_prompt(self._system_prompt_template, self._tools)

    def think(self) -> Union[ThinkResponse, FinalAnswer]:
        self.messages.append({"role": "user", "content": "Think about the current question or observation and decide what to do next. Can we answer the question with the information we have? If so, respond with FinalAnswer. If not, respond with ThinkResponse."})
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=Union[ThinkResponse, FinalAnswer],
        )
        
    def action(self) -> Tuple[str, BaseModel]:
        if not self._tools:
            raise RuntimeError("No tools registered for this agent")

        self.messages.append({
            "role": "user", 
            "content": "Choose and call an available tool based on the latest thoughts"
        })

        tool_models = [tool.response_model for tool in self._tools.values()]
        available_tool_response_models = tool_models[0] if len(tool_models) == 1 else Union[*tool_models]
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=available_tool_response_models,
        )

        # Identify which tool the language model implied by checking the response type.
        for tool_name, spec in self._tools.items():
            if isinstance(response, spec.response_model):
                return tool_name, response

        raise ValueError("Received tool payload that matches no registered tool")
       

    def observation(self) -> Dict[str, Any]:
        self.messages.append({"role": "user", "content": "Review the outcome of your recent Action. Use it to inform your next Thought."})
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.messages,
            response_model=ObservationResponse,
        )

    def run(self, prompt: str) -> Dict[str, Any]:
        """Run the ReAct loop until we reach a final answer or max steps."""
        if not prompt:
            raise ValueError("Empty prompt")
        if self.client is None:
            raise RuntimeError("Instructor client is not configured")

        self.messages.append({"role": "user", "content": prompt})

        for _ in range(self.max_steps):
            # ReAct step order: Thought -> Action -> Observation.
            think_response = self.think()

            if think_response.type == "final":
                self.messages.append({"role": "assistant", "content": think_response.content})
                return think_response.content

            if think_response.type != "think":
                raise ValueError("Invalid think response")

            self.messages.append({"role": "assistant", "content": think_response.content})

            if not self._tools:
                raise RuntimeError("No tools registered for this agent")

            tool_name, action_payload = self.action()
            payload_dict = action_payload.model_dump()
            self.messages.append({"role": "assistant", "content": f"Action: {tool_name} -> {payload_dict}"})

            handler = self._tools[tool_name].handler
            tool_call_result = handler(payload_dict)
            self.messages.append({"role": "tool", "content": tool_call_result})

            observation_response = self.observation()
            self.messages.append({"role": "assistant", "content": observation_response.content})

        raise RuntimeError("Reached max steps without a final answer")

      

class CalcArgs(BaseModel):
    expression: str

OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def _eval_expression(node: ast.AST) -> float:
    """Evaluate a safe arithmetic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval_expression(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left = _eval_expression(node.left)
        right = _eval_expression(node.right)
        return OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


def calculate(raw_args: Dict[str, Any]) -> Dict[str, Any]:
    print(f"CALC_ARGS_RAW: {raw_args}")
    args = CalcArgs(**raw_args)
    parsed = ast.parse(args.expression, mode="eval")
    value = _eval_expression(parsed.body)
    return {"expression": args.expression, "value": value}


class CalculateResponse(BaseModel):
    expression: str
    value: float


if __name__ == "__main__":
    agent = ReActAgent(model='openai/gpt-oss-20b')
    calculate_tool = ToolSpec(name="calculate", description="Evaluate a mathematical expression.", response_model=CalcArgs, handler=calculate)
    agent.add_tool(calculate_tool)
    agent.run("Find the value of 42 + 3")
    print(agent.messages)
    