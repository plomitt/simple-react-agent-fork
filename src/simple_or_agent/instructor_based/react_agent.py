from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type
import os
import json

import instructor
from openai import OpenAI
from pydantic import BaseModel, create_model
from typing import Literal

from simple_or_agent.instructor_based import create_instructor_client

# ========== ReAct primitives (schemas) ==========

class FinalAnswer(BaseModel):
    """Stop condition: produce a final user-facing answer."""
    kind: Literal["final"] = "final"
    answer: str


@dataclass
class ToolSpec:
    """
    Describe a tool:
      - name: unique tool identifier (used as `kind`)
      - description: for prompting (optional—kept for clarity)
      - args_model: pydantic model for tool arguments
      - handler: function(dict) -> any  (your side-effect / API call)
    """
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[[Dict[str, Any]], Any]
    action_model: Optional[Type[BaseModel]] = None  # auto-generated: {kind, thought, params}

    def build_action_model(self) -> Type[BaseModel]:
        """Create a discriminated action model: {kind=<tool name>, thought:str, params:<args_model>}"""
        if self.action_model is not None:
            return self.action_model
        self.action_model = create_model(
            f"{self.name.title().replace('_','')}Action",
            kind=(Literal[self.name], self.name),
            thought=(str, ...),                 # ReAct "Thought" (private; don't show to end user)
            params=(self.args_model, ...),      # structured tool arguments
        )
        return self.action_model


# ========== Minimal ReAct Agent ==========

DEFAULT_REACT_SYSTEM_PROMPT = (
    "You are a ReAct-style assistant.\n"
    "Follow an internal loop: Thought → Action → Observation.\n"
    "• Thought: brief private notes for yourself (do NOT reveal in the final answer).\n"
    "• Action: when external info is needed, choose exactly one tool action using the provided schemas.\n"
    "• Observation: you'll receive a tool result via a message with role='tool'. Use it to continue.\n"
    "Stop when you can answer, returning FinalAnswer.\n"
    "Keep the final answer concise and helpful."
)


class ReActAgent:
    """
    A tiny ReAct agent:
      - At each step, the model returns either FinalAnswer or a tool action (with Thought + params).
      - If a tool is chosen, we execute it, feed back an Observation (role='tool'), and loop.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: str = DEFAULT_REACT_SYSTEM_PROMPT,
        client: Optional[OpenAI] = None,
        temperature: float = 0.1,
        max_steps: int = 6,
    ) -> None:
        self.model = model or os.getenv("MODEL_ID", "qwen/qwen3-next-80b-a3b-thinking")
        self.client = client or create_instructor_client(self.model)
        self.temperature = temperature
        self.max_steps = max(1, int(max_steps))
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._tools: Dict[str, ToolSpec] = {}

    # ---- Tool registry ----
    def add_tool(self, tool: ToolSpec) -> None:
        tool.build_action_model()
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    # ---- Core loop ----
    def ask(self, prompt: str) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("Empty prompt")
        self.messages.append({"role": "user", "content": prompt})

        # Build dynamic union: FinalAnswer | ActionTool1 | ActionTool2 | ...
        action_models = [t.action_model for t in self._tools.values()]
        Decision = FinalAnswer
        for am in action_models:
            Decision = Decision | am  # PEP 604 unions

        print(Decision)

        for step in range(self.max_steps):
            # Step: Thought or Final
            decision = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                response_model=Decision,
            )

            # If FinalAnswer → stop
            if isinstance(decision, FinalAnswer) or getattr(decision, "kind", None) == "final":
                self.messages.append({"role": "assistant", "content": decision.answer})
                return {"answer": decision.answer, "messages": self.messages}

            # Otherwise it's a tool action
            tool_name: str = decision.kind
            spec = self._tools.get(tool_name)

            # Prepare minimal action trace for the model (no internal Thought revealed to end user)
            # We do NOT display the 'thought' to the user. It's only kept in the context.
            action_trace = {
                "action": {"name": tool_name, "params": _model_dump(decision.params)},
                "note": "internal-thought-hidden",
            }
            self.messages.append({"role": "assistant", "content": json.dumps(action_trace, ensure_ascii=False)})

            # Execute tool
            if spec is None:
                observation = {"error": f"unknown tool '{tool_name}'"}
            else:
                try:
                    args = decision.params.model_dump() if isinstance(decision.params, BaseModel) else dict(decision.params or {})
                    out = spec.handler(args)
                    observation = out if isinstance(out, (dict, list, str, int, float, bool)) else {"result": str(out)}
                except Exception as e:
                    observation = {"error": str(e)}

            # Feed Observation
            self.messages.append({"role": "tool", "content": json.dumps(observation, ensure_ascii=False)})

            # Loop continues: the model sees Observation and can choose another Action or FinalAnswer

        # Fallback if max_steps reached
        self.messages.append({"role": "assistant", "content": "Sorry, I couldn't complete that within the step limit."})
        return {"answer": "Sorry, I couldn't complete that within the step limit.", "messages": self.messages}


def _model_dump(x: Any) -> Any:
    if isinstance(x, BaseModel):
        return x.model_dump()
    return x


# ========== Tiny example (optional) ==========

# Example tool: a toy calculator (safe-ish numeric eval of + - * / and parentheses)
# Replace with your real tools (web search, DB lookup, etc.)

if __name__ == "__main__":
    from typing import Union
    import ast
    import operator as op

    class CalcArgs(BaseModel):
        expr: str

    OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Unsupported expression")

    def calc_handler(args: Dict[str, Any]) -> Dict[str, Union[str, float]]:
        expr = args.get("expr", "")
        value = _eval(ast.parse(expr, mode="eval").body)
        return {"expr": expr, "value": value}

    agent = ReActAgent(model='google/gemini-2.5-flash-lite')
    agent.add_tool(ToolSpec(
        name="calc",
        description="Evaluate simple arithmetic expressions.",
        args_model=CalcArgs,
        handler=calc_handler,
    ))

    # Ask something that may require tool use:
    result = agent.ask("What is (2+3)*4? Then tell me if it's greater than 15.")
    print(result["answer"])
