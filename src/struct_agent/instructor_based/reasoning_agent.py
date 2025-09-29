from __future__ import annotations
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from enum import Enum
import json

from struct_agent.instructor_based.client_manager import build_client, resolve_model
from struct_agent.instructor_based.prompt_manager import get_system_prompt
from struct_agent.instructor_based.tool_manager import ToolRegistry, ToolSpec

load_dotenv()

class ObservationResponse(BaseModel):
    """Observation response"""
    observation: str

class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"

class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="Short step title.")
    action: Optional[str] = Field(None, description="Planned action written in first person.")
    result: Optional[str] = Field(None, description="Outcome summary for the step.")
    reasoning: Optional[str] = Field(None, description="Why this step matters.")
    tool: Optional[str] = Field(None, description="Tool name when you need external help.")
    next_action: Optional[NextAction] = Field(None, description="continue, validate, final_answer, or reset.")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0.")

class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="Ordered reasoning steps.")

class MaybeToolCall(BaseModel):
    result: Optional[Dict[str, Any]] = Field(default=None, description="Tool arguments when the call is valid.")
    error: bool = Field(default=False, description="True when the tool call failed.")
    message: Optional[str] = Field(default=None, description="Explanation of what went wrong.")

class ReasoningAgent:
    def __init__(
        self,
        *,
        min_steps: int = 1,
        max_steps: int = 10,
        temperature: float = 0.1,
        thinking_model: Optional[str] = None,
        action_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_lmstudio: bool = False
    ) -> None:
        self.model_id = thinking_model or resolve_model(use_lmstudio)
        self.action_model = action_model or self.model_id
        self.client = build_client(self.model_id, api_key, base_url, use_lmstudio)
        self.temperature = temperature
        self.min_steps = max(1, int(min_steps))
        self.max_steps = max(self.min_steps, int(max_steps))
        self._tools = ToolRegistry()
        self._prompt_template = get_system_prompt(self.min_steps, self.max_steps)
        self._system_prompt = self._compose_system_prompt()
        self._steps: List[ReasoningStep] = []

    def _compose_system_prompt(self) -> str:
        base = self._prompt_template.strip()
        catalog = self._tools.tool_names_and_descriptions()
        if catalog:
            base = f"{base}\n\nAvailable tools:\n{catalog}\n\nSet the `tool` field only to one of these names."
        return base

    def _history_text(self) -> str:
        if not self._steps:
            return "No steps recorded yet."
        blocks: List[str] = []
        for index, step in enumerate(self._steps, start=1):
            parts = [f"Step {index}: {step.title or 'Untitled step'}"]
            if step.action:
                parts.append(f"Action: {step.action}")
            if step.reasoning:
                parts.append(f"Reasoning: {step.reasoning}")
            if step.tool:
                parts.append(f"Tool: {step.tool}")
            if step.result:
                parts.append(f"Result: {step.result}")
            if step.next_action:
                choice = step.next_action.value if isinstance(step.next_action, NextAction) else str(step.next_action)
                parts.append(f"Next Action: {choice}")
            if step.confidence is not None:
                parts.append(f"Confidence: {step.confidence}")
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def _user_message(self, prompt: str, history: str, *blocks: str) -> str:
        lines = [prompt, "", "Reasoning history:", history]
        for block in blocks:
            if block:
                lines.extend(["", block])
        return "\n".join(lines)

    def _chat(self, user_content: str, response_model: Type[BaseModel]) -> BaseModel:
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_model=response_model,
            temperature=self.temperature,
            extra_body={"provider": {"require_parameters": True}}
        )

    def _stringify(self, value: Any) -> str:
        if isinstance(value, BaseModel):
            return json.dumps(value.model_dump(), indent=2, sort_keys=True)
        if isinstance(value, dict):
            return json.dumps(value, indent=2, sort_keys=True)
        if isinstance(value, (list, tuple)):
            return json.dumps(list(value), indent=2, sort_keys=True)
        return str(value)

    def add_tool(self, tool: ToolSpec) -> None:
        self._tools.add(tool)
        self._system_prompt = self._compose_system_prompt()

    def remove_tool(self, name: str) -> None:
        self._tools.remove(name)
        self._system_prompt = self._compose_system_prompt()

    def run(self, prompt: str) -> ReasoningSteps:
        if not prompt:
            raise ValueError("Prompt must not be empty.")
        self._system_prompt = self._compose_system_prompt()
        self._steps = []

        for _ in range(self.max_steps * 2):
            history = self._history_text()
            directive = (
                "Return the next ReasoningStep JSON object. Provide exactly one step. "
                "If you need to run a tool set `tool` to its exact name and leave `result` empty until the tool is observed."
            )
            step = self._chat(
                self._user_message(prompt, history, directive),
                ReasoningStep,
            )
            self._steps.append(step)
            if step.next_action == NextAction.RESET:
                self._steps.clear()
                continue
            if len(self._steps) > self.max_steps:
                raise RuntimeError("Exceeded configured max_steps before final answer.")
            if step.tool:  # Ask the model for structured tool args and run the handler.
                name = step.tool.strip()
                if not name:
                    raise ValueError("Tool field is present but empty.")
                if not self._tools.has_tools():
                    raise RuntimeError("A tool was requested but no tools are registered.")
                tools_map = self._tools.as_mapping()
                if name not in tools_map:
                    known = ", ".join(self._tools.tool_names()) or "no tools"
                    raise ValueError(f"Unknown tool '{name}'. Known tools: {known}")
                spec = tools_map[name]
                maybe = self._chat(
                    self._user_message(
                        prompt,
                        self._history_text(),
                        (
                            f"Return a MaybeToolCall JSON object for the `{name}` tool. "
                            "Populate `result` with the exact arguments when the call is valid. "
                            "If the call cannot proceed, set `error` to true and explain why in `message`."
                        ),
                    ),
                    MaybeToolCall,
                )
                step.tool = name
                if maybe.error or maybe.result is None:
                    failure_note = maybe.message or "Tool call failed without an explanation."
                    failure_prompt = self._user_message(
                        prompt,
                        self._history_text(),
                        f"Tool `{name}` reported an error.",
                        f"Error message: {failure_note}",
                        f"Partial payload: {self._stringify(maybe.result) if maybe.result else 'None available'}",
                        "Explain how this affects the plan and suggest a next step.",
                    )
                    observation = self._chat(
                        failure_prompt,
                        ObservationResponse,
                    )
                    step.result = observation.observation
                    continue
                validated_args = spec.model_class()(**(maybe.result or {}))
                tool_output = spec.handler(validated_args.model_dump())
                step.result = self._chat(
                    self._user_message(
                        prompt,
                        self._history_text(),
                        f"You called `{name}`.",
                        f"Tool args:\n{self._stringify(validated_args.model_dump())}",
                        f"Tool output:\n{self._stringify(tool_output)}",
                        "Summarize what this output means and how it affects the plan.",
                    ),
                    ObservationResponse,
                ).observation

            if step.next_action == NextAction.FINAL_ANSWER and len(self._steps) >= self.min_steps:
                break

        if not self._steps:
            raise RuntimeError("No reasoning steps were produced.")
        if self._steps[-1].next_action != NextAction.FINAL_ANSWER:
            raise RuntimeError("Reasoning agent stopped without a final answer.")
        return ReasoningSteps(reasoning_steps=self._steps)

def run_reasoning_agent(
    prompt: str,
    *,
    tools: Optional[List[ToolSpec]] = None,
    **agent_kwargs: Any,
) -> ReasoningSteps:
    agent = ReasoningAgent(**agent_kwargs)
    for tool in tools or []:
        agent.add_tool(tool)
    return agent.run(prompt)

__all__ = ["ReasoningAgent", "run_reasoning_agent"]

if __name__ == "__main__":
    from struct_agent.tools import MetaSearchToolkit, VectorIndexToolkit, MathsToolkit
    
    demo_agent = ReasoningAgent()

    toolkits = [MetaSearchToolkit(), VectorIndexToolkit(), MathsToolkit()]
    for toolkit in toolkits:
        for tool in toolkit:
            demo_agent.add_tool(tool)

    # demo_steps = demo_agent.run("Who is the current CEO of OpenAI? Where did that person go to university? Look up their most recent public talk, including title and date.")
    # demo_steps = demo_agent.run("what is the exact value of '((7 * (3 + 5) - (12 / 4)) * (2 ** 3) + (19 - (6 * 2))) / (4 + (15 - 13) * 2)'?")
    demo_steps = demo_agent.run("What is 2 + 2?")
    print("Final answer:", demo_steps.reasoning_steps[-1].result)