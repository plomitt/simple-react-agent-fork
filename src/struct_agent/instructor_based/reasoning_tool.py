from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from struct_agent.instructor_based.reasoning_prompt import get_reasoning_prompt
from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec

class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"

class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="A concise title summarizing the step's purpose")
    action: Optional[str] = Field(None, description="The action derived from this step. Talk in first person like I will ...")
    result: Optional[str] = Field(None, description="The result of executing the action. Talk in first person like I did this and got ... ")
    reasoning: Optional[str] = Field(None, description="The thought process and considerations behind this step")
    next_action: Optional[NextAction] = Field(None, description="Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer")
    confidence: Optional[float] = Field(None, description="Confidence score for this step (0.0 to 1.0)")

class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="A list of reasoning steps")

class ReasoningToolArgs(BaseModel):
    """Inputs for the reasoning tool."""
    query: str = Field(..., description="The user query or problem to reason about")
    history: List[str] = Field(default_factory=list, description="Previous reasoning steps or context")

class ReasoningToolResponse(BaseModel):
    """Response model for reasoning tool output."""
    reasoning_steps: Optional[List[ReasoningStep]] = Field(None, description="The processed reasoning steps after filtering")
    final_answer: Optional[str] = Field(None, description="The final result from the last reasoning step")
    error: Optional[str] = Field(None, description="Error message if reasoning failed")

def get_reasoning_messages(query, history):
    reasoning_prompt = get_reasoning_prompt()

    return [
        {"role": "system", "content": reasoning_prompt},
        {"role": "user", "content": f"User query: {query}"},
        {"role": "system", "content": f"History:\n{'\n'.join(history)}"},
    ]

def filter_reasoning_steps_after_reset(reasoning_steps: List[ReasoningStep]) -> List[ReasoningStep]:
    last_reset_index = -1

    for i, step in enumerate(reasoning_steps):
        if step.next_action == NextAction.RESET:
            last_reset_index = i

    if last_reset_index >= 0:
        return reasoning_steps[last_reset_index + 1:]
    return reasoning_steps

def make_reasoning_tool() -> ToolSpec:
    """Create a reasoning tool that breaks down complex problems into step-by-step reasoning."""

    def handler(args: Dict[str, Any]) -> ReasoningToolResponse:
        try:
            parsed_args = ReasoningToolArgs(**args)

            client = build_client()
            messages = get_reasoning_messages(parsed_args.query, parsed_args.history)

            reasoning: ReasoningSteps = client.chat.completions.create(
                messages=messages,
                response_model=ReasoningSteps,
            )

            if not reasoning.reasoning_steps:
                return ReasoningToolResponse(
                    error="No reasoning steps generated"
                )

            filtered_steps = filter_reasoning_steps_after_reset(reasoning.reasoning_steps)
            last_step = filtered_steps[-1]

            if not last_step.result:
                return ReasoningToolResponse(
                    error="No result for the last step"
                )

            return ReasoningToolResponse(
                reasoning_steps=filtered_steps,
                final_answer=last_step.result,
            )

        except Exception as e:
            return ReasoningToolResponse(
                reasoning_steps=[],
                final_answer="",
                success=False,
                error=f"Reasoning tool failed: {str(e)}"
            )

    return ToolSpec(
        name="reasoning_tool",
        description="Breaks down complex queries into step-by-step reasoning with clear actions and conclusions.",
        args_model=ReasoningToolArgs,
        handler=handler,
        parameters={
            "query": "the user query or problem to reason about",
            "history": "list of previous reasoning steps or context"
        }
    )

__all__ = [
    "make_reasoning_tool",
    "ReasoningStep",
    "ReasoningSteps",
    "NextAction",
    "ReasoningToolArgs",
    "ReasoningToolResponse",
]