from enum import Enum
import os
from typing import List, Optional, Dict, Any, Tuple
from __future__ import annotations
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from instructor import Mode
from pathlib import Path
import sys

from struct_agent.instructor_based import simple_client
from struct_agent.instructor_based.reasoning_prompt import get_system_prompt
from struct_agent.instructor_based import openrouter_client
from struct_agent.instructor_based.tool_manager import ToolRegistry, ToolSpec
from struct_agent.tools.searxng_tools import make_searxng_search_tool

if __package__ in {None, ''}:
    # Ensure direct execution can resolve the src/ package namespace.
    project_src = Path(__file__).resolve().parent.parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))

load_dotenv()


class NextAction(str, Enum):
    CONTINUE = "continue"
    VALIDATE = "validate"
    FINAL_ANSWER = "final_answer"
    RESET = "reset"


class ReasoningStep(BaseModel):
    title: Optional[str] = Field(None, description="A concise title summarizing the step's purpose")
    action: Optional[str] = Field(
        None, description="The action derived from this step. Talk in first person like I will ..."
    )
    result: Optional[str] = Field(
        None, description="The result of executing the action. Talk in first person like I did this and got ... "
    )
    reasoning: Optional[str] = Field(None, description="The thought process and considerations behind this step")
    next_action: Optional[NextAction] = Field(
        None,
        description="Indicates whether to continue reasoning, validate the provided result, or confirm that the result is the final answer",
    )
    confidence: Optional[float] = Field(None, description="Confidence score for this step (0.0 to 1.0)")


class ReasoningSteps(BaseModel):
    reasoning_steps: List[ReasoningStep] = Field(..., description="A list of reasoning steps")


class ThinkResponse(BaseModel):
    """Thought response"""
    is_final: bool
    thoughts: str


class ObservationResponse(BaseModel):
    """Observation response"""
    content: str


def _resolve_api_key() -> Optional[str]:
    """Read the preferred API key from the environment."""
    return openrouter_client.resolve_api_key_from_env()


def _derive_model_id(model: Optional[str], provider_id: Optional[str]) -> str:
    """Return the completion model id based on the hints provided."""
    if model:
        return model
    if provider_id:
        if openrouter_client.is_openrouter(provider_id):
            return openrouter_client.provider_model_hint(provider_id)
        if provider_id.startswith("openai/"):
            return provider_id.split("/", 1)[1]
        return provider_id
    return "qwen/qwen3-next-80b-a3b-instruct"


def _build_client(
    api_key: str,
    provider_id: str,
    mode: Optional[Mode],
) -> Any:
    """Create an Instructor client using the OpenRouter settings."""
    return openrouter_client.build_client(
        api_key=api_key,
        provider_id=provider_id,
        mode=mode,
    )


class HybridAgent:
    """Hybrid Reasoning-ReAct agent that combines comprehensive reasoning with tool chaining."""

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
 
        self.model_id = resolved_model
        self.temperature = temperature
        self.max_steps = max_steps
        self._tools = ToolRegistry()

        # Add SearXNG search tool
        self._tools.add(make_searxng_search_tool())

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": get_system_prompt()}
        ]

    def think(self, context: str) -> ReasoningSteps:
        """Generate comprehensive reasoning steps using the reasoning agent approach."""
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": context}
        ]

        return self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            response_model=ReasoningSteps,
        )

    def act(self, reasoning_steps: ReasoningSteps) -> Tuple[str, ToolSpec, BaseModel]:
        """Execute the first tool action found in reasoning steps."""
        if not self._tools.has_tools():
            raise RuntimeError("No tools registered for this agent")

        # Find first step with an action that requires a tool
        for step in reasoning_steps.reasoning_steps:
            if step.action and ("search" in step.action.lower() or "look up" in step.action.lower()):
                # Create search query from action
                action_content = step.action.lower()
                queries = []

                # Extract search terms from action
                if "search for" in action_content:
                    query = action_content.split("search for")[-1].strip()
                    queries.append(query)
                elif "look up" in action_content:
                    query = action_content.split("look up")[-1].strip()
                    queries.append(query)
                else:
                    # Fallback: use action as query
                    queries.append(step.action)

                # Execute search
                search_args = {"queries": queries, "max_results": 5}
                tool_name, tool_spec = "searxng_search", self._tools._tools["searxng_search"]

                # Create response model
                response_model = tool_spec.model_class()
                response_payload = response_model(**search_args)

                return tool_name, tool_spec, response_payload

        # If no tool action found, raise error
        raise RuntimeError("No tool action found in reasoning steps")

    def observe(self, tool_result: Any) -> str:
        """Process tool result and return observation."""
        if isinstance(tool_result, dict):
            if "error" in tool_result:
                return f"Tool execution failed: {tool_result['error']}"
            elif "results" in tool_result:
                results = tool_result["results"]
                if results:
                    return f"Found {len(results)} search results. Top result: {results[0].get('title', 'No title')} - {results[0].get('content', 'No content')[:200]}..."
                else:
                    return "No search results found."
            else:
                return f"Tool executed successfully. Result: {str(tool_result)[:200]}..."
        else:
            return f"Tool executed successfully. Result: {str(tool_result)[:200]}..."

    def run(self, prompt: str) -> str:
        """Run the hybrid agent loop."""
        if not prompt:
            raise ValueError("Empty prompt")
        if self.client is None:
            raise RuntimeError("Instructor client is not configured")

        iteration_messages = []
        iteration_messages.append(f'User: {prompt}\n')

        for step_num in range(self.max_steps):
            # Build context for thinking
            context = "".join(iteration_messages) + f"\n\nAvailable tools: {', '.join(self._tools.tool_names())}."
            if step_num > 0:
                context += f"\n\nThis is step {step_num + 1} of the conversation. Continue from where we left off."

            # Think: Generate comprehensive reasoning steps
            reasoning_steps = self.think(context)

            # Check if we have a final answer
            final_steps = [step for step in reasoning_steps.reasoning_steps if step.next_action == NextAction.FINAL_ANSWER]
            if final_steps and final_steps[-1].result:
                return final_steps[-1].result

            # Find first step that needs action
            action_steps = [step for step in reasoning_steps.reasoning_steps if step.action and step.next_action != NextAction.FINAL_ANSWER]
            if not action_steps:
                raise RuntimeError("No actionable steps found in reasoning")

            # Act: Execute first tool action
            try:
                tool_name, tool_spec, action_payload = self.act(reasoning_steps)
                func_result = tool_spec.handler(action_payload.model_dump())

                # Observe: Process tool result
                observation = self.observe(func_result)

                # Add to iteration messages
                iteration_messages.append(f'Thought: {action_steps[0].reasoning}\n')
                iteration_messages.append(f'Action: {action_steps[0].action}\n')
                iteration_messages.append(f'Result: {observation}\n')

            except Exception as e:
                iteration_messages.append(f'Error: Failed to execute action - {str(e)}\n')
                # Continue to next iteration with error information

        raise RuntimeError("Reached max steps without a final answer")


if __name__ == "__main__":
    # Basic testing
    agent = HybridAgent(model='qwen/qwen3-30b-a3b-instruct-2507')

    print(f"Registered tools: {agent._tools.tool_names()}")

    # Test queries
    test_queries = [
        "What is the current population of China?",
        "Search for information about artificial intelligence trends in 2024",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        try:
            answer = agent.run(query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
        print('='*50)