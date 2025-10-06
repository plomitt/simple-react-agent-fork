from __future__ import annotations

from typing import List, Type, Union, Dict, Set
from pydantic import BaseModel
from enum import Enum
from dotenv import load_dotenv
from pprint import pprint
import instructor
import hashlib

from struct_agent.instructor_based.reasoning_tool import ReasoningStep, ReasoningToolArgs, ReasoningToolResponse, make_reasoning_tool
from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec
from struct_agent.instructor_based.utils import merge_configs
from struct_agent.tools.searxng_tools import make_searxng_search_tool

load_dotenv()

class AgentState(Enum):
    """Agent execution states"""
    INITIALIZING = "initializing"
    REASONING = "reasoning"
    ACTING = "acting"
    COMPLETED = "completed"
    FAILED = "failed"

class FinalAnswerTool(BaseModel):
    """Deliver the final answer when reasoning is complete."""

    answer: str

class ThinkResponse(BaseModel):
    """Thought response"""

    thought: str

def response_union(tools: List[ToolSpec]) -> Type[BaseModel]:
    models = [tool.model_class() for tool in tools]
    return models[0] if len(models) == 1 else Union[*models]

def tool_names(tools: List[ToolSpec]) -> List[str]:
    return [tool.name for tool in tools]

def tool_names_and_descriptions(tools: List[ToolSpec]) -> str:
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

def resolve_tool(payload: ToolSpec, tools: List[ToolSpec]) -> ToolSpec:
    for tool in tools:
        if isinstance(payload, tool.model_class()):
            return tool
    return None

def get_reasoning_text(reasoning_steps: List[ReasoningStep]) -> str:
    return "\n".join([
        f"\nStep {i+1}: {step.title or 'Untitled'}" +
        (f"\n  Action: {step.action}" if step.action else "") +
        (f"\n  Result: {step.result}" if step.result else "") +
        (f"\n  Reasoning: {step.reasoning}" if step.reasoning else "")
        for i, step in enumerate(reasoning_steps or [])
    ]) 

def get_messages(system_prompt: str, query: str, history: List[str] = []) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {query}"},
        {"role": "system", "content": f"History:\n{'\n'.join(history)}"},
    ]

def get_thought_messages(tools: List[ToolSpec], query: str, history: List[str] = []) -> List[dict]:
    system_prompt = (
        "You are an agent that uses structured reasoning to solve problems.\n"
        f"Available tools: {tool_names_and_descriptions(tools)}.\n"
        "Current step: Structured Reasoning.\n"
        "Review the history above to see what information you have already gathered from previous observations.\n"
        "You MUST use the reasoning_tool to create a step-by-step plan for what to do next.\n"
        "The reasoning tool will help you break down the problem and decide on the next action.\n"
        "If you have enough information to provide a final answer, the reasoning tool will indicate this in its final response.\n"
    )

    return get_messages(system_prompt, query, history)

def get_action_messages(tools: List[ToolSpec], query: str, history: List[str] = [], reasoning_steps: List[ReasoningStep] = []) -> List[dict]:
    # Extract the latest reasoning step to get the planned action
    latest_action = ""
    if reasoning_steps:
        latest_step = reasoning_steps[-1]
        if latest_step.action:
            latest_action = latest_step.action

    system_prompt = (
        "You are an agent that follows a reasoning-based plan.\n"
        f"Available tools: {tool_names_and_descriptions(tools)}.\n"
        "Current step: Action.\n"
        "You MUST follow the reasoning plan that was just created.\n"
        f"Latest reasoning step action: {latest_action}\n\n"
        "IMPORTANT: Execute exactly what the reasoning step specifies. Do not deviate from the plan.\n"
        "If the reasoning suggests searching for specific information, use those exact search terms.\n"
    )

    return get_messages(system_prompt, query, history)

def run_react_loop(query: str, client: instructor.Client, user_config: dict = {}) -> str:
    """Run the deterministic reasoning loop until the agent returns a final answer."""

    # Configure
    reasoning_tool = make_reasoning_tool()
    searxng_tool = make_searxng_search_tool()

    default_config = {
        'max_steps': 10,
        'tools': [searxng_tool]
    }

    config = merge_configs(user_config, default_config)

    tools: List[ToolSpec] = config['tools']
    history: List[str] = []

    # Tracking structures to prevent redundancy
    completed_steps: Set[str] = set()  # Track completed reasoning step titles
    executed_actions: Set[str] = set()  # Track executed action signatures
    gathered_info: Dict[str, str] = {}  # Track what information has been gathered

    # State machine
    current_state = AgentState.INITIALIZING

    # Run deterministic loop
    for step_num in range(config['max_steps']):
        print(f"\n=== STEP {step_num + 1} ===")
        print(f"Current state: {current_state.value}")

        # State: REASONING
        current_state = AgentState.REASONING
        print("\nReasoning...")

        # Add context about completed steps and gathered info
        context_info = ""
        if completed_steps:
            context_info += f"\nAlready completed steps: {len(completed_steps)} unique reasoning steps"
        if gathered_info:
            context_info += f"\nGathered information: {len(gathered_info)} data points"
            for title, result in list(gathered_info.items())[:3]:  # Show top 3
                context_info += f"\n- {title}: {result[:100]}..."

        # Add this context to the history
        enhanced_history = history.copy()
        if context_info:
            enhanced_history.append(f"Context: {context_info}")

        messages = get_thought_messages(tools, query, enhanced_history)
        thought = client.chat.completions.create(
            messages=messages,
            response_model=ReasoningToolArgs,
        )
        
        if isinstance(thought, ReasoningToolArgs):
            thought_payload = thought.model_dump()
            reasoning: ReasoningToolResponse = reasoning_tool.handler(thought_payload)

            if reasoning.error:
                print(f"Reasoning failed: {reasoning.error}")
                # If reasoning fails, we need to try again or fail gracefully
                if step_num == config['max_steps'] - 1:
                    return f"Failed to reason about the query: {reasoning.error}"
                continue

            # Check if reasoning produced a final answer
            if reasoning.final_answer and reasoning.final_answer.strip():
                current_state = AgentState.COMPLETED
                print("Reasoning produced final answer, terminating...")
                return reasoning.final_answer

            # Track completed reasoning steps to prevent redundancy
            current_step_titles = set()
            for step in reasoning.reasoning_steps:
                if step.title:
                    step_hash = hashlib.md5(step.title.encode()).hexdigest()[:8]
                    current_step_titles.add(step_hash)
                    if step_hash not in completed_steps:
                        completed_steps.add(step_hash)
                        # Update gathered info based on step results
                        if step.result and step.title:
                            gathered_info[step.title] = step.result

            # Check if we're repeating steps
            if current_step_titles.issubset(completed_steps):
                print("All reasoning steps have been completed before, checking for final answer...")
                # If we've completed all these steps before but don't have a final answer,
                # we might be stuck. Try to synthesize from gathered info
                if gathered_info:
                    current_state = AgentState.COMPLETED
                    synthesis = "\n".join([f"- {title}: {result}" for title, result in gathered_info.items()])
                    return f"Based on the information gathered:\n{synthesis}\n\nPlease let me know if you need more specific details on any aspect."

            thought_content = f"Thought (reasoning): {get_reasoning_text(reasoning.reasoning_steps)}"
        else:
            thought_content = f"Thought (malformed): {str(thought)}"
            if step_num == config['max_steps'] - 1:
                return f"Failed to generate proper reasoning: {str(thought)}"
            continue

        history.append(thought_content)

        # State: ACTING
        current_state = AgentState.ACTING
        print("\nActing...")
        messages = get_action_messages(tools, query, history, reasoning.reasoning_steps)
        act_model = response_union(tools)
        action = client.chat.completions.create(
            messages=messages,
            response_model=act_model,
        )

        # Observe
        print("\nObserving...")
        tool = resolve_tool(action, tools)
        payload = action.model_dump()

        if tool is None:
            action_content = str(action)
            observation_content = f"Unknown tool: {str(action)}"
        else:
            action_content = f"{tool.name}({payload})"

            # Create action signature for deduplication
            action_signature = hashlib.md5(str(payload).encode()).hexdigest()[:8]

            # Check if this action has been executed before
            if action_signature in executed_actions:
                print(f"Action {action_signature} already executed, skipping...")
                observation_content = "This action was already performed. Please proceed with new reasoning based on existing results."
            else:
                executed_actions.add(action_signature)
                observation_content = tool.handler(payload)

        history.append(f"Action: {action_content}")
        history.append(f"Observation: {observation_content}")

        #---
        print("\nThought:", "="*50)
        print(thought_content)

        print("\nAction:", "="*50)
        print(action_content)

        print("\nObservation:", "="*50)
        print(str(observation_content)[:100])
        #---

    current_state = AgentState.FAILED
    return f"Max steps ({config['max_steps']}) reached before final answer."


if __name__ == "__main__":
    query = "What is the weather in the capital of France, and what is that city known for?"

    client = build_client()
    answer = run_react_loop(query, client)
    print("\nFinal Answer:", answer)