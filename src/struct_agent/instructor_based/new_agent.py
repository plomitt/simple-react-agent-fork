from __future__ import annotations

from typing import Any, Dict, List, Type, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pprint import pprint
from instructor import Instructor
import instructor

from struct_agent.instructor_based.reasoning_modules import NextAction, ReasoningStep, ReasoningSteps
from struct_agent.instructor_based.reasoning_prompt import get_reasoning_prompt
from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec
from struct_agent.instructor_based.utils import merge_configs
from struct_agent.tools.searxng_tools import make_searxng_search_tool

load_dotenv()

class FinalAnswerTool(BaseModel):
    """Deliver the final answer when reasoning is complete."""

    answer: str

class ThoughtTopic(BaseModel):
    """Thought response"""
    thought_topic: str = Field(..., description="Topic of the thought. What do I need to think about?")

def response_union(tools: List[ToolSpec]) -> BaseModel | Union[BaseModel]:
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
        "You are an agent that uses a Thought → Action → Observation loop.\n"
        f"Available tools: {tool_names(tools)}.\n"
        "Call FinalAnswerTool when you can answer the user.\n"
    )

    return get_messages(system_prompt, query, history)

def get_action_messages(tools: List[ToolSpec], query: str, history: List[str] = []) -> List[dict]:
    system_prompt = (
        "You are an agent that uses a Thought → Action → Observation loop.\n"
        f"Available tools: {tool_names_and_descriptions(tools)}.\n"
        "Based on the latest Thought, perform the next Action by calling one appropriate tool.\n"
    )

    return get_messages(system_prompt, query, history)

def summarize_action(action_name: str, action_args: Dict[str, Any]) -> str:
    formatted_args = ", ".join(f"{key}={value}" for key, value in action_args.items())
    return f"{action_name}({formatted_args})"

def filter_reasoning_steps_after_reset(reasoning_steps: List[ReasoningStep]) -> List[ReasoningStep]:
    last_reset_index = -1

    for i, step in enumerate(reasoning_steps):
        if step.next_action == NextAction.RESET:
            last_reset_index = i

    if last_reset_index >= 0:
        return reasoning_steps[last_reset_index + 1:]
    return reasoning_steps

def format_reasoning_outline(entries: List[Dict[str, str]]) -> str:
    """Build a readable outline so we can inspect each reasoning hop quickly."""
    if not entries:
        return "Reasoning outline is empty."

    lines: List[str] = ["\nReasoning Outline"]
    lines.append("------------------")
    for entry in entries:
        lines.append(f"Step {entry['step']}: {entry['topic']}")
        lines.append(f"    Thought: {entry['thought']}")
        lines.append(f"    Action: {entry['action']}")
        lines.append(f"    Observation: {entry['observation']}")
        lines.append("")
    return "\n".join(lines).rstrip()

def format_thought_steps(steps: List[ReasoningStep]) -> str:
    """Build a readable log that shows every step returned by the thought model."""
    if not steps:
        return "Thought agent returned no steps."

    lines: List[str] = ["\nThought Agent Steps"]
    lines.append("-------------------")
    for index, step in enumerate(steps, start=1):
        title = step.title or f"Step {index}"
        lines.append(f"{index}. {title}")
        if step.reasoning:
            lines.append(f"    Reasoning: {step.reasoning}")
        if step.action:
            lines.append(f"    Action: {step.action}")
        if step.next_action:
            lines.append(f"    Next Action: {step.next_action.value}")
        if step.confidence is not None:
            lines.append(f"    Confidence: {step.confidence}")
        lines.append("")
    return "\n".join(lines).rstrip()

def generate_tought(query, topic, history, client: Instructor) -> str:
    history_text = "\n".join(history) if history else "No previous steps."
    user_payload = (
        f"User query: {query}\n"
        f"Thought topic: {topic}\n"
        f"History:\n{history_text}\n"
        "Write the next Thought now."
    )

    messages = [
        {"role": "system", "content": get_reasoning_prompt()},
        {"role": "user", "content": user_payload},
    ]

    thought: ReasoningSteps = client.chat.completions.create(
        messages=messages,
        response_model=ReasoningSteps,
        extra_body={"provider": {"require_parameters": True}}
    )

    steps = thought.reasoning_steps

    if not steps:
        return "No structured reasoning returned."
    
    print(format_thought_steps(steps))
    print("\n")
    
    final_step = steps[-1]
    return final_step.reasoning or final_step.action or "No structured reasoning returned."

def run_react_loop(query: str, client: Instructor, user_config: dict = {}) -> str:
    """Run the ReAct loop until the agent returns a final answer."""

    # Configure
    searxng_tool = make_searxng_search_tool()

    default_config = {
        'max_steps': 10,
        'tools': [searxng_tool]
    }

    config = merge_configs(user_config, default_config)

    tools: List[ToolSpec] = config['tools']
    history: List[str] = []
    reasoning_outline: List[Dict[str, str]] = []

    # Run loop
    for step_num in range(config['max_steps']):
        print(f"\n=== STEP {step_num + 1} ===")

        # Think
        print("\nThinking...")
        messages = get_thought_messages(tools, query, history)
        thought = client.chat.completions.create(
            messages=messages,
            response_model=Union[ThoughtTopic, FinalAnswerTool],
            extra_body={"provider": {"require_parameters": True}}
        )

        if isinstance(thought, FinalAnswerTool):
            print(format_reasoning_outline(reasoning_outline))
            print("\n")
            return thought.answer
        
        if isinstance(thought, ThoughtTopic):
            thought_topic = thought.thought_topic
            thought = generate_tought(query, thought_topic, history, client)
            thought_content = f"Thought: {thought}"
        else:
            thought_content = f"Thought (malformed): {str(thought)}"

        history.append(thought_content)

        # Act
        print("\nActing...")
        messages = get_action_messages(tools, query, history)
        act_model = response_union(tools)
        action: BaseModel = client.chat.completions.create(
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
            action_content = summarize_action(tool.name, payload)
            observation_content = tool.handler(payload)

        history.append(f"Action: {action_content}")
        history.append(f"Observation: {observation_content}")

        reasoning_outline.append({
            "step": str(len(reasoning_outline) + 1),
            "topic": thought_topic,
            "thought": thought_content,
            "action": action_content,
            "observation": observation_content,
        })
    
    print(format_reasoning_outline(reasoning_outline))
    print("\n")

    return f"Max steps ({config['max_steps']}) reached before final answer."

if __name__ == "__main__":
    query = "What is the weather in the capital of France, and what is that city known for?"

    client = build_client()
    answer = run_react_loop(query, client)
    print("\nFinal Answer:", answer)