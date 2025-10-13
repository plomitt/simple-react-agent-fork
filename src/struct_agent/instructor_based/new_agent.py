from __future__ import annotations

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field
from instructor import Instructor
from dotenv import load_dotenv

from struct_agent.instructor_based.prompt_manager import get_action_prompt, get_reasoning_prompt, get_thought_prompt
from struct_agent.instructor_based.reasoning_modules import NextAction, ReasoningStep, ReasoningSteps
from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec
from struct_agent.instructor_based.utils import merge_configs

from struct_agent.tools.searxng_tools import make_searxng_search_tool
from struct_agent.tools.blank_tool import make_blank_tool

load_dotenv()

class FinalAnswer(BaseModel):
    """Deliver the final answer."""
    answer: str = Field(..., description="The final answer to the user's question.")

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

def get_messages(system_prompt: str, query: str, history: List[str] = []) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User query: {query}"},
        {"role": "system", "content": f"History:\n{'\n'.join(history)}"},
    ]

def get_thought_messages(_tools: List[ToolSpec], query: str, history: List[str] = [], is_last_step: bool = False) -> List[dict]:
    system_prompt = get_thought_prompt(is_last_step)
    return get_messages(system_prompt, query, history)

def get_action_messages(tools: List[ToolSpec], query: str, history: List[str] = []) -> List[dict]:
    system_prompt = get_action_prompt(tool_names_and_descriptions(tools))
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

    final_step = steps[-1]
    return final_step.reasoning or final_step.action or "No structured reasoning returned."

def run_react_loop(query: str, client: Instructor, user_config: dict = {}) -> str:
    """Run the ReAct loop until the agent returns a final answer."""

    # Configure
    searxng_tool = make_searxng_search_tool()
    blank_tool = make_blank_tool()

    default_config = {
        'max_steps': 10,
        'tools': [searxng_tool, blank_tool]
    }

    config = merge_configs(user_config, default_config)

    tools: List[ToolSpec] = config['tools']
    history: List[str] = []

    # Run loop
    for step_num in range(config['max_steps']):

        # Think
        is_last_step = step_num == config['max_steps'] - 1
        messages = get_thought_messages(tools, query, history, is_last_step)

        thought: BaseModel = client.chat.completions.create(
            messages=messages,
            response_model=Union[ThoughtTopic, FinalAnswer],
            extra_body={"provider": {"require_parameters": True}}
        )

        if isinstance(thought, FinalAnswer):
            return thought.answer

        if isinstance(thought, ThoughtTopic):
            thought_topic = thought.thought_topic
            thought = generate_tought(query, thought_topic, history, client)
            thought_content = f"Thought: {thought}"
        else:
            thought_content = f"Thought (malformed): {str(thought)}"

        history.append(thought_content)

        # Act
        messages = get_action_messages(tools, query, history)
        act_model = response_union(tools)
        action: BaseModel = client.chat.completions.create(
            messages=messages,
            response_model=act_model,
            extra_body={"provider": {"require_parameters": True}}
        )

        # Observe
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

    return f"Max steps ({config['max_steps']}) reached before final answer."

if __name__ == "__main__":
    query = "What is the weather in the capital of France, and what is that city known for?"

    client = build_client()
    answer = run_react_loop(query, client)
    print("\nFinal Answer:", answer)