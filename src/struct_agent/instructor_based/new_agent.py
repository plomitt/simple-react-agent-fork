from __future__ import annotations

from typing import List, Type, Union
from pydantic import BaseModel
from dotenv import load_dotenv
from pprint import pprint
import instructor

from struct_agent.instructor_based.reasoning_tool import ReasoningStep, ReasoningToolArgs, ReasoningToolResponse, make_reasoning_tool
from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec
from struct_agent.instructor_based.utils import merge_configs
from struct_agent.tools.searxng_tools import make_searxng_search_tool

load_dotenv()

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
        "You are an agent that uses a Thought → Action → Observation loop.\n"
        f"Available tools: {tool_names_and_descriptions(tools)}.\n"
        "Current step: Thought.\n"
        "Review the history above to see what information you have already gathered from previous observations.\n"
        "If you have sufficient information from previous observations to answer the user's question completely, call FinalAnswerTool immediately to provide the final answer.\n"
        "If you still need more information, call reasoning_tool (ReasoningToolArgs) to think what to do next.\n"
    )

    return get_messages(system_prompt, query, history)

def get_action_messages(tools: List[ToolSpec], query: str, history: List[str] = []) -> List[dict]:
    system_prompt = (
        "You are an agent that uses a Thought → Action → Observation loop.\n"
        f"Available tools: {tool_names_and_descriptions(tools)}.\n"
        "Current step: Action.\n"
        "Based on the latest thought, call the appropriate tool to answer the user's question.\n"
    )

    return get_messages(system_prompt, query, history)

def run_react_loop(query: str, client: instructor.Client, user_config: dict = {}) -> str:
    """Run the ReAct loop until the agent returns a final answer."""

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

    # Run loop
    for step_num in range(config['max_steps']):
        print(f"\n=== STEP {step_num + 1} ===")

        # Think
        print("\nThinking...")
        messages = get_thought_messages(tools, query, history)
        thought = client.chat.completions.create(
            messages=messages,
            response_model=Union[ReasoningToolArgs, FinalAnswerTool],
        )

        if isinstance(thought, FinalAnswerTool):
            return thought.answer
        
        if isinstance(thought, ReasoningToolArgs):
            thought_payload = thought.model_dump()
            reasoning: ReasoningToolResponse = reasoning_tool.handler(thought_payload)

            if reasoning.error:
                thought_content = f"Thought (error): {reasoning.error}"
            else:
                thought_content = f"Thought (reasoning): {get_reasoning_text(reasoning.reasoning_steps)}"
        else:
            thought_content = f"Thought (malformed): {str(thought)}"

        history.append(thought_content)

        # Act
        print("\nActing...")
        messages = get_action_messages(tools, query, history)
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

    return f"Max steps ({config['max_steps']}) reached before final answer."


if __name__ == "__main__":
    query = "What is the weather in the capital of France, and what is that city known for?"

    client = build_client()
    answer = run_react_loop(query, client)
    print("\nFinal Answer:", answer)