from __future__ import annotations

from typing import List, Type, Union

import instructor
from pydantic import BaseModel
from dotenv import load_dotenv

from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec
from struct_agent.instructor_based.utils import merge_configs
from struct_agent.tools.searxng_tools import make_searxng_search_tool
from pprint import pprint

load_dotenv()

class FinalAnswerTool(BaseModel):
    """Deliver the final answer when reasoning is complete."""

    answer: str

class ThinkResponse(BaseModel):
    """Thought response"""
    
    thought: str

def run_react_loop(query: str, client: instructor.Client, config: dict = {}) -> str:
    """Run the ReAct loop until the agent returns a final answer."""

    # Configure
    default_config = {
        'max_steps': 10,
        'tools': [make_searxng_search_tool()]
    }

    config = merge_configs(config, default_config)

    tools: List[ToolSpec] = config['tools']

    def response_union() -> Type[BaseModel]:
        models = [tool.model_class() for tool in tools]
        # models.append(FinalAnswerTool)
        return models[0] if len(models) == 1 else Union[*models]

    def tool_names() -> List[str]:
        return [tool.name for tool in tools]
    
    def resolve_tool(payload: ToolSpec) -> ToolSpec:
        for tool in tools:
            if isinstance(payload, tool.model_class()):
                return tool
        return None
    
    def get_messages(system_prompt):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User query: {query}"},
            {"role": "system", "content": f"History:\n{'\n'.join(history)}"},
        ]
    
    def get_thought_messages():
        system_prompt = (
            "You are an agent that uses a Thought → Action → Observation loop.\n"
            f"Available tools: {tool_names()}.\n"
            "Current step: Thought/FinalAnswer.\n"
            "Review the history above to see what information you have already gathered from previous observations.\n"
            "If you have sufficient information from previous observations to answer the user's question completely, call FinalAnswerTool immediately to provide the final answer.\n"
            "If you still need more information, use ThinkResponse to explain what additional information you need to gather.\n"
        )

        return get_messages(system_prompt)
    
    def get_action_messages():
        system_prompt = (
            "You are an agent that uses a Thought → Action → Observation loop.\n"
            f"Available tools: {tool_names()}.\n"
            "Current step: Action.\n"
            "Based on the latest thought, call the appropriate tool to answer the user's question.\n"
        )

        return get_messages(system_prompt)
    
    # Run loop
    history: List[str] = []

    for step_num in range(config['max_steps']):
        print(f"\n=== STEP {step_num + 1} ===")

        # Think
        print("Thinking...")
        messages = get_thought_messages()
        thought = client.chat.completions.create(
            messages=messages,
            response_model=Union[ThinkResponse, FinalAnswerTool],
        )

        if isinstance(thought, FinalAnswerTool):
            return thought.answer

        thought_content = thought.thought
        history.append(f"Thought: {thought_content}")

        # Act
        print("Acting...")
        messages = get_action_messages()
        act_model = response_union()
        action = client.chat.completions.create(
            messages=messages,
            response_model=act_model,
        )

        # Observe
        print("Observing...")
        tool = resolve_tool(action)
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
        print("Thought:", "="*50)
        print(thought_content)

        print("Action:", "="*50)
        print(action_content)

        print("Observation:", "="*50)
        print(str(observation_content)[:100])
        #---

    return f"Max steps ({config['max_steps']}) reached before final answer."


if __name__ == "__main__":
    query = "What is the weather in the capital of France, and what is that city known for?"

    client = build_client(use_lmstudio=True)
    answer = run_react_loop(query, client)
    print("\nFinal Answer:", answer)