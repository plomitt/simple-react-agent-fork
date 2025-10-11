from __future__ import annotations

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from instructor import Instructor
import logging

from struct_agent.instructor_based.reasoning_modules import NextAction, ReasoningStep, ReasoningSteps
from struct_agent.instructor_based.reasoning_prompt import get_reasoning_prompt
from struct_agent.instructor_based.client_manager import build_client
from struct_agent.instructor_based.tool_manager import ToolSpec
from struct_agent.instructor_based.utils import merge_configs
from struct_agent.tools.searxng_tools import make_searxng_search_tool

load_dotenv()

# Verbosity level constants
VERBOSITY_SILENT = 0      # Only final answer
VERBOSITY_ESSENTIAL = 1    # Final answer + reasoning outline (default)
VERBOSITY_STANDARD = 2     # + thought/action/observation steps
VERBOSITY_VERBOSE = 3      # + debug info and instructor logs

# Configure logging based on verbosity level
def configure_logging(verbosity: int):
    """Configure logging level based on verbosity setting."""
    if verbosity >= VERBOSITY_VERBOSE:
        level = logging.DEBUG
    elif verbosity >= VERBOSITY_STANDARD:
        level = logging.INFO
    elif verbosity >= VERBOSITY_ESSENTIAL:
        level = logging.WARNING
    else:
        level = logging.ERROR

    # Configure root logger to suppress unwanted output
    logging.basicConfig(level=level, force=True)

    # Suppress httpx and other noisy loggers unless in verbose mode
    if verbosity < VERBOSITY_VERBOSE:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    if verbosity >= VERBOSITY_VERBOSE:
        logger.info("Using standard logging only")
    return logger

# Initialize with default logging (will be reconfigured when verbosity is known)
logger = logging.getLogger(__name__)

def print_verbose(verbosity_level: int, min_level: int, message: str, *args, **kwargs):
    """Print message only if verbosity level meets minimum requirement."""
    if verbosity_level >= min_level:
        print(message, *args, **kwargs)

def print_debug(verbosity_level: int, message: str, *args, **kwargs):
    """Print debug message only in verbose mode."""
    print_verbose(verbosity_level, VERBOSITY_VERBOSE, message, *args, **kwargs)

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

def get_thought_messages(_tools: List[ToolSpec], query: str, history: List[str] = [], is_last_step: bool = False) -> List[dict]:
    if is_last_step:
        system_prompt = (
            "You are an agent that uses a Thought → Action → Observation loop.\n"
            "This is the FINAL step - you MUST provide your final answer now.\n"
            "Call FinalAnswer with a complete answer to the user's question.\n"
            "Do NOT call ThoughtTopic anymore.\n"
        )
    else:
        system_prompt = (
            "You are an agent that uses a Thought → Action → Observation loop.\n"
            "Your goal is to gather information to answer the user's question.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "1. Call FinalAnswer ONLY when you have sufficient information to provide a complete answer\n"
            "2. Call ThoughtTopic to continue thinking and gathering more information\n"
            "3. Do not call FinalAnswer prematurely - ensure you have enough context\n\n"
            "Response models available:\n"
            "- FinalAnswer: Use when ready to provide the final answer\n"
            "- ThoughtTopic: Use to continue the reasoning process\n"
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
        lines.append(f"    Observation: {str(entry['observation'])[:500]}")
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

def generate_tought(query, topic, history, client: Instructor, verbosity: int = VERBOSITY_ESSENTIAL) -> str:
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

    print_verbose(verbosity, VERBOSITY_STANDARD, format_thought_steps(steps))
    print_verbose(verbosity, VERBOSITY_STANDARD, "\n")

    final_step = steps[-1]
    return final_step.reasoning or final_step.action or "No structured reasoning returned."

def run_react_loop(query: str, client: Instructor, user_config: dict = {}) -> str:
    """Run the ReAct loop until the agent returns a final answer."""

    # Configure
    searxng_tool = make_searxng_search_tool()

    default_config = {
        'max_steps': 10,
        'tools': [searxng_tool],
        'verbosity': VERBOSITY_STANDARD  # Default to standard verbosity
    }

    config = merge_configs(user_config, default_config)
    verbosity = config['verbosity']

    global logger
    logger = configure_logging(verbosity)

    tools: List[ToolSpec] = config['tools']
    history: List[str] = []
    reasoning_outline: List[Dict[str, str]] = []

    # Run loop
    for step_num in range(config['max_steps']):
        print_verbose(verbosity, VERBOSITY_STANDARD, f"\n=== STEP {step_num + 1} ===")

        # Think
        print_verbose(verbosity, VERBOSITY_STANDARD, "\nThinking...")
        is_last_step = step_num == config['max_steps'] - 1
        messages = get_thought_messages(tools, query, history, is_last_step)

        print_debug(verbosity, f"Step {step_num + 1} - Making instructor call with response_model Union[ThoughtTopic, FinalAnswer]")
        print_debug(verbosity, f"Messages: {messages}")

        try:
            thought_response = client.chat.completions.create(
                messages=messages,
                response_model=Union[ThoughtTopic, FinalAnswer],
                extra_body={"provider": {"require_parameters": True}}
            )

            # Store the response for debugging
            thought = thought_response

            print_debug(verbosity, f"Response type: {type(thought)}")
            print_debug(verbosity, f"Response value: {thought}")

            # Access raw response for debugging
            if hasattr(thought, '_raw_response'):
                print_debug(verbosity, f"Raw response: {thought._raw_response}")
                print_debug(verbosity, "Raw LLM response available")
                # Print a snippet of the raw response for debugging
                raw_resp = thought._raw_response
                if hasattr(raw_resp, 'choices') and raw_resp.choices:
                    choice = raw_resp.choices[0]
                    if hasattr(choice, 'message') and choice.message:
                        content = getattr(choice.message, 'content', 'No content')
                        print_debug(verbosity, f"Raw content snippet: {str(content)[:200]}...")

            print_debug(verbosity, f"Response type = {type(thought).__name__}")
            print_debug(verbosity, f"Response value = {thought}")

            # Validate response structure
            if isinstance(thought, (ThoughtTopic, FinalAnswer)):
                print_debug(verbosity, "Valid response structure detected")
            else:
                print_debug(verbosity, "Unexpected response structure!")

        except Exception as e:
            logger.error(f"Error in instructor call: {e}")
            print_debug(verbosity, f"ERROR: Instructor call failed: {e}")
            print_debug(verbosity, f"ERROR: Exception type: {type(e).__name__}")
            # Continue with a malformed thought to keep the loop going
            thought = f"Error: {str(e)}"

        if isinstance(thought, FinalAnswer):
            print_debug(verbosity, "FinalAnswer detected!")
            print_verbose(verbosity, VERBOSITY_ESSENTIAL, format_reasoning_outline(reasoning_outline))
            print_verbose(verbosity, VERBOSITY_ESSENTIAL, "\n")
            return thought.answer, step_num

        if isinstance(thought, ThoughtTopic):
            print_debug(verbosity, "ThoughtTopic detected!")
            thought_topic = thought.thought_topic
            thought = generate_tought(query, thought_topic, history, client, verbosity)
            thought_content = f"Thought: {thought}"
        else:
            print_debug(verbosity, f"Malformed response detected: {type(thought)}")
            thought_content = f"Thought (malformed): {str(thought)}"

        history.append(thought_content)

        # Act
        print_verbose(verbosity, VERBOSITY_STANDARD, "\nActing...")
        messages = get_action_messages(tools, query, history)
        act_model = response_union(tools)
        action: BaseModel = client.chat.completions.create(
            messages=messages,
            response_model=act_model,
        )

        # Observe
        print_verbose(verbosity, VERBOSITY_STANDARD, "\nObserving...")
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

        print_verbose(verbosity, VERBOSITY_STANDARD, '\n\nAction', '>'*10)
        print_verbose(verbosity, VERBOSITY_STANDARD, action_content)
        print_verbose(verbosity, VERBOSITY_STANDARD, '\n\nObservation', '>'*10)
        print_verbose(verbosity, VERBOSITY_STANDARD, str(observation_content)[:500])

        reasoning_outline.append({
            "step": str(len(reasoning_outline) + 1),
            "topic": thought_topic,
            "thought": thought_content,
            "action": action_content,
            "observation": observation_content,
        })  

    print_verbose(verbosity, VERBOSITY_ESSENTIAL, format_reasoning_outline(reasoning_outline))
    print_verbose(verbosity, VERBOSITY_ESSENTIAL, "\n")

    return f"Max steps ({config['max_steps']}) reached before final answer.", config['max_steps']

if __name__ == "__main__":
    query = "What is the weather in the capital of France, and what is that city known for?"

    client = build_client()
    answer, step_num = run_react_loop(query, client)
    print("\nFinal Answer:", answer)