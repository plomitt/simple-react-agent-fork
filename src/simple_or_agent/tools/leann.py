from leann import LeannBuilder, LeannSearcher, LeannChat
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
import os

from simple_or_agent.instructor_based.tools import ToolSpec

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENROUTER_BASE_URL")

INDEX_PATH = str(os.getenv("LEANN_INDEX_PATH", Path("./").resolve()) / "/leann_index" / "vector.leann")
BACKEND_NAME = "hnsw"
CHAT_MODEL = os.getenv("LEANN_CHAT_MODEL", "openai/gpt-oss-20b")

def make_leann_add_text_tool() -> ToolSpec:
    """Add text content to the LEANN index."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            text_content = args["text_content"]

            builder = LeannBuilder(backend_name=BACKEND_NAME)
            builder.add_text(text_content)
            builder.build_index(INDEX_PATH)

            return {
                "message": "Text added to index",
            }
        except Exception as e:
            return {"error": f"Failed to add text to index: {str(e)}"}

    return ToolSpec(
        name="leann_add_text",
        description="Add text content to the LEANN vector index for semantic search and retrieval.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "text_content": {
                    "type": "string",
                    "description": "The text content to add to the index.",
                }
            },
            "required": ["text_content"],
        },
    )

def make_leann_search_tool() -> ToolSpec:
    """Search the LEANN index for relevant content."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            query = args["query"]
            top_k = args.get("top_k", 5)

            searcher = LeannSearcher(INDEX_PATH)
            results = searcher.search(query, top_k=top_k)

            return {
                # "query": query,
                # "top_k": top_k,
                "results": results
            }
        except Exception as e:
            return {"error": f"Failed to search index: {str(e)}"}

    return ToolSpec(
        name="leann_search",
        description="Search the LEANN vector index for semantically similar content.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find similar content.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return. Default: 5",
                }
            },
            "required": ["query"],
        },
    )

def make_leann_chat_tool() -> ToolSpec:
    """Chat with the LEANN index using RAG + LLM."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            query = args["query"]
            top_k = args.get("top_k", 5)

            chat = LeannChat(INDEX_PATH, llm_config={"type": "openai", "model": CHAT_MODEL})
            response = chat.ask(query, top_k=top_k)

            return {
                # "query": query,
                # "top_k": top_k,
                "response": response,
            }
        except Exception as e:
            return {"error": f"Failed to chat with index: {str(e)}"}

    return ToolSpec(
        name="leann_chat",
        description="Chat with the LEANN vector index using RAG (Retrieval-Augmented Generation) + LLM.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or message to send to the RAG system.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to retrieve for context. Default: 5",
                }
            },
            "required": ["query"],
        },
    )

__all__ = [
    "make_leann_add_text_tool",
    "make_leann_search_tool",
    "make_leann_chat_tool"
]