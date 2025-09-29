from instructor import Mode
from typing import Optional
from openai import OpenAI
import instructor
import os

def resolve_model(use_lmstudio: bool = False):
    if use_lmstudio:
        return os.getenv("LMSTUDIO_MODEL_ID")
    return os.getenv("OPENROUTER_MODEL_ID")

def build_or_client(model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
    resolved_base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
    resolved_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    resolved_model = model or resolve_model()
    provider = f'openrouter/{resolved_model}'
    return instructor.from_provider(provider, api_key=resolved_api_key, mode=Mode.TOOLS, base_url=resolved_base_url)

def build_lm_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    resolved_base_url = base_url or os.getenv("LMSTUDIO_BASE_URL")
    resolved_api_key = api_key or os.getenv("LMSTUDIO_API_KEY", '123')
    openai_client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
    return instructor.from_openai(openai_client, mode=Mode.JSON)

def build_client(model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, use_lmstudio: bool = False):
    if use_lmstudio:
        return build_lm_client(api_key, base_url)
    return build_or_client(model, api_key, base_url)

__all__ = ["resolve_model", "build_or_client", "build_lm_client", "build_client"]