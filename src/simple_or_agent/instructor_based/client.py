import instructor
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai
from instructor import from_openai
import lmstudio as lms

load_dotenv()

def build_instructor_client(model_id: str=None, url: str=None, key: str=None) -> instructor.AsyncInstructor:
    # provider_id = model_id or os.getenv("MODEL_ID") or os.getenv("LMSTUDIO_MODEL") or "qwen/qwen3-next-80b-a3b-thinking"
    # base_url = url or os.getenv("LMSTUDIO_BASE_URL")
    # api_key = key or os.getenv("OPENROUTER_API_KEY")

    # provider_id = f"openrouter/{provider_id}"
    # # return instructor.from_provider(provider_id, api_key=api_key)
    # return instructor.from_provider(model='openai/gpt-oss-20b', base_url='http://192.168.1.157:1234/v1', api_key='123')

    # lm_client = lms.llm("openai/gpt-oss-20b", base_url="http://192.168.1.157:1234")

    # instructor_client = instructor.patch(lm_client)

    # return instructor_client

    return instructor.from_openai(
        openai.OpenAI(base_url="http://192.168.1.157:1234/v1", api_key="lm-studio"),
        mode=instructor.Mode.MD_JSON,
    )

    # return from_openai(
    #     openai.OpenAI(api_key='123'),
    #     model='openai/gpt-oss-20b',
    #     base_url='http://192.168.1.157:1234',
    #     mode=instructor.Mode.TOOLS,
    # )

if __name__ == '__main__':
    pass