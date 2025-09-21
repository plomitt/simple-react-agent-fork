import instructor
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

# Define what you want
class User(BaseModel):
    name: str
    age: int


# Extract it from natural language
client = instructor.from_provider("openrouter/google/gemini-2.5-flash-lite", api_key=os.getenv("OPENROUTER_API_KEY"))
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

if __name__ == '__main__':
    print(user)  # User(name='John', age=25)