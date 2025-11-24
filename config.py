
from openai import OpenAI
import os
client = OpenAI(
    api_key=os.environ.get("HF_TOKEN"),
    base_url="https://api.groq.com/openai/v1",
)

DEPLOYMENT = "llama-3.3-70b-versatile"
