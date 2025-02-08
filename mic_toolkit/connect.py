import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class Agent:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "API Key not found. Please set the OPENAI_API_KEY environment variable."
            )
        self.model = model
        self.client = OpenAI()

    def chat(self, data):
        return self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": data,
                }
            ],
            model=self.model,
        )
