import openai
from dotenv import load_dotenv
import os

load_dotenv()


class ChatOpenAI:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        # Configure legacy OpenAI SDK
        openai.api_key = self.openai_api_key

    def run(self, messages, text_only: bool = True, **kwargs):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        response = openai.ChatCompletion.create(
            model=self.model_name, messages=messages, **kwargs
        )

        if text_only:
            return response.choices[0].message.content

        return response
