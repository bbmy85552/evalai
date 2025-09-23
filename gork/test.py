from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

class ChatApp:
    def __init__(
        self,
        x_ai_api_key: str,
        base_url: str = "https://api.x.ai/v1",
        system_prompt: str | None = None,
    ) -> None:
        self.x_ai_api_key = x_ai_api_key
        self.grok_client = OpenAI(base_url=base_url, api_key=self.x_ai_api_key)
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def converse(self, model: str = "grok-code-fast"):
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Exiting...")
                break

            print(f"You: {user_input}", flush=True)
            self.messages.append({"role": "user", "content": user_input})

            model_response = ""
            stream = self.grok_client.chat.completions.create(
                model=model, messages=self.messages, stream=True
            )

            print("Grok: ", end="", flush=True)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    model_response += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
            self.messages.append({"role": "assistant", "content": model_response})


SYSTEM_PROMPT = """
You are Grok, a customer service assistant created by xAI for a food delivery app similar to Deliveroo.
Your role is to assist users with questions about their orders.
Respond in a clear, friendly, and professional tone. Never stray off topic and focus exclusively on answering customer service queries.
"""

api_key=os.getenv('XAI_API_KEY')
app = ChatApp(
    x_ai_api_key=api_key,
    system_prompt=SYSTEM_PROMPT,
)
app.converse()