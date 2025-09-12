import os

from xai_sdk import Client
from xai_sdk.chat import user, system
from dotenv import load_dotenv

load_dotenv()




client = Client(
api_key=os.getenv('XAI_API_KEY'),
timeout=3600, # Override default timeout with longer timeout for reasoning models
)

chat = client.chat.create(model="grok-code-fast")
chat.append(
system("You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."),
)
chat.append(
user("什么是css，前端方面?")
)

for response, chunk in chat.stream():
    print(chunk.content, end="", flush=True) # Each chunk's content
    print(response.content, end="", flush=True) # The response object auto-accumulates the chunks

    print(response.content) # The full response
