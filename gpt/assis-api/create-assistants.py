from openai import OpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 直接初始化客户端，它会自动读取名为 "OPENAI_API_KEY" 的环境变量
# 你不再需要手动传递 api_key 参数
client = OpenAI()

my_assistant = client.beta.assistants.create(
    instructions="You are a helpful assistant and need to respond based on the language entered by the user.",
    name="basic assistant-gpt-5-nano",
    # tools=[{"type": "code_interpreter"}],
    tools=[],
    model="gpt-5-nano",
)
print(my_assistant)
