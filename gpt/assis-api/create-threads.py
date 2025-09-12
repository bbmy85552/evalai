from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 直接初始化客户端，它会自动读取名为 "OPENAI_API_KEY" 的环境变量
# 你不再需要手动传递 api_key 参数
client = OpenAI()
empty_thread = client.beta.threads.create()
print(empty_thread)

"""
2dqy003@2dqy003deMac-mini evalai % python gpt/create-threads.py
/Users/2dqy003/PycharmProjects/evalai/gpt/create-threads.py:11: DeprecationWarning: The Assistants API is deprecated in favor of the Responses API
  empty_thread = client.beta.threads.create()

Thread(id='thread_2rhXLMcGID1yuANDaRR1OGA5', 
created_at=1757409938, 
metadata={}, 
object='thread', 
tool_resources=ToolResources(code_interpreter=None, file_search=None))
"""
