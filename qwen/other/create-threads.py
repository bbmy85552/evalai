import json
from dashscope import Threads
import dashscope
import os
from dotenv import load_dotenv

load_dotenv()
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

thread = Threads.create(
    # 建议您优先配置环境变量。若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    messages=[{"role": "user", "content": "How does AI work? Explain it in simple terms."}]
)
print(json.dumps(thread, default=lambda o: o.__dict__, sort_keys=True, indent=4))