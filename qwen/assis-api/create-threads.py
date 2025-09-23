
import json
import os
from dashscope import Threads
from dotenv import load_dotenv

load_dotenv()
thread = Threads.create(
    # 建议您优先配置环境变量。若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    messages=[{"role": "user", "content": "How does AI work? Explain it in simple terms."}]
)
print(json.dumps(thread, default=lambda o: o.__dict__, sort_keys=True, indent=4))

