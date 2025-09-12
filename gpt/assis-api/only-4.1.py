
"""
这是25-08-20更新的新的api使用方法，需要assistant和thread，仅支持gpt4.1

流式输出以及相关变量
https://platform.openai.com/docs/api-reference/runs/createRun

assistans相关
https://platform.openai.com/docs/api-reference/assistants/createAssistant

threads相关
https://platform.openai.com/docs/api-reference/threads/createThread

gpt于2025.08.20更新了API调用方法，旧版不支持assistants与threads：
https://platform.openai.com/docs/guides/migrate-to-responses?tool-use=chat-completions#about-the-responses-api
"""


from openai import OpenAI

# import os
api_key = "xxx"
from openai import OpenAI
import datetime

client = OpenAI(api_key=api_key)

response = client.beta.threads.runs.create(
    model="gpt-5-nano",
    max_completion_tokens=1000,  # 控制思考token + 回复token总数
    input=[
        {
            "role": "user",
            "content": "讲一下什么是ssr，前端的",
        },
    ],
    stream=True,

    # 该instructions参数为模型提供了生成响应时应如何操作的高级指令，包括语气、目标以及正确响应的示例。任何以这种方式提供的指令都将优先于input参数中的提示。
    # instructions= "Talk like a pirate.",  
)


# --- 在这里修改你的循环 ---
final_response = None
print("模型输出: ", end="", flush=True)

for event in response:
    # 检查事件类型是否为文本增量
    if event.type == 'response.output_text.delta':
        # 直接打印出增量内容，不换行
        print(event.delta, end="", flush=True)
    
    # 检查事件类型是否为请求完成
    elif event.type == 'response.completed':
        # 保存最终的响应对象以备后用
        final_response = event.response

# 循环结束后，打印一个换行符，让格式更美观
print("\n" + "="*50)

# 如果成功接收到最终响应，则打印其元数据
if final_response:
    # 提取所需信息
    request_id = final_response.id
    created_timestamp = final_response.created_at
    usage_info = final_response.usage

    # 将 Unix 时间戳转换为可读格式
    created_time = datetime.datetime.fromtimestamp(created_timestamp).strftime('%Y-%m-%d %H:%M:%S')

    print(f"请求完成 ✓")
    print(f"请求 ID: {request_id}")
    print(f"创建时间: {created_time}")
    print("Token 使用情况:")
    print(f"  - 输入 Tokens: {usage_info.input_tokens}")
    print(f"  - 输出 Tokens: {usage_info.output_tokens}")
    print(f"  - 总 Tokens: {usage_info.total_tokens}")
else:
    print("未能获取到最终响应信息。")