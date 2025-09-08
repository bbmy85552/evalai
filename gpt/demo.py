from openai import OpenAI

import os
api_key = os.getenv("OPENAI_API_KEY")


from openai import OpenAI
import datetime

client = OpenAI(api_key=api_key)

stream = client.responses.create(
    model="gpt-5-nano",
    input=[
        {
            "role": "user",
            "content": "讲一下什么是ssr，前端的",
        },
    ],
    stream=True,
)


# --- 在这里修改你的循环 ---
final_response = None
print("模型输出: ", end="", flush=True)

for event in stream:
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