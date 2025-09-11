import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# 参数配置
API_KEY        = None                        # None=自动读环境变量 DASHSCOPE_API_KEY
BASE_URL       = None                        # None=默认北京节点
MODEL_NAME     = "qwen-plus"                 # 模型名
SYSTEM_PROMPT  = "You are a helpful assistant."
PROMPT         = "讲一下什么是SSR，前端的"
MAX_TOKENS     = 200                       # ← 新增：输出 token 上限
ENABLE_REASONING = False                     # 是否启用思考（仅部分模型支持）
REASONING_EFFORT = "medium"                  # minimal / medium / high

class QwenChatError(Exception):
    pass


class QwenStream:
    """纯流式调用，支持 max_tokens / reasoning 等全部参数"""
    def __init__(self) -> None:
        key = API_KEY or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise QwenChatError("缺少 DASHSCOPE_API_KEY")
        url = (BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.client = OpenAI(api_key=key, base_url=url)

    def stream(
        self,
        prompt: str = PROMPT,
        max_tokens: int = MAX_TOKENS,   # ← 使用配置区变量
        model: str = MODEL_NAME,
        system: str = SYSTEM_PROMPT,
        enable_reasoning: bool = ENABLE_REASONING,
        reasoning_effort: str = REASONING_EFFORT,
        **extra,
    ) -> Iterator[str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        reasoning_dict = (
            {"reasoning_effort": reasoning_effort} if enable_reasoning else {}
        )

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=max_tokens,
                **reasoning_dict,
                **extra,
            )
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                elif chunk.usage:
                    yield {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
        except APIError as e:
            raise QwenChatError(f"API 请求失败: {e}") from e


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    bot = QwenStream()
    usage = None
    for seg in bot.stream():          # 默认用配置区参数
        if isinstance(seg, dict):
            usage = seg
        else:
            print(seg, end="", flush=True)
    if usage:
        print("\n--- 用量 ---", usage)
