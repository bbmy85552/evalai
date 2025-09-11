import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()


class QwenChatError(Exception):
    pass


class QwenStream:
    """
    纯流式调用，支持 max_tokens 等全部额外参数
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen-plus",
        system: str = "You are a helpful assistant.",
    ):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise QwenChatError("缺少 DASHSCOPE_API_KEY")
        base_url = (base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system = system

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[str]:
        """
        仅 yield 文本片段；最后 yield 一个 dict 带用量
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=max_tokens,  # 👈 关键参数
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
    for seg in bot.stream("讲一下什么是Spring Boot", max_tokens=20):  # 限制最多 20 个输出 token
        if isinstance(seg, dict):
            usage = seg
        else:
            print(seg, end="", flush=True)
    if usage:
        print("\n--- 用量 ---", usage)
