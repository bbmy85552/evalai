import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# -------------------- 配置区 --------------------
prompt = "讲一下什么是 Spring Boot"
max_tokens = 20                       # 通过提示词限制输出长度
model_name = "kimi-k2-0905-preview"    # Kimi 流式预览模型
system_message = "You are a helpful assistant."
# ----------------------------------------------


class KimiChatError(Exception):
    """Kimi 专属异常"""
    pass


class KimiStream:
    """
    纯流式调用，支持 max_tokens 等全部额外参数
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise KimiChatError("缺少 MOONSHOT_API_KEY")

        base_url = (base_url or "https://api.moonshot.cn/v1").rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        # 默认值
        self.model = model if model is not None else model_name
        self.system = system if system is not None else system_message

    # ------------- 内部工具 -------------
    def _build_prompt_with_token_limit(self, base_prompt: str, token_limit: int) -> str:
        return f"{base_prompt}，用{token_limit}个token完成回复"

    # ------------- 核心流式生成器 -------------
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[str]:
        if max_tokens is not None:
            prompt = self._build_prompt_with_token_limit(prompt, max_tokens)

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
            raise KimiChatError(f"API 请求失败: {e}") from e

    # ------------- 带打印的便利封装 -------------
    def chat_stream(self, prompt: str, max_tokens: Optional[int] = None, **extra):
        print("正在向 Kimi 发送流式请求...")
        print("模型输出: ", end="", flush=True)

        usage = None
        try:
            for seg in self.stream(prompt, max_tokens=max_tokens, **extra):
                if isinstance(seg, dict):
                    usage = seg
                else:
                    print(seg, end="", flush=True)

            print("\n" + "=" * 50)
            self._print_usage_info(usage)
            return usage

        except Exception as e:
            print(f"\n发生错误: {e}")
            return None

    # ------------- 用量打印 -------------
    def _print_usage_info(self, usage_info: Optional[Dict]):
        if usage_info:
            print("请求完成 ✓")
            print("Token 使用情况:")
            print(f"  - 输入 Tokens: {usage_info['prompt_tokens']}")
            print(f"  - 输出 Tokens: {usage_info['completion_tokens']}")
            print(f"  - 总 Tokens: {usage_info['total_tokens']}")
        else:
            print("未能获取到使用信息。")


# -------------------- 运行入口 --------------------
if __name__ == "__main__":
    bot = KimiStream(model=model_name, system=system_message)
    bot.chat_stream(prompt=prompt, max_tokens=max_tokens)