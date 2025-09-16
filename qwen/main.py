import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 200  # 通过提示词限制输出token数
model_name = "qwen-plus"
system_message = "You are a helpful assistant."


class QwenStream:
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
        """
        初始化Qwen客户端

        Args:
            api_key: DASHSCOPE API密钥，如果为None则从环境变量获取
            base_url: API基础URL，如果为None则使用默认值
            model: 使用的模型名称，如果为None则使用默认值
            system: 系统消息，如果为None则使用默认值
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise QwenChatError("缺少 DASHSCOPE_API_KEY")
        
        base_url = (base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        # 设置默认值
        self.model = model if model is not None else model_name
        self.system = system if system is not None else system_message
    
    def _build_prompt_with_token_limit(self, base_prompt, token_limit):
        """
        将token限制集成到prompt中
        
        Args:
            base_prompt: 基础提示词
            token_limit: token限制数量
            
        Returns:
            完整的提示词
        """
        return f"{base_prompt}，用{token_limit}个token完成回复"

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[str]:
        """
        仅 yield 文本片段；最后 yield 一个 dict 带用量
        """
        # 如果指定了max_tokens，将其集成到提示词中
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
            raise QwenChatError(f"API 请求失败: {e}") from e
    
    def chat_stream(self, prompt: str, max_tokens: Optional[int] = None, **extra):
        """
        发送流式聊天请求并处理输出显示
        
        Args:
            prompt: 用户提示词
            max_tokens: 最大输出token数
            **extra: 额外参数
            
        Returns:
            使用信息字典或None
        """
        print("正在向模型发送请求并等待流式响应...")
        print("模型输出: ", end="", flush=True)
        
        try:
            usage = None
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
    
    def _print_usage_info(self, usage_info):
        """
        打印使用信息
        
        Args:
            usage_info: 使用信息字典
        """
        if usage_info:
            print("请求完成 ✓")
            print("Token 使用情况:")
            print(f"  - 输入 Tokens: {usage_info['prompt_tokens']}")
            print(f"  - 输出 Tokens: {usage_info['completion_tokens']}")
            print(f"  - 总 Tokens: {usage_info['total_tokens']}")
        else:
            print("未能获取到使用信息。")


# --- 使用示例 ---
if __name__ == "__main__":
    # 创建客户端实例（使用配置参数）
    bot = QwenStream(
        model=model_name,
        system=system_message
    )
    
    # 发送请求
    response = bot.chat_stream(
        prompt=prompt,
        max_tokens=max_tokens
    )