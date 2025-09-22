import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 100
model_name = "deepseek-chat"
system_message = "You are a helpful assistant."


class DeepSeekChatError(Exception):
    pass


class DeepSeekChatStream:
    """
    DeepSeek Chat 流式调用
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekChatError("缺少 DEEPSEEK_API_KEY")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url or "https://api.deepseek.com")
        self.model = model or model_name
        self.system = system or system_message
    
    def _build_prompt_with_token_limit(self, base_prompt, token_limit):
        """将token限制集成到prompt中"""
        return f"{base_prompt}，用{token_limit}个token完成回复"

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[str]:
        """
        流式输出文本片段；最后返回使用信息
        """
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
            raise DeepSeekChatError(f"API请求失败: {e}")

    def chat_stream(self, prompt: str, max_tokens: Optional[int] = None, **extra):
        """
        发送流式聊天请求并处理输出显示
        """
        print("正在向DeepSeek Chat发送请求并等待流式响应...")
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
        """打印使用信息"""
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
    bot = DeepSeekChatStream()
    
    # 发送请求
    response = bot.chat_stream(
        prompt=prompt,
        max_tokens=max_tokens
    )

    '''
    PS E:\ai-test\evalai-main\evalai-main> python deepseek/chat-main.py
正在向DeepSeek Chat发送请求并等待流式响应...
模型输出: Spring Boot是一个基于Spring框架的简化开发工具，通过自动配置和约定优于配置的原则，快速创建独立、生产级的Spring应用。它内置了嵌入式服务器（如Tomcat），无需部署WAR文件，并提供starter
依赖简化Maven/Gradle配置，大幅减少开发时间和代码量。支持微服务架构，适用于REST API、Web应用等场景。
==================================================
请求完成 ✓
Token 使用情况:
  - 输入 Tokens: 22
  - 输出 Tokens: 77
  - 总 Tokens: 99
'''