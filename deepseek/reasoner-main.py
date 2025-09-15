import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 20
model_name = "deepseek-reasoner"
system_message = "You are a helpful assistant."


class DeepSeekReasonerError(Exception):
    pass


class DeepSeekReasonerStream:
    """
    DeepSeek Reasoner 流式调用
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
            raise DeepSeekReasonerError("缺少 DEEPSEEK_API_KEY")
        
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
    ) -> Iterator[Dict]:
        """流式输出推理过程和最终答案"""
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
                **extra,
            )
            
            for chunk in resp:
                delta = chunk.choices[0].delta
                result = {}
                
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    result['reasoning'] = delta.reasoning_content
                
                if hasattr(delta, 'content') and delta.content:
                    result['content'] = delta.content
                
                if result:
                    yield result
                    
        except APIError as e:
            raise DeepSeekReasonerError(f"API请求失败: {e}")

    def chat_stream(self, prompt: str, max_tokens: Optional[int] = None, **extra):
        """发送流式聊天请求并显示输出"""
        print("正在向DeepSeek Reasoner发送请求并等待流式响应...")
        print("模型输出: ", end="", flush=True)
        
        try:
            for segment in self.stream(prompt, max_tokens=max_tokens, **extra):
                if 'reasoning' in segment:
                    print(f"\033[1;32m{segment['reasoning']}\033[0m", end="", flush=True)
                
                if 'content' in segment:
                    print(f"{segment['content']}", end="", flush=True)
            
            print("\n" + "=" * 50)
            print("请求完成 ✓")
            
        except Exception as e:
            print(f"\n发生错误: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    bot = DeepSeekReasonerStream()
    
    # 测试中文问题
    response = bot.chat_stream(
        prompt=prompt,
        max_tokens=max_tokens
    )