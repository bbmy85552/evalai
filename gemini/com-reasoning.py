import os
from typing import Iterator, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 20
model_name = "gemini-2.5-flash"
system_message = "You are a helpful assistant."
reasoning_effort = "low"  # 在配置参数里设置 reasoning_effort


class GeminiChatError(Exception):
    pass


class GeminiStream:
    """
    支持 reasoning_effort 的Gemini流式调用
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        初始化Gemini客户端

        Args:
            api_key: Gemini API密钥
            base_url: API基础URL
            model: 使用的模型名称
            system: 系统消息
            reasoning_effort: 推理努力程度
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiChatError("缺少 GEMINI_API_KEY")
        
        base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        # 设置默认值
        self.model = model if model is not None else model_name
        self.system = system if system is not None else system_message
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else reasoning_effort

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        **extra,
    ) -> Iterator[str]:
        """
        流式调用，支持 reasoning_effort
        
        Args:
            prompt: 用户提示词
            max_tokens: 最大输出token数
            reasoning_effort: 推理努力程度
            **extra: 额外参数
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # 使用实例级别的 reasoning_effort 或传入的参数
            final_reasoning_effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
            
            request_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **extra
            }
            
            # 添加 reasoning_effort（如果提供了）
            if final_reasoning_effort is not None:
                request_params["reasoning_effort"] = final_reasoning_effort
            
            resp = self.client.chat.completions.create(**request_params)
            
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                
        except APIError as e:
            raise GeminiChatError(f"API 请求失败: {e}") from e
        except Exception as e:
            raise GeminiChatError(f"请求处理失败: {e}") from e
    
    def chat_stream(self, prompt: str, max_tokens: Optional[int] = None, 
                   reasoning_effort: Optional[str] = None, **extra):
        """
        发送流式聊天请求
        
        Args:
            prompt: 用户提示词
            max_tokens: 最大输出token数
            reasoning_effort: 推理努力程度
            **extra: 额外参数
        """
        print(f"正在向Gemini模型 {self.model} 发送请求...")
        if reasoning_effort or self.reasoning_effort:
            eff = reasoning_effort if reasoning_effort else self.reasoning_effort
            print(f"推理努力程度: {eff}")
        print("模型输出: ", end="", flush=True)
        
        try:
            full_response = ""
            for seg in self.stream(
                prompt, 
                max_tokens=max_tokens, 
                reasoning_effort=reasoning_effort,
                **extra
            ):
                print(seg, end="", flush=True)
                full_response += seg
            
            print("\n" + "=" * 50)
            print("请求完成 ✓")
            print(f"响应长度: {len(full_response)} 字符")
            
            return full_response
            
        except Exception as e:
            print(f"\n发生错误: {e}")
            return None


# --- 使用示例 ---
if __name__ == "__main__":
    # 创建Gemini客户端实例，在初始化时设置 reasoning_effort
    bot = GeminiStream(
        model=model_name,
        system=system_message,
        reasoning_effort=reasoning_effort  # 使用配置参数中的 reasoning_effort
    )
    
    # 示例1：使用配置的 reasoning_effort
    print("=== 使用配置的 reasoning_effort (low) ===")
    response = bot.chat_stream(
        prompt=prompt,
        max_tokens=max_tokens
    )
    
    # 示例2：覆盖配置的 reasoning_effort
    print("\n\n=== 覆盖 reasoning_effort (medium) ===")
    response2 = bot.chat_stream(
        prompt="解释一下机器学习的基本概念",
        reasoning_effort="medium"  # 覆盖默认配置
    )
    
    # 示例3：使用 high reasoning_effort
    print("\n\n=== 使用 high reasoning_effort ===")
    response3 = bot.chat_stream(
        prompt="详细分析人工智能的未来发展趋势和挑战",
        reasoning_effort="high"
    )