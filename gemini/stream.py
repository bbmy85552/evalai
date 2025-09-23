import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 20  # 直接使用max_tokens参数限制输出
model_name = "gemini-2.0-flash"
system_message = "You are a helpful assistant."


class GeminiChatError(Exception):
    pass


class GeminiStream:
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
        初始化Gemini客户端

        Args:
            api_key: Gemini API密钥，如果为None则从环境变量获取
            base_url: API基础URL，如果为None则使用默认值
            model: 使用的模型名称，如果为None则使用默认值
            system: 系统消息，如果为None则使用默认值
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiChatError("缺少 GEMINI_API_KEY")
        
        # Gemini的OpenAI兼容端点
        base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        # 设置默认值
        self.model = model if model is not None else model_name
        self.system = system if system is not None else system_message

    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[str]:
        """
        仅 yield 文本片段
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        
        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **extra
            }
            
            # 添加max_tokens参数（如果提供了）
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            
            resp = self.client.chat.completions.create(**request_params)
            
            for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                
        except APIError as e:
            raise GeminiChatError(f"API 请求失败: {e}") from e
    
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
        if max_tokens is not None:
            print(f"最大输出token数: {max_tokens}")
        print("模型输出: ", end="", flush=True)
        
        try:
            full_response = ""
            for seg in self.stream(prompt, max_tokens=max_tokens, **extra):
                print(seg, end="", flush=True)
                full_response += seg
            
            print("\n" + "=" * 50)
            self._print_response_info(full_response, max_tokens)
            return {"response_length": len(full_response)}
            
        except Exception as e:
            print(f"\n发生错误: {e}")
            return None
    
    def _print_response_info(self, response_text, max_tokens=None):
        """
        打印响应信息
        
        Args:
            response_text: 响应文本
            max_tokens: 设置的max_tokens限制
        """
        if response_text:
            print("请求完成 ✓")
            print("响应信息:")
            print(f"  - 响应长度: {len(response_text)} 字符")
            if max_tokens is not None:
                print(f"  - Token限制: {max_tokens}")
            print(f"  - 响应内容: {response_text[:100]}...")  # 显示前100个字符
        else:
            print("未能获取到响应信息。")
    
    def get_usage_info(self, prompt: str, max_tokens: Optional[int] = None, **extra):
        """
        通过非流式请求获取usage信息
        
        Args:
            prompt: 用户提示词
            max_tokens: 最大输出token数
            **extra: 额外参数
            
        Returns:
            使用信息字典或None
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        
        try:
            request_params = {
                "model": self.model,
                "messages": messages,
                **extra
            }
            
            # 添加max_tokens参数（如果提供了）
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**request_params)
            
            if hasattr(response, 'usage'):
                return {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return None
            
        except Exception as e:
            print(f"获取usage信息失败: {e}")
            return None


# --- 使用示例 ---
if __name__ == "__main__":
    # 创建客户端实例（使用配置参数）
    bot = GeminiStream(
        model=model_name,
        system=system_message
    )
    
    # 测试1：不使用max_tokens限制
    print("=== 测试1：无token限制 ===")
    response1 = bot.chat_stream(
        prompt=prompt
    )
    
    # 测试2：使用max_tokens限制
    print("\n\n=== 测试2：限制20个token ===")
    response2 = bot.chat_stream(
        prompt=prompt,
        max_tokens=20  # 直接使用max_tokens参数
    )
    
    # 测试3：更严格的token限制
    print("\n\n=== 测试3：限制10个token ===")
    response3 = bot.chat_stream(
        prompt=prompt,
        max_tokens=10  # 更严格的限制
    )
    
    # 获取usage信息
    print("\n=== 获取Usage信息 ===")
    usage_info = bot.get_usage_info(
        prompt=prompt,
        max_tokens=20
    )
    
    if usage_info:
        print("Token 使用情况:")
        print(f"  - 输入 Tokens: {usage_info['prompt_tokens']}")
        print(f"  - 输出 Tokens: {usage_info['completion_tokens']}")
        print(f"  - 总 Tokens: {usage_info['total_tokens']}")
    else:
        print("未能获取到Token使用信息")