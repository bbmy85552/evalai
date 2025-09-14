import os
from typing import Iterator, Dict, Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 200  # 最大输出token数
model_name = "deepseek-chat"  # 使用普通聊天模型，不是reasoner模型
system_message = "You are a helpful assistant."


class DeepSeekChatError(Exception):
    pass


class DeepSeekStream:
    """
    DeepSeek 流式调用类（无推理过程版本）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ):
        """
        初始化DeepSeek客户端
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise DeepSeekChatError("缺少 DEEPSEEK_API_KEY")
        
        base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        # 设置默认值
        self.model = model if model is not None else model_name
        self.system = system if system is not None else system_message
        
        # 初始化编码器用于token计数
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
            print("警告: 无法初始化token编码器，将使用估算方法")
    
    def count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # 简单估算
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            other_chars = len(text) - chinese_chars
            return int(chinese_chars * 1.5 + other_chars * 0.25)
    
    def calculate_input_tokens(self, messages: list) -> int:
        """计算输入消息的token数量"""
        total_tokens = 0
        for message in messages:
            total_tokens += self.count_tokens(message["content"])
            total_tokens += 5  # 角色token估算
        return total_tokens
    
    def stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Iterator[Dict]:
        """
        流式响应生成器（无推理过程）
        """
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        
        # 计算输入token数量
        input_tokens = self.calculate_input_tokens(messages)
        yield {"type": "input_tokens", "value": input_tokens}
        
        try:
            # 准备请求参数
            params = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **extra
            }
            
            # 添加max_tokens参数（如果提供）
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            resp = self.client.chat.completions.create(**params)
            
            content = ""
            output_tokens = 0
            
            for chunk in resp:
                # 只处理回复内容（没有推理内容）
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content_chunk = chunk.choices[0].delta.content
                    content += content_chunk
                    token_count = self.count_tokens(content_chunk)
                    output_tokens += token_count
                    yield {
                        "type": "content", 
                        "content": content_chunk,
                        "tokens": token_count
                    }
                
                # 处理使用信息（如果有）
                if hasattr(chunk, 'usage') and chunk.usage:
                    yield {
                        "type": "usage",
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
            
            # 如果没有从API获取到usage信息，使用我们的估算
            yield {
                "type": "estimated_usage",
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
                    
        except APIError as e:
            raise DeepSeekChatError(f"API 请求失败: {e}") from e
    
    def chat_stream(self, prompt: str, max_tokens: Optional[int] = None, **extra):
        """
        发送流式聊天请求（无推理过程）
        """
        print("正在向DeepSeek模型发送请求并等待流式响应...")
        print("=" * 50)
        
        try:
            usage = None
            content_text = ""
            content_tokens = 0
            input_tokens = 0
            
            for segment in self.stream(prompt, max_tokens=max_tokens, **extra):
                if segment["type"] == "input_tokens":
                    input_tokens = segment["value"]
                    print(f"输入Token估算: {input_tokens}")
                
                elif segment["type"] == "content":
                    content_text += segment["content"]
                    content_tokens += segment["tokens"]
                    print(f"\033[92m{segment['content']}\033[0m", end="", flush=True)
                
                elif segment["type"] == "usage":
                    usage = segment
                    print("\n" + "=" * 50)
                    self._print_usage_info(usage)
                
                elif segment["type"] == "estimated_usage" and usage is None:
                    usage = segment
                    print("\n" + "=" * 50)
                    print("注意: 使用估算的Token数量")
                    self._print_usage_info(usage)
            
            # 保存完整的回复内容
            self.last_content = content_text
            self.last_content_tokens = content_tokens
            self.last_input_tokens = input_tokens
            
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
    
    def get_last_response(self):
        """获取最后一次请求的完整响应"""
        return {
            "content": getattr(self, 'last_content', ''),
            "input_tokens": getattr(self, 'last_input_tokens', 0),
            "output_tokens": getattr(self, 'last_content_tokens', 0),
        }


# --- 使用示例 ---
if __name__ == "__main__":
    # 创建DeepSeek客户端实例（使用deepseek-chat模型，无推理过程）
    bot = DeepSeekStream(
        model="deepseek-chat",  # 确保使用普通聊天模型
        system=system_message
    )
    
    # 发送请求
    response = bot.chat_stream(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    
    # 获取完整响应
    full_response = bot.get_last_response()
    
    print("\n完整回复内容:")
    print(full_response["content"])
    
    print("\n详细Token统计:")
    print(f"输入Tokens: {full_response['input_tokens']}")
    print(f"输出Tokens: {full_response['output_tokens']}")
    print(f"总Tokens: {full_response['input_tokens'] + full_response['output_tokens']}")