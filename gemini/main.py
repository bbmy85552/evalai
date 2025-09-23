import os
import datetime
from typing import Optional, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是Spring Boot"
max_tokens = 18 #原定token为20，考虑到部分超过阈值，调整为18
model_name = "gemini-2.0-flash"
system_message = "你是一个智能助手"
ENABLE_STREAM_OUTPUT = True  # 控制是否流式输出内容（True=流式展示，False=一次性展示）


class GeminiClient:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 base_url: Optional[str] = None,
                 model: Optional[str] = None, 
                 system: Optional[str] = None):
        """初始化Gemini客户端（兼容流式输出与Token统计）"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API密钥未提供且环境变量中未找到GEMINI_API_KEY")
        
        # Gemini的OpenAI兼容API基础URL
        self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 设置默认参数
        self.model = model if model is not None else model_name
        self.system = system if system is not None else system_message

    def _get_chat_messages(self, prompt: str) -> list:
        """组装对话结构"""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]

    def _print_token_info(self, response: Any, full_content: str) -> None:
        """打印核心信息（含Token统计）"""
        print("\n" + "=" * 50)
        print(f"请求完成 ✓")
        print(f"请求 ID: {response.id if hasattr(response, 'id') else '未知'}")
        print(f"创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"响应字符数: {len(full_content)} | 配置输出上限: {max_tokens}")
        print("Token 使用情况:")
        
        # 提取非流式响应中的真实Token数据
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            print(f"  - 输入 Tokens: {usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else '未知'}")
            print(f"  - 输出 Tokens: {usage.completion_tokens if hasattr(usage, 'completion_tokens') else '未知'}")
            print(f"  - 总 Tokens: {usage.total_tokens if hasattr(usage, 'total_tokens') else '未知'}")
        else:
            print("  - 输入 Tokens: 无可用信息")
            print("  - 输出 Tokens: 无可用信息")
            print("  - 总 Tokens: 无可用信息")

    def chat_with_tokens(self, prompt: str, max_tokens: int) -> None:
        """核心方法：根据配置自动选择流式/非流式，确保Token信息获取"""
        try:
            # 1. 先发送非流式请求获取完整响应（含Token信息）
            print("正在获取完整响应（用于Token统计）...")
            full_response = self.client.chat.completions.create(
                model=self.model,
                messages=self._get_chat_messages(prompt),
                stream=False,  # 非流式确保获取usage
                max_tokens=max_tokens,
                temperature=0.7
            )
            full_content = full_response.choices[0].message.content or ""

            # 2. 根据配置决定输出方式
            if ENABLE_STREAM_OUTPUT:
                # 流式展示内容（提升体验）
                print("\n正在流式展示内容...")
                print("模型输出: ", end="", flush=True)
                for char in full_content:
                    print(char, end="", flush=True)
            else:
                # 非流式一次性展示
                print("\n模型输出: ", full_content)

            # 3. 打印Token信息（基于非流式响应的真实数据）
            self._print_token_info(full_response, full_content)

        except Exception as e:
            print(f"\n发生错误: {type(e).__name__}-{str(e)}")


# --- 运行示例 ---
if __name__ == "__main__":
    client = GeminiClient(model=model_name, system=system_message)
    client.chat_with_tokens(prompt=prompt, max_tokens=max_tokens)
    

    '''
    PS E:\ai-test\evalai-main\evalai-main> python gemini/main.py
正在获取完整响应（用于Token统计）...

正在流式展示内容...
模型输出: 好的，Spring Boot 是一个用于简化 Spring 应用开发的框架。它让你能够快速创建
==================================================
请求完成 ✓
请求 ID: je7PaOTgCYSL6dkP1s2-kAY
创建时间: 2025-09-21 20:24:45
响应字符数: 47 | 配置输出上限: 18
Token 使用情况:
  - 输入 Tokens: 9
  - 输出 Tokens: 20
  - 总 Tokens: 29
    '''