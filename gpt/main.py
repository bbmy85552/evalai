import os
import datetime
from openai import OpenAI

# --- (建议) 使用环境变量管理 API Key，更安全 ---
from dotenv import load_dotenv

load_dotenv()

# --- 配置参数 ---
prompt = "讲一下什么是ssr，前端的"
word_limit = 200  # 字数限制
model_name = "gpt-5-nano"
enable_reasoning = True  # 是否启用思考
reasoning_effort = "minimal"  # 思考程度：minimal, medium, high


class OpenAIClient:
    def __init__(self, api_key=None, model=None, enable_reasoning=None,
                 reasoning_effort=None):
        """
        初始化OpenAI客户端

        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的模型名称，如果为None则使用默认值
            enable_reasoning: 是否启用推理思考，如果为None则使用默认值
            reasoning_effort: 推理思考的程度，如果为None则使用默认值
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

        # 设置默认值
        self.model = model if model is not None else model_name
        self.enable_reasoning = enable_reasoning if enable_reasoning is not None else enable_reasoning
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else reasoning_effort
    
    def _build_prompt_with_word_limit(self, base_prompt, word_limit):
        """
        将字数限制集成到prompt中
        
        Args:
            base_prompt: 基础提示词
            word_limit: 字数限制
            
        Returns:
            完整的提示词
        """
        return f"{base_prompt}，用{word_limit}字完成回复"
    
    def chat_stream(self, prompt, word_limit=word_limit, instructions=None):
        """
        发送流式聊天请求
        
        Args:
            prompt: 用户提示词
            word_limit: 字数限制
            instructions: 额外指令
            
        Returns:
            最终响应对象或None
        """
        print("正在向模型发送请求并等待流式响应...")
        
        # 构建完整的提示词（包含字数限制）
        full_prompt = self._build_prompt_with_word_limit(prompt, word_limit)
        
        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "input": [
                    {
                        "role": "user",
                        "content": full_prompt,
                    },
                ],
                "stream": True,
            }
            
            # 根据设置添加推理参数
            if self.enable_reasoning:
                request_params["reasoning"] = {"effort": self.reasoning_effort}
            
            # 如果有额外指令，添加到请求中
            if instructions:
                request_params["instructions"] = instructions
            
            response_stream = self.client.responses.create(**request_params)
            
            # 处理流式响应
            final_response = None
            print("模型输出: ", end="", flush=True)
            
            for event in response_stream:
                # 检查事件类型是否为文本增量
                if event.type == "response.output_text.delta":
                    # 直接打印出增量内容，不换行
                    print(event.delta, end="", flush=True)
                
                # 检查事件类型是否为请求完成
                elif event.type == "response.completed":
                    # 保存最终的响应对象以备后用
                    final_response = event.response
            
            # 循环结束后，打印一个换行符，让格式更美观
            print("\n" + "=" * 50)
            
            # 打印响应元数据
            self._print_response_info(final_response)
            
            return final_response
            
        except Exception as e:
            print(f"\n发生错误: {e}")
            return None
    
    def _print_response_info(self, final_response):
        """
        打印响应信息
        
        Args:
            final_response: 最终响应对象
        """
        if final_response:
            # 提取所需信息
            request_id = final_response.id
            created_timestamp = final_response.created_at
            usage_info = final_response.usage
            
            # 将 Unix 时间戳转换为可读格式
            created_time = datetime.datetime.fromtimestamp(created_timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            
            print(f"请求完成 ✓")
            print(f"请求 ID: {request_id}")
            print(f"创建时间: {created_time}")
            print("Token 使用情况:")
            print(f"  - 输入 Tokens: {usage_info.input_tokens}")
            print(f"  - 输出 Tokens: {usage_info.output_tokens}")
            print(f"  - 总 Tokens: {usage_info.total_tokens}")
        else:
            print("未能获取到最终响应信息。")


# --- 使用示例 ---
if __name__ == "__main__":
    # 创建客户端实例（必须传参）
    ai_client = OpenAIClient(
        model=model_name,
        enable_reasoning=enable_reasoning,
        reasoning_effort=reasoning_effort
    )

    # 发送请求
    response = ai_client.chat_stream(
        prompt=prompt,
        word_limit=word_limit
    )



"""
(evalai) 2dqy003@2dqy003deMac-mini evalai % python gpt/main.py     
正在向模型发送请求并等待流式响应...
模型输出: SSR（Server-Side Rendering，服务器端渲染）是指将网页的初始 HTML 由服务器生成并直接返回给浏览器，而不是在浏览器端通过 JavaScript 逐步渲染。前端领域常见两种渲染模式：CSR（客户端渲染）和 SSR。优点是首屏加载快、对 SEO友好、初始渲染稳定；缺点是服务端压力增大、实现复杂度高、缓存和数据同步需要额外处理。工作流程大致是：浏览器请求页面，服务器根据请求获取数据、渲染成 HTML（可预取数据），发送给浏览器，浏览器接收到后完成客户端的交互逻辑。常用实现方案有 React/Vue 的服务端渲染（如 Next.js、Nuxt.js）以及自定义 SSR。一些场景也会采用混合渲染（同时有 SSR 与 CSR），以兼顾首屏性能和后续交互体验。
==================================================
请求完成 ✓
请求 ID: resp_68c10db4876c819fab479f1a294cc61a0b17e6cab2120add
创建时间: 2025-09-10 13:33:40
Token 使用情况:
  - 输入 Tokens: 21
  - 输出 Tokens: 223
  - 总 Tokens: 244
"""
