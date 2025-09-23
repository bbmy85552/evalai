import os
import datetime
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 基础配置（仅保留必要参数，按需修改） ---
load_dotenv()  # 加载.env文件中的API密钥

# ===================== 必要配置（仅改这里） =====================
API_KEY = os.getenv("ZHIPU_API_KEY")  # 智谱API密钥（.env中配置ZHIPU_API_KEY）
MODEL_NAME = "glm-4-flash"  # 模型名称
USER_PROMPT = "讲一下什么是Spring Boot"  # 用户提问
MAX_TOKENS = 20  # 模型输出Token上限（避免生成过长）
SYSTEM_MESSAGE = "You are a helpful assistant."  # 系统指令（引导模型）
# ==================================================================

# --- 固定配置（无需修改：强制流式，确保返回tokens） ---
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"  # GLM固定API地址
ENABLE_STREAM = True  # 强制开启流式（GLM流式自动返回tokens）


class GLMStreamTokenClient:
    def __init__(self):
        """初始化客户端：仅加载必要配置，强制流式"""
        # 校验API密钥
        if not API_KEY:
            raise ValueError("请在.env文件中添加 ZHIPU_API_KEY=你的智谱API密钥")
        
        # 初始化GLM客户端（兼容OpenAI SDK）
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def _get_chat_messages(self) -> list:
        """组装对话结构（符合GLM API格式）"""
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_PROMPT}
        ]

    def _print_core_info(self, final_chunk: any, full_content: str) -> None:
        """打印核心信息（与需求格式对齐：请求完成/ID/Token使用情况）"""
        print("\n" + "="*50)
        print(f"请求完成 ✓")
        print(f"请求 ID: {final_chunk.id if (final_chunk and hasattr(final_chunk, 'id')) else '未知'}")
        print(f"创建时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"响应字符数: {len(full_content)} | 配置输出上限: {MAX_TOKENS}")
        print("Token 使用情况:")
        
        # 提取GLM返回的真实tokens（流式最后一块含usage）
        if final_chunk and hasattr(final_chunk, 'usage') and final_chunk.usage:
            usage = final_chunk.usage
            print(f"  - 输入 Tokens: {usage.prompt_tokens}")
            print(f"  - 输出 Tokens: {usage.completion_tokens}")
            print(f"  - 总 Tokens: {usage.total_tokens}")
        else:
            # 极端情况备用：本地估算（仅作参考）
            import tiktoken  # 按需加载，减少初始化依赖
            encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(SYSTEM_MESSAGE + USER_PROMPT))
            output_tokens = len(encoding.encode(full_content))
            print(f"  - 输入 Tokens: 估算{input_tokens}")
            print(f"  - 输出 Tokens: 估算{output_tokens}")
            print(f"  - 总 Tokens: 估算{input_tokens + output_tokens}")

    def stream_with_tokens(self) -> None:
        """核心方法：仅流式返回，自动获取并打印核心信息"""
        print("正在向模型发送请求并等待流式响应...")
        print("模型输出: ", end="", flush=True)
        
        try:
            # 1. 构建流式请求参数（强制stream=True）
            response_stream = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=self._get_chat_messages(),
                stream=ENABLE_STREAM,  # 固定流式，确保返回tokens
                max_tokens=MAX_TOKENS,  # 控制输出长度
                temperature=0.7  # 随机性（按需调整0-1）
            )
            
            # 2. 处理流式响应：实时打印+记录最后一块（含tokens）
            full_content = ""  # 存储完整回复
            final_chunk = None  # 存储最后一块（关键：含usage）
            
            for chunk in response_stream:
                # 实时打印模型输出（仅处理文本增量）
                if chunk.choices and chunk.choices[0].delta.content:
                    content_delta = chunk.choices[0].delta.content
                    print(content_delta, end="", flush=True)
                    full_content += content_delta
                # 记录最后一块（GLM真实tokens在最后一块）
                final_chunk = chunk
            
            # 3. 打印核心信息（与需求格式一致）
            self._print_core_info(final_chunk, full_content)
        
        except Exception as e:
            print(f"\n发生错误: {e}")


# --- 一键运行：直接流式返回并显示核心信息 ---
if __name__ == "__main__":
    client = GLMStreamTokenClient()
    client.stream_with_tokens()

    '''
    PS E:\ai-test\evalai-main\evalai-main> python glm/main.py
正在向模型发送请求并等待流式响应...
模型输出: Spring Boot 是一个开源的Java框架，它旨在简化Spring应用的创建和部署过程。
==================================================
请求完成 ✓
请求 ID: 20250921195825ca14db977a4445d5
创建时间: 2025-09-21 19:58:24
响应字符数: 47 | 配置输出上限: 20
Token 使用情况:
  - 输入 Tokens: 18
  - 输出 Tokens: 20
  - 总 Tokens: 38
  '''