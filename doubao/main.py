import os
import datetime
from typing import Optional, Any
from openai import OpenAI, APIError
from dotenv import load_dotenv

# --- 环境变量加载 ---
load_dotenv()

# ===================== 基础配置区（按需修改以下参数） =====================
MODEL_NAME = "doubao-seed-1-6-250615"  # 调用的模型ID/Endpoint ID
SYSTEM_PROMPT = "You are a helpful assistant."  # 系统指令（模型遵循的角色定位）
USER_PROMPT = "讲一下什么是Springboot"  # 用户提问（测试SSR相关内容）
WORD_LIMIT = 200  # 字数限制（参考gpt/main.py的实现方式）
# ==========================================================================

# --- 固定配置（基于火山方舟Chat API文档设置） ---
API_KEY = os.getenv("ARK_API_KEY")  # 火山方舟API密钥（从.env文件读取）
TEMPERATURE = 0.7  # 采样温度（0-2，值越低输出越确定）
TOP_P = 0.9  # 核采样阈值（0-1，与temperature二选一调整随机性）
ENABLE_STREAM = True  # 开启流式响应（符合测试需求）
STREAM_INCLUDE_USAGE = True  # 流式响应中包含Token用量统计（文档stream_options.include_usage）
RESPONSE_FORMAT = {"type": "text"}  # 响应格式（文档response_format，默认文本格式）


class VolcanoArkChatClient:
    def __init__(self):
        """初始化客户端（基于火山方舟Chat API规范）"""
        # 校验API Key（文档要求必选）
        if not API_KEY:
            raise ValueError("请在.env文件中配置ARK_API_KEY（火山方舟API密钥）")
        
        # 初始化OpenAI风格客户端（火山方舟兼容该SDK，文档推荐调用方式）
        self.client = OpenAI(
            api_key=API_KEY,
            base_url="https://ark.cn-beijing.volces.com/api/v3"  # 火山方舟固定BaseURL
        )
        
        self.full_content = ""  # 存储流式响应拼接的完整内容

    def _build_prompt_with_word_limit(self, base_prompt: str, word_limit: int) -> str:
        """
        将字数限制集成到prompt中（参考gpt/main.py的实现方式）
        
        Args:
            base_prompt: 基础提示词
            word_limit: 字数限制
            
        Returns:
            完整的提示词
        """
        return f"{base_prompt}，用{word_limit}字完成回复"

    def _build_chat_messages(self, word_limit: int = WORD_LIMIT) -> list[dict]:
        """构建API要求的messages结构（文档messages参数规范：system+user角色）"""
        # 构建包含字数限制的完整提示词（参考gpt/main.py的实现方式）
        full_user_prompt = self._build_prompt_with_word_limit(USER_PROMPT, word_limit)
        
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT  # 系统指令（文档中system角色必选，定义模型行为）
            },
            {
                "role": "user",
                "content": full_user_prompt  # 用户提问（包含字数限制）
            }
        ]

    def stream_chat_request(self, word_limit: int = WORD_LIMIT) -> Optional[str]:
        """核心方法：发送流式Chat请求（完全遵循文档参数规范）"""
        print("正在向模型发送请求并等待流式响应...")
        
        # 1. 构建API请求参数（严格对应文档中Chat API的请求体字段）
        request_params = {
            "model": MODEL_NAME,  # 必选，模型ID/Endpoint ID（文档model参数）
            "messages": self._build_chat_messages(word_limit),  # 必选，对话消息列表（文档messages参数）
            "stream": ENABLE_STREAM,  # 必选，开启流式（文档stream参数）
            "stream_options": {"include_usage": STREAM_INCLUDE_USAGE},  # 流式包含用量（文档stream_options）
            "temperature": TEMPERATURE,  # 可选，采样温度（文档temperature参数）
            "top_p": TOP_P,  # 可选，核采样（文档top_p参数）
            "response_format": RESPONSE_FORMAT  # 可选，响应格式（文档response_format参数）
        }

        try:
            self.full_content = ""  # 重置完整内容存储
            usage_info = None  # 存储usage信息（参考test.py的方式）
            final_chunk = None  # 存储最后一个Chunk（含完整usage用量数据，文档stream结束标识）

            # 2. 发送流式请求并处理响应（文档中流式响应按SSE协议逐块返回）
            print("模型输出: ", end="", flush=True)
            for chunk in self.client.chat.completions.create(** request_params):
                # 拼接流式增量内容（文档中delta.content为逐块文本）
                if chunk.choices and chunk.choices[0].delta.content:
                    content_delta = chunk.choices[0].delta.content
                    print(content_delta, end="", flush=True)  # 实时打印流式输出
                    self.full_content += content_delta
                
                # 获取usage信息（参考test.py的方式，通常在最后一个chunk中）
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage
                    
                # 记录最后一个Chunk（包含完整的usage和id信息，文档stream结束时返回）
                final_chunk = chunk

            # 3. 打印响应总结（匹配目标输出格式）
            print("\n" + "=" * 50)
            self._print_response_summary(final_chunk, usage_info)
            return self.full_content

        except APIError as e:
            # API错误处理（文档中提及的APIError场景，如密钥无效、模型不存在）
            print(f"\nAPI请求失败（遵循文档错误场景）: {e}")
            return None
        except Exception as e:
            # 其他未知错误处理
            print(f"\n未知错误: {e}")
            return None

    def _print_response_summary(self, final_chunk: Any, usage_info: Any) -> None:
        """打印响应总结（包含文档中API返回的核心字段：id、created、usage）"""
        # 1. 获取请求ID（文档中id字段，唯一标识请求）
        request_id = final_chunk.id if (final_chunk and hasattr(final_chunk, 'id')) else "未知"
        
        # 2. 获取创建时间（文档中created字段，Unix时间戳转换为可读格式）
        if final_chunk and hasattr(final_chunk, 'created'):
            created_time = datetime.datetime.fromtimestamp(final_chunk.created).strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 3. 打印基础信息（匹配目标输出格式）
        print(f"请求完成 ✓")
        print(f"请求 ID: {request_id}")
        print(f"创建时间: {created_time}")
        print("Token 使用情况:")

        # 4. 处理Token用量（参考test.py的方式，优先使用流式响应中获取的usage_info）
        if usage_info:
            print(f"  - 输入 Tokens: {usage_info.prompt_tokens}")  # 文档usage.prompt_tokens（输入Token）
            print(f"  - 输出 Tokens: {usage_info.completion_tokens}")  # 文档usage.completion_tokens（输出Token）
            print(f"  - 总 Tokens: {usage_info.total_tokens}")  # 文档usage.total_tokens（总Token）
        else:
            print("  - ⚠️  未获取到token使用信息")


# --- 执行测试（流式响应调用入口） ---
if __name__ == "__main__":
    chat_client = VolcanoArkChatClient()
    chat_client.stream_chat_request()
