import os
from openai import OpenAI

# 配置豆包API（火山方舟）
def setup_doubao_client():
    """配置豆包客户端"""
    # 从环境变量获取API Key，或直接填写
    api_key =  "12e95067-cd7e-4b65-bdd7-84207558fa24"
    
    # 火山方舟的Base URL
    base_url = "https://ark.cn-beijing.volces.com/api/v3"
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client

def stream_chat_with_usage():
    """流式输出并统计token使用量"""
    # 初始化客户端
    client = setup_doubao_client()
    
    # 这里需要替换为您的实际模型ID（Endpoint ID）
    # 可以在火山方舟控制台的"在线推理"页面获取
    model_id = "doubao-seed-1-6-250615"  # 替换为您的实际Endpoint ID
    
    try:
        # 创建流式请求
        stream = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是豆包助手，由字节跳动开发的AI助手"},
                {"role": "user", "content": "解释什么是深度学习，请详细说明其原理和应用"}
            ],
            stream=True,
            stream_options={"include_usage": True},  # 启用token统计
            temperature=0.7,
            max_tokens=2000
        )
        
        content = ""
        usage_info = None
        
        print("🤖 豆包回复:")
        print("-" * 50)
        
        # 处理流式响应
        for chunk in stream:
            # 处理内容流
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content_piece = delta.content
                    content += content_piece
                    print(content_piece, end="", flush=True)
            
            # 获取usage信息（通常在最后一个chunk中）
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_info = chunk.usage
        
        print("\n")
        print("-" * 50)
        print("📊 Token 使用统计")
        print("-" * 50)
        
        if usage_info:
            print(f"📥 输入 tokens: {usage_info.prompt_tokens}")
            print(f"📤 输出 tokens: {usage_info.completion_tokens}")
            print(f"📊 总计 tokens: {usage_info.total_tokens}")
        else:
            print("⚠️  未获取到token使用信息")
        
        return content, usage_info
        
    except Exception as e:
        print(f"❌ 调用失败: {str(e)}")
        print("\n🔧 请检查以下配置:")
        print("1. API Key 是否正确设置")
        print("2. 模型ID (Endpoint ID) 是否正确")
        print("3. 网络连接是否正常")
        return None, None


def main():
    """主函数"""
    print("🌟 豆包流式输出演示")
    print("=" * 60)
    
    # 检查API Key配置
    api_key = "12e95067-cd7e-4b65-bdd7-84207558fa24"
        

    print("\n🔄 开始流式对话...")
    
    # 执行流式聊天
    content, usage = stream_chat_with_usage()
    
    if content:
        print("\n" + "=" * 60)
        print("✨ 对话完成!")
        print(f"📝 总共生成内容长度: {len(content)} 字符")
    else:
        print("\n❌ 对话失败，请检查配置")


if __name__ == "__main__":
    main()
    

