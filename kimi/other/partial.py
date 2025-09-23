#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kimi 流式角色扮演测试
用法：
    python kimi_role_play_stream.py
    python kimi_role_play_stream.py --role "灰太狼" --friend "红太狼" --question "今晚抓不到羊怎么办？"
"""
import os
import sys
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# ---------- 0. 加载环境变量 ----------
load_dotenv()
API_KEY = os.getenv("MOONSHOT_API_KEY")
if not API_KEY:
    sys.exit("❗ 请先设置环境变量 MOONSHOT_API_KEY（支持 .env 文件）")

# ---------- 1. 客户端 ----------
client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.cn/v1")

# ---------- 2. 消息模板 ----------
def build_messages(role: str, friend: str, question: str):
    return [
        {
            "role": "system",
            "content": f"下面你来扮演{role}，你有一个特别好的朋友叫做{friend}，你们从小一起长大，一起冒险，你要用{role}的口吻来说话。",
        },
        {"role": "user", "content": question},
    ]

# ---------- 3. 流式对话 ----------
def chat_stream(role: str, friend: str, question: str):
    messages = build_messages(role, friend, question)
    stream = client.chat.completions.create(
        model="kimi-k2-0905-preview",
        messages=messages,
        temperature=0.6,
        max_tokens=65536,
        stream=True,
        stream_options={"include_usage": True},  # 末尾带 usage
    )

    print("🟢 Kimi 正在思考…\n")
    print("——— 流式输出开始 ———\n")

    usage = None
    for chunk in stream:
        # 1. 文本片段
        if chunk.choices and chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
        # 2. usage 信息（最后一块）
        if chunk.usage:
            usage = chunk.usage

    print("\n\n——— 流式输出结束 ———")
    if usage:
        print(
            f"📊 本次调用 token 用量 → "
            f"prompt: {usage.prompt_tokens}, "
            f"completion: {usage.completion_tokens}, "
            f"total: {usage.total_tokens}"
        )

# ---------- 4. 命令行入口 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kimi 流式角色扮演")
    parser.add_argument("--role", default="喜羊羊", help="你想让 Kimi 扮演的角色")
    parser.add_argument("--friend", default="懒羊羊", help="该角色的好朋友")
    parser.add_argument("--question", default="你怎么看待懒羊羊？", help="问他们的问题")
    args = parser.parse_args()

    chat_stream(args.role, args.friend, args.question)