#!/usr/bin/env python3
"""测试 Paratera API 可用模型"""

import requests
import json

API_KEY = "sk-0oVqiF3DzxzxTcbxsaPEOg"
BASE_URL = "https://llmapi.paratera.com"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 测试 Embedding
print("测试 Embedding API...")
response = requests.post(
    f"{BASE_URL}/v1/embeddings",
    headers=headers,
    json={
        "model": "GLM-Embedding-3",
        "input": "测试文本",
        "encoding_format": "float"
    },
    timeout=30
)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    embedding = result["data"][0]["embedding"]
    print(f"✓ Embedding 成功，维度: {len(embedding)}")
else:
    print(f"✗ Error: {response.text[:200]}")

print()

# 测试不同 LLM 模型
models_to_test = [
    "Qwen3-235B-A22B-Instruct",
    "qwen3-235b-a22b-instruct",
    "glm-4-plus",
    "GLM-4-Plus",
    "deepseek-v3",
    "DeepSeek-V3",
    "Qwen3-32B",
    "qwen3-32b",
]

print("测试 Chat Completions API...")
for model in models_to_test:
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 50,
            "temperature": 0.0
        },
        timeout=30
    )
    status = "✓" if response.status_code == 200 else "✗"
    print(f"  {model}: {status} ({response.status_code})")
    if response.status_code == 200:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        print(f"    回复: {content[:50]}...")
        break
    elif response.status_code == 401:
        print(f"    未授权")
    else:
        print(f"    错误: {response.text[:100]}")

print()

# 测试 Rerank
print("测试 Rerank API...")
response = requests.post(
    f"{BASE_URL}/v1/rerank",
    headers=headers,
    json={
        "model": "GLM-Rerank",
        "query": "什么是机器学习",
        "documents": ["机器学习是人工智能的一个分支", "深度学习是机器学习的一种方法"],
        "top_k": 2
    },
    timeout=30
)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print(f"✓ Rerank 成功")
    print(f"  结果: {json.dumps(result, ensure_ascii=False, indent=2)[:200]}...")
else:
    print(f"✗ Error: {response.text[:200]}")
