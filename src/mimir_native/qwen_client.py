"""
Qwen Embedding Client for Mimir Memory V2
使用阿里云 DashScope API - OpenAI 兼容模式
"""
import json
import requests
from typing import List, Optional


class QwenEmbeddingClient:
    """
    Qwen3-Embedding-8B / DashScope 客户端
    支持 32-4096 维自定义，支持 100+ 语言
    使用 OpenAI 兼容 API
    """
    
    def __init__(self, api_key: str = None, model: str = "text-embedding-v3"):
        self.api_key = api_key or "sk-wzxwqcbzhzcpgszxerezpoletnmdodkvjxgegihyayohonkc"
        self.model = model  # text-embedding-v3 或 text-embedding-v4
        # 中国区域使用 aliyuncs.com，国际区域使用 dashscope-intl.aliyuncs.com
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed(self, text: str, dimensions: int = 1536) -> Optional[List[float]]:
        """
        获取单个文本的 embedding
        
        Args:
            text: 输入文本
            dimensions: 输出维度 (32-4096，默认1536匹配当前表)
        
        Returns:
            1536 维向量或 None
        """
        if not text or not text.strip():
            return None
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": text,
                    "dimensions": dimensions  # OpenAI 兼容格式
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            # OpenAI 兼容格式: {"data": [{"embedding": [...], "index": 0}]}
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0].get("embedding")
            
            print(f"[ERROR] Qwen API unexpected response: {result}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Qwen embed failed: {e}")
            return None
    
    def batch_embed(self, texts: List[str], dimensions: int = 1536) -> List[Optional[List[float]]]:
        """
        批量获取 embedding
        
        Args:
            texts: 文本列表
            dimensions: 输出维度
        
        Returns:
            embedding 列表（每个可能是 None）
        """
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid_texts:
            return [None] * len(texts)
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": [t for _, t in valid_texts],  # 列表形式
                    "dimensions": dimensions
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            # OpenAI 兼容格式: {"data": [{"embedding": [...], "index": 0}, ...]}
            if "data" in result:
                embeddings = result["data"]
                # 重建原始顺序
                result_map = {valid_texts[i][0]: e.get("embedding") 
                             for i, e in enumerate(embeddings)}
                return [result_map.get(i) for i in range(len(texts))]
            
            print(f"[ERROR] Qwen batch_embed unexpected response: {result}")
            return [None] * len(texts)
            
        except Exception as e:
            print(f"[ERROR] Qwen batch_embed failed: {e}")
            # 降级为逐个调用
            return [self.embed(t, dimensions) for t in texts]


# 简单测试
if __name__ == "__main__":
    client = QwenEmbeddingClient()
    
    # 测试单个
    result = client.embed("Hello World")
    print(f"Single embed dimension: {len(result) if result else 'None'}")
    
    # 测试批量
    results = client.batch_embed(["Hello", "World", "Test"])
    print(f"Batch embed count: {len(results)}")
    for i, r in enumerate(results):
        print(f"  [{i}] dimension: {len(r) if r else 'None'}")
