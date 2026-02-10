"""
Silicon Flow Client for Mimir Memory V2
支持 Qwen3-Embedding-8B 等多种模型 + Qwen2.5-14B-Instruct 文本生成
"""
import json
import requests
from typing import List, Optional, Dict, Any


class SiliconFlowClient:
    """
    Silicon Flow 客户端
    支持 Embedding 和文本生成
    
    Embedding 模型:
    - Qwen/Qwen3-Embedding-8B (推荐，MTEB #1)
    - Qwen/Qwen3-Embedding-4B
    - Qwen/Qwen3-Embedding-0.6B
    - BAAI/bge-m3 (支持 8192 tokens)
    
    文本生成模型:
    - Qwen/Qwen2.5-14B-Instruct (用于 fact extraction, conflict detection)
    """
    
    def __init__(self, api_key: str = None, embedding_model: str = "Qwen/Qwen3-Embedding-8B",
                 text_model: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Silicon Flow API key 不能为空")
        
        self.embedding_model = embedding_model
        self.text_model = text_model
        self.embed_url = "https://api.siliconflow.cn/v1/embeddings"
        self.chat_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed(self, text: str, dimensions: int = None) -> Optional[List[float]]:
        """
        获取单个文本的 embedding
        
        Args:
            text: 输入文本
            dimensions: 输出维度 (仅部分模型支持，如 Qwen3 支持 32-4096)
        
        Returns:
            embedding 向量或 None
        """
        if not text or not text.strip():
            return None
        
        try:
            payload = {
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float"
            }
            
            # Qwen3 支持自定义维度
            if dimensions and "Qwen" in self.embedding_model:
                payload["dimensions"] = dimensions
            
            response = requests.post(
                self.embed_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            # OpenAI 兼容格式: {"data": [{"embedding": [...], "index": 0}]}
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0].get("embedding")
                # 确保返回的是 list
                if isinstance(embedding, list):
                    return embedding
            
            print(f"[ERROR] Silicon Flow API unexpected response: {result}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Silicon Flow embed failed: {e}")
            return None
    
    def batch_embed(self, texts: List[str], dimensions: int = None) -> List[Optional[List[float]]]:
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
            payload = {
                "model": self.embedding_model,
                "input": [t for _, t in valid_texts],
                "encoding_format": "float"
            }
            
            if dimensions and "Qwen" in self.embedding_model:
                payload["dimensions"] = dimensions
            
            response = requests.post(
                self.embed_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            # OpenAI 兼容格式
            if "data" in result:
                embeddings = result["data"]
                # 重建原始顺序
                result_map = {}
                for i, e in enumerate(embeddings):
                    original_idx = valid_texts[i][0]
                    embedding = e.get("embedding")
                    if isinstance(embedding, list):
                        result_map[original_idx] = embedding
                
                return [result_map.get(i) for i in range(len(texts))]
            
            print(f"[ERROR] Silicon Flow batch_embed unexpected response: {result}")
            return [None] * len(texts)
            
        except Exception as e:
            print(f"[ERROR] Silicon Flow batch_embed failed: {e}")
            # 降级为逐个调用
            return [self.embed(t, dimensions) for t in texts]

    # ============================================================================
    # 文本生成方法 (Qwen/Qwen2.5-14B-Instruct)
    # ============================================================================

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        """
        文本生成（用于 fact extraction, conflict detection）
        
        使用 Qwen/Qwen2.5-14B-Instruct
        """
        response = requests.post(
            self.chat_url,
            headers=self.headers,
            json={
                "model": self.text_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def extract_facts(self, text: str, context: dict = None) -> List[Dict[str, Any]]:
        """
        从文本中提取结构化事实
        
        Args:
            text: 输入文本
            context: 上下文信息（包含日期等）

        Returns:
            List[Dict]: 提取的事实列表
        """
        context = context or {}

        prompt = f"""从以下文本中提取所有事实陈述和人物属性。

文本: {text}

上下文: {json.dumps(context, ensure_ascii=False)}

关键要求:
1. **提取所有有意义的陈述**，包括：
   - 事件（做了什么事）
   - 人物属性（身份、性别、职业、关系状态等）
   - 偏好（喜欢/不喜欢）
   - 计划（将要做什么）
2. **保持原文语言**：如果输入是英文，fact 字段必须是英文，不要翻译
3. **保留所有时间信息**：原始日期格式必须保留（如 "7 May 2023"）
4. 每个事实应该是自包含的，无需上下文就能理解
5. 识别涉及的实体（人名、地点、组织等）

事实类型: event | preference | relationship | work | personal_info | other

输出 JSON 数组格式:
[
  {{
    "fact": "事实陈述（保持原文语言，不要翻译）",
    "temporal_info": {{
      "absolute_time": "YYYY-MM-DD 格式或 null",
      "relative_time": "如 'yesterday' 或 null",
      "time_mentions": ["文中提到的所有时间，保持原始格式"]
    }},
    "entities": ["实体1", "实体2"],
    "fact_type": "personal_info",
    "confidence": 0.95
  }}
]

**重要**: 
- 不要过滤！提取所有信息，包括简单的属性陈述
- 不要翻译！保持输入文本的原始语言
- 只输出 JSON，不要其他文字。"""

        try:
            response = self.generate(prompt, max_tokens=1500, temperature=0.0)
            return self._parse_json_response(response)
        except Exception as e:
            print(f"[ERROR] Silicon Flow fact extraction failed: {e}")
            raise

    def check_conflict(self, new_fact: str, existing_fact: str,
                       new_time: str = None, existing_time: str = None) -> Dict[str, Any]:
        """
        检查两个事实是否冲突

        Args:
            new_fact: 新事实
            existing_fact: 已有事实
            new_time: 新事实的时间
            existing_time: 已有事实的时间

        Returns:
            Dict: 冲突检测结果
        """
        prompt = f"""判断以下两个事实是否冲突。

新事实: {new_fact}
新事实时间: {new_time or '未知'}

已有事实: {existing_fact}
已有事实时间: {existing_time or '未知'}

判断标准:
1. 如果两个事实描述的是同一事物的不同状态，且不能同时为真，则为冲突
2. 如果新事实是已有事实的补充或更新（且时间更近），则不是冲突，而是更新
3. 如果两个事实描述的是不同方面，则不是冲突

输出 JSON 格式:
{{
  "is_conflict": true/false,
  "resolution": "keep_existing/update/merge/new",
  "reason": "判断理由",
  "confidence": 0.9
}}

resolution 说明:
- keep_existing: 保留已有事实
- update: 用新事实更新（时间更近或更准确）
- merge: 合并两个事实
- new: 作为新事实保留

只输出 JSON，不要其他文字。"""

        try:
            response = self.generate(prompt, max_tokens=500, temperature=0.0)
            return self._parse_json_response(response)
        except Exception as e:
            print(f"[ERROR] Silicon Flow conflict check failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Any:
        """解析 LLM 返回的 JSON"""
        try:
            # 尝试直接解析
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试从 markdown 代码块中提取
            try:
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0]
                    return json.loads(json_str.strip())
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0]
                    return json.loads(json_str.strip())
            except Exception:
                pass

            # 尝试提取 [] 或 {} 包裹的内容
            try:
                start = response.find('[')
                end = response.rfind(']')
                if start != -1 and end != -1:
                    return json.loads(response[start:end+1])

                start = response.find('{')
                end = response.rfind('}')
                if start != -1 and end != -1:
                    return json.loads(response[start:end+1])
            except Exception:
                pass

            print(f"[ERROR] Cannot parse JSON response: {response[:200]}")
            raise ValueError(f"Invalid JSON response: {response[:200]}")


# 保留旧类名作为别名（向后兼容）
SiliconFlowEmbeddingClient = SiliconFlowClient


# 简单测试
if __name__ == "__main__":
    import os
    
    # 从环境变量或输入获取 API key
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("请设置 SILICONFLOW_API_KEY 环境变量")
        exit(1)
    
    client = SiliconFlowClient(api_key=api_key)
    
    # 测试 embedding
    print("Testing single embed...")
    result = client.embed("Hello World")
    print(f"Single embed dimension: {len(result) if result else 'None'}")
    
    # 测试批量 embedding
    print("\nTesting batch embed...")
    results = client.batch_embed(["Hello", "World", "Test"])
    print(f"Batch embed count: {len(results)}")
    for i, r in enumerate(results):
        print(f"  [{i}] dimension: {len(r) if r else 'None'}")
    
    # 测试文本生成
    print("\nTesting text generation...")
    gen_result = client.generate("你好，请介绍一下自己", max_tokens=200)
    print(f"Generation result: {gen_result[:100]}...")
    
    # 测试事实提取
    print("\nTesting fact extraction...")
    facts = client.extract_facts("我今天去了北京，明天要去上海。我喜欢吃川菜。")
    print(f"Extracted facts: {json.dumps(facts, ensure_ascii=False, indent=2)}")
