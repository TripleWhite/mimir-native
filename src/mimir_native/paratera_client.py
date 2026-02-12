"""
Paratera API Client for Mimir Memory
支持 GLM/Qwen/DeepSeek 系列模型
"""
import json
import requests
from typing import List, Optional, Dict, Any


class ParateraClient:
    """
    并行科技 LLM API 客户端
    
    API 端点: https://llmapi.paratera.com
    
    推荐模型配置:
    - Embedding: GLM-Embedding-3 或 Doubao-Embedding-Large-Text
    - Rerank: GLM-Rerank
    - LLM: Qwen3-235B-A22B-Instruct 或 DeepSeek-V3
    """
    
    def __init__(self, api_key: str = None, base_url: str = "https://llmapi.paratera.com",
                 embedding_model: str = "GLM-Embedding-3",
                 rerank_model: str = "GLM-Rerank",
                 llm_model: str = "Qwen3-235B-A22B-Instruct"):
        """
        初始化 Paratera 客户端
        
        Args:
            api_key: API Key (sk-0oVqiF3DzxzxTcbxsaPEOg)
            base_url: API 基础 URL
            embedding_model: Embedding 模型名称
            rerank_model: Rerank 模型名称
            llm_model: LLM 模型名称
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Paratera API key 不能为空")
        
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.llm_model = llm_model
        
        # OpenAI 兼容端点
        self.embed_url = f"{self.base_url}/v1/embeddings"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.rerank_url = f"{self.base_url}/v1/rerank"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed(self, text: str, dimensions: int = None) -> Optional[List[float]]:
        """获取单个文本的 embedding"""
        if not text or not text.strip():
            return None
        
        try:
            payload = {
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float"
            }
            
            if dimensions:
                payload["dimensions"] = dimensions
            
            response = requests.post(
                self.embed_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0].get("embedding")
                if isinstance(embedding, list):
                    return embedding
            
            print(f"[ERROR] Paratera API unexpected response: {result}")
            return None
            
        except Exception as e:
            print(f"[ERROR] Paratera embed failed: {e}")
            return None
    
    def batch_embed(self, texts: List[str], dimensions: int = None) -> List[Optional[List[float]]]:
        """批量获取 embedding"""
        if not texts:
            return []
        
        valid_texts = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not valid_texts:
            return [None] * len(texts)
        
        try:
            payload = {
                "model": self.embedding_model,
                "input": [t for _, t in valid_texts],
                "encoding_format": "float"
            }
            
            if dimensions:
                payload["dimensions"] = dimensions
            
            response = requests.post(
                self.embed_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "data" in result:
                embeddings = result["data"]
                result_map = {}
                for i, e in enumerate(embeddings):
                    original_idx = valid_texts[i][0]
                    embedding = e.get("embedding")
                    if isinstance(embedding, list):
                        result_map[original_idx] = embedding
                
                return [result_map.get(i) for i in range(len(texts))]
            
            print(f"[ERROR] Paratera batch_embed unexpected response: {result}")
            return [None] * len(texts)
            
        except Exception as e:
            print(f"[ERROR] Paratera batch_embed failed: {e}")
            return [self.embed(t, dimensions) for t in texts]
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用 Rerank 模型对文档重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果
        
        Returns:
            排序后的文档列表，包含 relevance_score
        """
        if not documents:
            return []
        
        try:
            payload = {
                "model": self.rerank_model,
                "query": query,
                "documents": documents,
                "top_k": top_k
            }
            
            response = requests.post(
                self.rerank_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            if "results" in result:
                return result["results"]
            
            print(f"[ERROR] Paratera rerank unexpected response: {result}")
            return []
            
        except Exception as e:
            print(f"[ERROR] Paratera rerank failed: {e}")
            return []
    
    def chat(self, prompt: str, system_prompt: str = None, max_tokens: int = 1000, 
             temperature: float = 0.0) -> str:
        """
        调用 LLM 进行文本生成
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            max_tokens: 最大生成 token 数
            temperature: 温度参数
        
        Returns:
            生成的文本
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            
            print(f"[ERROR] Paratera chat unexpected response: {result}")
            return ""
            
        except Exception as e:
            print(f"[ERROR] Paratera chat failed: {e}")
            return ""
    
    def extract_facts(self, conversation_text: str, session_date: str = None) -> List[str]:
        """
        从对话中提取事实
        
        Args:
            conversation_text: 对话文本
            session_date: 会话日期
        
        Returns:
            提取的事实列表
        """
        system_prompt = """You are a fact extraction assistant. Extract specific facts from the conversation.
        
Rules:
1. Extract one fact per line
2. Include temporal information when available
3. Be specific and concrete
4. Use the exact wording from the conversation when possible"""

        date_context = f"\nSession date: {session_date}" if session_date else ""
        prompt = f"Extract facts from this conversation:{date_context}\n\n{conversation_text}\n\nFacts:"
        
        response = self.chat(prompt, system_prompt, max_tokens=2000, temperature=0.0)
        
        # 解析响应为事实列表
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # 去除编号前缀
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:].strip()
                facts.append(line)
        
        return facts
    
    def detect_conflicts(self, fact1: str, fact2: str) -> Dict[str, Any]:
        """
        检测两个事实是否冲突
        
        Args:
            fact1: 事实1
            fact2: 事实2
        
        Returns:
            冲突检测结果
        """
        system_prompt = """You are a conflict detection assistant. Determine if two facts contradict each other.

Output format (JSON):
{
    "conflict": true/false,
    "type": "temporal"/"factual"/"preference"/"none",
    "explanation": "brief explanation"
}"""

        prompt = f"Fact 1: {fact1}\n\nFact 2: {fact2}\n\nDo these facts conflict?"
        
        response = self.chat(prompt, system_prompt, max_tokens=500, temperature=0.0)
        
        try:
            # 尝试解析 JSON 响应
            import json as json_module
            result = json_module.loads(response.strip())
            return result
        except:
            # 降级处理
            return {
                "conflict": "conflict" in response.lower() or "contradict" in response.lower(),
                "type": "unknown",
                "explanation": response[:200]
            }


# 全局客户端实例（懒加载）
_paratera_client = None

def get_paratera_client() -> ParateraClient:
    """获取全局 Paratera 客户端实例"""
    global _paratera_client
    if _paratera_client is None:
        _paratera_client = ParateraClient(
            api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
            base_url="https://llmapi.paratera.com",
            embedding_model="GLM-Embedding-3",
            rerank_model="GLM-Rerank",
            llm_model="Qwen3-235B-A22B-Instruct"
        )
    return _paratera_client
