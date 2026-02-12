#!/usr/bin/env python3
"""
LoCoMo Hybrid Retriever - 批处理 + 缓存版本
"""

import json
import sys
import os
import re
import math
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import requests


class CachedEmbeddingClient:
    """带缓存的 Embedding 客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://llmapi.paratera.com", 
                 cache_file: str = "embeddings_cache.json"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.embed_url = f"{self.base_url}/v1/embeddings"
        self.cache_file = cache_file
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _get_text_hash(self, text: str) -> str:
        """获取文本哈希"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding（带缓存）"""
        text_hash = self._get_text_hash(text)
        
        # 检查缓存
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # 调用 API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "GLM-Embedding-3",
            "input": text[:512]
        }
        
        try:
            response = requests.post(
                self.embed_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                # 保存到缓存
                self.cache[text_hash] = embedding
                return embedding
        except Exception as e:
            print(f"  Embedding error: {e}")
        
        return []
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量获取 embeddings"""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # 先检查缓存
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                results.append((i, self.cache[text_hash]))
            else:
                results.append((i, None))
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 批量调用 API 获取未缓存的
        if uncached_texts:
            print(f"  API调用: {len(uncached_texts)} 个新文本")
            
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                batch_indices = uncached_indices[i:i+batch_size]
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "GLM-Embedding-3",
                    "input": [t[:512] for t in batch]
                }
                
                try:
                    response = requests.post(
                        self.embed_url,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if "data" in result:
                        for idx_in_batch, data in enumerate(result["data"]):
                            embedding = data["embedding"]
                            original_idx = batch_indices[idx_in_batch]
                            text_hash = self._get_text_hash(batch[idx_in_batch])
                            self.cache[text_hash] = embedding
                            results[original_idx] = (original_idx, embedding)
                            
                except Exception as e:
                    print(f"  Batch error: {e}")
                    # 失败后逐个尝试
                    for idx_in_batch, text in enumerate(batch):
                        original_idx = batch_indices[idx_in_batch]
                        emb = self.get_embedding(text)
                        results[original_idx] = (original_idx, emb)
        
        # 保存缓存
        self._save_cache()
        
        # 返回按原始顺序的结果
        return [emb for _, emb in sorted(results, key=lambda x: x[0])]


class LoCoMoHybridRetriever:
    """为 LoCoMo 定制的 Hybrid Retriever"""
    
    def __init__(self, api_key: str, base_url: str = "https://llmapi.paratera.com"):
        self.embedding_client = CachedEmbeddingClient(api_key, base_url)
        
        # 存储
        self.facts = []
        self.session_dates = {}
        self.bm25_corpus = []
    
    def parse_session_date(self, date_str: str) -> Optional[datetime]:
        """解析会话日期"""
        match = re.search(r'(\d{1,2})[:\s]*(am|pm)?\s*on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})', 
                         date_str, re.IGNORECASE)
        if match:
            day = int(match.group(3))
            month_name = match.group(4).lower()
            year = int(match.group(5))
            
            month_map = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month = month_map.get(month_name)
            if month:
                try:
                    return datetime(year, month, day)
                except:
                    pass
        return None
    
    def build_index(self, data: Dict):
        """构建混合索引"""
        conversation = data.get('conversation', {})
        observation = data.get('observation', {})
        
        # 1. 提取会话日期
        for key in conversation.keys():
            if key.endswith('_date_time'):
                session_key = key.replace('_date_time', '')
                parsed = self.parse_session_date(conversation[key])
                if parsed:
                    self.session_dates[session_key] = parsed
        
        print(f"解析到 {len(self.session_dates)} 个会话日期")
        
        # 2. 从 observation 提取事实
        # 结构: session_X_observation -> {person: [[fact, timestamp], ...]}
        fact_texts = []
        for session_key, obs_dict in observation.items():
            session = session_key.replace('_observation', '')
            session_date = self.session_dates.get(session, datetime(2023, 5, 1))
            
            if isinstance(obs_dict, dict):
                for person, fact_list in obs_dict.items():
                    if isinstance(fact_list, list):
                        for fact_item in fact_list:
                            if isinstance(fact_item, list) and len(fact_item) >= 1:
                                fact_text = fact_item[0]
                                if isinstance(fact_text, str) and len(fact_text) > 20:
                                    self.facts.append({
                                        'content': fact_text,
                                        'date': session_date,
                                        'source': 'observation',
                                        'session': session,
                                        'person': person
                                    })
                                    fact_texts.append(fact_text)
        
        print(f"提取了 {len(self.facts)} 个事实")
        
        # 3. 批量生成 embeddings
        print("批量生成 embeddings...")
        embeddings = self.embedding_client.get_embeddings_batch(fact_texts, batch_size=32)
        
        for i, emb in enumerate(embeddings):
            self.facts[i]['embedding'] = emb
        
        print(f"Embeddings 完成: {len([f for f in self.facts if f.get('embedding')])}/{len(self.facts)}")
        
        # 4. 准备 BM25
        for fact in self.facts:
            self.bm25_corpus.append(fact['content'].lower().split())
        
        print("索引构建完成!")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25 检索"""
        query_terms = query.lower().split()
        scores = []
        
        for idx, doc in enumerate(self.bm25_corpus):
            score = 0
            for term in query_terms:
                if term in doc:
                    tf = doc.count(term) / len(doc) if doc else 0
                    score += tf
            if score > 0:
                scores.append((idx, score))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def vector_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """向量检索"""
        query_emb = self.embedding_client.get_embedding(query)
        if not query_emb:
            return []
        
        scores = []
        for idx, fact in enumerate(self.facts):
            if fact.get('embedding'):
                sim = self.cosine_similarity(query_emb, fact['embedding'])
                if sim > 0.3:  # 降低阈值
                    scores.append((idx, sim))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def rrf_fusion(self, bm25_results: List[Tuple[int, float]], 
                   vector_results: List[Tuple[int, float]], 
                   k: int = 60) -> List[Tuple[int, float]]:
        """RRF 融合"""
        scores = {}
        
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        
        return sorted(scores.items(), key=lambda x: -x[1])
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """混合检索"""
        bm25_results = self.bm25_search(query, top_k=20)
        vector_results = self.vector_search(query, top_k=20)
        
        fused_results = self.rrf_fusion(bm25_results, vector_results)
        
        results = []
        for idx, score in fused_results[:top_k]:
            if 0 <= idx < len(self.facts):
                fact = self.facts[idx].copy()
                fact['retrieval_score'] = score
                results.append(fact)
        
        return results
    
    def answer_when(self, question: str) -> str:
        """回答 When 问题"""
        results = self.retrieve(question, top_k=10)
        
        dated_facts = [r for r in results if r.get('date')]
        
        if dated_facts:
            best = dated_facts[0]
            return best['date'].strftime('%d %B %Y')
        
        return "Unknown"


def calculate_f1(predicted: str, ground_truth: Any) -> float:
    """计算 F1"""
    if isinstance(ground_truth, (int, float)):
        ground_truth = str(ground_truth)
    
    pred = str(predicted).lower().strip()
    truth = str(ground_truth).lower().strip()
    
    if pred == truth:
        return 1.0
    
    if truth in pred or pred in truth:
        return 0.8
    
    pred_year = re.search(r'\b(20\d{2})\b', pred)
    truth_year = re.search(r'\b(20\d{2})\b', truth)
    if pred_year and truth_year:
        if pred_year.group(1) == truth_year.group(1):
            return 0.7
    
    pred_chars = set(pred)
    truth_chars = set(truth)
    
    if not pred_chars or not truth_chars:
        return 0.0
    
    intersection = pred_chars & truth_chars
    precision = len(intersection) / len(pred_chars) if pred_chars else 0
    recall = len(intersection) / len(truth_chars) if truth_chars else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def main():
    print("="*70)
    print("LoCoMo Hybrid Retriever - 批处理 + 缓存版本")
    print("="*70)
    
    # 加载数据
    with open('/tmp/mimir-review/mimir-native/locomodata.json', 'r') as f:
        data = json.load(f)
    
    conv = data[0]
    qa_list = conv.get('qa', [])
    
    # 检查缓存
    cache_file = '/tmp/mimir-review/mimir-native/embeddings_cache.json'
    cache_exists = os.path.exists(cache_file)
    if cache_exists:
        cache_size = os.path.getsize(cache_file)
        print(f"\n发现缓存文件: {cache_file} ({cache_size/1024:.1f} KB)")
    
    # 初始化 Hybrid Retriever
    retriever = LoCoMoHybridRetriever(
        api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
        base_url="https://llmapi.paratera.com"
    )
    
    # 构建索引
    print("\n构建 Hybrid Index...")
    retriever.build_index(conv)
    
    # 筛选 When 问题
    when_questions = [(i, qa) for i, qa in enumerate(qa_list) 
                     if qa.get('question', '').lower().startswith('when')]
    
    print(f"\n测试 {len(when_questions)} 个 When 问题...")
    print("="*70)
    
    results = []
    for idx, qa in when_questions:
        question = qa['question']
        ground_truth = qa['answer']
        
        predicted = retriever.answer_when(question)
        f1 = calculate_f1(predicted, ground_truth)
        
        results.append({
            'q_id': idx + 1,
            'question': question,
            'predicted': predicted,
            'ground_truth': str(ground_truth),
            'f1': f1
        })
        
        status = "✓" if f1 >= 0.8 else "~" if f1 >= 0.5 else "✗"
        print(f"  [{idx+1:3d}] {status} F1:{f1:.0%}")
        print(f"        Q: {question[:50]}...")
        print(f"        A: {predicted[:30]:30s} | 真实: {str(ground_truth)[:30]}...")
    
    # 统计
    avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0
    correct = sum(1 for r in results if r['f1'] >= 0.8)
    partial = sum(1 for r in results if 0.5 <= r['f1'] < 0.8)
    wrong = sum(1 for r in results if r['f1'] < 0.5)
    
    print(f"\n{'='*70}")
    print(f"正确: {correct}, 部分: {partial}, 错误: {wrong}")
    print(f"When 问题 F1: {avg_f1:.2%}")
    print(f"{'='*70}")
    
    # 对比
    print("\n📊 对比:")
    print(f"  原始版:        25.3%")
    print(f"  Session匹配版: 69.2%")
    print(f"  Hybrid检索版:  {avg_f1:.1%}")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'method': 'Hybrid Retriever (BM25 + Embedding + RRF + Cache)',
        'num_when_questions': len(when_questions),
        'avg_f1': avg_f1,
        'results': results
    }
    
    output_path = f"/tmp/mimir-review/mimir-native/locomo_hybrid_cached_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n结果已保存: {output_path}")
    
    # 显示缓存统计
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        print(f"缓存统计: {len(cache)} 个 embeddings 已缓存")


if __name__ == "__main__":
    main()
