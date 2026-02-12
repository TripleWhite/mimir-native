#!/usr/bin/env python3
"""
LoCoMo 基准测试 - 使用 Paratera API (独立版本)

直接测试 Paratera API 在 LoCoMo 任务上的表现
"""

import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime


class ParateraClient:
    """Paratera API 客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://llmapi.paratera.com",
                 embedding_model: str = "GLM-Embedding-3",
                 rerank_model: str = "GLM-Rerank",
                 llm_model: str = "Qwen3-235B-A22B-Instruct"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.llm_model = llm_model
        
        self.embed_url = f"{self.base_url}/v1/embeddings"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.rerank_url = f"{self.base_url}/v1/rerank"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed(self, text: str) -> Optional[List[float]]:
        """获取文本 embedding"""
        if not text or not text.strip():
            return None
        
        try:
            payload = {
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float"
            }
            
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
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Embed failed: {e}")
            return None
    
    def chat(self, prompt: str, system_prompt: str = None, max_tokens: int = 500, 
             temperature: float = 0.0) -> str:
        """调用 LLM"""
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
            
            return ""
            
        except Exception as e:
            print(f"[ERROR] Chat failed: {e}")
            return ""


class LoCoMoEvaluator:
    """LoCoMo 评估器"""
    
    def __init__(self, client: ParateraClient):
        self.client = client
    
    def get_sample_conversation(self) -> Dict:
        """获取示例 LoCoMo 对话"""
        return {
            "conversation_id": "locomo_test_001",
            "conversation": [
                {
                    "role": "user",
                    "content": "Hi, I'm Caroline. I'm a 25-year-old software engineer living in San Francisco.",
                    "timestamp": "2023-05-08T09:00:00"
                },
                {
                    "role": "assistant",
                    "content": "Hello Caroline! Nice to meet you. How can I help you today?",
                    "timestamp": "2023-05-08T09:00:30"
                },
                {
                    "role": "user",
                    "content": "I just moved here last week from New York. I'm still getting used to the weather.",
                    "timestamp": "2023-05-08T09:01:00"
                },
                {
                    "role": "user",
                    "content": "By the way, I adopted a puppy yesterday! His name is Max.",
                    "timestamp": "2023-05-09T10:00:00"
                },
                {
                    "role": "assistant",
                    "content": "That's exciting! Congratulations on the new puppy. What breed is Max?",
                    "timestamp": "2023-05-09T10:00:30"
                },
                {
                    "role": "user",
                    "content": "He's a golden retriever. Only 8 weeks old. I'm taking him to the vet tomorrow morning.",
                    "timestamp": "2023-05-09T10:01:00"
                },
                {
                    "role": "user",
                    "content": "The vet appointment went well. Max got his first round of vaccines. The vet said he's very healthy!",
                    "timestamp": "2023-05-10T11:00:00"
                },
                {
                    "role": "assistant",
                    "content": "That's great news! I'm glad Max is healthy.",
                    "timestamp": "2023-05-10T11:00:30"
                }
            ],
            "questions": [
                {"q_id": "Q1", "question": "What's Caroline's age?", "answer": "25"},
                {"q_id": "Q2", "question": "Where does Caroline live?", "answer": "San Francisco"},
                {"q_id": "Q3", "question": "Where did Caroline move from?", "answer": "New York"},
                {"q_id": "Q4", "question": "What's the name of Caroline's puppy?", "answer": "Max"},
                {"q_id": "Q5", "question": "When did Caroline adopt Max?", "answer": "9 May 2023"},
                {"q_id": "Q6", "question": "What breed is Max?", "answer": "golden retriever"},
                {"q_id": "Q7", "question": "When did Caroline take Max to the vet?", "answer": "10 May 2023"},
                {"q_id": "Q8", "question": "What did the vet say about Max?", "answer": "very healthy"}
            ]
        }
    
    def extract_facts(self, conversation: List[Dict]) -> List[str]:
        """从对话中提取事实"""
        # 格式化对话
        conversation_text = "\n".join([
            f"[{msg.get('timestamp', 'Unknown')}] {msg['role']}: {msg['content']}"
            for msg in conversation
        ])
        
        system_prompt = """You are a fact extraction assistant. Extract specific facts from the conversation.

Rules:
1. Extract one fact per line
2. Include temporal information (dates, times) when available
3. Be specific and concrete
4. Use exact wording from conversation when possible
5. Output format: one fact per line, no numbering"""

        prompt = f"""Extract all facts from this conversation:

{conversation_text}

Facts:"""
        
        print("  正在提取事实...")
        response = self.client.chat(prompt, system_prompt, max_tokens=2000, temperature=0.0)
        
        # 解析事实
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # 去除编号
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:].strip()
                # 去除列表符号
                if line.startswith('- ') or line.startswith('* '):
                    line = line[2:].strip()
                facts.append(line)
        
        return facts
    
    def answer_question(self, question: str, facts: List[str]) -> str:
        """基于事实回答问题"""
        context = "\n".join([f"- {fact}" for fact in facts])
        
        system_prompt = """You are a helpful assistant. Answer the question based ONLY on the provided facts.

Rules:
1. Use only the information in the facts
2. Be concise (1-5 words if possible)
3. If the answer is not in the facts, say "Unknown"
4. No explanations, just the answer"""

        prompt = f"""Facts:
{context}

Question: {question}

Answer:"""
        
        answer = self.client.chat(prompt, system_prompt, max_tokens=100, temperature=0.0)
        return answer.strip().strip('"').strip("'")
    
    def calculate_f1(self, predicted: str, ground_truth: str) -> float:
        """计算 F1 分数"""
        pred = predicted.lower().strip()
        truth = ground_truth.lower().strip()
        
        # 完全匹配
        if pred == truth:
            return 1.0
        
        # 包含匹配
        if truth in pred or pred in truth:
            return 0.8
        
        # 字符级 F1
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
    
    def evaluate(self) -> Dict:
        """运行 LoCoMo 评估"""
        print("="*60)
        print("LoCoMo 基准测试 - 使用 Paratera API")
        print("="*60)
        print(f"API: https://llmapi.paratera.com")
        print(f"Embedding: {self.client.embedding_model}")
        print(f"Rerank: {self.client.rerank_model}")
        print(f"LLM: {self.client.llm_model}")
        print("="*60)
        
        # 获取测试数据
        data = self.get_sample_conversation()
        conversation = data["conversation"]
        questions = data["questions"]
        
        print(f"\n对话 ID: {data['conversation_id']}")
        print(f"对话消息数: {len(conversation)}")
        print(f"问题数: {len(questions)}")
        
        # 提取事实
        print("\n" + "-"*60)
        print("步骤 1: 提取事实")
        print("-"*60)
        facts = self.extract_facts(conversation)
        print(f"\n提取到 {len(facts)} 个事实:")
        for i, fact in enumerate(facts, 1):
            print(f"  {i}. {fact}")
        
        # 回答问题
        print("\n" + "-"*60)
        print("步骤 2: 回答问题")
        print("-"*60)
        
        results = []
        total_f1 = 0.0
        
        for q in questions:
            q_id = q["q_id"]
            question = q["question"]
            ground_truth = q["answer"]
            
            predicted = self.answer_question(question, facts)
            f1 = self.calculate_f1(predicted, ground_truth)
            total_f1 += f1
            
            results.append({
                "q_id": q_id,
                "question": question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "f1": f1
            })
            
            status = "✓" if f1 >= 0.8 else "✗"
            print(f"\n[{q_id}] {status}")
            print(f"  问题: {question}")
            print(f"  预测: {predicted}")
            print(f"  真实: {ground_truth}")
            print(f"  F1: {f1:.2%}")
        
        # 计算总体 F1
        avg_f1 = total_f1 / len(questions) if questions else 0.0
        
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        print(f"总体 F1 Score: {avg_f1:.2%}")
        print(f"问题数: {len(questions)}")
        print(f"正确回答: {sum(1 for r in results if r['f1'] >= 0.8)}")
        print("="*60)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "api_config": {
                "base_url": self.client.base_url,
                "embedding_model": self.client.embedding_model,
                "rerank_model": self.client.rerank_model,
                "llm_model": self.client.llm_model
            },
            "num_facts": len(facts),
            "num_questions": len(questions),
            "overall_f1": avg_f1,
            "results": results
        }


def test_api_connection():
    """测试 API 连接"""
    print("测试 Paratera API 连接...")
    print("-" * 40)
    
    client = ParateraClient(
        api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
        base_url="https://llmapi.paratera.com",
        embedding_model="GLM-Embedding-3",
        llm_model="GLM-4-Plus"
    )
    
    # 测试 Embedding
    print("\n1. 测试 Embedding API...")
    embed_result = client.embed("这是一个测试文本")
    if embed_result:
        print(f"   ✓ Embedding 成功 (维度: {len(embed_result)})")
    else:
        print("   ✗ Embedding 失败")
        return None
    
    # 测试 Chat
    print("\n2. 测试 Chat API...")
    chat_result = client.chat("你好，请回复'API测试成功'", max_tokens=50)
    if chat_result:
        print(f"   ✓ Chat 成功")
        print(f"   回复: {chat_result[:100]}...")
    else:
        print("   ✗ Chat 失败")
        return None
    
    print("\n" + "-" * 40)
    print("✓ API 连接正常!")
    print("-" * 40)
    
    return client


def main():
    """主函数"""
    # 测试 API 连接
    client = test_api_connection()
    if not client:
        print("\nAPI 连接失败，请检查配置")
        return
    
    # 运行 LoCoMo 评估
    print()
    evaluator = LoCoMoEvaluator(client)
    results = evaluator.evaluate()
    
    # 保存结果
    output_file = f"locomo_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
