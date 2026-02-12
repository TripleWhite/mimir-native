#!/usr/bin/env python3
"""
LoCoMo 基准测试 - 使用 Paratera API

LoCoMo (Long Conversation Memory) 测试记忆系统在长对话中的表现
测试指标: F1 Score (精确率和召回率的调和平均)

API 配置:
- Base URL: https://llmapi.paratera.com
- Embedding: GLM-Embedding-3
- Rerank: GLM-Rerank  
- LLM: Qwen3-235B-A22B-Instruct
"""

import json
import sys
import os
from typing import List, Dict, Any
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from mimir_native.paratera_client import ParateraClient, get_paratera_client


class LoCoMoEvaluator:
    """LoCoMo 评估器"""
    
    def __init__(self):
        self.client = get_paratera_client()
        self.results = []
        
    def load_locomo_data(self, data_path: str = None) -> List[Dict]:
        """
        加载 LoCoMo 测试数据
        
        LoCoMo 数据格式:
        {
            "conversation_id": str,
            "conversation": [
                {"role": "user"/"assistant", "content": str, "timestamp": str}
            ],
            "questions": [
                {
                    "question_id": str,
                    "question": str,
                    "answer": str,
                    "answer_type": "temporal"/"factual"/"preference"
                }
            ]
        }
        """
        # 如果没有提供数据路径，使用示例数据
        if not data_path or not os.path.exists(data_path):
            print("使用示例 LoCoMo 数据...")
            return self._get_sample_data()
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data if isinstance(data, list) else [data]
    
    def _get_sample_data(self) -> List[Dict]:
        """示例 LoCoMo 数据"""
        return [
            {
                "conversation_id": "locomo_sample_001",
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
                    {
                        "question_id": "Q1",
                        "question": "What's Caroline's age?",
                        "answer": "25",
                        "answer_type": "factual"
                    },
                    {
                        "question_id": "Q2", 
                        "question": "Where does Caroline live?",
                        "answer": "San Francisco",
                        "answer_type": "factual"
                    },
                    {
                        "question_id": "Q3",
                        "question": "Where did Caroline move from?",
                        "answer": "New York",
                        "answer_type": "factual"
                    },
                    {
                        "question_id": "Q4",
                        "question": "What's the name of Caroline's puppy?",
                        "answer": "Max",
                        "answer_type": "factual"
                    },
                    {
                        "question_id": "Q5",
                        "question": "When did Caroline adopt Max?",
                        "answer": "9 May 2023",
                        "answer_type": "temporal"
                    },
                    {
                        "question_id": "Q6",
                        "question": "What breed is Max?",
                        "answer": "golden retriever",
                        "answer_type": "factual"
                    },
                    {
                        "question_id": "Q7",
                        "question": "When did Caroline take Max to the vet?",
                        "answer": "10 May 2023",
                        "answer_type": "temporal"
                    },
                    {
                        "question_id": "Q8",
                        "question": "What did the vet say about Max?",
                        "answer": "very healthy",
                        "answer_type": "factual"
                    }
                ]
            }
        ]
    
    def extract_facts_from_conversation(self, conversation: List[Dict]) -> List[str]:
        """从对话中提取事实"""
        # 合并对话文本
        conversation_text = "\n".join([
            f"[{msg.get('timestamp', 'Unknown')}] {msg['role']}: {msg['content']}"
            for msg in conversation
        ])
        
        # 获取会话日期
        session_date = conversation[0].get('timestamp', '').split('T')[0] if conversation else None
        
        # 使用 Paratera API 提取事实
        print(f"  提取事实 (对话 {len(conversation)} 条消息)...")
        facts = self.client.extract_facts(conversation_text, session_date)
        
        print(f"  提取到 {len(facts)} 个事实")
        for i, fact in enumerate(facts[:5], 1):
            print(f"    {i}. {fact[:80]}...")
        
        return facts
    
    def answer_question(self, question: str, facts: List[str]) -> str:
        """基于事实回答问题"""
        # 构建上下文
        context = "\n".join([f"- {fact}" for fact in facts])
        
        system_prompt = """You are a helpful assistant. Answer the question based ONLY on the provided facts.
        
Rules:
1. Use only the information in the facts
2. Be concise (1-5 words if possible)
3. If the answer is not in the facts, say "Unknown"
4. Do not add explanations"""

        prompt = f"Facts:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        answer = self.client.chat(prompt, system_prompt, max_tokens=100, temperature=0.0)
        
        # 清理答案
        answer = answer.strip().strip('"').strip("'")
        
        return answer
    
    def calculate_f1(self, predicted: str, ground_truth: str) -> float:
        """
        计算 F1 分数
        
        使用字符级别的 F1 计算
        """
        pred = predicted.lower().strip()
        truth = ground_truth.lower().strip()
        
        # 完全匹配
        if pred == truth:
            return 1.0
        
        # 包含匹配
        if truth in pred or pred in truth:
            return 0.8
        
        # 计算字符级别的 F1
        pred_chars = set(pred)
        truth_chars = set(truth)
        
        if not pred_chars or not truth_chars:
            return 0.0
        
        intersection = pred_chars & truth_chars
        
        precision = len(intersection) / len(pred_chars) if pred_chars else 0
        recall = len(intersection) / len(truth_chars) if truth_chars else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def evaluate_conversation(self, data: Dict) -> Dict:
        """评估单个对话"""
        conversation_id = data['conversation_id']
        conversation = data['conversation']
        questions = data['questions']
        
        print(f"\n{'='*60}")
        print(f"评估对话: {conversation_id}")
        print(f"{'='*60}")
        
        # 提取事实
        facts = self.extract_facts_from_conversation(conversation)
        
        # 评估每个问题
        results = []
        total_f1 = 0.0
        
        for q in questions:
            q_id = q['question_id']
            question = q['question']
            ground_truth = q['answer']
            answer_type = q.get('answer_type', 'factual')
            
            print(f"\n  [{q_id}] {question}")
            
            # 生成答案
            predicted = self.answer_question(question, facts)
            
            # 计算 F1
            f1 = self.calculate_f1(predicted, ground_truth)
            total_f1 += f1
            
            print(f"    预测: {predicted}")
            print(f"    真实: {ground_truth}")
            print(f"    F1: {f1:.2%}")
            
            results.append({
                'question_id': q_id,
                'question': question,
                'predicted': predicted,
                'ground_truth': ground_truth,
                'f1': f1,
                'type': answer_type
            })
        
        # 计算平均 F1
        avg_f1 = total_f1 / len(questions) if questions else 0.0
        
        print(f"\n  对话平均 F1: {avg_f1:.2%}")
        
        return {
            'conversation_id': conversation_id,
            'num_facts': len(facts),
            'num_questions': len(questions),
            'avg_f1': avg_f1,
            'results': results
        }
    
    def run_evaluation(self, data_path: str = None) -> Dict:
        """运行完整评估"""
        print("="*60)
        print("LoCoMo 基准测试 - 使用 Paratera API")
        print("="*60)
        print(f"API: https://llmapi.paratera.com")
        print(f"Embedding: {self.client.embedding_model}")
        print(f"Rerank: {self.client.rerank_model}")
        print(f"LLM: {self.client.llm_model}")
        print("="*60)
        
        # 加载数据
        data_list = self.load_locomo_data(data_path)
        print(f"\n加载了 {len(data_list)} 个对话")
        
        # 评估每个对话
        all_results = []
        total_f1 = 0.0
        
        for i, data in enumerate(data_list, 1):
            print(f"\n处理第 {i}/{len(data_list)} 个对话...")
            result = self.evaluate_conversation(data)
            all_results.append(result)
            total_f1 += result['avg_f1']
        
        # 计算总体 F1
        overall_f1 = total_f1 / len(data_list) if data_list else 0.0
        
        # 汇总结果
        summary = {
            'timestamp': datetime.now().isoformat(),
            'api_config': {
                'base_url': self.client.base_url,
                'embedding_model': self.client.embedding_model,
                'rerank_model': self.client.rerank_model,
                'llm_model': self.client.llm_model
            },
            'num_conversations': len(data_list),
            'overall_f1': overall_f1,
            'conversations': all_results
        }
        
        # 打印汇总
        print(f"\n{'='*60}")
        print("评估完成!")
        print(f"{'='*60}")
        print(f"总体 F1 Score: {overall_f1:.2%}")
        print(f"{'='*60}")
        
        return summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """保存结果到文件"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"locomo_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    # 检查 API 可用性
    print("检查 Paratera API 连接...")
    try:
        client = get_paratera_client()
        
        # 测试 embedding
        test_embed = client.embed("测试文本")
        if test_embed:
            print(f"✓ Embedding API 正常 (维度: {len(test_embed)})")
        else:
            print("✗ Embedding API 失败")
            return
        
        # 测试 chat
        test_chat = client.chat("你好", max_tokens=50)
        if test_chat:
            print(f"✓ Chat API 正常")
        else:
            print("✗ Chat API 失败")
            return
            
    except Exception as e:
        print(f"✗ API 连接失败: {e}")
        return
    
    # 运行评估
    evaluator = LoCoMoEvaluator()
    
    # 可以从命令行参数指定数据文件
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    results = evaluator.run_evaluation(data_path)
    
    # 保存结果
    evaluator.save_results(results)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
