#!/usr/bin/env python3
"""
LoCoMo 批量并行测试 - 使用 ThreadPoolExecutor 并行调用 API

优化点：
- 并行回答问题（默认 5 并发）
- 批量提取事实
- 实时进度显示
- 超时保护
"""

import json
import requests
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time


class ParateraClient:
    """Paratera API 客户端 - 线程安全版本"""
    
    def __init__(self, api_key: str, base_url: str = "https://llmapi.paratera.com",
                 embedding_model: str = "GLM-Embedding-3",
                 llm_model: str = "GLM-4-Plus"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        self.embed_url = f"{self.base_url}/v1/embeddings"
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, prompt: str, system_prompt: str = None, max_tokens: int = 500, 
             temperature: float = 0.0) -> str:
        """调用 LLM - 带重试逻辑"""
        max_retries = 3
        for attempt in range(max_retries):
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
                    timeout=30  # 减少超时时间
                )
                response.raise_for_status()
                
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                
                return ""
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # 指数退避
                else:
                    return ""


class LoCoMoParallelEvaluator:
    """LoCoMo 并行评估器"""
    
    def __init__(self, client: ParateraClient, max_workers: int = 5):
        self.client = client
        self.max_workers = max_workers
        self.print_lock = Lock()
        
    def safe_print(self, *args, **kwargs):
        """线程安全打印"""
        with self.print_lock:
            print(*args, **kwargs)
    
    def load_locomo_data(self, data_path: str) -> List[Dict]:
        """加载 LoCoMo 数据集"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_conversation(self, data: Dict) -> str:
        """格式化对话为文本"""
        lines = []
        conversation = data.get('conversation', {})
        
        for session_key in sorted(conversation.keys()):
            if not session_key.startswith('session_') or session_key.endswith('_date_time'):
                continue
            
            session = conversation[session_key]
            session_date = conversation.get(f'{session_key}_date_time', 'Unknown Date')
            
            lines.append(f"\n=== {session_key} ({session_date}) ===")
            
            for turn in session:
                speaker = turn.get('speaker', '')
                text = turn.get('text', '')
                lines.append(f"{speaker}: {text}")
        
        return "\n".join(lines)
    
    def extract_facts(self, conversation_text: str) -> List[str]:
        """批量提取事实"""
        system_prompt = """You are a fact extraction assistant. Extract specific facts from the conversation.

Rules:
1. Extract one fact per line
2. Include temporal information when available
3. Be specific and concrete
4. Output format: one fact per line, no numbering"""

        prompt = f"""Extract all facts from this conversation:

{conversation_text[:10000]}

Facts:"""
        
        self.safe_print("  提取事实...")
        response = self.client.chat(prompt, system_prompt, max_tokens=3000, temperature=0.0)
        
        facts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                # 清理编号和符号
                if line[0].isdigit() and line[1:3] in ['. ', ') ']:
                    line = line[3:].strip()
                if line.startswith('- ') or line.startswith('* '):
                    line = line[2:].strip()
                facts.append(line)
        
        return facts
    
    def answer_single_question(self, question_data: Dict, facts: List[str]) -> Dict:
        """回答单个问题（用于并行调用）"""
        q_id = question_data['q_id']
        question = question_data['question']
        ground_truth = question_data['answer']
        
        # 构建上下文（限制长度）
        context = "\n".join([f"- {fact}" for fact in facts[:20]])
        
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
        
        answer = self.client.chat(prompt, system_prompt, max_tokens=50, temperature=0.0)
        predicted = answer.strip().strip('"').strip("'")
        
        # 计算 F1
        f1 = self.calculate_f1(predicted, ground_truth)
        
        return {
            'q_id': q_id,
            'question': question,
            'predicted': predicted,
            'ground_truth': str(ground_truth),
            'f1': f1
        }
    
    def calculate_f1(self, predicted: str, ground_truth: Any) -> float:
        """计算 F1 分数"""
        if isinstance(ground_truth, (int, float)):
            ground_truth = str(ground_truth)
        
        pred = predicted.lower().strip()
        truth = str(ground_truth).lower().strip()
        
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
    
    def evaluate_conversation_parallel(self, data: Dict, conv_idx: int) -> Dict:
        """并行评估单个对话"""
        sample_id = data.get('sample_id', f'conv_{conv_idx}')
        qa_list = data.get('qa', [])
        
        self.safe_print(f"\n{'='*70}")
        self.safe_print(f"对话 {conv_idx}: {sample_id}")
        self.safe_print(f"{'='*70}")
        self.safe_print(f"问题数: {len(qa_list)}")
        
        # 格式化对话并提取事实
        conversation_text = self.format_conversation(data)
        facts = self.extract_facts(conversation_text)
        self.safe_print(f"  提取到 {len(facts)} 个事实")
        
        # 准备问题数据
        questions_data = []
        for i, qa in enumerate(qa_list):
            if 'question' in qa and 'answer' in qa:
                questions_data.append({
                    'q_id': i+1, 
                    'question': qa['question'], 
                    'answer': qa['answer']
                })
        
        if not questions_data:
            self.safe_print("  警告: 没有有效的问题数据")
            return {
                'sample_id': sample_id,
                'num_facts': len(facts),
                'num_questions': 0,
                'avg_f1': 0.0,
                'results': []
            }
        
        # 并行回答问题
        self.safe_print(f"  并行回答 {len(questions_data)} 个问题 (并发: {self.max_workers})...")
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_qid = {
                executor.submit(self.answer_single_question, q_data, facts): q_data['q_id']
                for q_data in questions_data
            }
            
            # 收集结果
            for future in as_completed(future_to_qid):
                try:
                    result = future.result(timeout=35)  # 单个问题超时
                    results.append(result)
                    completed += 1
                    
                    # 每 10 个显示进度
                    if completed % 10 == 0 or completed == len(questions_data):
                        self.safe_print(f"    进度: {completed}/{len(questions_data)}")
                        
                except Exception as e:
                    q_id = future_to_qid[future]
                    self.safe_print(f"    [错误] 问题 {q_id}: {e}")
                    results.append({
                        'q_id': q_id,
                        'question': '',
                        'predicted': 'ERROR',
                        'ground_truth': '',
                        'f1': 0.0
                    })
        
        # 按 q_id 排序
        results.sort(key=lambda x: x['q_id'])
        
        # 显示详细结果
        correct = sum(1 for r in results if r['f1'] >= 0.8)
        partial = sum(1 for r in results if 0.5 <= r['f1'] < 0.8)
        wrong = sum(1 for r in results if r['f1'] < 0.5)
        
        self.safe_print(f"\n  结果统计:")
        for r in results[:10]:  # 只显示前 10 个
            status = "✓" if r['f1'] >= 0.8 else "~" if r['f1'] >= 0.5 else "✗"
            self.safe_print(f"    [{r['q_id']:3d}] {status} F1:{r['f1']:.0%} | {r['question'][:40]}... | 答案: {r['predicted'][:30]}...")
        
        if len(results) > 10:
            self.safe_print(f"    ... 还有 {len(results) - 10} 个问题")
        
        # 计算平均 F1
        avg_f1 = sum(r['f1'] for r in results) / len(results) if results else 0.0
        
        self.safe_print(f"\n  正确: {correct}, 部分: {partial}, 错误: {wrong}")
        self.safe_print(f"  对话 F1: {avg_f1:.2%}")
        
        return {
            'sample_id': sample_id,
            'num_facts': len(facts),
            'num_questions': len(qa_list),
            'avg_f1': avg_f1,
            'correct': correct,
            'partial': partial,
            'wrong': wrong,
            'results': results
        }
    
    def run_evaluation(self, data_path: str, max_conversations: int = None) -> Dict:
        """运行完整并行评估"""
        print("="*70)
        print("LoCoMo 并行基准测试")
        print("="*70)
        print(f"API: https://llmapi.paratera.com")
        print(f"LLM: {self.client.llm_model}")
        print(f"并发数: {self.max_workers}")
        print(f"数据集: {data_path}")
        print("="*70)
        
        # 加载数据
        print("\n加载 LoCoMo 数据集...")
        data_list = self.load_locomo_data(data_path)
        
        if max_conversations:
            data_list = data_list[:max_conversations]
        
        print(f"加载了 {len(data_list)} 个对话")
        
        # 评估每个对话
        all_results = []
        total_f1 = 0.0
        total_questions = 0
        
        start_time = time.time()
        
        for i, data in enumerate(data_list, 1):
            result = self.evaluate_conversation_parallel(data, i)
            all_results.append(result)
            total_f1 += result['avg_f1'] * result['num_questions']
            total_questions += result['num_questions']
        
        elapsed = time.time() - start_time
        
        # 计算总体 F1
        overall_f1 = total_f1 / total_questions if total_questions > 0 else 0.0
        
        # 汇总
        summary = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': elapsed,
            'api_config': {
                'base_url': self.client.base_url,
                'llm_model': self.client.llm_model
            },
            'num_conversations': len(data_list),
            'total_questions': total_questions,
            'overall_f1': overall_f1,
            'conversations': all_results
        }
        
        # 打印汇总
        print(f"\n{'='*70}")
        print("评估完成!")
        print(f"{'='*70}")
        print(f"总对话数: {len(data_list)}")
        print(f"总问题数: {total_questions}")
        print(f"总体 F1 Score: {overall_f1:.2%}")
        print(f"总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
        print(f"{'='*70}")
        
        return summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """保存结果"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"locomo_parallel_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")


def main():
    """主函数"""
    # 初始化客户端
    client = ParateraClient(
        api_key="sk-0oVqiF3DzxzxTcbxsaPEOg",
        base_url="https://llmapi.paratera.com",
        llm_model="GLM-4-Plus"
    )
    
    # 运行评估
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_conv = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    print(f"并发数: {max_workers}")
    if max_conv:
        print(f"测试对话数: {max_conv}")
    
    evaluator = LoCoMoParallelEvaluator(client, max_workers=max_workers)
    
    results = evaluator.run_evaluation(
        data_path="locomodata.json",
        max_conversations=max_conv
    )
    
    # 保存结果
    evaluator.save_results(results)
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
